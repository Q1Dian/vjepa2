# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from evals.representation_anticipation_frozen.dataloader import filter_annotations, init_data
from evals.representation_anticipation_frozen.losses import topk_representation_loss
from evals.representation_anticipation_frozen.models import init_module
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _encode_tokens(model, clips):
    model_core = _unwrap_model(model)
    target_full = model_core.encoder(clips)
    embed_dim = model_core.embed_dim
    if target_full.size(-1) > embed_dim:
        return target_full[:, :, -embed_dim:]
    return target_full


def _split_context_future(clips, frames_per_clip, future_frames_per_clip):
    context = clips[:, :, :frames_per_clip, :, :]
    future = clips[:, :, frames_per_clip : frames_per_clip + future_frames_per_clip, :, :]
    return context, future


def _distributed_mean(value, device):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = torch.tensor([value], device=device, dtype=torch.float32)
        torch.distributed.all_reduce(tensor)
        tensor /= torch.distributed.get_world_size()
        return float(tensor.item())
    return float(value)


def _align_predictions_to_targets(predictions, targets):
    """Align predictor output tokens with target token count.

    Some wrappers return accumulated context + predicted tokens. For latent
    future loss, only the last target-length tokens should be compared.
    """
    if predictions.size(-1) != targets.size(-1):
        raise ValueError(f"feature dim mismatch: predictions={predictions.shape}, targets={targets.shape}")

    if predictions.size(1) == targets.size(1):
        return predictions

    if predictions.size(1) > targets.size(1):
        return predictions[:, -targets.size(1) :, :]

    raise ValueError(
        f"prediction has fewer tokens than target: predictions={predictions.shape}, targets={targets.shape}"
    )


def main(args_eval, resume_preempt=False):
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_data = args_exp.get("data")
    args_opt = args_exp.get("optimization")

    dataset = args_data.get("dataset")
    base_path = args_data.get("base_path")
    file_format = args_data.get("file_format", 1)
    num_workers = args_data.get("num_workers", 12)
    pin_mem = args_data.get("pin_memory", True)
    frames_per_clip = args_data.get("frames_per_clip")
    future_frames_per_clip = args_data.get("future_frames_per_clip", 16)
    frames_per_second = args_data.get("frames_per_second")
    resolution = args_data.get("resolution", 224)
    train_anticipation_time_sec = args_data.get("train_anticipation_time_sec")
    train_anticipation_point = args_data.get("train_anticipation_point")
    val_anticipation_point = args_data.get("val_anticipation_point", [0.0, 0.0])
    val_anticipation_time_sec = args_data.get("anticipation_time_sec")
    auto_augment = args_data.get("auto_augment")
    motion_shift = args_data.get("motion_shift")
    reprob = args_data.get("reprob")
    random_resize_scale = args_data.get("random_resize_scale")
    train_annotations_path = args_data.get("dataset_train")
    val_annotations_path = args_data.get("dataset_val")

    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    topk_ratio = args_opt.get("topk_ratio", 0.25)
    first_opt = (args_opt.get("multihead_kwargs") or [{}])[0]
    optimizer_lr = first_opt.get("lr", 1e-4)
    optimizer_wd = first_opt.get("weight_decay", 1e-4)

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()

    folder = os.path.join(pretrain_folder, "representation_anticipation_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train-loss"), ("%.5f", "val-loss"))
        tb_writer = SummaryWriter(log_dir=os.path.join(folder, "runs"))

    annotations = filter_annotations(
        dataset,
        base_path,
        train_annotations_path,
        val_annotations_path,
        file_format=file_format,
    )
    train_annotations = annotations["train"]
    val_annotations = annotations["val"]

    model = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    model_core = _unwrap_model(model)
    patches_per_step = int(model_core.grid_size**2) if hasattr(model_core, "grid_size") else None
    if world_size > 1:
        model = DistributedDataParallel(model, static_graph=True)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_lr,
        weight_decay=optimizer_wd,
    )

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        checkpoint_state = robust_checkpoint_loader(latest_path, map_location=torch.device("cpu"))
        _unwrap_model(model).load_state_dict(checkpoint_state["model"])
        optimizer.load_state_dict(checkpoint_state["opt"])
        start_epoch = checkpoint_state.get("epoch", 0)
        if val_only:
            start_epoch = 0

    _, train_loader, train_data_info = init_data(
        dataset=dataset,
        training=True,
        base_path=base_path,
        annotations_path=train_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        future_frames_per_clip=future_frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=train_anticipation_time_sec,
        anticipation_point=train_anticipation_point,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )
    _, val_loader, _ = init_data(
        dataset=dataset,
        training=False,
        base_path=base_path,
        annotations_path=val_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        future_frames_per_clip=future_frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=val_anticipation_time_sec,
        anticipation_point=val_anticipation_point,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )

    for epoch in range(start_epoch, num_epochs):
        train_data_info.set_epoch(epoch)
        if not val_only:
            train_loss = train_one_epoch(
                device=device,
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                frames_per_clip=frames_per_clip,
                future_frames_per_clip=future_frames_per_clip,
                use_bfloat16=use_bfloat16,
                topk_ratio=topk_ratio,
                patches_per_step=patches_per_step,
            )
        else:
            train_loss = float("nan")

        val_loss = validate(
            device=device,
            model=model,
            data_loader=val_loader,
            frames_per_clip=frames_per_clip,
            future_frames_per_clip=future_frames_per_clip,
            use_bfloat16=use_bfloat16,
            topk_ratio=topk_ratio,
            patches_per_step=patches_per_step,
        )

        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, val_loss)
            tb_writer.add_scalar("loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("loss/val", val_loss, epoch + 1)
            tb_writer.flush()
            print(f"[{epoch + 1:5d}] train loss: {train_loss:.5f} val loss: {val_loss:.5f}")
            torch.save(
                {
                    "model": _unwrap_model(model).state_dict(),
                    "opt": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "batch_size": batch_size,
                    "world_size": world_size,
                },
                latest_path,
            )

    if rank == 0:
        tb_writer.close()


def train_one_epoch(
    device,
    model,
    optimizer,
    data_loader,
    frames_per_clip,
    future_frames_per_clip,
    use_bfloat16,
    topk_ratio,
    patches_per_step,
):
    model_core = _unwrap_model(model)
    model_core.encoder.eval()
    model_core.predictor.train()
    losses = AverageMeter()
    data_iter = iter(data_loader)

    for _ in range(data_loader.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        clips = batch[0].to(device)
        anticipation_times = batch[-1].to(device)

        context_clips, future_clips = _split_context_future(clips, frames_per_clip, future_frames_per_clip)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            predictions = model(context_clips, anticipation_times)
            with torch.no_grad():
                targets = _encode_tokens(model, future_clips)
            predictions = _align_predictions_to_targets(predictions, targets)
            loss = topk_representation_loss(
                predictions,
                targets,
                topk_ratio=topk_ratio,
                patches_per_step=patches_per_step,
            )

        loss.backward()
        optimizer.step()

        reduced_loss = _distributed_mean(loss.detach().item(), device)
        losses.update(reduced_loss, clips.size(0))

    return losses.avg


@torch.no_grad()
def validate(
    device,
    model,
    data_loader,
    frames_per_clip,
    future_frames_per_clip,
    use_bfloat16,
    topk_ratio,
    patches_per_step,
):
    model_core = _unwrap_model(model)
    model_core.encoder.eval()
    model_core.predictor.eval()
    losses = AverageMeter()
    data_iter = iter(data_loader)

    for _ in range(data_loader.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        clips = batch[0].to(device)
        anticipation_times = batch[-1].to(device)

        context_clips, future_clips = _split_context_future(clips, frames_per_clip, future_frames_per_clip)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            predictions = model(context_clips, anticipation_times)
            targets = _encode_tokens(model, future_clips)
            predictions = _align_predictions_to_targets(predictions, targets)
            loss = topk_representation_loss(
                predictions,
                targets,
                topk_ratio=topk_ratio,
                patches_per_step=patches_per_step,
            )

        reduced_loss = _distributed_mean(loss.detach().item(), device)
        losses.update(reduced_loss, clips.size(0))

    return losses.avg
