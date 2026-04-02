# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import json
import logging
import math

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from evals.representation_anticipation_frozen_hybrid.dataloader import init_data
from evals.representation_anticipation_frozen_hybrid.losses import topk_representation_loss
from evals.representation_anticipation_frozen_hybrid.models import init_module
from evals.representation_anticipation_frozen_hybrid.utils import WarmupCosineLRSchedule, CosineWDSchedule
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

    args_pretrain = args_eval.get("model_kwargs", {})
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs", {})
    args_wrapper = args_pretrain.get("wrapper_kwargs", {})

    args_exp = args_eval.get("experiment", {})
    args_data = args_exp.get("data", {})
    args_opt = args_exp.get("optimization", {})

    dataset = args_data.get("dataset", "EK100")
    base_path = args_data.get("base_path")
    file_format = args_data.get("file_format", 1)
    num_workers = args_data.get("num_workers", 8)
    pin_mem = args_data.get("pin_memory", True)
    frames_per_clip = args_data.get("frames_per_clip", 32)
    frames_per_second = args_data.get("frames_per_second", 8)
    resolution = args_data.get("resolution", 224)
    anticipation_frames = args_data.get("anticipation_frames", 8)
    anticipation_gap = args_data.get("anticipation_gap", 0.0)
    sliding_window_stride_frames = args_data.get("sliding_window_stride_frames", args_data.get("stride", 4))
    auto_augment = args_data.get("auto_augment", False)
    motion_shift = args_data.get("motion_shift", False)
    reprob = args_data.get("reprob", 0.0)
    random_resize_scale = args_data.get("random_resize_scale", (0.3, 1.0))
    train_video_path = args_data.get("dataset_train")
    val_video_path = args_data.get("dataset_val")

    batch_size = args_opt.get("batch_size", 2)
    num_epochs = args_opt.get("num_epochs", 20)
    use_bfloat16 = args_opt.get("use_bfloat16", True)
    topk_ratio = args_opt.get("topk_ratio", 0.25)
    first_opt = (args_opt.get("multihead_kwargs") or [{}])[0]
    optimizer_lr = first_opt.get("lr", args_opt.get("lr", 1e-4))
    optimizer_wd = first_opt.get("weight_decay", args_opt.get("weight_decay", 1e-4))
    # -- Scheduler parameters
    warmup_fraction = first_opt.get("warmup", args_opt.get("warmup", 0.1))
    start_lr = first_opt.get("start_lr", args_opt.get("start_lr", 0.0))
    final_lr = first_opt.get("final_lr", args_opt.get("final_lr", 0.0))
    final_wd = first_opt.get("final_weight_decay", args_opt.get("final_weight_decay", 0.0))

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
    logger.info(f"Initialized distributed runtime rank/world_size={rank}/{world_size} on device={device}")

    folder = os.path.join(pretrain_folder, "representation_anticipation_frozen_hybrid/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    config_path = os.path.join(folder, "config_eval.json")

    if rank == 0:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(args_eval, f, indent=2, sort_keys=True)
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train-loss"), ("%.5f", "val-loss"))
        tb_writer = SummaryWriter(log_dir=os.path.join(folder, "runs"))
        logger.info(f"Logging to {log_file}")

    logger.info(
        "Config summary: "
        f"dataset={dataset} batch_size={batch_size} epochs={num_epochs} "
        f"frames_per_clip={frames_per_clip} anticipation_frames={anticipation_frames} "
        f"anticipation_gap={anticipation_gap} stride={sliding_window_stride_frames}"
    )

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
    # -- Use custom param groups for scheduler support
    param_groups = [
        {
            "params": (p for p in model.parameters() if p.requires_grad),
            "mc_warmup_steps": None,  # Will be set after dataloader is ready
            "mc_start_lr": start_lr,
            "mc_ref_lr": optimizer_lr,
            "mc_final_lr": final_lr,
            "mc_ref_wd": optimizer_wd,
            "mc_final_wd": final_wd,
        }
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=optimizer_lr,
        weight_decay=optimizer_wd,
    )
    # Placeholder schedulers (will be replaced after dataloader init)
    lr_scheduler = None
    wd_scheduler = None

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
        csv_path=train_video_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_frames=anticipation_frames,
        anticipation_gap=anticipation_gap,
        sliding_window_stride_frames=sliding_window_stride_frames,
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
        file_format=file_format,
    )
    logger.info(f"Train dataloader ready with num_batches={train_loader.num_batches}")

    # -- Initialize schedulers now that we know iterations per epoch
    ite = train_loader.num_batches
    optimizer.param_groups[0]["mc_warmup_steps"] = int(warmup_fraction * ite)
    lr_scheduler = WarmupCosineLRSchedule(optimizer, T_max=int(num_epochs * ite))
    wd_scheduler = CosineWDSchedule(optimizer, T_max=int(num_epochs * ite))

    # -- Skip past already-completed steps if resuming
    if resume_checkpoint and start_epoch > 0:
        for _ in range(start_epoch * ite):
            lr_scheduler.step()
            wd_scheduler.step()

    _, val_loader, _ = init_data(
        dataset=dataset,
        training=False,
        base_path=base_path,
        csv_path=val_video_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_frames=anticipation_frames,
        anticipation_gap=anticipation_gap,
        sliding_window_stride_frames=sliding_window_stride_frames,
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
        file_format=file_format,
    )
    logger.info(f"Val dataloader ready with num_batches={val_loader.num_batches}")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        train_data_info.set_epoch(epoch)
        if not val_only:
            train_loss = train_one_epoch(
                device=device,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                anticipation_gap=anticipation_gap,
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
            anticipation_gap=anticipation_gap,
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
            logger.info(f"Saved checkpoint to {latest_path} at epoch={epoch + 1}")

        logger.info(f"Finished epoch {epoch + 1}/{num_epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

    if rank == 0:
        tb_writer.close()


def train_one_epoch(
    device,
    model,
    optimizer,
    lr_scheduler,
    wd_scheduler,
    data_loader,
    anticipation_gap,
    use_bfloat16,
    topk_ratio,
    patches_per_step,
):
    model_core = _unwrap_model(model)
    model_core.encoder.eval()
    model_core.predictor.train()
    losses = AverageMeter()
    data_iter = iter(data_loader)
    steps = 0

    for step_idx in range(data_loader.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info(f"Training dataloader exhausted early at step {step_idx}/{data_loader.num_batches}")
            break

        # -- Step learning rate and weight decay schedulers
        lr_scheduler.step()
        wd_scheduler.step()

        context_clips = batch[0].to(device)
        future_clips = batch[1].to(device)
        anticipation_times = torch.full((context_clips.size(0),), anticipation_gap, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16, device_type=device.type):
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
        losses.update(reduced_loss, context_clips.size(0))
        steps += 1

        if step_idx % 10 == 0:
            logger.info(f"train step {step_idx}/{data_loader.num_batches} loss={losses.avg:.5f}")

    logger.info(f"Train epoch summary: steps={steps} avg_loss={losses.avg:.5f}")

    return losses.avg


@torch.no_grad()
def validate(
    device,
    model,
    data_loader,
    anticipation_gap,
    use_bfloat16,
    topk_ratio,
    patches_per_step,
):
    model_core = _unwrap_model(model)
    model_core.encoder.eval()
    model_core.predictor.eval()
    losses = AverageMeter()
    data_iter = iter(data_loader)
    steps = 0

    for step_idx in range(data_loader.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info(f"Validation dataloader exhausted early at step {step_idx}/{data_loader.num_batches}")
            break
        context_clips = batch[0].to(device)
        future_clips = batch[1].to(device)
        anticipation_times = torch.full((context_clips.size(0),), anticipation_gap, device=device)

        with torch.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16, device_type=device.type):
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
        losses.update(reduced_loss, context_clips.size(0))
        steps += 1

        if step_idx % 10 == 0:
            logger.info(f"val step {step_idx}/{data_loader.num_batches} loss={losses.avg:.5f}")

    logger.info(f"Validation summary: steps={steps} avg_loss={losses.avg:.5f}")

    return losses.avg
