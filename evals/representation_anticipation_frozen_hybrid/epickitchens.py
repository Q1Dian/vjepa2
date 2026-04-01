# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from src.datasets.utils.worker_init_fn import pl_worker_init_function

multiprocessing.set_start_method("spawn", force=True)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    num_shards = len(shards_list)
    total_size = num_shards
    return total_size, num_shards


def log_and_continue(exn):
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class split_by_node(wds.PipelineStage):
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def run(self, src):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


class decode_videos_to_clips(wds.PipelineStage):
    def __init__(
        self,
        sliding_window_stride_frames=4,
        frames_per_clip=32,
        fps=8,
        transform=None,
        anticipation_frames=8,
        anticipation_gap=0.0,
        training=True,
    ):
        self.sliding_window_stride_frames = max(1, int(sliding_window_stride_frames))
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_frames = anticipation_frames
        self.anticipation_gap = anticipation_gap
        self.training = training

    def run(self, src):
        for path in src:
            try:
                vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
                vfps = vr.get_avg_fps()
                total_frames = len(vr)
                fstp = max(1, int(vfps / self.fps))

                context_nframes = int(self.frames_per_clip * fstp)
                gap_nframes = int(self.anticipation_gap * vfps)
                target_nframes = int(self.anticipation_frames * fstp)

                total_window = context_nframes + gap_nframes + target_nframes
            except Exception as e:
                logging.info(f"Encountered exception loading video {e=}")
                continue

            # Allow context to start before frame 0 (action-like behavior),
            # but keep target indices strictly inside the video (no end padding).
            min_start_idx = -context_nframes + fstp
            max_start_idx = total_frames - total_window - 1
            if max_start_idx < min_start_idx:
                continue

            for start_f in range(min_start_idx, max_start_idx + 1, self.sliding_window_stride_frames):

                context_indices = np.arange(start_f, start_f + context_nframes, fstp).astype(np.int64)
                # Action-like left padding: missing past context repeats first frame.
                context_indices[context_indices < 0] = 0
                context_indices = context_indices[: self.frames_per_clip]

                target_start_f = start_f + context_nframes + gap_nframes
                target_indices = np.arange(target_start_f, target_start_f + target_nframes, fstp).astype(np.int64)
                target_indices = target_indices[: self.anticipation_frames]

                if len(context_indices) < self.frames_per_clip or len(target_indices) < self.anticipation_frames:
                    continue

                try:
                    combined_indices = np.concatenate([context_indices, target_indices])
                    buffer = vr.get_batch(combined_indices).asnumpy()
                except Exception as e:
                    logging.info(f"Encountered exception getting indices {e=}")
                    continue

                if self.transform is not None:
                    buffer = self.transform(buffer)

                context_buffer = buffer[:, : self.frames_per_clip]
                target_buffer = buffer[:, self.frames_per_clip :]

                yield dict(
                    context=context_buffer,
                    target=target_buffer,
                )


def _estimate_num_clips(
    paths,
    frames_per_clip,
    fps,
    anticipation_frames,
    anticipation_gap,
    sliding_window_stride_frames,
):
    total = 0
    stride = max(1, int(sliding_window_stride_frames))
    for path in paths:
        try:
            vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
            vfps = vr.get_avg_fps()
            total_frames = len(vr)
            fstp = max(1, int(vfps / fps))

            context_nframes = int(frames_per_clip * fstp)
            gap_nframes = int(anticipation_gap * vfps)
            target_nframes = int(anticipation_frames * fstp)
            total_window = context_nframes + gap_nframes + target_nframes

            min_start_idx = -context_nframes + fstp
            max_start_idx = total_frames - total_window - 1
            if max_start_idx < min_start_idx:
                continue

            total += ((max_start_idx - min_start_idx) // stride) + 1
        except Exception as e:
            logging.info(f"Encountered exception estimating clips for {path}: {e=}")
            continue
    return total


class ResampledShards(IterableDataset):
    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)

    def __iter__(self):
        if self.training:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            yield from self.urls[torch.randperm(len(self.urls), generator=gen)]
        else:
            yield from self.urls[torch.arange(len(self.urls))]


def get_video_wds_dataset(
    batch_size,
    input_shards,
    video_decoder,
    training,
    epoch=0,
    world_size=1,
    rank=0,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
):
    assert input_shards is not None

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple("context", "target"),
        wds.batched(batch_size, partial=True, collation_fn=torch.utils.data.default_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=pl_worker_init_function,
        pin_memory=pin_memory,
    )

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def prepare_video_paths(
    base_path,
    csv_path,
    file_format=1,
):
    df = pd.read_csv(csv_path)
    video_paths = []
    unique_videos = list(dict.fromkeys(df["video_id"].values))
    for uv in unique_videos:
        pid = uv.split("_")[0]
        if file_format == 0:
            fpath = os.path.join(base_path, pid, "videos", f"{uv}.MP4")
        else:
            # handle case variation .MP4 vs .mp4
            fpath = os.path.join(base_path, pid, f"{uv}.MP4")
            if not os.path.exists(fpath):
                fpath = os.path.join(base_path, pid, f"{uv}.mp4")

        if not os.path.exists(fpath):
            logging.info(f"file path not found {fpath=}")
            continue
        video_paths.append(fpath)

    return video_paths


def make_webvid(
    base_path,
    video_list_path,
    batch_size,
    transform,
    frames_per_clip=32,
    fps=8,
    num_workers=8,
    world_size=1,
    rank=0,
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_frames=8,
    anticipation_gap=0.0,
    sliding_window_stride_frames=4,
    file_format=1,
    **kwargs,
):

    paths = prepare_video_paths(
        base_path=base_path,
        csv_path=video_list_path,
        file_format=file_format,
    )

    num_clips = _estimate_num_clips(
        paths=paths,
        frames_per_clip=frames_per_clip,
        fps=fps,
        anticipation_frames=anticipation_frames,
        anticipation_gap=anticipation_gap,
        sliding_window_stride_frames=sliding_window_stride_frames,
    )

    video_decoder = decode_videos_to_clips(
        sliding_window_stride_frames=sliding_window_stride_frames,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_frames=anticipation_frames,
        anticipation_gap=anticipation_gap,
        training=training,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        input_shards=paths,
        epoch=0,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        video_decoder=video_decoder,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        training=training,
    )

    datainfo.dataloader.num_batches = num_clips // max(1, (world_size * batch_size))
    datainfo.dataloader.num_samples = num_clips

    return dataset, datainfo.dataloader, datainfo
