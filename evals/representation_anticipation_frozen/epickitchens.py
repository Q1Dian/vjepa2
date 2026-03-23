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
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class split_by_node(wds.PipelineStage):
    """Node splitter that uses provided rank/world_size instead of from torch.distributed"""

    def __init__(
        self,
        rank=0,
        world_size=1,
    ):
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
        annotations,
        frames_per_clip=16,
        future_frames_per_clip=16,
        fps=5,
        transform=None,
        anticipation_time_sec=[0.0, 0.0],
        anticipation_point=[0.25, 0.75],
    ):
        self.annotations = annotations
        self.frames_per_clip = frames_per_clip
        self.future_frames_per_clip = future_frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.anticipation_point = anticipation_point

    def run(self, src):
        for path in src:
            # -- get all action annotations for the video
            video_id = path.split("/")[-1].split(".")[0]
            ano = self.annotations[video_id]

            # -- get action annotations and frame stamps
            start_frames = ano["start_frame"].values
            stop_frames = ano["stop_frame"].values

            # -- load clips corresponding to action annotations
            try:
                vr = VideoReader(path, num_threads=4, ctx=cpu(0))
                vr.seek(0)
                # --
                vfps = vr.get_avg_fps()
                fpc = self.frames_per_clip
                future_fpc = self.future_frames_per_clip
                fstp = int(vfps / self.fps)
                nframes = int(fpc * fstp)
                future_nframes = int(future_fpc * fstp)
            except Exception as e:
                logging.info(f"Encountered exception loading video {e=}")
                continue

            for i, (sf, ef) in enumerate(zip(start_frames, stop_frames)):
                # sample an anticipation time
                at = random.uniform(*self.anticipation_time)
                aframes = int(at * vfps)

                # sample an anticipation frame b/w start and end of action
                ap = random.uniform(*self.anticipation_point)
                af = int(sf * ap + (1 - ap) * ef - aframes)

                context_indices = np.arange(af - nframes, af, fstp).astype(np.int64)
                future_indices = np.arange(af, af + future_nframes, fstp).astype(np.int64)

                context_indices[context_indices < 0] = 0
                future_indices[future_indices < 0] = 0
                context_indices[context_indices >= len(vr)] = len(vr) - 1
                future_indices[future_indices >= len(vr)] = len(vr) - 1

                try:
                    context_buffer = vr.get_batch(context_indices).asnumpy()
                    vr.seek(0)
                    future_buffer = vr.get_batch(future_indices).asnumpy()
                    vr.seek(0)
                except Exception as e:
                    logging.info(f"Encountered exception getting indices {e=}")
                    continue

                buffer = np.concatenate([context_buffer, future_buffer], axis=0)
                if self.transform is not None:
                    buffer = self.transform(buffer)

                yield dict(
                    video=buffer,
                    anticipation_time=at,
                )


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)
        logging.info("Done initializing ResampledShards")

    def __iter__(self):
        """Return an iterator over the shards."""
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
    _, num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is {num_shards=}")

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple("video", "anticipation_time"),
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


def filter_annotations(
    base_path,
    train_annotations_path,
    val_annotations_path,
    file_format=1,
):

    tdf = pd.read_csv(train_annotations_path)
    vdf = pd.read_csv(val_annotations_path)

    def build_annotations(df):
        video_paths, annotations = [], {}
        unique_videos = list(dict.fromkeys(df["video_id"].values))
        for uv in unique_videos:
            pid = uv.split("_")[0]
            candidate_paths = []
            if file_format == 0:
                candidate_paths.extend(
                    [
                        os.path.join(base_path, pid, "videos", uv + ".MP4"),
                        os.path.join(base_path, "train", pid, "videos", uv + ".MP4"),
                        os.path.join(base_path, "test", pid, "videos", uv + ".MP4"),
                    ]
                )
            else:
                candidate_paths.extend(
                    [
                        os.path.join(base_path, pid, uv + ".MP4"),
                        os.path.join(base_path, "train", pid, uv + ".MP4"),
                        os.path.join(base_path, "test", pid, uv + ".MP4"),
                    ]
                )

            fpath = next((p for p in candidate_paths if os.path.exists(p)), None)
            if fpath is None:
                logging.info(f"file path not found for {uv=}; tried {candidate_paths}")
                continue
            video_paths += [fpath]
            annotations[uv] = df[df["video_id"] == uv].sort_values(by="start_frame")
        return video_paths, annotations

    train_annotations = build_annotations(tdf)
    val_annotations = build_annotations(vdf)

    return dict(
        train=train_annotations,
        val=val_annotations,
    )


def make_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,
    fps=5,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=[0.0, 0.0],
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=[0.1, 0.1],
    **kwargs,
):

    paths, annotations = annotations_path
    num_clips = sum([len(a) for a in annotations.values()])
    future_frames_per_clip = kwargs.get("future_frames_per_clip", 16)

    video_decoder = decode_videos_to_clips(
        annotations=annotations,
        frames_per_clip=frames_per_clip,
        future_frames_per_clip=future_frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
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

    datainfo.dataloader.num_batches = num_clips // (world_size * batch_size)
    datainfo.dataloader.num_samples = num_clips

    return dataset, datainfo.dataloader, datainfo
