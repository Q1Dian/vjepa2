#!/usr/bin/env python3
"""Create a small preprocessed EPIC-KITCHENS toy subset.

This script:
- samples a fixed number of unique videos from train/val CSVs,
- re-encodes sampled source videos to a lighter format,
- writes filtered annotation CSVs for the sampled videos.
"""

import argparse
import random
import subprocess
from pathlib import Path

import pandas as pd


def _resolve_video_path(video_id: str, base_path: Path, file_format: int) -> Path | None:
    pid = video_id.split("_")[0]
    if file_format == 0:
        candidates = [
            base_path / pid / "videos" / f"{video_id}.MP4",
            base_path / "train" / pid / "videos" / f"{video_id}.MP4",
            base_path / "test" / pid / "videos" / f"{video_id}.MP4",
        ]
    else:
        candidates = [
            base_path / pid / f"{video_id}.MP4",
            base_path / "train" / pid / f"{video_id}.MP4",
            base_path / "test" / pid / f"{video_id}.MP4",
        ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _existing_video_paths(df: pd.DataFrame, base_path: Path, file_format: int) -> dict[str, Path]:
    unique_videos = list(dict.fromkeys(df["video_id"].tolist()))
    resolved = {}
    for video_id in unique_videos:
        src_path = _resolve_video_path(video_id, base_path, file_format)
        if src_path is not None:
            resolved[video_id] = src_path
    return resolved


def _sample_video_ids(
    available_ids: list[str],
    count: int,
    seed: int,
    preferred_pids: tuple[str, ...],
    min_per_pid: int,
) -> list[str]:
    rng = random.Random(seed)
    if count >= len(available_ids):
        return sorted(available_ids)

    selected = []
    selected_set = set()
    for pid in preferred_pids:
        pid_candidates = [vid for vid in available_ids if vid.startswith(f"{pid}_")]
        if not pid_candidates:
            continue
        k = min(min_per_pid, len(pid_candidates), count - len(selected))
        if k <= 0:
            continue
        for vid in rng.sample(pid_candidates, k):
            if vid not in selected_set:
                selected.append(vid)
                selected_set.add(vid)
        if len(selected) >= count:
            return sorted(selected)

    remaining = [vid for vid in available_ids if vid not in selected_set]
    need = count - len(selected)
    if need > 0:
        selected.extend(rng.sample(remaining, need))

    return sorted(selected)


def _transcode_video(
    src_path: Path,
    dst_path: Path,
    short_side: int,
    target_fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not overwrite:
        return

    scale_filter = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_path),
        "-vf",
        scale_filter,
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-an",
        str(dst_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Fallback path for ffmpeg builds/options that do not support x264 preset/crf.
        fallback_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_path),
            "-vf",
            scale_filter,
            "-r",
            str(target_fps),
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-an",
            str(dst_path),
        ]
        subprocess.run(fallback_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy EPIC-KITCHENS subset with preprocessed videos")
    parser.add_argument("--train-csv", required=True, help="Path to EPIC_100_train.csv")
    parser.add_argument("--val-csv", required=True, help="Path to EPIC_100_validation.csv")
    parser.add_argument("--src-base-path", required=True, help="Path to source videos root used by training config")
    parser.add_argument("--output-root", required=True, help="Output directory for toy set")
    parser.add_argument("--train-videos", type=int, default=24, help="Number of unique train videos")
    parser.add_argument("--val-videos", type=int, default=8, help="Number of unique val videos")
    parser.add_argument("--file-format", type=int, default=1, choices=[0, 1], help="EPIC file format mode")
    parser.add_argument("--short-side", type=int, default=384, help="Resize short side to this value")
    parser.add_argument("--fps", type=int, default=8, help="Target output FPS")
    parser.add_argument("--crf", type=int, default=28, help="H264 CRF (higher is smaller, lower quality)")
    parser.add_argument("--preset", default="veryfast", help="FFmpeg x264 preset")
    parser.add_argument(
        "--preferred-pids",
        nargs="+",
        default=["P01", "P02"],
        help="Participant IDs to prioritize in both train and val sampling",
    )
    parser.add_argument(
        "--min-preferred-per-split",
        type=int,
        default=1,
        help="Minimum sampled videos per preferred participant in each split when available",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already transcoded files")
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    src_base_path = Path(args.src_base_path)
    out_root = Path(args.output_root)
    out_videos_root = out_root / "videos"
    out_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_path_map = _existing_video_paths(train_df, src_base_path, args.file_format)
    val_path_map = _existing_video_paths(val_df, src_base_path, args.file_format)

    train_ids = _sample_video_ids(
        available_ids=sorted(train_path_map.keys()),
        count=args.train_videos,
        seed=args.seed,
        preferred_pids=tuple(args.preferred_pids),
        min_per_pid=args.min_preferred_per_split,
    )
    val_ids = _sample_video_ids(
        available_ids=sorted(val_path_map.keys()),
        count=args.val_videos,
        seed=args.seed + 1,
        preferred_pids=tuple(args.preferred_pids),
        min_per_pid=args.min_preferred_per_split,
    )

    selected_ids = set(train_ids) | set(val_ids)
    print(f"Selected {len(train_ids)} train videos and {len(val_ids)} val videos ({len(selected_ids)} total).")
    print(f"Available source videos: train={len(train_path_map)} val={len(val_path_map)}")

    missing = []
    failed = []
    done = 0
    total = len(selected_ids)
    for idx, video_id in enumerate(sorted(selected_ids), start=1):
        src_path = train_path_map.get(video_id, val_path_map.get(video_id))
        if src_path is None:
            missing.append(video_id)
            print(f"[{idx}/{total}] missing source for {video_id}")
            continue

        pid = video_id.split("_")[0]
        dst_path = out_videos_root / pid / f"{video_id}.MP4"
        print(f"[{idx}/{total}] transcoding {video_id}")
        try:
            _transcode_video(
                src_path=src_path,
                dst_path=dst_path,
                short_side=args.short_side,
                target_fps=args.fps,
                crf=args.crf,
                preset=args.preset,
                overwrite=args.overwrite,
            )
            done += 1
        except subprocess.CalledProcessError as exc:
            failed.append(video_id)
            print(f"[{idx}/{total}] ffmpeg failed for {video_id}: {exc}")

    toy_train_df = train_df[train_df["video_id"].isin(train_ids)].copy()
    toy_val_df = val_df[val_df["video_id"].isin(val_ids)].copy()

    toy_train_csv = out_root / "EPIC_100_train_toy.csv"
    toy_val_csv = out_root / "EPIC_100_validation_toy.csv"
    toy_train_df.to_csv(toy_train_csv, index=False)
    toy_val_df.to_csv(toy_val_csv, index=False)

    print("\nToy dataset created:")
    print(f"  videos_root: {out_videos_root}")
    print(f"  train_csv:   {toy_train_csv} ({len(toy_train_df)} rows)")
    print(f"  val_csv:     {toy_val_csv} ({len(toy_val_df)} rows)")
    print(f"  transcoded:  {done}/{total} videos")

    train_pid_counts = pd.Series([vid.split("_")[0] for vid in train_ids]).value_counts()
    val_pid_counts = pd.Series([vid.split("_")[0] for vid in val_ids]).value_counts()
    print("  preferred participant coverage:")
    for pid in args.preferred_pids:
        print(f"    {pid}: train_videos={train_pid_counts.get(pid, 0)} val_videos={val_pid_counts.get(pid, 0)}")

    if missing:
        print(f"  missing:     {len(missing)} videos")
    if failed:
        print(f"  failed:      {len(failed)} videos")


if __name__ == "__main__":
    main()
