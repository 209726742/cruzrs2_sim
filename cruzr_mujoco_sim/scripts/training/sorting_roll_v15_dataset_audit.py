#!/usr/bin/env python3
"""Audit the formal Sorting Roll v15 LeRobot v3.0 dataset."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK_VERSION = "sorting_roll_v15_diverse_sim"
COLLECTION_PROFILE = "sorting_roll_d405_candidate_v6"
CAMERAS = (
    "observation.images.stereo_left",
    "observation.images.left_wrist_realsense",
    "observation.images.right_wrist_realsense",
)
SAMPLED_EPISODES = (0, 3, 6, 239, 240, 269, 270, 299)


def ffprobe(path):
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def finite_parquet_contract(root, task_by_index, prompt_by_episode):
    frames = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        for key in ("observation.state", "action"):
            field = schema.field(key)
            if not pa.types.is_list(field.type):
                raise ValueError(f"{key} is not a list column: {path}")
            if field.type.value_type != pa.float32():
                raise ValueError(f"{key} is not float32: {path}")
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=[
                    "observation.state",
                    "action",
                    "episode_index",
                    "task_index",
                ],
            )
            frames += table.num_rows
            for key in ("observation.state", "action"):
                values = np.asarray(table[key].to_pylist(), dtype=np.float32)
                if values.shape != (table.num_rows, 18):
                    raise ValueError(f"{key} has invalid shape in {path}")
                if not np.isfinite(values).all():
                    raise ValueError(f"{key} contains NaN or Inf in {path}")
            episode_indices = table["episode_index"].to_pylist()
            task_indices = table["task_index"].to_pylist()
            for episode_index, task_index in set(zip(
                episode_indices, task_indices
            )):
                if task_by_index.get(task_index) != prompt_by_episode.get(
                    episode_index
                ):
                    raise ValueError(
                        "frame task_index does not match episode prompt: "
                        f"episode={episode_index} task_index={task_index}"
                    )
    return frames


def source_diversity_counts(rows):
    fields = (
        "split",
        "pose_bin",
        "prompt_id",
        "object_profile",
        "appearance_profile",
        "lighting_profile",
        "dynamics_profile",
        "image_profile",
    )
    counts = {}
    for field in fields:
        values = []
        for row in rows:
            assignment = row["source_diversity"]["assignment"]
            value = row["source_split"] if field == "split" else assignment[field]
            values.append(value["name"] if isinstance(value, dict) else value)
        counts[field] = dict(sorted(Counter(values).items()))
    return counts


def audit(args):
    root = args.dataset.resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    stats = json.loads((root / "meta" / "stats.json").read_text())
    selection = json.loads(args.selection_report.read_text())
    errors = []

    expected_info = {
        "codebase_version": "v3.0",
        "source_task_version": TASK_VERSION,
        "source_campaign": args.campaign,
        "collection_profile": COLLECTION_PROFILE,
        "total_episodes": 300,
        "total_source_episodes": 300,
        "fps": 30,
        "total_tasks": 5,
        "splits": {"train": "0:240", "val": "240:270", "test": "270:300"},
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            errors.append(f"info.{key}={info.get(key)!r}, expected {expected!r}")
    if info.get("source_diversity_counts") != selection.get("counts"):
        errors.append("info diversity counts do not match source selection")

    features = info.get("features", {})
    actual_cameras = {
        key for key, value in features.items() if value.get("dtype") == "video"
    }
    if actual_cameras != set(CAMERAS):
        errors.append(f"camera feature mismatch: {sorted(actual_cameras)}")
    for camera in CAMERAS:
        feature = features.get(camera, {})
        video = feature.get("info", {})
        if feature.get("shape") != [224, 224, 3]:
            errors.append(f"{camera} shape mismatch")
        for key, expected in (
            ("video.fps", 30),
            ("video.codec", "h264"),
            ("video.pix_fmt", "yuv420p"),
        ):
            if video.get(key) != expected:
                errors.append(f"{camera} {key} mismatch")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [18]:
            errors.append(f"{key} must be 18-dimensional float32")
        if not {"q01", "q99"}.issubset(stats.get(key, {})):
            errors.append(f"{key} is missing q01/q99 statistics")

    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_rows = [
        row
        for path in episode_files
        for row in pq.read_table(path).to_pylist()
    ]
    task_path = root / "meta" / "tasks.parquet"
    task_rows = pq.read_table(task_path).to_pylist()
    task_by_index = {
        int(row["task_index"]): row["task"] for row in task_rows
    }
    prompt_by_episode = {
        int(row["episode_index"]):
        row["source_diversity"]["assignment"]["prompt"]
        for row in episode_rows
    }
    if len(task_by_index) != len(task_rows):
        errors.append("task metadata contains duplicate task indices")
    if sorted(task_by_index) != list(range(5)):
        errors.append("task metadata must contain indices 0 through 4")
    if set(task_by_index.values()) != set(prompt_by_episode.values()):
        errors.append("task metadata does not match source prompt texts")
    if len(episode_rows) != 300:
        errors.append(f"episode metadata count is {len(episode_rows)}, expected 300")
    else:
        seeds = [int(row["source_seed"]) for row in episode_rows]
        if len(set(seeds)) != 300:
            errors.append("episode source seeds are not unique")
        if any(row["source_task_version"] != TASK_VERSION for row in episode_rows):
            errors.append("episode metadata contains another task version")
        if any(
            row["source_collection_profile"] != COLLECTION_PROFILE
            for row in episode_rows
        ):
            errors.append("episode metadata contains another camera profile")
        if source_diversity_counts(episode_rows) != selection.get("counts"):
            errors.append("episode diversity counts do not match source selection")
        if any(
            row.get("tasks")
            != [row["source_diversity"]["assignment"]["prompt"]]
            for row in episode_rows
        ):
            errors.append("episode task text does not match source prompt")

    parquet_frames = finite_parquet_contract(
        root, task_by_index, prompt_by_episode
    )
    if parquet_frames != info.get("total_frames"):
        errors.append("Parquet frame count does not match info.json")

    video_streams = {}
    for camera in CAMERAS:
        paths = sorted((root / "videos" / camera).rglob("*.mp4"))
        streams = [ffprobe(path) for path in paths]
        frame_count = sum(int(stream["nb_read_frames"]) for stream in streams)
        for stream in streams:
            expected = {
                "codec_name": "h264",
                "width": 224,
                "height": 224,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
            }
            if any(stream.get(key) != value for key, value in expected.items()):
                errors.append(f"invalid encoded stream for {camera}")
        if frame_count != info.get("total_frames"):
            errors.append(f"{camera} frame count mismatch")
        video_streams[camera] = {
            "file_count": len(paths),
            "frame_count": frame_count,
            "streams": streams,
        }

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        video_backend="pyav",
    )
    if dataset.num_episodes != 300 or len(dataset) != info.get("total_frames"):
        errors.append("LeRobotDataset length mismatch")
    decoded = []
    for episode_index in SAMPLED_EPISODES:
        row = episode_rows[episode_index]
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        for frame_index in (start, (start + stop - 1) // 2, stop - 1):
            sample = dataset[frame_index]
            if tuple(sample["observation.state"].shape) != (18,):
                errors.append("decoded state shape mismatch")
            if tuple(sample["action"].shape) != (18,):
                errors.append("decoded action shape mismatch")
            for camera in CAMERAS:
                if tuple(sample[camera].shape) != (3, 224, 224):
                    errors.append(f"decoded camera shape mismatch: {camera}")
            numeric = np.concatenate([
                sample["observation.state"].cpu().numpy(),
                sample["action"].cpu().numpy(),
            ])
            if not all(math.isfinite(float(value)) for value in numeric):
                errors.append("decoded state/action contains NaN or Inf")
            decoded.append({
                "episode_index": episode_index,
                "source_seed": int(row["source_seed"]),
                "frame_index": frame_index,
            })

    report = {
        "schema_version": 1,
        "dataset": str(root),
        "repo_id": args.repo_id,
        "task_version": TASK_VERSION,
        "collection_profile": COLLECTION_PROFILE,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "cameras": list(CAMERAS),
        "state_dtype": features.get("observation.state", {}).get("dtype"),
        "state_dim": 18,
        "action_dtype": features.get("action", {}).get("dtype"),
        "action_dim": 18,
        "splits": info.get("splits"),
        "diversity_counts": info.get("source_diversity_counts"),
        "video_streams": video_streams,
        "sampled_episode_count": len(SAMPLED_EPISODES),
        "decoded_sample_count": len(decoded),
        "tasks": task_by_index,
        "sampled_episodes": list(SAMPLED_EPISODES),
        "decoded_samples": decoded,
        "errors": errors,
        "passed": not errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({
        "episodes": report["episodes"],
        "frames": report["frames"],
        "sampled_episode_count": report["sampled_episode_count"],
        "decoded_sample_count": report["decoded_sample_count"],
        "errors": errors,
        "passed": report["passed"],
    }, indent=2))
    return 0 if report["passed"] else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
