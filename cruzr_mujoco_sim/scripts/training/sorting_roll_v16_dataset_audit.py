#!/usr/bin/env python3
"""Audit the v15-train + v16-pilot LeRobot v3.0 candidate dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sorting_roll_v15_dataset_audit import (  # noqa: E402
    CAMERAS,
    COLLECTION_PROFILE,
    ffprobe,
    finite_parquet_contract,
)
from src.lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


V15_TASK_VERSION = "sorting_roll_v15_diverse_sim"
V16_TASK_VERSION = "sorting_roll_v16_expansion_pilot_sim"
EXPECTED_SPLITS = {"train": "0:252", "val": "252:254", "test": "254:256"}
EXPECTED_FAMILIES = {"H": 4, "R": 8, "T": 4}
SAMPLED_EPISODES = (0, 3, 239, 240, 251, 252, 254, 255)


def load_episode_rows(root):
    return [
        row
        for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]


def decode_json_extension(value):
    """Decode Arrow JSON extension values, including legacy double encoding."""
    for _ in range(2):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    return value


def scenario_errors(rows, v16_task_version=V16_TASK_VERSION):
    errors = []
    v15 = [row for row in rows if row["source_task_version"] == V15_TASK_VERSION]
    v16 = [row for row in rows if row["source_task_version"] == v16_task_version]
    if len(v15) != 240 or len(v16) != 16:
        errors.append(f"task-version counts are v15={len(v15)} v16={len(v16)}")
    if any(row.get("source_split") != "train" for row in v15):
        errors.append("old v15 val/test leaked into the mixed dataset")
    if any(row.get("source_scenario") is not None for row in v15):
        errors.append("v15 rows unexpectedly contain v16 scenario metadata")
    family_counts = dict(sorted(Counter(
        (row.get("source_scenario") or {}).get("scenario_family")
        for row in v16
    ).items(), key=lambda item: str(item[0])))
    if family_counts != EXPECTED_FAMILIES:
        errors.append(f"v16 family counts {family_counts} != {EXPECTED_FAMILIES}")
    group_splits = defaultdict(set)
    for row in v16:
        scenario = row.get("source_scenario") or {}
        group = scenario.get("scene_group_id")
        if not group:
            errors.append("v16 row is missing scene_group_id")
            continue
        group_splits[group].add(row.get("source_split"))
        if scenario.get("recorded_start_phase") is None:
            errors.append("v16 row is missing recorded_start_phase")
        if scenario.get("recorded_terminal_phase") is None:
            errors.append("v16 row is missing recorded_terminal_phase")
        if scenario.get("scenario_family") == "R":
            evidence = decode_json_extension(scenario.get("intervention_evidence")) or {}
            if (
                scenario.get("intervention_frame") != -1
                or scenario.get("recovery_start_frame") != 0
                or evidence.get("completed_before_recording") is not True
            ):
                errors.append("v16 recovery boundary metadata is invalid")
    if any(len(splits) != 1 for splits in group_splits.values()):
        errors.append("a v16 scene group crosses dataset splits")
    return errors, family_counts


def audit(args):
    root = args.dataset.resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    stats = json.loads((root / "meta" / "stats.json").read_text())
    build = json.loads(args.build_report.read_text())
    errors = []
    expected_info = {
        "codebase_version": "v3.0",
        "source_task_version": "mixed",
        "source_task_versions": sorted((V15_TASK_VERSION, args.v16_task_version)),
        "source_campaign": None,
        "source_campaigns": sorted((args.v15_campaign, args.v16_campaign)),
        "collection_profile": COLLECTION_PROFILE,
        "total_episodes": 256,
        "total_source_episodes": 256,
        "fps": 30,
        "total_tasks": 5,
        "splits": EXPECTED_SPLITS,
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            errors.append(f"info.{key}={info.get(key)!r}, expected {expected!r}")
    if build.get("passed") is not True or build.get("split_counts") != {
        "test": 2, "train": 252, "val": 2
    }:
        errors.append("mixed build report is not admitted")
    if info.get("source_diversity_counts") != {}:
        errors.append("mixed dataset must not mislabel v15-only diversity counts as global")

    features = info.get("features", {})
    cameras = {key for key, value in features.items() if value.get("dtype") == "video"}
    if cameras != set(CAMERAS):
        errors.append(f"camera feature mismatch: {sorted(cameras)}")
    for camera in CAMERAS:
        feature = features.get(camera, {})
        video = feature.get("info", {})
        if feature.get("shape") != [224, 224, 3]:
            errors.append(f"{camera} shape mismatch")
        if any(video.get(key) != expected for key, expected in (
            ("video.fps", 30),
            ("video.codec", "h264"),
            ("video.pix_fmt", "yuv420p"),
        )):
            errors.append(f"{camera} video contract mismatch")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [18]:
            errors.append(f"{key} must be 18-dimensional float32")
        if not {"q01", "q99"}.issubset(stats.get(key, {})):
            errors.append(f"{key} is missing q01/q99 statistics")

    episode_rows = load_episode_rows(root)
    task_rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
    task_by_index = {int(row["task_index"]): row["task"] for row in task_rows}
    prompt_by_episode = {
        int(row["episode_index"]): row["tasks"][0] for row in episode_rows
    }
    if len(episode_rows) != 256:
        errors.append(f"episode metadata count is {len(episode_rows)}, expected 256")
    else:
        seeds = [int(row["source_seed"]) for row in episode_rows]
        if len(set(seeds)) != 256:
            errors.append("episode source seeds are not unique")
        if Counter(row["source_split"] for row in episode_rows) != {
            "train": 252, "val": 2, "test": 2
        }:
            errors.append("episode split counts are invalid")
        if any(
            row["source_collection_profile"] != COLLECTION_PROFILE
            for row in episode_rows
        ):
            errors.append("episode metadata contains another camera profile")
        if any(row.get("tasks") != [prompt_by_episode[index]] for index, row in enumerate(episode_rows)):
            errors.append("episode task text is malformed")
    scenario_audit_errors, family_counts = scenario_errors(
        episode_rows, args.v16_task_version
    )
    errors.extend(scenario_audit_errors)

    parquet_frames = finite_parquet_contract(root, task_by_index, prompt_by_episode)
    if parquet_frames != info.get("total_frames"):
        errors.append("Parquet frame count does not match info.json")

    video_streams = {}
    for camera in CAMERAS:
        paths = sorted((root / "videos" / camera).rglob("*.mp4"))
        referenced_files = {
            (row[f"videos/{camera}/chunk_index"], row[f"videos/{camera}/file_index"])
            for row in episode_rows
        }
        streams = [ffprobe(path) for path in paths]
        frame_count = sum(int(stream["nb_read_frames"]) for stream in streams)
        expected_stream = {
            "codec_name": "h264",
            "width": 224,
            "height": 224,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "30/1",
        }
        if len(paths) != len(referenced_files):
            errors.append(f"{camera} video file count mismatch")
        if any(
            any(stream.get(key) != value for key, value in expected_stream.items())
            for stream in streams
        ):
            errors.append(f"invalid encoded stream for {camera}")
        if frame_count != info.get("total_frames"):
            errors.append(f"{camera} frame count mismatch")
        video_streams[camera] = {"file_count": len(paths), "frame_count": frame_count}

    dataset = LeRobotDataset(repo_id=args.repo_id, root=root, video_backend="pyav")
    if dataset.num_episodes != 256 or len(dataset) != info.get("total_frames"):
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
            decoded.append({"episode_index": episode_index, "frame_index": frame_index})

    report = {
        "schema_version": 1,
        "dataset": str(root),
        "repo_id": args.repo_id,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "splits": info.get("splits"),
        "task_versions": info.get("source_task_versions"),
        "v16_family_counts": family_counts,
        "cameras": list(CAMERAS),
        "state_dim": 18,
        "action_dim": 18,
        "video_streams": video_streams,
        "sampled_episodes": list(SAMPLED_EPISODES),
        "decoded_sample_count": len(decoded),
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
        "v16_family_counts": family_counts,
        "decoded_sample_count": report["decoded_sample_count"],
        "errors": errors,
        "passed": report["passed"],
    }, indent=2))
    return 0 if report["passed"] else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--v15-campaign", required=True)
    parser.add_argument("--v16-campaign", required=True)
    parser.add_argument("--v16-task-version", default=V16_TASK_VERSION)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
