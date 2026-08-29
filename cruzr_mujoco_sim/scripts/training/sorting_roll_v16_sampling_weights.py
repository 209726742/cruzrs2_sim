#!/usr/bin/env python3
"""Build auditable per-frame sampling weights for the v16 pilot train split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset


V15_TASK_VERSION = "sorting_roll_v15_diverse_sim"
SAMPLING_PROFILES = {
    "pilot_old50": {"old": 0.50, "H": 0.15, "T": 0.15, "R": 0.20},
    "full_v2_old70": {"old": 0.70, "H": 0.10, "T": 0.10, "R": 0.10},
    "stage80_old50": {
        "old": 0.50,
        "H": 0.15,
        "T": 0.15,
        "R": 0.15,
        "C": 0.05,
    },
}
DEFAULT_PROFILE = "pilot_old50"
TARGET_FRACTIONS = SAMPLING_PROFILES[DEFAULT_PROFILE]


def load_episode_rows(root: Path) -> list[dict]:
    return [
        row
        for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]


def episode_family(row: dict, supported_families=TARGET_FRACTIONS) -> str:
    if row["source_task_version"] == V15_TASK_VERSION:
        return "old"
    scenario = row.get("source_scenario") or {}
    family = scenario.get("scenario_family")
    if family not in supported_families:
        raise ValueError(f"unsupported v16 family for episode {row['episode_index']}: {family}")
    return family


def build_frame_weights(
    frame_episode_indices: np.ndarray,
    episode_rows: list[dict],
    target_fractions: dict[str, float] = TARGET_FRACTIONS,
) -> tuple[np.ndarray, dict]:
    if not np.isclose(sum(target_fractions.values()), 1.0):
        raise ValueError("target fractions must sum to one")
    rows = {int(row["episode_index"]): row for row in episode_rows}
    selected_episodes = sorted({int(index) for index in frame_episode_indices})
    missing_rows = [index for index in selected_episodes if index not in rows]
    if missing_rows:
        raise ValueError(f"missing episode metadata: {missing_rows}")
    non_train = [index for index in selected_episodes if rows[index].get("source_split") != "train"]
    if non_train:
        raise ValueError(f"sampling weights may only contain train episodes: {non_train}")

    families = np.asarray(
        [
            episode_family(rows[int(index)], target_fractions)
            for index in frame_episode_indices
        ],
        dtype=object,
    )
    frame_counts = Counter(families.tolist())
    missing_families = sorted(set(target_fractions) - set(frame_counts))
    if missing_families:
        raise ValueError(f"target families have no train frames: {missing_families}")

    weights = np.asarray(
        [target_fractions[family] / frame_counts[family] for family in families],
        dtype=np.float64,
    )
    expected_mass = {
        family: float(weights[families == family].sum()) for family in sorted(target_fractions)
    }
    total_frames = len(weights)
    report = {
        "schema_version": 1,
        "frame_count": total_frames,
        "episode_count": len(selected_episodes),
        "selected_episode_min": selected_episodes[0],
        "selected_episode_max": selected_episodes[-1],
        "frame_counts": dict(sorted(frame_counts.items())),
        "natural_frame_fractions": {
            family: count / total_frames for family, count in sorted(frame_counts.items())
        },
        "target_fractions": dict(sorted(target_fractions.items())),
        "expected_sampling_mass": expected_mass,
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "replacement": True,
        "num_samples_per_sampler_cycle": total_frames,
    }
    return weights, report


def parse_episode_range(value: str) -> list[int]:
    start, stop = value.split(":", maxsplit=1)
    return list(range(int(start), int(stop)))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", default="0:252")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(SAMPLING_PROFILES),
        default=DEFAULT_PROFILE,
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    episodes = parse_episode_range(args.episodes)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset,
        episodes=episodes,
        download_videos=False,
        video_backend="pyav",
    )
    frame_episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    target_fractions = SAMPLING_PROFILES[args.profile]
    weights, report = build_frame_weights(
        frame_episode_indices,
        load_episode_rows(args.dataset),
        target_fractions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, weights, allow_pickle=False)
    os.replace(temporary, args.output)
    report.update({
        "dataset": str(args.dataset.resolve()),
        "repo_id": args.repo_id,
        "sampling_profile": args.profile,
        "episodes": args.episodes,
        "weights_path": str(args.output.resolve()),
        "weights_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "passed": all(
            np.isclose(report["expected_sampling_mass"][family], target, atol=1e-12)
            for family, target in target_fractions.items()
        ),
    })
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
