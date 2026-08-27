#!/usr/bin/env python3
"""Build the admitted v15-train + v16-pilot LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
sys.path[:0] = [str(SCRIPT_DIR), str(CORE_DIR)]

import sorting_roll_build_v21 as builder  # noqa: E402
from sorting_roll_v16_pilot_contract import (  # noqa: E402
    TASK_VERSION,
    load_manifest,
)
from sorting_roll_v16_validate import validate_episode  # noqa: E402


EXPECTED_COUNTS = {"train": 252, "val": 2, "test": 2}
SCENARIO_FIELDS = (
    "scenario_family",
    "scenario_variant",
    "scene_group_id",
    "counterfactual_pair_id",
    "start_phase",
    "terminal_phase",
    "recorded_start_phase",
    "recorded_terminal_phase",
    "intervention_type",
    "intervention_frame",
    "recovery_start_frame",
    "intervention_evidence",
    "target_object_id",
    "target_color",
    "distractor_object_ids",
    "requested_transforms",
    "applied_transforms",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_v15_train_sources(validation_report):
    report = load_json(validation_report)
    if (
        report.get("passed") is not True
        or report.get("episode_count") != 300
        or report.get("passed_count") != 300
        or report.get("failed_count") != 0
    ):
        raise ValueError("v15 validation report is not an admitted 300/300 report")
    sources = [
        dict(record["info"])
        for record in report["records"]
        if record.get("passed") is True
        and (record.get("info") or {}).get("split") == "train"
    ]
    if len(sources) != 240:
        raise ValueError(f"expected 240 admitted v15 train sources, got {len(sources)}")
    for source in sources:
        source["campaign"] = report["campaign"]
        source["scenario"] = None
    return builder.sort_sources(sources)


def old_v21_source_rows(dataset):
    rows = [
        json.loads(line)
        for line in (dataset / "meta" / "source_episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return sorted(rows, key=lambda row: row["dataset_episode_index"])


def reuse_video_paths(dataset, episode_index):
    paths = {
        camera: (
            dataset
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / f"episode_{episode_index:06d}.mp4"
        )
        for camera in builder.POLICY_CAMERAS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("missing reusable v15 videos: " + ", ".join(missing))
    return paths


def attach_v15_reuse(sources, dataset):
    rows = [row for row in old_v21_source_rows(dataset) if row["split"] == "train"]
    if len(rows) != 240:
        raise ValueError(f"expected 240 v15 train rows in old v2.1 dataset, got {len(rows)}")
    expected_paths = [str(Path(source["path"]).resolve()) for source in sources]
    actual_paths = [str(Path(row["path"]).resolve()) for row in rows]
    if actual_paths != expected_paths:
        raise ValueError("old v2.1 train order does not match admitted v15 source order")
    for source, row in zip(sources, rows):
        source["reuse_video_paths"] = reuse_video_paths(
            dataset, row["dataset_episode_index"]
        )


def load_v16_sources(root, manifest_path):
    manifest = load_manifest(manifest_path)
    sources = []
    for assignment in manifest["assignments"]:
        episode = root / f"seed_{assignment['seed']}"
        info, errors = validate_episode(episode, assignment)
        if errors:
            raise ValueError(
                f"v16 source validation failed for seed {assignment['seed']}: "
                + "; ".join(errors)
            )
        meta = load_json(episode / "meta.json")
        episode_meta = meta["episode_metadata"]
        source = dict(info)
        source.update({
            "task_version": TASK_VERSION,
            "collection_profile": episode_meta["collection_profile"],
            "prompt": meta["prompt"],
            "diversity": meta["diversity"],
            "campaign": manifest["campaign"],
            "scenario": {
                field: meta.get(field) for field in SCENARIO_FIELDS
            },
        })
        sources.append(source)
    if len(sources) != 16:
        raise ValueError(f"expected 16 admitted v16 sources, got {len(sources)}")
    return builder.sort_sources(sources)


def split_counts(sources):
    return dict(sorted(Counter(source["split"] for source in sources).items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15-validation", type=Path, required=True)
    parser.add_argument("--v15-v21", type=Path, required=True)
    parser.add_argument("--v16-root", type=Path, required=True)
    parser.add_argument("--v16-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--encode-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.encode_workers < 1:
        parser.error("--encode-workers must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite report: {args.report}")
    v15 = load_v15_train_sources(args.v15_validation)
    attach_v15_reuse(v15, args.v15_v21.resolve())
    v16 = load_v16_sources(args.v16_root.resolve(), args.v16_manifest.resolve())
    sources = builder.sort_sources(v15 + v16)
    counts = split_counts(sources)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"mixed split counts {counts} != {EXPECTED_COUNTS}")
    if len({source["seed"] for source in sources}) != len(sources):
        raise ValueError("mixed sources contain duplicate seeds")
    builder.build_dataset(sources, args.out.resolve(), args.encode_workers)
    family_counts = dict(sorted(Counter(
        source["scenario"]["scenario_family"]
        for source in v16
    ).items()))
    report = {
        "schema_version": 1,
        "dataset": str(args.out.resolve()),
        "v15_train_count": len(v15),
        "v16_pilot_count": len(v16),
        "total_count": len(sources),
        "split_counts": counts,
        "v16_family_counts": family_counts,
        "old_v15_val_test_excluded": True,
        "old_v15_videos_reused": len(v15) * len(builder.POLICY_CAMERAS),
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
