#!/usr/bin/env python3
"""Validate grouped stage80/stage160 H/T/R/C expansion episodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
sys.path[:0] = [str(SCRIPT_DIR), str(CORE_DIR)]

from sorting_roll_v16_expansion_contract import (  # noqa: E402
    TASK_VERSION,
    load_manifest,
)
from sorting_roll_v16_validate import validate_episode  # noqa: E402
from sorting_roll_validate import TASK  # noqa: E402


def counterfactual_pair_errors(manifest, records):
    errors = []
    by_seed = {
        record["info"]["seed"]: record
        for record in records if record.get("info")
    }
    groups = defaultdict(list)
    for assignment in manifest["assignments"]:
        if assignment["scenario_family"] == "C":
            groups[assignment["counterfactual_pair_id"]].append(assignment)
    for pair_id, assignments in groups.items():
        present = [item for item in assignments if item["seed"] in by_seed]
        if not present:
            continue
        if len(present) != 2:
            errors.append(f"counterfactual pair is incomplete: {pair_id}")
            continue
        if any(not by_seed[item["seed"]]["passed"] for item in present):
            errors.append(f"counterfactual pair contains failed episode: {pair_id}")
            continue
        scenes = [item["counterfactual_scene"] for item in present]
        if (
            {item["target_lane"] for item in present} != {"left", "right"}
            or len({item["split"] for item in present}) != 1
            or scenes[0]["lane_x_m"] != scenes[1]["lane_x_m"]
            or scenes[0]["lane_colors"] != scenes[1]["lane_colors"]
            or scenes[0]["scene_randomization_seed"]
            != scenes[1]["scene_randomization_seed"]
            or len({item["prompt"] for item in present}) != 2
        ):
            errors.append(f"counterfactual pair invariants failed: {pair_id}")
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    records = []
    for assignment in manifest["assignments"]:
        episode = args.episode_root / f"seed_{assignment['seed']}"
        if not (episode / "result.json").is_file():
            if args.require_complete:
                records.append({
                    "path": str(episode.resolve()),
                    "passed": False,
                    "info": None,
                    "errors": ["episode is missing"],
                })
            continue
        info, errors = validate_episode(episode, assignment)
        records.append({
            "path": str(episode.resolve()),
            "passed": not errors,
            "info": info,
            "errors": errors,
        })
        print(
            f"[v16 expansion validate] seed={assignment['seed']} "
            f"{'PASS' if not errors else 'FAIL'}"
            + ("" if not errors else f" {'; '.join(errors)}"),
            flush=True,
        )
    pair_errors = counterfactual_pair_errors(manifest, records)
    passed_count = sum(record["passed"] for record in records)
    family_counts = Counter(
        record["info"]["scenario_family"]
        for record in records
        if record["passed"] and record.get("info")
    )
    complete = len(records) == manifest["count"]
    report = {
        "schema_version": 1,
        "task": TASK,
        "task_version": TASK_VERSION,
        "campaign": manifest["campaign"],
        "stage": manifest["stage"],
        "manifest": str(args.manifest.resolve()),
        "episode_root": str(args.episode_root.resolve()),
        "episode_count": len(records),
        "passed_count": passed_count,
        "failed_count": len(records) - passed_count,
        "family_counts": dict(sorted(family_counts.items())),
        "counterfactual_pair_errors": pair_errors,
        "complete": complete,
        "passed": bool(records)
        and passed_count == len(records)
        and not pair_errors,
        "records": records,
    }
    if args.require_complete and not complete:
        report["passed"] = False
    if args.report:
        if args.report.exists():
            raise SystemExit(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        key: report[key]
        for key in (
            "episode_count", "passed_count", "failed_count", "complete",
            "counterfactual_pair_errors", "passed",
        )
    }, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
