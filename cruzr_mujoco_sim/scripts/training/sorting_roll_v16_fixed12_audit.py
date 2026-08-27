#!/usr/bin/env python3
"""Audit the matched 12-scenario Sorting Roll checkpoint comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


LABELS = ("original36", "control3k", "treatment3k")
EXPECTED_FAMILIES = {"nominal": 3, "T": 3, "H": 3, "R": 3}


def summarize(results):
    return {
        "episodes": len(results),
        "families": dict(sorted(Counter(
            item.get("scenario_family") for item in results
        ).items())),
        "reached_grasp_workzone": sum(
            item.get("reached_grasp_workzone") is True for item in results
        ),
        "strict_bimanual_grasp": sum(
            item.get("first_bimanual_grasp_step") is not None for item in results
        ),
        "stable_lift_at_least_70mm": sum(
            item.get("stable_lift_at_least_70mm") is True for item in results
        ),
        "success": sum(item.get("success") is True for item in results),
        "unsafe_collision": sum(
            item.get("unsafe_collision") is True for item in results
        ),
        "continuous_rotation": sum(
            item.get("continuous_rotation") is True for item in results
        ),
    }


def absolute_gate(summary):
    return (
        summary["reached_grasp_workzone"] >= 10
        and summary["strict_bimanual_grasp"] >= 9
        and summary["stable_lift_at_least_70mm"] >= 8
        and summary["success"] >= 6
        and summary["unsafe_collision"] == 0
        and summary["continuous_rotation"] == 0
    )


def matched_improvement(control, treatment):
    safety_not_worse = (
        treatment["unsafe_collision"] <= control["unsafe_collision"]
        and treatment["continuous_rotation"] <= control["continuous_rotation"]
    )
    stage_not_worse = (
        treatment["strict_bimanual_grasp"] >= control["strict_bimanual_grasp"]
        and treatment["stable_lift_at_least_70mm"]
        >= control["stable_lift_at_least_70mm"]
        and treatment["success"] >= control["success"]
    )
    strict_gain = any((
        treatment["strict_bimanual_grasp"] > control["strict_bimanual_grasp"],
        treatment["stable_lift_at_least_70mm"]
        > control["stable_lift_at_least_70mm"],
        treatment["success"] > control["success"],
    ))
    return safety_not_worse and stage_not_worse and strict_gain


def audit(root):
    root = Path(root).resolve()
    errors = []
    summaries = {}
    result_paths = {}
    for label in LABELS:
        paths = sorted((root / label).glob("*/result.json"))
        results = []
        keys = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("checkpoint_label") != label:
                errors.append(f"{path}: checkpoint_label mismatch")
            key = (payload.get("manifest_kind"), payload.get("seed"))
            if key in keys:
                errors.append(f"{path}: duplicate scenario {key}")
            keys.add(key)
            results.append(payload)
        summary = summarize(results)
        if summary["episodes"] != 12:
            errors.append(
                f"{label}: expected 12 results, got {summary['episodes']}"
            )
        if summary["families"] != EXPECTED_FAMILIES:
            errors.append(
                f"{label}: family counts {summary['families']} "
                f"!= {EXPECTED_FAMILIES}"
            )
        summaries[label] = summary
        result_paths[label] = [str(path) for path in paths]

    ready = False
    if not errors:
        ready = absolute_gate(summaries["treatment3k"]) and matched_improvement(
            summaries["control3k"], summaries["treatment3k"]
        )
    return {
        "schema_version": 1,
        "purpose": "sorting_roll_v16_fixed12_matched_checkpoint_gate",
        "root": str(root),
        "passed": not errors,
        "errors": errors,
        "summaries": summaries,
        "result_paths": result_paths,
        "treatment_absolute_gate": (
            absolute_gate(summaries["treatment3k"]) if not errors else False
        ),
        "treatment_improves_matched_control": (
            matched_improvement(
                summaries["control3k"], summaries["treatment3k"]
            )
            if not errors else False
        ),
        "ready_to_expand": ready,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    if args.require_ready and not report["ready_to_expand"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
