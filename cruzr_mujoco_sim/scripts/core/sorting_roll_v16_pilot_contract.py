#!/usr/bin/env python3
"""Manifest contract for the 16-episode Sorting Roll v16 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from sorting_roll_diversity import (
    DYNAMICS_PROFILES,
    DIVERSE_TASK_VERSION,
    OBJECT_PROFILES,
    POSE_BINS,
    _assignment_id,
    assignment_errors,
    generate_manifest,
)


SCHEMA_VERSION = 1
TASK_VERSION = "sorting_roll_v16_expansion_pilot_sim"
PILOT_COUNT = 16
SEED_START = 5000
SPLIT_COUNTS = {"train": 12, "val": 2, "test": 2}
FAMILY_COUNTS = {"H": 4, "T": 4, "R": 8}

SCENARIOS = (
    ("T", "clean_grasp_lift", {}),
    ("T", "clean_grasp_lift", {}),
    ("R", "single_hand_contact_left", {}),
    ("H", "pickup_support_height_high", {"z_m": 0.010}),
    ("T", "clean_grasp_lift", {}),
    ("R", "double_hand_miss", {}),
    ("R", "single_hand_contact_right", {}),
    ("H", "pickup_support_height_low", {"z_m": -0.010}),
    ("T", "clean_grasp_lift", {}),
    ("R", "double_hand_miss", {}),
    ("R", "partial_lift_slip_left", {}),
    ("H", "pickup_support_y_positive", {"y_m": 0.010}),
    ("R", "double_hand_miss", {}),
    ("R", "double_hand_miss", {}),
    ("R", "partial_lift_slip_right", {}),
    ("H", "pickup_support_y_negative", {"y_m": -0.010}),
)


def _pilot_assignment_id(assignment):
    payload = dict(assignment)
    payload.pop("assignment_id", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _force_high_risk(base_assignment):
    assignment = copy.deepcopy(base_assignment)
    assignment["object_profile"] = {
        "name": "long_baseline",
        **OBJECT_PROFILES["long_baseline"],
    }
    assignment["dynamics_profile"] = {
        "name": "heavy_low_friction",
        **DYNAMICS_PROFILES["heavy_low_friction"],
    }
    assignment["pose_bin"] = "boundary"
    assignment["assignment_id"] = _assignment_id(assignment)
    return assignment


def generate_pilot_manifest(campaign, seed_start=SEED_START):
    if not campaign or "/" in campaign:
        raise ValueError("campaign must be a non-empty path-free name")
    base_manifest = generate_manifest(campaign, seed_start, PILOT_COUNT)
    assignments = []
    for index, (base, scenario) in enumerate(
        zip(base_manifest["assignments"], SCENARIOS)
    ):
        family, variant, support_transform = scenario
        if family == "T" or variant.startswith("partial_lift"):
            base = _force_high_risk(base)
        intervention_type = None
        if family == "R":
            intervention_type = variant
        assignment = {
            "schema_version": SCHEMA_VERSION,
            "task_version": TASK_VERSION,
            "campaign": campaign,
            "seed": int(base["seed"]),
            "split": base["split"],
            "scenario_family": family,
            "scenario_variant": variant,
            "scene_group_id": f"v16_pilot_{index:02d}",
            "counterfactual_pair_id": None,
            "start_phase": (
                "initial_hold"
                if family == "H"
                else (
                    "approach_table_with_arms_staged"
                    if family == "T"
                    else "recovery_start"
                )
            ),
            "terminal_phase": (
                "terminal_success_hold"
                if family == "H"
                else (
                    "lift_flat_from_pickup_support"
                    if family == "T"
                    else "clear_table"
                )
            ),
            "intervention_type": intervention_type,
            "intervention_frame": -1 if family == "R" else None,
            "recovery_start_frame": 0 if family == "R" else None,
            "target_object_id": "sorting_roll",
            "target_color": base["appearance_profile"]["name"],
            "distractor_object_ids": [],
            "requested_transforms": {
                "pickup_support_and_roll": {
                    "x_m": float(support_transform.get("x_m", 0.0)),
                    "y_m": float(support_transform.get("y_m", 0.0)),
                    "z_m": float(support_transform.get("z_m", 0.0)),
                    "yaw_rad": 0.0,
                }
            },
            "base_diversity_assignment": base,
            "prompt": base["prompt"],
        }
        assignment["assignment_id"] = _pilot_assignment_id(assignment)
        assignments.append(assignment)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_version": TASK_VERSION,
        "campaign": campaign,
        "seed_start": seed_start,
        "count": PILOT_COUNT,
        "assignments": assignments,
        "counts": manifest_counts(assignments),
    }
    errors = manifest_errors(manifest)
    if errors:
        raise ValueError("generated invalid v16 pilot manifest: " + "; ".join(errors))
    return manifest


def manifest_counts(assignments):
    def count(field):
        return dict(sorted(Counter(item[field] for item in assignments).items()))

    return {
        "split": count("split"),
        "scenario_family": count("scenario_family"),
        "scenario_variant": count("scenario_variant"),
    }


def assignment_errors_v16(assignment):
    errors = []
    if not isinstance(assignment, dict):
        return ["assignment is not an object"]
    if assignment.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if assignment.get("task_version") != TASK_VERSION:
        errors.append("task_version mismatch")
    if assignment.get("scenario_family") not in FAMILY_COUNTS:
        errors.append("scenario_family is invalid")
    if assignment.get("split") not in SPLIT_COUNTS:
        errors.append("split is invalid")
    if not assignment.get("scene_group_id"):
        errors.append("scene_group_id is missing")
    if assignment.get("target_object_id") != "sorting_roll":
        errors.append("target_object_id mismatch")
    if assignment.get("distractor_object_ids") != []:
        errors.append("pilot must not contain distractors")
    base = assignment.get("base_diversity_assignment")
    for error in assignment_errors(base):
        errors.append(f"base_diversity_assignment: {error}")
    if isinstance(base, dict):
        for field in ("seed", "split", "campaign", "prompt"):
            if assignment.get(field) != base.get(field):
                errors.append(f"assignment/base {field} mismatch")
        if assignment.get("target_color") != (
            base.get("appearance_profile") or {}
        ).get("name"):
            errors.append("target_color mismatch")
    transform = (
        (assignment.get("requested_transforms") or {})
        .get("pickup_support_and_roll")
    )
    if not isinstance(transform, dict) or set(transform) != {
        "x_m", "y_m", "z_m", "yaw_rad"
    }:
        errors.append("pickup support transform is invalid")
    family = assignment.get("scenario_family")
    if family == "R":
        if assignment.get("intervention_type") != assignment.get("scenario_variant"):
            errors.append("recovery intervention_type mismatch")
        if assignment.get("intervention_frame") != -1:
            errors.append("recovery intervention must end before recording")
        if assignment.get("recovery_start_frame") != 0:
            errors.append("recovery must start at frame zero")
    elif any(
        assignment.get(field) is not None
        for field in (
            "intervention_type", "intervention_frame", "recovery_start_frame"
        )
    ):
        errors.append("non-recovery assignment contains intervention metadata")
    if assignment.get("assignment_id") != _pilot_assignment_id(assignment):
        errors.append("assignment_id mismatch")
    return errors


def manifest_errors(payload):
    if not isinstance(payload, dict):
        return ["manifest is not an object"]
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if payload.get("task_version") != TASK_VERSION:
        errors.append("manifest task_version mismatch")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        return errors + ["assignments is not a list"]
    if payload.get("count") != PILOT_COUNT or len(assignments) != PILOT_COUNT:
        errors.append("pilot must contain exactly 16 assignments")
    seeds = [item.get("seed") for item in assignments if isinstance(item, dict)]
    if seeds != list(range(payload.get("seed_start", 0), payload.get("seed_start", 0) + PILOT_COUNT)):
        errors.append("pilot seeds are not contiguous")
    if len(set(seeds)) != len(seeds):
        errors.append("pilot contains duplicate seeds")
    for index, assignment in enumerate(assignments):
        for error in assignment_errors_v16(assignment):
            errors.append(f"assignment[{index}]: {error}")
        if isinstance(assignment, dict) and assignment.get("campaign") != payload.get("campaign"):
            errors.append(f"assignment[{index}]: campaign mismatch")
    if assignments and manifest_counts(assignments) != payload.get("counts"):
        errors.append("manifest counts mismatch")
    counts = payload.get("counts") or {}
    if counts.get("split") != SPLIT_COUNTS:
        errors.append("pilot split counts mismatch")
    if counts.get("scenario_family") != FAMILY_COUNTS:
        errors.append("pilot family counts mismatch")
    groups = [item.get("scene_group_id") for item in assignments if isinstance(item, dict)]
    if len(set(groups)) != len(groups):
        errors.append("pilot scene_group_id values must be unique")
    return errors


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = manifest_errors(payload)
    if errors:
        raise ValueError(f"invalid v16 pilot manifest {path}: {'; '.join(errors)}")
    return payload


def assignment_for_seed(payload, seed):
    matches = [
        item for item in payload["assignments"] if item["seed"] == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest has {len(matches)} assignments for seed {seed}")
    return matches[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--campaign", required=True)
    generate.add_argument("--seed-start", type=int, default=SEED_START)
    check = subparsers.add_parser("check")
    check.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "generate":
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite manifest: {args.out}")
        payload = generate_pilot_manifest(args.campaign, args.seed_start)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        payload = load_manifest(args.manifest)
    print(json.dumps({
        "campaign": payload["campaign"],
        "count": payload["count"],
        "counts": payload["counts"],
        "passed": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
