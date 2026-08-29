#!/usr/bin/env python3
"""Grouped H/T/R/C manifest contract for Sorting Roll v16 expansion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from sorting_roll_diversity import (
    APPEARANCE_PROFILES,
    DYNAMICS_PROFILES,
    OBJECT_PROFILES,
    _assignment_id,
    assignment_errors,
    generate_manifest,
)


SCHEMA_VERSION = 1
TASK_VERSION = "sorting_roll_v16_expansion_stage_sim"
DEFAULT_STAGE = 80
DEFAULT_SEED_START = 6000
STAGE_COUNTS = {
    80: {
        "split": {"train": 64, "val": 8, "test": 8},
        "family": {"H": 20, "T": 20, "R": 28, "C": 12},
        "family_split": {
            "train": {"H": 16, "T": 16, "R": 24, "C": 8},
            "val": {"H": 2, "T": 2, "R": 2, "C": 2},
            "test": {"H": 2, "T": 2, "R": 2, "C": 2},
        },
    },
    160: {
        "split": {"train": 128, "val": 16, "test": 16},
        "family": {"H": 40, "T": 40, "R": 56, "C": 24},
        "family_split": {
            "train": {"H": 32, "T": 32, "R": 48, "C": 16},
            "val": {"H": 4, "T": 4, "R": 4, "C": 4},
            "test": {"H": 4, "T": 4, "R": 4, "C": 4},
        },
    },
}

H_VARIANTS = (
    ("pickup_support_height_high", {"z_m": 0.010}),
    ("pickup_support_height_low", {"z_m": -0.010}),
    ("pickup_support_y_positive", {"y_m": 0.010}),
    ("pickup_support_y_negative", {"y_m": -0.010}),
)
R_VARIANTS = (
    "double_hand_miss",
    "single_hand_contact_left",
    "single_hand_contact_right",
    "partial_lift_slip_left",
    "partial_lift_slip_right",
)
COLOR_PAIRS = (
    ("red", "blue"),
    ("green", "yellow"),
    ("orange", "blue"),
    ("red", "green"),
    ("yellow", "blue"),
    ("orange", "green"),
)
LANE_X_M = {"left": -0.62, "right": 0.0}


def _stable_id(assignment):
    payload = dict(assignment)
    payload.pop("assignment_id", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _transform(values=None):
    values = values or {}
    return {
        "x_m": float(values.get("x_m", 0.0)),
        "y_m": float(values.get("y_m", 0.0)),
        "z_m": float(values.get("z_m", 0.0)),
        "yaw_rad": 0.0,
    }


def _forced_base(source, seed, *, color=None, high_risk=False):
    base = copy.deepcopy(source)
    base["seed"] = int(seed)
    if high_risk or color is not None:
        base["object_profile"] = {
            "name": "long_baseline",
            **OBJECT_PROFILES["long_baseline"],
        }
    if high_risk:
        base["dynamics_profile"] = {
            "name": "heavy_low_friction",
            **DYNAMICS_PROFILES["heavy_low_friction"],
        }
        base["pose_bin"] = "boundary"
    elif color is not None:
        base["dynamics_profile"] = {
            "name": "nominal",
            **DYNAMICS_PROFILES["nominal"],
        }
        base["pose_bin"] = "easy"
        base["appearance_profile"] = {
            "name": color,
            **APPEARANCE_PROFILES[color],
        }
    base["assignment_id"] = _assignment_id(base)
    return base


def _descriptors(stage, split):
    counts = STAGE_COUNTS[stage]["family_split"][split]
    items = []
    for index in range(counts["H"]):
        variant, transform = H_VARIANTS[index % len(H_VARIANTS)]
        items.append({
            "family": "H",
            "variant": variant,
            "transform": transform,
        })
    items.extend(
        {"family": "T", "variant": "clean_grasp_lift", "transform": {}}
        for _ in range(counts["T"])
    )
    items.extend(
        {
            "family": "R",
            "variant": R_VARIANTS[index % len(R_VARIANTS)],
            "transform": {},
        }
        for index in range(counts["R"])
    )
    if counts["C"] % 2:
        raise ValueError("counterfactual family count must be even per split")
    for pair_index in range(counts["C"] // 2):
        left_color, right_color = COLOR_PAIRS[pair_index % len(COLOR_PAIRS)]
        pair_id = f"v16_stage{stage}_{split}_c_{pair_index:02d}"
        shared = {
            "family": "C",
            "variant": "color_target_counterfactual",
            "transform": {},
            "pair_id": pair_id,
            "lane_colors": {"left": left_color, "right": right_color},
        }
        items.append({**shared, "target_lane": "left"})
        items.append({**shared, "target_lane": "right"})
    return items


def _assignment(campaign, stage, index, base, descriptor, scene_seed=None):
    family = descriptor["family"]
    target_lane = descriptor.get("target_lane")
    lane_colors = descriptor.get("lane_colors")
    target_color = base["appearance_profile"]["name"]
    distractor_color = None
    counterfactual_scene = None
    pair_id = descriptor.get("pair_id")
    prompt = base["prompt"]
    distractors = []
    if family == "C":
        target_color = lane_colors[target_lane]
        distractor_lane = "right" if target_lane == "left" else "left"
        distractor_color = lane_colors[distractor_lane]
        distractors = ["sorting_roll_distractor"]
        prompt = (
            f"Pick up the {target_color} roll and place it stably in the "
            "integrated top shelf slot"
        )
        counterfactual_scene = {
            "target_lane": target_lane,
            "distractor_lane": distractor_lane,
            "lane_x_m": dict(LANE_X_M),
            "lane_colors": dict(lane_colors),
            "scene_randomization_seed": int(scene_seed),
        }
    assignment = {
        "schema_version": SCHEMA_VERSION,
        "task_version": TASK_VERSION,
        "campaign": campaign,
        "stage": stage,
        "seed": int(base["seed"]),
        "split": base["split"],
        "scenario_family": family,
        "scenario_variant": descriptor["variant"],
        "scene_group_id": pair_id or f"v16_stage{stage}_{index:03d}",
        "counterfactual_pair_id": pair_id,
        "start_phase": (
            "initial_hold"
            if family in ("H", "C")
            else (
                "approach_table_with_arms_staged"
                if family == "T"
                else "recovery_start"
            )
        ),
        "terminal_phase": "terminal_success_hold",
        "intervention_type": descriptor["variant"] if family == "R" else None,
        "intervention_frame": -1 if family == "R" else None,
        "recovery_start_frame": 0 if family == "R" else None,
        "target_object_id": "sorting_roll",
        "target_color": target_color,
        "target_lane": target_lane,
        "distractor_object_ids": distractors,
        "distractor_color": distractor_color,
        "counterfactual_scene": counterfactual_scene,
        "requested_transforms": {
            "pickup_support_and_roll": _transform(descriptor["transform"]),
        },
        "base_diversity_assignment": base,
        "prompt": prompt,
    }
    assignment["assignment_id"] = _stable_id(assignment)
    return assignment


def generate_expansion_manifest(
    campaign,
    stage=DEFAULT_STAGE,
    seed_start=DEFAULT_SEED_START,
):
    if stage not in STAGE_COUNTS:
        raise ValueError(f"unsupported stage: {stage}")
    if seed_start <= 0 or seed_start % 10:
        raise ValueError("seed_start must be a positive multiple of 10")
    count = stage
    base_manifest = generate_manifest(campaign, seed_start, count)
    bases = {item["seed"]: item for item in base_manifest["assignments"]}
    seeds_by_split = defaultdict(list)
    for base in base_manifest["assignments"]:
        seeds_by_split[base["split"]].append(base["seed"])

    assignments = []
    for split in ("train", "val", "test"):
        descriptors = _descriptors(stage, split)
        seeds = seeds_by_split[split]
        if len(seeds) != len(descriptors):
            raise ValueError(
                f"stage {stage} split {split} has {len(seeds)} seeds but "
                f"requires {len(descriptors)} assignments"
            )
        pair_templates = {}
        pair_scene_seeds = {}
        for seed, descriptor in zip(seeds, descriptors):
            pair_id = descriptor.get("pair_id")
            if pair_id and pair_id not in pair_templates:
                pair_templates[pair_id] = bases[seed]
                pair_scene_seeds[pair_id] = seed
        for seed, descriptor in zip(seeds, descriptors):
            base = bases[seed]
            family = descriptor["family"]
            high_risk = family == "T" or descriptor["variant"].startswith(
                "partial_lift"
            )
            scene_seed = None
            if family == "C":
                pair_id = descriptor["pair_id"]
                target_color = descriptor["lane_colors"][
                    descriptor["target_lane"]
                ]
                base = _forced_base(
                    pair_templates[pair_id], seed, color=target_color
                )
                scene_seed = pair_scene_seeds[pair_id]
            elif high_risk:
                base = _forced_base(base, seed, high_risk=True)
            assignments.append(
                _assignment(
                    campaign,
                    stage,
                    len(assignments),
                    base,
                    descriptor,
                    scene_seed=scene_seed,
                )
            )
    assignments.sort(key=lambda item: item["seed"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_version": TASK_VERSION,
        "campaign": campaign,
        "stage": stage,
        "seed_start": seed_start,
        "count": count,
        "assignments": assignments,
        "counts": manifest_counts(assignments),
    }
    errors = manifest_errors(payload)
    if errors:
        raise ValueError("generated invalid expansion manifest: " + "; ".join(errors))
    return payload


def manifest_counts(assignments):
    def count(field):
        return dict(sorted(Counter(item[field] for item in assignments).items()))

    return {
        "split": count("split"),
        "scenario_family": count("scenario_family"),
        "scenario_variant": count("scenario_variant"),
    }


def assignment_errors_v16_expansion(assignment):
    if not isinstance(assignment, dict):
        return ["assignment is not an object"]
    errors = []
    if assignment.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if assignment.get("task_version") != TASK_VERSION:
        errors.append("task_version mismatch")
    stage = assignment.get("stage")
    if stage not in STAGE_COUNTS:
        errors.append("stage is invalid")
    family = assignment.get("scenario_family")
    if family not in ("H", "T", "R", "C"):
        errors.append("scenario_family is invalid")
    base = assignment.get("base_diversity_assignment")
    for error in assignment_errors(base):
        errors.append(f"base_diversity_assignment: {error}")
    if isinstance(base, dict):
        for field in ("seed", "split", "campaign"):
            if assignment.get(field) != base.get(field):
                errors.append(f"assignment/base {field} mismatch")
        if assignment.get("target_color") != (
            base.get("appearance_profile") or {}
        ).get("name"):
            errors.append("target_color mismatch")
        if family != "C" and assignment.get("prompt") != base.get("prompt"):
            errors.append("assignment/base prompt mismatch")
    transform = (
        (assignment.get("requested_transforms") or {})
        .get("pickup_support_and_roll")
    )
    if not isinstance(transform, dict) or set(transform) != {
        "x_m", "y_m", "z_m", "yaw_rad"
    }:
        errors.append("pickup support transform is invalid")
    if family == "R":
        if assignment.get("intervention_type") != assignment.get("scenario_variant"):
            errors.append("recovery intervention_type mismatch")
        if assignment.get("intervention_frame") != -1:
            errors.append("recovery intervention must end before recording")
        if assignment.get("recovery_start_frame") != 0:
            errors.append("recovery must start at frame zero")
    elif any(
        assignment.get(field) is not None
        for field in ("intervention_type", "intervention_frame", "recovery_start_frame")
    ):
        errors.append("non-recovery assignment contains intervention metadata")
    if family == "C":
        scene = assignment.get("counterfactual_scene")
        lane = assignment.get("target_lane")
        if assignment.get("counterfactual_pair_id") != assignment.get("scene_group_id"):
            errors.append("counterfactual pair/group mismatch")
        if assignment.get("distractor_object_ids") != ["sorting_roll_distractor"]:
            errors.append("counterfactual distractor id mismatch")
        if lane not in LANE_X_M or not isinstance(scene, dict):
            errors.append("counterfactual target lane/scene is invalid")
        else:
            colors = scene.get("lane_colors") or {}
            other = "right" if lane == "left" else "left"
            if scene.get("target_lane") != lane or scene.get("distractor_lane") != other:
                errors.append("counterfactual lane binding mismatch")
            if scene.get("lane_x_m") != LANE_X_M:
                errors.append("counterfactual lane positions mismatch")
            if colors.get(lane) != assignment.get("target_color"):
                errors.append("counterfactual target color mismatch")
            if colors.get(other) != assignment.get("distractor_color"):
                errors.append("counterfactual distractor color mismatch")
            expected_prompt = (
                f"Pick up the {assignment.get('target_color')} roll and place it "
                "stably in the integrated top shelf slot"
            )
            if assignment.get("prompt") != expected_prompt:
                errors.append("counterfactual prompt mismatch")
    elif any(
        assignment.get(field) is not None
        for field in (
            "counterfactual_pair_id", "target_lane", "distractor_color",
            "counterfactual_scene",
        )
    ) or assignment.get("distractor_object_ids") != []:
        errors.append("non-counterfactual assignment contains distractor metadata")
    if assignment.get("target_object_id") != "sorting_roll":
        errors.append("target_object_id mismatch")
    if assignment.get("assignment_id") != _stable_id(assignment):
        errors.append("assignment_id mismatch")
    return errors


def manifest_errors(payload):
    if not isinstance(payload, dict):
        return ["manifest is not an object"]
    errors = []
    stage = payload.get("stage")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if payload.get("task_version") != TASK_VERSION:
        errors.append("manifest task_version mismatch")
    if stage not in STAGE_COUNTS:
        return errors + ["manifest stage is invalid"]
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        return errors + ["assignments is not a list"]
    if payload.get("count") != stage or len(assignments) != stage:
        errors.append("manifest count does not match stage")
    seeds = [item.get("seed") for item in assignments if isinstance(item, dict)]
    expected = list(range(payload.get("seed_start", 0), payload.get("seed_start", 0) + stage))
    if seeds != expected:
        errors.append("manifest seeds are not the expected contiguous range")
    for index, assignment in enumerate(assignments):
        for error in assignment_errors_v16_expansion(assignment):
            errors.append(f"assignment[{index}]: {error}")
        if isinstance(assignment, dict) and assignment.get("campaign") != payload.get("campaign"):
            errors.append(f"assignment[{index}]: campaign mismatch")
    counts = payload.get("counts") or {}
    if assignments and manifest_counts(assignments) != counts:
        errors.append("manifest counts mismatch")
    expected_counts = STAGE_COUNTS[stage]
    if counts.get("split") != expected_counts["split"]:
        errors.append("manifest split counts mismatch")
    if counts.get("scenario_family") != expected_counts["family"]:
        errors.append("manifest family counts mismatch")
    family_split = Counter(
        (item.get("split"), item.get("scenario_family"))
        for item in assignments if isinstance(item, dict)
    )
    for split, family_counts in expected_counts["family_split"].items():
        for family, count in family_counts.items():
            if family_split[(split, family)] != count:
                errors.append(f"manifest {split}/{family} count mismatch")
    groups = defaultdict(list)
    for item in assignments:
        if isinstance(item, dict):
            groups[item.get("scene_group_id")].append(item)
    for group_id, members in groups.items():
        if not group_id:
            errors.append("scene_group_id is missing")
            continue
        families = {item.get("scenario_family") for item in members}
        splits = {item.get("split") for item in members}
        if len(splits) != 1:
            errors.append(f"scene group crosses splits: {group_id}")
        if families == {"C"}:
            if len(members) != 2 or {item.get("target_lane") for item in members} != {"left", "right"}:
                errors.append(f"counterfactual group is not a complete pair: {group_id}")
            scenes = [item.get("counterfactual_scene") or {} for item in members]
            if len({json.dumps(scene.get("lane_colors"), sort_keys=True) for scene in scenes}) != 1:
                errors.append(f"counterfactual pair colors differ: {group_id}")
            if len({scene.get("scene_randomization_seed") for scene in scenes}) != 1:
                errors.append(f"counterfactual pair scene seeds differ: {group_id}")
        elif len(members) != 1:
            errors.append(f"non-counterfactual scene group is duplicated: {group_id}")
    return errors


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = manifest_errors(payload)
    if errors:
        raise ValueError(f"invalid expansion manifest {path}: {'; '.join(errors)}")
    return payload


def assignment_for_seed(payload, seed):
    matches = [
        item for item in payload["assignments"] if item["seed"] == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest has {len(matches)} assignments for seed {seed}")
    return matches[0]


def representative_seeds(payload):
    selected = []
    for family in ("H", "T", "R"):
        selected.append(next(
            item["seed"] for item in payload["assignments"]
            if item["split"] == "train" and item["scenario_family"] == family
        ))
    pair = next(
        item["counterfactual_pair_id"] for item in payload["assignments"]
        if item["split"] == "train" and item["scenario_family"] == "C"
    )
    selected.extend(
        item["seed"] for item in payload["assignments"]
        if item["counterfactual_pair_id"] == pair
    )
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--campaign", required=True)
    generate.add_argument("--stage", type=int, choices=tuple(STAGE_COUNTS), default=DEFAULT_STAGE)
    generate.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    check = subparsers.add_parser("check")
    check.add_argument("manifest", type=Path)
    select = subparsers.add_parser("select")
    select.add_argument("manifest", type=Path)
    select.add_argument("--mode", choices=("representative", "all"), default="representative")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "generate":
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite manifest: {args.out}")
        payload = generate_expansion_manifest(args.campaign, args.stage, args.seed_start)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "campaign": payload["campaign"],
            "stage": payload["stage"],
            "counts": payload["counts"],
            "passed": True,
        }, ensure_ascii=False, indent=2))
        return 0
    payload = load_manifest(args.manifest)
    if args.action == "select":
        seeds = (
            representative_seeds(payload)
            if args.mode == "representative"
            else [item["seed"] for item in payload["assignments"]]
        )
        print("\n".join(str(seed) for seed in seeds))
        return 0
    print(json.dumps({
        "campaign": payload["campaign"],
        "stage": payload["stage"],
        "counts": payload["counts"],
        "passed": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
