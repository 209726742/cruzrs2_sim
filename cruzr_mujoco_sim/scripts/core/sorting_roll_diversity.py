#!/usr/bin/env python3
"""Versioned diversity manifest and MuJoCo model overrides for Sorting Roll."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np


SCHEMA_VERSION = 1
DIVERSE_TASK_VERSION = "sorting_roll_v15_diverse_sim"
BASE_ROLL_LENGTH_M = 0.500
BASE_ROLL_DIAMETER_M = 0.025

OBJECT_PROFILES = {
    "short_slim": {
        "length_m": 0.4674,
        "diameter_m": 0.0225,
        "shelf_width_ratio": 0.82,
    },
    "medium": {
        "length_m": 0.4845,
        "diameter_m": 0.0240,
        "shelf_width_ratio": 0.85,
    },
    "long_baseline": {
        "length_m": 0.5000,
        "diameter_m": 0.0240,
        "shelf_width_ratio": 0.877193,
    },
}

DYNAMICS_PROFILES = {
    "nominal": {
        "mass_kg": 0.2500,
        "sliding_friction": 1.2500,
    },
    "light_high_friction": {
        "mass_kg": 0.2125,
        "sliding_friction": 1.4375,
    },
    "heavy_low_friction": {
        "mass_kg": 0.2875,
        "sliding_friction": 1.0625,
    },
}

APPEARANCE_PROFILES = {
    "red": {"rgba": [0.95, 0.10, 0.18, 1.0]},
    "orange": {"rgba": [1.00, 0.32, 0.06, 1.0]},
    "yellow": {"rgba": [0.95, 0.72, 0.08, 1.0]},
    "green": {"rgba": [0.10, 0.68, 0.24, 1.0]},
    "blue": {"rgba": [0.10, 0.32, 0.92, 1.0]},
}

LIGHTING_PROFILES = {
    "normal": {"diffuse_scale": 1.00},
    "dim": {"diffuse_scale": 0.82},
    "bright": {"diffuse_scale": 1.18},
}

IMAGE_PROFILES = {
    "clean": {"jpeg_quality": 92},
    "mild_compression": {"jpeg_quality": 84},
    "strong_compression": {"jpeg_quality": 76},
}

POSE_BINS = {
    "easy": {"normalized_min": 0.00, "normalized_max": 0.40},
    "medium": {"normalized_min": 0.40, "normalized_max": 0.75},
    "boundary": {"normalized_min": 0.75, "normalized_max": 1.00},
}

PROMPTS = {
    "prompt_0": "Pick up the roll and place it stably in the integrated top shelf slot",
    "prompt_1": "Grasp the roll and set it securely in the top integrated shelf slot",
    "prompt_2": "Move the roll from the table into the integrated slot on the top shelf",
    "prompt_3": "Place the roll securely inside the top shelf's integrated slot",
    "prompt_4": "Pick up the rod and leave it stable in the integrated top shelf slot",
}

ADMISSION_GROUPS = {
    "geometry_short": {
        "object_profile": "short_slim",
        "dynamics_profile": "nominal",
    },
    "geometry_medium": {
        "object_profile": "medium",
        "dynamics_profile": "nominal",
    },
    "geometry_long": {
        "object_profile": "long_baseline",
        "dynamics_profile": "nominal",
    },
    "dynamics_light_high_friction": {
        "object_profile": "long_baseline",
        "dynamics_profile": "light_high_friction",
    },
    "dynamics_heavy_low_friction": {
        "object_profile": "long_baseline",
        "dynamics_profile": "heavy_low_friction",
    },
}


def source_split(seed):
    seed = int(seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    if seed % 10 == 1:
        return "val"
    if seed % 10 == 0:
        return "test"
    return "train"


def _rng(campaign, dimension, split):
    digest = hashlib.sha256(
        f"{campaign}:{dimension}:{split}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _balanced_values(names, weights, count, rng):
    names = tuple(names)
    weights = tuple(int(value) for value in weights)
    if len(names) != len(weights) or not names or any(value <= 0 for value in weights):
        raise ValueError("balanced value names/weights are invalid")
    total = sum(weights)
    exact = [count * value / total for value in weights]
    counts = [int(value) for value in exact]
    remainder = count - sum(counts)
    order = sorted(
        range(len(names)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    values = [
        name
        for name, item_count in zip(names, counts)
        for _ in range(item_count)
    ]
    rng.shuffle(values)
    return values


def _stratified_values(seeds, campaign, dimension, names, weights):
    values = [None] * len(seeds)
    for split in ("train", "val", "test"):
        indices = [
            index for index, seed in enumerate(seeds)
            if source_split(seed) == split
        ]
        selected = _balanced_values(
            names,
            weights,
            len(indices),
            _rng(campaign, dimension, split),
        )
        for index, value in zip(indices, selected):
            values[index] = value
    if any(value is None for value in values):
        raise RuntimeError(f"failed to assign diversity dimension {dimension}")
    return values


def _assignment_id(assignment):
    payload = dict(assignment)
    payload.pop("assignment_id", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def generate_manifest(campaign, seed_start, count, admission_group=None):
    if not campaign or "/" in campaign:
        raise ValueError("campaign must be a non-empty path-free name")
    if seed_start <= 0 or count <= 0:
        raise ValueError("seed_start and count must be positive")
    if admission_group is not None and admission_group not in ADMISSION_GROUPS:
        raise ValueError(f"unknown admission group: {admission_group}")

    seeds = list(range(seed_start, seed_start + count))
    forced = ADMISSION_GROUPS.get(admission_group, {})
    dimensions = {
        "object_profile": _stratified_values(
            seeds, campaign, "object", OBJECT_PROFILES, (1, 1, 1)
        ),
        "pose_bin": _stratified_values(
            seeds, campaign, "pose", POSE_BINS, (2, 2, 1)
        ),
        "appearance_profile": _stratified_values(
            seeds, campaign, "appearance", APPEARANCE_PROFILES, (1, 1, 1, 1, 1)
        ),
        "lighting_profile": _stratified_values(
            seeds, campaign, "lighting", LIGHTING_PROFILES, (3, 1, 1)
        ),
        "dynamics_profile": _stratified_values(
            seeds, campaign, "dynamics", DYNAMICS_PROFILES, (3, 1, 1)
        ),
        "image_profile": _stratified_values(
            seeds, campaign, "image", IMAGE_PROFILES, (3, 1, 1)
        ),
        "prompt_id": _stratified_values(
            seeds, campaign, "prompt", PROMPTS, (1, 1, 1, 1, 1)
        ),
    }
    for name, value in forced.items():
        dimensions[name] = [value] * count

    assignments = []
    for index, seed in enumerate(seeds):
        object_name = dimensions["object_profile"][index]
        appearance_name = dimensions["appearance_profile"][index]
        lighting_name = dimensions["lighting_profile"][index]
        dynamics_name = dimensions["dynamics_profile"][index]
        image_name = dimensions["image_profile"][index]
        prompt_id = dimensions["prompt_id"][index]
        assignment = {
            "schema_version": SCHEMA_VERSION,
            "task_version": DIVERSE_TASK_VERSION,
            "campaign": campaign,
            "seed": seed,
            "split": source_split(seed),
            "admission_group": admission_group,
            "object_profile": {
                "name": object_name,
                **OBJECT_PROFILES[object_name],
            },
            "pose_bin": dimensions["pose_bin"][index],
            "appearance_profile": {
                "name": appearance_name,
                **APPEARANCE_PROFILES[appearance_name],
            },
            "lighting_profile": {
                "name": lighting_name,
                **LIGHTING_PROFILES[lighting_name],
            },
            "dynamics_profile": {
                "name": dynamics_name,
                **DYNAMICS_PROFILES[dynamics_name],
            },
            "image_profile": {
                "name": image_name,
                **IMAGE_PROFILES[image_name],
            },
            "prompt_id": prompt_id,
            "prompt": PROMPTS[prompt_id],
        }
        assignment["assignment_id"] = _assignment_id(assignment)
        assignments.append(assignment)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_version": DIVERSE_TASK_VERSION,
        "campaign": campaign,
        "seed_start": seed_start,
        "count": count,
        "admission_group": admission_group,
        "assignments": assignments,
        "counts": manifest_counts(assignments),
    }
    errors = manifest_errors(payload)
    if errors:
        raise ValueError("generated manifest is invalid: " + "; ".join(errors))
    return payload


def manifest_counts(assignments):
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
        for assignment in assignments:
            value = assignment[field]
            values.append(value["name"] if isinstance(value, dict) else value)
        counts[field] = dict(sorted(Counter(values).items()))
    return counts


def assignment_errors(assignment):
    errors = []
    if not isinstance(assignment, dict):
        return ["assignment is not an object"]
    if assignment.get("schema_version") != SCHEMA_VERSION:
        errors.append("assignment schema_version mismatch")
    if assignment.get("task_version") != DIVERSE_TASK_VERSION:
        errors.append("assignment task_version mismatch")
    seed = assignment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        errors.append("assignment seed is invalid")
    elif assignment.get("split") != source_split(seed):
        errors.append("assignment split does not match seed")
    for field, catalog in (
        ("object_profile", OBJECT_PROFILES),
        ("appearance_profile", APPEARANCE_PROFILES),
        ("lighting_profile", LIGHTING_PROFILES),
        ("dynamics_profile", DYNAMICS_PROFILES),
        ("image_profile", IMAGE_PROFILES),
    ):
        value = assignment.get(field)
        name = value.get("name") if isinstance(value, dict) else None
        expected = {"name": name, **catalog.get(name, {})}
        if name not in catalog or value != expected:
            errors.append(f"assignment {field} is invalid")
    if assignment.get("pose_bin") not in POSE_BINS:
        errors.append("assignment pose_bin is invalid")
    prompt_id = assignment.get("prompt_id")
    if prompt_id not in PROMPTS or assignment.get("prompt") != PROMPTS.get(prompt_id):
        errors.append("assignment prompt is invalid")
    group = assignment.get("admission_group")
    if group is not None:
        expected = ADMISSION_GROUPS.get(group)
        if expected is None:
            errors.append("assignment admission_group is invalid")
        elif any(
            not isinstance(assignment.get(field), dict)
            or assignment[field].get("name") != value
            for field, value in expected.items()
        ):
            errors.append("assignment does not satisfy admission_group")
    if assignment.get("assignment_id") != _assignment_id(assignment):
        errors.append("assignment_id mismatch")
    return errors


def manifest_errors(payload):
    errors = []
    if not isinstance(payload, dict):
        return ["manifest is not an object"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if payload.get("task_version") != DIVERSE_TASK_VERSION:
        errors.append("manifest task_version mismatch")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        return errors + ["manifest assignments is not a list"]
    if payload.get("count") != len(assignments):
        errors.append("manifest count mismatch")
    seeds = [item.get("seed") for item in assignments if isinstance(item, dict)]
    if len(seeds) != len(set(seeds)):
        errors.append("manifest contains duplicate seeds")
    expected_seeds = list(range(
        payload.get("seed_start", 0),
        payload.get("seed_start", 0) + len(assignments),
    ))
    if seeds != expected_seeds:
        errors.append("manifest seeds are not the expected contiguous range")
    for index, assignment in enumerate(assignments):
        for error in assignment_errors(assignment):
            errors.append(f"assignment[{index}]: {error}")
        if not isinstance(assignment, dict):
            continue
        if assignment.get("campaign") != payload.get("campaign"):
            errors.append(f"assignment[{index}]: campaign mismatch")
        if assignment.get("admission_group") != payload.get("admission_group"):
            errors.append(f"assignment[{index}]: admission_group mismatch")
    assignments_are_valid = all(
        isinstance(assignment, dict) and not assignment_errors(assignment)
        for assignment in assignments
    )
    if (
        assignments_are_valid
        and payload.get("counts") != manifest_counts(assignments)
    ):
        errors.append("manifest counts mismatch")
    return errors


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = manifest_errors(payload)
    if errors:
        raise ValueError(f"invalid diversity manifest {path}: {'; '.join(errors)}")
    return payload


def assignment_for_seed(payload, seed):
    matches = [
        assignment for assignment in payload["assignments"]
        if assignment["seed"] == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest has {len(matches)} assignments for seed {seed}")
    return matches[0]


def replacement_assignment(source, seed):
    errors = assignment_errors(source)
    if errors:
        raise ValueError("invalid source assignment: " + "; ".join(errors))
    replacement = copy.deepcopy(source)
    replacement["seed"] = int(seed)
    replacement["split"] = source_split(seed)
    replacement.pop("assignment_id", None)
    replacement["assignment_id"] = _assignment_id(replacement)
    errors = assignment_errors(replacement)
    if errors:
        raise ValueError("invalid replacement assignment: " + "; ".join(errors))
    return replacement


def apply_model_diversity(mujoco, model, data, assignment):
    errors = assignment_errors(assignment)
    if errors:
        raise ValueError("invalid diversity assignment: " + "; ".join(errors))

    def named(object_type, name):
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"scene is missing {name}")
        return object_id

    roll_body = named(mujoco.mjtObj.mjOBJ_BODY, "sorting_roll")
    roll_visual = named(mujoco.mjtObj.mjOBJ_GEOM, "sorting_roll_visual")
    roll_collider = named(mujoco.mjtObj.mjOBJ_GEOM, "sorting_roll_col")
    object_profile = assignment["object_profile"]
    dynamics = assignment["dynamics_profile"]
    length = float(object_profile["length_m"])
    diameter = float(object_profile["diameter_m"])
    radius = 0.5 * diameter
    mass = float(dynamics["mass_kg"])

    model.geom_size[roll_collider, 0] = radius
    model.geom_size[roll_collider, 1] = 0.5 * length
    model.geom_friction[roll_collider, 0] = float(
        dynamics["sliding_friction"]
    )

    mesh_id = int(model.geom_dataid[roll_visual])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count]
    spans = np.ptp(vertices, axis=0)
    length_axis = int(np.argmax(spans))
    scale = np.full(3, diameter / BASE_ROLL_DIAMETER_M)
    scale[length_axis] = length / BASE_ROLL_LENGTH_M
    vertices[:] *= scale

    axis_inertia = 0.5 * mass * radius * radius
    transverse_inertia = mass * (3.0 * radius * radius + length * length) / 12.0
    model.body_mass[roll_body] = mass
    model.body_inertia[roll_body] = [
        transverse_inertia,
        transverse_inertia,
        axis_inertia,
    ]

    rgba = np.asarray(assignment["appearance_profile"]["rgba"], dtype=float)
    material_id = int(model.geom_matid[roll_visual])
    if material_id >= 0:
        model.mat_texid[material_id, :] = -1
        model.mat_rgba[material_id] = rgba
    model.geom_rgba[roll_visual] = rgba
    model.geom_rgba[roll_collider] = rgba

    light_scale = float(assignment["lighting_profile"]["diffuse_scale"])
    model.light_diffuse[:] = np.clip(
        model.light_diffuse * light_scale, 0.05, 1.0
    )
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    actual_spans = np.ptp(vertices, axis=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "assignment_id": assignment["assignment_id"],
        "roll_length_m": length,
        "roll_diameter_m": diameter,
        "roll_mass_kg": mass,
        "roll_sliding_friction": float(model.geom_friction[roll_collider, 0]),
        "visual_mesh_span_m": np.round(actual_spans, 7).tolist(),
        "visual_length_axis": length_axis,
        "appearance_rgba": np.round(rgba, 5).tolist(),
        "visual_texture_disabled": bool(
            material_id >= 0
            and np.all(model.mat_texid[material_id] == -1)
        ),
        "light_diffuse_scale": light_scale,
        "jpeg_quality": int(assignment["image_profile"]["jpeg_quality"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--campaign", required=True)
    generate.add_argument("--seed-start", type=int, required=True)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument(
        "--admission-group", choices=tuple(ADMISSION_GROUPS)
    )
    check = subparsers.add_parser("check")
    check.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "generate":
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite manifest: {args.out}")
        payload = generate_manifest(
            args.campaign,
            args.seed_start,
            args.count,
            args.admission_group,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
        return 0
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
