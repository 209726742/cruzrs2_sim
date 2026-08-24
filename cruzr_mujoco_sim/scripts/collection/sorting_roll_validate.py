#!/usr/bin/env python3
"""Validate rendered Sorting Roll simulation episodes before ingestion."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_realsense_profile import (  # noqa: E402
    POLICY_IMAGE_MAP,
    PROFILE_NAME,
)
from sorting_roll_diversity import (  # noqa: E402
    DIVERSE_TASK_VERSION,
    assignment_errors,
    load_manifest,
    manifest_counts,
)


TASK = "sorting_roll_cruzr"
TASK_VERSION = "sorting_roll_v11_d405_upright_support_sim"
SUPPORTED_TASK_VERSIONS = (TASK_VERSION, DIVERSE_TASK_VERSION)
FPS = 30
MAX_CAMERA_STATE_SKEW_S = 0.020
MIN_FRAMES = 51
POLICY_CAMERAS = tuple(
    value.rsplit(".", 1)[-1]
    for value in POLICY_IMAGE_MAP.values()
)
ARRAY_SHAPES = {
    "timestamp": (),
    "state": (16,),
    "action": (16,),
    "action_real": (),
    "base": (3,),
    "base_velocity": (2,),
    "base_action": (2,),
    "phase": (),
    "roll_qpos": (7,),
    "roll_qvel": (6,),
}
FINAL_CHECKS = (
    "center_inside_integrated_top_tier",
    "fully_inside_shelf_width",
    "axis_aligned_with_shelf",
    "supported_by_integrated_top_tier",
    "resting_on_integrated_top_tier_geometry",
    "released_from_both_grippers",
    "not_supported_by_table",
    "low_linear_speed",
    "low_angular_speed",
)
REQUIRED_GATES = (
    "released_during_guarded_backoff",
    "rod_remains_in_integrated_top_tier_after_open",
    "arms_retracted",
    "sorting_roll_success_after_retract",
    "episode_under_one_minute",
)


def source_split(seed):
    seed = int(seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    if seed % 10 == 1:
        return "val"
    if seed % 10 == 0:
        return "test"
    return "train"


def payload_errors(payload, num_frames):
    errors = []
    for name, tail in ARRAY_SHAPES.items():
        if name not in payload:
            errors.append(f"episode_data.npz missing {name}")
            continue
        value = np.asarray(payload[name])
        expected = (num_frames, *tail)
        if value.shape != expected:
            errors.append(f"{name} shape {value.shape} != {expected}")
            continue
        if name != "phase" and not np.isfinite(value).all():
            errors.append(f"{name} contains NaN/Inf")
    action_real = np.asarray(payload.get("action_real", []))
    if action_real.shape == (num_frames,) and not action_real.all():
        errors.append("action_real must be true for every frame")
    timestamp = np.asarray(payload.get("timestamp", []), dtype=float)
    if timestamp.shape == (num_frames,):
        expected = (np.arange(num_frames) + 1) / FPS
        if not np.allclose(timestamp, expected, atol=2e-6, rtol=0):
            errors.append("timestamp is not the expected uniform 30 FPS grid")
    return errors


def timestamp_errors(payload, num_frames):
    errors = []
    expected_keys = {"state_timestamp"} | {
        f"camera_{camera}_timestamp" for camera in POLICY_CAMERAS
    }
    if set(payload) != expected_keys:
        errors.append(
            "sdk_timestamps.npz keys do not match the three-camera contract"
        )
        return errors
    state = np.asarray(payload["state_timestamp"], dtype=float)
    if state.shape != (num_frames,) or not np.isfinite(state).all():
        errors.append("state_timestamp is missing, non-finite, or the wrong length")
        return errors
    if num_frames > 1 and np.any(np.diff(state) <= 0.0):
        errors.append("state_timestamp is not strictly increasing")
    for camera in POLICY_CAMERAS:
        values = np.asarray(
            payload[f"camera_{camera}_timestamp"], dtype=float
        )
        if values.shape != (num_frames,) or not np.isfinite(values).all():
            errors.append(f"{camera} timestamps are invalid")
            continue
        skew = float(np.max(np.abs(values - state)))
        if skew > MAX_CAMERA_STATE_SKEW_S:
            errors.append(f"{camera} timestamp skew exceeds 20 ms")
    return errors


def diversity_errors(meta, result, episode_meta):
    errors = []
    task_version = result.get("task_version")
    values = (
        meta.get("diversity"),
        episode_meta.get("diversity"),
        result.get("diversity"),
    )
    if task_version == TASK_VERSION:
        if any(value is not None for value in values):
            errors.append("base episode unexpectedly contains diversity metadata")
        return errors
    if task_version != DIVERSE_TASK_VERSION:
        return errors
    if not isinstance(values[0], dict) or values[0] != values[1] or values[0] != values[2]:
        return ["diversity metadata is missing or inconsistent"]

    diversity = values[0]
    assignment = diversity.get("assignment")
    applied = diversity.get("applied")
    assignment_validation_errors = assignment_errors(assignment)
    for error in assignment_validation_errors:
        errors.append(f"diversity assignment: {error}")
    if assignment_validation_errors:
        return errors
    if not isinstance(applied, dict):
        return errors + ["diversity applied report is missing"]
    if applied.get("assignment_id") != assignment.get("assignment_id"):
        errors.append("applied assignment_id mismatch")

    expected = {
        "roll_length_m": assignment["object_profile"]["length_m"],
        "roll_diameter_m": assignment["object_profile"]["diameter_m"],
        "roll_mass_kg": assignment["dynamics_profile"]["mass_kg"],
        "roll_sliding_friction": assignment["dynamics_profile"][
            "sliding_friction"
        ],
        "light_diffuse_scale": assignment["lighting_profile"][
            "diffuse_scale"
        ],
        "jpeg_quality": assignment["image_profile"]["jpeg_quality"],
    }
    for name, expected_value in expected.items():
        actual = applied.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isclose(actual, expected_value, atol=1e-6, rtol=0)
        ):
            errors.append(f"applied {name} mismatch")

    actual_rgba = np.asarray(applied.get("appearance_rgba", []), dtype=float)
    expected_rgba = np.asarray(
        assignment["appearance_profile"]["rgba"], dtype=float
    )
    if (
        actual_rgba.shape != (4,)
        or not np.isfinite(actual_rgba).all()
        or not np.allclose(actual_rgba, expected_rgba, atol=1e-6, rtol=0)
    ):
        errors.append("applied appearance_rgba mismatch")
    if applied.get("visual_texture_disabled") is not True:
        errors.append("visual texture must be disabled for true color diversity")

    spans = np.asarray(applied.get("visual_mesh_span_m", []), dtype=float)
    length_axis = applied.get("visual_length_axis")
    if spans.shape != (3,) or length_axis not in (0, 1, 2):
        errors.append("applied visual mesh dimensions are invalid")
    else:
        if not np.isclose(
            spans[length_axis],
            assignment["object_profile"]["length_m"],
            atol=2e-4,
            rtol=0,
        ):
            errors.append("visual mesh length does not match object profile")
        radial = np.delete(spans, length_axis)
        if not np.allclose(
            radial,
            assignment["object_profile"]["diameter_m"],
            atol=2e-4,
            rtol=0,
        ):
            errors.append("visual mesh diameter does not match object profile")
    if meta.get("prompt") != assignment.get("prompt"):
        errors.append("meta prompt does not match diversity assignment")
    if result.get("prompt") != assignment.get("prompt"):
        errors.append("result prompt does not match diversity assignment")
    randomization = result.get("scene_randomization") or {}
    if randomization.get("pose_bin") != assignment.get("pose_bin"):
        errors.append("scene randomization pose_bin mismatch")
    return errors


def metadata_errors(meta, result, num_frames):
    errors = []
    episode_meta = meta.get("episode_metadata") or {}
    if meta.get("task") != TASK or result.get("task") != TASK:
        errors.append("task name mismatch")
    task_version = result.get("task_version")
    if (
        task_version not in SUPPORTED_TASK_VERSIONS
        or episode_meta.get("task_version") != task_version
        or meta.get("task_version") != task_version
    ):
        errors.append("task version mismatch")
    seed = result.get("seed")
    if meta.get("seed") != seed or episode_meta.get("seed") != seed:
        errors.append("seed mismatch across meta/result")
    if meta.get("fps") != FPS:
        errors.append("meta.fps must be 30")
    if meta.get("num_frames") != num_frames or result.get("num_frames") != num_frames:
        errors.append("num_frames mismatch across meta/result/data")
    if set(meta.get("cameras") or {}) != set(POLICY_CAMERAS):
        errors.append("meta cameras do not match the three-camera contract")
    if tuple(episode_meta.get("policy_cameras") or ()) != POLICY_CAMERAS:
        errors.append("episode_metadata.policy_cameras order mismatch")
    if tuple(episode_meta.get("recorded_cameras") or ()) != POLICY_CAMERAS:
        errors.append("episode_metadata.recorded_cameras order mismatch")
    if episode_meta.get("policy_image_map") != POLICY_IMAGE_MAP:
        errors.append("policy image map mismatch")
    if episode_meta.get("collection_profile") != PROFILE_NAME:
        errors.append("collection profile mismatch")
    eligibility = (
        meta.get("simulation_canary_eligible"),
        episode_meta.get("simulation_canary_eligible"),
        result.get("simulation_canary_eligible"),
    )
    if eligibility != (True, True, True):
        errors.append("simulation canary eligibility is not consistently true")
    if meta.get("training_eligible") is not False:
        errors.append("simulation meta.training_eligible must remain false")
    if episode_meta.get("training_eligible") is not False:
        errors.append("simulation nested training_eligible must remain false")
    if result.get("training_eligible") is not False:
        errors.append("simulation result.training_eligible must remain false")
    if meta.get("success") is not True or result.get("success") is not True:
        errors.append("episode success is not true")
    if result.get("error") is not None:
        errors.append("result.error is not null")
    seconds = result.get("sim_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        errors.append("sim_seconds is invalid")
    elif seconds > 60.0:
        errors.append("sim_seconds exceeds 60")
    final = result.get("final_evidence") or {}
    checks = final.get("checks") or {}
    if final.get("instantaneous_success") is not True:
        errors.append("final evidence is not successful")
    if final.get("stable_seconds", 0.0) < 2.0:
        errors.append("final stable window is shorter than 2 seconds")
    for check in FINAL_CHECKS:
        if checks.get(check) is not True:
            errors.append(f"final check failed: {check}")
    for gate in REQUIRED_GATES:
        if (result.get("gates") or {}).get(gate, {}).get("passed") is not True:
            errors.append(f"required gate failed: {gate}")
    if result.get("scene_randomization") != episode_meta.get("scene_randomization"):
        errors.append("scene randomization mismatch across meta/result")
    elif (result.get("scene_randomization") or {}).get("enabled") is not True:
        errors.append("scene randomization is not enabled")
    errors.extend(diversity_errors(meta, result, episode_meta))
    return errors


def frame_errors(path, num_frames, resolution_hw):
    errors = []
    if (
        not isinstance(resolution_hw, list)
        or len(resolution_hw) != 2
        or any(not isinstance(value, int) for value in resolution_hw)
    ):
        return ["meta.resolution_hw is invalid"]
    expected_size = (resolution_hw[1], resolution_hw[0])
    frame_root = path / "frames"
    actual_dirs = {
        item.name for item in frame_root.iterdir() if item.is_dir()
    } if frame_root.is_dir() else set()
    if actual_dirs != set(POLICY_CAMERAS):
        errors.append("frame directories do not match the policy cameras")
    for camera in POLICY_CAMERAS:
        directory = frame_root / camera
        files = sorted(directory.glob("frame_*.jpg"))
        if len(files) != num_frames:
            errors.append(f"{camera} frame count {len(files)} != {num_frames}")
            continue
        for index, file in enumerate(files):
            if file.name != f"frame_{index:06d}.jpg":
                errors.append(f"{camera} frame numbering is not contiguous")
                break
        for index in sorted({0, num_frames // 2, num_frames - 1}):
            try:
                with Image.open(files[index]) as image:
                    image.load()
                    if image.size != expected_size or image.mode != "RGB":
                        errors.append(
                            f"{camera} sample frame has wrong size or mode"
                        )
            except (OSError, ValueError) as exc:
                errors.append(f"{camera} sample frame cannot be decoded: {exc}")
    return errors


def validate_episode(path, manifest_assignments=None):
    path = Path(path).resolve()
    errors = []
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"cannot read meta/result JSON: {exc}"]
    num_frames = meta.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        return None, ["meta.num_frames is not an integer"]
    if num_frames < MIN_FRAMES:
        errors.append(f"num_frames is shorter than {MIN_FRAMES}")
    try:
        with np.load(path / "episode_data.npz", allow_pickle=False) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        errors.extend(payload_errors(payload, num_frames))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read episode_data.npz: {exc}")
    try:
        with np.load(path / "sdk_timestamps.npz", allow_pickle=False) as data:
            timestamps = {name: np.asarray(data[name]) for name in data.files}
        errors.extend(timestamp_errors(timestamps, num_frames))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read sdk_timestamps.npz: {exc}")
    errors.extend(metadata_errors(meta, result, num_frames))
    errors.extend(frame_errors(path, num_frames, meta.get("resolution_hw")))
    seed = result.get("seed")
    info = {
        "path": str(path),
        "seed": seed,
        "split": source_split(seed) if isinstance(seed, int) and seed > 0 else None,
        "num_frames": num_frames,
        "sim_seconds": result.get("sim_seconds"),
        "cameras": list(POLICY_CAMERAS),
        "resolution_hw": meta.get("resolution_hw"),
        "task_version": result.get("task_version"),
        "collection_profile": (meta.get("episode_metadata") or {}).get(
            "collection_profile"
        ),
        "prompt": meta.get("prompt"),
        "diversity": meta.get("diversity"),
    }
    if manifest_assignments is not None and result.get("task_version") == DIVERSE_TASK_VERSION:
        expected = manifest_assignments.get(seed)
        actual = (meta.get("diversity") or {}).get("assignment")
        if expected is None:
            errors.append("seed is missing from campaign manifest")
        elif actual != expected:
            errors.append("episode assignment does not match campaign manifest")
    return info, errors


def expand_episode_paths(paths):
    episodes = []
    for path in paths:
        path = Path(path)
        if (path / "result.json").is_file():
            episodes.append(path)
        else:
            episodes.extend(sorted(
                item for item in path.glob("seed_*")
                if (item / "result.json").is_file()
            ))
    return episodes


def episode_diversity_counts(infos):
    assignments = [
        info["diversity"]["assignment"]
        for info in infos
        if info.get("task_version") == DIVERSE_TASK_VERSION
        and isinstance(info.get("diversity"), dict)
        and isinstance(info["diversity"].get("assignment"), dict)
    ]
    return manifest_counts(assignments) if assignments else {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_assignments = None
    manifest = None
    if args.manifest:
        manifest = load_manifest(args.manifest)
        manifest_assignments = {
            assignment["seed"]: assignment
            for assignment in manifest["assignments"]
        }
    episodes = expand_episode_paths(args.episodes)
    if not episodes:
        raise SystemExit("no complete episode directories found")
    records = []
    seeds = []
    for episode in episodes:
        info, errors = validate_episode(episode, manifest_assignments)
        records.append({
            "path": str(episode.resolve()),
            "passed": not errors,
            "info": info,
            "errors": errors,
        })
        if info and isinstance(info.get("seed"), int):
            seeds.append(info["seed"])
        print(
            f"[validate] {episode.name} {'PASS' if not errors else 'FAIL'}"
            + ("" if not errors else f" {'; '.join(errors)}"),
            flush=True,
        )
    duplicate_seeds = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicate_seeds:
        records.append({
            "path": None,
            "passed": False,
            "info": None,
            "errors": [f"duplicate seeds: {duplicate_seeds}"],
        })
    infos = [record["info"] for record in records if record.get("info")]
    task_versions = sorted(
        {info.get("task_version") for info in infos}, key=str
    )
    collection_profiles = sorted(
        {info.get("collection_profile") for info in infos}, key=str
    )
    campaigns = sorted({
        info["diversity"]["assignment"].get("campaign")
        for info in infos
        if info.get("task_version") == DIVERSE_TASK_VERSION
        and isinstance(info.get("diversity"), dict)
        and isinstance(info["diversity"].get("assignment"), dict)
    }, key=str)
    collection_errors = []
    if len(task_versions) != 1:
        collection_errors.append(
            f"task versions cannot be mixed: {task_versions}"
        )
    if len(collection_profiles) != 1:
        collection_errors.append(
            f"collection profiles cannot be mixed: {collection_profiles}"
        )
    if DIVERSE_TASK_VERSION in task_versions and len(campaigns) != 1:
        collection_errors.append(
            f"diversity campaigns cannot be mixed: {campaigns}"
        )
    if collection_errors:
        records.append({
            "path": None,
            "passed": False,
            "info": None,
            "errors": collection_errors,
        })
    passed = sum(record["passed"] for record in records)
    passed_infos = [
        record["info"]
        for record in records
        if record.get("passed") and record.get("info")
    ]
    report = {
        "schema_version": 1,
        "task": TASK,
        "task_version": task_versions[0] if len(task_versions) == 1 else None,
        "task_versions": task_versions,
        "collection_profiles": collection_profiles,
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "campaign": manifest.get("campaign") if manifest else None,
        "policy_image_map": POLICY_IMAGE_MAP,
        "episode_count": len(episodes),
        "passed_count": passed,
        "failed_count": len(records) - passed,
        "diversity_counts": episode_diversity_counts(passed_infos),
        "passed": passed == len(records) == len(episodes),
        "records": records,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        key: report[key]
        for key in ("episode_count", "passed_count", "failed_count", "passed")
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
