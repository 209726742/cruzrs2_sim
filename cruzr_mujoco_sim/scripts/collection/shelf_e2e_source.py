#!/usr/bin/env python3
"""Validate a recorded dual-material source episode before dataset ingestion."""

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from cruzr_s2_sdk_contract import (
    SDK_COLLECTION_PROFILE,
    SDK_DOC_REVISION,
    SDK_TASK_HEAD_POSE_RAD,
    audit_sdk_episode,
    load_sdk_timestamp_sidecar,
)
from shelf_e2e_contract import CHUNK_SIZE, FPS, IMAGE_SHAPE
from shelf_e2e_flex_state import internal_state_errors
from shelf_e2e_profiles import (
    STRICT_COLLECTION_PROFILE,
    collection_cameras,
    normalize_collection_profile,
    policy_image_map,
)


SOURCE_STATE_DIM = 16
SOURCE_ACTION_DIM = 16
MIN_SOURCE_FRAMES = CHUNK_SIZE + 1
SPLIT_ORDER = ("train", "val", "test")
BOUNDARY_LAYOUT_AXES = {
    "cart_x", "cart_y", "rack_y", "robot_x", "robot_y", "robot_yaw"
}
LAYOUT_AXIS_LIMITS = {
    "cart_x": 0.20,
    "cart_y": 0.30,
    "rack_y": 0.24,
    "robot_x": 0.08,
    "robot_y": 0.08,
    "robot_yaw": 0.12,
}


def source_split(seed):
    """Deterministic 80/10/10 split based only on the independent source seed."""
    seed = int(seed)
    if seed <= 0:
        raise ValueError(f"source seed must be positive, got {seed}")
    bucket = seed % 10
    if bucket == 1:
        return "val"
    if bucket == 0:
        return "test"
    return "train"


def quality_errors(meta, result, num_frames):
    """Return final-result/endpoint errors without touching episode payload files."""
    errors = []
    episode_meta = meta.get("episode_metadata") or {}
    validation = episode_meta.get("validation") or {}
    if meta.get("success") is not True:
        errors.append("meta.success is not true")
    if meta.get("aborted", False):
        errors.append("meta.aborted is true")
    if validation.get("passed") is not True:
        errors.append("episode_metadata.validation.passed is not true")
    if result.get("passed") is not True:
        errors.append("result.passed is not true")

    for label, motion in (
        ("episode_metadata.validation.motion_quality", validation.get("motion_quality") or {}),
        ("result.motion_quality", result.get("motion_quality") or {}),
    ):
        if motion.get("passed") is not True:
            errors.append(f"{label}.passed is not true")
        if motion.get("tracking_passed") is not True:
            errors.append(f"{label}.tracking_passed is not true")
        if motion.get("tracking_enforced") is not True:
            errors.append(f"{label}.tracking_enforced is not true")
        if motion.get("num_frames") != num_frames:
            errors.append(f"{label}.num_frames != {num_frames}")

    endpoint = episode_meta.get("policy_episode_end") or {}
    result_endpoint = result.get("policy_episode_end") or {}
    if endpoint.get("reason") != "both_objects_released_and_stable":
        errors.append("policy_episode_end.reason is not both_objects_released_and_stable")
    if endpoint.get("recorded_frames") != num_frames:
        errors.append(f"policy_episode_end.recorded_frames != {num_frames}")
    if endpoint.get("audit_frames") != num_frames:
        errors.append(f"policy_episode_end.audit_frames != {num_frames}")
    if result_endpoint != endpoint:
        errors.append("result/meta policy_episode_end mismatch")

    safety_home = result.get("safety_home") or {}
    if safety_home.get("recorded_in_policy_episode") is not False:
        errors.append("safety_home was not explicitly kept outside the policy episode")
    if safety_home.get("objects_stable") is not True:
        errors.append("objects were not stable after safety_home")
    return errors


def diversity_errors(meta, result):
    """Validate labeled scene/recovery diversity without rejecting legacy sources."""
    errors = []
    episode_meta = meta.get("episode_metadata") or {}
    diversity = episode_meta.get("diversity")
    result_diversity = result.get("diversity")
    if diversity is None and result_diversity is None:
        return "legacy_unlabeled", None, errors
    if not isinstance(diversity, dict):
        return None, None, ["episode_metadata.diversity must be an object"]
    if result_diversity != diversity:
        errors.append("result/meta diversity mismatch")

    mode = diversity.get("mode")
    if mode not in ("clean", "recovery"):
        errors.append(f"unsupported diversity mode: {mode!r}")
    if diversity.get("schema_version") != 1:
        errors.append("diversity.schema_version must be 1")
    scene = diversity.get("scene_randomization")
    if not isinstance(scene, dict):
        errors.append("diversity.scene_randomization must be an object")
        scene = {}
    layout_mode = scene.get("layout_mode")
    boundary_axis = scene.get("boundary_axis")
    if layout_mode not in ("random", "boundary"):
        errors.append(f"unsupported layout mode: {layout_mode!r}")
    elif layout_mode == "random" and boundary_axis is not None:
        errors.append("random layout mode must not set boundary_axis")
    elif layout_mode == "boundary" and boundary_axis not in BOUNDARY_LAYOUT_AXES:
        errors.append(f"unsupported boundary axis: {boundary_axis!r}")
    cart_offset = scene.get("cart_offset_xy_m")
    robot_initial = scene.get("robot_initial_xyyaw")
    axis_values = {
        "cart_x": cart_offset[0] if isinstance(cart_offset, list) and len(cart_offset) == 2 else None,
        "cart_y": cart_offset[1] if isinstance(cart_offset, list) and len(cart_offset) == 2 else None,
        "rack_y": scene.get("rack_y_offset_m"),
        "robot_x": robot_initial[0] if isinstance(robot_initial, list) and len(robot_initial) == 3 else None,
        "robot_y": robot_initial[1] if isinstance(robot_initial, list) and len(robot_initial) == 3 else None,
        "robot_yaw": robot_initial[2] if isinstance(robot_initial, list) and len(robot_initial) == 3 else None,
    }
    for axis, limit in LAYOUT_AXIS_LIMITS.items():
        value = axis_values[axis]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            errors.append(f"scene randomization axis {axis} is missing or invalid")
        elif abs(value) > limit + 1e-9:
            errors.append(f"scene randomization axis {axis} exceeds {limit}")
    if layout_mode == "boundary" and boundary_axis in LAYOUT_AXIS_LIMITS:
        value = axis_values[boundary_axis]
        limit = LAYOUT_AXIS_LIMITS[boundary_axis]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if np.isfinite(value) and abs(value) < 0.60 * limit - 1e-9:
                errors.append(
                    f"boundary axis {boundary_axis} is not in the outer 20%"
                )

    requested = diversity.get("requested_event_count")
    actual = diversity.get("actual_event_count")
    events = diversity.get("events")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        errors.append("diversity.requested_event_count must be a non-negative integer")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        errors.append("diversity.actual_event_count must be a non-negative integer")
    if not isinstance(events, list):
        errors.append("diversity.events must be a list")
        events = []
    if isinstance(actual, int) and not isinstance(actual, bool) and actual != len(events):
        errors.append("diversity.actual_event_count does not match events")

    if mode == "clean":
        if requested != 0 or actual != 0 or events:
            errors.append("clean diversity mode must contain zero perturbation events")
        if diversity.get("perturbation_type") != "none":
            errors.append("clean diversity perturbation_type must be none")
    elif mode == "recovery":
        if requested != 1:
            errors.append("recovery diversity mode must request exactly one event")
        if actual != 1:
            errors.append("recovery diversity mode must record exactly one actual event")
        if diversity.get("perturbation_type") != "controlled_empty_navigation_base_pose_shift":
            errors.append("unsupported recovery perturbation_type")

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"diversity.events[{index}] must be an object")
            continue
        if not isinstance(event.get("phase"), str) or not event.get("phase"):
            errors.append(f"diversity.events[{index}].phase must be a non-empty string")
        if mode == "recovery":
            if event.get("trigger") != "controlled_empty_navigation_entry":
                errors.append(f"diversity.events[{index}] has an unsupported trigger")
            if event.get("phase") != "pillar_navigate_to_grasp":
                errors.append(f"diversity.events[{index}] is outside empty navigation")
        delta = event.get("base_pose_delta")
        if not isinstance(delta, dict):
            errors.append(f"diversity.events[{index}].base_pose_delta must be an object")
            continue
        limits = (
            (("x_m", 0.035), ("y_m", 0.035), ("yaw_rad", 0.0525))
            if mode == "recovery"
            else (("x_m", 0.10), ("y_m", 0.10), ("yaw_rad", 0.15))
        )
        for field, limit in limits:
            value = delta.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"diversity.events[{index}].base_pose_delta.{field} is invalid")
            elif not np.isfinite(value) or abs(value) > limit + 1e-9:
                errors.append(
                    f"diversity.events[{index}].base_pose_delta.{field} exceeds {limit}"
                )
    return mode, diversity, errors


def validate_source_dir(path):
    """Return (source_info, errors); source_info is populated only when parsing succeeds."""
    errors = []
    meta_path = os.path.join(path, "meta.json")
    result_path = os.path.join(path, "result.json")
    data_path = os.path.join(path, "episode_data.npz")
    pose_path = os.path.join(path, "object_poses.npz")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, [f"cannot read meta.json: {exc}"]
    try:
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, [f"cannot read result.json: {exc}"]

    episode_meta = meta.get("episode_metadata") or {}
    try:
        collection_profile = normalize_collection_profile(
            episode_meta.get("collection_profile")
        )
    except ValueError as exc:
        errors.append(str(exc))
        collection_profile = STRICT_COLLECTION_PROFILE
    expected_cameras = collection_cameras(collection_profile)
    seeds = [meta.get("seed"), episode_meta.get("seed"), result.get("seed")]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        errors.append(f"invalid seed fields: {seeds}")
        seed = None
    elif len(set(seeds)) != 1:
        errors.append(f"seed mismatch: {seeds}")
        seed = seeds[0]
    else:
        seed = seeds[0]
        if seed <= 0:
            errors.append(f"source seed must be positive, got {seed}")

    num_frames = meta.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        errors.append(f"invalid meta.num_frames: {num_frames!r}")
        num_frames = 0
    elif num_frames < MIN_SOURCE_FRAMES:
        errors.append(f"source has {num_frames} frames, need at least {MIN_SOURCE_FRAMES}")
    errors.extend(quality_errors(meta, result, num_frames))
    diversity_mode, diversity, diversity_validation_errors = diversity_errors(meta, result)
    errors.extend(diversity_validation_errors)
    layout_mode = (
        "legacy_unlabeled"
        if diversity is None
        else (diversity.get("scene_randomization") or {}).get("layout_mode")
    )
    errors.extend(internal_state_errors(path, episode_meta, num_frames))

    if meta.get("fps") != FPS:
        errors.append(f"meta.fps {meta.get('fps')!r} != {FPS}")
    if list(meta.get("resolution_hw") or []) != list(IMAGE_SHAPE[:2]):
        errors.append(f"resolution_hw {meta.get('resolution_hw')!r} != {list(IMAGE_SHAPE[:2])}")
    actual_cameras = tuple((meta.get("cameras") or {}).keys())
    if collection_profile == SDK_COLLECTION_PROFILE:
        if actual_cameras != expected_cameras:
            errors.append(
                f"SDK camera order {actual_cameras} != {expected_cameras}"
            )
    elif set(actual_cameras) != set(expected_cameras):
        errors.append("camera keys do not match the deployable contract")
    if len(meta.get("state_joint_names") or []) != SOURCE_STATE_DIM:
        errors.append(f"state_joint_names must have {SOURCE_STATE_DIM} entries")
    if len(meta.get("action_names") or []) != SOURCE_ACTION_DIM:
        errors.append(f"action_names must have {SOURCE_ACTION_DIM} entries")

    source_arrays = {}
    try:
        with np.load(data_path, allow_pickle=False) as data:
            required_shapes = {
                "timestamp": (num_frames,),
                "state": (num_frames, SOURCE_STATE_DIM),
                "action": (num_frames, SOURCE_ACTION_DIM),
                "base": (num_frames, 3),
                "base_velocity": (num_frames, 2),
                "base_action": (num_frames, 2),
                "phase": (num_frames,),
            }
            for key, shape in required_shapes.items():
                if key not in data:
                    errors.append(f"episode_data.npz missing {key}")
                    continue
                if data[key].shape != shape:
                    errors.append(f"{key} shape {data[key].shape} != {shape}")
            for key in ("timestamp", "state", "action", "base", "base_velocity", "base_action"):
                if key in data and not np.isfinite(data[key]).all():
                    errors.append(f"{key} contains NaN/Inf")
            if "timestamp" in data and data["timestamp"].shape == (num_frames,) and num_frames > 1:
                expected = ((np.arange(num_frames) + 1) / FPS).astype(np.float32)
                if not np.allclose(data["timestamp"], expected, atol=1e-6, rtol=0):
                    errors.append("timestamp does not match the recorder's uniform 30 FPS grid")
            for key in ("timestamp", "state", "action", "base_action"):
                if key in data:
                    source_arrays[key] = np.asarray(data[key]).copy()
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot read episode_data.npz: {exc}")

    if collection_profile == SDK_COLLECTION_PROFILE:
        stored_sdk = (episode_meta.get("validation") or {}).get("sdk_alignment")
        result_sdk = result.get("sdk_alignment")
        if episode_meta.get("sdk_document_revision") != SDK_DOC_REVISION:
            errors.append(
                f"SDK document revision must be {SDK_DOC_REVISION}"
            )
        if episode_meta.get("sdk_task_head_pose_rad") != SDK_TASK_HEAD_POSE_RAD:
            errors.append(
                f"SDK task head pose must be {SDK_TASK_HEAD_POSE_RAD}"
            )
        if result.get("collection_profile") != SDK_COLLECTION_PROFILE:
            errors.append("result collection_profile is not sdk_recovery_v1")
        if not isinstance(stored_sdk, dict) or stored_sdk.get("passed") is not True:
            errors.append("episode_metadata.validation.sdk_alignment.passed is not true")
        if not isinstance(result_sdk, dict) or result_sdk.get("passed") is not True:
            errors.append("result.sdk_alignment.passed is not true")
        if stored_sdk != result_sdk:
            errors.append("result/meta SDK alignment mismatch")
        required = {"timestamp", "state", "action", "base_action"}
        if required.issubset(source_arrays):
            sdk_state_timestamp, camera_timestamps = load_sdk_timestamp_sidecar(path)
            sdk_audit = audit_sdk_episode(
                source_arrays["state"],
                source_arrays["action"],
                source_arrays["base_action"],
                fps=FPS,
                joint_names=meta.get("action_names") or [],
                cameras=actual_cameras,
                timestamp=source_arrays["timestamp"],
                sdk_state_timestamp=sdk_state_timestamp,
                camera_timestamps=camera_timestamps,
                require_camera_timestamps=True,
                enforce_rated_speed=True,
            )
            if not sdk_audit["passed"]:
                errors.extend(
                    f"SDK audit: {message}" for message in sdk_audit["errors"]
                )

    try:
        with np.load(pose_path, allow_pickle=False) as poses:
            if poses.get("pose", np.empty(0)).shape != (num_frames, 14):
                errors.append(f"object pose shape must be ({num_frames}, 14)")
            elif not np.isfinite(poses["pose"]).all():
                errors.append("object poses contain NaN/Inf")
            if list(poses.get("names", [])) != ["pillar", "strip"]:
                errors.append("object pose names must be pillar, strip")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot read object_poses.npz: {exc}")

    for camera in expected_cameras:
        frame_dir = os.path.join(path, "frames", camera)
        frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
        if len(frames) != num_frames:
            errors.append(f"camera {camera} has {len(frames)} frames, expected {num_frames}")
            continue
        bad_name = next(
            (name for index, name in enumerate(frames)
             if os.path.basename(name) != f"frame_{index:06d}.jpg"),
            None,
        )
        if bad_name:
            errors.append(f"camera {camera} frame sequence has a gap at {os.path.basename(bad_name)}")
            continue
        for frame in (frames[0], frames[-1]):
            try:
                with Image.open(frame) as image:
                    if image.size != (IMAGE_SHAPE[1], IMAGE_SHAPE[0]) or image.mode != "RGB":
                        errors.append(
                            f"camera {camera} frame {os.path.basename(frame)} is "
                            f"{image.size}/{image.mode}, expected 224x224/RGB"
                        )
            except (OSError, ValueError) as exc:
                errors.append(f"cannot decode {frame}: {exc}")

    info = None
    if seed is not None:
        info = {
            "path": os.path.abspath(path),
            "seed": int(seed),
            "split": source_split(seed) if seed > 0 else None,
            "num_frames": int(num_frames),
            "task_version": episode_meta.get("task_version"),
            "collection_profile": collection_profile,
            "diversity_mode": diversity_mode,
            "layout_mode": layout_mode,
            "diversity": diversity,
            "cameras": list(expected_cameras),
            "policy_image_map": policy_image_map(collection_profile),
        }
    return info, errors


def require_unique_seeds(sources):
    seen = {}
    for source in sources:
        seed = source["seed"]
        if seed in seen:
            raise ValueError(
                f"duplicate source seed {seed}: {seen[seed]} and {source['path']}"
            )
        seen[seed] = source["path"]


def require_single_task_version(sources):
    versions = sorted({source["task_version"] for source in sources})
    if len(versions) != 1:
        raise ValueError(f"source task versions must not be mixed: {versions}")
    return versions[0]


def require_single_collection_profile(sources):
    profiles = sorted({source["collection_profile"] for source in sources})
    if len(profiles) != 1:
        raise ValueError(f"source collection profiles must not be mixed: {profiles}")
    cameras = {tuple(source["cameras"]) for source in sources}
    if len(cameras) != 1:
        raise ValueError(f"source camera contracts must not be mixed: {sorted(cameras)}")
    return profiles[0], next(iter(cameras))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-diversity-mode", choices=("clean", "recovery"))
    parser.add_argument("--expected-layout-mode", choices=("random", "boundary"))
    args = parser.parse_args()
    info, errors = validate_source_dir(args.episode)
    if args.expected_seed is not None and (info or {}).get("seed") != args.expected_seed:
        errors.append(
            f"source seed {(info or {}).get('seed')} != expected seed {args.expected_seed}"
        )
    if args.expected_profile is not None:
        try:
            expected_profile = normalize_collection_profile(args.expected_profile)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if (info or {}).get("collection_profile") != expected_profile:
                errors.append(
                    f"source profile {(info or {}).get('collection_profile')} "
                    f"!= expected profile {expected_profile}"
                )
    if (
        args.expected_diversity_mode is not None
        and (info or {}).get("diversity_mode") != args.expected_diversity_mode
    ):
        errors.append(
            f"source diversity mode {(info or {}).get('diversity_mode')} "
            f"!= expected diversity mode {args.expected_diversity_mode}"
        )
    if (
        args.expected_layout_mode is not None
        and (info or {}).get("layout_mode") != args.expected_layout_mode
    ):
        errors.append(
            f"source layout mode {(info or {}).get('layout_mode')} "
            f"!= expected layout mode {args.expected_layout_mode}"
        )
    print(json.dumps({"passed": not errors, "source": info, "errors": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
