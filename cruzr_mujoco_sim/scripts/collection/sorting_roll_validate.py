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


TASK = "sorting_roll_cruzr"
TASK_VERSION = "sorting_roll_v9_d405_sim"
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


def metadata_errors(meta, result, num_frames):
    errors = []
    episode_meta = meta.get("episode_metadata") or {}
    if meta.get("task") != TASK or result.get("task") != TASK:
        errors.append("task name mismatch")
    if (
        episode_meta.get("task_version") != TASK_VERSION
        or result.get("task_version") != TASK_VERSION
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


def validate_episode(path):
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
    }
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    episodes = expand_episode_paths(args.episodes)
    if not episodes:
        raise SystemExit("no complete episode directories found")
    records = []
    seeds = []
    for episode in episodes:
        info, errors = validate_episode(episode)
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
    passed = sum(record["passed"] for record in records)
    report = {
        "schema_version": 1,
        "task": TASK,
        "task_version": TASK_VERSION,
        "policy_image_map": POLICY_IMAGE_MAP,
        "episode_count": len(episodes),
        "passed_count": passed,
        "failed_count": len(records) - passed,
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
