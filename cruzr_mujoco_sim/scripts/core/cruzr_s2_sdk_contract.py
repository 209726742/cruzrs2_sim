#!/usr/bin/env python3
"""Machine-readable Cruzr S2 SDK contract and source-episode audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence

import numpy as np


SDK_DOC_REVISION = "v2.1-2026-06-23"
SDK_MOTION_CONTROL_HZ = 500
SDK_DATASET_FPS = 30
SDK_CAMERA_STATE_MAX_SKEW_S = 0.020
SDK_STATE_POSITION_TOLERANCE_RAD = 0.001
SDK_RAW_TIMESTAMP_MAX_PERIOD_ERROR_S = 0.005
SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_S = 0.005
SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_FRACTION = 0.001

LEFT_ARM_JOINT_NAMES = (
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
)
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

ARM_POSITION_LIMITS_RAD = np.asarray(
    [
        (-2.83, 2.83),
        (-1.86, 0.08),
        (-2.92, 2.92),
        (-2.60, 0.02),
        (-2.88, 2.88),
        (-1.57, 1.60),
        (-1.98, 1.98),
        (-2.83, 2.83),
        (-1.86, 0.08),
        (-2.92, 2.92),
        (-2.60, 0.02),
        (-2.88, 2.88),
        (-1.60, 1.57),
        (-1.98, 1.98),
    ],
    dtype=np.float64,
)
ARM_RATED_SPEED_RAD_S = 20.0 * 2.0 * math.pi / 60.0
ARM_MAX_SPEED_RAD_S = 30.0 * 2.0 * math.pi / 60.0
ARM_RATED_DELTA_RAD_AT_DATASET_FPS = ARM_RATED_SPEED_RAD_S / SDK_DATASET_FPS
ARM_MAX_DELTA_RAD_AT_DATASET_FPS = ARM_MAX_SPEED_RAD_S / SDK_DATASET_FPS
SDK_COMMAND_SPEED_FRACTION_OF_RATED = 0.95
# The representative SDK preflight observed up to 4.18 mrad of simulated
# contact-driven state overshoot beyond a command target.  Six mrad keeps the
# command away from the mechanical stop without relaxing the state audit.
SDK_COMMAND_POSITION_MARGIN_RAD = 0.006
SDK_COMMAND_SPEED_LIMIT_RAD_S = (
    SDK_COMMAND_SPEED_FRACTION_OF_RATED * ARM_RATED_SPEED_RAD_S
)
SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS = (
    SDK_COMMAND_SPEED_LIMIT_RAD_S / SDK_DATASET_FPS
)
SDK_ARM_COMMAND_LIMITS_RAD = ARM_POSITION_LIMITS_RAD + np.asarray(
    (SDK_COMMAND_POSITION_MARGIN_RAD, -SDK_COMMAND_POSITION_MARGIN_RAD),
    dtype=np.float64,
)

SDK_COLLECTION_PROFILE = "sdk_recovery_v1"
SDK_POLICY_IMAGE_MAP = {
    "observation/image": "observation.images.stereo_left",
    "observation/left_wrist_image": "observation.images.waist_front",
    "observation/right_wrist_image": "observation.images.chassis_front",
}
SDK_CAMERAS = tuple(
    key.rsplit(".", 1)[-1] for key in SDK_POLICY_IMAGE_MAP.values()
)
SDK_CAMERA_TOPICS = {
    "stereo_left": "/sensor/camera/stereo_left/image/raw",
    "waist_front": "/sensor/camera/waist_front_rgbd/color/raw",
    "chassis_front": "/sensor/camera/chassis_front_rgbd/color/raw",
}
SDK_DOCUMENTED_RGB_CAMERA_TOPICS = {
    "chassis_front": "/sensor/camera/chassis_front_rgbd/color/raw",
    "waist_front": "/sensor/camera/waist_front_rgbd/color/raw",
    "fisheye_left": "/sensor/camera/fisheye_left/image/raw",
    "fisheye_right": "/sensor/camera/fisheye_right/image/raw",
    "stereo_left": "/sensor/camera/stereo_left/image/raw",
    "stereo_right": "/sensor/camera/stereo_right/image/raw",
}
SDK_SENSOR_EXTRINSICS_ZYX = {
    "waist_front": {
        "parent_link": "waist_yaw_link",
        "xyz_m": (0.07754007, 0.0, 0.02319591),
        "rpy_deg": (0.0, 51.0, 0.0),
    },
    "chassis_front": {
        "parent_link": "base_link",
        "xyz_m": (0.30533127, 0.00128866, 0.155386),
        "rpy_deg": (0.0, -10.0, 0.0),
    },
    "stereo_left": {
        "parent_link": "head_pitch_link",
        "xyz_m": (0.10675004, 0.03018686, 0.13200105),
        "rpy_deg": (-90.0, 0.0, 0.0),
    },
    "stereo_right": {
        "parent_link": "head_pitch_link",
        "xyz_m": (0.10675004, -0.02981314, 0.13200105),
        "rpy_deg": (-90.0, 0.0, 0.0),
    },
    "fisheye_left": {
        "parent_link": "head_pitch_link",
        "xyz_m": (0.02681314, 0.07925004, 0.10490105),
        "rpy_deg": (-90.0, 90.0, 0.0),
    },
    "fisheye_right": {
        "parent_link": "head_pitch_link",
        "xyz_m": (0.02718686, -0.07925004, 0.10490105),
        "rpy_deg": (-90.0, -90.0, 0.0),
    },
}
SDK_SENSOR_ROTATION_ORDER = "ZYX"
SDK_CAMERA_INTRINSICS_VERIFIED = False
SDK_WRIST_CAMERAS = ()
SDK_TASK_HEAD_POSE_RAD = {
    "head_yaw_joint": 0.0,
    "head_pitch_joint": -0.65,
}
SDK_ROBOT_STATE_TOPIC = "/mc/sdk/robot_state"
SDK_ROBOT_COMMAND_TOPIC = "/mc/sdk/robot_command"
SDK_LEFT_GRIP_COMMAND_TOPIC = "/ecat/left_grip/cmd"
SDK_RIGHT_GRIP_COMMAND_TOPIC = "/ecat/right_grip/cmd"
SDK_LEFT_GRIP_STATE_TOPIC = "/ecat/left_grip/state"
SDK_RIGHT_GRIP_STATE_TOPIC = "/ecat/right_grip/state"

SDK_GRIP_POSITION_RANGE_M = (0.0, 0.05)
SDK_GRIP_FORCE_RANGE_N = (0.0, 100.0)
SDK_BASE_V_FWD_RANGE_M_S = (-0.3, 0.8)
SDK_BASE_WZ_RANGE_RAD_S = (-0.6, 0.6)
SDK_BASE_WATCHDOG_S = 2.0


def contract_summary() -> dict:
    """Return a JSON-serializable summary without claiming unresolved gripper polarity."""
    return {
        "sdk_document_revision": SDK_DOC_REVISION,
        "motion_control_hz": SDK_MOTION_CONTROL_HZ,
        "dataset_fps": SDK_DATASET_FPS,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "arm_position_limits_rad": ARM_POSITION_LIMITS_RAD.tolist(),
        "arm_rated_speed_rad_s": ARM_RATED_SPEED_RAD_S,
        "arm_max_speed_rad_s": ARM_MAX_SPEED_RAD_S,
        "arm_rated_delta_rad_at_dataset_fps": ARM_RATED_DELTA_RAD_AT_DATASET_FPS,
        "arm_max_delta_rad_at_dataset_fps": ARM_MAX_DELTA_RAD_AT_DATASET_FPS,
        "command_speed_fraction_of_rated": SDK_COMMAND_SPEED_FRACTION_OF_RATED,
        "command_position_margin_rad": SDK_COMMAND_POSITION_MARGIN_RAD,
        "arm_command_limits_rad": SDK_ARM_COMMAND_LIMITS_RAD.tolist(),
        "command_speed_limit_rad_s": SDK_COMMAND_SPEED_LIMIT_RAD_S,
        "command_delta_rad_at_dataset_fps": SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS,
        "collection_profile": SDK_COLLECTION_PROFILE,
        "policy_image_map": dict(SDK_POLICY_IMAGE_MAP),
        "sdk_cameras": list(SDK_CAMERAS),
        "sdk_camera_topics": dict(SDK_CAMERA_TOPICS),
        "documented_rgb_camera_count": len(
            SDK_DOCUMENTED_RGB_CAMERA_TOPICS
        ),
        "documented_rgb_camera_topics": dict(
            SDK_DOCUMENTED_RGB_CAMERA_TOPICS
        ),
        "sensor_extrinsics_zyx": {
            name: {
                "parent_link": values["parent_link"],
                "xyz_m": list(values["xyz_m"]),
                "rpy_deg": list(values["rpy_deg"]),
            }
            for name, values in SDK_SENSOR_EXTRINSICS_ZYX.items()
        },
        "sensor_rotation_order": SDK_SENSOR_ROTATION_ORDER,
        "camera_intrinsics_verified": SDK_CAMERA_INTRINSICS_VERIFIED,
        "wrist_cameras": list(SDK_WRIST_CAMERAS),
        "task_head_pose_rad": dict(SDK_TASK_HEAD_POSE_RAD),
        "gripper": {
            "position_range_m": list(SDK_GRIP_POSITION_RANGE_M),
            "force_range_n": list(SDK_GRIP_FORCE_RANGE_N),
            "position_polarity_verified": False,
        },
        "base": {
            "v_fwd_range_m_s": list(SDK_BASE_V_FWD_RANGE_M_S),
            "wz_range_rad_s": list(SDK_BASE_WZ_RANGE_RAD_S),
            "watchdog_s": SDK_BASE_WATCHDOG_S,
        },
        "project_camera_state_max_skew_s": SDK_CAMERA_STATE_MAX_SKEW_S,
        "project_raw_timestamp_cadence": {
            "max_period_error_s": SDK_RAW_TIMESTAMP_MAX_PERIOD_ERROR_S,
            "max_duration_drift_s": SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_S,
            "max_duration_drift_fraction": SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_FRACTION,
        },
        "state_position_tolerance_rad": SDK_STATE_POSITION_TOLERANCE_RAD,
    }


def model_contract_errors(
    joint_names: Sequence[str],
    joint_ranges_rad,
    *,
    atol: float = 1e-6,
) -> list[str]:
    """Compare a model's ordered 14-arm contract with the official v2.1 table."""
    errors = []
    names = tuple(joint_names)
    ranges = np.asarray(joint_ranges_rad, dtype=np.float64)
    if names != ARM_JOINT_NAMES:
        errors.append(
            f"arm joint order must be {list(ARM_JOINT_NAMES)}, got {list(names)}"
        )
    if ranges.shape != ARM_POSITION_LIMITS_RAD.shape:
        errors.append(
            f"arm joint ranges must have shape {ARM_POSITION_LIMITS_RAD.shape}, got {ranges.shape}"
        )
    elif not np.isfinite(ranges).all():
        errors.append("arm joint ranges contain NaN/Inf")
    elif not np.allclose(ranges, ARM_POSITION_LIMITS_RAD, atol=atol, rtol=0):
        bad = np.argwhere(
            ~np.isclose(ranges, ARM_POSITION_LIMITS_RAD, atol=atol, rtol=0)
        )[0]
        joint_index = int(bad[0])
        errors.append(
            f"{ARM_JOINT_NAMES[joint_index]} range {ranges[joint_index].tolist()} "
            f"!= SDK {ARM_POSITION_LIMITS_RAD[joint_index].tolist()}"
        )
    return errors


def rate_limit_arm_target(current, target, max_delta_rad=None) -> np.ndarray:
    """Limit one dataset-cadence arm target update to the SDK rated speed."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if current.shape != target.shape or current.ndim != 1:
        raise ValueError(
            f"current/target must be same-shape vectors, got {current.shape}/{target.shape}"
        )
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("current/target contains NaN/Inf")
    limit = (
        SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS
        if max_delta_rad is None
        else float(max_delta_rad)
    )
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"max_delta_rad must be positive and finite, got {limit}")
    return current + np.clip(target - current, -limit, limit)


def clip_arm_target_to_operational_limits(target) -> np.ndarray:
    """Keep SDK commands inside, rather than exactly on, mechanical limits."""
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(
            f"arm target must have shape ({len(ARM_JOINT_NAMES)},), got {target.shape}"
        )
    if not np.isfinite(target).all():
        raise ValueError("arm target contains NaN/Inf")
    return np.clip(
        target,
        SDK_ARM_COMMAND_LIMITS_RAD[:, 0],
        SDK_ARM_COMMAND_LIMITS_RAD[:, 1],
    )


def _position_issues(
    label: str,
    values: np.ndarray,
    *,
    tolerance_rad: float,
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    lo = ARM_POSITION_LIMITS_RAD[:, 0]
    hi = ARM_POSITION_LIMITS_RAD[:, 1]
    excess = np.maximum(values - hi, lo - values)
    hard = np.argwhere(excess > tolerance_rad)
    outside = np.argwhere(excess > 0.0)
    if hard.size:
        frame, joint = (int(x) for x in hard[0])
        errors.append(
            f"{label} {ARM_JOINT_NAMES[joint]} at frame {frame} is "
            f"{values[frame, joint]:.6f} rad outside "
            f"[{lo[joint]:.6f}, {hi[joint]:.6f}]"
        )
    elif outside.size:
        worst = np.unravel_index(int(np.argmax(excess)), excess.shape)
        frame, joint = (int(x) for x in worst)
        warnings.append(
            f"{label} {ARM_JOINT_NAMES[joint]} at frame {frame} is "
            f"{excess[worst]:.6f} rad beyond the documented range but within "
            f"the {tolerance_rad:.6f} rad state tolerance"
        )
    return errors, warnings


def _uniform_timestamp_errors(timestamp: np.ndarray, n: int, fps: float) -> list[str]:
    original_dtype = np.asarray(timestamp).dtype
    timestamp = np.asarray(timestamp, dtype=np.float64)
    if timestamp.shape != (n,):
        return [f"timestamp shape {timestamp.shape} != ({n},)"]
    if not np.isfinite(timestamp).all():
        return ["timestamp contains NaN/Inf"]
    if n > 1 and not np.all(np.diff(timestamp) > 0):
        return ["timestamp is not strictly increasing"]
    expected = (np.arange(n, dtype=np.float64) + 1.0) / float(fps)
    if np.issubdtype(original_dtype, np.floating) and original_dtype.itemsize <= 4:
        expected = expected.astype(original_dtype).astype(np.float64)
    if not np.allclose(timestamp, expected, atol=1e-6, rtol=0):
        return ["timestamp is not the uniform recorder grid"]
    return []


def _raw_timestamp_cadence(
    timestamp: np.ndarray,
    fps: float,
) -> tuple[list[str], dict]:
    """Reject local jitter or cumulative drift hidden by monotonic timestamps."""
    periods = np.diff(timestamp)
    expected_period_s = 1.0 / float(fps)
    mean_period_s = float(np.mean(periods))
    max_period_error_s = float(np.max(np.abs(periods - expected_period_s)))
    expected_duration_s = float((len(timestamp) - 1) * expected_period_s)
    actual_duration_s = float(timestamp[-1] - timestamp[0])
    duration_drift_s = float(abs(actual_duration_s - expected_duration_s))
    duration_drift_limit_s = float(max(
        SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_S,
        expected_duration_s * SDK_RAW_TIMESTAMP_MAX_DURATION_DRIFT_FRACTION,
    ))

    errors = []
    if max_period_error_s > SDK_RAW_TIMESTAMP_MAX_PERIOD_ERROR_S + 1e-9:
        errors.append(
            f"SDK raw timestamp period error {max_period_error_s:.6f}s > "
            f"{SDK_RAW_TIMESTAMP_MAX_PERIOD_ERROR_S:.6f}s"
        )
    if duration_drift_s > duration_drift_limit_s + 1e-9:
        errors.append(
            f"SDK raw timestamp duration drift {duration_drift_s:.6f}s > "
            f"{duration_drift_limit_s:.6f}s"
        )
    return errors, {
        "expected_period_s": expected_period_s,
        "mean_period_s": mean_period_s,
        "max_period_error_s": max_period_error_s,
        "max_period_error_limit_s": SDK_RAW_TIMESTAMP_MAX_PERIOD_ERROR_S,
        "expected_duration_s": expected_duration_s,
        "actual_duration_s": actual_duration_s,
        "duration_drift_s": duration_drift_s,
        "duration_drift_limit_s": duration_drift_limit_s,
    }


def _camera_timestamp_audit(
    state_timestamp,
    camera_timestamps: Mapping[str, np.ndarray] | None,
    n: int,
    *,
    fps: float,
    require_camera_timestamps: bool,
    max_skew_s: float,
) -> tuple[list[str], float | None, dict | None]:
    if state_timestamp is None or camera_timestamps is None:
        if require_camera_timestamps:
            return ["SDK state/camera header timestamps are required"], None, None
        return [], None, None

    reference = np.asarray(state_timestamp, dtype=np.float64)
    errors = []
    if reference.shape != (n,):
        errors.append(f"SDK state timestamp shape {reference.shape} != ({n},)")
        return errors, None, None
    if not np.isfinite(reference).all() or (n > 1 and not np.all(np.diff(reference) > 0)):
        errors.append("SDK state timestamp must be finite and strictly increasing")
        return errors, None, None

    cadence_errors, cadence = _raw_timestamp_cadence(reference, fps)
    errors.extend(cadence_errors)

    if tuple(camera_timestamps) != SDK_CAMERAS:
        errors.append(
            f"SDK camera timestamp keys must be {list(SDK_CAMERAS)}, "
            f"got {list(camera_timestamps)}"
        )
        return errors, None, cadence

    worst = 0.0
    for camera in SDK_CAMERAS:
        values = np.asarray(camera_timestamps[camera], dtype=np.float64)
        if values.shape != (n,):
            errors.append(f"{camera} timestamp shape {values.shape} != ({n},)")
            continue
        if not np.isfinite(values).all() or (n > 1 and not np.all(np.diff(values) > 0)):
            errors.append(f"{camera} timestamp must be finite and strictly increasing")
            continue
        worst = max(worst, float(np.max(np.abs(values - reference))))
    if not errors and worst > max_skew_s + 1e-9:
        errors.append(
            f"camera/state timestamp skew {worst:.6f}s > {max_skew_s:.6f}s"
        )
    return errors, worst, cadence


def audit_sdk_episode(
    state,
    action,
    base_action,
    *,
    fps: float,
    joint_names: Sequence[str],
    cameras: Sequence[str],
    timestamp=None,
    sdk_state_timestamp=None,
    camera_timestamps: Mapping[str, np.ndarray] | None = None,
    require_camera_timestamps: bool = False,
    enforce_rated_speed: bool = True,
    max_camera_state_skew_s: float = SDK_CAMERA_STATE_MAX_SKEW_S,
) -> dict:
    """Audit a source episode at dataset cadence against the SDK recovery contract."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    base_action = np.asarray(base_action, dtype=np.float64)
    errors: list[str] = []
    warnings: list[str] = []

    if state.ndim != 2 or state.shape[1] < 16:
        errors.append(f"state must be (n, >=16), got {state.shape}")
    if action.ndim != 2 or action.shape[1] < 16:
        errors.append(f"action must be (n, >=16), got {action.shape}")
    if base_action.ndim != 2 or base_action.shape[1] < 2:
        errors.append(f"base_action must be (n, >=2), got {base_action.shape}")

    lengths = [
        len(array)
        for array in (state, action, base_action)
        if array.ndim == 2
    ]
    n = lengths[0] if lengths else 0
    if lengths and any(length != n for length in lengths):
        errors.append(f"state/action/base_action lengths are not aligned: {lengths}")
    if n < 2:
        errors.append("episode needs at least two aligned frames")

    if tuple(joint_names[:14]) != ARM_JOINT_NAMES:
        errors.append(
            f"first 14 joint names must be {list(ARM_JOINT_NAMES)}, "
            f"got {list(joint_names[:14])}"
        )
    if tuple(cameras) != SDK_CAMERAS:
        errors.append(
            f"camera order must be {list(SDK_CAMERAS)}, got {list(cameras)}"
        )
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        errors.append(f"fps must be positive and finite, got {fps}")

    joint_speed_max = None
    joint_speed_joint = None
    joint_speed_frame = None
    if not errors or (
        state.ndim == 2
        and action.ndim == 2
        and base_action.ndim == 2
        and len(state) == len(action) == len(base_action)
        and len(action) >= 2
        and state.shape[1] >= 16
        and action.shape[1] >= 16
        and base_action.shape[1] >= 2
        and math.isfinite(float(fps))
        and float(fps) > 0
    ):
        finite = (
            np.isfinite(state[:, :16]).all()
            and np.isfinite(action[:, :16]).all()
            and np.isfinite(base_action[:, :2]).all()
        )
        if not finite:
            errors.append("state/action/base_action contains NaN/Inf")
        else:
            position_errors, position_warnings = _position_issues(
                "state",
                state[:, :14],
                tolerance_rad=SDK_STATE_POSITION_TOLERANCE_RAD,
            )
            errors.extend(position_errors)
            warnings.extend(position_warnings)
            position_errors, position_warnings = _position_issues(
                "action", action[:, :14], tolerance_rad=1e-6
            )
            errors.extend(position_errors)
            warnings.extend(position_warnings)
            operational_excess = np.maximum(
                action[:, :14] - SDK_ARM_COMMAND_LIMITS_RAD[:, 1],
                SDK_ARM_COMMAND_LIMITS_RAD[:, 0] - action[:, :14],
            )
            operational_outside = np.argwhere(operational_excess > 1e-6)
            if operational_outside.size:
                frame, joint = (int(value) for value in operational_outside[0])
                errors.append(
                    f"action {ARM_JOINT_NAMES[joint]} at frame {frame} is "
                    f"{action[frame, joint]:.6f} rad outside SDK operational limits "
                    f"{SDK_ARM_COMMAND_LIMITS_RAD[joint].tolist()}"
                )
            for label, values in (
                ("state gripper open_frac", state[:, 14:16]),
                ("action gripper open_frac", action[:, 14:16]),
            ):
                if np.any((values < -1e-6) | (values > 1.0 + 1e-6)):
                    errors.append(f"{label} must stay in [0, 1]")

            speed = np.abs(np.diff(action[:, :14], axis=0)) * float(fps)
            speed_index = np.unravel_index(int(np.argmax(speed)), speed.shape)
            joint_speed_max = float(speed[speed_index])
            joint_speed_frame = int(speed_index[0]) + 1
            joint_speed_joint = ARM_JOINT_NAMES[int(speed_index[1])]
            if joint_speed_max > ARM_MAX_SPEED_RAD_S + 1e-6:
                errors.append(
                    f"joint command speed {joint_speed_max:.6f} rad/s > SDK maximum "
                    f"{ARM_MAX_SPEED_RAD_S:.6f} rad/s"
                )
            elif joint_speed_max > ARM_RATED_SPEED_RAD_S + 1e-6:
                message = (
                    f"joint command speed {joint_speed_max:.6f} rad/s > SDK rated "
                    f"{ARM_RATED_SPEED_RAD_S:.6f} rad/s"
                )
                if enforce_rated_speed:
                    errors.append(message)
                else:
                    warnings.append(message)

            v = base_action[:, 0]
            wz = base_action[:, 1]
            if np.any((v < SDK_BASE_V_FWD_RANGE_M_S[0] - 1e-6)
                      | (v > SDK_BASE_V_FWD_RANGE_M_S[1] + 1e-6)):
                errors.append(
                    f"base v_fwd must stay in {SDK_BASE_V_FWD_RANGE_M_S} m/s"
                )
            if np.any((wz < SDK_BASE_WZ_RANGE_RAD_S[0] - 1e-6)
                      | (wz > SDK_BASE_WZ_RANGE_RAD_S[1] + 1e-6)):
                errors.append(
                    f"base wz must stay in {SDK_BASE_WZ_RANGE_RAD_S} rad/s"
                )

    if timestamp is not None and n:
        errors.extend(
            _uniform_timestamp_errors(
                np.asarray(timestamp), n, float(fps)
            )
        )

    timestamp_errors, timestamp_skew, timestamp_cadence = _camera_timestamp_audit(
        sdk_state_timestamp,
        camera_timestamps,
        n,
        fps=float(fps),
        require_camera_timestamps=require_camera_timestamps,
        max_skew_s=float(max_camera_state_skew_s),
    )
    errors.extend(timestamp_errors)

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_frames": int(n),
        "sdk_document_revision": SDK_DOC_REVISION,
        "joint_command_speed": {
            "max_rad_s": joint_speed_max,
            "max_joint": joint_speed_joint,
            "max_frame": joint_speed_frame,
            "rated_limit_rad_s": ARM_RATED_SPEED_RAD_S,
            "absolute_max_rad_s": ARM_MAX_SPEED_RAD_S,
            "rated_limit_enforced": bool(enforce_rated_speed),
        },
        "camera_state_timestamp": {
            "required": bool(require_camera_timestamps),
            "max_skew_s": timestamp_skew,
            "limit_s": float(max_camera_state_skew_s),
            "raw_cadence": timestamp_cadence,
        },
        "gripper_position_polarity_verified": False,
    }


def load_sdk_timestamp_sidecar(path: str):
    sidecar = os.path.join(path, "sdk_timestamps.npz")
    if not os.path.exists(sidecar):
        return None, None
    with np.load(sidecar, allow_pickle=False) as data:
        state_timestamp = (
            np.asarray(data["state_timestamp"])
            if "state_timestamp" in data
            else None
        )
        camera_timestamps = {}
        for camera in SDK_CAMERAS:
            key = f"camera_{camera}_timestamp"
            if key in data:
                camera_timestamps[camera] = np.asarray(data[key])
        if len(camera_timestamps) != len(SDK_CAMERAS):
            camera_timestamps = None
    return state_timestamp, camera_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="source episode directory")
    parser.add_argument(
        "--require-sdk-timestamps",
        action="store_true",
        help="require sdk_timestamps.npz with state and all three camera header stamps",
    )
    parser.add_argument(
        "--allow-above-rated-speed",
        action="store_true",
        help="warn between rated and maximum speed instead of rejecting",
    )
    args = parser.parse_args()

    with open(os.path.join(args.episode, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with np.load(
        os.path.join(args.episode, "episode_data.npz"), allow_pickle=False
    ) as data:
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        base_action = np.asarray(data["base_action"])
        timestamp = np.asarray(data["timestamp"])

    sdk_state_timestamp, camera_timestamps = load_sdk_timestamp_sidecar(
        args.episode
    )
    result = audit_sdk_episode(
        state,
        action,
        base_action,
        fps=float(meta.get("fps", 0)),
        joint_names=meta.get("action_names") or [],
        cameras=list((meta.get("cameras") or {}).keys()),
        timestamp=timestamp,
        sdk_state_timestamp=sdk_state_timestamp,
        camera_timestamps=camera_timestamps,
        require_camera_timestamps=args.require_sdk_timestamps,
        enforce_rated_speed=not args.allow_above_rated_speed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
