#!/usr/bin/env python3
"""Single-episode Sorting Roll expert with physical success gating and review video."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
from cruzr_s2_sdk_contract import (
    SDK_CAMERA_INTRINSICS_VERIFIED,
    SDK_DOCUMENTED_RGB_CAMERA_TOPICS,
    SDK_DOC_REVISION,
    SDK_SENSOR_EXTRINSICS_ZYX,
    SDK_TASK_HEAD_POSE_RAD,
    SDK_WRIST_CAMERAS,
)
from sorting_roll_scene import (
    ROLL_SUPPORT_TOP_Z_M,
    TARGET_AXIS as SCENE_TARGET_AXIS,
    TARGET_CENTER as SCENE_TARGET_CENTER,
    TOP_TIER_BACK_INNER_X_M,
    TOP_TIER_FRONT_LIP_PEAK_Z_M,
    TOP_TIER_FRONT_LIP_X_M,
    TOP_TIER_TROUGH_TOP_Z_M,
)
from sorting_roll_diversity import (
    DIVERSE_TASK_VERSION,
    POSE_BINS,
    apply_model_diversity,
    assignment_for_seed,
    load_manifest,
)
from sorting_roll_realsense_profile import (
    CAMERA_ROLES,
    D405_DEPTH_POLICY_INPUT,
    D405_FOV_DEG,
    D405_IDEAL_RANGE_M,
    D405_MODEL,
    D405_RGB_FPS,
    D405_RGB_RESOLUTION_WH,
    D405_SHUTTER,
    MODEL_CAMERA_SOURCES,
    POLICY_IMAGE_MAP,
    PROFILE_NAME,
    apply_model_camera_overrides,
    profile_report,
    wrist_camera_initialization_report,
)
from sorting_roll_task import REQUIRED_STABLE_SECONDS


TASK_VERSION = "sorting_roll_v15_d405_isomorphic_forward_park_sim"
POLICY_CAMERAS = tuple(MODEL_CAMERA_SOURCES)
REVIEW_ONLY_CAMERAS = ("third_person",)
RECORDED_CAMERAS = POLICY_CAMERAS
UNMODELED_SDK_RGB_CAMERAS = tuple(
    camera
    for camera in SDK_DOCUMENTED_RGB_CAMERA_TOPICS
    if camera not in MODEL_CAMERA_SOURCES.values()
)
FLAT_REGRASP_ORDER = (("r", "l", -1.0), ("l", "r", 1.0))
TARGET_CENTER = np.array(SCENE_TARGET_CENTER, dtype=float, copy=True)
TARGET_AXIS = np.array(SCENE_TARGET_AXIS, dtype=float, copy=True)
ROLL_HALF_LENGTH = 0.25
ROLL_RADIUS = 0.012
HELD_MIN_ROLL_Z_M = TOP_TIER_TROUGH_TOP_Z_M + 0.008
TABLE_OBSERVATION_XY = np.array([0.0, -0.22])
TABLE_GRASP_XY = np.array([0.0, -0.47])
SHELF_STAGE_OFFSET_X = -0.050
TABLE_CLEAR_REVERSE_M = 0.22
FLAT_PICK_TARGET_ALONG_M = 0.160
FLAT_PICK_TIP_BIAS_Y_M = 0.034
FLAT_PICK_PREGRASP_CLEARANCE_Y_M = 0.080
SORTING_ROLL_INITIAL_ARM_PARK = {
    "l": (
        0.49416359, -0.75437207, -0.34254118, -2.21999880,
        0.81456269, 0.35954162, -1.28679103,
    ),
    "r": (
        0.01646851, -0.47162549, 0.97093201, -2.30892391,
        2.20681147, -0.04700812, 0.88190000,
    ),
}
EARLY_COLLISION_MONITOR_PHASES = frozenset({
    "initial_hold",
    "navigate_to_table_observation",
    "localize_roll_with_head_stereo",
    "confirm_task_ready_arm_park_after_stereo_localization",
    "approach_table_with_arms_staged",
    "coordinated_flat_pick_pregrasp_after_stereo_localization",
    "horizontal_approach_and_grasp",
    "lift_flat_from_pickup_support",
})
FLAT_PICK_JOINT_WAYPOINTS = {"l": (), "r": ()}

FLAT_PICK_GOAL_IK_SEEDS = {
    "l": (0.49416359, -0.75437207, -0.34254118, -2.21999880,
          0.81456269, 0.35954162, -1.28679103),
    "r": (0.01646851, -0.47162549, 0.97093201, -2.30892391,
          2.20681147, -0.04700812, 0.88190000),
}
FLAT_PICK_COORDINATION_GRID_STEPS = 120
FLAT_PICK_COORDINATION_CLEARANCE_CELLS = 1
FLAT_PICK_ROLL_CLEARANCE_MARGIN_M = 0.008
FLAT_PICK_COLLISION_STEP_RAD = 0.005
FLAT_PICK_SIDE_MARGIN_M = 0.050
FLAT_PICK_PAD_X_RANGE_M = (-0.030, 0.480)
FLAT_PICK_PAD_ABS_Y_MAX_M = 0.320
FLAT_PICK_PAD_Z_RANGE_M = (0.470, 1.250)
FLAT_PICK_PAD_TABLE_CLEARANCE_M = 0.012
FLAT_PICK_PAD_PATH_MAX_M = 1.150
FLAT_PICK_PAD_BACKTRACK_MAX_M = 0.040
FLAT_PICK_LIFT_M = 0.085
PRE_RELEASE_Y_TOLERANCE_M = 0.003
PRE_RELEASE_ENDPOINT_MARGIN_M = 0.020
ARM_RETRACT_M = 0.082
HAND_FLAT_ROLL_Z = 1.240
RELEASE_APPROACH_Y_BIAS_M = 0.000
RELEASE_CLEARANCE_ROLL_Z = 0.955
RELEASE_REFERENCE_DIAMETER_M = 0.024
RELEASE_GUARDED_DROP_Z_M = 0.951
RELEASE_INSERT_TARGET_X_M = float(TARGET_CENTER[0]) + 0.040
RELEASE_DROP_MAX_M = 0.025
RELEASE_PAD_SHELF_CLEARANCE_MIN_M = 0.002
RELEASE_WRIST_LEVEL_DEG = 4.0
RELEASE_INSERT_STEP_M = 0.006
RELEASE_CLEARANCE_LIFT_M = 0.050
RELEASE_OPEN_INITIAL_BACKOFF_M = 0.010
RELEASE_OPEN_BACKOFF_STEP_M = 0.004
RELEASE_OPEN_BACKOFF_MAX_M = 0.050
RELEASE_OPEN_CLEARANCE_LIFT_MAX_M = 0.010
RELEASE_OPEN_FINAL_SETTLE_TICKS = 12
RELEASE_PAD_SLIDING_FRICTION = 1.0
RELEASE_FRICTION_SETTLE_TICKS = 12
GRASP_YAW_DEG = 14.0
FLAT_REGRASP_ANGLE_DEG = 94.0
FLAT_REGRASP_TARGET_ALONG_M = 0.160
FLAT_REGRASP_COUPLED_START_M = 0.180
FLAT_REGRASP_NEAR_END_M = 0.270
FLAT_REGRASP_FAR_END_M = 0.290
FLAT_REGRASP_CLEARANCE = np.array([0.0, 0.043, 0.020])
FLAT_REGRASP_CLEARANCE_ONSET = 0.45
FLAT_REGRASP_ROTATION_EXPONENT = 2.0
FLAT_REGRASP_COUPLED_MIN_STEPS = 60
FLAT_REGRASP_ABSOLUTE_ROTATION_STEP_DEG = 1.0
FLAT_REGRASP_CART_STEP_M = 0.003
FLAT_REGRASP_COLLISION_STEP_RAD = 0.005
FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M = 0.005
FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M = 0.003
FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M = 0.008
FLAT_REGRASP_ANCHOR_CORRECTION_ATTEMPTS = 3
FLAT_REGRASP_HEIGHT_RESTORE_MAX_STEP_M = 0.015
FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M = 0.003
FLAT_REGRASP_HEIGHT_RESTORE_ATTEMPTS = 4
FLAT_REGRASP_LEVEL_MAX_STEP_M = 0.004
FLAT_REGRASP_LEVEL_TARGET_AXIS_Z = 0.010
FLAT_REGRASP_LEVEL_ATTEMPTS = 4
ENTRY_AXIS_ARM_MAX_STEP_M = 0.004
ENTRY_AXIS_ARM_ATTEMPTS = 4
INSERT_AXIS_X_SAFETY_LIMIT = 0.0012
INSERT_AXIS_Z_SAFETY_LIMIT = 0.02
INSERT_AXIS_CORRECTION_MAX_STEP_M = 0.001
INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M = 0.008
EMPTY_HAND_SERVO_MAX_STEP_RAD = 0.012
ONE_HAND_SUPPORT_DROP_TOLERANCE_M = 0.025
IK_ROTATION_TOLERANCE_DEG = 5.0
ENTRY_X_COMMAND_BIAS = -0.0003
RELEASE_GRIP_RATE = 0.035
BASE_ACCEL = 1.0
BASE_YAW_ACCEL = 0.8
BASE_MAX_SPEED = 0.30
BASE_MAX_YAW_RATE = 0.55
CONTROL_FPS = 60.0
GRIP_FORCE_MIN_N = 0.2
HOLD_CONTACT_RECOVERY_TICKS = 30
ARM_TRACK_TOL_RAD = 0.03
ARM_TRACK_STABLE_TICKS = 6
ARM_TRACK_MAX_TICKS = 900
ARM_SERVO_MAX_STEP_RAD = 0.016
ARM_SERVO_MIN_TICKS = 12
ARM_SERVO_SETTLE_TICKS = 2
GRASP_SETTLE_TICKS = 60
RANDOM_BASE_XY_LIMIT_M = 0.015
RANDOM_BASE_YAW_LIMIT_RAD = 0.025
RANDOM_ROLL_XY_LIMIT_M = 0.004
RANDOM_ROLL_YAW_LIMIT_RAD = 0.012
SLOT_VISUAL_REVIEW_CAMERA = (0.85, -45.0, -45.0)
SLOT_PHYSICS_REVIEW_CAMERA = (0.85, -45.0, -45.0)


class ExpertFailure(RuntimeError):
    pass


def angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def cosine_steps(distance, max_step, minimum=1):
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    return max(
        int(minimum), int(math.ceil(math.pi * float(distance) / (2.0 * max_step)))
    )


def joint_polyline_at_progress(waypoints, progress):
    waypoints = np.asarray(waypoints, dtype=float)
    progress = float(progress)
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise ValueError("joint polyline requires at least two waypoints")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    segment_lengths = np.max(
        np.abs(np.diff(waypoints, axis=0)),
        axis=1,
    )
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("joint polyline contains a zero-length segment")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if progress == 1.0:
        return waypoints[-1].copy()
    distance = progress * float(cumulative[-1])
    segment = min(
        int(np.searchsorted(cumulative, distance, side="right") - 1),
        len(segment_lengths) - 1,
    )
    blend = (
        (distance - cumulative[segment])
        / segment_lengths[segment]
    )
    return (
        waypoints[segment]
        + blend * (waypoints[segment + 1] - waypoints[segment])
    )


def flat_pick_workspace_is_safe(left_pad_base, right_pad_base):
    left_pad_base = np.asarray(left_pad_base, dtype=float)
    right_pad_base = np.asarray(right_pad_base, dtype=float)
    if left_pad_base.shape != (3,) or right_pad_base.shape != (3,):
        raise ValueError("flat-pick pad positions must be 3-D")
    return bool(
        FLAT_PICK_PAD_X_RANGE_M[0]
        <= left_pad_base[0]
        <= FLAT_PICK_PAD_X_RANGE_M[1]
        and FLAT_PICK_PAD_X_RANGE_M[0]
        <= right_pad_base[0]
        <= FLAT_PICK_PAD_X_RANGE_M[1]
        and FLAT_PICK_SIDE_MARGIN_M
        <= left_pad_base[1]
        <= FLAT_PICK_PAD_ABS_Y_MAX_M
        and -FLAT_PICK_PAD_ABS_Y_MAX_M
        <= right_pad_base[1]
        <= -FLAT_PICK_SIDE_MARGIN_M
        and FLAT_PICK_PAD_Z_RANGE_M[0]
        <= left_pad_base[2]
        <= FLAT_PICK_PAD_Z_RANGE_M[1]
        and FLAT_PICK_PAD_Z_RANGE_M[0]
        <= right_pad_base[2]
        <= FLAT_PICK_PAD_Z_RANGE_M[1]
    )


def coordination_clearance_mask(validity, clearance_cells):
    validity = np.asarray(validity, dtype=bool)
    clearance_cells = int(clearance_cells)
    if validity.ndim != 2 or min(validity.shape) < 2:
        raise ValueError("coordination validity must be a 2-D grid")
    if clearance_cells < 0:
        raise ValueError("clearance_cells must be non-negative")
    result = validity.copy()
    if clearance_cells == 0:
        return result
    rows, columns = result.shape
    for row, column in np.argwhere(~validity):
        result[
            max(0, row - clearance_cells):
            min(rows, row + clearance_cells + 1),
            max(0, column - clearance_cells):
            min(columns, column + clearance_cells + 1),
        ] = False
    return result


def monotonic_coordination_indices(validity, edge_is_safe=None):
    validity = np.asarray(validity, dtype=bool)
    if (
        validity.ndim != 2
        or validity.shape[0] != validity.shape[1]
        or validity.shape[0] < 2
    ):
        raise ValueError("coordination validity must be a square grid")
    final = validity.shape[0] - 1
    if not validity[0, 0] or not validity[final, final]:
        return ()

    scores = np.full(validity.shape, np.inf)
    scores[0, 0] = 0.0
    parents = {}
    for left in range(final + 1):
        for right in range(final + 1):
            if (left == 0 and right == 0) or not validity[left, right]:
                continue
            candidates = []
            for delta_left, delta_right in ((1, 1), (1, 0), (0, 1)):
                previous = (
                    left - delta_left,
                    right - delta_right,
                )
                if (
                    previous[0] < 0
                    or previous[1] < 0
                    or not np.isfinite(scores[previous])
                ):
                    continue
                score = (
                    scores[previous]
                    + abs(left - right)
                    + (0.001 if delta_left != delta_right else 0.0)
                )
                direction_rank = 0 if delta_left == delta_right else 1
                candidates.append((score, direction_rank, previous))
            for score, _, previous in sorted(candidates):
                if edge_is_safe is not None and not edge_is_safe(
                    previous,
                    (left, right),
                ):
                    continue
                scores[left, right] = score
                parents[(left, right)] = previous
                break

    if not np.isfinite(scores[final, final]):
        return ()
    path = []
    cell = (final, final)
    while True:
        path.append(cell)
        if cell == (0, 0):
            break
        cell = parents[cell]
    return tuple(reversed(path))

def bounded_vector(vector, max_norm):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if norm <= max_norm:
        return vector.copy()
    return vector * (max_norm / norm)


def anchor_feedback_mount_position(
    command_mount_position,
    target_anchor_position,
    actual_anchor_position,
    max_correction,
):
    return np.asarray(command_mount_position, dtype=float) + bounded_vector(
        np.asarray(target_anchor_position, dtype=float)
        - np.asarray(actual_anchor_position, dtype=float),
        max_correction,
    )


def symmetric_level_correction(
    roll_axis,
    left_anchor,
    right_anchor,
    max_step,
):
    roll_axis = np.asarray(roll_axis, dtype=float)
    left_anchor = np.asarray(left_anchor, dtype=float)
    right_anchor = np.asarray(right_anchor, dtype=float)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    axis_norm = float(np.linalg.norm(roll_axis))
    if axis_norm <= 1e-9:
        raise ValueError("roll_axis must be non-zero")
    roll_axis = roll_axis / axis_norm
    hand_axis = left_anchor - right_anchor
    if float(np.dot(roll_axis, hand_axis)) < 0.0:
        roll_axis = -roll_axis
    horizontal_axis_norm = float(np.linalg.norm(roll_axis[:2]))
    horizontal_separation = float(np.linalg.norm(hand_axis[:2]))
    if horizontal_axis_norm <= 1e-9 or horizontal_separation <= 1e-9:
        raise ValueError("grasp axis must have horizontal separation")
    correction = (
        -0.5
        * float(roll_axis[2])
        / horizontal_axis_norm
        * horizontal_separation
    )
    return float(np.clip(correction, -max_step, max_step))


def symmetric_axis_correction(
    roll_axis,
    left_anchor,
    right_anchor,
    target_axis,
    max_step,
):
    roll_axis = np.asarray(roll_axis, dtype=float)
    left_anchor = np.asarray(left_anchor, dtype=float)
    right_anchor = np.asarray(right_anchor, dtype=float)
    target_axis = np.asarray(target_axis, dtype=float)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    roll_norm = float(np.linalg.norm(roll_axis))
    target_norm = float(np.linalg.norm(target_axis))
    if roll_norm <= 1e-9 or target_norm <= 1e-9:
        raise ValueError("roll_axis and target_axis must be non-zero")
    roll_axis = roll_axis / roll_norm
    target_axis = target_axis / target_norm
    hand_axis = left_anchor - right_anchor
    if float(np.dot(roll_axis, hand_axis)) < 0.0:
        roll_axis = -roll_axis
    if float(np.dot(roll_axis, target_axis)) < 0.0:
        target_axis = -target_axis
    target_component = float(np.dot(roll_axis, target_axis))
    target_separation = abs(float(np.dot(hand_axis, target_axis)))
    if target_component <= 1e-6 or target_separation <= 1e-9:
        raise ValueError("grasp axis must have target-axis separation")
    perpendicular_axis = roll_axis - target_component * target_axis
    correction = (
        -0.5
        * target_separation
        * perpendicular_axis
        / target_component
    )
    return bounded_vector(correction, max_step)


def rotation_x(radians):
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ])


def rotation_z(radians):
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_y(radians):
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])


def vector_angle_degrees(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= float(np.linalg.norm(first))
    second /= float(np.linalg.norm(second))
    return math.degrees(math.acos(float(np.clip(
        np.dot(first, second), -1.0, 1.0
    ))))


def camera_mount_report(mujoco, model, data):
    mujoco.mj_forward(model, data)
    chassis = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
    )
    robot_rotation = data.xmat[chassis].reshape(3, 3)
    task_pitch = abs(float(SDK_TASK_HEAD_POSE_RAD["head_pitch_joint"]))
    head_rotation = rotation_y(task_pitch)

    expected = {}
    for camera in ("stereo_left",):
        raw = SDK_SENSOR_EXTRINSICS_ZYX[camera]
        offset = np.asarray(raw["xyz_m"], dtype=float)
        if camera.startswith("stereo_"):
            offset = head_rotation @ offset
            forward = head_rotation @ np.array([1.0, 0.0, 0.0])
        else:
            pitch = math.radians(float(raw["rpy_deg"][1]))
            forward = np.array([
                math.cos(pitch),
                0.0,
                -math.sin(pitch),
            ])
        expected[camera] = {
            "parent": (
                "chassis"
                if raw["parent_link"] == "base_link"
                else raw["parent_link"]
            ),
            "offset": offset,
            "forward": forward,
            "right": np.array([0.0, -1.0, 0.0]),
        }

    cameras = {}
    for camera, target in expected.items():
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        parent_id = int(model.cam_bodyid[camera_id])
        parent = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, parent_id
        )
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        offset = (
            robot_rotation.T
            @ (data.cam_xpos[camera_id] - data.xpos[parent_id])
        )
        forward = robot_rotation.T @ (-rotation[:, 2])
        right = robot_rotation.T @ rotation[:, 0]
        cameras[camera] = {
            "parent": parent,
            "expected_parent": target["parent"],
            "offset_robot_m": np.round(offset, 8).tolist(),
            "expected_offset_robot_m": np.round(
                target["offset"], 8
            ).tolist(),
            "position_error_mm": round(
                1000.0 * float(np.linalg.norm(offset - target["offset"])),
                5,
            ),
            "forward_robot": np.round(forward, 8).tolist(),
            "forward_error_deg": round(
                vector_angle_degrees(forward, target["forward"]), 5
            ),
            "right_error_deg": round(
                vector_angle_degrees(right, target["right"]), 5
            ),
            "fovy_deg": round(float(model.cam_fovy[camera_id]), 5),
        }
    passed = all(
        item["parent"] == item["expected_parent"]
        and item["position_error_mm"] <= 0.05
        and item["forward_error_deg"] <= 0.05
        and item["right_error_deg"] <= 0.05
        for item in cameras.values()
    )
    return {
        "passed": passed,
        "source": f"CRUZR S2 SDK {SDK_DOC_REVISION} section 1.4.2",
        "intrinsics_verified": SDK_CAMERA_INTRINSICS_VERIFIED,
        "fovy_status": "simulation_assumption_pending_real_CameraInfo",
        "cameras": cameras,
    }




def rotation_axis_angle(axis, radians):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("rotation axis must be non-zero")
    axis = axis / norm
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    radians = float(radians)
    return (
        np.eye(3)
        + math.sin(radians) * skew
        + (1.0 - math.cos(radians)) * (skew @ skew)
    )


def grasp_target_rotation(base_rotation, direction):
    return (
        rotation_z(float(direction) * math.radians(GRASP_YAW_DEG))
        @ np.asarray(base_rotation, dtype=float)
    )


def flatten_target_rotation(initial_rotation, progress):
    return (
        rotation_x(
            -float(progress) * math.radians(FLAT_REGRASP_ANGLE_DEG)
        )
        @ np.asarray(initial_rotation, dtype=float)
    )


def coupled_regrasp_progress(progress):
    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    clearance_progress = 0.0
    if progress > FLAT_REGRASP_CLEARANCE_ONSET:
        clearance_progress = (
            progress - FLAT_REGRASP_CLEARANCE_ONSET
        ) / (1.0 - FLAT_REGRASP_CLEARANCE_ONSET)
    return progress**FLAT_REGRASP_ROTATION_EXPONENT, clearance_progress


def cartesian_waypoints(start, target, max_step):
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(target - start))
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    steps = max(1, int(math.ceil(distance / float(max_step))))
    return [
        start + (target - start) * (index / steps)
        for index in range(1, steps + 1)
    ]


def flat_regrasp_anchors(roll_position, roll_axis, current_anchor, direction):
    roll_position = np.asarray(roll_position, dtype=float)
    roll_axis = np.asarray(roll_axis, dtype=float)
    current_anchor = np.asarray(current_anchor, dtype=float)
    norm = float(np.linalg.norm(roll_axis))
    if norm <= 1e-9:
        raise ValueError("roll_axis must be non-zero")
    roll_axis = roll_axis / norm
    direction = math.copysign(1.0, float(direction))
    radial = (
        current_anchor
        - roll_position
        - np.dot(current_anchor - roll_position, roll_axis) * roll_axis
    )
    far_end = (
        roll_position
        + direction * FLAT_REGRASP_FAR_END_M * roll_axis
        + radial
    )
    axis_far = (
        roll_position
        + direction * FLAT_REGRASP_FAR_END_M * roll_axis
    )
    target = (
        roll_position
        + direction * FLAT_REGRASP_TARGET_ALONG_M * roll_axis
    )
    return {
        "far_end": far_end,
        "axis_far": axis_far,
        "target": target,
    }


def anchored_mount_position(
    mount_position,
    mount_rotation,
    anchor_position,
    target_rotation,
    target_anchor_position=None,
):
    mount_position = np.asarray(mount_position, dtype=float)
    mount_rotation = np.asarray(mount_rotation, dtype=float)
    anchor_position = np.asarray(anchor_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if target_anchor_position is None:
        target_anchor_position = anchor_position
    target_anchor_position = np.asarray(
        target_anchor_position, dtype=float
    )
    anchor_in_mount = mount_rotation.T @ (anchor_position - mount_position)
    return target_anchor_position - target_rotation @ anchor_in_mount


def mount_position_for_pad_target(
    target_pad_position,
    target_rotation,
    pad_offset_in_mount,
):
    return np.asarray(target_pad_position, dtype=float) - (
        np.asarray(target_rotation, dtype=float)
        @ np.asarray(pad_offset_in_mount, dtype=float)
    )


def roll_half_extent_x(
    axis_x,
    half_length=ROLL_HALF_LENGTH,
    radius=ROLL_RADIUS,
):
    axis_x = abs(float(axis_x))
    return (
        float(half_length) * axis_x
        + float(radius) * math.sqrt(max(0.0, 1.0 - axis_x * axis_x))
    )


def integrated_depth_margin(
    center_x,
    axis_x,
    half_length=ROLL_HALF_LENGTH,
    radius=ROLL_RADIUS,
):
    half_x = roll_half_extent_x(axis_x, half_length, radius)
    center_x = float(center_x)
    return min(
        center_x - half_x - TOP_TIER_FRONT_LIP_X_M,
        TOP_TIER_BACK_INNER_X_M - center_x - half_x,
    )


def guarded_release_center_z(roll_radius):
    return (
        RELEASE_GUARDED_DROP_Z_M
        + float(roll_radius) - 0.5 * RELEASE_REFERENCE_DIAMETER_M
    )


def guarded_release_is_ready(roll_clearance, pad_clearance):
    return bool(
        0.0 <= float(roll_clearance) <= RELEASE_DROP_MAX_M
        and float(pad_clearance)
        >= RELEASE_PAD_SHELF_CLEARANCE_MIN_M
    )


def guarded_release_geometry_is_ready(
    endpoint_margins,
    axis_error_deg,
    depth_margin,
):
    return bool(
        min(float(value) for value in endpoint_margins.values())
        >= PRE_RELEASE_ENDPOINT_MARGIN_M
        and float(axis_error_deg) <= 5.0
        and float(depth_margin) >= 0.005
    )


def resolved_geom_clearance(raw_distance, witness_distance, has_contact):
    raw_distance = float(raw_distance)
    witness_distance = float(witness_distance)
    if has_contact or raw_distance < 0.0:
        return min(raw_distance, 0.0)
    if raw_distance == 0.0 and witness_distance > 0.0:
        return witness_distance
    return raw_distance


def insertion_axis_is_safe(roll_axis):
    roll_axis = np.asarray(roll_axis, dtype=float)
    return bool(
        abs(float(roll_axis[0])) <= INSERT_AXIS_X_SAFETY_LIMIT
        and abs(float(roll_axis[2])) <= INSERT_AXIS_Z_SAFETY_LIMIT
    )


def insertion_axis_correction_has_clearance(
    roll_clearance,
    pad_clearance,
):
    return bool(
        min(float(roll_clearance), float(pad_clearance))
        >= INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M
    )


def seed_randomization(seed, pose_bin=None):
    rng = np.random.default_rng(int(seed))
    limits = np.asarray([
        RANDOM_BASE_XY_LIMIT_M,
        RANDOM_BASE_XY_LIMIT_M,
        RANDOM_BASE_YAW_LIMIT_RAD,
        RANDOM_ROLL_XY_LIMIT_M,
        RANDOM_ROLL_XY_LIMIT_M,
        RANDOM_ROLL_YAW_LIMIT_RAD,
    ])
    if pose_bin is None:
        normalized = rng.uniform(-1.0, 1.0, len(limits))
    else:
        if pose_bin not in POSE_BINS:
            raise ValueError(f"unknown pose bin: {pose_bin}")
        upper = POSE_BINS[pose_bin]["normalized_max"]
        normalized = rng.uniform(-upper, upper, len(limits))
        lower = POSE_BINS[pose_bin]["normalized_min"]
        if lower > 0.0:
            focus = int(rng.integers(0, len(limits)))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            normalized[focus] = sign * rng.uniform(lower, upper)
    values = normalized * limits
    return {
        "pose_bin": pose_bin,
        "base_delta_xyyaw": values[:3].tolist(),
        "roll_delta_xy_m": values[3:5].tolist(),
        "roll_yaw_rad": float(values[5]),
    }


def release_is_clear(left, right):
    return (
        not left["pads"]
        and not right["pads"]
        and left["force_n"] <= 0.05
        and right["force_n"] <= 0.05
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new episode output directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--review-videos", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.seed < 1:
        parser.error("--seed must be positive")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.width < 224 or args.height < 224:
        parser.error("record dimensions must both be at least 224")
    if args.no_render and args.review_videos:
        parser.error("--review-videos requires rendering")
    if args.manifest and not args.randomize:
        parser.error("--manifest requires --randomize")
    return args


def load_teleop(scene_path, gpu, seed, prompt=None):
    os.environ["TELEOP_SCENE_XML"] = str(scene_path)
    os.environ["TELEOP_VIEWER"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["TELEOP_RECORD_GPU"] = str(gpu)
    os.environ["CRUZR_GRIP_CLOSE"] = "0.025"
    os.environ["CRUZR_EP_SEED"] = str(seed)
    os.environ["REC_CAMS"] = ",".join(RECORDED_CAMERAS)
    os.environ["REC_CAMERA_SOURCES"] = ",".join(
        f"{logical}={source}"
        for logical, source in MODEL_CAMERA_SOURCES.items()
    )
    os.environ["REC_SAVE_RAW_TIMESTAMPS"] = "1"
    os.environ["REC_PROMPT"] = prompt or (
        "Pick up the roll from the table and place it stably in the top shelf slot"
    )
    spec = importlib.util.spec_from_file_location(
        "cruzr_teleop", CORE_DIR / "cruzr_teleop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_sorting_roll_initial_park(ct, mujoco):
    for hand, values in SORTING_ROLL_INITIAL_ARM_PARK.items():
        arm = ct.ARMS[hand]
        target = np.asarray(values, dtype=float)
        if target.shape != (7,):
            raise RuntimeError(f"invalid {hand} initial arm park shape")
        if np.any(target < arm.lo) or np.any(target > arm.hi):
            raise RuntimeError(f"{hand} initial arm park exceeds joint limits")
        for address, value in zip(arm.qadr, target):
            ct.d.qpos[address] = value
        for actuator, value in zip(arm.arm_acts, target):
            ct.d.ctrl[actuator] = value
        ct.qtgt[hand] = target.copy()
        ct.grip_cmd[hand] = ct.GRIP_OPEN
        ct.set_gripper_state(arm, ct.GRIP_OPEN)
    ct.d.qvel[:] = 0.0
    mujoco.mj_forward(ct.m, ct.d)
    for _ in range(240):
        mujoco.mj_step(ct.m, ct.d)
    measured = {}
    maximum_error = 0.0
    for hand, values in SORTING_ROLL_INITIAL_ARM_PARK.items():
        arm = ct.ARMS[hand]
        target = np.asarray(values, dtype=float)
        measured[hand] = np.asarray([
            ct.d.qpos[address] for address in arm.qadr
        ])
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(measured[hand] - target))),
        )
        ct.qtgt[hand] = target.copy()
        ct.grip_cmd[hand] = ct.GRIP_OPEN
    ct.d.qvel[:] = 0.0
    mujoco.mj_forward(ct.m, ct.d)
    return {
        "passed": maximum_error <= ARM_TRACK_TOL_RAD,
        "target_joint_positions_rad": {
            hand: list(values)
            for hand, values in SORTING_ROLL_INITIAL_ARM_PARK.items()
        },
        "measured_joint_positions_rad": {
            hand: np.round(values, 6).tolist()
            for hand, values in measured.items()
        },
        "maximum_tracking_error_rad": round(maximum_error, 6),
        "settle_steps": 240,
    }


def render_third_person(
    recorder,
    mujoco,
    model,
    data,
    camera,
    output_path,
    *,
    scene_option=None,
    hidden_geom_ids=(),
):
    from PIL import Image

    recorder._ensure_gl()
    recorder._gl.make_current()
    saved_alpha = {
        int(geom): float(model.geom_rgba[int(geom), 3])
        for geom in hidden_geom_ids
    }
    try:
        for geom in saved_alpha:
            model.geom_rgba[geom, 3] = 0.0
        mujoco.mjv_updateScene(
            model,
            data,
            recorder._opt if scene_option is None else scene_option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            recorder._scn,
        )
        mujoco.mjr_render(recorder._vp, recorder._scn, recorder._con)
        width, height = recorder._vp.width, recorder._vp.height
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, recorder._vp, recorder._con)
        Image.fromarray(np.flipud(rgb)).save(output_path, quality=90)
    finally:
        for geom, alpha in saved_alpha.items():
            model.geom_rgba[geom, 3] = alpha


class SortingRollExpert:
    def __init__(self, args, ct, mujoco, scheduler, evaluate_placement, tracker_cls):
        self.args = args
        self.ct = ct
        self.mujoco = mujoco
        self.scheduler = scheduler
        self.evaluate_placement = evaluate_placement
        self.tracker_cls = tracker_cls
        self.model = ct.m
        self.data = ct.d
        apply_model_camera_overrides(
            self.mujoco,
            self.model,
        )
        self.out = Path(args.out).resolve()
        self.review_dir = self.out / "diagnostics" / "third_person"
        self.review_video = self.out / "sorting_roll_review.mp4"
        self.slot_visual_review_dir = (
            self.out / "diagnostics" / "slot_visual_closeup"
        )
        self.slot_visual_review_video = (
            self.out / "sorting_roll_slot_visual_closeup.mp4"
        )
        self.slot_physics_review_dir = (
            self.out / "diagnostics" / "slot_physics_closeup"
        )
        self.slot_physics_review_video = (
            self.out / "sorting_roll_slot_physics_closeup.mp4"
        )
        self.robot_camera_videos = {
            camera: self.out / f"sorting_roll_{camera}.mp4"
            for camera in RECORDED_CAMERAS
        }
        self.robot_multiview_video = (
            self.out / "sorting_roll_robot_multiview.mp4"
        )
        self.result_path = self.out / "result.json"
        self.roll_body = ct.bid("sorting_roll")
        self.roll_geom = ct.gid("sorting_roll_col")
        self.shelf_visual_geom = ct.gid("sorting_shelf_visual")
        self.slot_visual_geom_ids = set()
        self.roll_joint = ct.jid("sorting_roll_free")
        self.roll_qpos_adr = int(self.model.jnt_qposadr[self.roll_joint])
        self.roll_dof_adr = int(self.model.jnt_dofadr[self.roll_joint])
        self.diversity_assignment = getattr(
            args, "diversity_assignment", None
        )
        self.task_version = (
            DIVERSE_TASK_VERSION
            if self.diversity_assignment is not None
            else TASK_VERSION
        )
        self.diversity = None
        if self.diversity_assignment is not None:
            applied = apply_model_diversity(
                self.mujoco,
                self.model,
                self.data,
                self.diversity_assignment,
            )
            roll_radius = 0.5 * float(
                self.diversity_assignment["object_profile"]["diameter_m"]
            )
            self.data.qpos[self.roll_qpos_adr + 2] = (
                ROLL_SUPPORT_TOP_Z_M + roll_radius + 0.0015
            )
            self.ct.REC_JPEG_Q = int(
                self.diversity_assignment["image_profile"]["jpeg_quality"]
            )
            self.mujoco.mj_forward(self.model, self.data)
            self.diversity = {
                "assignment": self.diversity_assignment,
                "applied": applied,
                "manifest": str(args.manifest.resolve()),
            }
        args.initial_arm_park_report = apply_sorting_roll_initial_park(
            self.ct, self.mujoco
        )
        self.pad_ids = {
            ct.gid(name)
            for name in ("L_pad1", "L_pad2", "R_pad1", "R_pad2")
        }
        self.table_top_geom = ct.gid("table_top_col")
        self.pickup_support_geom_ids = {
            ct.gid(name)
            for name in (
                "roll_support_x_negative_base_col",
                "roll_support_x_positive_base_col",
                "roll_support_x_negative_robot_lip_col",
                "roll_support_x_negative_far_lip_col",
                "roll_support_x_positive_robot_lip_col",
                "roll_support_x_positive_far_lip_col",
            )
        }
        self.integrated_support_geom_ids = {
            ct.gid(name)
            for name in (
                "shelf_top_front_lip_col",
                "shelf_top_trough_col",
                "shelf_top_back_slope_col",
            )
        }
        self.shelf_geom_ids = {
            ct.gid(name)
            for name in (
                "shelf_post_front_left_col",
                "shelf_post_front_right_col",
                "shelf_post_rear_left_col",
                "shelf_post_rear_right_col",
                "shelf_tier1_col",
                "shelf_tier2_col",
                "shelf_tier3_col",
                "shelf_top_front_lip_col",
                "shelf_top_trough_col",
                "shelf_top_back_slope_col",
                "shelf_top_back_panel_col",
            )
        }
        self.arm_geom_ids = {}
        for hand, root_name in (
            ("l", "L_shoulder_pitch_link"),
            ("r", "R_shoulder_pitch_link"),
        ):
            root_body = ct.bid(root_name)
            body_ids = {root_body}
            for body in range(self.model.nbody):
                parent = int(self.model.body_parentid[body])
                if parent in body_ids:
                    body_ids.add(body)
            self.arm_geom_ids[hand] = {
                geom
                for geom in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom]) in body_ids
            }
        self.recorded_roll_qpos = []
        self.recorded_roll_qvel = []
        self.gates = {}
        self.final_evidence = None
        self.sim_seconds = 0.0
        self.early_collision_checks = 0
        self.early_collision_events = []

        self.out.mkdir(parents=True)
        if args.review_videos:
            self.review_dir.mkdir(parents=True)
            self.slot_visual_review_dir.mkdir(parents=True)
            self.slot_physics_review_dir.mkdir(parents=True)
        ct.REC_WH = (args.width, args.height)
        self.recorder = ct.EpisodeRecorder(str(self.out))
        ct.REC.update({
            "rec": self.recorder,
            "on": True,
            "count": 0,
            "phase": "initial_hold",
            "metadata": {
                "task_version": self.task_version,
                "seed": args.seed,
                "collection_profile": PROFILE_NAME,
                "initial_arm_park": args.initial_arm_park_report,
                "diversity": self.diversity,
                "training_eligible": False,
                "simulation_canary_eligible": False,
                "success_source": "sorting_roll_task.SortingRollSuccessTracker",
                "policy_cameras": list(POLICY_CAMERAS),
                "policy_image_map": dict(POLICY_IMAGE_MAP),
                "review_only_cameras": list(REVIEW_ONLY_CAMERAS),
                "review_videos_enabled": bool(args.review_videos),
                "recorded_cameras": list(RECORDED_CAMERAS),
                "model_camera_sources": dict(MODEL_CAMERA_SOURCES),
                "camera_roles": dict(CAMERA_ROLES),
                "camera_model": D405_MODEL,
                "d405_rgb_resolution_wh": list(D405_RGB_RESOLUTION_WH),
                "d405_rgb_fps": D405_RGB_FPS,
                "d405_fov_deg": list(D405_FOV_DEG),
                "d405_ideal_range_m": list(D405_IDEAL_RANGE_M),
                "d405_shutter": D405_SHUTTER,
                "depth_policy_input": D405_DEPTH_POLICY_INPUT,
                "realsense_candidate_profile": profile_report(),
                "sdk_document_revision": SDK_DOC_REVISION,
                "sdk_documented_rgb_cameras": list(
                    SDK_DOCUMENTED_RGB_CAMERA_TOPICS
                ),
                "sdk_wrist_cameras": list(SDK_WRIST_CAMERAS),
                "unmodeled_sdk_rgb_cameras": list(
                    UNMODELED_SDK_RGB_CAMERAS
                ),
                "synthetic_wrist_cameras_recorded": True,
                "camera_extrinsics_source": (
                    "stereo_left_from_CRUZR_SDK; "
                    "dual_D405_same_local_installation_transform_"
                    "pending_real_mount_measurement"
                ),
                "camera_intrinsics_verified": SDK_CAMERA_INTRINSICS_VERIFIED,
                "camera_fovy_status": (
                    "D405_nominal_FOV_pending_real_CameraInfo"
                ),
                "review_camera": "free_camera_azimuth_45_not_policy_input",
                "slot_visual_review_camera": (
                    "free_camera_azimuth_-45_elevation_-45_"
                    "integrated_shelf_visible_geometry_not_policy_input"
                ),
                "slot_physics_review_camera": (
                    "free_camera_azimuth_-45_elevation_-45_"
                    "integrated_shelf_collision_geometry_not_policy_input"
                ),
                "release_pad_sliding_friction": (
                    RELEASE_PAD_SLIDING_FRICTION
                ),
            },
        })

        self.review_camera = mujoco.MjvCamera()
        self.review_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.review_camera.lookat[:] = [0.40, -0.45, 0.82]
        self.review_camera.distance = 2.55
        self.review_camera.azimuth = 45.0
        self.review_camera.elevation = -22.0


        self.visual_review_option = mujoco.MjvOption()
        self.visual_review_option.geomgroup[3] = 0

        self.slot_visual_review_camera = mujoco.MjvCamera()
        self.slot_visual_review_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.slot_visual_review_camera.lookat[:] = TARGET_CENTER
        (
            self.slot_visual_review_camera.distance,
            self.slot_visual_review_camera.azimuth,
            self.slot_visual_review_camera.elevation,
        ) = SLOT_VISUAL_REVIEW_CAMERA
        self.slot_physics_review_camera = mujoco.MjvCamera()
        self.slot_physics_review_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.slot_physics_review_camera.lookat[:] = TARGET_CENTER
        (
            self.slot_physics_review_camera.distance,
            self.slot_physics_review_camera.azimuth,
            self.slot_physics_review_camera.elevation,
        ) = SLOT_PHYSICS_REVIEW_CAMERA
        self.slot_physics_review_option = mujoco.MjvOption()
        self.slot_physics_review_option.geomgroup[3] = 1
        self.apply_scene_randomization()

    def apply_scene_randomization(self):
        report = {
            "enabled": bool(self.args.randomize),
            "seed": self.args.seed,
        }
        if not self.args.randomize:
            self.scene_randomization = report
            self.ct.REC["metadata"]["scene_randomization"] = report
            return
        pose_bin = (
            self.diversity_assignment["pose_bin"]
            if self.diversity_assignment is not None
            else None
        )
        values = seed_randomization(self.args.seed, pose_bin=pose_bin)
        for address, delta in zip(
            self.ct.BQ,
            values["base_delta_xyyaw"],
        ):
            self.data.qpos[address] += delta
        self.data.qpos[
            self.roll_qpos_adr:self.roll_qpos_adr + 2
        ] += values["roll_delta_xy_m"]
        half_yaw = 0.5 * values["roll_yaw_rad"]
        yaw_quaternion = np.asarray([
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ])
        rotated = np.empty(4, dtype=float)
        self.mujoco.mju_mulQuat(
            rotated,
            yaw_quaternion,
            self.data.qpos[
                self.roll_qpos_adr + 3:self.roll_qpos_adr + 7
            ],
        )
        self.data.qpos[
            self.roll_qpos_adr + 3:self.roll_qpos_adr + 7
        ] = rotated
        self.mujoco.mj_forward(self.model, self.data)
        report.update(values)
        self.scene_randomization = report
        self.ct.REC["metadata"]["scene_randomization"] = report

    def phase(self, name):
        self.ct.REC["phase"] = name
        print(f"[phase] {name}", flush=True)

    def tick(self):
        previous = self.recorder.n
        substeps = self.scheduler.next_substeps()
        self.ct.control_step(substeps)
        dt = float(substeps) * float(self.model.opt.timestep)
        self.sim_seconds += dt
        phase = self.ct.REC["phase"]
        if phase in EARLY_COLLISION_MONITOR_PHASES:
            self.early_collision_checks += 1
            contacts = self.early_unintended_arm_contacts(
                allow_pad_roll=(
                    phase in {
                        "horizontal_approach_and_grasp",
                        "lift_flat_from_pickup_support",
                    }
                )
            )
            if contacts:
                event = {
                    "phase": phase,
                    "sim_seconds": round(self.sim_seconds, 4),
                    "contacts": contacts,
                }
                self.early_collision_events.append(event)
                raise ExpertFailure(
                    "unintended early arm collision "
                    + json.dumps(event, ensure_ascii=False)
                )
        if self.recorder.n != previous:
            self.recorded_roll_qpos.append(
                self.data.qpos[self.roll_qpos_adr:self.roll_qpos_adr + 7].copy()
            )
            self.recorded_roll_qvel.append(
                self.data.qvel[self.roll_dof_adr:self.roll_dof_adr + 6].copy()
            )
            if self.args.review_videos:
                render_third_person(
                    self.recorder,
                    self.mujoco,
                    self.model,
                    self.data,
                    self.review_camera,
                    self.review_dir
                    / f"frame_{self.recorder.n - 1:06d}.jpg",
                    scene_option=self.visual_review_option,
                )
                render_third_person(
                    self.recorder,
                    self.mujoco,
                    self.model,
                    self.data,
                    self.slot_visual_review_camera,
                    self.slot_visual_review_dir
                    / f"frame_{self.recorder.n - 1:06d}.jpg",
                    scene_option=self.visual_review_option,
                )
                render_third_person(
                    self.recorder,
                    self.mujoco,
                    self.model,
                    self.data,
                    self.slot_physics_review_camera,
                    self.slot_physics_review_dir
                    / f"frame_{self.recorder.n - 1:06d}.jpg",
                    scene_option=self.slot_physics_review_option,
                    hidden_geom_ids=(
                        self.shelf_visual_geom,
                        *sorted(self.slot_visual_geom_ids),
                    ),
                )
        return dt

    def frames(self, count):
        for _ in range(int(count)):
            self.tick()

    def gate(self, name, passed, detail):
        passed = bool(passed)
        self.gates[name] = {"passed": passed, "detail": detail}
        print(f"[gate:{name}] {'PASS' if passed else 'FAIL'} {detail}", flush=True)
        if not passed:
            raise ExpertFailure(f"{name}: {detail}")

    def roll_position(self):
        return self.data.xpos[self.roll_body].copy()

    def arm_joint_positions(self):
        return {
            hand: np.array(
                [self.data.qpos[address] for address in arm.qadr],
                dtype=float,
            )
            for hand, arm in (("l", self.ct.L), ("r", self.ct.R))
        }

    def early_unintended_arm_contacts(self, allow_pad_roll=False):
        contacts = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if (
                geom1 in self.arm_geom_ids["l"]
                and geom2 in self.arm_geom_ids["l"]
            ) or (
                geom1 in self.arm_geom_ids["r"]
                and geom2 in self.arm_geom_ids["r"]
            ):
                continue
            pair = {geom1, geom2}
            if not pair & (
                self.arm_geom_ids["l"] | self.arm_geom_ids["r"]
            ):
                continue
            if allow_pad_roll and self.roll_geom in pair:
                other = geom2 if geom1 == self.roll_geom else geom1
                if other in self.pad_ids:
                    continue
            contacts.append({
                "pair": [self.geom_label(geom1), self.geom_label(geom2)],
                "penetration_mm": round(
                    -1000.0 * min(
                        0.0, float(self.data.contact[index].dist)
                    ),
                    3,
                ),
            })
        return contacts

    def contact_evidence(self, first_ids, second_ids):
        first_ids = set(first_ids)
        second_ids = set(second_ids)
        force = np.zeros(6, dtype=float)
        total = 0.0
        pairs = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            pair = {geom1, geom2}
            if not pair & first_ids or not pair & second_ids:
                continue
            self.mujoco.mj_contactForce(
                self.model, self.data, index, force
            )
            total += abs(float(force[0]))
            pairs.append([
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom1
                ),
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom2
                ),
            ])
        return {"force_n": total, "pairs": pairs}

    def arm_shelf_contacts(self):
        all_arm = self.arm_geom_ids["l"] | self.arm_geom_ids["r"]
        contacts = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if not (
                (geom1 in all_arm and geom2 in self.shelf_geom_ids)
                or (geom2 in all_arm and geom1 in self.shelf_geom_ids)
            ):
                continue
            contacts.append({
                "pair": [self.geom_label(geom1), self.geom_label(geom2)],
                "penetration_mm": round(
                    -1000.0 * min(
                        0.0, float(self.data.contact[index].dist)
                    ),
                    3,
                ),
            })
        return contacts

    def minimum_geom_clearance(self, first_ids, second_ids):
        active_pairs = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            active_pairs.add(frozenset((
                int(contact.geom1),
                int(contact.geom2),
            )))
        best = None
        for first in sorted(first_ids):
            for second in sorted(second_ids):
                fromto = np.zeros(6, dtype=float)
                raw_distance = float(self.mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    int(first),
                    int(second),
                    0.25,
                    fromto,
                ))
                witness_distance = float(np.linalg.norm(
                    fromto[3:] - fromto[:3]
                ))
                has_contact = (
                    frozenset((int(first), int(second))) in active_pairs
                )
                distance = resolved_geom_clearance(
                    raw_distance,
                    witness_distance,
                    has_contact,
                )
                if best is None or distance < best["distance_m"]:
                    best = {
                        "distance_m": distance,
                        "raw_distance_m": raw_distance,
                        "witness_distance_m": witness_distance,
                        "active_contact": has_contact,
                        "pair": [
                            self.geom_label(first),
                            self.geom_label(second),
                        ],
                        "fromto_m": np.round(fromto, 6).tolist(),
                    }
        if best is None:
            raise RuntimeError("geom clearance requires non-empty sets")
        return best


    def require_arms_clear_shelf(self, label):
        contacts = self.arm_shelf_contacts()
        if contacts:
            raise ExpertFailure(
                f"arm-shelf collision phase={label} contacts={contacts}"
            )

    def geom_label(self, geom):
        name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom)
        )
        if name:
            return name
        body = int(self.model.geom_bodyid[int(geom)])
        body_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body
        )
        return body_name or f"geom_{int(geom)}"

    def moving_arm_contacts(self, hand):
        moving = self.arm_geom_ids[hand]
        contacts = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if geom1 in moving and geom2 in moving:
                continue
            if geom1 not in moving and geom2 not in moving:
                continue
            contacts.append({
                "pair": [self.geom_label(geom1), self.geom_label(geom2)],
                "penetration_mm": round(
                    -1000.0 * min(0.0, float(self.data.contact[index].dist)),
                    3,
                ),
            })
        return contacts

    def pad_vertical_span(self):
        spans = {}
        for geom in self.pad_ids:
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            half_span = float(
                np.abs(rotation[2]) @ self.model.geom_size[geom, :3]
            )
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom
            )
            spans[name] = 2.0 * half_span
        return spans

    def grip_evidence(self, hand):
        pad_ids = {
            self.ct.gid(f"{hand}_pad1"),
            self.ct.gid(f"{hand}_pad2"),
        }
        force = np.zeros(6, dtype=float)
        total = 0.0
        contacts = set()
        for index in range(self.data.ncon):
            pair = {
                int(self.data.contact[index].geom1),
                int(self.data.contact[index].geom2),
            }
            if self.roll_geom in pair and pair & pad_ids:
                self.mujoco.mj_contactForce(
                    self.model, self.data, index, force
                )
                total += abs(float(force[0]))
                contacts.update(pair & pad_ids)
        return {
            "force_n": total,
            "pads": sorted(
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom
                )
                for geom in contacts
            ),
        }

    def roll_pad_contact_geometry(self):
        details = []
        force = np.zeros(6, dtype=float)
        roll_center = self.roll_position()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            pad_pair = pair & self.pad_ids
            if self.roll_geom not in pair or not pad_pair:
                continue
            pad = next(iter(pad_pair))
            self.mujoco.mj_contactForce(
                self.model, self.data, index, force
            )
            pad_center = self.data.geom_xpos[pad].copy()
            details.append({
                "pad": self.geom_label(pad),
                "geom1": self.geom_label(int(contact.geom1)),
                "geom2": self.geom_label(int(contact.geom2)),
                "force_n": round(abs(float(force[0])), 6),
                "contact_position_m": np.round(
                    contact.pos, 6
                ).tolist(),
                "contact_normal_world": np.round(
                    contact.frame[:3], 6
                ).tolist(),
                "pad_center_m": np.round(pad_center, 6).tolist(),
                "roll_center_m": np.round(roll_center, 6).tolist(),
                "roll_to_pad_center_m": np.round(
                    pad_center - roll_center, 6
                ).tolist(),
            })
        return details

    def set_base_velocity(self, forward, yaw_rate):
        dt = (1.0 / CONTROL_FPS) * self.ct.REC_DECIM
        self.ct.base_vel[0] += float(np.clip(
            forward - self.ct.base_vel[0],
            -BASE_ACCEL * dt,
            BASE_ACCEL * dt,
        ))
        self.ct.base_vel[1] += float(np.clip(
            yaw_rate - self.ct.base_vel[1],
            -BASE_YAW_ACCEL * dt,
            BASE_YAW_ACCEL * dt,
        ))

    def stop_base(self):
        for _ in range(90):
            if max(abs(float(v)) for v in self.ct.base_vel) < 1e-3:
                break
            self.set_base_velocity(0.0, 0.0)
            self.frames(self.ct.REC_DECIM)
        self.ct.base_vel[:] = 0.0
        self.frames(6)

    @staticmethod
    def brake_cap(remaining, acceleration):
        return math.sqrt(max(0.0, 2.0 * acceleration * abs(float(remaining))))

    def turn_in_place(
        self,
        target_yaw,
        max_rate=BASE_MAX_YAW_RATE,
        tolerance=0.012,
    ):
        self.stop_base()
        for _ in range(2400):
            error = angle(target_yaw - self.ct.base_pose()[2])
            if abs(error) <= tolerance and abs(self.ct.base_vel[1]) <= 0.015:
                break
            rate_cap = min(
                max_rate,
                self.brake_cap(error, BASE_YAW_ACCEL),
            )
            self.set_base_velocity(
                0.0,
                float(np.clip(1.8 * error, -rate_cap, rate_cap)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure(f"turn timeout target={target_yaw:.3f}")
        self.stop_base()

    def go_to(
        self,
        target_xy,
        target_yaw,
        max_speed=BASE_MAX_SPEED,
        tolerance=0.008,
    ):
        target_xy = np.asarray(target_xy, dtype=float)
        pose = self.ct.base_pose()
        if float(np.linalg.norm(target_xy - pose[:2])) > tolerance:
            heading = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
            self.turn_in_place(heading)
        for _ in range(5000):
            x, y, yaw = self.ct.base_pose()
            delta = target_xy - np.array([x, y])
            distance = float(np.linalg.norm(delta))
            if distance <= tolerance:
                break
            heading = math.atan2(delta[1], delta[0])
            heading_error = angle(heading - yaw)
            if abs(heading_error) > 0.25:
                self.turn_in_place(heading)
                continue
            speed_cap = min(
                max_speed,
                self.brake_cap(distance - tolerance, BASE_ACCEL),
            )
            speed = min(max(0.025, distance), speed_cap)
            yaw_cap = min(
                BASE_MAX_YAW_RATE,
                self.brake_cap(heading_error, BASE_YAW_ACCEL),
            )
            self.set_base_velocity(
                speed,
                float(np.clip(1.5 * heading_error, -yaw_cap, yaw_cap)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure(f"navigation timeout target={target_xy.tolist()}")
        self.stop_base()
        self.turn_in_place(target_yaw)

    def reverse(self, distance, max_speed=0.16):
        start = self.ct.base_pose()
        yaw = float(start[2])
        target = start[:2] - float(distance) * np.array(
            [math.cos(yaw), math.sin(yaw)]
        )
        for _ in range(2500):
            x, y, measured_yaw = self.ct.base_pose()
            remaining = float(np.linalg.norm(target - np.array([x, y])))
            if remaining <= 0.008:
                break
            reverse_heading = angle(
                math.atan2(target[1] - y, target[0] - x) + math.pi
            )
            yaw_error = angle(reverse_heading - measured_yaw)
            speed_cap = min(
                max_speed,
                self.brake_cap(remaining - 0.008, BASE_ACCEL),
            )
            self.set_base_velocity(
                -min(max(0.02, remaining), speed_cap),
                float(np.clip(1.2 * yaw_error, -0.10, 0.10)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure("reverse clearance timeout")
        self.stop_base()

    def solve_mounts(self, positions, rotations, iterations=240, base_pose=None):
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        targets = {}
        residuals = {}
        rotation_residuals = {}
        if base_pose is not None:
            for address, value in zip(self.ct.BQ, base_pose):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
        for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
            for address, value in zip(arm.qadr, self.ct.qtgt[hand]):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
            self.ct.ik(
                arm,
                np.asarray(positions[hand], dtype=float),
                np.asarray(rotations[hand], dtype=float),
                iters=iterations,
                w=0.6,
            )
            targets[hand] = np.array(
                [self.data.qpos[address] for address in arm.qadr],
                dtype=float,
            )
            residuals[hand] = float(np.linalg.norm(
                self.data.xpos[arm.mount] - positions[hand]
            ))
            rotation_residuals[hand] = math.degrees(float(np.linalg.norm(
                self.ct.rot_err(
                    rotations[hand],
                    self.data.xmat[arm.mount].reshape(3, 3),
                )
            )))
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return targets, residuals, rotation_residuals

    def validate_shelf_arm_path(self, targets, label):
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        starts = {
            hand: self.ct.qtgt[hand].copy()
            for hand in ("l", "r")
        }
        distance = max(
            float(np.max(np.abs(targets[hand] - starts[hand])))
            for hand in ("l", "r")
        )
        steps = cosine_steps(
            distance,
            FLAT_REGRASP_COLLISION_STEP_RAD,
            minimum=2,
        )
        collision = None
        try:
            for index in range(steps + 1):
                blend = 0.5 - 0.5 * math.cos(
                    math.pi * index / steps
                )
                for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
                    configuration = (
                        starts[hand]
                        + (targets[hand] - starts[hand]) * blend
                    )
                    for address, value in zip(arm.qadr, configuration):
                        self.data.qpos[address] = value
                self.mujoco.mj_forward(self.model, self.data)
                contacts = self.arm_shelf_contacts()
                if contacts:
                    collision = {
                        "sample": index,
                        "samples": steps,
                        "contacts": contacts,
                    }
                    break
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.mujoco.mj_forward(self.model, self.data)
        if collision is not None:
            raise ExpertFailure(
                f"shelf arm path collision phase={label} "
                f"evidence={collision}"
            )
        print(
            f"[shelf_path:{label}] PASS samples={steps + 1}",
            flush=True,
        )

    def servo_arms(
        self,
        targets,
        max_step=ARM_SERVO_MAX_STEP_RAD,
        minimum=ARM_SERVO_MIN_TICKS,
        shelf_safe=False,
    ):
        distance = max(
            float(np.max(np.abs(targets[hand] - self.ct.qtgt[hand])))
            for hand in ("l", "r")
        )
        effective_max_step = (
            min(max_step, 0.012)
            if shelf_safe
            else max_step
        )
        steps = cosine_steps(
            distance,
            effective_max_step,
            minimum=minimum,
        )
        starts = {hand: self.ct.qtgt[hand].copy() for hand in ("l", "r")}
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(math.pi * (index + 1) / steps)
            for hand in ("l", "r"):
                self.ct.qtgt[hand][:] = (
                    starts[hand] + (targets[hand] - starts[hand]) * blend
                )
            self.ct.base_vel[:] = 0.0
            self.frames(1)
            if shelf_safe:
                self.require_arms_clear_shelf(self.ct.REC["phase"])
        if shelf_safe:
            for _ in range(ARM_SERVO_SETTLE_TICKS):
                self.frames(1)
                self.require_arms_clear_shelf(self.ct.REC["phase"])
        else:
            self.frames(ARM_SERVO_SETTLE_TICKS)

    def wait_arm_tracking(
        self,
        label,
        tolerance=ARM_TRACK_TOL_RAD,
        collision_free_hand=None,
        collision_free_hands=(),
        shelf_safe=False,
    ):
        stable_ticks = 0
        error = float("inf")
        for tick in range(1, ARM_TRACK_MAX_TICKS + 1):
            hands = list(collision_free_hands)
            if collision_free_hand is not None:
                hands.append(collision_free_hand)
            for collision_hand in hands:
                contacts = self.moving_arm_contacts(collision_hand)
                if contacts:
                    raise ExpertFailure(
                        f"moving arm collision phase={label} "
                        f"hand={collision_hand} contacts={contacts}"
                    )
            if shelf_safe:
                self.require_arms_clear_shelf(label)
            error = max(
                abs(float(self.data.qpos[address]) - float(target))
                for hand, arm in (("l", self.ct.L), ("r", self.ct.R))
                for address, target in zip(arm.qadr, self.ct.qtgt[hand])
            )
            stable_ticks = stable_ticks + 1 if error <= tolerance else 0
            if stable_ticks >= ARM_TRACK_STABLE_TICKS:
                print(
                    f"[track:{label}] PASS error={error:.4f}rad ticks={tick}",
                    flush=True,
                )
                return
            self.frames(1)
        raise ExpertFailure(
            f"arm tracking timeout phase={label} error={error:.4f}rad"
        )

    def move_mounts(
        self,
        positions,
        rotations,
        iterations=240,
        solve_base_pose=None,
        tracking_tolerance=ARM_TRACK_TOL_RAD,
        shelf_safe=False,
    ):
        targets, residuals, rotation_residuals = self.solve_mounts(
            positions,
            rotations,
            iterations=iterations,
            base_pose=solve_base_pose,
        )
        self.gate(
            "ik_reachable",
            max(residuals.values()) <= 0.012
            and max(rotation_residuals.values())
            <= IK_ROTATION_TOLERANCE_DEG,
            "residual_mm="
            + ",".join(
                f"{hand}:{1000.0 * residuals[hand]:.1f}"
                for hand in ("l", "r")
            )
            + " rotation_deg="
            + ",".join(
                f"{hand}:{rotation_residuals[hand]:.2f}"
                for hand in ("l", "r")
            ),
        )
        if shelf_safe:
            self.validate_shelf_arm_path(
                targets,
                self.ct.REC["phase"],
            )
        self.servo_arms(targets, shelf_safe=shelf_safe)
        self.wait_arm_tracking(
            self.ct.REC["phase"],
            tolerance=tracking_tolerance,
            shelf_safe=shelf_safe,
        )

    def solve_one_mount_target(
        self, hand, seed, position, rotation, iterations=400
    ):
        arms = {"l": self.ct.L, "r": self.ct.R}
        arm = arms[hand]
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        for address, value in zip(arm.qadr, seed):
            self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        self.ct.ik(
            arm,
            np.asarray(position, dtype=float),
            np.asarray(rotation, dtype=float),
            iters=iterations,
            w=0.6,
        )
        target = np.array(
            [self.data.qpos[address] for address in arm.qadr],
            dtype=float,
        )
        residual = float(np.linalg.norm(
            self.data.xpos[arm.mount] - position
        ))
        rotation_residual = math.degrees(float(np.linalg.norm(
            self.ct.rot_err(
                rotation,
                self.data.xmat[arm.mount].reshape(3, 3),
            )
        )))
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return target, residual, rotation_residual

    def validate_one_arm_segment(self, hand, start, target, label):
        arm = self.ct.L if hand == "l" else self.ct.R
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        distance = float(np.max(np.abs(target - start)))
        steps = cosine_steps(
            distance,
            FLAT_REGRASP_COLLISION_STEP_RAD,
            minimum=2,
        )
        collision = None
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * (index + 1) / steps
            )
            configuration = start + (target - start) * blend
            for address, value in zip(arm.qadr, configuration):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
            contacts = self.moving_arm_contacts(hand)
            if contacts:
                collision = {
                    "sample": index + 1,
                    "samples": steps,
                    "contacts": contacts,
                }
                break
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        if collision is not None:
            raise ExpertFailure(
                f"collision-free path failed phase={label} "
                f"hand={hand} evidence={collision}"
            )

    def servo_one_arm_target(self, hand, target, label):
        start = self.ct.qtgt[hand].copy()
        distance = float(np.max(np.abs(target - start)))
        steps = cosine_steps(
            distance,
            EMPTY_HAND_SERVO_MAX_STEP_RAD,
            minimum=1,
        )
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * (index + 1) / steps
            )
            self.ct.qtgt[hand][:] = start + (target - start) * blend
            self.ct.base_vel[:] = 0.0
            self.frames(1)
            contacts = self.moving_arm_contacts(hand)
            if contacts:
                raise ExpertFailure(
                    f"moving arm collision phase={label} "
                    f"hand={hand} contacts={contacts}"
                )

    def follow_empty_hand_stage(
        self,
        hand,
        stage,
        targets,
        initial_ik_seed=None,
    ):
        max_position_residual = 0.0
        max_rotation_residual = 0.0
        waypoint_count = 0
        for position, rotation in targets:
            waypoint_count += 1
            start = self.ct.qtgt[hand].copy()
            seed = (
                np.asarray(initial_ik_seed, dtype=float)
                if waypoint_count == 1 and initial_ik_seed is not None
                else start
            )
            target, residual, rotation_residual = (
                self.solve_one_mount_target(
                    hand,
                    seed,
                    position,
                    rotation,
                )
            )
            if (
                residual > 0.012
                or rotation_residual > IK_ROTATION_TOLERANCE_DEG
            ):
                self.gate(
                    f"flat_regrasp_path_{hand}",
                    False,
                    f"stage={stage} residual_mm={1000.0 * residual:.1f} "
                    f"rotation_deg={rotation_residual:.2f}",
                )
            self.validate_one_arm_segment(
                hand,
                start,
                target,
                f"{hand}_{stage}",
            )
            self.servo_one_arm_target(
                hand,
                target,
                f"{hand}_{stage}",
            )
            max_position_residual = max(max_position_residual, residual)
            max_rotation_residual = max(
                max_rotation_residual, rotation_residual
            )
        self.wait_arm_tracking(
            f"{hand}_{stage}",
            collision_free_hand=hand,
        )
        contacts = self.moving_arm_contacts(hand)
        self.gate(
            f"collision_free_{hand}_{stage}",
            not contacts,
            f"waypoints={waypoint_count} "
            f"max_residual_mm={1000.0 * max_position_residual:.2f} "
            f"max_rotation_deg={max_rotation_residual:.3f} "
            f"contacts={contacts}",
        )

    def bimanual_contacts(self):
        return {
            hand: contacts
            for hand in ("l", "r")
            if (contacts := self.moving_arm_contacts(hand))
        }

    def flat_pick_pad_positions_in_base(self):
        base = self.ct.base_pose()
        origin = np.array([base[0], base[1], 0.0])
        world_to_base = self.ct.base_rotz().T
        return {
            "l": world_to_base @ (self.ct.L.padmid() - origin),
            "r": world_to_base @ (self.ct.R.padmid() - origin),
        }

    def flat_pick_pad_table_clearances(self):
        return {
            hand: self.minimum_geom_clearance(
                {
                    self.ct.gid(f"{hand.upper()}_pad1"),
                    self.ct.gid(f"{hand.upper()}_pad2"),
                },
                {self.table_top_geom},
            )
            for hand in ("l", "r")
        }

    def verify_task_ready_arm_park(self):
        targets = {
            hand: np.asarray(values, dtype=float)
            for hand, values in SORTING_ROLL_INITIAL_ARM_PARK.items()
        }
        measured = self.arm_joint_positions()
        target_error = max(
            float(np.max(np.abs(self.ct.qtgt[hand] - targets[hand])))
            for hand in ("l", "r")
        )
        tracking_error = max(
            float(np.max(np.abs(measured[hand] - targets[hand])))
            for hand in ("l", "r")
        )
        pad_positions = self.flat_pick_pad_positions_in_base()
        table_clearances = self.flat_pick_pad_table_clearances()
        minimum_table_clearance_m = min(
            evidence["distance_m"]
            for evidence in table_clearances.values()
        )
        contacts = self.bimanual_contacts()
        self.gate(
            "task_ready_arm_park_held_after_stereo_localization",
            target_error <= 1e-12
            and tracking_error <= 0.03
            and flat_pick_workspace_is_safe(
                pad_positions["l"], pad_positions["r"]
            )
            and minimum_table_clearance_m
            >= FLAT_PICK_PAD_TABLE_CLEARANCE_M
            and not contacts,
            f"target_error_rad={target_error:.6f} "
            f"tracking_error_rad={tracking_error:.4f} "
            f"minimum_table_clearance_mm="
            f"{1000.0 * minimum_table_clearance_m:.3f} "
            f"pad_positions_base_m="
            f"{json.dumps({hand: np.round(position, 6).tolist() for hand, position in pad_positions.items()})} "
            f"contacts={contacts}",
        )

    def follow_coordinated_flat_pick_path(
        self,
        pregrasp_positions,
        rotations,
    ):
        goal_targets = {}
        residuals = {}
        rotation_residuals = {}
        for hand in ("l", "r"):
            target, residual, rotation_residual = (
                self.solve_one_mount_target(
                    hand,
                    np.asarray(
                        FLAT_PICK_GOAL_IK_SEEDS[hand],
                        dtype=float,
                    ),
                    pregrasp_positions[hand],
                    rotations[hand],
                )
            )
            goal_targets[hand] = target
            residuals[hand] = residual
            rotation_residuals[hand] = rotation_residual
        self.gate(
            "flat_pick_goal_ik_reachable",
            max(residuals.values()) <= 0.012
            and max(rotation_residuals.values())
            <= IK_ROTATION_TOLERANCE_DEG,
            "residual_mm="
            + json.dumps({
                hand: round(1000.0 * residuals[hand], 3)
                for hand in ("l", "r")
            })
            + " rotation_deg="
            + json.dumps({
                hand: round(rotation_residuals[hand], 4)
                for hand in ("l", "r")
            }),
        )

        start_joints = self.arm_joint_positions()
        joint_paths = {
            hand: np.asarray([
                start_joints[hand],
                *FLAT_PICK_JOINT_WAYPOINTS[hand],
                goal_targets[hand],
            ], dtype=float)
            for hand in ("l", "r")
        }
        grid_steps = FLAT_PICK_COORDINATION_GRID_STEPS
        grid_configurations = {
            hand: np.asarray([
                joint_polyline_at_progress(
                    joint_paths[hand],
                    index / grid_steps,
                )
                for index in range(grid_steps + 1)
            ])
            for hand in ("l", "r")
        }

        def configuration_for_cell(cell):
            return np.concatenate([
                grid_configurations["l"][cell[0]],
                grid_configurations["r"][cell[1]],
            ])

        def set_arm_configuration(configuration):
            for arm, offset in (
                (self.ct.L, 0),
                (self.ct.R, 7),
            ):
                for address, value in zip(
                    arm.qadr,
                    configuration[offset:offset + 7],
                ):
                    self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        original_roll_radius = float(
            self.model.geom_size[self.roll_geom, 0]
        )
        cells = ()
        collision = None
        edge_checks = 0
        edge_samples = 0
        valid_grid_nodes = 0
        planning_grid_nodes = 0
        execution_steps = 0
        execution = []
        max_command_delta = 0.0
        validated_samples = 0
        workspace_invalid_grid_nodes = 0
        clearance_invalid_grid_nodes = 0
        minimum_planned_clearance_m = math.inf
        workspace_violation = None
        clearance_violation = None

        def edge_is_safe(start_cell, target_cell):
            nonlocal edge_checks, edge_samples
            start = configuration_for_cell(start_cell)
            target = configuration_for_cell(target_cell)
            distance = float(np.max(np.abs(target - start)))
            samples = max(
                2,
                int(math.ceil(
                    distance / FLAT_PICK_COLLISION_STEP_RAD
                )) + 1,
            )
            edge_checks += 1
            for blend in np.linspace(0.0, 1.0, samples)[1:-1]:
                set_arm_configuration(
                    start + blend * (target - start)
                )
                edge_samples += 1
                pad_positions = self.flat_pick_pad_positions_in_base()
                if not flat_pick_workspace_is_safe(
                    pad_positions["l"], pad_positions["r"]
                ):
                    return False
                if self.bimanual_contacts():
                    return False
            return True

        try:
            self.model.geom_size[self.roll_geom, 0] = (
                original_roll_radius
                + FLAT_PICK_ROLL_CLEARANCE_MARGIN_M
            )
            grid_table_clearances = {
                hand: np.full(grid_steps + 1, math.inf)
                for hand in ("l", "r")
            }
            reference = configuration_for_cell((0, 0))
            for hand, offset in (("l", 0), ("r", 7)):
                for index in range(grid_steps + 1):
                    configuration = reference.copy()
                    configuration[offset:offset + 7] = (
                        grid_configurations[hand][index]
                    )
                    set_arm_configuration(configuration)
                    grid_table_clearances[hand][index] = (
                        self.flat_pick_pad_table_clearances()[hand][
                            "distance_m"
                        ]
                    )
            minimum_planned_clearance_m = min(
                float(np.min(clearances))
                for clearances in grid_table_clearances.values()
            )
            validity = np.zeros(
                (grid_steps + 1, grid_steps + 1),
                dtype=bool,
            )
            for left in range(grid_steps + 1):
                for right in range(grid_steps + 1):
                    set_arm_configuration(
                        configuration_for_cell((left, right))
                    )
                    pad_positions = self.flat_pick_pad_positions_in_base()
                    workspace_ok = flat_pick_workspace_is_safe(
                        pad_positions["l"], pad_positions["r"]
                    )
                    if not workspace_ok:
                        workspace_invalid_grid_nodes += 1
                    minimum_clearance_m = min(
                        grid_table_clearances["l"][left],
                        grid_table_clearances["r"][right],
                    )
                    clearance_ok = (
                        minimum_clearance_m
                        >= FLAT_PICK_PAD_TABLE_CLEARANCE_M
                    )
                    if not clearance_ok:
                        clearance_invalid_grid_nodes += 1
                    validity[left, right] = (
                        workspace_ok
                        and clearance_ok
                        and not self.bimanual_contacts()
                    )
            valid_grid_nodes = int(np.count_nonzero(validity))
            planning_validity = coordination_clearance_mask(
                validity,
                FLAT_PICK_COORDINATION_CLEARANCE_CELLS,
            )
            planning_validity[0, 0] = validity[0, 0]
            planning_validity[-1, -1] = validity[-1, -1]
            planning_grid_nodes = int(
                np.count_nonzero(planning_validity)
            )
            cells = monotonic_coordination_indices(
                planning_validity,
                edge_is_safe=edge_is_safe,
            )
            if not cells:
                collision = {"reason": "no_monotonic_coordination_path"}
            else:
                coordinated = np.asarray([
                    configuration_for_cell(cell)
                    for cell in cells
                ])
                segment_lengths = np.max(
                    np.abs(np.diff(coordinated, axis=0)),
                    axis=1,
                )
                total_distance = float(np.sum(segment_lengths))
                execution_steps = cosine_steps(
                    total_distance,
                    EMPTY_HAND_SERVO_MAX_STEP_RAD,
                    minimum=ARM_SERVO_MIN_TICKS,
                )
                execution = [
                    joint_polyline_at_progress(
                        coordinated,
                        0.5 - 0.5 * math.cos(
                            math.pi * (index + 1)
                            / execution_steps
                        ),
                    )
                    for index in range(execution_steps)
                ]
                validation = [coordinated[0], *execution]
                max_command_delta = max(
                    float(np.max(np.abs(target - start)))
                    for start, target in zip(
                        validation,
                        validation[1:],
                    )
                )

                for segment_index, (start, target) in enumerate(
                    zip(validation, validation[1:])
                ):
                    distance = float(np.max(np.abs(target - start)))
                    samples = max(
                        2,
                        int(math.ceil(
                            distance / FLAT_PICK_COLLISION_STEP_RAD
                        )) + 1,
                    )
                    for sample_index, blend in enumerate(
                        np.linspace(0.0, 1.0, samples)
                    ):
                        if segment_index and sample_index == 0:
                            continue
                        set_arm_configuration(
                            start + blend * (target - start)
                        )
                        validated_samples += 1
                        pad_positions = (
                            self.flat_pick_pad_positions_in_base()
                        )
                        if not flat_pick_workspace_is_safe(
                            pad_positions["l"], pad_positions["r"]
                        ):
                            workspace_violation = {
                                "segment": segment_index + 1,
                                "sample": sample_index + 1,
                                "pad_positions_base_m": {
                                    hand: np.round(position, 6).tolist()
                                    for hand, position
                                    in pad_positions.items()
                                },
                            }
                            collision = {
                                "workspace": workspace_violation
                            }
                            break
                        table_clearances = (
                            self.flat_pick_pad_table_clearances()
                        )
                        minimum_clearance_m = min(
                            evidence["distance_m"]
                            for evidence in table_clearances.values()
                        )
                        if (
                            minimum_clearance_m
                            < FLAT_PICK_PAD_TABLE_CLEARANCE_M
                        ):
                            clearance_violation = {
                                "segment": segment_index + 1,
                                "sample": sample_index + 1,
                                "minimum_clearance_mm": round(
                                    1000.0 * minimum_clearance_m, 3
                                ),
                                "clearances": table_clearances,
                            }
                            collision = {
                                "table_clearance": clearance_violation
                            }
                            break
                        contacts = self.bimanual_contacts()
                        if contacts:
                            collision = {
                                "segment": segment_index + 1,
                                "sample": sample_index + 1,
                                "contacts": contacts,
                            }
                            break
                    if collision:
                        break
        finally:
            self.model.geom_size[self.roll_geom, 0] = (
                original_roll_radius
            )
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.mujoco.mj_forward(self.model, self.data)

        max_progress_gap = (
            max(abs(left - right) for left, right in cells)
            / grid_steps
            if cells
            else math.inf
        )
        self.gate(
            "collision_free_coordinated_flat_pick_path",
            bool(cells) and collision is None,
            f"grid_nodes={(grid_steps + 1) ** 2} "
            f"valid_grid_nodes={valid_grid_nodes} "
            f"workspace_invalid_grid_nodes="
            f"{workspace_invalid_grid_nodes} "
            f"clearance_invalid_grid_nodes="
            f"{clearance_invalid_grid_nodes} "
            f"minimum_table_clearance_mm="
            f"{1000.0 * minimum_planned_clearance_m:.3f} "
            f"planning_grid_nodes={planning_grid_nodes} "
            f"path_nodes={len(cells)} "
            f"max_progress_gap={max_progress_gap:.4f} "
            f"edge_checks={edge_checks} "
            f"edge_samples={edge_samples} "
            f"execution_steps={execution_steps} "
            f"validated_samples={validated_samples} "
            f"roll_clearance_margin_mm="
            f"{1000.0 * FLAT_PICK_ROLL_CLEARANCE_MARGIN_M:.1f} "
            f"collision={collision}",
        )

        workspace_trace = {
            hand: [position.copy()]
            for hand, position
            in self.flat_pick_pad_positions_in_base().items()
        }
        table_clearance_trace = {
            hand: [evidence["distance_m"]]
            for hand, evidence
            in self.flat_pick_pad_table_clearances().items()
        }
        for target in execution:
            self.ct.qtgt["l"][:] = target[:7]
            self.ct.qtgt["r"][:] = target[7:]
            self.ct.base_vel[:] = 0.0
            self.frames(1)
            pad_positions = self.flat_pick_pad_positions_in_base()
            for hand in ("l", "r"):
                workspace_trace[hand].append(
                    pad_positions[hand].copy()
                )
            if not flat_pick_workspace_is_safe(
                pad_positions["l"], pad_positions["r"]
            ):
                self.gate(
                    "coordinated_flat_pick_workspace_execution",
                    False,
                    "pad_positions_base_m="
                    + json.dumps({
                        hand: np.round(position, 6).tolist()
                        for hand, position in pad_positions.items()
                    }),
                )
            table_clearances = self.flat_pick_pad_table_clearances()
            for hand in ("l", "r"):
                table_clearance_trace[hand].append(
                    table_clearances[hand]["distance_m"]
                )
            minimum_clearance_m = min(
                evidence["distance_m"]
                for evidence in table_clearances.values()
            )
            if minimum_clearance_m < FLAT_PICK_PAD_TABLE_CLEARANCE_M:
                self.gate(
                    "coordinated_flat_pick_table_clearance_execution",
                    False,
                    f"minimum_clearance_mm="
                    f"{1000.0 * minimum_clearance_m:.3f} "
                    f"clearances={table_clearances}",
                )
            contacts = self.bimanual_contacts()
            if contacts:
                self.gate(
                    "collision_free_coordinated_flat_pick_execution",
                    False,
                    f"contacts={contacts}",
                )
        self.wait_arm_tracking(
            "coordinated_flat_pick_pregrasp",
            collision_free_hands=("l", "r"),
        )
        final_pad_positions = self.flat_pick_pad_positions_in_base()
        final_table_clearances = (
            self.flat_pick_pad_table_clearances()
        )
        for hand in ("l", "r"):
            workspace_trace[hand].append(
                final_pad_positions[hand].copy()
            )
            table_clearance_trace[hand].append(
                final_table_clearances[hand]["distance_m"]
            )
        workspace_arrays = {
            hand: np.asarray(trace, dtype=float)
            for hand, trace in workspace_trace.items()
        }
        pad_path_lengths = {
            hand: float(np.sum(np.linalg.norm(
                np.diff(trace, axis=0), axis=1
            )))
            for hand, trace in workspace_arrays.items()
        }
        pad_backtracks = {
            hand: float(np.sum(np.maximum(
                0.0, -np.diff(trace[:, 0])
            )))
            for hand, trace in workspace_arrays.items()
        }
        minimum_side_margin = min(
            float(np.min(workspace_arrays["l"][:, 1])),
            float(np.min(-workspace_arrays["r"][:, 1])),
        )
        workspace_execution_ok = (
            flat_pick_workspace_is_safe(
                final_pad_positions["l"], final_pad_positions["r"]
            )
            and max(pad_path_lengths.values())
            <= FLAT_PICK_PAD_PATH_MAX_M
            and max(pad_backtracks.values())
            <= FLAT_PICK_PAD_BACKTRACK_MAX_M
        )
        self.gate(
            "coordinated_flat_pick_workspace_execution",
            workspace_execution_ok,
            f"minimum_side_margin_mm="
            f"{1000.0 * minimum_side_margin:.1f} "
            f"pad_path_length_m="
            f"{json.dumps({hand: round(value, 4) for hand, value in pad_path_lengths.items()})} "
            f"pad_backtrack_m="
            f"{json.dumps({hand: round(value, 4) for hand, value in pad_backtracks.items()})}",
        )
        minimum_table_clearance_m = min(
            min(trace) for trace in table_clearance_trace.values()
        )
        self.gate(
            "coordinated_flat_pick_table_clearance_execution",
            minimum_table_clearance_m
            >= FLAT_PICK_PAD_TABLE_CLEARANCE_M,
            f"minimum_clearance_mm="
            f"{1000.0 * minimum_table_clearance_m:.3f} "
            f"required_mm="
            f"{1000.0 * FLAT_PICK_PAD_TABLE_CLEARANCE_M:.1f}",
        )
        self.gate(
            "collision_free_coordinated_flat_pick_execution",
            not self.bimanual_contacts(),
            f"execution_steps={execution_steps} "
            f"max_command_delta_rad={max_command_delta:.6f} "
            f"max_command_speed_rad_s="
            f"{CONTROL_FPS * max_command_delta:.4f}",
        )

    def move_mounts_delta(self, delta, shelf_safe=False):
        delta = np.asarray(delta, dtype=float)
        steps = max(1, int(math.ceil(float(np.linalg.norm(delta)) / 0.02)))
        for _ in range(steps):
            step = delta / steps
            positions = {
                "l": self.data.xpos[self.ct.L.mount].copy() + step,
                "r": self.data.xpos[self.ct.R.mount].copy() + step,
            }
            rotations = {
                "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
                "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
            }
            self.move_mounts(
                positions,
                rotations,
                iterations=300,
                shelf_safe=shelf_safe,
            )

    def commanded_mount_poses(self):
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
            for address, value in zip(arm.qadr, self.ct.qtgt[hand]):
                self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        positions = {
            "l": self.data.xpos[self.ct.L.mount].copy(),
            "r": self.data.xpos[self.ct.R.mount].copy(),
        }
        rotations = {
            "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
            "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
        }
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return positions, rotations

    def move_mount_commands_delta(self, delta, shelf_safe=False):
        delta = np.asarray(delta, dtype=float)
        self.move_mount_command_deltas(
            {"l": delta, "r": delta},
            shelf_safe=shelf_safe,
        )

    def move_mount_command_deltas(self, deltas, shelf_safe=False):
        positions, rotations = self.commanded_mount_poses()
        for hand in ("l", "r"):
            positions[hand] += np.asarray(deltas[hand], dtype=float)
        self.move_mounts(
            positions,
            rotations,
            iterations=300,
            shelf_safe=shelf_safe,
        )

    def require_held(self, stage, minimum_force=GRIP_FORCE_MIN_N):
        recovery_ticks = 0
        while True:
            left = self.grip_evidence("L")
            right = self.grip_evidence("R")
            position = self.roll_position()
            held = (
                left["force_n"] >= minimum_force
                and right["force_n"] >= minimum_force
                and len(left["pads"]) == 2
                and len(right["pads"]) == 2
                and position[2] >= HELD_MIN_ROLL_Z_M
            )
            if held or recovery_ticks >= HOLD_CONTACT_RECOVERY_TICKS:
                break
            self.frames(1)
            recovery_ticks += 1
        self.gate(
            f"held_{stage}",
            held,
            f"position={np.round(position, 4).tolist()} "
            f"left={left['force_n']:.2f}N/{left['pads']} "
            f"right={right['force_n']:.2f}N/{right['pads']} "
            f"recovery_ticks={recovery_ticks}",
        )

    def require_hand_held(self, hand, stage, minimum_force=GRIP_FORCE_MIN_N):
        recovery_ticks = 0
        while True:
            evidence = self.grip_evidence(hand.upper())
            held = (
                evidence["force_n"] >= minimum_force
                and len(evidence["pads"]) == 2
                and self.roll_position()[2] >= 1.055
            )
            if held or recovery_ticks >= HOLD_CONTACT_RECOVERY_TICKS:
                break
            self.frames(1)
            recovery_ticks += 1
        self.gate(
            f"held_{hand}_{stage}",
            held,
            f"position={np.round(self.roll_position(), 4).tolist()} "
            f"force={evidence['force_n']:.2f}N/{evidence['pads']} "
            f"recovery_ticks={recovery_ticks}",
        )

    def require_hand_released(self, hand, stage):
        evidence = self.grip_evidence(hand.upper())
        self.gate(
            f"released_{hand}_{stage}",
            evidence["force_n"] <= 0.05 and not evidence["pads"],
            f"force={evidence['force_n']:.3f}N/{evidence['pads']}",
        )

    def open_hand_until_released(
        self, hand, stage, max_ticks=120, stable_ticks=6
    ):
        arm = self.ct.L if hand == "l" else self.ct.R
        stable = 0
        evidence = None
        open_fraction = 0.0
        for tick in range(1, int(max_ticks) + 1):
            self.frames(1)
            raw = float(np.mean([
                self.data.qpos[address] for address in arm.grip_qadr
            ]))
            span = self.ct.GRIP_OPEN - self.ct.GRIP_CLOSE
            open_fraction = float(np.clip(
                (raw - self.ct.GRIP_CLOSE) / span,
                0.0,
                1.0,
            ))
            evidence = self.grip_evidence(hand.upper())
            released = (
                open_fraction >= 0.95
                and evidence["force_n"] <= 0.05
                and not evidence["pads"]
            )
            stable = stable + 1 if released else 0
            if stable >= int(stable_ticks):
                self.gate(
                    f"released_{hand}_{stage}",
                    True,
                    f"open_fraction={open_fraction:.3f} "
                    f"force={evidence['force_n']:.3f}N/{evidence['pads']} "
                    f"ticks={tick} stable_ticks={stable}",
                )
                return
        self.gate(
            f"released_{hand}_{stage}",
            False,
            f"open_fraction={open_fraction:.3f} "
            f"force={evidence['force_n']:.3f}N/{evidence['pads']} "
            f"ticks={max_ticks} stable_ticks={stable}",
        )

    def flatten_hands(self):
        starting_height = float(self.roll_position()[2])
        arms = {"l": self.ct.L, "r": self.ct.R}
        for hand, support, direction in FLAT_REGRASP_ORDER:
            self.phase(f"release_{hand}_for_flat_regrasp")
            self.require_hand_held(support, f"supporting_{hand}_release")
            self.ct.grip_cmd[hand] = self.ct.GRIP_OPEN
            self.open_hand_until_released(hand, "before_flattening")
            self.require_hand_held(support, f"supporting_{hand}_flattening")
            arm = arms[hand]
            measured = self.arm_joint_positions()[hand]
            self.ct.qtgt[hand][:] = measured
            self.mujoco.mj_forward(self.model, self.data)

            def live_axis(reference=None):
                axis = (
                    self.data.xmat[self.roll_body]
                    .reshape(3, 3)[:, 0]
                    .copy()
                )
                if reference is None:
                    if axis[0] < 0.0:
                        axis = -axis
                elif float(np.dot(axis, reference)) < 0.0:
                    axis = -axis
                return axis / float(np.linalg.norm(axis))

            def verify_empty_stage(stage):
                self.require_hand_released(hand, f"after_{stage}")
                self.require_hand_held(
                    support,
                    f"supporting_{hand}_{stage}",
                )
                self.gate(
                    f"one_hand_support_height_{hand}_{stage}",
                    self.roll_position()[2]
                    >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                    f"starting_z={starting_height:.4f} "
                    f"actual_z={self.roll_position()[2]:.4f}",
                )

            slide_anchor = arm.padmid().copy()
            slide_mount = self.data.xpos[arm.mount].copy()
            slide_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            slide_roll = self.roll_position()
            slide_axis = live_axis()
            slide_along = float(np.dot(
                slide_anchor - slide_roll,
                slide_axis,
            ))
            slide_target_along = (
                float(direction) * FLAT_REGRASP_COUPLED_START_M
            )
            slide_radial = (
                slide_anchor
                - slide_roll
                - slide_along * slide_axis
            )
            slide_steps = max(1, int(math.ceil(
                abs(slide_target_along - slide_along)
                / FLAT_REGRASP_CART_STEP_M
            )))
            slide_alongs = np.linspace(
                slide_along,
                slide_target_along,
                slide_steps + 1,
            )[1:]

            def adaptive_slide_targets():
                for along in slide_alongs:
                    axis = live_axis(slide_axis)
                    radial = (
                        slide_radial
                        - np.dot(slide_radial, axis) * axis
                    )
                    anchor = self.roll_position() + along * axis + radial
                    yield (
                        anchored_mount_position(
                            slide_mount,
                            slide_rotation,
                            slide_anchor,
                            slide_rotation,
                            target_anchor_position=anchor,
                        ),
                        slide_rotation,
                    )

            self.phase(f"flatten_{hand}_slide_out")
            self.follow_empty_hand_stage(
                hand,
                "slide_out",
                adaptive_slide_targets(),
            )
            verify_empty_stage("slide_out")

            coupled_anchor = arm.padmid().copy()
            coupled_mount = self.data.xpos[arm.mount].copy()
            coupled_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            reference_axis = live_axis(slide_axis)
            coupled_roll = self.roll_position()
            coupled_along = float(np.dot(
                coupled_anchor - coupled_roll,
                reference_axis,
            ))
            coupled_target_along = (
                float(direction) * FLAT_REGRASP_NEAR_END_M
            )
            coupled_radial = (
                coupled_anchor
                - coupled_roll
                - coupled_along * reference_axis
            )
            coupled_steps = max(
                FLAT_REGRASP_COUPLED_MIN_STEPS,
                int(math.ceil(
                    abs(coupled_target_along - coupled_along)
                    / FLAT_REGRASP_CART_STEP_M
                )),
            )
            coupled_alongs = np.linspace(
                coupled_along,
                coupled_target_along,
                coupled_steps + 1,
            )[1:]

            def adaptive_coupled_targets():
                for index, along in enumerate(coupled_alongs, start=1):
                    progress = index / coupled_steps
                    rotation_progress, clearance_progress = (
                        coupled_regrasp_progress(progress)
                    )
                    axis = live_axis(reference_axis)
                    radial = (
                        coupled_radial
                        - np.dot(coupled_radial, axis) * axis
                    )
                    anchor = (
                        self.roll_position()
                        + along * axis
                        + radial
                        + clearance_progress * FLAT_REGRASP_CLEARANCE
                    )
                    rotation = flatten_target_rotation(
                        coupled_rotation,
                        rotation_progress,
                    )
                    yield (
                        anchored_mount_position(
                            coupled_mount,
                            coupled_rotation,
                            coupled_anchor,
                            rotation,
                            target_anchor_position=anchor,
                        ),
                        rotation,
                    )

            self.phase(f"flatten_{hand}_coupled_exit_and_flatten")
            self.follow_empty_hand_stage(
                hand,
                "coupled_exit_and_flatten",
                adaptive_coupled_targets(),
            )
            verify_empty_stage("coupled_exit_and_flatten")

            current_anchor = arm.padmid().copy()
            mount_position = self.data.xpos[arm.mount].copy()
            initial_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            roll_position = self.roll_position()
            reference_axis = live_axis(reference_axis)
            anchors = flat_regrasp_anchors(
                roll_position,
                reference_axis,
                current_anchor,
                direction,
            )
            target_rotation = flatten_target_rotation(
                grasp_target_rotation(self.ct.R_DES, direction),
                1.0,
            )

            def mount_targets(anchor_targets, rotations):
                return [
                    (
                        anchored_mount_position(
                            mount_position,
                            initial_rotation,
                            current_anchor,
                            rotation,
                            target_anchor_position=anchor,
                        ),
                        rotation,
                    )
                    for anchor, rotation in zip(anchor_targets, rotations)
                ]

            stages = []
            rotation_vector = self.ct.rot_err(
                target_rotation,
                initial_rotation,
            )
            rotation_angle = float(np.linalg.norm(rotation_vector))
            if rotation_angle > 1e-9:
                rotation_axis = rotation_vector / rotation_angle
                rotation_steps = max(1, int(math.ceil(
                    math.degrees(rotation_angle)
                    / FLAT_REGRASP_ABSOLUTE_ROTATION_STEP_DEG
                )))
                rotations = [
                    rotation_axis_angle(
                        rotation_axis,
                        rotation_angle * index / rotation_steps,
                    )
                    @ initial_rotation
                    for index in range(1, rotation_steps + 1)
                ]
                stages.append((
                    "finish_flat_rotation",
                    mount_targets(
                        [current_anchor] * len(rotations),
                        rotations,
                    ),
                ))
            for name, start, target in (
                (
                    "extend_flat",
                    current_anchor,
                    anchors["far_end"],
                ),
                (
                    "align_far",
                    anchors["far_end"],
                    anchors["axis_far"],
                ),
            ):
                anchor_targets = cartesian_waypoints(
                    start,
                    target,
                    FLAT_REGRASP_CART_STEP_M,
                )
                stages.append((
                    name,
                    mount_targets(
                        anchor_targets,
                        [target_rotation] * len(anchor_targets),
                    ),
                ))

            for stage, targets in stages:
                self.phase(f"flatten_{hand}_{stage}")
                self.follow_empty_hand_stage(hand, stage, targets)
                verify_empty_stage(stage)

            insert_start = float(direction) * FLAT_REGRASP_FAR_END_M
            insert_target = (
                float(direction) * FLAT_REGRASP_TARGET_ALONG_M
            )
            insert_steps = max(1, int(math.ceil(
                abs(insert_target - insert_start)
                / FLAT_REGRASP_CART_STEP_M
            )))
            insert_alongs = np.linspace(
                insert_start,
                insert_target,
                insert_steps + 1,
            )

            def adaptive_insert_targets():
                for along in insert_alongs:
                    axis = live_axis(reference_axis)
                    anchor = self.roll_position() + along * axis
                    yield (
                        anchored_mount_position(
                            mount_position,
                            initial_rotation,
                            current_anchor,
                            target_rotation,
                            target_anchor_position=anchor,
                        ),
                        target_rotation,
                    )

            self.phase(f"flatten_{hand}_insert_from_end")
            self.follow_empty_hand_stage(
                hand,
                "insert_from_end",
                adaptive_insert_targets(),
            )
            verify_empty_stage("insert_from_end")

            for attempt in range(
                1, FLAT_REGRASP_ANCHOR_CORRECTION_ATTEMPTS + 1
            ):
                axis = live_axis(reference_axis)
                target_anchor = (
                    self.roll_position()
                    + float(direction)
                    * FLAT_REGRASP_TARGET_ALONG_M
                    * axis
                )
                actual_anchor = arm.padmid().copy()
                anchor_error = target_anchor - actual_anchor
                if (
                    float(np.linalg.norm(anchor_error))
                    <= FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M
                ):
                    break

                command_positions, command_rotations = (
                    self.commanded_mount_poses()
                )
                command_mount = command_positions[hand]
                command_rotation = command_rotations[hand]

                correction_position = anchor_feedback_mount_position(
                    command_mount,
                    target_anchor,
                    actual_anchor,
                    FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M,
                )
                correction_stage = f"anchor_correction_{attempt}"
                self.phase(f"flatten_{hand}_{correction_stage}")
                self.follow_empty_hand_stage(
                    hand,
                    correction_stage,
                    [(correction_position, command_rotation)],
                )
                verify_empty_stage(correction_stage)

            axis = (
                self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
            )
            if float(np.dot(axis, reference_axis)) < 0.0:
                axis = -axis
            target_anchor = (
                self.roll_position()
                + float(direction) * FLAT_REGRASP_TARGET_ALONG_M * axis
            )
            anchor_error = target_anchor - arm.padmid()
            self.gate(
                f"flat_regrasp_anchor_{hand}",
                float(np.linalg.norm(anchor_error))
                <= FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M,
                f"error_mm={np.round(1000.0 * anchor_error, 1).tolist()} "
                f"roll={np.round(self.roll_position(), 4).tolist()}",
            )
            self.gate(
                f"one_hand_support_height_before_{hand}_regrasp",
                self.roll_position()[2]
                >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                f"starting_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

            self.phase(f"regrasp_{hand}_flat")
            self.ct.grip_cmd[hand] = self.ct.GRIP_CLOSE
            self.open_hand_until_released(
                hand,
                "before_release_tip_shift",
                max_ticks=60,
            )
            self.require_held(f"{hand}_flat_regrasp")
            self.gate(
                f"one_hand_support_height_{hand}",
                self.roll_position()[2]
                >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                f"starting_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

            self.phase(f"restore_height_after_{hand}_flat_regrasp")
            for _ in range(FLAT_REGRASP_HEIGHT_RESTORE_ATTEMPTS):
                height_error = starting_height - float(
                    self.roll_position()[2]
                )
                if (
                    height_error
                    <= FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M
                ):
                    break
                self.move_mount_commands_delta([
                    0.0,
                    0.0,
                    min(
                        FLAT_REGRASP_HEIGHT_RESTORE_MAX_STEP_M,
                        height_error + 0.002,
                    ),
                ])
                self.require_held(f"restoring_height_after_{hand}")
            self.gate(
                f"flat_regrasp_height_restored_{hand}",
                self.roll_position()[2]
                >= starting_height
                - FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M,
                f"target_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

        self.phase("level_roll_after_flat_regrasp")
        level_axis = None
        for _ in range(FLAT_REGRASP_LEVEL_ATTEMPTS):
            level_axis = (
                self.data.xmat[self.roll_body]
                .reshape(3, 3)[:, 0]
                .copy()
            )
            left_anchor = self.ct.L.padmid().copy()
            right_anchor = self.ct.R.padmid().copy()
            if float(np.dot(
                level_axis,
                left_anchor - right_anchor,
            )) < 0.0:
                level_axis = -level_axis
            if (
                abs(float(level_axis[2]))
                <= FLAT_REGRASP_LEVEL_TARGET_AXIS_Z
            ):
                break
            left_delta_z = symmetric_level_correction(
                level_axis,
                left_anchor,
                right_anchor,
                FLAT_REGRASP_LEVEL_MAX_STEP_M,
            )
            self.move_mount_command_deltas({
                "l": np.array([0.0, 0.0, left_delta_z]),
                "r": np.array([0.0, 0.0, -left_delta_z]),
            })
            self.require_held("leveling_flat_roll")
        level_axis = (
            self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        )
        self.gate(
            "flat_roll_levelled",
            abs(float(level_axis[2]))
            <= FLAT_REGRASP_LEVEL_TARGET_AXIS_Z,
            f"axis={np.round(level_axis, 6).tolist()}",
        )

        spans = self.pad_vertical_span()
        axis = self.roll_axis()
        self.gate(
            "hands_flat",
            max(spans.values()) <= 0.025
            and abs(float(axis[2])) <= 0.02
            and self.roll_position()[2] >= HAND_FLAT_ROLL_Z - 0.015,
            f"pad_vertical_span_mm="
            f"{json.dumps({name: round(1000.0 * value, 1) for name, value in spans.items()})} "
            f"roll_axis={np.round(axis, 6).tolist()} "
            f"roll={np.round(self.roll_position(), 4).tolist()}",
        )

    def align_roll_center(
        self,
        target,
        gate_name,
        tolerance,
        *,
        command_bias=None,
        command_space=False,
        max_step=0.02,
        attempts=12,
        shelf_safe=False,
        require_held_stage=None,
    ):
        target = np.asarray(target, dtype=float)
        tolerance = np.asarray(tolerance, dtype=float)
        command_target = target.copy()
        if command_bias is not None:
            command_target += np.asarray(command_bias, dtype=float)
        for attempt in range(attempts):
            error = target - self.roll_position()
            if np.all(np.abs(error) <= tolerance):
                break
            command_error = command_target - self.roll_position()
            move_delta = (
                self.move_mount_commands_delta
                if command_space
                else self.move_mounts_delta
            )
            move_delta(
                bounded_vector(command_error, max_step),
                shelf_safe=shelf_safe,
            )
            if require_held_stage is not None:
                self.require_held(
                    f"{require_held_stage}_{attempt + 1:02d}"
                )
        error = target - self.roll_position()
        self.gate(
            gate_name,
            np.all(np.abs(error) <= tolerance),
            f"target={np.round(target, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * error, 1).tolist()}",
        )

    def roll_axis(self):
        axis = self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        return -axis if axis[1] < 0.0 else axis

    def current_integrated_depth_margin(self):
        radius = float(self.model.geom_size[self.roll_geom, 0])
        half_length = float(self.model.geom_size[self.roll_geom, 1])
        return integrated_depth_margin(
            self.roll_position()[0],
            self.roll_axis()[0],
            half_length,
            radius,
        )

    def align_roll_axis(self, held_stage, gate_name):
        for _ in range(5):
            axis = self.roll_axis()
            if abs(float(axis[0])) <= 0.0008:
                break
            yaw_correction = math.atan2(float(axis[0]), float(axis[1]))
            target_yaw = angle(self.ct.base_pose()[2] + yaw_correction)
            self.turn_in_place(
                target_yaw, max_rate=0.20, tolerance=0.0002
            )
            self.require_held(held_stage)
        axis = self.roll_axis()
        self.gate(
            gate_name,
            abs(float(axis[0])) <= 0.0008,
            f"axis={np.round(axis, 6).tolist()}",
        )

    def align_roll_axis_with_arms(
        self,
        held_stage,
        gate_name,
        max_step=ENTRY_AXIS_ARM_MAX_STEP_M,
        shelf_safe=False,
    ):
        for _ in range(ENTRY_AXIS_ARM_ATTEMPTS):
            axis = self.roll_axis()
            if (
                abs(float(axis[0])) <= 0.0008
                and abs(float(axis[2])) <= 0.02
            ):
                break
            left_delta = symmetric_axis_correction(
                axis,
                self.ct.L.padmid(),
                self.ct.R.padmid(),
                TARGET_AXIS,
                max_step,
            )
            self.move_mount_command_deltas(
                {"l": left_delta, "r": -left_delta},
                shelf_safe=shelf_safe,
            )
            self.require_held(held_stage)
        axis = self.roll_axis()
        self.gate(
            gate_name,
            abs(float(axis[0])) <= 0.0008
            and abs(float(axis[2])) <= 0.02,
            f"axis={np.round(axis, 6).tolist()}",
        )

    def slowly_enter_integrated_top_tier(
        self,
        target,
        max_step=0.004,
    ):
        target = np.asarray(target, dtype=float)
        if max_step <= 0.0:
            raise ValueError("max_step must be positive")
        tolerance = np.array([
            0.0006,
            PRE_RELEASE_Y_TOLERANCE_M,
            0.002,
        ])
        axis = self.roll_axis()
        radius = float(self.model.geom_size[self.roll_geom, 0])
        half_length = float(self.model.geom_size[self.roll_geom, 1])
        roll_bottom_z = (
            float(self.roll_position()[2])
            - roll_half_extent_x(
                float(axis[2]),
                half_length,
                radius,
            )
        )
        lip_clearance = roll_bottom_z - TOP_TIER_FRONT_LIP_PEAK_Z_M
        self.gate(
            "front_lip_clearance_before_entry",
            lip_clearance >= 0.025,
            f"roll_bottom_z={roll_bottom_z:.6f} "
            f"lip_peak_z={TOP_TIER_FRONT_LIP_PEAK_Z_M:.6f} "
            f"clearance_mm={1000.0 * lip_clearance:.2f}",
        )
        command_target = target + np.array([
            ENTRY_X_COMMAND_BIAS,
            0.0,
            0.0,
        ])
        steps_taken = 0
        max_steps = max(
            12,
            int(math.ceil(
                float(np.linalg.norm(
                    command_target - self.roll_position()
                )) / max_step
            )) + 4,
        )
        for _ in range(max_steps):
            error = target - self.roll_position()
            if np.all(np.abs(error) <= tolerance):
                break
            self.move_mount_commands_delta(
                bounded_vector(
                    command_target - self.roll_position(),
                    max_step,
                ),
                shelf_safe=True,
            )
            steps_taken += 1
            self.require_held("integrated_top_tier_entry")
            axis = self.roll_axis()
            if not insertion_axis_is_safe(axis):
                roll_clearance = self.minimum_geom_clearance(
                    {self.roll_geom}, self.integrated_support_geom_ids
                )
                pad_clearance = self.minimum_geom_clearance(
                    self.pad_ids, self.shelf_geom_ids
                )
                correction_has_clearance = (
                    insertion_axis_correction_has_clearance(
                        roll_clearance["distance_m"],
                        pad_clearance["distance_m"],
                    )
                )
                self.gate(
                    "integrated_entry_axis_correction_clearance",
                    correction_has_clearance,
                    f"roll_clearance_mm="
                    f"{1000.0 * roll_clearance['distance_m']:.3f} "
                    f"pad_clearance_mm="
                    f"{1000.0 * pad_clearance['distance_m']:.3f}",
                )
                left_delta = symmetric_axis_correction(
                    axis,
                    self.ct.L.padmid(),
                    self.ct.R.padmid(),
                    TARGET_AXIS,
                    INSERT_AXIS_CORRECTION_MAX_STEP_M,
                )
                self.move_mount_command_deltas(
                    {"l": left_delta, "r": -left_delta},
                    shelf_safe=True,
                )
                self.require_held("correcting_integrated_entry_axis")
                axis = self.roll_axis()
            self.gate(
                "axis_safe_during_integrated_entry",
                insertion_axis_is_safe(axis),
                f"axis={np.round(axis, 6).tolist()}",
            )
        error = target - self.roll_position()
        self.gate(
            "integrated_entry_alignment",
            np.all(np.abs(error) <= tolerance),
            f"steps={steps_taken} "
            f"target={np.round(target, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * error, 1).tolist()}",
        )
        depth_margin = self.current_integrated_depth_margin()
        self.gate(
            "inside_integrated_top_tier_depth",
            depth_margin >= 0.005,
            f"depth_margin_mm={1000.0 * depth_margin:.2f} "
            f"roll={np.round(self.roll_position(), 6).tolist()}",
        )

    def level_release_support_surfaces(self):
        axis = self.roll_axis()
        turn = rotation_axis_angle(
            axis,
            math.radians(RELEASE_WRIST_LEVEL_DEG),
        )
        positions = {}
        rotations = {}
        for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
            mount_position = self.data.xpos[arm.mount].copy()
            mount_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            anchor = arm.padmid().copy()
            target_rotation = turn @ mount_rotation
            positions[hand] = anchored_mount_position(
                mount_position,
                mount_rotation,
                anchor,
                target_rotation,
            )
            rotations[hand] = target_rotation
        self.move_mounts(
            positions,
            rotations,
            iterations=400,
            shelf_safe=True,
        )
        self.require_held("levelling_release_support_surfaces")
        spans = self.pad_vertical_span()
        self.gate(
            "release_support_surfaces_levelled",
            max(spans.values()) <= 0.025,
            "pad_vertical_span_mm="
            + json.dumps({
                name: round(1000.0 * span, 1)
                for name, span in spans.items()
            }),
        )

    def release_into_integrated_top_tier(self):
        for geom in self.pad_ids:
            self.model.geom_friction[geom, 0] = (
                RELEASE_PAD_SLIDING_FRICTION
            )
        print(
            "[release_setup] pad_sliding_friction="
            f"{RELEASE_PAD_SLIDING_FRICTION:.3f}",
            flush=True,
        )
        self.frames(RELEASE_FRICTION_SETTLE_TICKS)
        self.require_held("guarded_release_friction_transition")

        before_release = self.evaluate_placement(self.model, self.data)
        depth_margin = self.current_integrated_depth_margin()
        pad_contact = self.contact_evidence(
            self.pad_ids, self.shelf_geom_ids
        )
        arm_shelf_contacts = self.arm_shelf_contacts()
        roll_clearance = self.minimum_geom_clearance(
            {self.roll_geom}, self.integrated_support_geom_ids
        )
        pad_clearance = self.minimum_geom_clearance(
            self.pad_ids, self.shelf_geom_ids
        )
        endpoint_margins = before_release["endpoint_margin_m"]
        self.gates["guarded_release_evidence"] = {
            "placement": before_release,
            "roll_support_clearance": roll_clearance,
            "pad_shelf_clearance": pad_clearance,
            "arm_shelf_contacts": arm_shelf_contacts,
        }
        self.gate(
            "guarded_release_ready",
            guarded_release_geometry_is_ready(
                endpoint_margins,
                before_release["axis_error_deg"],
                depth_margin,
            )
            and depth_margin - RELEASE_OPEN_INITIAL_BACKOFF_M >= 0.005
            and (
                roll_clearance["distance_m"]
                + RELEASE_OPEN_CLEARANCE_LIFT_MAX_M
                <= RELEASE_DROP_MAX_M
            )
            and not arm_shelf_contacts
            and pad_contact["force_n"] <= 0.2
            and guarded_release_is_ready(
                roll_clearance["distance_m"],
                pad_clearance["distance_m"],
            ),
            f"center={before_release['center_m']} "
            f"axis_error_deg={before_release['axis_error_deg']} "
            f"endpoint_margins={endpoint_margins} "
            f"depth_margin_mm={1000.0 * depth_margin:.2f} "
            f"bounded_backoff_depth_margin_mm="
            f"{1000.0 * (depth_margin - RELEASE_OPEN_INITIAL_BACKOFF_M):.2f} "
            f"bounded_open_drop_mm="
            f"{1000.0 * (roll_clearance['distance_m'] + RELEASE_OPEN_CLEARANCE_LIFT_MAX_M):.3f} "
            f"trough_gap_mm="
            f"{1000.0 * before_release['roll_bottom_to_trough_gap_m']:.3f} "
            f"roll_support_clearance_mm="
            f"{1000.0 * roll_clearance['distance_m']:.3f} "
            f"roll_pair={roll_clearance['pair']} "
            f"pad_shelf_clearance_mm="
            f"{1000.0 * pad_clearance['distance_m']:.3f} "
            f"pad_pair={pad_clearance['pair']} "
            f"pad_raw_mm="
            f"{1000.0 * pad_clearance['raw_distance_m']:.3f} "
            f"pad_witness_mm="
            f"{1000.0 * pad_clearance['witness_distance_m']:.3f} "
            f"pad_active_contact={pad_clearance['active_contact']} "
            f"pad_shelf_force_n={pad_contact['force_n']:.4f}",
        )

        start_mount_z = float(np.mean([
            self.data.xpos[self.ct.L.mount, 2],
            self.data.xpos[self.ct.R.mount, 2],
        ]))
        self.ct.GRIP_RATE = RELEASE_GRIP_RATE
        self.ct.grip_cmd["l"] = self.ct.GRIP_OPEN
        self.ct.grip_cmd["r"] = self.ct.GRIP_OPEN
        self.move_mount_commands_delta(
            [
                -RELEASE_OPEN_INITIAL_BACKOFF_M,
                0.0,
                RELEASE_OPEN_CLEARANCE_LIFT_MAX_M,
            ],
            shelf_safe=True,
        )
        max_steps = int(round(
            (
                RELEASE_OPEN_BACKOFF_MAX_M
                - RELEASE_OPEN_INITIAL_BACKOFF_M
            )
            / RELEASE_OPEN_BACKOFF_STEP_M
        ))
        release_steps = 0
        release_clear_confirmed = False
        for _ in range(RELEASE_OPEN_FINAL_SETTLE_TICKS):
            self.frames(1)
            self.require_arms_clear_shelf(
                "guarded_release_initial_clearance_settle"
            )
        left = self.grip_evidence("L")
        right = self.grip_evidence("R")
        for candidate_step in range(1, max_steps + 1):
            if release_is_clear(left, right):
                for _ in range(RELEASE_OPEN_FINAL_SETTLE_TICKS):
                    self.frames(1)
                    self.require_arms_clear_shelf(
                        "guarded_release_clear_confirmation"
                    )
                left = self.grip_evidence("L")
                right = self.grip_evidence("R")
                if release_is_clear(left, right):
                    release_clear_confirmed = True
                    break
            current_depth_margin = self.current_integrated_depth_margin()
            self.gate(
                "release_backoff_next_step_margin",
                current_depth_margin
                >= 0.005 + RELEASE_OPEN_BACKOFF_STEP_M,
                f"step={candidate_step}/{max_steps} "
                f"depth_margin_mm={1000.0 * current_depth_margin:.2f}",
            )
            self.move_mount_commands_delta(
                [-RELEASE_OPEN_BACKOFF_STEP_M, 0.0, 0.0],
                shelf_safe=True,
            )
            release_steps = candidate_step
            current_depth_margin = self.current_integrated_depth_margin()
            self.gate(
                "release_backoff_depth_margin",
                current_depth_margin >= 0.005,
                f"step={candidate_step}/{max_steps} "
                f"depth_margin_mm={1000.0 * current_depth_margin:.2f}",
            )
            left = self.grip_evidence("L")
            right = self.grip_evidence("R")
        if not release_clear_confirmed:
            for _ in range(RELEASE_OPEN_FINAL_SETTLE_TICKS):
                self.frames(1)
                self.require_arms_clear_shelf(
                    "guarded_release_final_settle"
                )

        left = self.grip_evidence("L")
        right = self.grip_evidence("R")
        actual_backoff = (
            RELEASE_OPEN_INITIAL_BACKOFF_M
            + release_steps * RELEASE_OPEN_BACKOFF_STEP_M
        )
        actual_open_lift = RELEASE_OPEN_CLEARANCE_LIFT_MAX_M
        self.gates["release_contact_geometry"] = {
            "steps": release_steps,
            "backoff_m": actual_backoff,
            "opening_lift_m": actual_open_lift,
            "after_feedback_backoff": self.roll_pad_contact_geometry(),
            "left": left,
            "right": right,
        }
        self.gate(
            "released_during_guarded_backoff",
            release_is_clear(left, right),
            f"left={left['force_n']:.3f}N/{left['pads']} "
            f"right={right['force_n']:.3f}N/{right['pads']}",
        )
        after_open = self.evaluate_placement(self.model, self.data)
        after_checks = after_open["checks"]
        self.gates["post_open_support_evidence"] = after_open
        self.gate(
            "rod_remains_in_integrated_top_tier_after_open",
            after_checks["center_inside_integrated_top_tier"]
            and after_checks["fully_inside_shelf_width"]
            and after_checks["axis_aligned_with_shelf"]
            and after_checks["supported_by_integrated_top_tier"]
            and after_checks["resting_on_integrated_top_tier_geometry"]
            and after_checks["released_from_both_grippers"],
            json.dumps(after_open, ensure_ascii=False),
        )
        self.move_mount_commands_delta(
            [
                0.0,
                0.0,
                RELEASE_CLEARANCE_LIFT_M
                - actual_open_lift,
            ],
            shelf_safe=True,
        )
        end_mount_z = float(np.mean([
            self.data.xpos[self.ct.L.mount, 2],
            self.data.xpos[self.ct.R.mount, 2],
        ]))
        lift = end_mount_z - start_mount_z
        contacts = self.arm_shelf_contacts()
        self.gate(
            "open_hands_lifted_clear_of_integrated_tier",
            lift >= RELEASE_CLEARANCE_LIFT_M - 0.005
            and not contacts,
            f"lifted_mm={1000.0 * lift:.1f} contacts={contacts}",
        )
        left = self.grip_evidence("L")
        right = self.grip_evidence("R")
        self.gate(
            "released_during_guarded_lift",
            release_is_clear(left, right),
            f"left={left['force_n']:.3f}N/{left['pads']} "
            f"right={right['force_n']:.3f}N/{right['pads']}",
        )
        self.gates["post_open_hand_lift_evidence"] = (
            self.evaluate_placement(self.model, self.data)
        )


    def track_success(
        self,
        ticks=240,
        required_seconds=REQUIRED_STABLE_SECONDS,
    ):
        tracker = self.tracker_cls(required_seconds=required_seconds)
        evidence = None
        for _ in range(int(ticks)):
            dt = self.tick()
            evidence = self.evaluate_placement(self.model, self.data)
            if tracker.update(evidence, dt):
                evidence = dict(evidence)
                evidence["stable_seconds"] = round(
                    tracker.stable_seconds, 4
                )
                return evidence
        return evidence

    def flat_pick_mount_poses(self, roll_position):
        roll_position = np.asarray(roll_position, dtype=float)
        roll_axis = (
            self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        )
        if roll_axis[0] < 0.0:
            roll_axis = -roll_axis
        roll_axis /= float(np.linalg.norm(roll_axis))
        positions = {}
        rotations = {}
        for hand, arm, direction in (
            ("l", self.ct.L, 1.0),
            ("r", self.ct.R, -1.0),
        ):
            mount_position = self.data.xpos[arm.mount].copy()
            mount_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            pad_offset_in_mount = (
                mount_rotation.T @ (arm.padmid() - mount_position)
            )
            target_rotation = flatten_target_rotation(
                grasp_target_rotation(self.ct.R_DES, direction),
                1.0,
            )
            target_pad_position = (
                roll_position
                + direction * FLAT_PICK_TARGET_ALONG_M * roll_axis
                + np.array([0.0, FLAT_PICK_TIP_BIAS_Y_M, 0.0])
            )
            positions[hand] = mount_position_for_pad_target(
                target_pad_position,
                target_rotation,
                pad_offset_in_mount,
            )
            rotations[hand] = target_rotation
        return positions, rotations

    def execute(self):
        park_report = self.args.initial_arm_park_report
        self.gates["sorting_roll_initial_arm_park"] = park_report
        self.gate(
            "sorting_roll_initial_arm_park",
            park_report["passed"],
            json.dumps(park_report, ensure_ascii=False),
        )
        initial_contacts = self.early_unintended_arm_contacts()
        self.gate(
            "initial_arm_park_collision_free",
            not initial_contacts,
            f"contacts={initial_contacts}",
        )

        camera_report = camera_mount_report(
            self.mujoco,
            self.model,
            self.data,
        )
        self.ct.REC["metadata"]["sdk_camera_mount_report"] = camera_report
        self.gates["sdk_camera_mounts"] = camera_report
        print(
            "[gate:sdk_camera_mounts] "
            + ("PASS " if camera_report["passed"] else "FAIL ")
            + json.dumps(camera_report, ensure_ascii=False),
            flush=True,
        )
        if not camera_report["passed"]:
            raise ExpertFailure(
                "recorded camera mounts do not match the SDK contract"
            )

        wrist_camera_report = wrist_camera_initialization_report(
            self.mujoco, self.model, self.data
        )
        self.ct.REC["metadata"][
            "wrist_d405_initialization_report"
        ] = wrist_camera_report
        self.gate(
            "wrist_d405_initialization",
            wrist_camera_report["passed"],
            json.dumps(wrist_camera_report, ensure_ascii=False),
        )

        self.frames(12)

        initial_targets = {
            hand: self.ct.qtgt[hand].copy()
            for hand in ("l", "r")
        }
        initial_joints = self.arm_joint_positions()

        self.phase("navigate_to_table_observation")
        self.go_to(
            TABLE_OBSERVATION_XY,
            -math.pi / 2.0,
            max_speed=0.26,
        )
        target_motion = max(
            float(np.max(np.abs(self.ct.qtgt[hand] - initial_targets[hand])))
            for hand in ("l", "r")
        )
        measured_joints = self.arm_joint_positions()
        measured_motion = max(
            float(np.max(np.abs(
                measured_joints[hand] - initial_joints[hand]
            )))
            for hand in ("l", "r")
        )
        self.gate(
            "arms_unchanged_before_observation",
            target_motion <= 1e-12 and measured_motion <= 0.03,
            f"target_motion_rad={target_motion:.6f} "
            f"measured_motion_rad={measured_motion:.4f}",
        )
        base = self.ct.base_pose()
        self.gate(
            "table_observation_park",
            float(np.linalg.norm(
                base[:2] - TABLE_OBSERVATION_XY
            )) <= 0.02
            and abs(angle(base[2] + math.pi / 2.0)) <= 0.02,
            f"base={np.round(base, 4).tolist()}",
        )

        self.phase("localize_roll_with_head_stereo")
        self.frames(30)

        self.phase(
            "confirm_task_ready_arm_park_after_stereo_localization"
        )
        self.verify_task_ready_arm_park()
        staged_targets = {
            hand: self.ct.qtgt[hand].copy()
            for hand in ("l", "r")
        }
        staged_joints = self.arm_joint_positions()

        self.phase("approach_table_with_arms_staged")
        self.go_to(
            TABLE_GRASP_XY,
            -math.pi / 2.0,
            max_speed=0.25,
            tolerance=0.003,
        )
        self.turn_in_place(
            -math.pi / 2.0,
            tolerance=0.003,
        )
        base = self.ct.base_pose()
        self.gate(
            "table_park",
            float(np.linalg.norm(base[:2] - TABLE_GRASP_XY)) <= 0.006
            and abs(angle(base[2] + math.pi / 2.0)) <= 0.004,
            f"base={np.round(base, 4).tolist()}",
        )
        measured_joints = self.arm_joint_positions()
        target_motion = max(
            float(np.max(np.abs(
                self.ct.qtgt[hand] - staged_targets[hand]
            )))
            for hand in ("l", "r")
        )
        measured_motion = max(
            float(np.max(np.abs(
                measured_joints[hand] - staged_joints[hand]
            )))
            for hand in ("l", "r")
        )
        self.gate(
            "staged_arms_unchanged_through_table_approach",
            target_motion <= 1e-12 and measured_motion <= 0.03,
            f"target_motion_rad={target_motion:.6f} "
            f"measured_motion_rad={measured_motion:.4f}",
        )
        self.gate(
            "collision_free_initial_observation_and_table_approach",
            not self.early_collision_events,
            f"checks={self.early_collision_checks} "
            f"events={self.early_collision_events}",
        )

        roll = self.roll_position()
        grasp_positions, rotations = self.flat_pick_mount_poses(roll)
        pregrasp_positions = {
            hand: position
            + np.array([0.0, FLAT_PICK_PREGRASP_CLEARANCE_Y_M, 0.0])
            for hand, position in grasp_positions.items()
        }
        self.phase(
            "coordinated_flat_pick_pregrasp_after_stereo_localization"
        )
        self.follow_coordinated_flat_pick_path(
            pregrasp_positions,
            rotations,
        )
        spans = self.pad_vertical_span()
        self.gate(
            "hands_flat_before_pick",
            max(spans.values()) <= 0.025,
            "pad_vertical_span_mm="
            + json.dumps({
                name: round(1000.0 * span, 1)
                for name, span in spans.items()
            }),
        )
        all_arm_ids = self.arm_geom_ids["l"] | self.arm_geom_ids["r"]
        support_contact = self.contact_evidence(
            all_arm_ids,
            self.pickup_support_geom_ids,
        )
        self.gate(
            "flat_hands_clear_pickup_support_before_grasp",
            support_contact["force_n"] <= 0.2,
            f"force_n={support_contact['force_n']:.4f} "
            f"pairs={support_contact['pairs']}",
        )

        self.phase("horizontal_approach_and_grasp")
        self.move_mounts(grasp_positions, rotations, iterations=1200)
        self.ct.grip_cmd["l"] = self.ct.GRIP_CLOSE
        self.ct.grip_cmd["r"] = self.ct.GRIP_CLOSE
        self.frames(GRASP_SETTLE_TICKS)
        self.require_held("flat_pickup")

        self.phase("lift_flat_from_pickup_support")
        support_height = float(self.roll_position()[2])
        self.move_mount_commands_delta([0.0, 0.0, FLAT_PICK_LIFT_M])
        self.frames(15)
        lifted = float(self.roll_position()[2] - support_height)
        self.require_held("flat_support_lift")
        self.gate(
            "flat_support_lift_height",
            lifted >= FLAT_PICK_LIFT_M - 0.015,
            f"lifted={lifted:.4f}m",
        )
        roll_support_contact = self.contact_evidence(
            {self.roll_geom},
            self.pickup_support_geom_ids,
        )
        hand_support_contact = self.contact_evidence(
            all_arm_ids,
            self.pickup_support_geom_ids,
        )
        self.gate(
            "pickup_support_cleared_after_lift",
            roll_support_contact["force_n"] <= 0.05
            and hand_support_contact["force_n"] <= 0.2,
            f"roll_force_n={roll_support_contact['force_n']:.4f} "
            f"hand_force_n={hand_support_contact['force_n']:.4f} "
            f"hand_pairs={hand_support_contact['pairs']}",
        )
        spans = self.pad_vertical_span()
        axis = self.data.xmat[self.roll_body].reshape(3, 3)[:, 0]
        self.gate(
            "hands_remain_flat_after_pick",
            max(spans.values()) <= 0.025
            and abs(float(axis[2])) <= 0.02,
            "pad_vertical_span_mm="
            + json.dumps({
                name: round(1000.0 * span, 1)
                for name, span in spans.items()
            })
            + f" roll_axis={np.round(axis, 6).tolist()}",
        )

        self.phase("clear_table")
        base = self.ct.base_pose()
        roll = self.roll_position()
        yaw = float(base[2])
        world_from_base = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        carried_offset_local = (
            world_from_base.T @ (roll[:2] - base[:2])
        )
        corridor_base_y = (
            RELEASE_APPROACH_Y_BIAS_M
            - float(carried_offset_local[1])
        )
        corridor_delta = np.array([
            0.0,
            corridor_base_y - float(base[1]),
        ])
        reverse_direction = -np.array([
            math.cos(yaw),
            math.sin(yaw),
        ])
        corridor_reverse_m = float(
            np.dot(corridor_delta, reverse_direction)
        )
        corridor_cross_error = float(np.linalg.norm(
            corridor_delta
            - corridor_reverse_m * reverse_direction
        ))
        self.gate(
            "shelf_corridor_reverse_plan",
            TABLE_CLEAR_REVERSE_M <= corridor_reverse_m <= 0.60
            and corridor_cross_error <= 0.010
            and abs(angle(yaw + math.pi / 2.0)) <= 0.004,
            f"distance_m={corridor_reverse_m:.4f} "
            f"cross_error_mm={1000.0 * corridor_cross_error:.2f} "
            f"target_y={corridor_base_y:.4f}",
        )
        self.reverse(corridor_reverse_m, max_speed=0.26)
        base = self.ct.base_pose()
        self.gate(
            "shelf_corridor_staged",
            abs(float(base[1]) - corridor_base_y) <= 0.010,
            f"base={np.round(base, 4).tolist()} "
            f"target_y={corridor_base_y:.4f}",
        )
        self.require_held("table_clear")

        self.phase("rotate_to_shelf")
        self.turn_in_place(0.0)
        self.require_held("rotated")

        self.phase("navigate_to_shelf_stage")
        base = self.ct.base_pose()
        roll = self.roll_position()
        carried_offset = roll[:2] - base[:2]
        stage_center = TARGET_CENTER.copy()
        stage_center[0] += SHELF_STAGE_OFFSET_X
        stage_center[1] = RELEASE_APPROACH_Y_BIAS_M
        stage_center[2] = roll[2]
        shelf_base = stage_center[:2] - carried_offset
        travel_heading = math.atan2(
            shelf_base[1] - base[1],
            shelf_base[0] - base[0],
        )
        print(
            f"[transport_plan] carried_offset="
            f"{np.round(carried_offset, 4).tolist()} "
            f"base_target={np.round(shelf_base, 4).tolist()} "
            f"travel_heading_deg={math.degrees(travel_heading):.2f}",
            flush=True,
        )
        self.gate(
            "straight_shelf_approach",
            abs(angle(travel_heading)) <= 0.05
            and abs(angle(base[2])) <= 0.02,
            f"base={np.round(base, 4).tolist()} "
            f"travel_heading_deg={math.degrees(travel_heading):.2f}",
        )
        self.go_to(
            shelf_base, 0.0, max_speed=0.28, tolerance=0.006
        )
        self.require_held("shelf_translation")
        parked_base = self.ct.base_pose()
        self.gate(
            "shelf_park_pose",
            float(np.linalg.norm(parked_base[:2] - shelf_base))
            <= 0.010
            and abs(angle(parked_base[2])) <= 0.015,
            f"base={np.round(parked_base, 4).tolist()} "
            f"target={np.round(shelf_base, 4).tolist()}",
        )
        self.require_held("shelf_park")

        self.phase("align_shelf_axis_above_front_lip")
        self.align_roll_axis(
            "aligning_shelf_axis_above_front_lip",
            "shelf_axis_x_above_front_lip",
        )

        self.phase("realign_shelf_stage_after_axis")
        self.align_roll_center(
            stage_center,
            "high_stage_realignment",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.02,
            attempts=12,
            shelf_safe=True,
        )
        self.require_held("axis_aligned_high_stage")

        self.phase("level_release_support_surfaces")
        self.level_release_support_surfaces()
        self.align_roll_center(
            stage_center,
            "after_release_surface_leveling",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=8,
            shelf_safe=True,
        )
        self.require_held("release_surfaces_levelled")

        self.phase("fine_align_axis_before_entry")
        self.align_roll_axis_with_arms(
            "fine_aligning_integrated_entry_axis",
            "integrated_entry_axis_alignment",
            shelf_safe=True,
        )
        self.align_roll_center(
            stage_center,
            "high_stage_centered_before_entry",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=8,
            shelf_safe=True,
        )
        self.require_held("integrated_entry_axis_ready")

        self.phase("lower_to_front_lip_clearance")
        clearance_target = stage_center.copy()
        clearance_target[2] = RELEASE_CLEARANCE_ROLL_Z
        self.align_roll_center(
            clearance_target,
            "front_lip_clearance_height",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.02,
            attempts=20,
            shelf_safe=True,
            require_held_stage="front_lip_clearance_lowering",
        )
        self.require_held("front_lip_clearance_height")

        entry_target = clearance_target.copy()
        entry_target[0] = RELEASE_INSERT_TARGET_X_M
        entry_start_x = float(self.roll_position()[0])
        self.phase("move_over_integrated_front_lip")
        self.slowly_enter_integrated_top_tier(
            entry_target,
            max_step=RELEASE_INSERT_STEP_M,
        )
        self.require_held("inside_integrated_top_tier")
        entry_end_x = float(self.roll_position()[0])
        forward_entry_distance = entry_end_x - entry_start_x
        self.gate(
            "forward_entry_motion",
            forward_entry_distance >= 0.045,
            f"start_x={entry_start_x:.6f} "
            f"end_x={entry_end_x:.6f} "
            f"distance_mm={1000.0 * forward_entry_distance:.2f}",
        )

        self.phase("position_guarded_release_clearance")
        guarded_release_target = entry_target.copy()
        roll_radius = float(
            self.model.geom_size[self.roll_geom, 0]
        )
        guarded_release_target[2] = guarded_release_center_z(roll_radius)
        self.align_roll_center(
            guarded_release_target,
            "guarded_release_height",
            [0.001, PRE_RELEASE_Y_TOLERANCE_M, 0.001],
            command_space=True,
            max_step=0.006,
            attempts=16,
            shelf_safe=True,
            require_held_stage="guarded_release_lowering",
        )
        self.require_held("guarded_release_height")

        self.phase("guarded_release_and_lift_open_hands")
        print(
            f"[release] gripper_rate={RELEASE_GRIP_RATE:.3f}m/s "
            "with bounded drop and simultaneous upward hand clearance",
            flush=True,
        )
        self.release_into_integrated_top_tier()

        self.phase("verify_after_guarded_release")
        post_release_evidence = self.track_success(
            required_seconds=0.5,
        )
        self.gates["post_guarded_release_evidence"] = post_release_evidence
        self.gate(
            "placed_after_guarded_release",
            post_release_evidence is not None
            and post_release_evidence.get("instantaneous_success") is True
            and post_release_evidence.get("stable_seconds", 0.0) >= 0.5,
            json.dumps(post_release_evidence, ensure_ascii=False),
        )

        self.phase("retract_arms_after_release")
        start_mount_x = float(np.mean([
            self.data.xpos[self.ct.L.mount, 0],
            self.data.xpos[self.ct.R.mount, 0],
        ]))
        retract_positions = {
            "l": self.data.xpos[self.ct.L.mount].copy()
            + [-ARM_RETRACT_M, 0.0, 0.0],
            "r": self.data.xpos[self.ct.R.mount].copy()
            + [-ARM_RETRACT_M, 0.0, 0.0],
        }
        retract_rotations = {
            "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
            "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
        }
        self.move_mounts(
            retract_positions,
            retract_rotations,
            iterations=400,
            shelf_safe=True,
        )
        end_mount_x = float(np.mean([
            self.data.xpos[self.ct.L.mount, 0],
            self.data.xpos[self.ct.R.mount, 0],
        ]))
        pad_contact = self.contact_evidence(
            self.pad_ids, self.shelf_geom_ids
        )
        self.gate(
            "arms_retracted",
            start_mount_x - end_mount_x >= 0.08
            and pad_contact["force_n"] <= 0.2
            and not self.arm_shelf_contacts(),
            f"retracted_mm={1000.0 * (start_mount_x - end_mount_x):.1f} "
            f"shelf_contact_n={pad_contact['force_n']:.4f} "
            f"pairs={pad_contact['pairs']}",
        )

        self.phase("terminal_success_hold")
        self.final_evidence = self.track_success()
        self.gate(
            "sorting_roll_success_after_retract",
            self.final_evidence is not None
            and self.final_evidence.get("instantaneous_success") is True
            and self.final_evidence.get("stable_seconds", 0.0)
            >= REQUIRED_STABLE_SECONDS,
            json.dumps(
                self.final_evidence
                or self.evaluate_placement(self.model, self.data),
                ensure_ascii=False,
            ),
        )
        self.gate(
            "episode_under_two_minutes",
            self.sim_seconds <= 120.0,
            f"sim_seconds={self.sim_seconds:.3f}",
        )
        self.gate(
            "episode_under_one_minute",
            self.sim_seconds <= 60.0,
            f"sim_seconds={self.sim_seconds:.3f}",
        )
        return True

    def encode_frame_video(self, frame_pattern, output_path):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(self.ct.REC_FPS),
            "-i",
            str(frame_pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(output_path),
        ]
        subprocess.run(command, check=True)

    def encode_review_videos(self):
        if self.args.review_videos:
            self.encode_frame_video(
                self.review_dir / "frame_%06d.jpg",
                self.review_video,
            )
            self.encode_frame_video(
                self.slot_visual_review_dir / "frame_%06d.jpg",
                self.slot_visual_review_video,
            )
            self.encode_frame_video(
                self.slot_physics_review_dir / "frame_%06d.jpg",
                self.slot_physics_review_video,
            )
        for camera, video_path in self.robot_camera_videos.items():
            self.encode_frame_video(
                self.out / "frames" / camera / "frame_%06d.jpg",
                video_path,
            )

        width, height = self.args.width, self.args.height
        command = ["ffmpeg", "-y", "-loglevel", "error"]
        filters = []
        labeled_streams = []
        font_size = max(14, height // 22)
        for index, camera in enumerate(RECORDED_CAMERAS):
            command.extend([
                "-framerate",
                str(self.ct.REC_FPS),
                "-i",
                str(
                    self.out
                    / "frames"
                    / camera
                    / "frame_%06d.jpg"
                ),
            ])
            output = f"camera_{index}"
            source = MODEL_CAMERA_SOURCES[camera]
            role = CAMERA_ROLES[camera]
            filters.append(
                f"[{index}:v]drawtext="
                f"text='{camera} | {role} | model={source}':"
                f"x=10:y=10:fontsize={font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.65:boxborderw=5[{output}]"
            )
            labeled_streams.append(f"[{output}]")
        filters.append(
            "".join(labeled_streams)
            + "xstack=inputs=3:layout="
            + f"{width // 2}_0|0_{height}|{width}_{height}:fill=black[v]"
        )
        command.extend([
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            str(self.robot_multiview_video),
        ])
        subprocess.run(command, check=True)

    def finalize(self, success, error=None):
        self.ct.REC["on"] = False
        self.ct.REC["metadata"]["simulation_canary_eligible"] = bool(
            success
        )
        self.ct.REC["metadata"]["gates"] = self.gates
        self.ct.REC["metadata"]["final_evidence"] = self.final_evidence
        self.recorder.finalize(success=bool(success))

        if self.recorder.n:
            episode_path = self.out / "episode_data.npz"
            with np.load(episode_path, allow_pickle=False) as data:
                payload = {name: np.asarray(data[name]) for name in data.files}
            if len(self.recorded_roll_qpos) != self.recorder.n:
                raise RuntimeError(
                    "roll-state log and recorded camera frame counts differ"
                )
            payload["roll_qpos"] = np.asarray(
                self.recorded_roll_qpos, dtype=np.float32
            )
            payload["roll_qvel"] = np.asarray(
                self.recorded_roll_qvel, dtype=np.float32
            )
            np.savez(episode_path, **payload)

            meta_path = self.out / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({
                "task": "sorting_roll_cruzr",
                "task_version": self.task_version,
                "prompt": os.environ["REC_PROMPT"],
                "diversity": self.diversity,
                "success": bool(success),
                "success_source": (
                    "sorting_roll_task.SortingRollSuccessTracker"
                ),
                "training_eligible": False,
                "simulation_canary_eligible": bool(success),
                "review_video": (
                    self.review_video.name
                    if self.args.review_videos else None
                ),
                "slot_visual_review_video": (
                    self.slot_visual_review_video.name
                    if self.args.review_videos else None
                ),
                "slot_physics_review_video": (
                    self.slot_physics_review_video.name
                    if self.args.review_videos else None
                ),
                "robot_camera_videos": {
                    camera: path.name
                    for camera, path in self.robot_camera_videos.items()
                },
                "robot_multiview_video": self.robot_multiview_video.name,
            })
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self.recorder.close()
        if self.recorder.n:
            self.encode_review_videos()

        result = {
            "task": "sorting_roll_cruzr",
            "task_version": self.task_version,
            "seed": self.args.seed,
            "scene_randomization": self.scene_randomization,
            "prompt": os.environ["REC_PROMPT"],
            "diversity": self.diversity,
            "success": bool(success),
            "training_eligible": False,
            "simulation_canary_eligible": bool(success),
            "error": error,
            "num_frames": int(self.recorder.n),
            "sim_seconds": round(float(self.sim_seconds), 3),
            "gates": self.gates,
            "final_evidence": self.final_evidence,
            "review_video": (
                str(self.review_video)
                if self.args.review_videos else None
            ),
            "slot_visual_review_video": (
                str(self.slot_visual_review_video)
                if self.args.review_videos else None
            ),
            "slot_physics_review_video": (
                str(self.slot_physics_review_video)
                if self.args.review_videos else None
            ),
            "robot_camera_videos": {
                camera: str(path)
                for camera, path in self.robot_camera_videos.items()
            },
            "robot_multiview_video": str(self.robot_multiview_video),
        }
        self.result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")

    sys.path.insert(0, str(CORE_DIR))
    import sorting_roll_scene

    args.diversity_assignment = None
    prompt = None
    if args.manifest:
        args.manifest = args.manifest.resolve()
        manifest = load_manifest(args.manifest)
        args.diversity_assignment = assignment_for_seed(
            manifest, args.seed
        )
        prompt = args.diversity_assignment["prompt"]
    scene_path = sorting_roll_scene.materialize_scene()
    ct = load_teleop(scene_path, args.gpu, args.seed, prompt=prompt)
    import mujoco
    apply_model_camera_overrides(mujoco, ct.m)
    from sorting_roll_task import evaluate_placement, SortingRollSuccessTracker
    from teleop_timing import CumulativeSubstepScheduler

    scheduler = CumulativeSubstepScheduler(
        ct.TARGET_FPS, ct.m.opt.timestep
    )
    expert = SortingRollExpert(
        args,
        ct,
        mujoco,
        scheduler,
        evaluate_placement,
        SortingRollSuccessTracker,
    )
    if args.no_render:
        ct.REC["on"] = False
    success = False
    error = None
    try:
        success = expert.execute()
    except ExpertFailure as exc:
        error = str(exc)
        print(f"[expert] FAIL {error}", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[expert] ERROR {error}", flush=True)
        raise
    finally:
        expert.finalize(success, error=error)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
