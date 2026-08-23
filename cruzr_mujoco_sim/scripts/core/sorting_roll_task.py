#!/usr/bin/env python3
"""Success contract for placing the rigid roll in the integrated top shelf."""

import math

import numpy as np

from sorting_roll_scene import (
    TARGET_AXIS,
    TARGET_CENTER,
    TOP_TIER_BACK_INNER_X_M,
    TOP_TIER_FRONT_LIP_PEAK_Z_M,
    TOP_TIER_FRONT_LIP_X_M,
    TOP_TIER_TROUGH_TOP_Z_M,
)


ROLL_LENGTH_M = 0.5
ROLL_VISUAL_DIAMETER_M = 0.025
ROLL_COLLISION_RADIUS_M = 0.012
SHELF_INNER_HALF_WIDTH_M = 0.285
TOP_TIER_CENTER_X_TOLERANCE_M = 0.020
TARGET_Y_TOLERANCE_M = 0.055
TARGET_Z_TOLERANCE_M = 0.012
TARGET_AXIS_TOLERANCE_DEG = 10.0
SUPPORT_FORCE_MIN_N = 0.5
RELEASE_FORCE_MAX_N = 0.2
TOP_TIER_TROUGH_GAP_TOLERANCE_M = 0.004
LINEAR_SPEED_MAX_M_S = 0.02
ANGULAR_SPEED_MAX_RAD_S = 0.10
REQUIRED_STABLE_SECONDS = 2.0

INSTANTANEOUS_CHECKS = (
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


def fit_report():
    shelf_usable_width = 2.0 * SHELF_INNER_HALF_WIDTH_M
    length_clearance = shelf_usable_width - ROLL_LENGTH_M
    roll_to_shelf_width_ratio = ROLL_LENGTH_M / shelf_usable_width
    integrated_pocket_depth = (
        TOP_TIER_BACK_INNER_X_M - TOP_TIER_FRONT_LIP_X_M
    )
    pocket_depth_clearance = (
        integrated_pocket_depth - ROLL_VISUAL_DIAMETER_M
    )
    front_lip_rise = (
        TOP_TIER_FRONT_LIP_PEAK_Z_M - TOP_TIER_TROUGH_TOP_Z_M
    )
    checks = {
        "roll_length_fits_shelf_inner_width": bool(length_clearance > 0.0),
        "roll_length_ratio_is_80_to_90_percent": bool(
            0.80 <= roll_to_shelf_width_ratio <= 0.90
        ),
        "integrated_pocket_deeper_than_roll_diameter": bool(
            pocket_depth_clearance > 0.0
        ),
        "front_lip_rises_above_roll_center": bool(
            front_lip_rise > ROLL_COLLISION_RADIUS_M
        ),
    }
    return {
        "shelf_usable_width_m": shelf_usable_width,
        "roll_length_m": ROLL_LENGTH_M,
        "length_clearance_total_m": length_clearance,
        "length_clearance_each_end_m": length_clearance / 2.0,
        "roll_to_shelf_width_ratio": roll_to_shelf_width_ratio,
        "roll_visual_diameter_m": ROLL_VISUAL_DIAMETER_M,
        "roll_collider_diameter_m": 2.0 * ROLL_COLLISION_RADIUS_M,
        "integrated_pocket_depth_m": integrated_pocket_depth,
        "pocket_depth_clearance_m": pocket_depth_clearance,
        "front_lip_rise_m": front_lip_rise,
        "checks": checks,
        "simulation_fits": all(checks.values()),
    }


def axis_alignment_degrees(axis, target=TARGET_AXIS):
    axis = np.array(axis, dtype=float, copy=True)
    target = np.array(target, dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    target /= np.linalg.norm(target)
    cosine = float(np.clip(abs(np.dot(axis, target)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _named_id(mujoco, model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"scene is missing {name}")
    return object_id


def _contact_report(mujoco, model, data, first, second):
    force = np.zeros(6, dtype=float)
    total = 0.0
    positions = []
    distances = []
    for index in range(data.ncon):
        pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        if pair & first and pair & second:
            mujoco.mj_contactForce(model, data, index, force)
            total += abs(float(force[0]))
            positions.append(
                np.round(data.contact[index].pos, 6).tolist()
            )
            distances.append(round(float(data.contact[index].dist), 7))
    return {
        "force_n": total,
        "count": len(positions),
        "positions_m": positions,
        "distances_m": distances,
    }


def _contact_force(mujoco, model, data, first, second):
    return _contact_report(
        mujoco,
        model,
        data,
        first,
        second,
    )["force_n"]


def evaluate_placement(model, data):
    import mujoco

    roll_body = _named_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "sorting_roll"
    )
    roll_geom = _named_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "sorting_roll_col"
    )
    trough_geom = _named_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "shelf_top_trough_col"
    )
    integrated_support_geoms = {
        _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "shelf_top_front_lip_col",
            "shelf_top_trough_col",
            "shelf_top_back_slope_col",
        )
    }
    table_top = _named_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_col"
    )
    pad_ids = {
        _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("L_pad1", "L_pad2", "R_pad1", "R_pad2")
    }

    mujoco.mj_forward(model, data)
    center = data.xpos[roll_body].copy()
    rotation = data.xmat[roll_body].reshape(3, 3)
    roll_axis = rotation[:, 0].copy()
    alignment_degrees = axis_alignment_degrees(roll_axis)
    half_y_span = (
        0.5 * ROLL_LENGTH_M * abs(float(roll_axis[1]))
        + ROLL_COLLISION_RADIUS_M
        * math.sqrt(max(0.0, 1.0 - float(roll_axis[1]) ** 2))
    )
    half_z_span = (
        0.5 * ROLL_LENGTH_M * abs(float(roll_axis[2]))
        + ROLL_COLLISION_RADIUS_M
        * math.sqrt(max(0.0, 1.0 - float(roll_axis[2]) ** 2))
    )
    trough_rotation = data.geom_xmat[trough_geom].reshape(3, 3)
    trough_half_z = float(
        np.abs(trough_rotation[2])
        @ model.geom_size[trough_geom, :3]
    )
    trough_top_z = float(
        data.geom_xpos[trough_geom, 2] + trough_half_z
    )
    roll_bottom_z = float(center[2] - half_z_span)
    trough_gap = roll_bottom_z - trough_top_z

    endpoint_margin_m = {
        "negative_y": float(
            center[1] - half_y_span + SHELF_INNER_HALF_WIDTH_M
        ),
        "positive_y": float(
            SHELF_INNER_HALF_WIDTH_M - center[1] - half_y_span
        ),
    }
    trough_contact = _contact_report(
        mujoco, model, data, {roll_geom}, {trough_geom}
    )
    integrated_contact = _contact_report(
        mujoco, model, data, {roll_geom}, integrated_support_geoms
    )
    integrated_support = integrated_contact["force_n"]
    table_support = _contact_force(
        mujoco, model, data, {roll_geom}, {table_top}
    )
    gripper_force = _contact_force(
        mujoco, model, data, {roll_geom}, pad_ids
    )
    angular_speed = float(np.linalg.norm(data.cvel[roll_body, :3]))
    linear_speed = float(np.linalg.norm(data.cvel[roll_body, 3:]))

    checks = {
        "center_inside_integrated_top_tier": bool(
            abs(center[0] - TARGET_CENTER[0])
            <= TOP_TIER_CENTER_X_TOLERANCE_M
            and abs(center[1] - TARGET_CENTER[1]) <= TARGET_Y_TOLERANCE_M
            and abs(center[2] - TARGET_CENTER[2]) <= TARGET_Z_TOLERANCE_M
        ),
        "fully_inside_shelf_width": bool(
            endpoint_margin_m["negative_y"] >= 0.0
            and endpoint_margin_m["positive_y"] >= 0.0
        ),
        "axis_aligned_with_shelf": bool(
            alignment_degrees <= TARGET_AXIS_TOLERANCE_DEG
        ),
        "supported_by_integrated_top_tier": bool(
            integrated_support >= SUPPORT_FORCE_MIN_N
        ),
        "resting_on_integrated_top_tier_geometry": bool(
            integrated_contact["count"] >= 2
            and abs(trough_gap) <= TOP_TIER_TROUGH_GAP_TOLERANCE_M
        ),
        "released_from_both_grippers": bool(gripper_force <= RELEASE_FORCE_MAX_N),
        "not_supported_by_table": bool(table_support < SUPPORT_FORCE_MIN_N),
        "low_linear_speed": bool(linear_speed <= LINEAR_SPEED_MAX_M_S),
        "low_angular_speed": bool(angular_speed <= ANGULAR_SPEED_MAX_RAD_S),
    }
    return {
        "center_m": np.round(center, 6).tolist(),
        "axis": np.round(roll_axis, 6).tolist(),
        "axis_error_deg": round(alignment_degrees, 4),
        "half_y_span_m": round(half_y_span, 6),
        "integrated_top_tier_support_force_n": round(
            integrated_support, 4
        ),
        "top_tier_contact_count": integrated_contact["count"],
        "top_tier_contact_positions_m": integrated_contact["positions_m"],
        "top_tier_contact_distances_m": integrated_contact["distances_m"],
        "trough_contact_count": trough_contact["count"],
        "trough_contact_positions_m": trough_contact["positions_m"],
        "trough_contact_distances_m": trough_contact["distances_m"],
        "roll_bottom_to_trough_gap_m": round(trough_gap, 7),
        "endpoint_margin_m": {
            side: round(margin, 6)
            for side, margin in endpoint_margin_m.items()
        },
        "table_support_force_n": round(table_support, 4),
        "gripper_contact_force_n": round(gripper_force, 4),
        "linear_speed_m_s": round(linear_speed, 6),
        "angular_speed_rad_s": round(angular_speed, 6),
        "checks": checks,
        "instantaneous_success": all(checks[name] for name in INSTANTANEOUS_CHECKS),
    }


class SortingRollSuccessTracker:
    def __init__(self, required_seconds=REQUIRED_STABLE_SECONDS):
        self.required_seconds = float(required_seconds)
        self.stable_seconds = 0.0

    def update(self, evidence, dt):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if evidence.get("instantaneous_success") is True:
            self.stable_seconds += float(dt)
        else:
            self.stable_seconds = 0.0
        return self.stable_seconds + 1e-12 >= self.required_seconds


def place_roll_at_target(model, data, height_offset_m=0.002):
    import mujoco

    joint = _named_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "sorting_roll_free"
    )
    qpos_adr = int(model.jnt_qposadr[joint])
    dof_adr = int(model.jnt_dofadr[joint])
    data.qpos[qpos_adr:qpos_adr + 3] = TARGET_CENTER + [0.0, 0.0, height_offset_m]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [
        math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)
    ]
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)


def target_placement_smoke(model, steps=1200):
    import mujoco

    data = mujoco.MjData(model)
    place_roll_at_target(model, data)
    tracker = SortingRollSuccessTracker()
    latched = False
    evidence = None
    executed_steps = 0
    for executed_steps in range(1, steps + 1):
        mujoco.mj_step(model, data)
        evidence = evaluate_placement(model, data)
        latched = tracker.update(evidence, float(model.opt.timestep))
        if latched:
            break
    evidence = dict(evidence)
    evidence["stable_seconds"] = round(tracker.stable_seconds, 4)
    evidence["required_stable_seconds"] = tracker.required_seconds
    evidence["simulated_seconds"] = round(
        executed_steps * float(model.opt.timestep), 4
    )
    evidence["success"] = bool(latched)
    return data, evidence
