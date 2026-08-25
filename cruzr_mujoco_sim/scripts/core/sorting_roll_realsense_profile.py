#!/usr/bin/env python3
"""Installation-diagram-constrained dual-wrist D405 candidate for Sorting Roll."""

import math

import numpy as np


PROFILE_NAME = "sorting_roll_d405_candidate_v6"
D405_MODEL = "RealSense D405"
D405_RGB_RESOLUTION_WH = (1280, 720)
D405_RGB_FPS = 30
D405_FOV_DEG = (87.0, 58.0)
D405_IDEAL_RANGE_M = (0.07, 0.50)
D405_SHUTTER = "global"
D405_DEPTH_POLICY_INPUT = False

# The wrist frames are task-specific additions. They do not replace or rename
# the frozen CRUZR SDK cameras or the legacy diagnostic hand cameras.
MODEL_CAMERA_SOURCES = {
    "stereo_left": "stereo_left",
    "left_wrist_realsense": "sorting_roll_left_wrist_d405",
    "right_wrist_realsense": "sorting_roll_right_wrist_d405",
}
LEFT_WRIST_D405_OPTICAL_POS_M = (0.0, -0.180, 0.070)
LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ = (0.5, 0.8660254, 0.0, 0.0)
RIGHT_WRIST_D405_OPTICAL_POS_M = LEFT_WRIST_D405_OPTICAL_POS_M
RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ = LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ
MODEL_CAMERA_OVERRIDES = {
    "left_wrist_realsense": {
        "pos_m": LEFT_WRIST_D405_OPTICAL_POS_M,
        "quat_wxyz": LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
        "fovy_deg": D405_FOV_DEG[1],
    },
    "right_wrist_realsense": {
        "pos_m": RIGHT_WRIST_D405_OPTICAL_POS_M,
        "quat_wxyz": RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ,
        "fovy_deg": D405_FOV_DEG[1],
    },
}
WRIST_D405_MOUNT_GEOMS = {
    "left_wrist_realsense": {
        "visual": (
            "L_sorting_roll_d405_body_visual",
            "L_sorting_roll_d405_face_visual",
            "L_sorting_roll_d405_rail_visual",
            "L_sorting_roll_d405_adapter_visual",
        ),
        "collision": (
            "L_sorting_roll_d405_body_collision",
            "L_sorting_roll_d405_rail_collision",
            "L_sorting_roll_d405_adapter_collision",
        ),
    },
    "right_wrist_realsense": {
        "visual": (
            "R_sorting_roll_d405_body_visual",
            "R_sorting_roll_d405_face_visual",
            "R_sorting_roll_d405_rail_visual",
            "R_sorting_roll_d405_adapter_visual",
        ),
        "collision": (
            "R_sorting_roll_d405_body_collision",
            "R_sorting_roll_d405_rail_collision",
            "R_sorting_roll_d405_adapter_collision",
        ),
    },
}
CAMERA_ROLES = {
    "stereo_left": "global",
    "left_wrist_realsense": "left_wrist",
    "right_wrist_realsense": "right_wrist",
}
POLICY_IMAGE_MAP = {
    "observation/image": "observation.images.stereo_left",
    "observation/left_wrist_image": (
        "observation.images.left_wrist_realsense"
    ),
    "observation/right_wrist_image": (
        "observation.images.right_wrist_realsense"
    ),
}

HARDWARE_VERIFIED = False
TRAINING_ELIGIBLE = False


def apply_model_camera_overrides(mujoco, model):
    for logical, override in MODEL_CAMERA_OVERRIDES.items():
        source = MODEL_CAMERA_SOURCES[logical]
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, source
        )
        if camera_id < 0:
            raise RuntimeError(
                f"scene is missing candidate camera source: {source}"
            )
        if "pos_m" in override:
            model.cam_pos[camera_id] = override["pos_m"]
        if "quat_wxyz" in override:
            model.cam_quat[camera_id] = override["quat_wxyz"]
        if "fovy_deg" in override:
            model.cam_fovy[camera_id] = override["fovy_deg"]
    for mount in WRIST_D405_MOUNT_GEOMS.values():
        for name in mount["visual"]:
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id < 0:
                raise RuntimeError(f"scene is missing D405 visual geom: {name}")
            model.geom_group[geom_id] = 1
            model.geom_rgba[geom_id, 3] = 1.0
        for name in mount["collision"]:
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id < 0:
                raise RuntimeError(f"scene is missing D405 collision geom: {name}")
            model.geom_group[geom_id] = 3
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1


def wrist_camera_initialization_report(mujoco, model, data):
    mujoco.mj_forward(model, data)
    chassis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
    )
    chassis_position = data.xpos[chassis_id].copy()
    world_to_base = data.xmat[chassis_id].reshape(3, 3).T
    positions = {}
    optical_forwards = {}
    optical_ups = {}
    pad_positions = {}
    gripper_view_angles_deg = {}
    gripper_distances_m = {}
    local_positions = {}
    local_quaternions = {}
    for logical, side in (
        ("left_wrist_realsense", "L"),
        ("right_wrist_realsense", "R"),
    ):
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            MODEL_CAMERA_SOURCES[logical],
        )
        if camera_id < 0:
            raise RuntimeError(f"scene is missing candidate camera: {logical}")
        positions[logical] = (
            world_to_base @ (data.cam_xpos[camera_id] - chassis_position)
        )
        camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
        optical_forwards[logical] = (
            world_to_base @ (-camera_rotation[:, 2])
        )
        optical_ups[logical] = world_to_base @ camera_rotation[:, 1]
        pads = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad{index}"
            )
            for index in (1, 2)
        ]
        if min(pads) < 0:
            raise RuntimeError(f"scene is missing {side} gripper pads")
        pad_positions[logical] = world_to_base @ (
            np.mean(data.geom_xpos[pads], axis=0) - chassis_position
        )
        camera_to_gripper = (
            pad_positions[logical] - positions[logical]
        )
        gripper_distances_m[logical] = float(
            np.linalg.norm(camera_to_gripper)
        )
        camera_to_gripper /= gripper_distances_m[logical]
        gripper_view_angles_deg[logical] = math.degrees(math.acos(
            float(np.clip(
                np.dot(optical_forwards[logical], camera_to_gripper),
                -1.0,
                1.0,
            ))
        ))
        local_positions[logical] = model.cam_pos[camera_id].copy()
        local_quaternions[logical] = model.cam_quat[camera_id].copy()
    mirror = np.diag([1.0, -1.0, 1.0])
    expected_right_pad_position = (
        mirror @ pad_positions["left_wrist_realsense"]
    )
    pad_position_error_m = float(np.linalg.norm(
        pad_positions["right_wrist_realsense"]
        - expected_right_pad_position
    ))
    camera_position_error_m = float(np.linalg.norm(
        positions["right_wrist_realsense"]
        - mirror @ positions["left_wrist_realsense"]
    ))

    def direction_error_deg(left, right):
        expected_right = mirror @ left
        cosine = float(np.dot(expected_right, right))
        cosine /= float(
            np.linalg.norm(expected_right) * np.linalg.norm(right)
        )
        return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

    optical_forward_error_deg = direction_error_deg(
        optical_forwards["left_wrist_realsense"],
        optical_forwards["right_wrist_realsense"],
    )
    optical_up_error_deg = direction_error_deg(
        optical_ups["left_wrist_realsense"],
        optical_ups["right_wrist_realsense"],
    )
    checks = {
        "both_cameras_in_front": bool(
            min(position[0] for position in positions.values()) >= 0.250
        ),
        "cameras_on_expected_sides": bool(
            positions["left_wrist_realsense"][1] > 0.0
            and positions["right_wrist_realsense"][1] < 0.0
        ),
        "grippers_in_front_on_expected_sides": bool(
            pad_positions["left_wrist_realsense"][0] >= 0.250
            and pad_positions["right_wrist_realsense"][0] >= 0.250
            and pad_positions["left_wrist_realsense"][1] > 0.0
            and pad_positions["right_wrist_realsense"][1] < 0.0
        ),
        "gripper_positions_sagittally_symmetric_within_12mm": (
            pad_position_error_m <= 0.012
        ),
        "camera_positions_sagittally_symmetric_within_12mm": (
            camera_position_error_m <= 0.012
        ),
        "camera_optical_axes_mirrored_within_1deg": (
            optical_forward_error_deg <= 1.0
            and optical_up_error_deg <= 1.0
        ),
        "same_local_installation_transform": bool(
            np.allclose(
                local_positions["left_wrist_realsense"],
                local_positions["right_wrist_realsense"],
                atol=1e-9,
            )
            and np.allclose(
                local_quaternions["left_wrist_realsense"],
                local_quaternions["right_wrist_realsense"],
                atol=1e-9,
            )
        ),
        "both_cameras_observe_own_gripper": bool(
            max(gripper_view_angles_deg.values())
            <= 0.5 * D405_FOV_DEG[1]
            and min(gripper_distances_m.values()) >= D405_IDEAL_RANGE_M[0]
            and max(gripper_distances_m.values()) <= D405_IDEAL_RANGE_M[1]
        ),
    }
    return {
        "passed": all(checks.values()),
        "frame": "chassis",
        "positions_base_m": {
            logical: np.round(position, 6).tolist()
            for logical, position in positions.items()
        },
        "gripper_positions_base_m": {
            logical: np.round(position, 6).tolist()
            for logical, position in pad_positions.items()
        },
        "optical_forwards_base": {
            logical: np.round(direction, 6).tolist()
            for logical, direction in optical_forwards.items()
        },
        "optical_ups_base": {
            logical: np.round(direction, 6).tolist()
            for logical, direction in optical_ups.items()
        },
        "gripper_view_angles_deg": {
            logical: round(angle, 4)
            for logical, angle in gripper_view_angles_deg.items()
        },
        "gripper_distances_m": {
            logical: round(distance, 6)
            for logical, distance in gripper_distances_m.items()
        },
        "gripper_sagittal_position_error_mm": round(
            1000.0 * pad_position_error_m, 3
        ),
        "camera_sagittal_position_error_mm": round(
            1000.0 * camera_position_error_m, 3
        ),
        "camera_optical_forward_mirror_error_deg": round(
            optical_forward_error_deg, 4
        ),
        "camera_optical_up_mirror_error_deg": round(
            optical_up_error_deg, 4
        ),
        "checks": checks,
    }


def profile_report():
    checks = {
        "exactly_three_policy_cameras": len(MODEL_CAMERA_SOURCES) == 3,
        "one_camera_per_required_role": set(CAMERA_ROLES.values())
        == {"global", "left_wrist", "right_wrist"},
        "logical_camera_names_are_unique": (
            len(set(MODEL_CAMERA_SOURCES)) == len(MODEL_CAMERA_SOURCES)
        ),
        "model_camera_sources_are_unique": (
            len(set(MODEL_CAMERA_SOURCES.values()))
            == len(MODEL_CAMERA_SOURCES)
        ),
        "policy_targets_match_logical_cameras": {
            value.rsplit(".", 1)[-1] for value in POLICY_IMAGE_MAP.values()
        }
        == set(MODEL_CAMERA_SOURCES),
    }
    return {
        "profile": PROFILE_NAME,
        "camera_model": D405_MODEL,
        "d405_rgb_resolution_wh": list(D405_RGB_RESOLUTION_WH),
        "d405_rgb_fps": D405_RGB_FPS,
        "d405_fov_deg": list(D405_FOV_DEG),
        "d405_ideal_range_m": list(D405_IDEAL_RANGE_M),
        "d405_shutter": D405_SHUTTER,
        "depth_policy_input": D405_DEPTH_POLICY_INPUT,
        "model_camera_sources": dict(MODEL_CAMERA_SOURCES),
        "model_camera_overrides": dict(MODEL_CAMERA_OVERRIDES),
        "wrist_mount_reference": {
            "mechanical_constraint": (
                "adapter_rotated_90deg_slider_rightmost_camera_topmost"
            ),
            "left_optical_position_in_gripper_mount_m": list(
                LEFT_WRIST_D405_OPTICAL_POS_M
            ),
            "left_optical_quaternion_wxyz": list(
                LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ
            ),
            "right_optical_position_in_gripper_mount_m": list(
                RIGHT_WRIST_D405_OPTICAL_POS_M
            ),
            "right_optical_quaternion_wxyz": list(
                RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ
            ),
            "left_right_transform": (
                "same_installation_transform_in_side_specific_gripper_frames"
            ),
            "cad_or_measured_transform_verified": False,
        },
        "camera_roles": dict(CAMERA_ROLES),
        "policy_image_map": dict(POLICY_IMAGE_MAP),
        "hardware_verified": HARDWARE_VERIFIED,
        "training_eligible": TRAINING_ELIGIBLE,
        "simulation_mount_status": (
            "installation_diagram_constrained_isomorphic_mount_pending_cad_measurement"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
