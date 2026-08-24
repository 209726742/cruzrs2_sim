#!/usr/bin/env python3
"""Installation-diagram-constrained dual-wrist D405 candidate for Sorting Roll."""


PROFILE_NAME = "sorting_roll_d405_candidate_v3"
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
WRIST_D405_OPTICAL_POS_M = (0.0, -0.090, 0.070)
WRIST_D405_OPTICAL_QUAT_WXYZ = (0.5, 0.8660254, 0.0, 0.0)
MODEL_CAMERA_OVERRIDES = {
    "left_wrist_realsense": {
        "pos_m": WRIST_D405_OPTICAL_POS_M,
        "quat_wxyz": WRIST_D405_OPTICAL_QUAT_WXYZ,
        "fovy_deg": D405_FOV_DEG[1],
    },
    "right_wrist_realsense": {
        "pos_m": WRIST_D405_OPTICAL_POS_M,
        "quat_wxyz": WRIST_D405_OPTICAL_QUAT_WXYZ,
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
            "optical_position_in_gripper_mount_m": list(
                WRIST_D405_OPTICAL_POS_M
            ),
            "optical_quaternion_wxyz": list(
                WRIST_D405_OPTICAL_QUAT_WXYZ
            ),
            "left_right_transform": "identical_in_mirrored_gripper_frames",
            "cad_or_measured_transform_verified": False,
        },
        "camera_roles": dict(CAMERA_ROLES),
        "policy_image_map": dict(POLICY_IMAGE_MAP),
        "hardware_verified": HARDWARE_VERIFIED,
        "training_eligible": TRAINING_ELIGIBLE,
        "simulation_mount_status": (
            "installation_diagram_constrained_symmetric_mount_pending_cad_measurement"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
