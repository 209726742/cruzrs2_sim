#!/usr/bin/env python3
"""Unverified dual-wrist RealSense candidate for Sorting Roll."""


PROFILE_NAME = "sorting_roll_d405_candidate_v1"
D405_MODEL = "RealSense D405"
D405_RGB_RESOLUTION_WH = (1280, 720)
D405_RGB_FPS = 30
D405_FOV_DEG = (87.0, 58.0)
D405_IDEAL_RANGE_M = (0.07, 0.50)
D405_SHUTTER = "global"
D405_DEPTH_POLICY_INPUT = False

# Logical dataset cameras are deliberately separate from the frozen CRUZR SDK
# camera names. The two wrist sources are existing MuJoCo diagnostic mounts used
# only to screen geometry until a real camera model and measured mount exist.
MODEL_CAMERA_SOURCES = {
    "stereo_left": "stereo_left",
    "left_wrist_realsense": "hand_left_shelf",
    "right_wrist_realsense": "hand_right",
}
MODEL_CAMERA_OVERRIDES = {
    "left_wrist_realsense": {
        "fovy_deg": D405_FOV_DEG[1],
    },
    "right_wrist_realsense": {
        "quat_wxyz": (-0.2642198, 0.1318878, 0.9134162, -0.2801147),
        "fovy_deg": D405_FOV_DEG[1],
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
        if "quat_wxyz" in override:
            model.cam_quat[camera_id] = override["quat_wxyz"]
        if "fovy_deg" in override:
            model.cam_fovy[camera_id] = override["fovy_deg"]


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
        "camera_roles": dict(CAMERA_ROLES),
        "policy_image_map": dict(POLICY_IMAGE_MAP),
        "hardware_verified": HARDWARE_VERIFIED,
        "training_eligible": TRAINING_ELIGIBLE,
        "simulation_mount_status": (
            "d405_intrinsics_with_asymmetric_proxy_extrinsics_pending_real_mount"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
