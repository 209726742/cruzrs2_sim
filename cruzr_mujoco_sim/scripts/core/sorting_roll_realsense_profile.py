#!/usr/bin/env python3
"""Unverified dual-wrist RealSense candidate for Sorting Roll."""


PROFILE_NAME = "sorting_roll_realsense_candidate_v1"

# Logical dataset cameras are deliberately separate from the frozen CRUZR SDK
# camera names. The two wrist sources are existing MuJoCo diagnostic mounts used
# only to screen geometry until a real camera model and measured mount exist.
MODEL_CAMERA_SOURCES = {
    "stereo_left": "stereo_left",
    "left_wrist_realsense": "hand_left_shelf",
    "right_wrist_realsense": "hand_right",
}
MODEL_CAMERA_OVERRIDES = {
    "right_wrist_realsense": {
        "quat_wxyz": (-0.2642198, 0.1318878, 0.9134162, -0.2801147),
        "fovy_deg": 75.0,
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
        model.cam_quat[camera_id] = override["quat_wxyz"]
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
        "model_camera_sources": dict(MODEL_CAMERA_SOURCES),
        "model_camera_overrides": dict(MODEL_CAMERA_OVERRIDES),
        "camera_roles": dict(CAMERA_ROLES),
        "policy_image_map": dict(POLICY_IMAGE_MAP),
        "hardware_verified": HARDWARE_VERIFIED,
        "training_eligible": TRAINING_ELIGIBLE,
        "simulation_mount_status": (
            "asymmetric_historical_diagnostic_proxies_pending_real_model_and_mount"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
