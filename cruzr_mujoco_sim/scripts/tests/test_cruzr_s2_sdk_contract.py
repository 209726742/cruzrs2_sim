#!/usr/bin/env python3
"""Tests for the Cruzr S2 SDK alignment contract."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from cruzr_s2_sdk_contract import (  # noqa: E402
    ARM_JOINT_NAMES,
    ARM_MAX_SPEED_RAD_S,
    ARM_POSITION_LIMITS_RAD,
    ARM_RATED_DELTA_RAD_AT_DATASET_FPS,
    ARM_RATED_SPEED_RAD_S,
    SDK_CAMERAS,
    SDK_DOCUMENTED_RGB_CAMERA_TOPICS,
    SDK_CAMERA_INTRINSICS_VERIFIED,
    SDK_SENSOR_EXTRINSICS_ZYX,
    SDK_SENSOR_ROTATION_ORDER,
    SDK_WRIST_CAMERAS,
    SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS,
    SDK_COMMAND_POSITION_MARGIN_RAD,
    SDK_COMMAND_SPEED_FRACTION_OF_RATED,
    SDK_COLLECTION_PROFILE,
    SDK_TASK_HEAD_POSE_RAD,
    audit_sdk_episode,
    clip_arm_target_to_operational_limits,
    contract_summary,
    model_contract_errors,
    rate_limit_arm_target,
)


class CruzrS2SdkContractTest(unittest.TestCase):
    def valid_episode(self, n=8):
        action = np.zeros((n, 16), dtype=np.float64)
        midpoint = ARM_POSITION_LIMITS_RAD.mean(axis=1)
        action[:, :14] = midpoint + np.arange(n)[:, None] * 0.01
        action[:, 14:16] = 0.5
        state = action.copy()
        base_action = np.zeros((n, 2), dtype=np.float64)
        base_action[:, 0] = np.linspace(0.0, 0.1, n)
        timestamp = (np.arange(n) + 1.0) / 30.0
        sdk_state_timestamp = 100.0 + timestamp
        camera_timestamps = {
            camera: sdk_state_timestamp + offset
            for camera, offset in zip(SDK_CAMERAS, (0.004, -0.003, 0.006))
        }
        return {
            "state": state,
            "action": action,
            "base_action": base_action,
            "fps": 30,
            "joint_names": ARM_JOINT_NAMES + ("grip_l", "grip_r"),
            "cameras": SDK_CAMERAS,
            "timestamp": timestamp,
            "sdk_state_timestamp": sdk_state_timestamp,
            "camera_timestamps": camera_timestamps,
            "require_camera_timestamps": True,
        }

    def test_contract_summary_keeps_unverified_gripper_polarity_explicit(self):
        summary = contract_summary()
        self.assertEqual(summary["sdk_document_revision"], "v2.1-2026-06-23")
        self.assertEqual(summary["collection_profile"], SDK_COLLECTION_PROFILE)
        self.assertEqual(summary["sdk_cameras"], list(SDK_CAMERAS))
        self.assertEqual(
            SDK_CAMERAS, ("stereo_left", "waist_front", "chassis_front")
        )
        self.assertEqual(summary["documented_rgb_camera_count"], 6)
        self.assertEqual(
            set(summary["documented_rgb_camera_topics"]),
            {
                "chassis_front",
                "waist_front",
                "fisheye_left",
                "fisheye_right",
                "stereo_left",
                "stereo_right",
            },
        )
        self.assertEqual(summary["wrist_cameras"], [])
        self.assertEqual(SDK_WRIST_CAMERAS, ())
        self.assertEqual(len(SDK_DOCUMENTED_RGB_CAMERA_TOPICS), 6)
        self.assertEqual(SDK_SENSOR_ROTATION_ORDER, "ZYX")
        self.assertFalse(SDK_CAMERA_INTRINSICS_VERIFIED)
        self.assertFalse(summary["camera_intrinsics_verified"])
        self.assertEqual(
            set(SDK_SENSOR_EXTRINSICS_ZYX),
            set(SDK_DOCUMENTED_RGB_CAMERA_TOPICS),
        )
        self.assertEqual(
            SDK_SENSOR_EXTRINSICS_ZYX["waist_front"],
            {
                "parent_link": "waist_yaw_link",
                "xyz_m": (0.07754007, 0.0, 0.02319591),
                "rpy_deg": (0.0, 51.0, 0.0),
            },
        )
        self.assertEqual(
            summary["sensor_extrinsics_zyx"]["stereo_left"]["parent_link"],
            "head_pitch_link",
        )
        self.assertEqual(
            summary["task_head_pose_rad"], SDK_TASK_HEAD_POSE_RAD
        )
        self.assertFalse(summary["gripper"]["position_polarity_verified"])
        self.assertAlmostEqual(ARM_RATED_SPEED_RAD_S, 2.0 * np.pi / 3.0)
        self.assertAlmostEqual(ARM_MAX_SPEED_RAD_S, np.pi)
        self.assertAlmostEqual(
            ARM_RATED_DELTA_RAD_AT_DATASET_FPS,
            ARM_RATED_SPEED_RAD_S / 30.0,
        )
        self.assertAlmostEqual(SDK_COMMAND_SPEED_FRACTION_OF_RATED, 0.95)
        self.assertAlmostEqual(SDK_COMMAND_POSITION_MARGIN_RAD, 0.006)

    def test_valid_sdk_episode_passes(self):
        result = audit_sdk_episode(**self.valid_episode())
        self.assertTrue(result["passed"], result["errors"])
        self.assertLess(result["joint_command_speed"]["max_rad_s"], ARM_RATED_SPEED_RAD_S)
        self.assertLessEqual(
            result["camera_state_timestamp"]["max_skew_s"], 0.020
        )

    def test_rated_and_absolute_speed_are_distinct(self):
        case = self.valid_episode()
        case["action"][1:, 0] = 0.08
        case["state"][:, 0] = case["action"][:, 0]
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(any("SDK rated" in error for error in result["errors"]))

        case["enforce_rated_speed"] = False
        result = audit_sdk_episode(**case)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(any("SDK rated" in warning for warning in result["warnings"]))

        case = self.valid_episode()
        case["action"][1:, 0] = 0.11
        case["state"][:, 0] = case["action"][:, 0]
        case["enforce_rated_speed"] = False
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(any("SDK maximum" in error for error in result["errors"]))

    def test_rollout_target_limiter_uses_rated_dataset_delta(self):
        current = np.array([0.0, 0.0, 0.1])
        limited = rate_limit_arm_target(current, np.array([1.0, -1.0, 0.11]))
        self.assertTrue(
            np.all(np.abs(limited - current) <= SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS)
        )
        self.assertAlmostEqual(limited[2], 0.11)
        with self.assertRaises(ValueError):
            rate_limit_arm_target(np.zeros(2), np.zeros(3))

    def test_operational_position_margin_clips_without_relaxing_sdk_limits(self):
        target = ARM_POSITION_LIMITS_RAD[:, 1].copy()
        clipped = clip_arm_target_to_operational_limits(target)
        np.testing.assert_allclose(
            ARM_POSITION_LIMITS_RAD[:, 1] - clipped,
            SDK_COMMAND_POSITION_MARGIN_RAD,
        )
        case = self.valid_episode()
        case["action"][:, :14] = clipped
        case["state"] = case["action"].copy()
        result = audit_sdk_episode(**case)
        self.assertTrue(result["passed"], result["errors"])

        case["action"][:, 0] = ARM_POSITION_LIMITS_RAD[0, 1]
        case["state"] = case["action"].copy()
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("operational limits" in error for error in result["errors"])
        )

    def test_joint_limit_camera_and_base_range_are_hard_gates(self):
        case = self.valid_episode()
        case["action"][2, 1] = ARM_POSITION_LIMITS_RAD[1, 1] + 0.01
        case["base_action"][3, 0] = -0.31
        case["cameras"] = (
            "head_stereo_l_shelf",
            "chassis_front",
            "hand_right_shelf",
        )
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        joined = "\n".join(result["errors"])
        self.assertIn("outside", joined)
        self.assertIn("base v_fwd", joined)
        self.assertIn("camera order", joined)

    def test_small_measured_limit_overshoot_warns_but_command_overshoot_fails(self):
        case = self.valid_episode()
        case["state"][2, 1] = ARM_POSITION_LIMITS_RAD[1, 1] + 0.0006
        result = audit_sdk_episode(**case)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(
            any("state tolerance" in warning for warning in result["warnings"])
        )

        case["action"][2, 1] = ARM_POSITION_LIMITS_RAD[1, 1] + 0.0006
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("action" in error and "outside" in error for error in result["errors"])
        )

    def test_long_float32_recorder_timestamp_grid_passes(self):
        case = self.valid_episode(n=8658)
        case["timestamp"] = (
            (np.arange(8658, dtype=np.float64) + 1.0) / 30.0
        ).astype(np.float32)
        case["action"][:, :14] = ARM_POSITION_LIMITS_RAD.mean(axis=1)
        case["state"] = case["action"].copy()
        case["sdk_state_timestamp"] = 100.0 + np.arange(8658) / 30.0
        case["camera_timestamps"] = {
            camera: case["sdk_state_timestamp"] + 0.001
            for camera in SDK_CAMERAS
        }
        result = audit_sdk_episode(**case)
        self.assertTrue(result["passed"], result["errors"])

    def test_timestamp_skew_and_missing_sidecar_are_rejected_when_required(self):
        case = self.valid_episode()
        case["camera_timestamps"][SDK_CAMERAS[-1]] += 0.025
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(any("timestamp skew" in error for error in result["errors"]))

        case = self.valid_episode()
        case["sdk_state_timestamp"] = None
        case["camera_timestamps"] = None
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(any("timestamps are required" in error for error in result["errors"]))

    def test_raw_timestamp_cumulative_drift_is_rejected(self):
        case = self.valid_episode(n=1000)
        case["action"][:, :14] = ARM_POSITION_LIMITS_RAD.mean(axis=1)
        case["state"] = case["action"].copy()
        case["sdk_state_timestamp"] = 100.0 + np.arange(1000) * 0.034
        case["camera_timestamps"] = {
            camera: case["sdk_state_timestamp"].copy()
            for camera in SDK_CAMERAS
        }
        result = audit_sdk_episode(**case)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("duration drift" in error for error in result["errors"]),
            result["errors"],
        )

    def test_drift_free_integer_substep_timestamps_pass(self):
        n = 300
        case = self.valid_episode(n=n)
        case["action"][:, :14] = ARM_POSITION_LIMITS_RAD.mean(axis=1)
        case["state"] = case["action"].copy()
        # 60 Hz control over 1 ms physics alternates 16/17 substeps; recording
        # every two ticks produces this exact-average 30 Hz cadence.
        control_steps = np.diff(
            np.rint(np.arange(2 * n + 1) * (1000.0 / 60.0)).astype(int)
        )
        frame_steps = control_steps.reshape(n, 2).sum(axis=1)
        case["sdk_state_timestamp"] = 100.0 + np.cumsum(frame_steps) * 0.001
        case["camera_timestamps"] = {
            camera: case["sdk_state_timestamp"].copy()
            for camera in SDK_CAMERAS
        }
        result = audit_sdk_episode(**case)
        self.assertTrue(result["passed"], result["errors"])
        cadence = result["camera_state_timestamp"]["raw_cadence"]
        self.assertLess(cadence["duration_drift_s"], 0.0011)

    def test_model_joint_ranges_match_sdk_v21(self):
        import mujoco

        model_path = os.path.join(ROOT, "assets", "e2e_dual_scene_1.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        ranges = []
        for name in ARM_JOINT_NAMES:
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            self.assertGreaterEqual(joint, 0, name)
            ranges.append(model.jnt_range[joint].copy())
        self.assertEqual(model_contract_errors(ARM_JOINT_NAMES, ranges), [])

    def test_model_contract_rejects_old_wider_urdf_range(self):
        ranges = ARM_POSITION_LIMITS_RAD.copy()
        ranges[0] = (-2.8623, 2.8623)
        errors = model_contract_errors(ARM_JOINT_NAMES, ranges)
        self.assertEqual(len(errors), 1)
        self.assertIn("L_shoulder_pitch_joint", errors[0])


if __name__ == "__main__":
    unittest.main()
