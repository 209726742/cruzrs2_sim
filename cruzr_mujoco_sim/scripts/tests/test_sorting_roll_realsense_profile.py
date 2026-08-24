#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_realsense_profile import (
    CAMERA_ROLES,
    D405_DEPTH_POLICY_INPUT,
    D405_FOV_DEG,
    D405_IDEAL_RANGE_M,
    D405_MODEL,
    D405_RGB_FPS,
    D405_RGB_RESOLUTION_WH,
    D405_SHUTTER,
    HARDWARE_VERIFIED,
    MODEL_CAMERA_OVERRIDES,
    MODEL_CAMERA_SOURCES,
    POLICY_IMAGE_MAP,
    TRAINING_ELIGIBLE,
    RIGHT_WRIST_UPRIGHT_QUAT_WXYZ,
    apply_model_camera_overrides,
    profile_report,
)


class SortingRollRealSenseProfileTest(unittest.TestCase):
    def test_candidate_has_one_global_and_two_distinct_wrist_roles(self):
        self.assertEqual(
            set(CAMERA_ROLES.values()),
            {"global", "left_wrist", "right_wrist"},
        )
        self.assertEqual(len(MODEL_CAMERA_SOURCES), 3)
        self.assertEqual(len(set(MODEL_CAMERA_SOURCES.values())), 3)
        self.assertEqual(
            MODEL_CAMERA_SOURCES["left_wrist_realsense"],
            "hand_left_shelf",
        )
        self.assertEqual(
            MODEL_CAMERA_SOURCES["right_wrist_realsense"],
            "hand_right",
        )

    def test_policy_slots_keep_real_wrist_semantics(self):
        self.assertEqual(
            POLICY_IMAGE_MAP["observation/left_wrist_image"],
            "observation.images.left_wrist_realsense",
        )
        self.assertEqual(
            POLICY_IMAGE_MAP["observation/right_wrist_image"],
            "observation.images.right_wrist_realsense",
        )

    def test_unverified_simulation_candidate_is_not_training_eligible(self):
        report = profile_report()
        self.assertEqual(report["profile"], "sorting_roll_d405_candidate_v2")
        self.assertTrue(report["passed"])
        self.assertFalse(HARDWARE_VERIFIED)
        self.assertFalse(TRAINING_ELIGIBLE)

    def test_d405_nominal_rgb_contract_is_explicit(self):
        self.assertEqual(D405_MODEL, "RealSense D405")
        self.assertEqual(D405_RGB_RESOLUTION_WH, (1280, 720))
        self.assertEqual(D405_RGB_FPS, 30)
        self.assertEqual(D405_FOV_DEG, (87.0, 58.0))
        self.assertEqual(D405_IDEAL_RANGE_M, (0.07, 0.50))
        self.assertEqual(D405_SHUTTER, "global")
        self.assertFalse(D405_DEPTH_POLICY_INPUT)
        for wrist in ("left_wrist_realsense", "right_wrist_realsense"):
            self.assertEqual(
                MODEL_CAMERA_OVERRIDES[wrist]["fovy_deg"],
                D405_FOV_DEG[1],
            )
        self.assertNotIn(
            "quat_wxyz",
            MODEL_CAMERA_OVERRIDES["left_wrist_realsense"],
        )
        self.assertEqual(
            MODEL_CAMERA_OVERRIDES["right_wrist_realsense"]["quat_wxyz"],
            RIGHT_WRIST_UPRIGHT_QUAT_WXYZ,
        )

    def test_right_override_preserves_left_and_corrects_image_roll(self):
        class CameraObject:
            mjOBJ_CAMERA = 0

        class MujocoStub:
            mjtObj = CameraObject

            @staticmethod
            def mj_name2id(model, object_type, name):
                return model.camera_ids.get(name, -1)

        class ModelStub:
            camera_ids = {
                "hand_left_shelf": 0,
                "hand_right": 1,
                "stereo_left": 2,
            }
            cam_quat = np.asarray([
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ])
            cam_fovy = np.asarray([75.0, 75.0, 70.0])

        model = ModelStub()
        left_quat = model.cam_quat[0].copy()
        apply_model_camera_overrides(MujocoStub(), model)
        np.testing.assert_array_equal(model.cam_quat[0], left_quat)
        np.testing.assert_allclose(
            model.cam_quat[1],
            RIGHT_WRIST_UPRIGHT_QUAT_WXYZ,
        )
        self.assertEqual(model.cam_fovy[0], 58.0)
        self.assertEqual(model.cam_fovy[1], 58.0)


if __name__ == "__main__":
    unittest.main()
