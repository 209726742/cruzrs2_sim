#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(COLLECTION_DIR))

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
    LEFT_WRIST_D405_OPTICAL_POS_M,
    LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
    MODEL_CAMERA_OVERRIDES,
    MODEL_CAMERA_SOURCES,
    POLICY_IMAGE_MAP,
    RIGHT_WRIST_D405_OPTICAL_POS_M,
    RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ,
    TRAINING_ELIGIBLE,
    WRIST_D405_MOUNT_GEOMS,
    apply_model_camera_overrides,
    profile_report,
    wrist_camera_initialization_report,
)
from sorting_roll_scene import materialize_scene
from sorting_roll_expert import SORTING_ROLL_INITIAL_ARM_PARK


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
            "sorting_roll_left_wrist_d405",
        )
        self.assertEqual(
            MODEL_CAMERA_SOURCES["right_wrist_realsense"],
            "sorting_roll_right_wrist_d405",
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
        self.assertEqual(report["profile"], "sorting_roll_d405_candidate_v6")
        self.assertTrue(report["passed"])
        self.assertFalse(HARDWARE_VERIFIED)
        self.assertFalse(TRAINING_ELIGIBLE)

    def test_d405_nominal_rgb_contract_is_explicit(self):
        self.assertEqual(D405_MODEL, "RealSense D405")
        self.assertEqual(
            LEFT_WRIST_D405_OPTICAL_POS_M, (0.0, -0.180, 0.070)
        )
        self.assertEqual(
            RIGHT_WRIST_D405_OPTICAL_POS_M,
            LEFT_WRIST_D405_OPTICAL_POS_M,
        )
        self.assertEqual(
            LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            (0.5, 0.8660254, 0.0, 0.0),
        )
        self.assertEqual(
            RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
        )
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
        expected = {
            "left_wrist_realsense": (
                LEFT_WRIST_D405_OPTICAL_POS_M,
                LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            ),
            "right_wrist_realsense": (
                RIGHT_WRIST_D405_OPTICAL_POS_M,
                RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            ),
        }
        for wrist, (position, quaternion) in expected.items():
            self.assertEqual(
                MODEL_CAMERA_OVERRIDES[wrist]["pos_m"], position
            )
            self.assertEqual(
                MODEL_CAMERA_OVERRIDES[wrist]["quat_wxyz"], quaternion
            )

    def test_symmetric_mounts_attach_to_the_gripper_mounts(self):
        import mujoco

        scene = Path(__file__).resolve().parents[2] / "assets" / "sorting_roll_scene.xml"
        materialize_scene(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        apply_model_camera_overrides(mujoco, model)
        expected = {
            "left_wrist_realsense": (
                "L_pgc140_mount",
                LEFT_WRIST_D405_OPTICAL_POS_M,
                LEFT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            ),
            "right_wrist_realsense": (
                "R_pgc140_mount",
                RIGHT_WRIST_D405_OPTICAL_POS_M,
                RIGHT_WRIST_D405_OPTICAL_QUAT_WXYZ,
            ),
        }
        for logical, (expected_parent, position, quaternion) in expected.items():
            camera_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                MODEL_CAMERA_SOURCES[logical],
            )
            self.assertGreaterEqual(camera_id, 0)
            self.assertEqual(
                model.body(int(model.cam_bodyid[camera_id])).name,
                expected_parent,
            )
            np.testing.assert_allclose(
                model.cam_pos[camera_id], position
            )
            np.testing.assert_allclose(
                model.cam_quat[camera_id], quaternion
            )
            for name in WRIST_D405_MOUNT_GEOMS[logical]["visual"]:
                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                self.assertEqual(int(model.geom_group[geom_id]), 1)
                self.assertEqual(float(model.geom_rgba[geom_id, 3]), 1.0)
            for name in WRIST_D405_MOUNT_GEOMS[logical]["collision"]:
                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                self.assertEqual(int(model.geom_contype[geom_id]), 1)
                self.assertEqual(int(model.geom_conaffinity[geom_id]), 1)


    def test_task_ready_park_places_d405_views_forward_and_mirrored(self):
        import mujoco

        scene = Path(__file__).resolve().parents[2] / "assets" / "sorting_roll_scene.xml"
        materialize_scene(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        for side in ("L", "R"):
            for joint, value in zip(
                (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow_roll",
                    "elbow_yaw",
                    "wrist_pitch",
                    "wrist_roll",
                ),
                SORTING_ROLL_INITIAL_ARM_PARK[side.lower()],
            ):
                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{joint}_joint",
                )
                data.qpos[model.jnt_qposadr[joint_id]] = value
        apply_model_camera_overrides(mujoco, model)
        report = wrist_camera_initialization_report(
            mujoco, model, data
        )
        self.assertTrue(report["passed"], report)
        for position in report["positions_base_m"].values():
            self.assertGreater(position[0], 0.25)
        self.assertLessEqual(
            report["camera_sagittal_position_error_mm"], 12.0
        )
        self.assertLessEqual(
            report["camera_optical_forward_mirror_error_deg"], 1.0
        )
        self.assertLessEqual(
            report["camera_optical_up_mirror_error_deg"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
