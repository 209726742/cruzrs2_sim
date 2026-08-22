#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_realsense_profile import (
    CAMERA_ROLES,
    HARDWARE_VERIFIED,
    MODEL_CAMERA_OVERRIDES,
    MODEL_CAMERA_SOURCES,
    POLICY_IMAGE_MAP,
    TRAINING_ELIGIBLE,
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
        self.assertTrue(report["passed"])
        self.assertFalse(HARDWARE_VERIFIED)
        self.assertFalse(TRAINING_ELIGIBLE)

    def test_right_wrist_override_is_explicit_and_provisional(self):
        override = MODEL_CAMERA_OVERRIDES["right_wrist_realsense"]
        self.assertEqual(len(override["quat_wxyz"]), 4)
        self.assertEqual(override["fovy_deg"], 75.0)


if __name__ == "__main__":
    unittest.main()
