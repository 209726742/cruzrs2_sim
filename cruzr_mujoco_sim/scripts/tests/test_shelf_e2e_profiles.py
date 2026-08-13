#!/usr/bin/env python3
"""Tests for isolated strict and Cruzr S2 SDK collection profiles."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from cruzr_s2_sdk_contract import SDK_COLLECTION_PROFILE  # noqa: E402
from shelf_e2e_profiles import (  # noqa: E402
    STRICT_COLLECTION_PROFILE,
    collection_cameras,
    normalize_collection_profile,
    policy_image_map,
)


class ShelfE2EProfilesTest(unittest.TestCase):
    def test_missing_profile_preserves_legacy_strict_contract(self):
        self.assertEqual(normalize_collection_profile(None), STRICT_COLLECTION_PROFILE)
        self.assertEqual(
            collection_cameras(None),
            ("head_stereo_l_shelf", "chassis_front", "hand_right_shelf"),
        )

    def test_sdk_profile_uses_only_real_sdk_camera_counterparts(self):
        self.assertEqual(
            collection_cameras(SDK_COLLECTION_PROFILE),
            ("stereo_left", "waist_front", "chassis_front"),
        )
        self.assertEqual(
            tuple(policy_image_map(SDK_COLLECTION_PROFILE)),
            (
                "observation/image",
                "observation/left_wrist_image",
                "observation/right_wrist_image",
            ),
        )

    def test_returned_mapping_cannot_mutate_the_profile(self):
        mapping = policy_image_map(SDK_COLLECTION_PROFILE)
        mapping["observation/image"] = "bad"
        self.assertNotEqual(
            policy_image_map(SDK_COLLECTION_PROFILE)["observation/image"], "bad"
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported collection profile"):
            normalize_collection_profile("sdk_latest")


if __name__ == "__main__":
    unittest.main()
