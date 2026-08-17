#!/usr/bin/env python3
import os
import sys
import unittest

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_build_v2 import episode_specs  # noqa: E402
from shelf_e2e_source import (  # noqa: E402
    quality_errors,
    require_single_collection_profile,
    require_single_task_version,
    require_unique_seeds,
    source_split,
)
from cruzr_s2_sdk_contract import SDK_CAMERAS, SDK_COLLECTION_PROFILE  # noqa: E402
from shelf_e2e_profiles import STRICT_COLLECTION_PROFILE  # noqa: E402


def valid_quality_records(num_frames=300):
    motion = {
        "passed": True,
        "tracking_passed": True,
        "tracking_enforced": True,
        "num_frames": num_frames,
    }
    endpoint = {
        "reason": "both_objects_released_and_stable",
        "recorded_frames": num_frames,
        "audit_frames": num_frames,
    }
    meta = {
        "success": True,
        "episode_metadata": {
            "validation": {"passed": True, "motion_quality": dict(motion)},
            "policy_episode_end": dict(endpoint),
        },
    }
    result = {
        "passed": True,
        "motion_quality": dict(motion),
        "policy_episode_end": dict(endpoint),
        "safety_home": {
            "recorded_in_policy_episode": False,
            "tracking_passed": True,
            "release_passed": True,
            "strip_contact_force_peak_n": 0.0,
            "objects_stable": True,
        },
    }
    return meta, result


class ShelfE2EBuildV2Test(unittest.TestCase):
    def test_split_is_determined_only_by_source_seed(self):
        self.assertEqual(source_split(1), "val")
        self.assertEqual(source_split(10), "test")
        self.assertEqual(source_split(2), "train")
        self.assertEqual(source_split(29), "train")

    def test_duplicate_source_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source seed 2"):
            require_unique_seeds([
                {"seed": 2, "path": "/a"},
                {"seed": 2, "path": "/b"},
            ])

    def test_rigid_and_flex_sources_cannot_be_mixed(self):
        rigid = {"task_version": "dual_two_trip_v1"}
        flex = {"task_version": "dual_two_trip_flex_v1"}
        self.assertEqual(require_single_task_version([rigid, rigid]), "dual_two_trip_v1")
        with self.assertRaisesRegex(ValueError, "must not be mixed"):
            require_single_task_version([rigid, flex])

    def test_strict_and_sdk_collection_profiles_cannot_be_mixed(self):
        strict = {
            "collection_profile": STRICT_COLLECTION_PROFILE,
            "cameras": [
                "head_stereo_l_shelf",
                "chassis_front",
                "hand_right_shelf",
            ],
        }
        sdk = {
            "collection_profile": SDK_COLLECTION_PROFILE,
            "cameras": list(SDK_CAMERAS),
        }
        self.assertEqual(
            require_single_collection_profile([strict, strict]),
            (STRICT_COLLECTION_PROFILE, tuple(strict["cameras"])),
        )
        with self.assertRaisesRegex(ValueError, "profiles must not be mixed"):
            require_single_collection_profile([strict, sdk])

    def test_each_output_spec_is_one_continuous_source_interval(self):
        n = 300
        action = np.zeros((n, 16), dtype=np.float32)
        action[:90, 0] = np.arange(90) * 0.01
        action[210:, 0] = np.arange(90) * 0.01
        base_action = np.zeros((n, 2), dtype=np.float32)
        base_action[20:70, 0] = 0.2
        base_action[230:280, 0] = 0.2
        specs = episode_specs({"action": action, "base_action": base_action})

        self.assertGreaterEqual(len(specs), 4)
        for variant, _, start, stop in specs:
            self.assertGreaterEqual(stop - start, 60)
            self.assertIsInstance(start, int)
            self.assertIsInstance(stop, int)
            if variant == "full":
                self.assertFalse(start < 100 and stop > 200)

    def test_latest_quality_endpoint_passes(self):
        meta, result = valid_quality_records()
        self.assertEqual(quality_errors(meta, result, 300), [])

    def test_soft_tracking_or_missing_endpoint_is_rejected(self):
        meta, result = valid_quality_records()
        result["motion_quality"]["tracking_enforced"] = False
        meta["episode_metadata"].pop("policy_episode_end")
        errors = quality_errors(meta, result, 300)
        self.assertTrue(any("tracking_enforced" in error for error in errors))
        self.assertTrue(any("policy_episode_end.reason" in error for error in errors))

    def test_terminal_hold_contact_or_tracking_failure_is_rejected(self):
        meta, result = valid_quality_records()
        result["safety_home"]["tracking_passed"] = False
        result["safety_home"]["strip_contact_force_peak_n"] = 0.201
        errors = quality_errors(meta, result, 300)
        self.assertIn("safety_home.tracking_passed is not true", errors)
        self.assertIn("safety_home.strip_contact_force_peak_n exceeds 0.2 N", errors)


if __name__ == "__main__":
    unittest.main()
