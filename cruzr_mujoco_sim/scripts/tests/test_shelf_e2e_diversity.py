#!/usr/bin/env python3
"""Tests for labeled clean/recovery source metadata."""

import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_source import diversity_errors  # noqa: E402


def payload(mode, requested, events, layout_mode="random", boundary_axis=None):
    rack_y = 0.20 if layout_mode == "boundary" and boundary_axis == "rack_y" else 0.0
    return {
        "schema_version": 1,
        "mode": mode,
        "scene_randomization": {
            "layout_mode": layout_mode,
            "boundary_axis": boundary_axis,
            "cart_offset_xy_m": [0.0, 0.0],
            "rack_y_offset_m": rack_y,
            "robot_initial_xyyaw": [0.0, 0.0, 0.0],
        },
        "perturbation_type": (
            "none" if mode == "clean"
            else "controlled_empty_navigation_base_pose_shift"
        ),
        "requested_event_count": requested,
        "actual_event_count": len(events),
        "events": events,
    }


class ShelfE2EDiversityTest(unittest.TestCase):
    def validate(self, diversity):
        meta = {"episode_metadata": {"diversity": diversity}}
        result = {"diversity": copy.deepcopy(diversity)}
        return diversity_errors(meta, result)

    def test_clean_has_no_perturbation(self):
        mode, stored, errors = self.validate(payload("clean", 0, []))
        self.assertEqual(mode, "clean")
        self.assertIsNotNone(stored)
        self.assertEqual(errors, [])

    def test_recovery_requires_and_validates_actual_event(self):
        event = {
            "trigger": "controlled_empty_navigation_entry",
            "phase": "pillar_navigate_to_grasp",
            "recorded_frame": 0,
            "sim_time_s": 0.5,
            "base_pose_delta": {"x_m": 0.02, "y_m": -0.01, "yaw_rad": 0.03},
        }
        mode, _, errors = self.validate(payload("recovery", 1, [event]))
        self.assertEqual(mode, "recovery")
        self.assertEqual(errors, [])

        _, _, missing_errors = self.validate(payload("recovery", 1, []))
        self.assertTrue(any("actual event" in error for error in missing_errors))

    def test_legacy_source_remains_readable_but_is_labeled(self):
        mode, stored, errors = diversity_errors(
            {"episode_metadata": {}}, {}
        )
        self.assertEqual(mode, "legacy_unlabeled")
        self.assertIsNone(stored)
        self.assertEqual(errors, [])

    def test_result_and_meta_must_match(self):
        diversity = payload("clean", 0, [])
        meta = {"episode_metadata": {"diversity": diversity}}
        _, _, errors = diversity_errors(meta, {"diversity": {}})
        self.assertIn("result/meta diversity mismatch", errors)

    def test_boundary_layout_requires_a_known_single_axis(self):
        _, _, errors = self.validate(
            payload("clean", 0, [], layout_mode="boundary", boundary_axis="rack_y")
        )
        self.assertEqual(errors, [])
        _, _, bad_errors = self.validate(
            payload("clean", 0, [], layout_mode="boundary", boundary_axis="all_axes")
        )
        self.assertTrue(any("boundary axis" in error for error in bad_errors))


if __name__ == "__main__":
    unittest.main()
