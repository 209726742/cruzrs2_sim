#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]

from sorting_roll_build_v21 import (  # noqa: E402
    ACTION_NAMES,
    STATE_NAMES,
    VIDEO_FILTER,
    parse_args,
    policy_state_action,
    sort_sources,
)


class SortingRollBuildV21Test(unittest.TestCase):
    def test_policy_state_action_builds_deployable_18_dimensions(self):
        payload = {
            "state": np.zeros((4, 16), dtype=np.float32),
            "base_velocity": np.ones((4, 2), dtype=np.float32),
            "action": np.zeros((4, 16), dtype=np.float32),
            "base_action": np.full((4, 2), 2.0, dtype=np.float32),
        }
        state, action = policy_state_action(payload)
        self.assertEqual(state.shape, (4, len(STATE_NAMES)))
        self.assertEqual(action.shape, (4, len(ACTION_NAMES)))
        np.testing.assert_array_equal(state[:, -2:], 1.0)
        np.testing.assert_array_equal(action[:, -2:], 2.0)

    def test_sources_are_grouped_by_split_without_seed_leakage(self):
        sources = [
            {"path": "/200", "seed": 200, "split": "test"},
            {"path": "/202", "seed": 202, "split": "train"},
            {"path": "/201", "seed": 201, "split": "val"},
            {"path": "/203", "seed": 203, "split": "train"},
        ]
        ordered = sort_sources(sources)
        self.assertEqual([item["seed"] for item in ordered], [202, 203, 201, 200])

    def test_video_resize_preserves_aspect_ratio(self):
        self.assertIn("force_original_aspect_ratio=decrease", VIDEO_FILTER)
        self.assertIn("pad=224:224", VIDEO_FILTER)

    def test_builder_accepts_explicit_campaign_manifest(self):
        args = parse_args([
            "source", "--out", "dataset", "--manifest", "campaign.json",
        ])
        self.assertEqual(args.manifest, Path("campaign.json"))


if __name__ == "__main__":
    unittest.main()
