#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_batch import result_errors


def valid_result():
    return {
        "success": True,
        "simulation_canary_eligible": True,
        "sim_seconds": 59.9,
        "final_evidence": {
            "instantaneous_success": True,
            "stable_seconds": 2.0,
        },
        "gates": {
            "episode_under_one_minute": {"passed": True},
        },
    }


class SortingRollBatchTest(unittest.TestCase):
    def test_result_requires_physics_stability_and_one_minute(self):
        self.assertEqual(result_errors(valid_result()), [])

    def test_result_rejects_short_stability_window(self):
        result = valid_result()
        result["final_evidence"]["stable_seconds"] = 1.99
        self.assertIn(
            "final stable window is shorter than 2 seconds",
            result_errors(result),
        )

    def test_result_rejects_over_one_minute(self):
        result = valid_result()
        result["sim_seconds"] = 60.01
        result["gates"]["episode_under_one_minute"]["passed"] = False
        errors = result_errors(result)
        self.assertIn("sim_seconds exceeds 60 seconds", errors)
        self.assertIn("one-minute gate did not pass", errors)


if __name__ == "__main__":
    unittest.main()
