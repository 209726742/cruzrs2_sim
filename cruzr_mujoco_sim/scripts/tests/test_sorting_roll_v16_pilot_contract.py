#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_v16_pilot_contract import (  # noqa: E402
    FAMILY_COUNTS,
    PILOT_COUNT,
    SPLIT_COUNTS,
    assignment_for_seed,
    generate_pilot_manifest,
    manifest_errors,
)


class SortingRollV16PilotContractTest(unittest.TestCase):
    def setUp(self):
        self.manifest = generate_pilot_manifest("v16_test")

    def test_pilot_has_required_family_and_split_counts(self):
        self.assertEqual(self.manifest["count"], PILOT_COUNT)
        self.assertEqual(
            self.manifest["counts"]["scenario_family"], FAMILY_COUNTS
        )
        self.assertEqual(self.manifest["counts"]["split"], SPLIT_COUNTS)
        self.assertEqual(manifest_errors(self.manifest), [])

    def test_recovery_has_no_bad_action_prefix(self):
        recoveries = [
            item for item in self.manifest["assignments"]
            if item["scenario_family"] == "R"
        ]
        self.assertEqual(len(recoveries), 8)
        self.assertTrue(all(item["intervention_frame"] == -1 for item in recoveries))
        self.assertTrue(all(item["recovery_start_frame"] == 0 for item in recoveries))

    def test_all_families_continue_to_task_success(self):
        self.assertTrue(all(
            item["terminal_phase"] == "terminal_success_hold"
            for item in self.manifest["assignments"]
        ))

    def test_high_risk_stage_scenarios_are_forced(self):
        stages = [
            item for item in self.manifest["assignments"]
            if item["scenario_family"] == "T"
        ]
        for item in stages:
            base = item["base_diversity_assignment"]
            self.assertEqual(base["object_profile"]["name"], "long_baseline")
            self.assertEqual(base["dynamics_profile"]["name"], "heavy_low_friction")
            self.assertEqual(base["pose_bin"], "boundary")

    def test_assignment_lookup_is_exact(self):
        assignment = assignment_for_seed(self.manifest, 5000)
        self.assertEqual(assignment["seed"], 5000)
        with self.assertRaises(ValueError):
            assignment_for_seed(self.manifest, 4999)


if __name__ == "__main__":
    unittest.main()
