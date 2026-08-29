#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]

from sorting_roll_v16_expansion_contract import (  # noqa: E402
    generate_expansion_manifest,
)
from sorting_roll_v16_expansion_validate import (  # noqa: E402
    counterfactual_pair_errors,
)


class SortingRollV16ExpansionValidateTests(unittest.TestCase):
    def setUp(self):
        self.manifest = generate_expansion_manifest("expansion_validate_test")
        assignments = [
            item for item in self.manifest["assignments"]
            if item["scenario_family"] == "C"
        ]
        pair_id = assignments[0]["counterfactual_pair_id"]
        self.pair = [
            item for item in assignments
            if item["counterfactual_pair_id"] == pair_id
        ]

    def records(self, assignments):
        return [
            {
                "passed": True,
                "info": {"seed": item["seed"]},
                "errors": [],
            }
            for item in assignments
        ]

    def test_complete_counterfactual_pair_passes(self):
        self.assertEqual(
            counterfactual_pair_errors(
                self.manifest, self.records(self.pair)
            ),
            [],
        )

    def test_partial_counterfactual_pair_fails(self):
        errors = counterfactual_pair_errors(
            self.manifest, self.records(self.pair[:1])
        )
        self.assertTrue(any("incomplete" in error for error in errors))

    def test_pair_scene_drift_fails(self):
        manifest = copy.deepcopy(self.manifest)
        second = next(
            item for item in manifest["assignments"]
            if item["seed"] == self.pair[1]["seed"]
        )
        second["counterfactual_scene"]["lane_x_m"]["right"] -= 0.01
        errors = counterfactual_pair_errors(
            manifest, self.records(self.pair)
        )
        self.assertTrue(any("invariants" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
