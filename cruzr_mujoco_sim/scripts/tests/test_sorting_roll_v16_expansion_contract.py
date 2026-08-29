#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import unittest


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_v16_expansion_contract import (  # noqa: E402
    STAGE_COUNTS,
    assignment_for_seed,
    generate_expansion_manifest,
    manifest_errors,
    representative_seeds,
)


class SortingRollV16ExpansionContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = generate_expansion_manifest("stage80_test")

    def test_stage80_has_exact_family_and_split_counts(self):
        self.assertEqual(self.manifest["count"], 80)
        self.assertEqual(
            self.manifest["counts"]["scenario_family"],
            STAGE_COUNTS[80]["family"],
        )
        self.assertEqual(
            self.manifest["counts"]["split"],
            STAGE_COUNTS[80]["split"],
        )
        self.assertEqual(manifest_errors(self.manifest), [])

    def test_counterfactual_pairs_are_grouped_with_fixed_scene(self):
        counterfactuals = [
            item for item in self.manifest["assignments"]
            if item["scenario_family"] == "C"
        ]
        groups = {}
        for item in counterfactuals:
            groups.setdefault(item["scene_group_id"], []).append(item)
        self.assertEqual(len(groups), 6)
        for members in groups.values():
            self.assertEqual(len(members), 2)
            self.assertEqual({item["target_lane"] for item in members}, {"left", "right"})
            self.assertEqual(len({item["split"] for item in members}), 1)
            self.assertEqual(
                len({
                    item["counterfactual_scene"]["scene_randomization_seed"]
                    for item in members
                }),
                1,
            )
            self.assertEqual(
                len({
                    tuple(sorted(item["counterfactual_scene"]["lane_colors"].items()))
                    for item in members
                }),
                1,
            )

    def test_broken_pair_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        item = next(
            assignment for assignment in broken["assignments"]
            if assignment["scenario_family"] == "C"
        )
        item["counterfactual_scene"]["scene_randomization_seed"] += 1
        item.pop("assignment_id")
        errors = manifest_errors(broken)
        self.assertTrue(any("scene seeds differ" in error for error in errors))

    def test_representative_selection_includes_one_complete_c_pair(self):
        seeds = representative_seeds(self.manifest)
        self.assertEqual(len(seeds), 5)
        assignments = [assignment_for_seed(self.manifest, seed) for seed in seeds]
        self.assertEqual(
            [item["scenario_family"] for item in assignments[:3]],
            ["H", "T", "R"],
        )
        self.assertEqual(
            assignments[3]["counterfactual_pair_id"],
            assignments[4]["counterfactual_pair_id"],
        )

    def test_stage160_is_supported_by_same_contract(self):
        manifest = generate_expansion_manifest(
            "stage160_test", stage=160, seed_start=7000
        )
        self.assertEqual(manifest["counts"]["scenario_family"], STAGE_COUNTS[160]["family"])
        self.assertEqual(manifest_errors(manifest), [])


if __name__ == "__main__":
    unittest.main()
