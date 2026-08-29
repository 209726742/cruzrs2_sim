from pathlib import Path
import sys
import unittest

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

import sorting_roll_v16_sampling_weights as sampling  # noqa: E402


class SortingRollV16SamplingWeightsTests(unittest.TestCase):
    def test_builds_requested_mass_by_family(self):
        rows = [
            {
                "episode_index": 0,
                "source_split": "train",
                "source_task_version": sampling.V15_TASK_VERSION,
                "source_scenario": None,
            },
            *[
                {
                    "episode_index": index,
                    "source_split": "train",
                    "source_task_version": "sorting_roll_v16_expansion_pilot_sim",
                    "source_scenario": {"scenario_family": family},
                }
                for index, family in enumerate(("H", "T", "R"), start=1)
            ],
        ]
        episode_indices = np.asarray([0, 0, 1, 2, 2, 2, 3], dtype=np.int64)

        weights, report = sampling.build_frame_weights(episode_indices, rows)

        self.assertEqual(len(weights), len(episode_indices))
        for family, target in sampling.TARGET_FRACTIONS.items():
            self.assertAlmostEqual(report["expected_sampling_mass"][family], target)

    def test_rejects_validation_episode(self):
        rows = [{
            "episode_index": 0,
            "source_split": "val",
            "source_task_version": sampling.V15_TASK_VERSION,
            "source_scenario": None,
        }]
        with self.assertRaisesRegex(ValueError, "train episodes"):
            sampling.build_frame_weights(np.asarray([0]), rows)


    def test_full_v2_profile_is_conservative(self):
        self.assertEqual(
            sampling.SAMPLING_PROFILES["full_v2_old70"],
            {"old": 0.70, "H": 0.10, "T": 0.10, "R": 0.10},
        )

    def test_stage80_profile_includes_counterfactual_family(self):
        profile = sampling.SAMPLING_PROFILES["stage80_old50"]
        self.assertEqual(
            profile,
            {"old": 0.50, "H": 0.15, "T": 0.15, "R": 0.15, "C": 0.05},
        )
        rows = [
            {
                "episode_index": index,
                "source_split": "train",
                "source_task_version": (
                    sampling.V15_TASK_VERSION if family == "old" else "v16"
                ),
                "source_scenario": (
                    None if family == "old" else {"scenario_family": family}
                ),
            }
            for index, family in enumerate(profile)
        ]
        episode_indices = np.arange(len(rows), dtype=np.int64)
        _, report = sampling.build_frame_weights(
            episode_indices,
            rows,
            profile,
        )
        self.assertEqual(report["expected_sampling_mass"], profile)


if __name__ == "__main__":
    unittest.main()
