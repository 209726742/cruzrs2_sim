#!/usr/bin/env python3

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "pi05_sorting_roll_v13_train.sh"
)


class Pi05SortingRollV13TrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_launcher_is_pinned_to_audited_v13_dataset(self):
        self.assertIn("sorting_roll_v13_diverse300_lerobot_v30_20260825", self.source)
        self.assertIn('info["source_task_version"] == "sorting_roll_v13_diverse_sim"', self.source)
        self.assertIn('info["collection_profile"] == "sorting_roll_d405_candidate_v4"', self.source)
        self.assertIn('info["splits"] == {"train": "0:240", "val": "240:270", "test": "270:300"}', self.source)

    def test_launcher_preserves_pi05_data_contract(self):
        for key in (
            "observation.images.stereo_left",
            "observation.images.left_wrist_realsense",
            "observation.images.right_wrist_realsense",
            "observation.state",
            "action",
        ):
            self.assertIn(key, self.source)
        self.assertIn('features[key]["dtype"] == "float32"', self.source)
        self.assertIn('features[key]["shape"] == [18]', self.source)

    def test_launcher_uses_the_proven_four_gpu_configuration(self):
        self.assertIn("GPU_IDS=${GPU_IDS:-0,1,2,3}", self.source)
        self.assertIn("NUM_PROCESSES=${NUM_PROCESSES:-4}", self.source)
        self.assertIn("BATCH_SIZE=${BATCH_SIZE:-1}", self.source)
        self.assertIn("--allow-small-batch true", self.source)
        self.assertIn("--gradient-checkpointing true", self.source)
        self.assertIn("--train-expert-only true", self.source)


if __name__ == "__main__":
    unittest.main()
