#!/usr/bin/env python3

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "pi05_sorting_roll_v13_h100x2_train.sh"
)


class Pi05SortingRollV13H100x2TrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_hardware_gate_requires_two_full_h100s(self):
        self.assertIn('nvidia-smi -i 0,1', self.source)
        self.assertIn('assert "H100" in name', self.source)
        self.assertIn('assert int(memory) >= 80000', self.source)
        self.assertIn('assert mig.lower() == "disabled"', self.source)

    def test_canary_proves_fresh_and_resume_paths(self):
        self.assertIn("TARGET_STEPS=${1:-200}", self.source)
        self.assertIn("set_canary_env 250", self.source)
        self.assertIn("pi05_sorting_roll_v13_h100x2_canary", self.source)

    def test_formal_configuration_matches_approved_plan(self):
        for setting in (
            "GPU_IDS=0,1",
            "NUM_PROCESSES=2",
            "BATCH_SIZE=16",
            "H100_NUM_WORKERS=${H100_NUM_WORKERS:-8}",
            "NUM_WORKERS=$H100_NUM_WORKERS",
            "TARGET_STEPS=20000",
            "WARMUP_STEPS=1000",
            "SAVE_FREQ=1000",
        ):
            self.assertIn(setting, self.source)
        self.assertIn("h100x2_expert20k_seed1000", self.source)


if __name__ == "__main__":
    unittest.main()
