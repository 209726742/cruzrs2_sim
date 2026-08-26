#!/usr/bin/env python3

import subprocess
from pathlib import Path
import unittest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = SCRIPTS_ROOT / "training" / "pi05_sorting_roll_v15_h100x4_fullft20h.sh"
AUDITOR = SCRIPTS_ROOT / "training" / "sorting_roll_v15_fullft_canary_audit.py"
DATASET_AUDITOR = SCRIPTS_ROOT / "training" / "sorting_roll_v15_dataset_audit.py"


class Pi05SortingRollV15H100x4Fullft20hTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.audit_source = AUDITOR.read_text(encoding="utf-8")
        cls.dataset_audit_source = DATASET_AUDITOR.read_text(encoding="utf-8")

    def test_launcher_has_valid_shell_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_hardware_gate_requires_four_full_h100s(self):
        self.assertIn("GPU_IDS=0,1,2,3", self.source)
        self.assertIn("NUM_PROCESSES=4", self.source)
        self.assertIn("assert len(rows) == 4", self.source)
        self.assertIn('assert "H100" in name', self.source)
        self.assertIn("assert int(memory) >= 80000", self.source)
        self.assertIn('assert mig.lower() == "disabled"', self.source)

    def test_formal_run_is_full_parameter_and_sized_for_twenty_hours(self):
        for setting in (
            "BATCH_SIZE=${BATCH_SIZE:-16}",
            "NUM_WORKERS=${NUM_WORKERS:-8}",
            "TARGET_STEPS=${TARGET_STEPS:-28000}",
            "SAVE_FREQ=${SAVE_FREQ:-1000}",
            "LEARNING_RATE=${LEARNING_RATE:-2.5e-5}",
            "WARMUP_STEPS=${WARMUP_STEPS:-1000}",
            "--min-effective-batch 64",
            "--dtype bfloat16",
            "--gradient-checkpointing true",
            "--train-expert-only false",
            "pi05_sorting_roll_v15_h100x4_fullft28k",
        ):
            self.assertIn(setting, self.source)

    def test_data_and_canary_gates_precede_formal_training(self):
        self.assertIn('info["source_task_version"] == "sorting_roll_v15_diverse_sim"', self.source)
        self.assertIn('info["total_episodes"] == info["total_source_episodes"] == 300', self.source)
        self.assertIn('info["total_tasks"] == 5', self.source)
        self.assertIn("frame task_index does not match episode prompt", self.dataset_audit_source)
        self.assertIn("canary_args 200 200", self.source)
        self.assertIn("canary_args 250 50", self.source)
        self.assertIn("formal_preflight", self.source)
        self.assertIn("EXPECTED_STEPS = (200, 250)", self.audit_source)
        self.assertIn("EXPECTED_PARAMETER_COUNT = 4_143_404_816", self.audit_source)
        self.assertIn('"freeze_vision_encoder": False', self.audit_source)
        self.assertIn('"train_expert_only": False', self.audit_source)
        self.assertIn('report["full_parameter_count_verified"] is True', self.source)

    def test_tmux_and_resume_entrypoints_are_available(self):
        for action in (
            "tmux-canary",
            "tmux-canary-resume",
            "tmux-start",
            "tmux-resume",
        ):
            self.assertIn(action, self.source)


if __name__ == "__main__":
    unittest.main()
