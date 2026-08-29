#!/usr/bin/env python3

import subprocess
from pathlib import Path
import unittest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = SCRIPTS_ROOT / "training" / "pi05_sorting_roll_v16_h100x4_fullft20h.sh"
AUDITOR = SCRIPTS_ROOT / "training" / "sorting_roll_v15_fullft_canary_audit.py"


class Pi05SortingRollV16H100x4Fullft20hTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.audit_source = AUDITOR.read_text(encoding="utf-8")

    def test_launcher_has_valid_shell_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_data_gate_matches_admitted_stage80_dataset(self):
        self.assertIn("sorting_roll_v16_stage80_mixed320_20260829", self.source)
        for setting in (
            'info["total_episodes"] == info["total_source_episodes"] == 320',
            'info["splits"] == {"train": "0:304", "val": "304:312", "test": "312:320"}',
            'audit["v16_family_counts"] == {"C": 12, "H": 20, "R": 28, "T": 20}',
            'readiness["ready_for_full_parameter_canary"] is True',
            'sampling["passed"] is True',
            'sampling["target_fractions"] == {',
            '"old": 0.50',
            '"C": 0.05',
            'observation.images.left_wrist_realsense',
            'observation.images.right_wrist_realsense',
            'features[key]["shape"] == [18]',
        ):
            self.assertIn(setting, self.source)

    def test_formal_run_uses_mature_checkpoint_with_fresh_full_ft(self):
        for setting in (
            "checkpoints/036000/pretrained_model",
            "--frame-sampling-weights",
            "--use-pretrained-stats true",
            "GPU_IDS=0,1,2,3",
            "NUM_PROCESSES=4",
            "BATCH_SIZE=${BATCH_SIZE:-16}",
            "TARGET_STEPS=${TARGET_STEPS:-28000}",
            "LEARNING_RATE=${LEARNING_RATE:-1e-5}",
            "--min-effective-batch 64",
            "--dtype bfloat16",
            "--gradient-checkpointing true",
            "--train-expert-only false",
        ):
            self.assertIn(setting, self.source)

    def test_fresh_resume_canary_and_tmux_are_hard_gates(self):
        for action in (
            "tmux-canary",
            "tmux-canary-resume",
            "canary-audit",
            "tmux-start",
            "tmux-resume",
        ):
            self.assertIn(action, self.source)
        self.assertIn("canary_args 200 200", self.source)
        self.assertIn("canary_args 250 50", self.source)
        self.assertIn("formal_preflight", self.source)
        self.assertIn('report["full_parameter_count_verified"] is True', self.source)

    def test_shared_canary_auditor_accepts_v16_contract_parameters(self):
        for argument in (
            '"--task-version"',
            '"--expected-learning-rate"',
        ):
            self.assertIn(argument, self.audit_source)
        self.assertIn('"task_version": args.task_version', self.audit_source)
        self.assertIn('"optimizer_lr": args.expected_learning_rate', self.audit_source)


if __name__ == "__main__":
    unittest.main()
