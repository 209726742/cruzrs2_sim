#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from action_quality import analyze_motion


class ActionQualityTest(unittest.TestCase):
    def analyze(self, state, action, *, enforce_tracking=True):
        return analyze_motion(
            state,
            action,
            np.zeros((len(action), 2), dtype=np.float32),
            fps=30,
            phases=np.full(len(action), "test_phase"),
            joint_names=[f"joint_{index}" for index in range(16)],
            action_delta_limit=0.15,
            tracking_p95_limit=0.03,
            tracking_max_limit=0.15,
            terminal_tracking_limit=0.05,
            enforce_tracking=enforce_tracking,
        )

    def test_aligned_state_action_passes_hard_gate(self):
        action = np.zeros((100, 16), dtype=np.float32)
        result = self.analyze(action.copy(), action)
        self.assertTrue(result["passed"])
        self.assertTrue(result["tracking_passed"])
        self.assertTrue(result["tracking_enforced"])

    def test_terminal_tracking_mismatch_fails_hard_gate(self):
        action = np.zeros((100, 16), dtype=np.float32)
        state = action.copy()
        state[-1, 0] = 2.0
        result = self.analyze(state, action)
        self.assertFalse(result["passed"])
        self.assertFalse(result["tracking_passed"])
        self.assertTrue(any("terminal tracking" in failure for failure in result["failures"]))
        self.assertEqual(result["tracking_error_rad"]["max_phase"], "test_phase")
        self.assertEqual(result["tracking_error_rad"]["max_joint"], "joint_0")

    def test_tracking_can_only_be_warning_when_explicitly_disabled(self):
        action = np.full((100, 16), 2.0, dtype=np.float32)
        state = np.zeros_like(action)
        result = self.analyze(state, action, enforce_tracking=False)
        self.assertTrue(result["passed"])
        self.assertFalse(result["tracking_passed"])
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
