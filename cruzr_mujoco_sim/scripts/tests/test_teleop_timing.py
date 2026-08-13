#!/usr/bin/env python3
"""Tests for drift-free integer MuJoCo control substep scheduling."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from teleop_timing import CumulativeSubstepScheduler  # noqa: E402


class TeleopTimingTest(unittest.TestCase):
    def test_sixty_hz_on_one_ms_physics_has_no_cumulative_drift(self):
        scheduler = CumulativeSubstepScheduler(60, 0.001)
        steps = [scheduler.next_substeps() for _ in range(600)]
        self.assertEqual(steps[:6], [17, 16, 17, 17, 16, 17])
        self.assertEqual(scheduler.physics_steps, 10000)
        frame_steps = [sum(steps[index:index + 2]) for index in range(0, 6, 2)]
        self.assertEqual(frame_steps, [33, 34, 33])

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            CumulativeSubstepScheduler(0, 0.001)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            CumulativeSubstepScheduler(2000, 0.001)


if __name__ == "__main__":
    unittest.main()
