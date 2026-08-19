import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from sorting_roll_task import (
    SortingRollSuccessTracker,
    axis_alignment_degrees,
)


class SortingRollTaskTest(unittest.TestCase):
    def test_axis_alignment_accepts_both_slot_directions(self):
        self.assertAlmostEqual(axis_alignment_degrees([0, 1, 0]), 0.0)
        self.assertAlmostEqual(axis_alignment_degrees([0, -1, 0]), 0.0)
        self.assertAlmostEqual(axis_alignment_degrees([1, 0, 0]), 90.0)

    def test_success_requires_full_stable_window(self):
        tracker = SortingRollSuccessTracker(required_seconds=0.5)
        evidence = {"instantaneous_success": True}
        for _ in range(49):
            self.assertFalse(tracker.update(evidence, 0.01))
        self.assertTrue(tracker.update(evidence, 0.01))

    def test_failed_frame_resets_stable_window(self):
        tracker = SortingRollSuccessTracker(required_seconds=0.5)
        for _ in range(40):
            tracker.update({"instantaneous_success": True}, 0.01)
        self.assertFalse(
            tracker.update({"instantaneous_success": False}, 0.01)
        )
        self.assertEqual(tracker.stable_seconds, 0.0)

    def test_axis_input_is_not_mutated(self):
        axis = np.array([0.0, 2.0, 0.0])
        axis_alignment_degrees(axis)
        np.testing.assert_array_equal(axis, [0.0, 2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
