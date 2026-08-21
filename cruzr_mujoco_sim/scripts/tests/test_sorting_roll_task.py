import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from sorting_roll_task import (
    INSTANTANEOUS_CHECKS,
    SLOT_FLOOR_GAP_TOLERANCE_M,
    SortingRollSuccessTracker,
    axis_alignment_degrees,
    fit_report,
)


class SortingRollTaskTest(unittest.TestCase):
    def test_roll_and_visual_geometry_have_positive_slot_clearance(self):
        report = fit_report()
        self.assertTrue(report["simulation_fits"], report)
        self.assertAlmostEqual(report["length_clearance_total_m"], 0.070)
        self.assertAlmostEqual(report["length_clearance_each_end_m"], 0.035)
        self.assertAlmostEqual(report["simulated_slot_clearance_m"], 0.006)
        self.assertAlmostEqual(report["physical_nominal_slot_clearance_m"], 0.005)
        self.assertAlmostEqual(report["middle_bar_vertical_clearance_m"], 0.095)
        self.assertTrue(
            report["checks"]["physical_nominal_has_positive_slot_clearance"]
        )
    def test_success_contract_requires_physical_slot_floor_contact(self):
        self.assertIn(
            "resting_on_slot_floor_geometry",
            INSTANTANEOUS_CHECKS,
        )
        self.assertAlmostEqual(SLOT_FLOOR_GAP_TOLERANCE_M, 0.002)


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
