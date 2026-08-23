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
    REQUIRED_STABLE_SECONDS,
    TOP_TIER_TROUGH_GAP_TOLERANCE_M,
    SortingRollSuccessTracker,
    axis_alignment_degrees,
    evaluate_placement,
    fit_report,
)

import sorting_roll_scene as scene


class SortingRollTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        cls.mujoco = mujoco
        scene.materialize_scene()
        cls.model = mujoco.MjModel.from_xml_path(str(scene.SCENE_PATH))
        cls.roll_joint = mujoco.mj_name2id(
            cls.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "sorting_roll_free",
        )

    @classmethod
    def placement_evidence(cls, position):
        data = cls.mujoco.MjData(cls.model)
        qpos_adr = int(cls.model.jnt_qposadr[cls.roll_joint])
        data.qpos[qpos_adr:qpos_adr + 3] = position
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [
            np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)
        ]
        cls.mujoco.mj_forward(cls.model, data)
        return evaluate_placement(cls.model, data)

    def test_roll_fits_integrated_top_tier(self):
        report = fit_report()
        self.assertTrue(report["simulation_fits"], report)
        self.assertAlmostEqual(report["length_clearance_total_m"], 0.070)
        self.assertAlmostEqual(report["length_clearance_each_end_m"], 0.035)
        self.assertAlmostEqual(report["roll_to_shelf_width_ratio"], 0.5 / 0.57)
        self.assertAlmostEqual(report["integrated_pocket_depth_m"], 0.1045)
        self.assertAlmostEqual(
            report["pocket_depth_clearance_m"],
            0.1045 - 0.025,
        )
        self.assertAlmostEqual(report["front_lip_rise_m"], 0.027)
        self.assertNotIn("slot_clear_width_m", report)
        self.assertTrue(
            report["checks"]["roll_length_ratio_is_80_to_90_percent"]
        )

    def test_success_contract_requires_integrated_multi_point_support(self):
        self.assertIn(
            "resting_on_integrated_top_tier_geometry",
            INSTANTANEOUS_CHECKS,
        )
        self.assertAlmostEqual(TOP_TIER_TROUGH_GAP_TOLERANCE_M, 0.004)

    def test_old_external_target_is_rejected(self):
        evidence = self.placement_evidence([0.7825, 0.0, 1.0125])
        self.assertFalse(
            evidence["checks"]["center_inside_integrated_top_tier"]
        )
        self.assertFalse(
            evidence["checks"]["supported_by_integrated_top_tier"]
        )
        self.assertFalse(evidence["instantaneous_success"])

    def test_floating_inside_shelf_is_rejected(self):
        evidence = self.placement_evidence([0.950, 0.0, 0.950])
        self.assertFalse(
            evidence["checks"]["supported_by_integrated_top_tier"]
        )
        self.assertFalse(
            evidence["checks"]["resting_on_integrated_top_tier_geometry"]
        )
        self.assertFalse(evidence["instantaneous_success"])


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

    def test_default_success_window_is_two_seconds(self):
        tracker = SortingRollSuccessTracker()
        self.assertEqual(REQUIRED_STABLE_SECONDS, 2.0)
        self.assertEqual(tracker.required_seconds, 2.0)

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
