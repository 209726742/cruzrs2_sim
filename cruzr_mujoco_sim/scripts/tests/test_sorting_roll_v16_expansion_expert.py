#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
SPEC = importlib.util.spec_from_file_location(
    "sorting_roll_v16_expansion_expert",
    COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SortingRollV16ExpansionExpertTests(unittest.TestCase):
    def test_c_release_accepts_only_bounded_pad_front_lip_grazing(self):
        measured = [{
            "pair": ["shelf_top_front_lip_col", "L_pad1"],
            "penetration_mm": 0.019,
            "normal_force_n": 0.08809,
        }]
        self.assertTrue(MODULE.c_release_contacts_are_incidental(measured))
        for field, value in (
            ("penetration_mm", 0.051),
            ("normal_force_n", 0.501),
        ):
            unsafe = [dict(measured[0], **{field: value})]
            self.assertFalse(
                MODULE.c_release_contacts_are_incidental(unsafe)
            )
        structural = [dict(
            measured[0],
            pair=["shelf_top_front_lip_col", "left_wrist_camera"],
        )]
        self.assertFalse(
            MODULE.c_release_contacts_are_incidental(structural)
        )

    def test_rigid_motion_metrics_ignore_quaternion_sign(self):
        initial = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        final = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        metrics = MODULE.rigid_motion_metrics(initial, final, np.zeros(6))
        self.assertAlmostEqual(metrics["translation_m"], 0.0)
        self.assertAlmostEqual(metrics["rotation_deg"], 0.0)
        self.assertAlmostEqual(metrics["speed"], 0.0)

    def test_rigid_motion_metrics_measure_translation(self):
        initial = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        final = np.array([0.006, 0.008, 1.0, 1.0, 0.0, 0.0, 0.0])
        metrics = MODULE.rigid_motion_metrics(initial, final, np.zeros(6))
        self.assertAlmostEqual(metrics["translation_m"], 0.010)

    def test_rigid_motion_metrics_ignore_cylinder_axial_spin(self):
        initial = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        half_sqrt = np.sqrt(0.5)
        axial = np.array([
            0.0, 0.0, 1.0, half_sqrt, half_sqrt, 0.0, 0.0
        ])
        transverse = np.array([
            0.0, 0.0, 1.0, half_sqrt, 0.0, 0.0, half_sqrt
        ])
        axial_metrics = MODULE.rigid_motion_metrics(
            initial, axial, np.zeros(6)
        )
        transverse_metrics = MODULE.rigid_motion_metrics(
            initial, transverse, np.zeros(6)
        )
        self.assertAlmostEqual(axial_metrics["rotation_deg"], 0.0)
        self.assertAlmostEqual(axial_metrics["raw_rotation_deg"], 90.0)
        self.assertAlmostEqual(transverse_metrics["rotation_deg"], 90.0)

    def test_c_release_target_keeps_forward_entry_and_camera_clearance(self):
        self.assertEqual(MODULE.V16_SHELF_STAGE_OFFSET_X_M, -0.075)
        self.assertEqual(MODULE.V16_RELEASE_CLEARANCE_ROLL_Z_M, 0.970)
        self.assertEqual(
            MODULE.V16_RELEASE_LATERAL_STEP_M,
            0.004,
        )
        self.assertEqual(MODULE.V16_RELEASE_LATERAL_MAX_STEPS, 3)
        self.assertEqual(MODULE.v15.SHELF_STAGE_OFFSET_X, -0.060)
        for family in ("H", "T", "R"):
            self.assertEqual(MODULE.shelf_stage_offset_x_m(family), -0.075)
        self.assertEqual(MODULE.shelf_stage_offset_x_m("C"), -0.060)
        self.assertEqual(MODULE.v15.RELEASE_CLEARANCE_ROLL_Z, 0.958)
        entry_start = MODULE.v15.TARGET_CENTER[0] + MODULE.v15.SHELF_STAGE_OFFSET_X
        self.assertGreaterEqual(
            MODULE.C_RELEASE_INSERT_TARGET_X_M - entry_start,
            0.045,
        )
        self.assertGreater(
            MODULE.C_RELEASE_INSERT_TARGET_X_M,
            MODULE.v15.TARGET_CENTER[0],
        )
        self.assertEqual(
            MODULE.C_RELEASE_INSERT_TARGET_X_M,
            MODULE.v15.TARGET_CENTER[0] + 0.0218,
        )
        self.assertEqual(MODULE.C_RELEASE_CLOSED_PREBACKOFF_M, 0.002)
        self.assertEqual(MODULE.C_RELEASE_CLOSED_PRELIFT_M, 0.0015)
        self.assertEqual(MODULE.C_RELEASE_OPEN_INITIAL_BACKOFF_M, 0.0)
        self.assertEqual(MODULE.C_RELEASE_NEAR_SHELF_GRIP_TARGET_M, 0.0105)
        self.assertGreater(MODULE.C_RELEASE_NEAR_SHELF_GRIP_TARGET_M, 0.0)
        self.assertLess(MODULE.C_RELEASE_NEAR_SHELF_GRIP_TARGET_M, 0.0110)
        self.assertEqual(MODULE.C_RELEASE_WRIST_LEVEL_DEG, 8.0)
        self.assertGreaterEqual(
            MODULE.v15.integrated_depth_margin(
                MODULE.C_RELEASE_INSERT_TARGET_X_M
                - MODULE.C_RELEASE_CLOSED_PREBACKOFF_M
                - MODULE.C_RELEASE_OPEN_INITIAL_BACKOFF_M,
                0.0,
            ),
            0.005 + MODULE.v15.RELEASE_OPEN_BACKOFF_STEP_M,
        )
        self.assertLess(
            MODULE.C_RELEASE_GUARDED_DROP_Z_M,
            MODULE.v15.RELEASE_GUARDED_DROP_Z_M,
        )
        self.assertEqual(MODULE.C_RELEASE_GUARDED_DROP_Z_M, 0.9470)
        self.assertEqual(MODULE.C_RELEASE_OPEN_CLEARANCE_LIFT_MAX_M, 0.0)
        release_center_x = (
            MODULE.C_RELEASE_INSERT_TARGET_X_M
            - MODULE.C_RELEASE_CLOSED_PREBACKOFF_M
            - MODULE.C_RELEASE_OPEN_INITIAL_BACKOFF_M
        )
        self.assertLess(
            abs(release_center_x - MODULE.v15.TARGET_CENTER[0]),
            0.020,
        )

    def test_c_release_prebackoff_keeps_grippers_closed(self):
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        method = source[
            source.index("    def release_into_integrated_top_tier(self):"):
            source.index("    def execute(self):")
        ]
        self.assertIn("self.move_mount_commands_delta(", method)
        self.assertIn('self.require_held("v16_c_closed_prebackoff")', method)
        self.assertNotIn("grip_cmd", method)

    def test_c_release_clears_front_lip_after_release_before_lift(self):
        self.assertEqual(MODULE.C_RELEASE_POST_OPEN_FORWARD_M, 0.0)
        base_source = (
            COLLECTION_DIR / "sorting_roll_expert.py"
        ).read_text(encoding="utf-8")
        base_release = base_source[
            base_source.index("    def release_into_integrated_top_tier(self):"):
            base_source.index("    def track_success(")
        ]
        clear_call = "self.clear_released_hands_before_lift()"
        lift_call = "RELEASE_CLEARANCE_LIFT_M\n                - actual_open_lift"
        self.assertIn(clear_call, base_release)
        self.assertLess(base_release.index(clear_call), base_release.index(lift_call))

        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        method = source[
            source.index("    def clear_released_hands_before_lift(self):"):
            source.index("    def release_into_integrated_top_tier(self):")
        ]
        self.assertNotIn("self.move_mount_commands_delta(", method)
        self.assertIn("c_release_contacts_are_incidental(contacts)", method)
        self.assertIn("v16_c_released_hands_ready_for_lift", method)

    def test_rigid_motion_metrics_reject_bad_shapes(self):
        with self.assertRaises(ValueError):
            MODULE.rigid_motion_metrics(np.zeros(6), np.zeros(7), np.zeros(6))

    def test_c_navigation_speedup_is_pregrasp_only_and_bounded(self):
        self.assertEqual(MODULE.C_TABLE_OBSERVATION_MAX_SPEED_M_S, 0.28)
        self.assertEqual(
            MODULE.C_TABLE_NAVIGATION_MAX_YAW_RATE_RAD_S,
            0.65,
        )
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        execute = source[
            source.index("    def execute(self):"):
            source.index("    def finalize(")
        ]
        self.assertIn("observation_offset = grasp_offset", execute)

    def test_v16_stage_offset_is_applied_without_mutating_v15_default(self):
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "v15.SHELF_STAGE_OFFSET_X = expert.shelf_stage_offset_x_m",
            source,
        )
        self.assertIn(
            "v15.SHELF_STAGE_OFFSET_X = old_shelf_stage_offset",
            source,
        )

    def test_non_c_stage_recentering_is_submillimeter(self):
        self.assertEqual(
            MODULE.V16_RELEASE_STAGE_CENTER_TOLERANCE_XZ_M,
            0.0005,
        )
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        self.assertIn('if self.family != "C":', source)
        self.assertIn("self.release_stage_center_tolerance_m", source)

    def test_non_c_release_descent_has_bounded_lateral_feedback(self):
        self.assertEqual(
            MODULE.V16_RELEASE_CLEARANCE_LATERAL_STEP_MAX_M,
            0.0015,
        )
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.release_clearance_axis_step_limits_m", source)
        self.assertIn("0.020", source)

    def test_short_boundary_adjusts_only_left_flat_pick_bias(self):
        self.assertEqual(
            MODULE.V16_SHORT_BOUNDARY_LEFT_FLAT_PICK_TIP_BIAS_Y_M,
            0.031,
        )
        source = (
            COLLECTION_DIR / "sorting_roll_v16_expansion_expert.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.flat_pick_tip_bias_y_m_by_hand", source)
        self.assertIn(
            '"l": V16_SHORT_BOUNDARY_LEFT_FLAT_PICK_TIP_BIAS_Y_M',
            source,
        )
        self.assertIn('self.family == "H"', source)
        self.assertIn('== "short_slim"', source)
        self.assertIn('== "boundary"', source)


if __name__ == "__main__":
    unittest.main()
