#!/usr/bin/env python3

import math
import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_expert import (
    ARM_RETRACT_M,
    BASE_MAX_SPEED,
    BASE_MAX_YAW_RATE,
    RANDOM_BASE_XY_LIMIT_M,
    RANDOM_BASE_YAW_LIMIT_RAD,
    RANDOM_ROLL_XY_LIMIT_M,
    RANDOM_ROLL_YAW_LIMIT_RAD,
    FLAT_REGRASP_ANGLE_DEG,
    FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M,
    FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M,
    FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M,
    FLAT_REGRASP_CLEARANCE,
    FLAT_REGRASP_CLEARANCE_ONSET,
    FLAT_REGRASP_COUPLED_START_M,
    FLAT_REGRASP_FAR_END_M,
    FLAT_REGRASP_NEAR_END_M,
    FLAT_REGRASP_ORDER,
    FLAT_REGRASP_TARGET_ALONG_M,
    FLAT_PICK_COORDINATION_GRID_STEPS,
    FLAT_PICK_COORDINATION_CLEARANCE_CELLS,
    FLAT_PICK_GOAL_IK_SEEDS,
    FLAT_PICK_JOINT_WAYPOINTS,
    FLAT_PICK_ROLL_CLEARANCE_MARGIN_M,
    FLAT_PICK_COLLISION_STEP_RAD,
    FLAT_PICK_PREGRASP_CLEARANCE_Y_M,
    FLAT_PICK_TARGET_ALONG_M,
    FLAT_PICK_TIP_BIAS_Y_M,
    GRASP_YAW_DEG,
    INSERT_AXIS_X_SAFETY_LIMIT,
    INSERT_AXIS_Z_SAFETY_LIMIT,
    INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M,
    INSERT_AXIS_CORRECTION_MAX_STEP_M,
    RELEASE_APPROACH_Y_BIAS_M,
    RELEASE_CLEARANCE_LIFT_M,
    RELEASE_CLEARANCE_ROLL_Z,
    RELEASE_DROP_MAX_M,
    RELEASE_FRICTION_SETTLE_TICKS,
    RELEASE_GUARDED_DROP_Z_M,
    RELEASE_INSERT_STEP_M,
    RELEASE_INSERT_TARGET_X_M,
    RELEASE_OPEN_BACKOFF_MAX_M,
    RELEASE_OPEN_BACKOFF_STEP_M,
    RELEASE_OPEN_CLEARANCE_LIFT_MAX_M,
    RELEASE_OPEN_INITIAL_BACKOFF_M,
    RELEASE_PAD_SLIDING_FRICTION,
    RELEASE_PAD_SHELF_CLEARANCE_MIN_M,
    ROLL_RADIUS,
    SHELF_STAGE_OFFSET_X,
    POLICY_CAMERAS,
    RECORDED_CAMERAS,
    REVIEW_ONLY_CAMERAS,
    TASK_VERSION,
    TARGET_CENTER,
    TARGET_AXIS,
    SLOT_PHYSICS_REVIEW_CAMERA,
    SLOT_VISUAL_REVIEW_CAMERA,
    anchor_feedback_mount_position,
    anchored_mount_position,
    angle,
    bounded_vector,
    camera_mount_report,
    cartesian_waypoints,
    cosine_steps,
    coupled_regrasp_progress,
    coordination_clearance_mask,
    monotonic_coordination_indices,
    flat_regrasp_anchors,
    integrated_depth_margin,
    guarded_release_geometry_is_ready,
    guarded_release_is_ready,
    flatten_target_rotation,
    grasp_target_rotation,
    insertion_axis_is_safe,
    insertion_axis_correction_has_clearance,
    mount_position_for_pad_target,
    joint_polyline_at_progress,
    parse_args,
    rotation_x,
    rotation_axis_angle,
    rotation_z,
    roll_half_extent_x,
    seed_randomization,
    resolved_geom_clearance,
    release_is_clear,
    symmetric_level_correction,
    symmetric_axis_correction,
)
from sorting_roll_scene import (
    SCENE_PATH,
    TARGET_CENTER as SCENE_TARGET_CENTER,
)


class SortingRollExpertTest(unittest.TestCase):
    def test_release_sequence_keeps_strict_contact_gate(self):
        clear = {"force_n": 0.0, "pads": []}
        touching = {"force_n": 1.2, "pads": ["L_pad1"]}
        weak_contact = {"force_n": 0.051, "pads": []}
        self.assertTrue(release_is_clear(clear, clear))
        self.assertFalse(release_is_clear(touching, clear))
        self.assertFalse(release_is_clear(clear, weak_contact))

    def test_angle_wraps_to_shortest_signed_distance(self):
        self.assertAlmostEqual(angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(angle(-3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(angle(0.25), 0.25)

    def test_bounded_vector_preserves_direction_and_limits_norm(self):
        vector = np.array([3.0, 4.0, 0.0])
        bounded = bounded_vector(vector, 2.0)
        np.testing.assert_allclose(bounded, [1.2, 1.6, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(bounded)), 2.0)
        np.testing.assert_allclose(
            bounded_vector([0.1, 0.0, 0.0], 2.0), [0.1, 0.0, 0.0]
        )

    def test_integrated_depth_margin_accounts_for_center_and_axis(self):
        center_x = float(TARGET_CENTER[0])
        self.assertAlmostEqual(roll_half_extent_x(0.0), 0.012)
        self.assertAlmostEqual(roll_half_extent_x(1.0), 0.25)
        self.assertAlmostEqual(
            integrated_depth_margin(center_x, 0.0),
            0.011,
        )
        self.assertGreater(
            integrated_depth_margin(center_x, 0.0),
            0.005,
        )
        self.assertLess(
            integrated_depth_margin(0.7825, 0.0),
            0.0,
        )
        self.assertLess(
            integrated_depth_margin(center_x, 0.05),
            0.0,
        )
    def test_insert_axis_monitor_accepts_dev050_drift_but_stays_bounded(self):
        self.assertTrue(insertion_axis_is_safe([0.00091, 0.999997, -0.0021]))
        self.assertFalse(
            insertion_axis_is_safe([
                INSERT_AXIS_X_SAFETY_LIMIT + 1e-6,
                1.0,
                0.0,
            ])
        )

    def test_insert_axis_correction_requires_roll_and_pad_clearance(self):
        self.assertAlmostEqual(INSERT_AXIS_CORRECTION_MAX_STEP_M, 0.001)
        self.assertAlmostEqual(INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M, 0.008)
        self.assertTrue(
            insertion_axis_correction_has_clearance(0.01234, 0.02665)
        )
        self.assertTrue(
            insertion_axis_correction_has_clearance(
                INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M,
                INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M,
            )
        )
        self.assertFalse(
            insertion_axis_correction_has_clearance(
                INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M - 1e-6,
                0.1,
            )
        )
        self.assertFalse(
            insertion_axis_is_safe([
                0.0,
                1.0,
                INSERT_AXIS_Z_SAFETY_LIMIT + 1e-6,
            ])
        )

    def test_release_geometry_enters_before_lowering_and_lifts_to_exit(self):
        self.assertAlmostEqual(ARM_RETRACT_M, 0.082)
        self.assertAlmostEqual(SHELF_STAGE_OFFSET_X, -0.050)
        self.assertAlmostEqual(RELEASE_CLEARANCE_ROLL_Z, 0.955)
        self.assertAlmostEqual(RELEASE_GUARDED_DROP_Z_M, 0.951)
        self.assertAlmostEqual(RELEASE_APPROACH_Y_BIAS_M, 0.0)
        self.assertAlmostEqual(
            RELEASE_INSERT_TARGET_X_M,
            float(TARGET_CENTER[0]) + 0.040,
        )
        self.assertGreaterEqual(
            RELEASE_CLEARANCE_ROLL_Z - ROLL_RADIUS - 0.915,
            0.025,
        )
        self.assertGreaterEqual(
            integrated_depth_margin(
                RELEASE_INSERT_TARGET_X_M,
                INSERT_AXIS_X_SAFETY_LIMIT,
            ),
            0.020,
        )
        self.assertAlmostEqual(RELEASE_INSERT_STEP_M, 0.006)
        self.assertLessEqual(RELEASE_INSERT_STEP_M, 0.010)
        self.assertAlmostEqual(RELEASE_DROP_MAX_M, 0.025)
        self.assertAlmostEqual(
            RELEASE_PAD_SHELF_CLEARANCE_MIN_M,
            0.002,
        )
        self.assertAlmostEqual(RELEASE_CLEARANCE_LIFT_M, 0.050)
        self.assertAlmostEqual(RELEASE_PAD_SLIDING_FRICTION, 1.0)
        self.assertEqual(RELEASE_FRICTION_SETTLE_TICKS, 12)

    def test_guarded_release_bounds_drop_and_pad_clearance(self):
        self.assertTrue(guarded_release_is_ready(0.020, 0.003))
        self.assertFalse(guarded_release_is_ready(-1e-6, 0.003))
        self.assertFalse(guarded_release_is_ready(0.026, 0.003))
        self.assertFalse(guarded_release_is_ready(0.020, 0.001))

    def test_guarded_release_geometry_does_not_reuse_final_height(self):
        margins = {"negative_y": 0.030, "positive_y": 0.030}
        self.assertTrue(
            guarded_release_geometry_is_ready(margins, 4.0, 0.006)
        )
        self.assertFalse(
            guarded_release_geometry_is_ready(margins, 5.1, 0.006)
        )
        self.assertFalse(
            guarded_release_geometry_is_ready(margins, 4.0, 0.004)
        )

    def test_resolved_clearance_uses_witness_only_without_contact(self):
        self.assertAlmostEqual(
            resolved_geom_clearance(0.0, 0.020, False),
            0.020,
        )
        self.assertEqual(
            resolved_geom_clearance(0.0, 0.020, True),
            0.0,
        )
        self.assertEqual(
            resolved_geom_clearance(-0.001, 0.020, False),
            -0.001,
        )

    def test_cosine_steps_respects_peak_step_bound(self):
        steps = cosine_steps(1.0, 0.01, minimum=2)
        self.assertGreaterEqual(steps, 158)
        self.assertEqual(cosine_steps(0.0, 0.01, minimum=7), 7)
        with self.assertRaises(ValueError):
            cosine_steps(1.0, 0.0)

    def test_joint_polyline_uses_max_joint_distance(self):
        waypoints = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
        ])
        np.testing.assert_allclose(
            joint_polyline_at_progress(waypoints, 0.0),
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            joint_polyline_at_progress(waypoints, 1.0 / 3.0),
            [1.0, 0.0],
        )
        np.testing.assert_allclose(
            joint_polyline_at_progress(waypoints, 2.0 / 3.0),
            [1.0, 1.0],
        )
        np.testing.assert_allclose(
            joint_polyline_at_progress(waypoints, 1.0),
            [1.0, 2.0],
        )
        with self.assertRaises(ValueError):
            joint_polyline_at_progress(waypoints, -0.01)
    def test_v13_target_is_sourced_from_scene_contract(self):
        self.assertEqual(
            TASK_VERSION,
            "sorting_roll_v13_d405_rearward_mount_sim",
        )
        np.testing.assert_allclose(TARGET_CENTER, SCENE_TARGET_CENTER)

    def test_seed_randomization_is_bounded_and_reproducible(self):
        first = seed_randomization(17)
        self.assertEqual(first, seed_randomization(17))
        self.assertNotEqual(first, seed_randomization(18))
        self.assertTrue(all(
            abs(value) <= RANDOM_BASE_XY_LIMIT_M
            for value in first["base_delta_xyyaw"][:2]
        ))
        self.assertLessEqual(
            abs(first["base_delta_xyyaw"][2]),
            RANDOM_BASE_YAW_LIMIT_RAD,
        )
        self.assertTrue(all(
            abs(value) <= RANDOM_ROLL_XY_LIMIT_M
            for value in first["roll_delta_xy_m"]
        ))
        self.assertLessEqual(
            abs(first["roll_yaw_rad"]),
            RANDOM_ROLL_YAW_LIMIT_RAD,
        )

    def test_pose_bins_are_reproducible_and_cover_the_requested_range(self):
        limits = np.asarray([
            RANDOM_BASE_XY_LIMIT_M,
            RANDOM_BASE_XY_LIMIT_M,
            RANDOM_BASE_YAW_LIMIT_RAD,
            RANDOM_ROLL_XY_LIMIT_M,
            RANDOM_ROLL_XY_LIMIT_M,
            RANDOM_ROLL_YAW_LIMIT_RAD,
        ])
        expected = {
            "easy": (0.0, 0.40),
            "medium": (0.40, 0.75),
            "boundary": (0.75, 1.00),
        }
        for pose_bin, (lower, upper) in expected.items():
            with self.subTest(pose_bin=pose_bin):
                value = seed_randomization(37, pose_bin)
                self.assertEqual(value, seed_randomization(37, pose_bin))
                normalized = np.abs(np.asarray([
                    *value["base_delta_xyyaw"],
                    *value["roll_delta_xy_m"],
                    value["roll_yaw_rad"],
                ]) / limits)
                self.assertLessEqual(float(normalized.max()), upper)
                self.assertGreaterEqual(float(normalized.max()), lower)

    def test_v13_records_three_d405_candidate_policy_cameras(self):
        self.assertEqual(
            POLICY_CAMERAS,
            (
                "stereo_left",
                "left_wrist_realsense",
                "right_wrist_realsense",
            ),
        )
        self.assertEqual(REVIEW_ONLY_CAMERAS, ("third_person",))
        self.assertEqual(RECORDED_CAMERAS, POLICY_CAMERAS)

    def test_v13_global_camera_mount_matches_sdk_extrinsics(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        data = mujoco.MjData(model)
        report = camera_mount_report(mujoco, model, data)
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["intrinsics_verified"])
        self.assertEqual(
            set(report["cameras"]),
            {"stereo_left"},
        )
        for camera in report["cameras"].values():
            self.assertLessEqual(camera["position_error_mm"], 0.05)
            self.assertLessEqual(camera["forward_error_deg"], 0.05)
            self.assertLessEqual(camera["right_error_deg"], 0.05)

    def test_supported_pick_targets_flat_hands_without_loaded_rotation(self):
        self.assertAlmostEqual(FLAT_PICK_TARGET_ALONG_M, 0.160)
        self.assertAlmostEqual(FLAT_PICK_TIP_BIAS_Y_M, 0.034)
        self.assertGreater(FLAT_PICK_PREGRASP_CLEARANCE_Y_M, 0.05)
        self.assertAlmostEqual(
            FLAT_PICK_ROLL_CLEARANCE_MARGIN_M,
            0.008,
        )
        self.assertAlmostEqual(FLAT_PICK_COLLISION_STEP_RAD, 0.005)
        self.assertEqual(FLAT_PICK_COORDINATION_GRID_STEPS, 120)
        self.assertEqual(len(FLAT_PICK_JOINT_WAYPOINTS["l"]), 3)
        self.assertEqual(len(FLAT_PICK_JOINT_WAYPOINTS["r"]), 5)
        self.assertGreater(FLAT_PICK_GOAL_IK_SEEDS["l"][4], 0.0)
        self.assertGreater(FLAT_PICK_GOAL_IK_SEEDS["r"][4], 0.0)
        self.assertEqual(FLAT_PICK_COORDINATION_CLEARANCE_CELLS, 1)
        self.assertEqual(SLOT_VISUAL_REVIEW_CAMERA, (0.85, -45.0, -45.0))
        self.assertEqual(SLOT_PHYSICS_REVIEW_CAMERA, (0.85, -45.0, -45.0))
        target_pad = np.array([0.16, -0.93, 1.112])
        rotation = flatten_target_rotation(
            grasp_target_rotation(np.diag([-1.0, 1.0, -1.0]), 1.0),
            1.0,
        )
        local_pad = np.array([0.0, 0.0, 0.0955])
        mount = mount_position_for_pad_target(
            target_pad,
            rotation,
            local_pad,
        )
        np.testing.assert_allclose(
            mount + rotation @ local_pad,
            target_pad,
            atol=1e-12,
        )
        self.assertGreater(BASE_MAX_SPEED, 0.20)
        self.assertGreater(BASE_MAX_YAW_RATE, 0.40)

    def test_dynamic_coordination_expands_obstacles_and_checks_edges(self):
        validity = np.ones((5, 5), dtype=bool)
        validity[2, 2] = False
        clearance = coordination_clearance_mask(validity, 1)
        self.assertFalse(np.any(clearance[1:4, 1:4]))
        self.assertTrue(clearance[0, 0])
        self.assertTrue(clearance[-1, -1])

        checked_edges = []

        def edge_is_safe(start, target):
            checked_edges.append((start, target))
            return (start, target) != ((1, 1), (2, 2))

        path = monotonic_coordination_indices(
            np.ones((4, 4), dtype=bool),
            edge_is_safe=edge_is_safe,
        )
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))
        self.assertNotIn(((1, 1), (2, 2)), zip(path, path[1:]))
        self.assertIn(((1, 1), (2, 2)), checked_edges)
        for start, target in zip(path, path[1:]):
            step = (
                target[0] - start[0],
                target[1] - start[1],
            )
            self.assertIn(step, ((1, 0), (0, 1), (1, 1)))

    def test_side_flat_rotation_makes_closing_axis_vertical(self):
        fingers_down = np.diag([-1.0, 1.0, -1.0])
        flat = rotation_x(math.radians(-94.0)) @ fingers_down
        self.assertGreater(abs(float(flat[2, 1])), 0.99)
        self.assertLess(abs(float(flat[2, 2])), 0.08)

    def test_grasp_yaw_is_symmetric(self):
        initial = np.eye(3)
        np.testing.assert_allclose(
            grasp_target_rotation(initial, 1.0),
            rotation_z(math.radians(GRASP_YAW_DEG)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grasp_target_rotation(initial, -1.0),
            rotation_z(math.radians(-GRASP_YAW_DEG)),
            atol=1e-12,
        )

    def test_axis_angle_rotation_matches_principal_axis_helper(self):
        np.testing.assert_allclose(
            rotation_axis_angle([2.0, 0.0, 0.0], math.pi / 3.0),
            rotation_x(math.pi / 3.0),
            atol=1e-12,
        )
        with self.assertRaises(ValueError):
            rotation_axis_angle([0.0, 0.0, 0.0], 0.1)

    def test_flatten_rotation_uses_absolute_progress(self):
        initial = np.diag([-1.0, 1.0, -1.0])
        halfway = flatten_target_rotation(initial, 0.5)
        complete = flatten_target_rotation(initial, 1.0)
        np.testing.assert_allclose(
            halfway,
            rotation_x(math.radians(-47.0)) @ initial,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            complete,
            rotation_x(math.radians(-94.0)) @ initial,
            atol=1e-12,
        )

    def test_flat_regrasp_always_has_the_other_hand_supporting(self):
        self.assertEqual(
            FLAT_REGRASP_ORDER,
            (("r", "l", -1.0), ("l", "r", 1.0)),
        )
        self.assertEqual(FLAT_REGRASP_ANGLE_DEG, 94.0)
        self.assertEqual(GRASP_YAW_DEG, 14.0)

    def test_regrasp_anchors_exit_and_reenter_through_roll_end(self):
        roll = np.array([0.0, -1.2, 1.24])
        axis = np.array([1.0, 0.0, 0.0])
        current = np.array([-0.12, -1.19, 1.27])
        right = flat_regrasp_anchors(roll, axis, current, -1.0)
        self.assertAlmostEqual(right["far_end"][0], -FLAT_REGRASP_FAR_END_M)
        np.testing.assert_allclose(right["far_end"][1:], current[1:])
        np.testing.assert_allclose(
            right["axis_far"],
            roll - FLAT_REGRASP_FAR_END_M * axis,
        )
        np.testing.assert_allclose(
            right["target"],
            roll - FLAT_REGRASP_TARGET_ALONG_M * axis,
        )
        left = flat_regrasp_anchors(roll, axis, current, 1.0)
        self.assertAlmostEqual(left["target"][0], FLAT_REGRASP_TARGET_ALONG_M)
        with self.assertRaises(ValueError):
            flat_regrasp_anchors(roll, np.zeros(3), current, 1.0)

    def test_coupled_regrasp_delays_rotation_and_clearance(self):
        self.assertLess(FLAT_REGRASP_COUPLED_START_M, FLAT_REGRASP_NEAR_END_M)
        np.testing.assert_allclose(
            FLAT_REGRASP_CLEARANCE,
            [0.0, 0.043, 0.020],
        )
        rotation, clearance = coupled_regrasp_progress(
            FLAT_REGRASP_CLEARANCE_ONSET
        )
        self.assertAlmostEqual(
            rotation,
            FLAT_REGRASP_CLEARANCE_ONSET**2,
        )
        self.assertEqual(clearance, 0.0)
        self.assertEqual(coupled_regrasp_progress(0.0), (0.0, 0.0))
        self.assertEqual(coupled_regrasp_progress(1.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            coupled_regrasp_progress(1.01)

    def test_anchor_feedback_uses_measured_error_and_bounds_correction(self):
        command_mount = np.array([0.1, -0.2, 1.0])
        target_anchor = np.array([0.0, 0.0, 0.010])
        actual_anchor = np.zeros(3)
        corrected = anchor_feedback_mount_position(
            command_mount,
            target_anchor,
            actual_anchor,
            FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M,
        )
        np.testing.assert_allclose(
            corrected,
            command_mount + [0.0, 0.0, 0.008],
        )
        self.assertLess(
            FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M,
            FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M,
        )

    def test_symmetric_level_correction_raises_lower_end_and_is_bounded(self):
        correction = symmetric_level_correction(
            [0.972445, 0.229194, -0.042671],
            [0.153454, -1.168040, 1.222291],
            [-0.154943, -1.239447, 1.235369],
            0.004,
        )
        self.assertEqual(correction, 0.004)
        self.assertAlmostEqual(
            symmetric_level_correction(
                [1.0, 0.0, 0.01],
                [0.16, 0.0, 1.0],
                [-0.16, 0.0, 1.0],
                0.004,
            ),
            -0.0016,
        )
        with self.assertRaises(ValueError):
            symmetric_level_correction(
                [0.0, 0.0, 0.0],
                [0.16, 0.0, 1.0],
                [-0.16, 0.0, 1.0],
                0.004,
            )

    def test_symmetric_axis_correction_removes_low_stage_tilt(self):
        correction = symmetric_axis_correction(
            [0.017872, 0.999695, 0.017058],
            [0.717267, 0.160634, 1.056904],
            [0.709690, -0.159636, 1.051461],
            TARGET_AXIS,
            0.004,
        )
        self.assertLess(float(correction[0]), 0.0)
        self.assertAlmostEqual(float(correction[1]), 0.0)
        self.assertLess(float(correction[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(correction)), 0.004)
        with self.assertRaises(ValueError):
            symmetric_axis_correction(
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.0],
                TARGET_AXIS,
                0.004,
            )

    def test_cartesian_waypoints_bound_step_and_reach_target(self):
        points = cartesian_waypoints([0.0, 0.0, 0.0], [0.01, 0.0, 0.0], 0.003)
        previous = np.zeros(3)
        for point in points:
            self.assertLessEqual(float(np.linalg.norm(point - previous)), 0.003)
            previous = point
        np.testing.assert_allclose(points[-1], [0.01, 0.0, 0.0])
        with self.assertRaises(ValueError):
            cartesian_waypoints([0, 0, 0], [1, 0, 0], 0.0)

    def test_anchored_mount_rotation_keeps_contact_point_fixed(self):
        mount = np.array([0.2, -0.1, 1.2])
        rotation = np.eye(3)
        anchor = np.array([0.2, -0.1, 1.1])
        target_rotation = rotation_x(math.pi / 2.0)
        target_mount = anchored_mount_position(
            mount, rotation, anchor, target_rotation
        )
        local_anchor = rotation.T @ (anchor - mount)
        np.testing.assert_allclose(
            target_mount + target_rotation @ local_anchor,
            anchor,
            atol=1e-12,
        )
        shifted_anchor = anchor + np.array([0.01, -0.02, 0.03])
        shifted_mount = anchored_mount_position(
            mount,
            rotation,
            anchor,
            target_rotation,
            target_anchor_position=shifted_anchor,
        )
        np.testing.assert_allclose(
            shifted_mount + target_rotation @ local_anchor,
            shifted_anchor,
            atol=1e-12,
        )

    def test_review_choreography_phase_order(self):
        source = (
            COLLECTION_DIR / "sorting_roll_expert.py"
        ).read_text(encoding="utf-8")
        execute_source = source[
            source.index("    def execute(self):"):
            source.index("    def encode_frame_video")
        ]
        phases = (
            "navigate_to_table_observation",
            "approach_table_with_arms_down",
            "localize_roll_with_head_stereo",
            "coordinated_flat_pick_pregrasp_after_stereo_localization",
            "horizontal_approach_and_grasp",
            "lift_flat_from_pickup_support",
            "clear_table",
            "rotate_to_shelf",
            "navigate_to_shelf_stage",
            "align_shelf_axis_above_front_lip",
            "realign_shelf_stage_after_axis",
            "level_release_support_surfaces",
            "fine_align_axis_before_entry",
            "lower_to_front_lip_clearance",
            "move_over_integrated_front_lip",
            "position_guarded_release_clearance",
            "guarded_release_and_lift_open_hands",
            "verify_after_guarded_release",
            "retract_arms_after_release",
            "terminal_success_hold",
        )
        offsets = [
            execute_source.index(f'"{phase}"')
            for phase in phases
        ]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn(
            '"arms_unchanged_through_table_approach"',
            execute_source,
        )
        self.assertNotIn('self.phase("observe_roll")', execute_source)
        self.assertNotIn(
            "clearance_rotation = grasp_target_rotation",
            execute_source,
        )
        self.assertNotIn("clearance_targets", execute_source)
        self.assertIn(
            "self.follow_coordinated_flat_pick_path(\n"
            "            pregrasp_positions,",
            execute_source,
        )
        self.assertNotIn(
            "self.follow_empty_hands_stage(",
            execute_source,
        )
        self.assertIn("tolerance=0.003", execute_source)
        self.assertIn('"sdk_camera_mounts"', execute_source)
        self.assertIn('self.require_held("shelf_translation")', execute_source)
        self.assertIn(
            '"shelf_corridor_reverse_plan"',
            execute_source,
        )
        self.assertIn(
            "self.reverse(corridor_reverse_m, max_speed=0.26)",
            execute_source,
        )
        self.assertIn("shelf_base, 0.0, max_speed=0.28", execute_source)
        self.assertEqual(execute_source.count("self.turn_in_place(0.0)"), 1)
        self.assertIn(
            '"flat_pick_goal_ik_reachable"',
            source,
        )
        self.assertIn(
            '"collision_free_coordinated_flat_pick_path"',
            source,
        )
        self.assertIn(
            "FLAT_PICK_ROLL_CLEARANCE_MARGIN_M",
            source,
        )
        self.assertIn(
            '"hands_flat_before_pick"',
            execute_source,
        )
        self.assertNotIn(
            '"hands_flat_after_localization_raise"',
            execute_source,
        )
        self.assertIn('"forward_entry_motion"', execute_source)
        self.assertNotIn('self.phase("flatten_hands")', execute_source)
        self.assertNotIn("position_grip_at_release_edge", execute_source)
        self.assertNotIn("regrasp_at_release_tips", execute_source)
        self.assertIn(
            "self.release_into_integrated_top_tier()",
            execute_source,
        )
        self.assertNotIn("release_with_backward_withdrawal", execute_source)

    def test_review_multiview_labels_d405_candidate_roles(self):

        source = (
            COLLECTION_DIR / "sorting_roll_expert.py"
        ).read_text(encoding="utf-8")
        encode_source = source[
            source.index("    def encode_review_videos"):
            source.index("    def finalize")
        ]
        self.assertIn("drawtext", encode_source)
        self.assertIn("MODEL_CAMERA_SOURCES", encode_source)
        self.assertIn("CAMERA_ROLES", encode_source)
        self.assertIn("xstack=inputs=3", encode_source)
        self.assertIn(
            'f"{width // 2}_0|0_{height}|{width}_{height}:fill=black[v]"',
            encode_source,
        )
        self.assertIn("slot_visual_review_video", encode_source)
        self.assertIn("slot_physics_review_video", encode_source)
        self.assertIn("if self.args.review_videos", encode_source)

    def test_review_videos_require_rendering(self):
        args = parse_args(["--out", "candidate"])
        self.assertFalse(args.review_videos)
        args = parse_args([
            "--out", "review", "--review-videos",
        ])
        self.assertTrue(args.review_videos)
        with self.assertRaises(SystemExit):
            parse_args([
                "--out", "invalid", "--no-render", "--review-videos",
            ])

    def test_manifest_requires_scene_randomization(self):
        with self.assertRaises(SystemExit):
            parse_args(["--out", "invalid", "--manifest", "campaign.json"])
        args = parse_args([
            "--out", "candidate", "--randomize", "--manifest", "campaign.json",
        ])
        self.assertEqual(args.manifest, Path("campaign.json"))

    def test_finalize_synchronizes_nested_canary_eligibility(self):
        source = (
            COLLECTION_DIR / "sorting_roll_expert.py"
        ).read_text(encoding="utf-8")
        finalize = source[
            source.index("    def finalize"):
            source.index("\ndef main")
        ]
        assignment = (
            'self.ct.REC["metadata"]'
            '["simulation_canary_eligible"] = bool('
        )
        self.assertIn(assignment, finalize)
        self.assertLess(
            finalize.index(assignment),
            finalize.index("self.recorder.finalize"),
        )

    def test_guarded_release_opens_before_lifting_and_retracting(self):
        source = (
            COLLECTION_DIR / "sorting_roll_expert.py"
        ).read_text(encoding="utf-8")
        release = source[
            source.index("def release_into_integrated_top_tier"):
            source.index("def track_success")
        ]
        self.assertLess(
            release.index("geom_friction"),
            release.index('self.ct.grip_cmd["l"]'),
        )
        self.assertIn('"guarded_release_ready"', release)
        self.assertLess(
            release.index('"guarded_release_ready"'),
            release.index('self.ct.grip_cmd["l"]'),
        )
        self.assertIn("minimum_geom_clearance", release)
        self.assertLess(
            release.index('self.ct.grip_cmd["l"]'),
            release.index("self.move_mount_commands_delta"),
        )
        self.assertIn("RELEASE_CLEARANCE_LIFT_M", release)
        self.assertEqual(RELEASE_OPEN_INITIAL_BACKOFF_M, 0.010)
        self.assertEqual(RELEASE_OPEN_BACKOFF_STEP_M, 0.004)
        self.assertEqual(RELEASE_OPEN_BACKOFF_MAX_M, 0.050)
        self.assertEqual(RELEASE_OPEN_CLEARANCE_LIFT_MAX_M, 0.010)
        self.assertIn("guarded_release_clear_confirmation", release)
        self.assertLess(
            release.index("guarded_release_clear_confirmation"),
            release.index("release_clear_confirmed = True"),
        )
        self.assertIn("release_contact_geometry", release)
        before_open = release[:release.index('self.ct.grip_cmd["l"]')]
        after_open = release[release.index('self.ct.grip_cmd["l"]'):]
        self.assertNotIn(
            'checks["resting_on_integrated_top_tier_geometry"]',
            before_open,
        )
        self.assertNotIn(
            'checks["supported_by_integrated_top_tier"]',
            before_open,
        )
        self.assertNotIn(
            'checks["center_inside_integrated_top_tier"]',
            before_open,
        )
        self.assertIn(
            'after_checks["resting_on_integrated_top_tier_geometry"]',
            after_open,
        )
        self.assertNotIn("RELEASE_BACKWARD_WITHDRAWAL_M", release)
        self.assertNotIn("RELEASE_RECENTER_MAX_M", release)
        self.assertNotIn("SLOT_CAPTURE_HALF_WIDTH", release)
if __name__ == "__main__":
    unittest.main()
