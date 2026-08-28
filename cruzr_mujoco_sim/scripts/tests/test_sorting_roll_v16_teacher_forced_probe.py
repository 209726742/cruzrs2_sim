#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

from sorting_roll_v16_teacher_forced_probe import (  # noqa: E402
    checkpoint_label,
    chunk_metrics,
    select_probe_frames,
)


class SortingRollV16TeacherForcedProbeTest(unittest.TestCase):
    def test_select_probe_frames_uses_phase_and_gripper_evidence(self):
        phases = np.array(
            ["initial_hold"] * 5
            + ["localize_roll_with_head_stereo"] * 5
            + ["coordinated_flat_pick_pregrasp_after_stereo_localization"] * 5
            + ["horizontal_approach_and_grasp"] * 10
            + ["lift_flat_from_pickup_support"] * 55
            + ["clear_table"] * 50
        )
        state = np.ones((len(phases), 18), dtype=np.float32)
        action = np.ones((len(phases), 18), dtype=np.float32)
        action[19:, 14:16] = 0.0
        state[21:, 14:16] = 0.6
        frames = select_probe_frames(phases, state, action)
        self.assertEqual(frames, {
            "table_observation": 5,
            "pregrasp": 10,
            "precontact": 18,
            "grasp_established": 21,
            "lift_start": 25,
            "support_cleared": 79,
        })

    def test_stage_episode_uses_approach_fallback_and_short_lift_future(self):
        phases = np.array(
            ["approach_table_with_arms_staged"] * 10
            + ["coordinated_flat_pick_pregrasp_after_stereo_localization"] * 5
            + ["horizontal_approach_and_grasp"] * 10
            + ["lift_flat_from_pickup_support"] * 24
        )
        state = np.ones((len(phases), 18), dtype=np.float32)
        action = np.ones((len(phases), 18), dtype=np.float32)
        action[20:, 14:16] = 0.0
        state[22:, 14:16] = 0.6
        self.assertEqual(select_probe_frames(phases, state, action), {
            "table_observation": 0,
            "pregrasp": 10,
            "precontact": 19,
            "grasp_established": 22,
            "lift_start": 25,
            "support_cleared": 48,
        })

    def test_chunk_metrics_uses_only_available_expert_future(self):
        predicted = np.zeros((50, 18), dtype=np.float32)
        expert = np.zeros((24, 18), dtype=np.float32)
        predicted[24:] = 1.0
        metrics = chunk_metrics(predicted, expert, np.zeros(18, dtype=np.float32))
        self.assertEqual(metrics["horizons"]["20"]["evaluated_steps"], 20)
        self.assertEqual(metrics["horizons"]["30"]["evaluated_steps"], 24)
        self.assertEqual(metrics["horizons"]["50"]["evaluated_steps"], 24)
        for horizon in ("20", "30", "50"):
            for group in metrics["horizons"][horizon]["groups"].values():
                self.assertEqual(group["mae"], 0.0)

    def test_chunk_metrics_reports_exact_match(self):
        expert = np.linspace(0.0, 1.0, 50 * 18, dtype=np.float32).reshape(50, 18)
        metrics = chunk_metrics(expert.copy(), expert, np.zeros(18, dtype=np.float32))
        for horizon in ("20", "30", "50"):
            for group in metrics["horizons"][horizon]["groups"].values():
                self.assertEqual(group["mae"], 0.0)
                self.assertEqual(group["rmse"], 0.0)
            self.assertAlmostEqual(
                metrics["horizons"][horizon]["arm_delta_cosine"], 1.0
            )
            self.assertEqual(
                metrics["horizons"][horizon]["base_sign_agreement"], 1.0
            )

    def test_chunk_metrics_ignores_sub_deadband_base_noise(self):
        expert = np.zeros((50, 18), dtype=np.float32)
        predicted = expert.copy()
        predicted[:, 16] = 0.005
        metrics = chunk_metrics(
            predicted, expert, np.zeros(18, dtype=np.float32)
        )
        for horizon in ("20", "30", "50"):
            self.assertEqual(
                metrics["horizons"][horizon]["base_sign_agreement"], 1.0
            )

    def test_checkpoint_label_requires_explicit_name(self):
        label, path = checkpoint_label("28k=/tmp/checkpoint")
        self.assertEqual(label, "28k")
        self.assertEqual(path, Path("/tmp/checkpoint"))
        with self.assertRaises(Exception):
            checkpoint_label("/tmp/checkpoint")


if __name__ == "__main__":
    unittest.main()
