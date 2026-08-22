#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_camera_audit import (
    aggregate_candidate,
    mask_metrics,
    phase_spans,
    principal_extent,
    project_points,
    sampled_indices,
)


class SortingRollCameraAuditTest(unittest.TestCase):
    def test_phase_spans_requires_one_contiguous_span_per_phase(self):
        self.assertEqual(
            phase_spans(np.asarray(["a", "a", "b", "b", "b"])),
            {"a": (0, 1), "b": (2, 4)},
        )
        with self.assertRaises(ValueError):
            phase_spans(np.asarray(["a", "b", "a"])),

    def test_sampled_indices_include_phase_boundaries(self):
        self.assertEqual(sampled_indices(10, 20, 3), [10, 15, 20])
        self.assertEqual(sampled_indices(4, 4, 5), [4])
        with self.assertRaises(ValueError):
            sampled_indices(2, 1, 3)

    def test_mask_metrics_measure_occlusion_and_axis_extent(self):
        full = np.zeros((12, 12), dtype=bool)
        visible = np.zeros_like(full)
        full[5:7, 1:11] = True
        visible[5:7, 3:9] = True
        metrics = mask_metrics(visible, full)
        self.assertEqual(metrics["visible_pixels"], 12)
        self.assertEqual(metrics["full_pixels"], 20)
        self.assertAlmostEqual(metrics["visible_fraction"], 0.6)
        self.assertGreater(principal_extent(full), principal_extent(visible))
        self.assertLess(metrics["extent_fraction"], 1.0)

    def test_project_points_uses_mujoco_camera_forward_axis(self):
        rotation = np.eye(3)
        points = np.asarray([[0.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
        pixels, depth, in_frame = project_points(
            np.zeros(3), rotation, points, 90.0, 100, 100
        )
        np.testing.assert_allclose(pixels[0], [50.0, 50.0])
        np.testing.assert_allclose(depth, [1.0, 1.0])
        self.assertEqual(in_frame.tolist(), [True, False])

    def test_candidate_coverage_counts_any_usable_camera(self):
        records = [
            {"stage": "pickup", "phase": "p", "frame": 1, "camera": "a", "usable": False},
            {"stage": "pickup", "phase": "p", "frame": 1, "camera": "b", "usable": True},
            {"stage": "pickup", "phase": "p", "frame": 2, "camera": "a", "usable": False},
            {"stage": "pickup", "phase": "p", "frame": 2, "camera": "b", "usable": False},
        ]
        report = aggregate_candidate(records, ("a", "b"))
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["covered"], 1)
        self.assertAlmostEqual(report["coverage_fraction"], 0.5)
        self.assertAlmostEqual(report["mean_usable_views"], 0.5)


if __name__ == "__main__":
    unittest.main()
