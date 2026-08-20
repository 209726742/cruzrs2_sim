#!/usr/bin/env python3

import math
import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_expert import (
    angle, bounded_vector, cosine_steps, cylinder_slot_fit_margin, TARGET_CENTER,
)


class SortingRollExpertTest(unittest.TestCase):
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

    def test_cylinder_slot_fit_margin_accounts_for_center_and_axis(self):
        center_x = float(TARGET_CENTER[0])
        self.assertAlmostEqual(
            cylinder_slot_fit_margin(center_x, 0.0), 0.0005
        )
        self.assertAlmostEqual(
            cylinder_slot_fit_margin(center_x + 0.0004, 0.0), 0.0001
        )
        self.assertLess(cylinder_slot_fit_margin(center_x, 0.003), 0.0)

    def test_cosine_steps_respects_peak_step_bound(self):
        steps = cosine_steps(1.0, 0.01, minimum=2)
        self.assertGreaterEqual(steps, 158)
        self.assertEqual(cosine_steps(0.0, 0.01, minimum=7), 7)
        with self.assertRaises(ValueError):
            cosine_steps(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
