#!/usr/bin/env python3

import json
import os
import sys
import unittest

import numpy as np


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from strip_cable_damping_fit import (  # noqa: E402
    calibrated_parameters,
    first_mode_curvature_weights,
    fit_damping,
    load_decay_source,
)
from strip_cable_isolated import DEFAULT_OBJ, load_parameter_candidate  # noqa: E402


PARAMETERS = os.path.join(
    ROOT, "assets", "shelf", "strip_cable_parameters_synthetic_v1.json"
)
MEASUREMENTS = os.path.join(
    ROOT, "docs", "strip_material_assumptions_simulation_v1.json"
)


class StripCableDampingFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, cls.parameter_sha = load_parameter_candidate(PARAMETERS)
        cls.displacement, cls.protocol, cls.measurement_sha = load_decay_source(
            MEASUREMENTS, cls.parameters["source_measurement_sha256"]
        )

    def test_first_mode_curvature_decays_smoothly_to_free_end(self):
        weights = first_mode_curvature_weights(13)
        self.assertEqual(weights.shape, (13,))
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertTrue(np.all(np.diff(weights) < 0.0))
        self.assertGreater(weights[-1], 0.0)
        self.assertLess(weights[-1], 0.02)

    def test_source_sha_and_initial_displacement_are_preserved(self):
        self.assertAlmostEqual(self.displacement, 0.1)
        self.assertIsNone(self.protocol)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_decay_source(MEASUREMENTS, "0" * 64)

    def test_fit_matches_decrement_without_using_period_as_gate(self):
        report = fit_damping(
            self.parameters,
            obj_path=DEFAULT_OBJ,
            initial_displacement_m=self.displacement,
            source_protocol=None,
            iterations=12,
        )
        self.assertTrue(report["fit_ready_for_scene_template"], report["checks"])
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["period_comparison"]["comparable"])
        damping = report["fit"]["joint_damping_nms_per_rad"]
        self.assertGreater(damping, 0.09)
        self.assertLess(damping, 0.12)

        calibrated = calibrated_parameters(
            self.parameters, report, self.parameter_sha, self.measurement_sha
        )
        self.assertEqual(calibrated["mode"], "isolated_cable_parameter_calibrated")
        self.assertFalse(calibrated["formal_collection_allowed"])
        self.assertTrue(calibrated["fit_ready_for_scene_template"])
        self.assertIsNotNone(calibrated["mujoco"]["joint_damping_nms_per_rad"])
        self.assertIsNone(self.parameters["mujoco"]["joint_damping_nms_per_rad"])


if __name__ == "__main__":
    unittest.main()
