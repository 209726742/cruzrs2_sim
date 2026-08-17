#!/usr/bin/env python3

import copy
import json
import os
import sys
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from strip_material_fit import fit_material, rectangle_torsion_constant  # noqa: E402


ASSUMPTIONS = os.path.join(
    ROOT,
    "docs",
    "strip_material_assumptions_simulation_v1.json",
)


class StripMaterialFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ASSUMPTIONS, encoding="utf-8") as handle:
            cls.document = json.load(handle)

    def test_synthetic_baseline_produces_auditable_cable_candidate(self):
        report = fit_material(self.document, measurement_sha256="abc")
        self.assertTrue(report["fit_ready_for_isolated_dynamics"], report["gates"])
        self.assertFalse(report["formal_collection_allowed"])
        self.assertEqual(report["source_measurement_sha256"], "abc")
        self.assertTrue(all(report["gates"].values()))
        fit = report["elastic_fit"]
        self.assertGreater(fit["bend_pa"], 50e6)
        self.assertLess(fit["bend_pa"], 65e6)
        self.assertGreater(fit["twist_pa"], 18e6)
        self.assertLess(fit["twist_pa"], 22e6)
        self.assertGreater(fit["implied_poisson_ratio"], 0.3)
        self.assertLess(fit["implied_poisson_ratio"], 0.5)
        self.assertIsNone(report["mujoco"]["joint_damping_nms_per_rad"])

    def test_torsion_constant_matches_current_full_section(self):
        constant = rectangle_torsion_constant(0.03, 0.008)
        self.assertAlmostEqual(constant, 4.260202470716049e-09)

    def test_geometry_mismatch_blocks_isolated_dynamics(self):
        document = copy.deepcopy(self.document)
        document["geometry"]["width_m"] = [0.04, 0.04]
        report = fit_material(document)
        self.assertFalse(report["fit_ready_for_isolated_dynamics"])
        self.assertFalse(report["gates"]["width_matches_obj"])

    def test_extensible_measurement_cannot_be_forced_into_cable_fit(self):
        document = copy.deepcopy(self.document)
        document["axial_tension"]["trials"][1]["points"][-1]["extension_m"] = 0.006
        with self.assertRaisesRegex(ValueError, "requires cable_candidate"):
            fit_material(document)


if __name__ == "__main__":
    unittest.main()
