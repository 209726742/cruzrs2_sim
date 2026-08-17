#!/usr/bin/env python3

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

from strip_measurement_check import validate_measurements  # noqa: E402


TEMPLATE = os.path.join(
    ROOT,
    "docs",
    "strip_material_measurements_template.json",
)


def synthetic_complete_document():
    with open(TEMPLATE, encoding="utf-8") as handle:
        document = json.load(handle)
    document["specimen_id"] = "synthetic_test_only"
    document["geometry"] = {
        "mass_kg": [0.400, 0.401],
        "length_m": [1.600, 1.601],
        "width_m": [0.0300, 0.0301],
        "thickness_m": [0.0080, 0.0081],
        "natural_arch_height_m": [0.060, 0.061],
        "balance_offset_m": [0.001, -0.001],
    }
    document["axial_tension"]["gauge_length_m"] = 1.5
    for trial_index, trial in enumerate(document["axial_tension"]["trials"]):
        for point, extension in zip(trial["points"], [0.0, 0.001, 0.002, 0.004 + 0.001 * trial_index]):
            point["extension_m"] = extension
        trial["residual_extension_m"] = 0.0001

    document["three_point_bending"]["support_span_m"] = 1.2
    for trial in document["three_point_bending"]["trials"]:
        for point, load, deflection in zip(
            trial["points"], [0.0, 1.0, 2.0, 3.0], [0.0, 0.010, 0.021, 0.033]
        ):
            point["load_n"] = load
            point["center_deflection_m"] = deflection
        trial["residual_deflection_m"] = 0.001

    document["torsion"]["gauge_length_m"] = 1.2
    for trial in document["torsion"]["trials"]:
        for point, torque, angle in zip(trial["points"], [0.0, 0.1, 0.2], [0.0, 0.12, 0.25]):
            point["torque_nm"] = torque
            point["angle_rad"] = angle
        trial["residual_angle_rad"] = 0.01

    for trial in document["free_decay"]["trials"]:
        trial["initial_displacement_m"] = 0.05
        trial["peak_time_s"] = [0.1, 0.3, 0.5, 0.7]
        trial["peak_displacement_m"] = [0.040, 0.028, 0.019, 0.012]
        trial["stable_time_s"] = 1.0
    document["friction"] = {
        "shelf_critical_angle_deg": [20.0, 21.0, 19.5],
        "gripper_pad_critical_angle_deg": [35.0, 36.0, 34.5],
    }
    return document


class StripMeasurementCheckTest(unittest.TestCase):
    def test_blank_template_is_incomplete_and_never_fits_parameters(self):
        with open(TEMPLATE, encoding="utf-8") as handle:
            report = validate_measurements(json.load(handle))
        self.assertFalse(report["complete"])
        self.assertFalse(report["physical_parameters_generated"])
        self.assertEqual(report["model_decision"], "undetermined")
        self.assertGreater(len(report["errors"]), 10)

    def test_complete_repeated_measurements_select_cable_candidate(self):
        report = validate_measurements(synthetic_complete_document())
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["model_decision"], "cable_candidate")
        self.assertTrue(report["cable_gate"]["passed"])
        self.assertAlmostEqual(report["cable_gate"]["worst_measured_extension_m"], 0.005)
        self.assertFalse(report["physical_parameters_generated"])

    def test_worst_repeat_over_five_mm_selects_extensible_model(self):
        document = synthetic_complete_document()
        document["axial_tension"]["trials"][1]["points"][-1]["extension_m"] = 0.0051
        report = validate_measurements(document)
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["model_decision"], "requires_extensible_model")
        self.assertFalse(report["cable_gate"]["passed"])

    def test_synthetic_assumptions_are_never_formal_collection_evidence(self):
        document = synthetic_complete_document()
        document["provenance"] = {
            "kind": "synthetic_engineering_baseline",
            "measured": False,
            "formal_collection_allowed": False,
        }
        report = validate_measurements(document)
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["model_decision"], "cable_candidate")
        self.assertFalse(report["formal_collection_allowed"])
        self.assertFalse(report["measurement_provenance"]["measured"])
        self.assertTrue(any("model development only" in item for item in report["warnings"]))

    def test_bad_units_order_and_missing_repeat_are_rejected(self):
        document = synthetic_complete_document()
        document["measurement_units"] = "mixed"
        document["axial_tension"]["trials"][0]["points"][1]["load_n"] = 31.0
        document["friction"]["shelf_critical_angle_deg"] = [20.0]
        report = validate_measurements(document)
        self.assertFalse(report["complete"])
        self.assertTrue(any("measurement_units" in error for error in report["errors"]))
        self.assertTrue(any("strictly increasing" in error for error in report["errors"]))
        self.assertTrue(any("at least 3 repeats" in error for error in report["errors"]))

    def test_malformed_sections_and_trials_are_rejected_without_crashing(self):
        document = synthetic_complete_document()
        document["geometry"] = []
        document["torsion"]["trials"] = [None, "bad"]
        report = validate_measurements(document)
        self.assertFalse(report["complete"])
        self.assertTrue(any("geometry must be an object" in error for error in report["errors"]))
        self.assertTrue(any("torsion.trials[0] must be an object" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
