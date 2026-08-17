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

from strip_cable_isolated import (  # noqa: E402
    build_isolated_xml,
    load_parameter_candidate,
    validate_isolated,
)


PARAMETERS = os.path.join(
    ROOT,
    "assets",
    "shelf",
    "strip_cable_parameters_synthetic_v1.json",
)


class StripCableIsolatedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, _ = load_parameter_candidate(PARAMETERS)

    def test_physical_candidate_compiles_with_expected_topology(self):
        xml, model, report = validate_isolated(self.parameters, duration_s=0.01)
        self.assertEqual((model.nq, model.nv), (59, 45))
        self.assertIn('value="56468444.5132"', xml)
        self.assertIn('value="19693228.6681"', xml)
        self.assertNotIn("STRUCTURE PROBE ONLY", xml)
        self.assertTrue(report["static_ready_for_damping_fit"], report["checks"])
        self.assertTrue(all(report["checks"].values()))

    def test_synthetic_provenance_remains_non_formal(self):
        _, _, report = validate_isolated(self.parameters, duration_s=0.001)
        self.assertEqual(
            report["source_measurement_provenance"]["kind"],
            "synthetic_engineering_baseline",
        )
        self.assertFalse(report["formal_collection_allowed"])

    def test_rejects_unpassed_or_flat_candidate(self):
        unpassed = copy.deepcopy(self.parameters)
        unpassed["fit_ready_for_isolated_dynamics"] = False
        flat = copy.deepcopy(self.parameters)
        flat["mujoco"]["flat"] = True
        for document, message in (
            (unpassed, "did not pass"),
            (flat, "flat=false"),
        ):
            with self.subTest(message=message):
                path = os.path.join("/tmp", f"strip_candidate_{message}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(document, handle)
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        load_parameter_candidate(path)
                finally:
                    os.unlink(path)

    def test_negative_damping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_isolated_xml(self.parameters, damping=-1.0)


if __name__ == "__main__":
    unittest.main()
