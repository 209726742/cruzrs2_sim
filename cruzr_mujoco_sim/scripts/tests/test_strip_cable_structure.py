#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from strip_cable_structure import (  # noqa: E402
    DEFAULT_OBJ,
    build_structure_probe_xml,
    compile_structure_probe,
    load_strip_geometry,
    main,
    sample_nodes,
)


class StripCableStructureTest(unittest.TestCase):
    def test_source_obj_geometry_is_stable(self):
        geometry = load_strip_geometry(DEFAULT_OBJ)
        self.assertEqual(geometry["vertices"].shape, (484, 3))
        self.assertEqual(len(geometry["section_y"]), 121)
        self.assertAlmostEqual(geometry["width_m"], 0.030, places=6)
        self.assertAlmostEqual(geometry["thickness_m"], 0.008, places=6)
        self.assertAlmostEqual(geometry["centerline_arc_m"], 1.598964, places=6)

    def test_nodes_preserve_endpoints_and_arch(self):
        geometry = load_strip_geometry(DEFAULT_OBJ)
        nodes = sample_nodes(geometry)
        self.assertEqual(nodes.shape, (15, 3))
        np.testing.assert_allclose(nodes[[0, -1], 1], [-0.796702, 0.796702])
        self.assertAlmostEqual(nodes[7, 2] - nodes[0, 2], 0.060, places=6)

    def test_probe_compiles_expected_topology_and_geometry(self):
        model, report = compile_structure_probe(DEFAULT_OBJ)
        self.assertEqual((model.nq, model.nv), (59, 45))
        self.assertEqual(report["mode"], "structure_probe")
        self.assertIs(report["physical_parameters"], False)
        self.assertTrue(all(report["checks"].values()))
        self.assertLessEqual(max(report["compiled"]["bbox_abs_error_mm"]), 1.1)

    def test_sentinel_use_requires_explicit_cli_flag(self):
        with self.assertRaisesRegex(SystemExit, "--structure-probe"):
            main([])
        xml, _, _ = build_structure_probe_xml(DEFAULT_OBJ)
        self.assertIn("STRUCTURE PROBE ONLY", xml)
        self.assertNotIn("cart_shelf", xml)


if __name__ == "__main__":
    unittest.main()
