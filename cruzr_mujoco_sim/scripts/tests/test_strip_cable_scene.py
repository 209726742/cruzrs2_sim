#!/usr/bin/env python3

import os
import sys
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from strip_cable_scene import (  # noqa: E402
    DEFAULT_OBJ,
    DEFAULT_TEMPLATE,
    build_scene,
    load_calibrated_parameters,
    RIGID_MODEL_ID,
    select_scene_template,
    SYNTHETIC_FLEX_MODEL_ID,
    SYNTHETIC_FLEX_V2_MODEL_ID,
    validate_scene,
)


CALIBRATED = os.path.join(
    ROOT,
    "assets",
    "shelf",
    "strip_cable_parameters_synthetic_v2_calibrated.json",
)
UNCALIBRATED = os.path.join(
    ROOT,
    "assets",
    "shelf",
    "strip_cable_parameters_synthetic_v2_candidate.json",
)


class StripCableSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, _ = load_calibrated_parameters(CALIBRATED)
        cls.scene = build_scene(DEFAULT_TEMPLATE, cls.parameters, DEFAULT_OBJ)
        cls.report = validate_scene(
            cls.scene,
            DEFAULT_TEMPLATE,
            cls.parameters,
            assets_root=os.path.join(ROOT, "assets"),
            settle_s=2.0,
        )

    def test_scene_compiles_contract_and_settles(self):
        self.assertTrue(
            self.report["scene_ready_for_norec"], self.report["checks"]
        )
        self.assertTrue(all(self.report["checks"].values()))
        self.assertEqual(self.report["compiled"]["strip_segments"], 14)
        self.assertEqual(self.report["compiled"]["pad_pair_count"], 56)
        self.assertLess(
            self.report["settle"]["max_segment_position_span_m_last_0p25s"],
            0.0001,
        )

    def test_scene_replaces_only_rigid_strip_model(self):
        self.assertEqual(self.scene.count('plugin="mujoco.elasticity.cable"'), 2)
        self.assertEqual(self.scene.count('<body name="strip"'), 1)
        self.assertNotIn('name="strip_col0"', self.scene)
        self.assertIn('name="pillar_col0"', self.scene)
        self.assertEqual(self.scene.count('<pair geom1="strip_G'), 56)

    def test_uncalibrated_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires calibrated"):
            load_calibrated_parameters(UNCALIBRATED)

    def test_synthetic_scene_templates_are_gated(self):
        rigid_path, rigid_manifest = select_scene_template(
            RIGID_MODEL_ID, norec=False
        )
        self.assertEqual(rigid_path, DEFAULT_TEMPLATE)
        self.assertTrue(rigid_manifest["formal_collection_allowed"])
        with self.assertRaisesRegex(ValueError, "not ready for NOREC"):
            select_scene_template(SYNTHETIC_FLEX_MODEL_ID, norec=True)
        flex_path, flex_manifest = select_scene_template(
            SYNTHETIC_FLEX_V2_MODEL_ID, norec=True
        )
        self.assertTrue(
            flex_path.endswith("template_strip_cable_v2_reinforced.xml")
        )
        self.assertFalse(flex_manifest["formal_collection_allowed"])
        with self.assertRaisesRegex(ValueError, "NOREC-only"):
            select_scene_template(SYNTHETIC_FLEX_V2_MODEL_ID, norec=False)


if __name__ == "__main__":
    unittest.main()
