#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest
import uuid
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_v16_counterfactual_scene import (  # noqa: E402
    SUPPORT_NAMES,
    materialize_counterfactual_scene,
)
from sorting_roll_v16_expansion_contract import (  # noqa: E402
    generate_expansion_manifest,
)


def named(root, tag, name):
    return next(item for item in root.iter(tag) if item.get("name") == name)


class SortingRollV16CounterfactualSceneTests(unittest.TestCase):
    def setUp(self):
        manifest = generate_expansion_manifest("counterfactual_scene_test")
        counterfactuals = [
            item for item in manifest["assignments"]
            if item["scenario_family"] == "C"
        ]
        pair_id = counterfactuals[0]["counterfactual_pair_id"]
        self.pair = [
            item for item in counterfactuals
            if item["counterfactual_pair_id"] == pair_id
        ]
        self.base = PACKAGE_ROOT / "assets" / "sorting_roll_scene.xml"
        token = uuid.uuid4().hex
        self.outputs = [
            self.base.parent / f"sorting_roll_v16_c_test_{token}_{index}.xml"
            for index in range(2)
        ]

    def tearDown(self):
        for path in self.outputs:
            path.unlink(missing_ok=True)

    def test_pair_has_two_supported_rolls_and_fixed_visible_lanes(self):
        signatures = []
        for assignment, output in zip(self.pair, self.outputs):
            report = materialize_counterfactual_scene(
                self.base, output, assignment
            )
            self.assertTrue(report["pair_visible_layout_invariant"])
            root = ET.parse(output).getroot()
            target = named(root, "body", "sorting_roll")
            distractor = named(root, "body", "sorting_roll_distractor")
            self.assertAlmostEqual(
                float(target.get("pos").split()[0]),
                assignment["counterfactual_scene"]["lane_x_m"][
                    assignment["target_lane"]
                ],
            )
            self.assertAlmostEqual(
                float(distractor.get("pos").split()[0]),
                assignment["counterfactual_scene"]["lane_x_m"][
                    assignment["counterfactual_scene"]["distractor_lane"]
                ],
            )
            for name in SUPPORT_NAMES:
                named(root, "geom", name)
                named(root, "geom", f"distractor_{name}")
            signatures.append((
                tuple(sorted(
                    assignment["counterfactual_scene"]["lane_x_m"].items()
                )),
                tuple(sorted(
                    assignment["counterfactual_scene"]["lane_colors"].items()
                )),
            ))
            self.assertAlmostEqual(
                float(named(root, "geom", "table_top_col").get("size").split()[0]),
                0.6,
            )
            self.assertAlmostEqual(
                float(named(root, "geom", "table_top_col").get("pos").split()[0]),
                -0.31,
            )
            self.assertAlmostEqual(
                float(named(root, "mesh", "sorting_table_mesh").get("scale").split()[0]),
                0.6,
            )
        self.assertEqual(signatures[0], signatures[1])

    def test_refuses_non_counterfactual_assignment(self):
        assignment = dict(self.pair[0])
        assignment["scenario_family"] = "T"
        with self.assertRaises(ValueError):
            materialize_counterfactual_scene(
                self.base, self.outputs[0], assignment
            )


if __name__ == "__main__":
    unittest.main()
