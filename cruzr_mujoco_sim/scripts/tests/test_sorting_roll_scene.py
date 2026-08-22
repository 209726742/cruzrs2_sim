import os
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import sorting_roll_scene as scene


class SortingRollSceneTest(unittest.TestCase):
    def test_layout_contract(self):
        report = scene.layout_report()
        self.assertTrue(all(report["checks"].values()), report)
        self.assertAlmostEqual(report["edge_gap_m"], 0.475)
        self.assertAlmostEqual(report["table_yaw_deg"], 180.0)
        self.assertAlmostEqual(
            report["roll_depth_from_robot_side_m"], 0.52 / 3.0
        )
        self.assertAlmostEqual(
            report["roll_depth_fraction_from_robot_side"], 1.0 / 3.0
        )

    def test_table_rotation_and_roll_spawn_are_encoded_in_xml(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        bodies = {
            element.attrib["name"]: element.attrib
            for element in root.iter("body")
            if "name" in element.attrib
        }
        geoms = {
            element.attrib["name"]: element.attrib
            for element in root.iter("geom")
            if "name" in element.attrib
        }
        np.testing.assert_allclose(
            np.fromstring(bodies["sorting_table"]["quat"], sep=" "),
            [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.fromstring(bodies["sorting_roll"]["pos"], sep=" "),
            scene.ROLL_SPAWN,
            atol=1e-9,
        )
        self.assertAlmostEqual(
            np.fromstring(bodies["sorting_table"]["pos"], sep=" ")[1],
            -1.050,
        )
        self.assertAlmostEqual(
            np.fromstring(geoms["table_pedestal_col"]["pos"], sep=" ")[0],
            -0.120,
        )
        self.assertAlmostEqual(
            np.fromstring(geoms["table_base_col"]["pos"], sep=" ")[0],
            -0.120,
        )
        self.assertEqual(geoms["sorting_roll_col"]["condim"], "6")

    def test_pickup_supports_raise_roll_and_leave_flat_grasp_clear(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        geoms = {
            element.attrib["name"]: element.attrib
            for element in root.iter("geom")
            if "name" in element.attrib
        }
        bases = (
            geoms["roll_support_x_negative_base_col"],
            geoms["roll_support_x_positive_base_col"],
        )
        base_positions = [
            np.fromstring(base["pos"], sep=" ") for base in bases
        ]
        base_sizes = [
            np.fromstring(base["size"], sep=" ") for base in bases
        ]
        self.assertAlmostEqual(base_positions[0][0], -scene.ROLL_SUPPORT_X_M)
        self.assertAlmostEqual(base_positions[1][0], scene.ROLL_SUPPORT_X_M)
        for position, size in zip(base_positions, base_sizes):
            self.assertAlmostEqual(
                position[2] + size[2],
                scene.ROLL_SUPPORT_TOP_Z_M,
            )
            self.assertGreaterEqual(
                abs(position[0]) - size[0],
                0.19,
            )
        self.assertAlmostEqual(
            scene.ROLL_SPAWN[2] - scene.ROLL_SUPPORT_TOP_Z_M,
            scene.ROLL_RADIUS_M + 0.0015,
        )

    def test_template_contains_named_task_geometry(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        names = {
            element.attrib["name"]
            for element in root.iter()
            if "name" in element.attrib
        }
        self.assertTrue({
            "sorting_shelf",
            "sorting_table",
            "sorting_roll",
            "sorting_roll_free",
            "sorting_roll_col",
            "table_top_col",
            "roll_support_x_negative_base_col",
            "roll_support_x_positive_base_col",
            "shelf_top_trough_col",
            "sorting_roll_target",
        }.issubset(names))

    def test_template_uses_separate_lowpoly_visual_meshes(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        mesh_files = {
            element.attrib["name"]: element.attrib["file"]
            for element in root.iter("mesh")
            if element.attrib.get("name", "").startswith("sorting_")
        }
        self.assertEqual(
            set(mesh_files),
            {"sorting_roll_mesh", "sorting_shelf_mesh", "sorting_table_mesh"},
        )
        for mesh_file in mesh_files.values():
            self.assertTrue(mesh_file.endswith("_lowpoly.obj"), mesh_file)

    def test_view_command_defaults_to_egl_with_cpu_override(self):
        script = (scene.SORTING_ROOT / "run_scene.sh").read_text()
        self.assertIn("VIEWER_MODE=${TELEOP_VIEWER:-egl}", script)
        self.assertIn("GL_BACKEND=egl", script)
        self.assertIn("GL_BACKEND=glfw", script)
        self.assertIn("passive|glfw)", script)
        self.assertIn("TELEOP_VIEWER=$VIEWER_MODE", script)
        self.assertIn("TELEOP_EGL_FAST=${TELEOP_EGL_FAST:-1}", script)
        self.assertIn("EGL_W=${EGL_W:-1280}", script)
        self.assertIn("EGL_H=${EGL_H:-720}", script)
        self.assertIn("TELEOP_FPS=${TELEOP_FPS:-60}", script)
        self.assertIn("opencv-python==4.11.0.86", script)

    def test_top_tier_collision_matches_integrated_shelf_cross_section(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        geoms = {
            element.attrib["name"]: element.attrib
            for element in root.iter("geom")
            if "name" in element.attrib
        }
        integrated_names = {
            "shelf_top_front_lip_col",
            "shelf_top_trough_col",
            "shelf_top_back_slope_col",
            "shelf_top_back_panel_col",
        }
        self.assertTrue(integrated_names.issubset(geoms))
        self.assertTrue({
            "shelf_tier4_col",
            "target_slot_floor_col",
            "target_slot_front_guard_col",
            "target_slot_back_guard_col",
            "target_middle_bar_col",
            "target_top_bar_col",
        }.isdisjoint(geoms))

        trough = geoms["shelf_top_trough_col"]
        trough_pos = np.fromstring(trough["pos"], sep=" ")
        trough_size = np.fromstring(trough["size"], sep=" ")
        np.testing.assert_allclose(
            trough_pos,
            [scene.TOP_TIER_TROUGH_CENTER_X_M, 0.0, 0.884],
            atol=1e-12,
        )
        np.testing.assert_allclose(trough_size, [0.010, 0.285, 0.004])
        self.assertAlmostEqual(
            trough_pos[2] + trough_size[2],
            scene.TOP_TIER_TROUGH_TOP_Z_M,
        )
        np.testing.assert_allclose(
            scene.TARGET_CENTER,
            [
                scene.TOP_TIER_TROUGH_CENTER_X_M,
                0.0,
                scene.TOP_TIER_TROUGH_TOP_Z_M + scene.ROLL_RADIUS_M,
            ],
        )
        self.assertGreaterEqual(
            scene.TARGET_CENTER[0],
            scene.SHELF_BOUNDS[0, 0],
        )
        self.assertLessEqual(
            scene.TARGET_CENTER[0],
            scene.SHELF_BOUNDS[1, 0],
        )
        for name in integrated_names:
            self.assertEqual(geoms[name]["group"], "3")
            self.assertEqual(geoms[name]["contype"], "1")
            self.assertEqual(geoms[name]["conaffinity"], "3")
        self.assertIn("quat", geoms["shelf_top_front_lip_col"])
        self.assertIn("quat", geoms["shelf_top_back_slope_col"])

    def test_integrated_top_tier_adds_no_external_visual_slot(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        geoms = {
            element.attrib["name"]: element.attrib
            for element in root.iter("geom")
            if "name" in element.attrib
        }
        self.assertFalse(
            any(name.startswith("target_slot_") for name in geoms),
            sorted(geoms),
        )
        shelf_visual = geoms["sorting_shelf_visual"]
        self.assertEqual(shelf_visual["group"], "1")
        self.assertEqual(shelf_visual["contype"], "0")

if __name__ == "__main__":
    unittest.main()
