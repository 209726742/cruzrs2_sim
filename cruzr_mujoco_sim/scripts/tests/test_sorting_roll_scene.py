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
        self.assertAlmostEqual(report["edge_gap_m"], 0.7)
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
            np.fromstring(geoms["table_pedestal_col"]["pos"], sep=" ")[0],
            -0.120,
        )
        self.assertAlmostEqual(
            np.fromstring(geoms["table_base_col"]["pos"], sep=" ")[0],
            -0.120,
        )
        self.assertEqual(geoms["sorting_roll_col"]["condim"], "6")

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
            "target_slot_floor_col",
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

    def test_top_tier_measurements(self):
        root = ET.parse(scene.TEMPLATE_PATH).getroot()
        geoms = {
            element.attrib["name"]: element.attrib
            for element in root.iter("geom")
            if "name" in element.attrib
        }
        front_guard = geoms["target_slot_front_guard_col"]
        back_guard = geoms["target_slot_back_guard_col"]
        middle_bar = geoms["target_middle_bar_col"]
        top_bar = geoms["target_top_bar_col"]
        floor = geoms["target_slot_floor_col"]
        front_pos = np.fromstring(front_guard["pos"], sep=" ")
        front_size = np.fromstring(front_guard["size"], sep=" ")
        back_pos = np.fromstring(back_guard["pos"], sep=" ")
        back_size = np.fromstring(back_guard["size"], sep=" ")
        middle_pos = np.fromstring(middle_bar["pos"], sep=" ")
        top_pos = np.fromstring(top_bar["pos"], sep=" ")
        top_size = np.fromstring(top_bar["size"], sep=" ")
        floor_pos = np.fromstring(floor["pos"], sep=" ")
        floor_size = np.fromstring(floor["size"], sep=" ")
        floor_top = floor_pos[2] + floor_size[2]
        slot_width = (
            back_pos[0] - back_size[0]
            - (front_pos[0] + front_size[0])
        )
        post_left = geoms["shelf_post_front_left_col"]
        post_right = geoms["shelf_post_front_right_col"]
        left_pos = np.fromstring(post_left["pos"], sep=" ")
        left_size = np.fromstring(post_left["size"], sep=" ")
        right_pos = np.fromstring(post_right["pos"], sep=" ")
        right_size = np.fromstring(post_right["size"], sep=" ")
        shelf_inner_half_width = left_pos[1] - left_size[1]
        self.assertAlmostEqual(shelf_inner_half_width, 0.285)
        self.assertAlmostEqual(
            right_pos[1] + right_size[1], -shelf_inner_half_width
        )
        top_height = top_pos[2] + top_size[2]
        self.assertAlmostEqual(floor_top, 1.0)
        self.assertAlmostEqual(2.0 * front_size[2], 0.05)
        self.assertAlmostEqual(slot_width, 0.030)
        self.assertAlmostEqual(middle_pos[2] - floor_top, 0.13)
        self.assertAlmostEqual(top_height, 1.2)
        self.assertAlmostEqual(scene.TARGET_CENTER[2], 1.0125)
        self.assertAlmostEqual(floor_pos[0], scene.TARGET_CENTER[0])
        self.assertAlmostEqual(front_pos[0] + 0.0225, scene.TARGET_CENTER[0])
        self.assertAlmostEqual(back_pos[0] - 0.0225, scene.TARGET_CENTER[0])

if __name__ == "__main__":
    unittest.main()
