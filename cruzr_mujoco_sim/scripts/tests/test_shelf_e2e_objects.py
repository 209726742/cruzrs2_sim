#!/usr/bin/env python3

import os
import sys
import unittest

import mujoco
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_objects import (  # noqa: E402
    internal_ball_quaternions,
    object_info,
    root_pose,
    subtree_com,
)


MODEL_XML = """
<mujoco model="object_topology_test">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="rigid" pos="-1 0 0">
      <freejoint name="rigid_free"/>
      <geom name="rigid_geom" type="box" size="0.1 0.1 0.1" mass="2"/>
    </body>
    <body name="flex" pos="1 2 3">
      <freejoint name="flex_free"/>
      <geom name="flex_root_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="flex_segment_1" pos="0 1 0">
        <joint name="flex_ball_1" type="ball"/>
        <geom name="flex_geom_1" type="box" size="0.1 0.1 0.1" mass="2"/>
        <body name="flex_segment_2" pos="0 0 2">
          <joint name="flex_ball_2" type="ball"/>
          <geom name="flex_geom_2" type="box" size="0.1 0.1 0.1" mass="3"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class ShelfE2EObjectTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_rigid_object_keeps_single_body_contract(self):
        info = object_info(self.model, "rigid")
        self.assertEqual(len(info["bodies"]), 1)
        self.assertEqual(len(info["geoms"]), 1)
        self.assertEqual(info["mass_kg"], 2.0)
        self.assertEqual(info["ball_qpos_adrs"], ())
        self.assertEqual(info["ball_joint_names"], ())
        self.assertEqual(root_pose(self.data, info).shape, (7,))
        self.assertEqual(internal_ball_quaternions(self.data, info).shape, (0,))

    def test_articulated_object_collects_descendants_and_state(self):
        info = object_info(self.model, "flex")
        self.assertEqual(len(info["bodies"]), 3)
        self.assertEqual(len(info["geoms"]), 3)
        self.assertEqual(info["mass_kg"], 6.0)
        self.assertEqual(len(info["ball_qpos_adrs"]), 2)
        self.assertEqual(info["ball_joint_names"], ("flex_ball_1", "flex_ball_2"))
        np.testing.assert_allclose(
            internal_ball_quaternions(self.data, info),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(subtree_com(self.model, self.data, info), [1.0, 17.0 / 6.0, 4.0])

    def test_missing_object_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing object body"):
            object_info(self.model, "missing")


if __name__ == "__main__":
    unittest.main()
