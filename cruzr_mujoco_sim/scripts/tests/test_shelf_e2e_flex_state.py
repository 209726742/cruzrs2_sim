#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest

import mujoco
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_flex_state import (  # noqa: E402
    FLEX_ENCODING,
    FLEX_JOINT_NAMES,
    FLEX_OBJECT,
    FLEX_STATE_DIM,
    FLEX_TASK_VERSION,
    INTERNAL_STATE_FILE,
    RIGID_TASK_VERSION,
    capture_internal_state,
    flex_contract,
    internal_state_errors,
    load_internal_state,
    object_state_contract,
    restore_internal_state,
    save_internal_state,
)
from shelf_e2e_objects import object_info  # noqa: E402
from strip_cable_structure import compile_structure_probe  # noqa: E402


class ShelfE2EFlexStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = compile_structure_probe()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.info = object_info(self.model, FLEX_OBJECT)

    def test_structure_probe_has_the_versioned_flex_contract(self):
        version, contract = object_state_contract({FLEX_OBJECT: self.info})
        self.assertEqual(version, FLEX_TASK_VERSION)
        self.assertEqual(contract, flex_contract())
        self.assertEqual(self.info["ball_joint_names"], FLEX_JOINT_NAMES)
        state = capture_internal_state(self.data, self.info)
        self.assertEqual(state.shape, (FLEX_STATE_DIM,))
        np.testing.assert_allclose(state.reshape(-1, 4)[:, 0], 1.0)

    def test_archive_round_trip_and_model_restore(self):
        first = capture_internal_state(self.data, self.info)
        second = first.copy()
        second[:4] = [0.0, 1.0, 0.0, 0.0]
        rows = np.stack([first, second])
        metadata = {
            "task_version": FLEX_TASK_VERSION,
            "object_internal_state": flex_contract(),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_internal_state(directory, rows)
            self.assertEqual(internal_state_errors(directory, metadata, 2), [])
            loaded = load_internal_state(directory, metadata, 2)
        np.testing.assert_allclose(loaded, rows)

        root_before = self.data.qpos[self.info["free_qpos_adr"]:self.info["free_qpos_adr"] + 7].copy()
        restore_internal_state(self.data, self.info, loaded[1])
        np.testing.assert_allclose(capture_internal_state(self.data, self.info), loaded[1])
        np.testing.assert_allclose(
            self.data.qpos[self.info["free_qpos_adr"]:self.info["free_qpos_adr"] + 7],
            root_before,
        )

    def test_corrupt_quaternion_and_joint_order_are_rejected(self):
        row = capture_internal_state(self.data, self.info)
        bad = np.stack([row])
        bad[0, 0] = 2.0
        metadata = {
            "task_version": FLEX_TASK_VERSION,
            "object_internal_state": flex_contract(),
        }
        with tempfile.TemporaryDirectory() as directory:
            np.savez_compressed(
                os.path.join(directory, INTERNAL_STATE_FILE),
                object=np.asarray(FLEX_OBJECT),
                encoding=np.asarray(FLEX_ENCODING),
                joint_names=np.asarray(tuple(reversed(FLEX_JOINT_NAMES))),
                quaternion=bad,
            )
            errors = internal_state_errors(directory, metadata, 1)
        self.assertTrue(any("joint names/order" in error for error in errors))
        self.assertTrue(any("unit length" in error for error in errors))

    def test_rigid_version_has_no_internal_sidecar(self):
        rigid_info = {"ball_qpos_adrs": (), "ball_joint_names": ()}
        version, contract = object_state_contract({FLEX_OBJECT: rigid_info})
        self.assertEqual((version, contract), (RIGID_TASK_VERSION, None))
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"task_version": RIGID_TASK_VERSION}
            self.assertEqual(internal_state_errors(directory, metadata, 3), [])
            save_internal_state(directory, np.tile([1.0, 0.0, 0.0, 0.0], (3, 13)))
            errors = internal_state_errors(directory, metadata, 3)
        self.assertIn("rigid task must not contain object_internal_state.npz", errors)

    def test_flex_version_requires_the_sidecar_and_known_task_version(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {
                "task_version": FLEX_TASK_VERSION,
                "object_internal_state": flex_contract(),
            }
            errors = internal_state_errors(directory, metadata, 3)
            self.assertTrue(any(f"cannot read {INTERNAL_STATE_FILE}" in error for error in errors))
            errors = internal_state_errors(directory, {"task_version": "unknown"}, 3)
            self.assertEqual(errors, ["unsupported task_version: 'unknown'"])


if __name__ == "__main__":
    unittest.main()
