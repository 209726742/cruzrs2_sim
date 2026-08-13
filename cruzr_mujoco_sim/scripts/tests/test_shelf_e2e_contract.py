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

from shelf_e2e_contract import (
    ACTION_DIM,
    CAMERAS,
    CHUNK_SIZE,
    POLICY_IMAGE_MAP,
    STATE_DIM,
    make_state,
    validate_action_chunk,
    validate_policy_observation,
)


class ShelfE2EContractTest(unittest.TestCase):
    def observation(self):
        observation = {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in POLICY_IMAGE_MAP
        }
        observation["observation/state"] = np.zeros(STATE_DIM, dtype=np.float32)
        observation["prompt"] = "test"
        return observation

    def test_declared_dimensions_and_cameras(self):
        self.assertEqual(STATE_DIM, 18)
        self.assertEqual(ACTION_DIM, 18)
        self.assertEqual(CHUNK_SIZE, 50)
        self.assertEqual(
            CAMERAS,
            ("head_stereo_l_shelf", "chassis_front", "hand_right_shelf"),
        )

    def test_make_state_rejects_privileged_or_missing_dimensions(self):
        self.assertEqual(make_state(np.zeros(16), np.zeros(2)).shape, (STATE_DIM,))
        with self.assertRaises(ValueError):
            make_state(np.zeros(16), np.zeros(8))

    def test_policy_observation_is_exact(self):
        validate_policy_observation(self.observation())
        missing = self.observation()
        del missing["observation/right_wrist_image"]
        with self.assertRaises(ValueError):
            validate_policy_observation(missing)
        extra = self.observation()
        extra["observation/extra_image"] = np.zeros((224, 224, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_policy_observation(extra)

    def test_action_chunk_is_exact(self):
        good = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        self.assertEqual(validate_action_chunk(good).shape, good.shape)
        with self.assertRaises(ValueError):
            validate_action_chunk(np.zeros((CHUNK_SIZE - 1, ACTION_DIM)))


if __name__ == "__main__":
    unittest.main()
