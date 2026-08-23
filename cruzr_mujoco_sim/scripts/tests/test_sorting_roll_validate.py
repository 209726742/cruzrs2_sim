#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_validate import (
    FPS,
    POLICY_CAMERAS,
    payload_errors,
    source_split,
    timestamp_errors,
)


class SortingRollValidateTest(unittest.TestCase):
    def test_source_split_is_seed_deterministic(self):
        self.assertEqual(source_split(200), "test")
        self.assertEqual(source_split(201), "val")
        self.assertEqual(source_split(202), "train")
        with self.assertRaises(ValueError):
            source_split(0)

    def test_payload_contract_accepts_complete_finite_arrays(self):
        count = 51
        payload = {
            "timestamp": ((np.arange(count) + 1) / FPS).astype(np.float32),
            "state": np.zeros((count, 16), dtype=np.float32),
            "action": np.zeros((count, 16), dtype=np.float32),
            "action_real": np.ones(count, dtype=bool),
            "base": np.zeros((count, 3), dtype=np.float32),
            "base_velocity": np.zeros((count, 2), dtype=np.float32),
            "base_action": np.zeros((count, 2), dtype=np.float32),
            "phase": np.full(count, "test"),
            "roll_qpos": np.zeros((count, 7), dtype=np.float32),
            "roll_qvel": np.zeros((count, 6), dtype=np.float32),
        }
        self.assertEqual(payload_errors(payload, count), [])
        payload["state"][0, 0] = np.nan
        self.assertIn("state contains NaN/Inf", payload_errors(payload, count))

    def test_timestamp_contract_requires_three_synchronized_cameras(self):
        count = 51
        state = (np.arange(count) + 1) / FPS
        payload = {"state_timestamp": state}
        payload.update({
            f"camera_{camera}_timestamp": state.copy()
            for camera in POLICY_CAMERAS
        })
        self.assertEqual(timestamp_errors(payload, count), [])
        payload[f"camera_{POLICY_CAMERAS[0]}_timestamp"] += 0.021
        self.assertIn(
            f"{POLICY_CAMERAS[0]} timestamp skew exceeds 20 ms",
            timestamp_errors(payload, count),
        )


if __name__ == "__main__":
    unittest.main()
