#!/usr/bin/env python3

import copy
import sys
from pathlib import Path
import unittest

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

from sorting_roll_validate import (
    DIVERSE_TASK_VERSION,
    FPS,
    POLICY_CAMERAS,
    diversity_errors,
    episode_diversity_counts,
    payload_errors,
    source_split,
    timestamp_errors,
)
from sorting_roll_diversity import generate_manifest


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

    def test_v11_diversity_evidence_matches_assignment_and_detects_tampering(self):
        assignment = generate_manifest(
            "validator_contract", 14000, 1, "geometry_medium"
        )["assignments"][0]
        profile = assignment["object_profile"]
        diversity = {
            "assignment": assignment,
            "applied": {
                "schema_version": 1,
                "assignment_id": assignment["assignment_id"],
                "roll_length_m": profile["length_m"],
                "roll_diameter_m": profile["diameter_m"],
                "roll_mass_kg": assignment["dynamics_profile"]["mass_kg"],
                "roll_sliding_friction": assignment["dynamics_profile"][
                    "sliding_friction"
                ],
                "visual_mesh_span_m": [
                    profile["diameter_m"],
                    profile["diameter_m"],
                    profile["length_m"],
                ],
                "visual_length_axis": 2,
                "appearance_rgba": assignment["appearance_profile"]["rgba"],
                "visual_texture_disabled": True,
                "light_diffuse_scale": assignment["lighting_profile"][
                    "diffuse_scale"
                ],
                "jpeg_quality": assignment["image_profile"]["jpeg_quality"],
            },
            "manifest": "/candidate/manifest.json",
        }
        meta = {"prompt": assignment["prompt"], "diversity": diversity}
        episode_meta = {"diversity": diversity}
        result = {
            "task_version": DIVERSE_TASK_VERSION,
            "prompt": assignment["prompt"],
            "scene_randomization": {"pose_bin": assignment["pose_bin"]},
            "diversity": diversity,
        }
        self.assertEqual(diversity_errors(meta, result, episode_meta), [])

        tampered = copy.deepcopy(result)
        tampered["diversity"]["applied"]["roll_length_m"] += 0.01
        errors = diversity_errors(meta, tampered, episode_meta)
        self.assertIn("diversity metadata is missing or inconsistent", errors)

    def test_episode_diversity_counts_report_all_strata(self):
        manifest = generate_manifest("counts_contract", 14100, 20)
        infos = [
            {
                "task_version": DIVERSE_TASK_VERSION,
                "diversity": {"assignment": assignment},
            }
            for assignment in manifest["assignments"]
        ]
        self.assertEqual(
            episode_diversity_counts(infos),
            manifest["counts"],
        )


if __name__ == "__main__":
    unittest.main()
