#!/usr/bin/env python3
"""End-to-end source validation tests for sdk_recovery_v1 episodes."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from cruzr_s2_sdk_contract import (  # noqa: E402
    ARM_JOINT_NAMES,
    ARM_POSITION_LIMITS_RAD,
    SDK_CAMERAS,
    SDK_COLLECTION_PROFILE,
    SDK_DOC_REVISION,
    SDK_TASK_HEAD_POSE_RAD,
    audit_sdk_episode,
)
from shelf_e2e_source import validate_source_dir  # noqa: E402


class ShelfE2ESourceSdkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = self.temp.name
        self.n = 51
        timestamp = ((np.arange(self.n) + 1.0) / 30.0).astype(np.float32)
        action = np.zeros((self.n, 16), dtype=np.float32)
        action[:, :14] = ARM_POSITION_LIMITS_RAD.mean(axis=1)
        action[:, 14:16] = 0.5
        state = action.copy()
        base_action = np.zeros((self.n, 2), dtype=np.float32)
        raw_timestamp = 100.0 + np.arange(self.n) / 30.0
        camera_timestamps = {
            camera: raw_timestamp.copy() for camera in SDK_CAMERAS
        }
        sdk_audit = audit_sdk_episode(
            state,
            action,
            base_action,
            fps=30,
            joint_names=ARM_JOINT_NAMES + ("grip_l", "grip_r"),
            cameras=SDK_CAMERAS,
            timestamp=timestamp,
            sdk_state_timestamp=raw_timestamp,
            camera_timestamps=camera_timestamps,
            require_camera_timestamps=True,
        )
        self.assertTrue(sdk_audit["passed"], sdk_audit["errors"])

        motion = {
            "passed": True,
            "tracking_passed": True,
            "tracking_enforced": True,
            "num_frames": self.n,
        }
        endpoint = {
            "reason": "both_objects_released_and_stable",
            "recorded_frames": self.n,
            "audit_frames": self.n,
        }
        episode_metadata = {
            "task_version": "dual_two_trip_v1",
            "seed": 2,
            "collection_profile": SDK_COLLECTION_PROFILE,
            "sdk_document_revision": SDK_DOC_REVISION,
            "sdk_task_head_pose_rad": dict(SDK_TASK_HEAD_POSE_RAD),
            "validation": {
                "passed": True,
                "motion_quality": dict(motion),
                "sdk_alignment": sdk_audit,
            },
            "policy_episode_end": endpoint,
        }
        self.meta = {
            "seed": 2,
            "success": True,
            "num_frames": self.n,
            "fps": 30,
            "resolution_hw": [224, 224],
            "cameras": {camera: camera for camera in SDK_CAMERAS},
            "state_joint_names": list(ARM_JOINT_NAMES) + ["grip_l", "grip_r"],
            "action_names": list(ARM_JOINT_NAMES) + ["grip_l", "grip_r"],
            "episode_metadata": episode_metadata,
        }
        self.result = {
            "seed": 2,
            "passed": True,
            "collection_profile": SDK_COLLECTION_PROFILE,
            "motion_quality": dict(motion),
            "sdk_alignment": sdk_audit,
            "policy_episode_end": endpoint,
            "safety_home": {
                "recorded_in_policy_episode": False,
                "objects_stable": True,
            },
        }
        self._write_json()
        np.savez(
            os.path.join(self.path, "episode_data.npz"),
            timestamp=timestamp,
            state=state,
            action=action,
            base=np.zeros((self.n, 3), dtype=np.float32),
            base_velocity=np.zeros((self.n, 2), dtype=np.float32),
            base_action=base_action,
            phase=np.full(self.n, "test"),
        )
        pose = np.zeros((self.n, 14), dtype=np.float32)
        pose[:, 3] = 1.0
        pose[:, 10] = 1.0
        np.savez(
            os.path.join(self.path, "object_poses.npz"),
            names=np.array(["pillar", "strip"]),
            pose=pose,
        )
        np.savez_compressed(
            os.path.join(self.path, "sdk_timestamps.npz"),
            state_timestamp=raw_timestamp,
            **{
                f"camera_{camera}_timestamp": values
                for camera, values in camera_timestamps.items()
            },
        )
        for camera in SDK_CAMERAS:
            frame_dir = os.path.join(self.path, "frames", camera)
            os.makedirs(frame_dir)
            first = os.path.join(frame_dir, "frame_000000.jpg")
            Image.new("RGB", (224, 224)).save(first)
            for index in range(1, self.n):
                os.link(first, os.path.join(frame_dir, f"frame_{index:06d}.jpg"))

    def tearDown(self):
        self.temp.cleanup()

    def _write_json(self):
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh)
        with open(os.path.join(self.path, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(self.result, fh)

    def test_complete_sdk_source_passes(self):
        info, errors = validate_source_dir(self.path)
        self.assertEqual(errors, [])
        self.assertEqual(info["collection_profile"], SDK_COLLECTION_PROFILE)
        self.assertEqual(tuple(info["cameras"]), SDK_CAMERAS)

    def test_camera_order_and_timestamp_sidecar_are_hard_gates(self):
        self.meta["cameras"] = {
            SDK_CAMERAS[1]: "waist",
            SDK_CAMERAS[0]: "left",
            SDK_CAMERAS[2]: "right",
        }
        self._write_json()
        os.unlink(os.path.join(self.path, "sdk_timestamps.npz"))
        _, errors = validate_source_dir(self.path)
        joined = "\n".join(errors)
        self.assertIn("SDK camera order", joined)
        self.assertIn("timestamps are required", joined)


if __name__ == "__main__":
    unittest.main()
