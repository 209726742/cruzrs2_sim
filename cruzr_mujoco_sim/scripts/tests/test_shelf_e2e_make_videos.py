#!/usr/bin/env python3
"""Tests for metadata-driven multi-camera preview selection."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from cruzr_s2_sdk_contract import SDK_CAMERAS  # noqa: E402
from shelf_e2e_make_videos import (  # noqa: E402
    episode_cameras,
    validate_frame_counts,
)


class ShelfE2EMakeVideosTest(unittest.TestCase):
    def test_sdk_camera_order_comes_from_metadata(self):
        with tempfile.TemporaryDirectory() as path:
            cameras = SDK_CAMERAS
            with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"cameras": {camera: camera for camera in cameras}}, fh)
            for camera in cameras:
                frame_dir = os.path.join(path, "frames", camera)
                os.makedirs(frame_dir)
                for index in range(2):
                    open(os.path.join(frame_dir, f"frame_{index:06d}.jpg"), "wb").close()
            self.assertEqual(episode_cameras(path), cameras)
            self.assertEqual(validate_frame_counts(path, cameras), 2)

    def test_unequal_camera_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            cameras = ("a", "b")
            for count, camera in enumerate(cameras, start=1):
                frame_dir = os.path.join(path, "frames", camera)
                os.makedirs(frame_dir)
                for index in range(count):
                    open(os.path.join(frame_dir, f"frame_{index:06d}.jpg"), "wb").close()
            with self.assertRaisesRegex(ValueError, "unequal"):
                validate_frame_counts(path, cameras)


if __name__ == "__main__":
    unittest.main()
