import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]
import sorting_roll_build_v21 as builder  # noqa: E402


class SortingRollBuildV21MixedTests(unittest.TestCase):
    def test_reused_videos_are_linked_without_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reuse = {}
            for camera in builder.POLICY_CAMERAS:
                path = root / f"{camera}.mp4"
                path.write_bytes(b"video")
                reuse[camera] = path
            with (
                mock.patch.object(builder, "encode_video") as encode,
                mock.patch.object(builder, "validate_video") as validate,
            ):
                builder.encode_episode_videos(
                    root / "source", root / "out", 0, 10, 1, reuse
                )
            encode.assert_not_called()
            self.assertEqual(validate.call_count, len(builder.POLICY_CAMERAS))

    def test_mixed_sources_are_declared_and_scenario_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index, version in enumerate((
                builder.DIVERSE_TASK_VERSION,
                "sorting_roll_v16_expansion_pilot_sim",
            )):
                episode = root / f"source_{index}"
                episode.mkdir()
                np.savez(
                    episode / "episode_data.npz",
                    state=np.zeros((2, 16), dtype=np.float32),
                    base_velocity=np.zeros((2, 2), dtype=np.float32),
                    action=np.zeros((2, 16), dtype=np.float32),
                    base_action=np.zeros((2, 2), dtype=np.float32),
                )
                sources.append({
                    "path": str(episode),
                    "seed": index + 1,
                    "split": "train" if index == 0 else "val",
                    "task_version": version,
                    "collection_profile": "profile",
                    "campaign": f"campaign_{index}",
                    "prompt": "move the roll",
                    "diversity": None,
                    "scenario": None if index == 0 else {"family": "R"},
                })
            out = root / "dataset"
            with mock.patch.object(builder, "encode_episode_videos"):
                builder.build_dataset(sources, out, 1)
            info = json.loads((out / "meta" / "info.json").read_text())
            self.assertEqual(info["source_task_version"], "mixed")
            self.assertEqual(
                info["source_task_versions"],
                sorted(source["task_version"] for source in sources),
            )
            self.assertEqual(
                info["source_campaigns"], ["campaign_0", "campaign_1"]
            )
            rows = [
                json.loads(line)
                for line in (out / "meta" / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[1]["source_scenario"], {"family": "R"})


if __name__ == "__main__":
    unittest.main()
