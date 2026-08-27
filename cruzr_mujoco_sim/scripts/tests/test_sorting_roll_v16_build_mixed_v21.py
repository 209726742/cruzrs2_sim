from pathlib import Path
import sys
import tempfile
import unittest


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]
import sorting_roll_v16_build_mixed_v21 as mixed  # noqa: E402


class SortingRollV16BuildMixedV21Tests(unittest.TestCase):
    def test_split_counts(self):
        sources = (
            [{"split": "train"}] * 252
            + [{"split": "val"}] * 2
            + [{"split": "test"}] * 2
        )
        self.assertEqual(mixed.split_counts(sources), mixed.EXPECTED_COUNTS)

    def test_reuse_video_paths_require_all_policy_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for camera in mixed.builder.POLICY_CAMERAS:
                path = (
                    root
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{camera}"
                    / "episode_000000.mp4"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            paths = mixed.reuse_video_paths(root, 0)
            self.assertEqual(set(paths), set(mixed.builder.POLICY_CAMERAS))


if __name__ == "__main__":
    unittest.main()
