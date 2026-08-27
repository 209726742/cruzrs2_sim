from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lerobot.datasets.sampler import load_frame_sampling_weights  # noqa: E402


class FrameSamplingWeightsTests(unittest.TestCase):
    def test_loads_positive_finite_vector(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "weights.npy"
            np.save(path, np.asarray([0.25, 0.75], dtype=np.float64))

            weights = load_frame_sampling_weights(path, expected_length=2)

        self.assertEqual(weights.dtype, torch.float64)
        self.assertEqual(weights.tolist(), [0.25, 0.75])

    def test_rejects_wrong_length_and_nonpositive_values(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "weights.npy"
            np.save(path, np.asarray([1.0, 0.0], dtype=np.float64))

            with self.assertRaisesRegex(ValueError, "strictly positive"):
                load_frame_sampling_weights(path, expected_length=2)
            with self.assertRaisesRegex(ValueError, "length"):
                load_frame_sampling_weights(path, expected_length=3)


if __name__ == "__main__":
    unittest.main()
