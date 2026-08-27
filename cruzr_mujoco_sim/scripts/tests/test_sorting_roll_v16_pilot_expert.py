import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]
SPEC = importlib.util.spec_from_file_location(
    "sorting_roll_v16_pilot_expert",
    COLLECTION_DIR / "sorting_roll_v16_pilot_expert.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SortingRollV16PilotExpertTests(unittest.TestCase):
    def test_support_stability_uses_window_excursion(self):
        metrics = MODULE.support_stability_metrics(
            [[0.0, 0.0, 0.0], [0.0005, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.001, 0.0]],
        )
        self.assertAlmostEqual(metrics["max_center_excursion_m"], 0.0005)
        self.assertLess(metrics["max_axis_excursion_deg"], 0.1)

    def test_support_stability_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.support_stability_metrics([[0.0, 0.0]], [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
