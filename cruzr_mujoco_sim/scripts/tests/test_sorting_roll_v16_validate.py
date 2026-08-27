import importlib.util
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path[:0] = [str(COLLECTION_DIR), str(CORE_DIR)]
SPEC = importlib.util.spec_from_file_location(
    "sorting_roll_v16_validate",
    COLLECTION_DIR / "sorting_roll_v16_validate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_assignment():
    return {
        "requested_transforms": {
            "pickup_support_and_roll": {
                "x_m": 0.0,
                "y_m": 0.01,
                "z_m": -0.01,
                "yaw_rad": 0.0,
            }
        }
    }


def valid_transform_report():
    before = [0.1, 0.2, 0.3]
    after = [0.1, 0.21, 0.29]
    return {
        "pickup_support_and_roll": {
            "applied_delta_m": [0.0, 0.01, -0.01],
            "visual_collision_and_roll_consistent": True,
            "support_geoms": {
                f"geom_{index}": {"before_m": before, "after_m": after}
                for index in range(12)
            },
            "roll_before_m": before,
            "roll_after_m": after,
        }
    }


class SortingRollV16ValidateTests(unittest.TestCase):
    def test_valid_transform_passes(self):
        self.assertEqual(
            MODULE.transform_errors(
                valid_assignment(), valid_transform_report()
            ),
            [],
        )

    def test_visual_collision_mismatch_fails(self):
        report = valid_transform_report()
        report["pickup_support_and_roll"][
            "visual_collision_and_roll_consistent"
        ] = False
        errors = MODULE.transform_errors(valid_assignment(), report)
        self.assertTrue(any("inconsistent" in error for error in errors))

    def test_roll_delta_mismatch_fails(self):
        report = valid_transform_report()
        report["pickup_support_and_roll"]["roll_after_m"] = [0.1, 0.2, 0.3]
        errors = MODULE.transform_errors(valid_assignment(), report)
        self.assertTrue(any("roll transform" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
