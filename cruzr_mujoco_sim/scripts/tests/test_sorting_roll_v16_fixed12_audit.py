import json
from pathlib import Path
import sys
import tempfile
import unittest


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

from sorting_roll_v16_fixed12_audit import audit  # noqa: E402


FAMILIES = ("nominal",) * 3 + ("T",) * 3 + ("H",) * 3 + ("R",) * 3


def write_group(root, label, reached, grasped, lifted, success):
    for index, family in enumerate(FAMILIES):
        directory = root / label / f"case_{index:02d}"
        directory.mkdir(parents=True)
        payload = {
            "checkpoint_label": label,
            "manifest_kind": "v15" if family == "nominal" else "v16",
            "seed": 3000 + index,
            "scenario_family": family,
            "reached_grasp_workzone": index < reached,
            "first_bimanual_grasp_step": 1 if index < grasped else None,
            "stable_lift_at_least_70mm": index < lifted,
            "success": index < success,
            "unsafe_collision": False,
            "continuous_rotation": False,
        }
        (directory / "result.json").write_text(json.dumps(payload))


class SortingRollV16Fixed12AuditTests(unittest.TestCase):
    def test_treatment_must_pass_absolute_and_matched_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_group(root, "original36", 7, 5, 3, 1)
            write_group(root, "control3k", 9, 8, 7, 5)
            write_group(root, "treatment3k", 10, 9, 8, 6)
            report = audit(root)
        self.assertTrue(report["passed"])
        self.assertTrue(report["treatment_absolute_gate"])
        self.assertTrue(report["treatment_improves_matched_control"])
        self.assertTrue(report["ready_to_expand"])

    def test_equal_treatment_is_not_an_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_group(root, "original36", 7, 5, 3, 1)
            write_group(root, "control3k", 10, 9, 8, 6)
            write_group(root, "treatment3k", 10, 9, 8, 6)
            report = audit(root)
        self.assertTrue(report["treatment_absolute_gate"])
        self.assertFalse(report["treatment_improves_matched_control"])
        self.assertFalse(report["ready_to_expand"])

    def test_missing_results_fail_structure_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            report = audit(Path(temp))
        self.assertFalse(report["passed"])
        self.assertFalse(report["ready_to_expand"])


if __name__ == "__main__":
    unittest.main()
