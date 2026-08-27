from pathlib import Path
import sys
import unittest


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
import sorting_roll_v16_dataset_audit as audit  # noqa: E402


class SortingRollV16DatasetAuditTests(unittest.TestCase):
    def test_scenario_audit_accepts_expected_families(self):
        rows = [
            {
                "source_task_version": audit.V15_TASK_VERSION,
                "source_split": "train",
                "source_scenario": None,
            }
            for _ in range(240)
        ]
        families = ["H"] * 4 + ["R"] * 8 + ["T"] * 4
        for index, family in enumerate(families):
            rows.append({
                "source_task_version": audit.V16_TASK_VERSION,
                "source_split": "train" if index < 12 else ("val" if index < 14 else "test"),
                "source_scenario": {
                    "scenario_family": family,
                    "scene_group_id": f"group_{index}",
                    "recorded_start_phase": "recovery_x" if family == "R" else "start",
                    "recorded_terminal_phase": "end",
                    "intervention_frame": -1 if family == "R" else None,
                    "recovery_start_frame": 0 if family == "R" else None,
                    "intervention_evidence": (
                        (
                            '"{\\"completed_before_recording\\":true}"'
                            if index == 0
                            else {"completed_before_recording": True}
                        )
                        if family == "R" else None
                    ),
                },
            })
        errors, counts = audit.scenario_errors(rows)
        self.assertEqual(errors, [])
        self.assertEqual(counts, audit.EXPECTED_FAMILIES)

    def test_scenario_audit_rejects_old_val_leakage(self):
        rows = [{
            "source_task_version": audit.V15_TASK_VERSION,
            "source_split": "val",
            "source_scenario": None,
        }]
        errors, _ = audit.scenario_errors(rows)
        self.assertTrue(any("leaked" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
