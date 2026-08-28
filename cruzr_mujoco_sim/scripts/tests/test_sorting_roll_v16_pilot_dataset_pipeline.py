from pathlib import Path
import subprocess
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "sorting_roll_v16_pilot_dataset_pipeline.sh"
)


class SortingRollV16PilotDatasetPipelineTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_pipeline_requires_audit_before_canary(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sorting_roll_v16_dataset_audit.py", text)
        self.assertIn("sorting_roll_v16_sampling_weights.py", text)
        self.assertIn("ready_for_full_parameter_canary", text)
        self.assertIn("SORTING_ROLL_V16_CAMPAIGN", text)
        self.assertIn("SORTING_ROLL_V16_SOURCE_CAMPAIGN", text)
        self.assertIn("SORTING_ROLL_V16_TASK_VERSION", text)
        self.assertIn("SORTING_ROLL_V16_SAMPLING_PROFILE", text)
        self.assertIn("full_v2_old70", text)
        self.assertIn('audit.get("passed") is True', text)
        self.assertIn('sampling.get("passed") is True', text)


if __name__ == "__main__":
    unittest.main()
