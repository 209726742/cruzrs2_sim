from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "training" / "sorting_roll_v16_pilot_matched_canary.sh"


class SortingRollV16PilotMatchedCanaryTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_is_matched_control_and_treatment(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("gpu_ids=0,1", text)
        self.assertIn("gpu_ids=2,3", text)
        self.assertIn("--train-expert-only true", text)
        self.assertIn("--use-pretrained-stats true", text)
        self.assertIn("sampling_weights_old50_h15_t15_r20.npy", text)
        self.assertIn("--learning-rate 5e-6", text)
        self.assertIn("CANARY_STEPS=${CANARY_STEPS:-3000}", text)
        self.assertIn("CANARY_BATCH_SIZE=${CANARY_BATCH_SIZE:-8}", text)
        self.assertIn("PREFLIGHT_STEPS=${PREFLIGHT_STEPS:-5}", text)
        self.assertIn("PREFLIGHT_BATCH_SIZE=${PREFLIGHT_BATCH_SIZE:-8}", text)
        self.assertIn("SORTING_ROLL_V16_TREATMENT_DATASET", text)
        self.assertIn("SORTING_ROLL_V16_TREATMENT_WEIGHTS", text)
        self.assertIn("SORTING_ROLL_V16_CANARY_GROUPS", text)


if __name__ == "__main__":
    unittest.main()
