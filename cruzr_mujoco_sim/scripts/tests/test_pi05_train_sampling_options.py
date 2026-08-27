from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / "pi05_train.sh"


class Pi05TrainSamplingOptionsTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_fresh_command_exposes_sampling_and_stats_options(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--frame-sampling-weights", text)
        self.assertIn('--dataset.frame_sampling_weights="$FRAME_SAMPLING_WEIGHTS"', text)
        self.assertIn('--dataset.use_pretrained_stats="$USE_PRETRAINED_STATS"', text)
        self.assertIn('PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT', text)


if __name__ == "__main__":
    unittest.main()
