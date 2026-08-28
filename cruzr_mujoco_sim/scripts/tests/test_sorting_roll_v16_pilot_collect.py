from pathlib import Path
import subprocess
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PACKAGE_ROOT
    / "scripts"
    / "collection"
    / "sorting_roll_v16_pilot_collect.sh"
)


class SortingRollV16PilotCollectTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_launcher_is_resume_safe(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("already successful; skip", text)
        self.assertIn("failed immutable result", text)
        self.assertNotIn("rm -", text)

    def test_representative_mode_covers_full_trajectory_families(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("SEEDS=(5000 5003 5005 5010)", text)


if __name__ == "__main__":
    unittest.main()
