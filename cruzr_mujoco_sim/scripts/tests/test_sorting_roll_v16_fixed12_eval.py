from pathlib import Path
import subprocess
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
TRAINING = SCRIPTS / "training"
LAUNCHER = TRAINING / "sorting_roll_v16_fixed12_eval.sh"
ROLLOUT = TRAINING / "sorting_roll_pi05_fixed_rollout.py"
SERVER = TRAINING / "sorting_roll_pi05_official_server.py"


class SortingRollV16Fixed12EvalTests(unittest.TestCase):
    def test_entrypoints_have_valid_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROLLOUT), str(SERVER)],
            check=True,
        )

    def test_launcher_defines_matched_checkpoints_and_twelve_cases(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("LABELS=(original36 control3k treatment3k)", text)
        self.assertIn("checkpoints/036000/pretrained_model", text)
        self.assertEqual(text.count("v15:30"), 3)
        self.assertEqual(text.count("v16:60"), 9)
        self.assertIn("sorting_roll_v16_fixed12_audit.py", text)
        self.assertIn("tmux new-session -d", text)

    def test_rollout_uses_d405_contract_and_physical_stage_metrics(self):
        text = ROLLOUT.read_text(encoding="utf-8")
        self.assertNotIn("/share/home/", text)
        self.assertIn('"left_wrist_realsense"', text)
        self.assertIn('"right_wrist_realsense"', text)
        self.assertIn('MANIFEST_KIND == "v16"', text)
        self.assertIn('"reached_grasp_workzone"', text)
        self.assertIn('"stable_lift_at_least_70mm"', text)
        self.assertIn('"unsafe_collision"', text)
        self.assertIn('"continuous_rotation"', text)

    def test_server_uses_official_pi05_api_and_d405_feature_names(self):
        text = SERVER.read_text(encoding="utf-8")
        self.assertNotIn("/share/home/", text)
        self.assertIn('"observation.images.left_wrist_realsense"', text)
        self.assertIn('"observation.images.right_wrist_realsense"', text)
        self.assertIn("PI05Policy.predict_action_chunk", text)
        self.assertIn("--default-policy-seed", text)


if __name__ == "__main__":
    unittest.main()
