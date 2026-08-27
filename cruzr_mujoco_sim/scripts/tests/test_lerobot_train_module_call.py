from pathlib import Path
import unittest


TRAINER = Path(__file__).resolve().parents[3] / "src" / "lerobot" / "scripts" / "lerobot_train.py"


class LerobotTrainModuleCallTests(unittest.TestCase):
    def test_update_policy_uses_module_call_for_distributed_hooks(self):
        text = TRAINER.read_text(encoding="utf-8")
        start = text.index("def update_policy(")
        end = text.index("\n\n@parser.wrap()", start)
        function_text = text[start:end]

        self.assertNotIn("policy.forward(", function_text)
        self.assertIn('policy(batch, reduction="none")', function_text)
        self.assertIn("policy(batch)", function_text)


if __name__ == "__main__":
    unittest.main()
