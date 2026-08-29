from pathlib import Path
import subprocess
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "sorting_roll_v16_expansion_dataset_pipeline.sh"
)


class SortingRollV16ExpansionDatasetPipelineTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_pipeline_uses_expansion_contract_and_stage80_weights(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sorting_roll_v16_stage80_20260829_v51_safe_fast_nav", text)
        self.assertIn("sorting_roll_v16_stage80_20260829_v93_final_v51_manifest", text)
        self.assertIn("sorting_roll_v16_stage80_mixed320_20260829", text)
        self.assertIn(
            "$SIM_ROOT/out/collection/$SOURCE_CAMPAIGN/campaign_manifest.json",
            text,
        )
        self.assertIn("--manifest-kind expansion", text)
        self.assertIn("sorting_roll_v16_expansion_stage_sim", text)
        self.assertIn("SORTING_ROLL_V16_SAMPLING_PROFILE=stage80_old50", text)
        self.assertIn("SORTING_ROLL_V16_TRAIN_EPISODES=0:304", text)
        self.assertIn("SORTING_ROLL_V16_CANDIDATE_STAGE=v16_expansion_80", text)
        self.assertIn("run-tmux", text)
        self.assertIn("readiness", text)

    def test_pipeline_reuses_admitted_v15_train_sources(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sorting_roll_v15_diverse300_20260826_8x4090", text)
        self.assertIn("--v15-validation", text)
        self.assertIn("--v15-v21", text)


if __name__ == "__main__":
    unittest.main()
