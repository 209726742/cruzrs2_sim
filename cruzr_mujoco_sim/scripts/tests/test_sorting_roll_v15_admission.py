#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path.insert(0, str(COLLECTION_DIR))
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_diversity import (  # noqa: E402
    generate_manifest,
    manifest_errors,
    source_split,
)
from sorting_roll_v15_finalize import make_replacement_plan  # noqa: E402


ADMISSION = COLLECTION_DIR / "sorting_roll_v15_admission.sh"
H100X2_ADMISSION = COLLECTION_DIR / "sorting_roll_v15_h100x2_admission.sh"
H100X2_COLLECT = COLLECTION_DIR / "sorting_roll_v15_h100x2_collect.sh"
H100X2_FINALIZE = COLLECTION_DIR / "sorting_roll_v15_h100x2_finalize.sh"
X4090_COLLECT = COLLECTION_DIR / "sorting_roll_v15_8x4090_collect.sh"
X4090_PIPELINE = COLLECTION_DIR / "sorting_roll_v15_8x4090_to_dataset.sh"
REVIEW_BUNDLE = COLLECTION_DIR / "sorting_roll_v15_review_bundle.sh"


class SortingRollV15AdmissionTest(unittest.TestCase):
    def test_shell_entrypoints_have_valid_syntax(self):
        for script in (
            ADMISSION,
            H100X2_ADMISSION,
            H100X2_COLLECT,
            H100X2_FINALIZE,
            X4090_COLLECT,
            X4090_PIPELINE,
            REVIEW_BUNDLE,
        ):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_admission_is_pinned_to_v15_camera_and_task(self):
        source = ADMISSION.read_text(encoding="utf-8")
        self.assertIn("sorting_roll_d405_candidate_v6", source)
        self.assertIn("sorting_roll_v15_diverse_sim", source)
        self.assertIn("--finalize-only", source)

    def test_configured_workers_cover_each_group_once(self):
        source = H100X2_ADMISSION.read_text(encoding="utf-8")
        self.assertIn("GPUS_CSV=0,1", source)
        self.assertIn("--gpus)", source)
        self.assertIn('slot=$((index % ${#gpus[@]}))', source)
        self.assertIn(
            'run_worker "${gpus[$index]}" "${worker_groups[$index]}"',
            source,
        )
        self.assertIn("--finalize-only", source)
        for group in (
            "dynamics_heavy_low_friction",
            "dynamics_light_high_friction",
            "geometry_long",
            "geometry_medium",
            "geometry_short",
        ):
            self.assertEqual(source.count(group), 1, group)

    def test_formal_collection_is_two_disjoint_150_episode_shards(self):
        source = H100X2_COLLECT.read_text(encoding="utf-8")
        self.assertIn('run_shard 0 3000', source)
        self.assertIn('run_shard 1 3150', source)
        self.assertEqual(source.count('--count 150'), 1)
        self.assertIn('--min-success 135', source)
        self.assertIn('"sorting_roll_v15_diverse_sim"', source)
        self.assertIn('successes >= 270', source)
    def test_eight_gpu_collection_is_disjoint_and_quota_gated(self):
        source = X4090_COLLECT.read_text(encoding="utf-8")
        self.assertIn("GPU_IDS=(0 1 2 3 4 5 6 7)", source)
        self.assertIn(
            "SEED_STARTS=(3000 3038 3076 3114 3152 3189 3226 3263)",
            source,
        )
        self.assertIn("COUNTS=(38 38 38 38 37 37 37 37)", source)
        self.assertIn("len(paths) != 8", source)
        self.assertIn("sorted(seeds) == list(range(3000, 3300))", source)
        self.assertIn("successes >= 270", source)
        finalize = (
            COLLECTION_DIR / "sorting_roll_v15_finalize.py"
        ).read_text(encoding="utf-8")
        self.assertIn('root.glob("shard_*/summary.json")', finalize)

    def test_pipeline_requires_validation_audit_and_review_bundle(self):
        source = X4090_PIPELINE.read_text(encoding="utf-8")
        for stage in (
            "sorting_roll_v15_8x4090_collect.sh",
            "sorting_roll_v15_h100x2_finalize.sh",
            "sorting_roll_v15_review_bundle.sh",
            "sorting_roll_v15_dataset_pipeline.sh",
        ):
            self.assertIn(stage, source)
        review = REVIEW_BUNDLE.read_text(encoding="utf-8")
        self.assertIn("--review-videos", review)
        self.assertIn('"object_profile"]["name"] == "short_slim"', review)
        self.assertIn('assignment["pose_bin"] == "boundary"', review)


    def test_replacement_plan_preserves_failed_strata(self):
        manifest = generate_manifest("v15_finalize_test", 3000, 300)
        failed_seeds = {3000, 3002}
        records = [
            {
                "seed": assignment["seed"],
                "passed": assignment["seed"] not in failed_seeds,
            }
            for assignment in manifest["assignments"]
        ]
        extended, plan = make_replacement_plan(manifest, records, 5)
        self.assertEqual(plan["initial_failed_count"], 2)
        self.assertEqual(manifest_errors(extended), [])
        self.assertEqual(
            extended["assignments"][:300], manifest["assignments"]
        )
        by_seed = {item["seed"]: item for item in extended["assignments"]}
        source_by_seed = {
            item["seed"]: item for item in manifest["assignments"]
        }
        for job in plan["jobs"]:
            source = source_by_seed[job["source_seed"]]
            self.assertEqual(len(job["candidate_seeds"]), 5)
            for seed in job["candidate_seeds"]:
                candidate = by_seed[seed]
                self.assertEqual(source_split(seed), source["split"])
                for field in (
                    "pose_bin",
                    "prompt_id",
                    "object_profile",
                    "appearance_profile",
                    "lighting_profile",
                    "dynamics_profile",
                    "image_profile",
                ):
                    self.assertEqual(candidate[field], source[field])


if __name__ == "__main__":
    unittest.main()
