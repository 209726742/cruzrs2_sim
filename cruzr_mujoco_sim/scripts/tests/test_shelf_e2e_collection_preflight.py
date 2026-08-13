#!/usr/bin/env python3
"""Tests for deterministic, non-overlapping collection preflight plans."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_collection_preflight import (  # noqa: E402
    analyze_sweep,
    build_shards,
    seed_overlap_errors,
)


class ShelfE2ECollectionPreflightTest(unittest.TestCase):
    def make_shards(self, gpu_count, total=50):
        return build_shards(
            gpu_count=gpu_count,
            target_success_total=total,
            seed_start=1,
            workers=1,
            attempt_factor=4,
            campaign=f"test{gpu_count}",
            output_root="/tmp/output",
            log_root="/tmp/logs",
            timeout_seconds=1400,
        )

    def test_four_and_eight_gpu_plans_cover_total_without_seed_overlap(self):
        for gpu_count in (4, 8):
            with self.subTest(gpu_count=gpu_count):
                shards = self.make_shards(gpu_count)
                self.assertEqual(len(shards), gpu_count)
                self.assertEqual(sum(s["target_success"] for s in shards), 50)
                self.assertEqual(seed_overlap_errors(shards), [])
                self.assertEqual(
                    [s["seed_start"] for s in shards],
                    list(range(1, gpu_count + 1)),
                )
                self.assertTrue(all(s["seed_stride"] == gpu_count for s in shards))

    def test_commands_are_sdk_profile_plan_only_inputs(self):
        shards = self.make_shards(4)
        for shard in shards:
            argv = shard["command_argv"]
            self.assertIn("--collection-profile", argv)
            self.assertEqual(
                argv[argv.index("--collection-profile") + 1], "sdk_recovery_v1"
            )
            self.assertEqual(
                argv[argv.index("--diversity-mode") + 1], "clean"
            )
            self.assertEqual(
                argv[argv.index("--layout-mode") + 1], "random"
            )
            self.assertNotIn("--resume", argv)

    def test_sweep_summary_keeps_task_sdk_and_motion_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            for seed, task_pass in ((2, False), (11, True)):
                path = os.path.join(temp, f"seed_{seed:06d}")
                os.makedirs(path)
                with open(os.path.join(path, "result.json"), "w") as fh:
                    json.dump({
                        "seed": seed,
                        "passed": task_pass,
                        "sdk_alignment": {"passed": True},
                        "motion_quality": {"passed": True},
                    }, fh)
            summary = analyze_sweep(temp)
            self.assertEqual(summary["result_count"], 2)
            self.assertEqual(summary["task_pass_count"], 1)
            self.assertEqual(summary["sdk_pass_count"], 2)
            self.assertEqual(summary["motion_pass_count"], 2)


if __name__ == "__main__":
    unittest.main()
