#!/usr/bin/env python3
"""Tests for guarded multi-GPU campaign planning."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
sys.path[:0] = [
    os.path.join(SCRIPTS_DIR, "collection"),
    os.path.join(SCRIPTS_DIR, "core"),
]

from shelf_e2e_multigpu_collect import (  # noqa: E402
    build_plan,
    parse_args,
    preflight_errors,
    seed_overlap_errors,
)


class ShelfE2EMultiGPUCollectTest(unittest.TestCase):
    def make_plan(self, gpu_count=4, total=50, boundary=20, recovery=10):
        args = parse_args([
            "--gpu-count", str(gpu_count),
            "--target-success-total", str(total),
            "--boundary-percent", str(boundary),
            "--recovery-percent", str(recovery),
            "--workers", "1",
            "--seed-start", "1",
            "--attempt-factor", "4",
            "--campaign", f"campaign{gpu_count}",
            "--output-root", "/tmp/e2e_output",
            "--log-root", "/tmp/e2e_logs",
        ])
        return build_plan(args)

    def test_four_and_eight_gpu_plans_have_exact_mix_and_unique_seeds(self):
        for gpu_count in (4, 8):
            with self.subTest(gpu_count=gpu_count):
                plan = self.make_plan(gpu_count)
                self.assertEqual(
                    plan["diversity_targets"],
                    {"normal": 35, "boundary": 10, "recovery": 5},
                )
                self.assertEqual(seed_overlap_errors(plan["waves"]), [])
                self.assertEqual(
                    sum(
                        shard["target_success"]
                        for wave in plan["waves"]
                        for shard in wave["shards"]
                    ),
                    50,
                )
                self.assertFalse(plan["launch_performed"])

    def test_commands_separate_normal_boundary_and_recovery_campaigns(self):
        plan = self.make_plan()
        run_ids = set()
        for wave in plan["waves"]:
            for shard in wave["shards"]:
                argv = shard["command_argv"]
                self.assertEqual(
                    argv[argv.index("--diversity-mode") + 1],
                    shard["diversity_mode"],
                )
                self.assertEqual(
                    argv[argv.index("--layout-mode") + 1],
                    shard["layout_mode"],
                )
                self.assertNotIn(shard["run_id"], run_ids)
                run_ids.add(shard["run_id"])

    def test_exact_ready_schema_v2_preflight_is_required(self):
        plan = self.make_plan()
        report = {
            "schema_version": 2,
            "mode": "plan_only_no_launch",
            "ready": True,
            "launch_performed": False,
            "collection_profile": plan["collection_profile"],
            "gpu_count": plan["gpu_count"],
            "target_success_total": plan["target_success_total"],
            "campaign": plan["campaign"],
            "output_root": plan["output_root"],
            "log_root": plan["log_root"],
            "settings": plan["settings"],
            "blockers": [],
        }
        self.assertEqual(preflight_errors(report, plan), [])
        report["ready"] = False
        report["blockers"] = ["task readiness 0/4"]
        errors = preflight_errors(report, plan)
        self.assertTrue(any("ready=False" in error for error in errors))
        self.assertTrue(any("blockers" in error for error in errors))

    def test_candidate_mode_waives_only_readiness_rate(self):
        args = parse_args([
            "--gpu-count", "4",
            "--target-success-total", "20",
            "--campaign", "candidate_canary",
            "--output-root", "/tmp/candidate_output",
            "--log-root", "/tmp/candidate_logs",
            "--candidate-accept-readiness-blocker",
        ])
        plan = build_plan(args)
        report = {
            "schema_version": 2,
            "mode": "plan_only_no_launch",
            "ready": False,
            "launch_performed": False,
            "collection_profile": plan["collection_profile"],
            "gpu_count": plan["gpu_count"],
            "target_success_total": plan["target_success_total"],
            "campaign": plan["campaign"],
            "output_root": plan["output_root"],
            "log_root": plan["log_root"],
            "settings": plan["settings"],
            "blockers": [
                "strict task readiness is 16/26; formal collection requires "
                "at least 90% (24/26)"
            ],
            "checks": {
                "representative_sweep": {
                    "result_count": 26,
                    "task_pass_count": 16,
                    "sdk_pass_count": 26,
                    "motion_pass_count": 26,
                    "collection_ready_pass_count": 16,
                }
            },
        }
        self.assertEqual(
            preflight_errors(report, plan, allow_candidate_readiness=True), []
        )
        report["blockers"].append("capture smoke did not pass")
        errors = preflight_errors(report, plan, allow_candidate_readiness=True)
        self.assertTrue(any("may waive only" in error for error in errors))



if __name__ == "__main__":
    unittest.main()
