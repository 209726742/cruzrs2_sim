#!/usr/bin/env python3
"""Plan or launch a guarded 4/8-GPU dual-material collection campaign.

The default is plan-only. ``--execute`` requires an exact, ready schema-v2
report from shelf_e2e_collection_preflight.py. An explicitly labeled candidate
campaign may waive only the representative readiness-rate blocker; all
per-episode publication gates remain unchanged. Normal, boundary, and recovery
data use separate waves/run-ids; recovery is labeled, never silent noise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from cruzr_s2_sdk_contract import SDK_COLLECTION_PROFILE
from shelf_e2e_collection_preflight import visible_gpus


BATCH = os.path.join(HERE, "shelf_e2e_batch.sh")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
READINESS_BLOCKER_PATTERN = re.compile(
    r"^strict task readiness is \d+/26; formal collection requires at least 90% \(24/26\)$"
)


def distribute_successes(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (index < remainder) for index in range(count)]


def percent_count(total: int, percent: int) -> int:
    return 0 if percent == 0 else max(1, (total * percent + 50) // 100)


def split_data_target(
    total: int, boundary_percent: int, recovery_percent: int
) -> dict[str, int]:
    boundary = percent_count(total, boundary_percent)
    recovery = percent_count(total, recovery_percent)
    return {
        "normal": total - boundary - recovery,
        "boundary": boundary,
        "recovery": recovery,
    }


def build_wave(
    *,
    name: str,
    diversity_mode: str,
    layout_mode: str,
    target: int,
    gpu_count: int,
    seed_start: int,
    workers: int,
    attempt_factor: int,
    timeout_seconds: int,
    campaign: str,
    output_root: str,
    log_root: str,
) -> list[dict]:
    shards = []
    for slot, shard_target in enumerate(distribute_successes(target, gpu_count)):
        if shard_target == 0:
            continue
        max_attempts = shard_target * attempt_factor
        shard_seed_start = seed_start + slot
        run_id = f"{campaign}_{name}_g{slot}"
        argv = [
            "bash", BATCH,
            "--target-success", str(shard_target),
            "--workers", str(workers),
            "--gpu-id", str(slot),
            "--seed-start", str(shard_seed_start),
            "--seed-stride", str(gpu_count),
            "--max-attempts", str(max_attempts),
            "--run-id", run_id,
            "--output-shard", output_root,
            "--log-shard", log_root,
            "--timeout", str(timeout_seconds),
            "--collection-profile", SDK_COLLECTION_PROFILE,
            "--diversity-mode", diversity_mode,
            "--layout-mode", layout_mode,
        ]
        shards.append({
            "name": name,
            "diversity_mode": diversity_mode,
            "layout_mode": layout_mode,
            "slot": slot,
            "gpu_id": slot,
            "run_id": run_id,
            "target_success": shard_target,
            "seed_start": shard_seed_start,
            "seed_stride": gpu_count,
            "max_attempts": max_attempts,
            "seed_preview": [
                shard_seed_start + offset * gpu_count
                for offset in range(min(max_attempts, 8))
            ],
            "command_argv": argv,
            "command": shlex.join(argv),
        })
    return shards


def last_planned_seed(shards: list[dict]) -> int | None:
    if not shards:
        return None
    return max(
        shard["seed_start"] + (shard["max_attempts"] - 1) * shard["seed_stride"]
        for shard in shards
    )


def seed_overlap_errors(waves: list[dict]) -> list[str]:
    owners = {}
    errors = []
    for wave in waves:
        for shard in wave["shards"]:
            for offset in range(shard["max_attempts"]):
                seed = shard["seed_start"] + offset * shard["seed_stride"]
                if seed in owners:
                    errors.append(
                        f"seed {seed} belongs to both {owners[seed]} and {shard['run_id']}"
                    )
                owners[seed] = shard["run_id"]
    return errors


def build_plan(args) -> dict:
    targets = split_data_target(
        args.target_success_total, args.boundary_percent, args.recovery_percent
    )
    common = {
        "gpu_count": args.gpu_count,
        "workers": args.workers,
        "attempt_factor": args.attempt_factor,
        "timeout_seconds": args.timeout,
        "campaign": args.campaign,
        "output_root": os.path.abspath(args.output_root),
        "log_root": os.path.abspath(args.log_root),
    }
    normal = build_wave(
        name="normal",
        diversity_mode="clean",
        layout_mode="random",
        target=targets["normal"],
        seed_start=args.seed_start,
        **common,
    )
    last_normal = last_planned_seed(normal)
    boundary_seed_start = (
        args.seed_start if last_normal is None else last_normal + 1
    )
    boundary = build_wave(
        name="boundary",
        diversity_mode="clean",
        layout_mode="boundary",
        target=targets["boundary"],
        seed_start=boundary_seed_start,
        **common,
    )
    last_boundary = last_planned_seed(boundary)
    recovery_seed_start = (
        boundary_seed_start if last_boundary is None else last_boundary + 1
    )
    recovery = build_wave(
        name="recovery",
        diversity_mode="recovery",
        layout_mode="random",
        target=targets["recovery"],
        seed_start=recovery_seed_start,
        **common,
    )
    waves = [
        {"name": "normal", "target_success": targets["normal"], "shards": normal},
        {
            "name": "boundary",
            "target_success": targets["boundary"],
            "shards": boundary,
        },
        {
            "name": "recovery",
            "target_success": targets["recovery"],
            "shards": recovery,
        },
    ]
    overlaps = seed_overlap_errors(waves)
    if overlaps:
        raise ValueError("; ".join(overlaps))
    return {
        "schema_version": 1,
        "mode": (
            "plan_only" if not args.execute else
            "guarded_candidate_execute" if args.candidate_accept_readiness_blocker
            else "guarded_execute"
        ),
        "collection_profile": SDK_COLLECTION_PROFILE,
        "gpu_count": args.gpu_count,
        "target_success_total": args.target_success_total,
        "diversity_targets": targets,
        "boundary_percent": args.boundary_percent,
        "recovery_percent": args.recovery_percent,
        "campaign": args.campaign,
        "output_root": common["output_root"],
        "log_root": common["log_root"],
        "settings": {
            "workers": args.workers,
            "seed_start": args.seed_start,
            "attempt_factor": args.attempt_factor,
            "timeout_seconds": args.timeout,
        },
        "seed_sequences_disjoint": True,
        "candidate_readiness_override": args.candidate_accept_readiness_blocker,
        "waves": waves,
        "launch_performed": False,
    }


def preflight_errors(
    report: dict, plan: dict, allow_candidate_readiness: bool = False
) -> list[str]:
    errors = []
    expected = {
        "schema_version": 2,
        "mode": "plan_only_no_launch",
        "ready": not allow_candidate_readiness,
        "launch_performed": False,
        "collection_profile": SDK_COLLECTION_PROFILE,
        "gpu_count": plan["gpu_count"],
        "target_success_total": plan["target_success_total"],
        "campaign": plan["campaign"],
        "output_root": plan["output_root"],
        "log_root": plan["log_root"],
        "settings": plan["settings"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            errors.append(
                f"preflight {key}={report.get(key)!r}, expected {value!r}"
            )
    blockers = report.get("blockers")
    if not allow_candidate_readiness and blockers:
        errors.append(f"preflight still has blockers: {blockers}")
    if allow_candidate_readiness:
        if not plan["campaign"].startswith("candidate_"):
            errors.append("readiness override requires a candidate_ campaign")
        if (
            not isinstance(blockers, list)
            or len(blockers) != 1
            or not isinstance(blockers[0], str)
            or READINESS_BLOCKER_PATTERN.fullmatch(blockers[0]) is None
        ):
            errors.append(
                "candidate mode may waive only the 16/26-style readiness-rate blocker"
            )
        sweep = ((report.get("checks") or {}).get("representative_sweep") or {})
        result_count = sweep.get("result_count")
        if (
            result_count != 26
            or sweep.get("sdk_pass_count") != result_count
            or sweep.get("motion_pass_count") != result_count
            or sweep.get("collection_ready_pass_count") != sweep.get("task_pass_count")
        ):
            errors.append(
                "candidate mode still requires all 26 SDK/motion audits and all "
                "terminal gates for every successful task"
            )
    return errors


def write_manifest(path: str, payload: dict) -> None:
    temp = f"{path}.tmp_{os.getpid()}"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def terminate_process_groups(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_wave(shards: list[dict], resume: bool) -> list[dict]:
    processes = []
    try:
        for shard in shards:
            argv = list(shard["command_argv"])
            if resume:
                argv.append("--resume")
            processes.append(
                subprocess.Popen(argv, start_new_session=True)  # noqa: S603
            )
        return [
            {"run_id": shard["run_id"], "returncode": process.wait()}
            for shard, process in zip(shards, processes)
        ]
    except (OSError, KeyboardInterrupt, SystemExit):
        terminate_process_groups(processes)
        raise


def execute(plan: dict, preflight_path: str, resume: bool) -> int:
    with open(preflight_path, encoding="utf-8") as fh:
        report = json.load(fh)
    errors = preflight_errors(
        report,
        plan,
        allow_candidate_readiness=plan["candidate_readiness_override"],
    )
    gpus, gpu_error = visible_gpus()
    if gpu_error:
        errors.append(f"GPU discovery failed: {gpu_error}")
    available = {gpu["index"] for gpu in gpus}
    missing = sorted(set(range(plan["gpu_count"])) - available)
    if missing:
        errors.append(f"GPU ids are no longer visible: {missing}")
    roots = (plan["output_root"], plan["log_root"])
    if not resume:
        existing = [root for root in roots if os.path.exists(root)]
        if existing:
            errors.append(f"fresh campaign roots already exist: {existing}")
    if errors:
        raise ValueError("execution refused:\n- " + "\n- ".join(errors))

    os.makedirs(plan["output_root"], exist_ok=resume)
    os.makedirs(plan["log_root"], exist_ok=resume)
    manifest_path = os.path.join(
        plan["log_root"], f"{plan['campaign']}_multigpu_manifest.json"
    )
    if os.path.exists(manifest_path) and not resume:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    plan["launch_performed"] = True
    plan["status"] = "running"
    plan["started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    plan["preflight_report"] = os.path.abspath(preflight_path)
    plan["wave_results"] = []
    write_manifest(manifest_path, plan)

    for wave in plan["waves"]:
        results = run_wave(wave["shards"], resume)
        plan["wave_results"].append({"name": wave["name"], "shards": results})
        write_manifest(manifest_path, plan)
        if any(item["returncode"] != 0 for item in results):
            plan["status"] = "failed"
            plan["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(manifest_path, plan)
            return 1
    plan["status"] = "complete"
    plan["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(manifest_path, plan)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=(4, 8), required=True)
    parser.add_argument("--target-success-total", type=int, default=50)
    parser.add_argument("--boundary-percent", type=int, default=20)
    parser.add_argument("--recovery-percent", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--attempt-factor", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1400)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--preflight-report")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate-accept-readiness-blocker",
        action="store_true",
        help="candidate_ campaigns only: waive the readiness-rate blocker, and no other",
    )
    args = parser.parse_args(argv)
    for name in (
        "target_success_total", "workers", "seed_start", "attempt_factor", "timeout"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_success_total < args.gpu_count:
        parser.error("--target-success-total must be at least --gpu-count")
    if not 0 <= args.boundary_percent <= 40:
        parser.error("--boundary-percent must be between 0 and 40")
    if not 0 <= args.recovery_percent <= 30:
        parser.error("--recovery-percent must be between 0 and 30")
    if args.boundary_percent + args.recovery_percent > 40:
        parser.error("boundary + recovery percentages must not exceed 40")
    if not RUN_ID_PATTERN.fullmatch(args.campaign):
        parser.error("--campaign may contain only A-Z a-z 0-9 . _ -")
    if args.execute and not args.preflight_report:
        parser.error("--execute requires --preflight-report")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if (
        args.candidate_accept_readiness_blocker
        and not args.campaign.startswith("candidate_")
    ):
        parser.error("--candidate-accept-readiness-blocker requires candidate_ campaign")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        return execute(plan, os.path.abspath(args.preflight_report), args.resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
