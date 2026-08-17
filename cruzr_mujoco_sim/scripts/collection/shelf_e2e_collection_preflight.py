#!/usr/bin/env python3
"""Generate and validate a 4/8-GPU collection plan without launching it."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from cruzr_s2_sdk_contract import SDK_CAMERAS, SDK_COLLECTION_PROFILE


DEFAULT_SWEEP = os.path.join(
    ROOT, "out", "sweeps", "sdk_recovery_v1_preflight4_margin6_final_20260813"
)
DEFAULT_SMOKE = os.path.join(
    ROOT, "out", "smoke", "sdk_recovery_v1_capture_v2_20260813"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
REQUIRED_READINESS_SEEDS = tuple(range(1, 27))
MIN_TASK_READINESS_RATE = 0.90


def distribute_successes(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (index < remainder) for index in range(count)]


def build_shards(
    *,
    gpu_count: int,
    target_success_total: int,
    seed_start: int,
    workers: int,
    attempt_factor: int,
    campaign: str,
    output_root: str,
    log_root: str,
    timeout_seconds: int,
) -> list[dict]:
    targets = distribute_successes(target_success_total, gpu_count)
    shards = []
    batch = os.path.join(HERE, "shelf_e2e_batch.sh")
    for slot, target in enumerate(targets):
        run_id = f"{campaign}_g{slot}"
        max_attempts = target * attempt_factor
        shard_seed_start = seed_start + slot
        args = [
            "bash", batch,
            "--target-success", str(target),
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
            "--diversity-mode", "clean",
            "--layout-mode", "random",
        ]
        shards.append({
            "slot": slot,
            "gpu_id": slot,
            "run_id": run_id,
            "target_success": target,
            "workers": workers,
            "seed_start": shard_seed_start,
            "seed_stride": gpu_count,
            "max_attempts": max_attempts,
            "seed_preview": [
                shard_seed_start + offset * gpu_count
                for offset in range(min(max_attempts, 8))
            ],
            "command_argv": args,
            "command": shlex.join(args),
        })
    return shards


def seed_overlap_errors(shards: list[dict]) -> list[str]:
    owners = {}
    errors = []
    for shard in shards:
        for offset in range(shard["max_attempts"]):
            seed = shard["seed_start"] + offset * shard["seed_stride"]
            if seed in owners:
                errors.append(
                    f"seed {seed} belongs to both {owners[seed]} and {shard['run_id']}"
                )
            owners[seed] = shard["run_id"]
    return errors


def visible_gpus() -> tuple[list[dict], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if result.returncode != 0:
        return [], result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            return [], f"cannot parse nvidia-smi row: {line!r}"
        gpus.append({
            "index": int(fields[0]),
            "name": fields[1],
            "memory_mib": int(fields[2]),
        })
    return gpus, None


def analyze_sweep(path: str) -> dict:
    results = []
    errors = []
    for result_path in sorted(glob.glob(os.path.join(path, "seed_*", "result.json"))):
        try:
            with open(result_path, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read {result_path}: {exc}")
    return {
        "path": os.path.abspath(path),
        "result_count": len(results),
        "seeds": [result.get("seed") for result in results],
        "task_pass_count": sum(result.get("passed") is True for result in results),
        "sdk_pass_count": sum(
            (result.get("sdk_alignment") or {}).get("passed") is True
            for result in results
        ),
        "motion_pass_count": sum(
            (result.get("motion_quality") or {}).get("passed") is True
            for result in results
        ),
        "terminal_hold_pass_count": sum(
            (result.get("safety_home") or {}).get("tracking_passed") is True
            and (result.get("safety_home") or {}).get("release_passed") is True
            and (result.get("safety_home") or {}).get("objects_stable") is True
            and isinstance(
                (result.get("safety_home") or {}).get("strip_contact_force_peak_n"),
                (int, float),
            )
            and (result.get("safety_home") or {}).get(
                "strip_contact_force_peak_n"
            ) <= 0.2
            for result in results
        ),
        "collection_ready_pass_count": sum(
            result.get("passed") is True
            and (result.get("sdk_alignment") or {}).get("passed") is True
            and (result.get("motion_quality") or {}).get("passed") is True
            and (result.get("safety_home") or {}).get("tracking_passed") is True
            and (result.get("safety_home") or {}).get("release_passed") is True
            and (result.get("safety_home") or {}).get("objects_stable") is True
            and isinstance(
                (result.get("safety_home") or {}).get("strip_contact_force_peak_n"),
                (int, float),
            )
            and (result.get("safety_home") or {}).get(
                "strip_contact_force_peak_n"
            ) <= 0.2
            for result in results
        ),
        "errors": errors,
    }


def readiness_seed_errors(sweep: dict) -> list[str]:
    seeds = sweep.get("seeds") or []
    if seeds == list(REQUIRED_READINESS_SEEDS):
        return []
    return [
        "representative readiness sweep must contain exactly seeds 1-26; "
        f"got {seeds}"
    ]


def readiness_gate_errors(sweep: dict) -> list[str]:
    result_count = sweep["result_count"]
    task_pass_count = sweep["task_pass_count"]
    collection_ready_pass_count = sweep["collection_ready_pass_count"]
    required_pass_count = math.ceil(result_count * MIN_TASK_READINESS_RATE)
    errors = []
    if sweep["sdk_pass_count"] != result_count:
        errors.append("not every representative seed passes the SDK audit")
    if sweep["motion_pass_count"] != result_count:
        errors.append("not every representative seed passes motion quality")
    if collection_ready_pass_count != task_pass_count:
        errors.append(
            f"only {collection_ready_pass_count}/{task_pass_count} successful tasks "
            "pass all terminal-hold collection gates"
        )
    if task_pass_count < required_pass_count:
        errors.append(
            f"strict task readiness is {task_pass_count}/{result_count}; formal "
            f"collection requires at least {MIN_TASK_READINESS_RATE:.0%} "
            f"({required_pass_count}/{result_count})"
        )
    return errors


def analyze_smoke(path: str) -> dict:
    result_path = os.path.join(path, "capture_smoke_result.json")
    try:
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"path": os.path.abspath(path), "passed": False, "error": str(exc)}
    return {
        "path": os.path.abspath(path),
        "passed": result.get("passed") is True,
        "profile": result.get("collection_profile"),
        "cameras": list((result.get("frame_counts") or {}).keys()),
        "frames": result.get("frame_counts"),
        "timestamp_audit": (result.get("audit") or {}).get("camera_state_timestamp"),
    }


def nearest_existing_parent(path: str) -> str:
    current = os.path.abspath(path)
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=(4, 8), required=True)
    parser.add_argument("--target-success-total", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--attempt-factor", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1400)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--readiness-sweep", default=DEFAULT_SWEEP)
    parser.add_argument("--capture-smoke", default=DEFAULT_SMOKE)
    parser.add_argument("--episode-size-mib", type=float, default=300.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    for label in (
        "target_success_total", "workers", "seed_start", "attempt_factor", "timeout"
    ):
        if getattr(args, label) <= 0:
            parser.error(f"--{label.replace('_', '-')} must be positive")
    if args.target_success_total < args.gpu_count:
        parser.error("--target-success-total must allocate at least one success per GPU")
    if not RUN_ID_PATTERN.fullmatch(args.campaign):
        parser.error("--campaign may contain only A-Z a-z 0-9 . _ -")
    if args.episode_size_mib <= 0 or not math.isfinite(args.episode_size_mib):
        parser.error("--episode-size-mib must be positive and finite")

    output_root = os.path.abspath(args.output_root)
    log_root = os.path.abspath(args.log_root)
    report_path = os.path.abspath(args.report)
    if os.path.exists(report_path):
        raise SystemExit(f"refusing to overwrite immutable preflight report: {report_path}")

    shards = build_shards(
        gpu_count=args.gpu_count,
        target_success_total=args.target_success_total,
        seed_start=args.seed_start,
        workers=args.workers,
        attempt_factor=args.attempt_factor,
        campaign=args.campaign,
        output_root=output_root,
        log_root=log_root,
        timeout_seconds=args.timeout,
    )
    blockers = seed_overlap_errors(shards)
    warnings = []

    python_path = os.path.join(os.path.dirname(ROOT), "envs", "mjx", "bin", "python")
    batch_path = os.path.join(HERE, "shelf_e2e_batch.sh")
    validator_path = os.path.join(HERE, "shelf_e2e_source.py")
    script_checks = {
        "python_executable": os.path.isfile(python_path) and os.access(python_path, os.X_OK),
        "batch_script_exists": os.path.isfile(batch_path),
        "validator_exists": os.path.isfile(validator_path),
        "batch_shell_syntax": False,
    }
    if script_checks["batch_script_exists"]:
        shell_check = subprocess.run(
            ["bash", "-n", batch_path], capture_output=True, text=True
        )
        script_checks["batch_shell_syntax"] = shell_check.returncode == 0
    for name, passed in script_checks.items():
        if not passed:
            blockers.append(f"script check failed: {name}")

    gpus, gpu_error = visible_gpus()
    if gpu_error:
        blockers.append(f"GPU discovery failed: {gpu_error}")
    available_ids = {gpu["index"] for gpu in gpus}
    required_ids = set(range(args.gpu_count))
    missing_ids = sorted(required_ids - available_ids)
    if missing_ids:
        blockers.append(
            f"requested GPUs 0-{args.gpu_count - 1}, but IDs {missing_ids} are not visible"
        )

    sweep = analyze_sweep(args.readiness_sweep)
    if sweep["errors"]:
        blockers.extend(sweep["errors"])
    blockers.extend(readiness_seed_errors(sweep))
    if sweep["result_count"] == 0:
        blockers.append("representative readiness sweep has no result.json files")
    else:
        blockers.extend(readiness_gate_errors(sweep))

    smoke = analyze_smoke(args.capture_smoke)
    if not smoke.get("passed"):
        blockers.append(f"capture smoke did not pass: {smoke.get('error', 'unknown error')}")
    if smoke.get("profile") != SDK_COLLECTION_PROFILE:
        blockers.append("capture smoke collection profile does not match sdk_recovery_v1")
    if tuple(smoke.get("cameras") or ()) != SDK_CAMERAS:
        blockers.append(
            f"capture smoke cameras {smoke.get('cameras')} != {list(SDK_CAMERAS)}"
        )

    collision_paths = [path for path in (output_root, log_root) if os.path.exists(path)]
    if collision_paths:
        blockers.append(
            "planned campaign roots already exist: " + ", ".join(collision_paths)
        )

    max_attempts_total = sum(shard["max_attempts"] for shard in shards)
    estimated_gib = max_attempts_total * args.episode_size_mib * 1.2 / 1024.0
    disk_anchor = nearest_existing_parent(output_root)
    free_gib = shutil.disk_usage(disk_anchor).free / (1024.0 ** 3)
    if free_gib < estimated_gib:
        blockers.append(
            f"free storage {free_gib:.1f} GiB < conservative campaign estimate "
            f"{estimated_gib:.1f} GiB"
        )
    warnings.append(
        "workers=1 is a safe canary setting; increase only after measuring render throughput "
        "and GPU memory on the actual multi-GPU host"
    )

    report = {
        "schema_version": 2,
        "mode": "plan_only_no_launch",
        "ready": not blockers,
        "collection_profile": SDK_COLLECTION_PROFILE,
        "camera_contract": list(SDK_CAMERAS),
        "readiness_policy": {
            "representative_seeds": list(REQUIRED_READINESS_SEEDS),
            "minimum_task_pass_rate": MIN_TASK_READINESS_RATE,
            "minimum_task_pass_count": math.ceil(
                len(REQUIRED_READINESS_SEEDS) * MIN_TASK_READINESS_RATE
            ),
            "sdk_audit_required_for_all": True,
            "motion_quality_required_for_all": True,
            "all_gates_required_for_published_episode": True,
        },
        "gpu_count": args.gpu_count,
        "target_success_total": args.target_success_total,
        "campaign": args.campaign,
        "output_root": output_root,
        "log_root": log_root,
        "settings": {
            "workers": args.workers,
            "seed_start": args.seed_start,
            "attempt_factor": args.attempt_factor,
            "timeout_seconds": args.timeout,
        },
        "blockers": blockers,
        "warnings": warnings,
        "checks": {
            "scripts": script_checks,
            "visible_gpus": gpus,
            "representative_sweep": sweep,
            "capture_smoke": smoke,
            "campaign_roots_absent": not collision_paths,
            "seed_sequences_disjoint": not seed_overlap_errors(shards),
            "storage": {
                "anchor": disk_anchor,
                "free_gib": round(free_gib, 2),
                "estimated_required_gib": round(estimated_gib, 2),
                "episode_size_mib_assumption": args.episode_size_mib,
                "max_attempts_total": max_attempts_total,
            },
        },
        "shards": shards,
        "launch_performed": False,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
