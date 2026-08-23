#!/usr/bin/env python3
"""Run recoverable sequential Sorting Roll seed validation or collection."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERT = SCRIPT_DIR / "sorting_roll_expert.py"
REQUIRED_STABLE_SECONDS = 2.0
MAX_EPISODE_SECONDS = 60.0


def result_errors(result):
    errors = []
    if result.get("success") is not True:
        errors.append("result.success is not true")
    if result.get("simulation_canary_eligible") is not True:
        errors.append("simulation_canary_eligible is not true")
    sim_seconds = result.get("sim_seconds")
    if (
        isinstance(sim_seconds, bool)
        or not isinstance(sim_seconds, (int, float))
        or sim_seconds > MAX_EPISODE_SECONDS
    ):
        errors.append("sim_seconds exceeds 60 seconds")
    evidence = result.get("final_evidence") or {}
    if evidence.get("instantaneous_success") is not True:
        errors.append("final evidence is not successful")
    if evidence.get("stable_seconds", 0.0) < REQUIRED_STABLE_SECONDS:
        errors.append("final stable window is shorter than 2 seconds")
    minute_gate = (result.get("gates") or {}).get(
        "episode_under_one_minute"
    ) or {}
    if minute_gate.get("passed") is not True:
        errors.append("one-minute gate did not pass")
    return errors


def write_summary(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--min-success", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--review-videos", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed_start < 1:
        parser.error("--seed-start must be positive")
    if args.count < 1:
        parser.error("--count must be positive")
    if not 1 <= args.min_success <= args.count:
        parser.error("--min-success must be in [1, count]")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.review_videos and not args.render:
        parser.error("--review-videos requires --render")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.out_root.resolve()
    if root.exists() and not args.resume:
        raise SystemExit(
            f"refusing to reuse existing batch root without --resume: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    records = []

    for seed in range(args.seed_start, args.seed_start + args.count):
        episode = root / f"seed_{seed:04d}"
        result_path = episode / "result.json"
        log_path = root / f"seed_{seed:04d}.log"
        returncode = None
        if result_path.exists() and args.resume:
            print(f"[batch] reuse seed={seed} result={result_path}", flush=True)
        else:
            if episode.exists():
                raise RuntimeError(
                    f"incomplete episode exists and is not reusable: {episode}"
                )
            command = [
                sys.executable,
                str(EXPERT),
                "--out",
                str(episode),
                "--seed",
                str(seed),
                "--gpu",
                str(args.gpu),
                "--randomize",
            ]
            if not args.render:
                command.append("--no-render")
            if args.review_videos:
                command.append("--review-videos")
            print(f"[batch] start seed={seed}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        command,
                        cwd=SCRIPT_DIR.parents[2],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124

        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            errors = result_errors(result)
            sim_seconds = result.get("sim_seconds")
        else:
            result = {}
            errors = ["result.json is missing"]
            sim_seconds = None
        record = {
            "seed": seed,
            "episode": str(episode),
            "log": str(log_path),
            "returncode": returncode,
            "passed": not errors,
            "sim_seconds": sim_seconds,
            "errors": errors,
        }
        records.append(record)
        successes = sum(item["passed"] for item in records)
        payload = {
            "seed_start": args.seed_start,
            "requested_count": args.count,
            "completed_count": len(records),
            "minimum_successes": args.min_success,
            "success_count": successes,
            "failed_count": len(records) - successes,
            "passed": (
                len(records) == args.count
                and successes >= args.min_success
            ),
            "render": bool(args.render),
            "review_videos": bool(args.review_videos),
            "records": records,
        }
        write_summary(summary_path, payload)
        print(
            f"[batch] seed={seed} passed={not errors} "
            f"sim_seconds={sim_seconds} successes={successes}/{len(records)}",
            flush=True,
        )

    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
