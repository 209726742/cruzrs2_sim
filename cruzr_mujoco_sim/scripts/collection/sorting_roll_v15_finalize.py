#!/usr/bin/env python3
"""Collect quota-preserving v15 replacements and select 300 valid sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from time import strftime


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path.insert(0, str(CORE_DIR))

from sorting_roll_diversity import (  # noqa: E402
    assignment_for_seed,
    generate_manifest,
    load_manifest,
    manifest_counts,
    manifest_errors,
    replacement_assignment,
    source_split,
)


TASK_VERSION = "sorting_roll_v15_diverse_sim"
EXPECTED_INITIAL_COUNT = 300
DEFAULT_ATTEMPTS_PER_FAILURE = 5


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_initial_records(root, manifest):
    summary_paths = sorted(root.glob("shard_*/summary.json"))
    if not summary_paths:
        raise ValueError("no initial shard summaries found")
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in summary_paths
    ]
    records = [
        record
        for summary in summaries
        for record in summary.get("records", [])
    ]
    seeds = [int(record["seed"]) for record in records]
    expected_seeds = [item["seed"] for item in manifest["assignments"]]
    if len(records) != EXPECTED_INITIAL_COUNT:
        raise ValueError(f"expected 300 initial records, got {len(records)}")
    if sorted(seeds) != expected_seeds or len(set(seeds)) != len(seeds):
        raise ValueError("initial shard seeds do not match the campaign manifest")
    for record in records:
        if record.get("task_version") != TASK_VERSION:
            raise ValueError(f"seed {record['seed']} has the wrong task version")
        recorded = (record.get("diversity") or {}).get("assignment")
        expected = assignment_for_seed(manifest, record["seed"])
        if recorded != expected:
            raise ValueError(f"seed {record['seed']} assignment mismatch")
    return sorted(records, key=lambda record: int(record["seed"]))


def make_replacement_plan(manifest, records, attempts_per_failure):
    failed = [record for record in records if not record.get("passed")]
    jobs = [
        {
            "job_index": index,
            "source_seed": int(record["seed"]),
            "split": assignment_for_seed(manifest, record["seed"])["split"],
            "candidate_seeds": [],
        }
        for index, record in enumerate(failed)
    ]
    candidate_seed = manifest["seed_start"] + manifest["count"]
    while any(
        len(job["candidate_seeds"]) < attempts_per_failure
        for job in jobs
    ):
        split = source_split(candidate_seed)
        for job in jobs:
            if (
                job["split"] == split
                and len(job["candidate_seeds"]) < attempts_per_failure
            ):
                job["candidate_seeds"].append(candidate_seed)
                break
        candidate_seed += 1

    if jobs:
        extended_count = candidate_seed - manifest["seed_start"]
        extended = generate_manifest(
            manifest["campaign"],
            manifest["seed_start"],
            extended_count,
        )
        extended["assignments"][:manifest["count"]] = manifest["assignments"]
        assignments = {
            item["seed"]: item for item in extended["assignments"]
        }
        for job in jobs:
            source = assignment_for_seed(manifest, job["source_seed"])
            for seed in job["candidate_seeds"]:
                assignments[seed] = replacement_assignment(source, seed)
        extended["assignments"] = [
            assignments[seed]
            for seed in range(
                extended["seed_start"],
                extended["seed_start"] + extended["count"],
            )
        ]
        extended["counts"] = manifest_counts(extended["assignments"])
    else:
        extended = json.loads(json.dumps(manifest))
    errors = manifest_errors(extended)
    if errors:
        raise ValueError("invalid extended manifest: " + "; ".join(errors))
    plan = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "campaign": manifest["campaign"],
        "initial_attempted_count": len(records),
        "initial_success_count": len(records) - len(failed),
        "initial_failed_count": len(failed),
        "attempts_per_failure": attempts_per_failure,
        "jobs": jobs,
    }
    return extended, plan


def reusable_candidate(candidate_root):
    summary_path = candidate_root / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is True and summary.get("success_count") == 1:
        record = summary["records"][0]
        return {
            "passed": True,
            "candidate_seed": int(record["seed"]),
            "episode": record["episode"],
            "reused": True,
        }
    return {"passed": False, "reused": True}


def run_replacement_job(root, manifest_path, python_bin, gpu, job):
    job_root = (
        root / "replacements"
        / f"job_{job['job_index']:03d}_source_{job['source_seed']}"
    )
    job_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for seed in job["candidate_seeds"]:
        candidate_root = job_root / f"candidate_{seed}"
        reusable = reusable_candidate(candidate_root)
        if reusable is not None:
            attempts.append({"seed": seed, **reusable})
            if reusable["passed"]:
                return {**job, "gpu": gpu, "attempts": attempts, **reusable}
            continue
        if candidate_root.exists():
            rejected = job_root / "rejected_incomplete"
            rejected.mkdir(exist_ok=True)
            destination = rejected / (
                f"candidate_{seed}_{strftime('%Y%m%dT%H%M%S')}"
            )
            candidate_root.replace(destination)
        candidate_root.mkdir(parents=True)
        command = [
            str(python_bin),
            str(SCRIPT_DIR / "sorting_roll_batch.py"),
            "--out-root", str(candidate_root),
            "--seed-start", str(seed),
            "--count", "1",
            "--min-success", "1",
            "--gpu", str(gpu),
            "--timeout", "1800",
            "--render",
            "--resume",
            "--manifest", str(manifest_path),
        ]
        with (candidate_root / "runner.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        reusable = reusable_candidate(candidate_root) or {"passed": False}
        attempt = {
            "seed": seed,
            "returncode": completed.returncode,
            **reusable,
        }
        attempts.append(attempt)
        if reusable["passed"]:
            return {**job, "gpu": gpu, "attempts": attempts, **reusable}
    return {**job, "gpu": gpu, "attempts": attempts, "passed": False}


def run_worker(root, manifest_path, python_bin, gpu, jobs):
    return [
        run_replacement_job(root, manifest_path, python_bin, gpu, job)
        for job in jobs
    ]


def collect_replacements(root, manifest_path, python_bin, gpus, jobs):
    if not jobs:
        return []
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                run_worker,
                root,
                manifest_path,
                python_bin,
                gpu,
                queue,
            )
            for gpu, queue in zip(gpus, queues)
            if queue
        ]
        results = [item for future in futures for item in future.result()]
    return sorted(results, key=lambda result: result["job_index"])


def select_sources(root, manifest, extended, records, replacements):
    unresolved = [result for result in replacements if not result.get("passed")]
    if unresolved:
        raise RuntimeError(
            "replacement jobs exhausted: "
            + ", ".join(str(item["source_seed"]) for item in unresolved)
        )
    selected = [record["episode"] for record in records if record.get("passed")]
    selected.extend(result["episode"] for result in replacements)
    selected_assignments = [
        assignment_for_seed(extended, int(Path(path).name.split("_")[-1]))
        for path in selected
    ]
    counts = manifest_counts(selected_assignments)
    selected_seeds = [item["seed"] for item in selected_assignments]
    passed = (
        len(selected) == EXPECTED_INITIAL_COUNT
        and len(set(selected_seeds)) == EXPECTED_INITIAL_COUNT
        and counts == manifest["counts"]
    )
    (root / "selected_sources.txt").write_text(
        "".join(f"{Path(path).resolve()}\n" for path in selected),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "campaign": manifest["campaign"],
        "selected_count": len(selected),
        "unique_seed_count": len(set(selected_seeds)),
        "replacement_seeds": [
            result["candidate_seed"] for result in replacements
        ],
        "counts": counts,
        "matches_original_300_quotas": counts == manifest["counts"],
        "passed": passed,
    }
    write_json(root / "selection_report.json", report)
    if not passed:
        raise RuntimeError("selected v15 sources do not preserve the 300 quotas")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--attempts-per-failure",
        type=int,
        default=DEFAULT_ATTEMPTS_PER_FAILURE,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    python_bin = PROJECT_ROOT / "envs" / "mjx" / "bin" / "python"
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus must contain distinct non-negative IDs")
    if args.attempts_per_failure < 1:
        raise ValueError("--attempts-per-failure must be positive")
    manifest_path = root / "campaign_manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest["task_version"] != TASK_VERSION:
        raise ValueError("campaign manifest is not v15")
    records = load_initial_records(root, manifest)
    extended, plan = make_replacement_plan(
        manifest, records, args.attempts_per_failure
    )
    extended_path = root / "campaign_manifest_with_replacements.json"
    write_json(extended_path, extended)
    plan["extended_manifest"] = str(extended_path)
    write_json(root / "replacement_plan.json", plan)
    write_json(root / "failed_sources.json", {
        "count": plan["initial_failed_count"],
        "records": [record for record in records if not record.get("passed")],
    })
    replacements = collect_replacements(
        root, extended_path, python_bin, gpus, plan["jobs"]
    )
    collection_report = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "job_count": len(plan["jobs"]),
        "resolved_count": sum(item.get("passed", False) for item in replacements),
        "passed": all(item.get("passed", False) for item in replacements),
        "records": replacements,
    }
    write_json(root / "replacement_collection_report.json", collection_report)
    if not collection_report["passed"]:
        raise RuntimeError("not all failed initial sources have replacements")
    report = select_sources(root, manifest, extended, records, replacements)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
