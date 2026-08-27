#!/usr/bin/env python3
"""Validate Sorting Roll v16 H/T/R pilot episodes before dataset ingestion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
sys.path[:0] = [str(SCRIPT_DIR), str(CORE_DIR)]

from sorting_roll_diversity import DIVERSE_TASK_VERSION  # noqa: E402
from sorting_roll_v16_pilot_contract import (  # noqa: E402
    TASK_VERSION,
    load_manifest,
)
from sorting_roll_validate import (  # noqa: E402
    FINAL_CHECKS,
    FPS,
    POLICY_CAMERAS,
    REQUIRED_GATES,
    TASK,
    diversity_errors,
    frame_errors,
    payload_errors,
    timestamp_errors,
)


MIN_FOCUSED_FRAMES = 150
FAMILY_PHASES = {
    "H": ("initial_hold", "terminal_success_hold"),
    "T": (
        "approach_table_with_arms_staged",
        "lift_flat_from_pickup_support",
    ),
    "R": ("recovery_", "clear_table"),
}
FAMILY_GATES = {
    "H": ("v16_pickup_support_physical_settle",),
    "T": (
        "held_flat_pickup",
        "held_flat_support_lift",
        "pickup_support_cleared_after_lift",
    ),
    "R": (
        "held_v16_recovered_flat_pickup",
        "held_flat_support_lift",
        "pickup_support_cleared_after_lift",
    ),
}


def _all_equal(values):
    return all(value == values[0] for value in values[1:])


def transform_errors(assignment, applied):
    errors = []
    requested = assignment["requested_transforms"][
        "pickup_support_and_roll"
    ]
    report = (applied or {}).get("pickup_support_and_roll")
    if not isinstance(report, dict):
        return ["pickup support applied transform report is missing"]
    expected_delta = np.asarray(
        [requested["x_m"], requested["y_m"], requested["z_m"]],
        dtype=float,
    )
    actual_delta = np.asarray(report.get("applied_delta_m", []), dtype=float)
    if (
        actual_delta.shape != (3,)
        or not np.isfinite(actual_delta).all()
        or not np.allclose(actual_delta, expected_delta, atol=1e-7, rtol=0)
    ):
        errors.append("pickup support applied delta does not match manifest")
    if requested["yaw_rad"] != 0.0:
        errors.append("v16 pilot supports translation only")
    if report.get("visual_collision_and_roll_consistent") is not True:
        errors.append("pickup support visual/collision/roll transform is inconsistent")
    support_geoms = report.get("support_geoms")
    if not isinstance(support_geoms, dict) or len(support_geoms) != 12:
        errors.append("pickup support transform must cover 12 visual/collision geoms")
    else:
        for name, item in support_geoms.items():
            before = np.asarray(item.get("before_m", []), dtype=float)
            after = np.asarray(item.get("after_m", []), dtype=float)
            if (
                before.shape != (3,)
                or after.shape != (3,)
                or not np.allclose(
                    after - before, expected_delta, atol=2e-7, rtol=0
                )
            ):
                errors.append(f"support geom transform mismatch: {name}")
                break
    roll_before = np.asarray(report.get("roll_before_m", []), dtype=float)
    roll_after = np.asarray(report.get("roll_after_m", []), dtype=float)
    if (
        roll_before.shape != (3,)
        or roll_after.shape != (3,)
        or not np.allclose(
            roll_after - roll_before, expected_delta, atol=2e-7, rtol=0
        )
    ):
        errors.append("roll transform does not match pickup support transform")
    return errors


def v16_metadata_errors(meta, result, assignment, phases, num_frames):
    errors = []
    episode_meta = meta.get("episode_metadata") or {}
    if meta.get("task") != TASK or result.get("task") != TASK:
        errors.append("task name mismatch")
    if not _all_equal([
        meta.get("task_version"),
        episode_meta.get("task_version"),
        result.get("task_version"),
        TASK_VERSION,
    ]):
        errors.append("task version mismatch")
    seed = assignment["seed"]
    if not _all_equal([
        meta.get("seed"), episode_meta.get("seed"), result.get("seed"), seed
    ]):
        errors.append("seed mismatch across manifest/meta/result")
    if meta.get("fps") != FPS:
        errors.append("meta.fps must be 30")
    if meta.get("num_frames") != num_frames or result.get("num_frames") != num_frames:
        errors.append("num_frames mismatch across meta/result/data")
    if num_frames < MIN_FOCUSED_FRAMES:
        errors.append(f"focused episode is shorter than {MIN_FOCUSED_FRAMES} frames")
    if set(meta.get("cameras") or {}) != set(POLICY_CAMERAS):
        errors.append("meta cameras do not match the three-camera contract")
    if tuple(episode_meta.get("policy_cameras") or ()) != POLICY_CAMERAS:
        errors.append("policy camera order mismatch")
    if tuple(episode_meta.get("recorded_cameras") or ()) != POLICY_CAMERAS:
        errors.append("recorded camera order mismatch")
    eligibility = (
        meta.get("simulation_canary_eligible"),
        episode_meta.get("simulation_canary_eligible"),
        result.get("simulation_canary_eligible"),
    )
    if eligibility != (True, True, True):
        errors.append("simulation canary eligibility is not consistently true")
    if any(
        value is not False
        for value in (
            meta.get("training_eligible"),
            episode_meta.get("training_eligible"),
            result.get("training_eligible"),
        )
    ):
        errors.append("simulation training_eligible must remain false")
    if meta.get("success") is not True or result.get("success") is not True:
        errors.append("episode success is not true")
    if result.get("error") is not None:
        errors.append("result.error is not null")

    seconds = result.get("sim_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        errors.append("sim_seconds is invalid")
    elif seconds > 60.0:
        errors.append("sim_seconds exceeds 60")
    final = result.get("final_evidence") or {}
    checks = final.get("checks") or {}
    if final.get("instantaneous_success") is not True:
        errors.append("final evidence is not successful")
    if final.get("stable_seconds", 0.0) < 2.0:
        errors.append("final stable window is shorter than 2 seconds")
    for check in FINAL_CHECKS:
        if checks.get(check) is not True:
            errors.append(f"final check failed: {check}")
    gates = result.get("gates") or {}
    for gate in REQUIRED_GATES:
        if (gates.get(gate) or {}).get("passed") is not True:
            errors.append(f"required gate failed: {gate}")

    scenario_fields = (
        "scenario_family",
        "scenario_variant",
        "scene_group_id",
        "counterfactual_pair_id",
        "start_phase",
        "terminal_phase",
        "intervention_type",
        "intervention_frame",
        "recovery_start_frame",
        "target_object_id",
        "target_color",
        "distractor_object_ids",
        "requested_transforms",
    )
    for field in scenario_fields:
        values = [meta.get(field), episode_meta.get(field), result.get(field)]
        if not _all_equal(values) or values[0] != assignment[field]:
            errors.append(f"{field} mismatch across manifest/meta/result")

    family = assignment["scenario_family"]
    expected_start, expected_terminal = FAMILY_PHASES[family]
    actual_start = result.get("recorded_start_phase")
    actual_terminal = result.get("recorded_terminal_phase")
    if family == "R":
        if not isinstance(actual_start, str) or not actual_start.startswith(expected_start):
            errors.append("recovery recording does not start at recovery frame zero")
    elif actual_start != expected_start:
        errors.append("recorded start phase mismatch")
    if actual_terminal != expected_terminal:
        errors.append("recorded terminal phase mismatch")
    if phases.shape != (num_frames,):
        errors.append("phase array shape is invalid")
    elif num_frames:
        first_phase = str(phases[0])
        if family == "R":
            if not first_phase.startswith(expected_start):
                errors.append("first recorded phase is not recovery")
        elif first_phase != expected_start:
            errors.append("first recorded phase mismatch")
        if str(phases[-1]) != expected_terminal:
            errors.append("last recorded phase mismatch")
    for gate in FAMILY_GATES[family]:
        if (gates.get(gate) or {}).get("passed") is not True:
            errors.append(f"family gate failed: {gate}")

    evidence_values = [
        meta.get("intervention_evidence"),
        episode_meta.get("intervention_evidence"),
        result.get("intervention_evidence"),
    ]
    if not _all_equal(evidence_values):
        errors.append("intervention evidence is inconsistent")
    if family == "R":
        evidence = evidence_values[0]
        if not isinstance(evidence, dict):
            errors.append("recovery intervention evidence is missing")
        elif (
            evidence.get("type") != assignment["scenario_variant"]
            or evidence.get("completed_before_recording") is not True
        ):
            errors.append("recovery intervention evidence is invalid")
        if assignment["scenario_variant"].startswith("partial_lift"):
            if (gates.get("v16_partial_lift_lowered_to_support") or {}).get("passed") is not True:
                errors.append("partial-lift recovery did not lower safely")
    elif evidence_values[0] is not None:
        errors.append("non-recovery episode contains intervention evidence")

    diversity_values = [
        meta.get("diversity"),
        episode_meta.get("diversity"),
        result.get("diversity"),
    ]
    if not _all_equal(diversity_values) or not isinstance(diversity_values[0], dict):
        errors.append("v16 diversity metadata is missing or inconsistent")
    else:
        diversity = diversity_values[0]
        if diversity.get("assignment") != assignment:
            errors.append("episode assignment does not match pilot manifest")
        base = diversity.get("base_diversity")
        if not isinstance(base, dict):
            errors.append("base v15 diversity report is missing")
        else:
            pseudo_meta = {"prompt": meta.get("prompt"), "diversity": base}
            pseudo_episode_meta = {"diversity": base}
            pseudo_result = {
                "task_version": DIVERSE_TASK_VERSION,
                "prompt": result.get("prompt"),
                "diversity": base,
                "scene_randomization": result.get("scene_randomization"),
            }
            errors.extend(
                diversity_errors(pseudo_meta, pseudo_result, pseudo_episode_meta)
            )
        errors.extend(transform_errors(assignment, meta.get("applied_transforms")))
    return errors


def validate_episode(path, assignment):
    path = Path(path).resolve()
    errors = []
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"cannot read meta/result JSON: {exc}"]
    num_frames = meta.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        return None, ["meta.num_frames is not an integer"]
    phases = np.asarray([])
    try:
        with np.load(path / "episode_data.npz", allow_pickle=False) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        phases = np.asarray(payload.get("phase", []))
        errors.extend(payload_errors(payload, num_frames))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read episode_data.npz: {exc}")
    try:
        with np.load(path / "sdk_timestamps.npz", allow_pickle=False) as data:
            timestamps = {name: np.asarray(data[name]) for name in data.files}
        errors.extend(timestamp_errors(timestamps, num_frames))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read sdk_timestamps.npz: {exc}")
    errors.extend(v16_metadata_errors(meta, result, assignment, phases, num_frames))
    errors.extend(frame_errors(path, num_frames, meta.get("resolution_hw")))
    return {
        "path": str(path),
        "seed": assignment["seed"],
        "split": assignment["split"],
        "scenario_family": assignment["scenario_family"],
        "scenario_variant": assignment["scenario_variant"],
        "num_frames": num_frames,
        "sim_seconds": result.get("sim_seconds"),
    }, errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    records = []
    for assignment in manifest["assignments"]:
        episode = args.episode_root / f"seed_{assignment['seed']}"
        if not (episode / "result.json").is_file():
            if args.require_complete:
                records.append({
                    "path": str(episode.resolve()),
                    "passed": False,
                    "info": None,
                    "errors": ["episode is missing"],
                })
            continue
        info, errors = validate_episode(episode, assignment)
        records.append({
            "path": str(episode.resolve()),
            "passed": not errors,
            "info": info,
            "errors": errors,
        })
        print(
            f"[v16 validate] seed={assignment['seed']} "
            f"{'PASS' if not errors else 'FAIL'}"
            + ("" if not errors else f" {'; '.join(errors)}"),
            flush=True,
        )
    family_counts = Counter(
        record["info"]["scenario_family"]
        for record in records
        if record["passed"] and record.get("info")
    )
    passed_count = sum(record["passed"] for record in records)
    report = {
        "schema_version": 1,
        "task": TASK,
        "task_version": TASK_VERSION,
        "campaign": manifest["campaign"],
        "manifest": str(args.manifest.resolve()),
        "episode_root": str(args.episode_root.resolve()),
        "episode_count": len(records),
        "passed_count": passed_count,
        "failed_count": len(records) - passed_count,
        "family_counts": dict(sorted(family_counts.items())),
        "complete": len(records) == manifest["count"],
        "passed": bool(records) and passed_count == len(records),
        "records": records,
    }
    if args.require_complete and not report["complete"]:
        report["passed"] = False
    if args.report:
        if args.report.exists():
            raise SystemExit(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        key: report[key]
        for key in (
            "episode_count", "passed_count", "failed_count", "complete", "passed"
        )
    }, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
