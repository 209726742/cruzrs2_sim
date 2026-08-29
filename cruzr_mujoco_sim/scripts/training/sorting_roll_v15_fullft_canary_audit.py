#!/usr/bin/env python3
"""Audit the Sorting Roll v15 full-parameter fresh/resume canary."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re

from safetensors import safe_open


EXPECTED_STEPS = (200, 250)
EXPECTED_PARAMETER_COUNT = 4_143_404_816
REQUIRED_JSON = (
    "pretrained_model/config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
)
REQUIRED_TENSORS = (
    "pretrained_model/model.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
)


def checkpoint_errors(root, expected_step):
    errors = []
    checkpoint = root / "checkpoints" / f"{expected_step:06d}"
    if not checkpoint.is_dir():
        return [f"missing checkpoint {expected_step}"], None
    for relative in REQUIRED_JSON:
        path = checkpoint / relative
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for relative in REQUIRED_TENSORS:
        path = checkpoint / relative
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    errors.append(f"empty safetensors: {path}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    state_path = checkpoint / "training_state/training_step.json"
    if state_path.is_file():
        step = json.loads(state_path.read_text(encoding="utf-8")).get("step")
        if int(step) != expected_step:
            errors.append(f"checkpoint step is {step}, expected {expected_step}")
    return errors, checkpoint


def audit(args):
    errors = []
    checkpoints = {}
    for step in EXPECTED_STEPS:
        step_errors, checkpoint = checkpoint_errors(args.output, step)
        errors.extend(step_errors)
        checkpoints[str(step)] = str(checkpoint) if checkpoint else None

    config_path = (
        args.output / "checkpoints/000250/pretrained_model/train_config.json"
    )
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    policy = config.get("policy", {})
    dataset = config.get("dataset", {})
    expected = {
        "batch_size": 16,
        "num_workers": 8,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"{key}={config.get(key)!r}, expected {value!r}")
    policy_expected = {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": False,
        "use_peft": False,
        "train_expert_only": False,
        "optimizer_lr": args.expected_learning_rate,
    }
    for key, value in policy_expected.items():
        if policy.get(key) != value:
            errors.append(f"policy.{key}={policy.get(key)!r}, expected {value!r}")
    if Path(dataset.get("root", "")).resolve() != args.dataset.resolve():
        errors.append("canary dataset root mismatch")

    text = args.log.read_text(encoding="utf-8", errors="replace")
    exit_codes = [int(value) for value in re.findall(
        r"train command exited rc=(\d+)", text
    )]
    if len(exit_codes) < 2 or exit_codes[-2:] != [0, 0]:
        errors.append(f"fresh/resume exit codes are {exit_codes[-2:]}")
    losses = [float(value) for value in re.findall(r"\bloss:([^\s]+)", text)]
    gradients = [float(value) for value in re.findall(r"\bgrdn:([^\s]+)", text)]
    if not losses or not gradients:
        errors.append("training log has no loss/gradient measurements")
    elif not all(math.isfinite(value) for value in losses + gradients):
        errors.append("training log contains non-finite loss or gradient")
    learnable_counts = [
        int(value)
        for value in re.findall(r"num_learnable_params=(\d+)", text)
    ]
    total_counts = [
        int(value)
        for value in re.findall(r"num_total_params=(\d+)", text)
    ]
    expected_counts = [EXPECTED_PARAMETER_COUNT, EXPECTED_PARAMETER_COUNT]
    full_parameter_count_verified = (
        learnable_counts[-2:] == expected_counts
        and total_counts[-2:] == expected_counts
    )
    if not full_parameter_count_verified:
        errors.append("fresh/resume runs did not train every policy parameter")

    timing_matches = re.findall(
        r"INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?step:(\d+)",
        text,
    )
    intervals = []
    for (left_time, left_step), (right_time, right_step) in zip(
        timing_matches, timing_matches[1:]
    ):
        step_delta = int(right_step) - int(left_step)
        time_delta = (
            datetime.fromisoformat(right_time)
            - datetime.fromisoformat(left_time)
        ).total_seconds()
        if step_delta > 0 and 0.0 < time_delta <= 600.0:
            intervals.append(time_delta / step_delta)
    mean_seconds_per_step = (
        sum(intervals) / len(intervals) if intervals else None
    )

    report = {
        "schema_version": 1,
        "task_version": args.task_version,
        "output": str(args.output.resolve()),
        "dataset": str(args.dataset.resolve()),
        "checkpoints": checkpoints,
        "fresh_target_step": 200,
        "resume_target_step": 250,
        "fresh_and_resume_exit_zero": exit_codes[-2:] == [0, 0],
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "learnable_parameter_counts": learnable_counts[-2:],
        "total_parameter_counts": total_counts[-2:],
        "full_parameter_count_verified": full_parameter_count_verified,
        "train_expert_only": policy.get("train_expert_only"),
        "batch_size_per_gpu": config.get("batch_size"),
        "effective_batch_size": 4 * int(config.get("batch_size", 0)),
        "learning_rate": policy.get("optimizer_lr"),
        "loss_measurements": len(losses),
        "gradient_measurements": len(gradients),
        "mean_seconds_per_step_without_checkpoint": mean_seconds_per_step,
        "historical_20h_target_steps": 28000,
        "errors": errors,
        "passed": not errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--task-version", default="sorting_roll_v15_diverse_sim"
    )
    parser.add_argument("--expected-learning-rate", type=float, default=2.5e-5)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
