#!/usr/bin/env python3
"""Compare pi0.5 action chunks on fixed Sorting Roll expert observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = (
    "observation.images.stereo_left",
    "observation.images.left_wrist_realsense",
    "observation.images.right_wrist_realsense",
)
ACTION_GROUPS = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "grippers": slice(14, 16),
    "base": slice(16, 18),
}
HORIZONS = (20, 30, 50)


def first_phase_index(phases, name):
    matches = np.flatnonzero(np.asarray(phases) == name)
    if len(matches) == 0:
        raise ValueError(f"missing required phase: {name}")
    return int(matches[0])


def last_phase_index(phases, name):
    matches = np.flatnonzero(np.asarray(phases) == name)
    if len(matches) == 0:
        raise ValueError(f"missing required phase: {name}")
    return int(matches[-1])


def select_probe_frames(phases, state, action):
    phases = np.asarray(phases)
    state = np.asarray(state)
    action = np.asarray(action)
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError("state and action must be two-dimensional")
    if len(phases) != len(state) or len(state) != len(action):
        raise ValueError("phase, state, and action lengths differ")
    if state.shape[1] < 16 or action.shape[1] < 16:
        raise ValueError("Sorting Roll state/action must include two grippers")

    grasp_start = first_phase_index(phases, "horizontal_approach_and_grasp")
    grasp_stop = first_phase_index(phases, "lift_flat_from_pickup_support")
    closing = np.flatnonzero(
        np.all(action[grasp_start:grasp_stop, 14:16] < 0.5, axis=1)
    )
    if len(closing) == 0:
        raise ValueError("grasp phase has no simultaneous close command")
    close_command = grasp_start + int(closing[0])
    established = np.flatnonzero(
        np.all(state[close_command:grasp_stop, 14:16] < 0.65, axis=1)
    )
    if len(established) == 0:
        raise ValueError("grasp phase never establishes two closed grippers")

    frames = {
        "table_observation": first_phase_index(
            phases, "localize_roll_with_head_stereo"
        ),
        "pregrasp": first_phase_index(
            phases, "coordinated_flat_pick_pregrasp_after_stereo_localization"
        ),
        "precontact": close_command - 1,
        "grasp_established": close_command + int(established[0]),
        "lift_start": grasp_stop,
        "support_cleared": last_phase_index(
            phases, "lift_flat_from_pickup_support"
        ),
    }
    if min(frames.values()) < 0 or max(frames.values()) + max(HORIZONS) > len(action):
        raise ValueError("probe frames do not have a complete 50-action future")
    return frames


def _cosine(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    return float(np.dot(first, second) / denominator)


def _first_close_index(actions, threshold=0.5):
    matches = np.flatnonzero(np.all(actions[:, 14:16] < threshold, axis=1))
    return int(matches[0]) if len(matches) else None


def _sign_with_deadband(values, deadband=0.01):
    values = np.asarray(values)
    return np.where(np.abs(values) <= deadband, 0, np.sign(values))


def chunk_metrics(predicted, expert, current_state):
    predicted = np.asarray(predicted, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    current_state = np.asarray(current_state, dtype=np.float32)
    if predicted.shape != (50, 18) or expert.shape != (50, 18):
        raise ValueError("predicted and expert chunks must have shape (50, 18)")
    if current_state.shape != (18,):
        raise ValueError("current_state must have shape (18,)")
    if not np.isfinite(predicted).all() or not np.isfinite(expert).all():
        raise ValueError("action chunks must contain only finite values")

    horizons = {}
    for horizon in HORIZONS:
        pred = predicted[:horizon]
        target = expert[:horizon]
        groups = {}
        for name, selector in ACTION_GROUPS.items():
            error = pred[:, selector] - target[:, selector]
            groups[name] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
            }
        pred_arm_delta = pred[:, :14] - current_state[None, :14]
        expert_arm_delta = target[:, :14] - current_state[None, :14]
        base_sign = _sign_with_deadband(
            pred[:, 16:18]
        ) == _sign_with_deadband(target[:, 16:18])
        horizons[str(horizon)] = {
            "groups": groups,
            "arm_delta_cosine": _cosine(pred_arm_delta, expert_arm_delta),
            "base_sign_agreement": float(np.mean(base_sign)),
            "predicted_base_mean": np.mean(pred[:, 16:18], axis=0).tolist(),
            "expert_base_mean": np.mean(target[:, 16:18], axis=0).tolist(),
        }
    return {
        "horizons": horizons,
        "predicted_first_close_index": _first_close_index(predicted),
        "expert_first_close_index": _first_close_index(expert),
        "predicted_first_action": predicted[0].tolist(),
        "expert_first_action": expert[0].tolist(),
    }


def checkpoint_label(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(raw_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=checkpoint_label, required=True
    )
    parser.add_argument("--source-seed", type=int, default=3000)
    parser.add_argument("--policy-seed", type=int, default=28000)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_episode_row(dataset_root, source_seed):
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    matches = [row for row in rows if int(row["source_seed"]) == source_seed]
    if len(matches) != 1:
        raise ValueError(
            f"expected one dataset episode for source seed {source_seed}, got {len(matches)}"
        )
    return matches[0]


def make_raw_observation(sample):
    return {
        "observation.state": sample["observation.state"],
        **{camera: sample[camera] for camera in CAMERAS},
        "task": sample["task"],
    }


def run(args):
    import torch

    from src.lerobot.configs.policies import PreTrainedConfig
    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
    from src.lerobot.policies.factory import (
        get_policy_class,
        make_pre_post_processors,
    )

    episode = args.episode.resolve()
    dataset_root = args.dataset.resolve()
    with np.load(episode / "episode_data.npz", allow_pickle=False) as payload:
        source_state = np.concatenate(
            [payload["state"], payload["base_velocity"]], axis=1
        ).astype(np.float32)
        source_action = np.concatenate(
            [payload["action"], payload["base_action"]], axis=1
        ).astype(np.float32)
        phases = payload["phase"].copy()
    frames = select_probe_frames(phases, source_state, source_action)
    episode_row = load_episode_row(dataset_root, args.source_seed)
    dataset_start = int(episode_row["dataset_from_index"])
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        video_backend="pyav",
    )
    samples = {
        name: dataset[dataset_start + frame]
        for name, frame in frames.items()
    }
    for name, sample in samples.items():
        if int(sample["frame_index"]) != frames[name]:
            raise ValueError(f"dataset/source frame mismatch for {name}")
        dataset_state = sample["observation.state"].cpu().numpy()
        if not np.allclose(dataset_state, source_state[frames[name]], atol=1e-6):
            raise ValueError(f"dataset/source state mismatch for {name}")

    results = {}
    arrays = {}
    for label, raw_checkpoint in args.checkpoint:
        checkpoint = raw_checkpoint.resolve()
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if config.type != "pi05" or config.n_action_steps != 50:
            raise ValueError(f"{label} is not a 50-step pi05 checkpoint")
        preprocessor, postprocessor = make_pre_post_processors(
            config, pretrained_path=str(checkpoint)
        )
        policy = get_policy_class(config.type).from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        checkpoint_results = {}
        for probe_index, (name, frame) in enumerate(frames.items()):
            seed = args.policy_seed + probe_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                processed = preprocessor(make_raw_observation(samples[name]))
                normalized = policy.predict_action_chunk(processed)
                predicted = postprocessor(normalized)[0].numpy().astype(np.float32)
            expert = source_action[frame:frame + 50]
            checkpoint_results[name] = {
                "frame": frame,
                "phase": str(phases[frame]),
                "policy_seed": seed,
                **chunk_metrics(predicted, expert, source_state[frame]),
            }
            arrays[f"{label}__{name}__predicted"] = predicted
            arrays[f"{label}__{name}__expert"] = expert
        results[label] = {
            "checkpoint": str(checkpoint),
            "probes": checkpoint_results,
        }
        del policy, preprocessor, postprocessor
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "purpose": "sorting_roll_v16_stage0_teacher_forced_probe",
        "source_seed": args.source_seed,
        "source_episode": str(episode),
        "dataset": str(dataset_root),
        "dataset_episode_index": int(episode_row["episode_index"]),
        "policy_seed": args.policy_seed,
        "probe_frames": frames,
        "horizons": list(HORIZONS),
        "checkpoints": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(json.dumps({
        "report": str(args.out.resolve()),
        "arrays": str(args.out.with_suffix('.npz').resolve()),
        "checkpoints": list(results),
        "probe_frames": frames,
    }, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if not args.episode.is_dir():
        raise SystemExit(f"episode directory is missing: {args.episode}")
    if not args.dataset.is_dir():
        raise SystemExit(f"dataset directory is missing: {args.dataset}")
    for label, checkpoint in args.checkpoint:
        if not (checkpoint / "model.safetensors").is_file():
            raise SystemExit(f"checkpoint is incomplete: {label}={checkpoint}")
    run(args)


if __name__ == "__main__":
    main()
