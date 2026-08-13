#!/usr/bin/env python3
"""Audit CRUZR joint commands against measured state at dataset cadence."""

import argparse
import json
import os

import numpy as np


def analyze_motion(
    state,
    action,
    base_action,
    *,
    fps,
    phases=None,
    joint_names=None,
    action_delta_limit=0.03,
    tracking_p95_limit=0.03,
    tracking_max_limit=0.15,
    terminal_tracking_limit=0.05,
    enforce_tracking=True,
):
    state = np.asarray(state)
    action = np.asarray(action)
    base_action = np.asarray(base_action)
    if state.ndim != 2 or action.ndim != 2 or len(state) != len(action) or len(action) < 2:
        return {"passed": False, "failures": ["need at least two aligned state/action rows"]}
    if state.shape[1] < 14 or action.shape[1] < 14:
        return {"passed": False, "failures": ["state/action must contain 14 arm joints"]}

    arm_state = state[:, :14]
    arm_action = action[:, :14]
    finite = bool(np.isfinite(arm_state).all() and np.isfinite(arm_action).all())
    if base_action.size:
        finite = finite and bool(np.isfinite(base_action).all())

    per_frame_delta = np.abs(np.diff(arm_action, axis=0)).max(axis=1)
    per_frame_tracking = np.abs(arm_state - arm_action).max(axis=1)
    delta_index = int(np.argmax(per_frame_delta)) + 1
    joint_index = int(np.argmax(np.abs(arm_action[delta_index] - arm_action[delta_index - 1])))
    tracking_index = int(np.argmax(per_frame_tracking))
    tracking_joint_index = int(
        np.argmax(np.abs(arm_state[tracking_index] - arm_action[tracking_index]))
    )
    terminal_tracking = float(per_frame_tracking[-1])

    if joint_names is None or len(joint_names) < 14:
        joint_names = [f"joint_{i}" for i in range(14)]
    delta_phase = None
    tracking_phase = None
    if phases is not None and len(phases) == len(action):
        delta_phase = str(phases[delta_index])
        tracking_phase = str(phases[tracking_index])

    failures = []
    delta_max = float(per_frame_delta.max())
    tracking_p95 = float(np.percentile(per_frame_tracking, 95))
    tracking_max = float(per_frame_tracking.max())
    if not finite:
        failures.append("non-finite state/action value")
    if delta_max > action_delta_limit:
        failures.append(f"action delta max {delta_max:.4f} > {action_delta_limit:.4f} rad")
    tracking_warnings = []
    if tracking_p95 > tracking_p95_limit:
        tracking_warnings.append(
            f"tracking p95 {tracking_p95:.4f} > {tracking_p95_limit:.4f} rad")
    if tracking_max > tracking_max_limit:
        tracking_warnings.append(
            f"tracking max {tracking_max:.4f} > {tracking_max_limit:.4f} rad")
    if terminal_tracking > terminal_tracking_limit:
        tracking_warnings.append(
            f"terminal tracking {terminal_tracking:.4f} > {terminal_tracking_limit:.4f} rad"
        )
    if enforce_tracking:
        failures.extend(tracking_warnings)

    base_accel_p95 = []
    base_accel_max = []
    if base_action.ndim == 2 and len(base_action) == len(action) and base_action.shape[1] >= 2:
        base_accel = np.abs(np.diff(base_action[:, :2], axis=0)) * float(fps)
        base_accel_p95 = np.percentile(base_accel, 95, axis=0).astype(float).tolist()
        base_accel_max = base_accel.max(axis=0).astype(float).tolist()

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": tracking_warnings if not enforce_tracking else [],
        "tracking_passed": not tracking_warnings,
        "tracking_enforced": bool(enforce_tracking),
        "num_frames": int(len(action)),
        "fps": float(fps),
        "finite": finite,
        "action_delta_rad": {
            "p95": float(np.percentile(per_frame_delta, 95)),
            "p99": float(np.percentile(per_frame_delta, 99)),
            "max": delta_max,
            "max_frame": delta_index,
            "max_time_s": float(delta_index / fps),
            "max_joint": str(joint_names[joint_index]),
            "max_phase": delta_phase,
        },
        "tracking_error_rad": {
            "p95": tracking_p95,
            "p99": float(np.percentile(per_frame_tracking, 99)),
            "max": tracking_max,
            "max_frame": tracking_index,
            "max_time_s": float(tracking_index / fps),
            "max_joint": str(joint_names[tracking_joint_index]),
            "max_phase": tracking_phase,
            "terminal_max": terminal_tracking,
        },
        "base_command_acceleration": {
            "p95": base_accel_p95,
            "max": base_accel_max,
            "units": ["m/s^2", "rad/s^2"],
        },
        "limits": {
            "action_delta_max_rad": float(action_delta_limit),
            "tracking_p95_max_rad": float(tracking_p95_limit),
            "tracking_max_rad": float(tracking_max_limit),
            "terminal_tracking_max_rad": float(terminal_tracking_limit),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="episode directory containing episode_data.npz")
    parser.add_argument("--action-delta-max", type=float, default=0.03)
    parser.add_argument("--tracking-p95-max", type=float, default=0.03)
    parser.add_argument("--tracking-max", type=float, default=0.15)
    parser.add_argument("--terminal-tracking-max", type=float, default=0.05)
    args = parser.parse_args()

    data = np.load(os.path.join(args.episode, "episode_data.npz"), allow_pickle=True)
    meta_path = os.path.join(args.episode, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        meta = {}
    result = analyze_motion(
        data["state"],
        data["action"],
        data["base_action"] if "base_action" in data else np.empty((len(data["action"]), 0)),
        fps=float(meta.get("fps", 30)),
        phases=data["phase"] if "phase" in data else None,
        joint_names=meta.get("action_names"),
        action_delta_limit=args.action_delta_max,
        tracking_p95_limit=args.tracking_p95_max,
        tracking_max_limit=args.tracking_max,
        terminal_tracking_limit=args.terminal_tracking_max,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
