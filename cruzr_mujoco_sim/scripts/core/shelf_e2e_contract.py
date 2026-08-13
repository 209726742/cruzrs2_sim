#!/usr/bin/env python3
"""Deployable observation/action contract for the dual-material shelf task."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FPS = 30
IMAGE_SHAPE = (224, 224, 3)
CHUNK_SIZE = 50

STATE_NAMES = tuple(
    [f"j{i}" for i in range(14)]
    + ["grip_l", "grip_r", "base_v_fwd", "base_wz"]
)
ACTION_NAMES = tuple(
    [f"j{i}" for i in range(14)]
    + ["grip_l", "grip_r", "base_cmd_v_fwd", "base_cmd_wz"]
)

# Only inputs that are available in the current rollout/deployment path belong
# in the policy contract. MuJoCo-only object/cart coordinates and the unserved
# left-hand camera are intentionally excluded.
POLICY_IMAGE_MAP = {
    "observation/image": "observation.images.head_stereo_l_shelf",
    "observation/left_wrist_image": "observation.images.chassis_front",
    "observation/right_wrist_image": "observation.images.hand_right_shelf",
}
CAMERAS = tuple(key.rsplit(".", 1)[-1] for key in POLICY_IMAGE_MAP.values())

STATE_DIM = len(STATE_NAMES)
ACTION_DIM = len(ACTION_NAMES)


def make_state(arm_and_gripper_state, base_velocity) -> np.ndarray:
    """Build one state vector while enforcing the declared field order."""
    state = np.concatenate(
        [
            np.asarray(arm_and_gripper_state, dtype=np.float32).reshape(-1),
            np.asarray(base_velocity, dtype=np.float32).reshape(-1),
        ]
    )
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or Inf")
    return state


def validate_policy_observation(observation: Mapping) -> None:
    """Reject missing/extra camera inputs and malformed state/image tensors."""
    camera_keys = {key for key in observation if key.endswith("image")}
    expected_camera_keys = set(POLICY_IMAGE_MAP)
    if camera_keys != expected_camera_keys:
        raise ValueError(
            f"policy camera keys must be {sorted(expected_camera_keys)}, got {sorted(camera_keys)}"
        )

    state = np.asarray(observation.get("observation/state"))
    if state.shape != (STATE_DIM,):
        raise ValueError(f"observation/state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("observation/state contains NaN or Inf")

    for key in POLICY_IMAGE_MAP:
        image = np.asarray(observation[key])
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"{key} must have shape {IMAGE_SHAPE}, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must have dtype uint8, got {image.dtype}")


def validate_action_chunk(actions) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.shape != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(
            f"actions must have shape ({CHUNK_SIZE}, {ACTION_DIM}), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("actions contain NaN or Inf")
    return chunk
