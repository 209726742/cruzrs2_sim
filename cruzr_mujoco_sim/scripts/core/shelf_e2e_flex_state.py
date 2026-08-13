#!/usr/bin/env python3
"""Versioned audit-state contract for the articulated strip model."""

import os

import numpy as np

from shelf_e2e_objects import internal_ball_quaternions


RIGID_TASK_VERSION = "dual_two_trip_v1"
FLEX_TASK_VERSION = "dual_two_trip_flex_v1"
INTERNAL_STATE_FILE = "object_internal_state.npz"
FLEX_OBJECT = "strip"
FLEX_ENCODING = "ball_joint_quaternion_wxyz"
FLEX_JOINT_NAMES = tuple([f"strip_J_{index}" for index in range(1, 13)] + ["strip_J_last"])
FLEX_STATE_DIM = 4 * len(FLEX_JOINT_NAMES)
QUATERNION_NORM_ATOL = 1e-4


def flex_contract():
    return {
        "file": INTERNAL_STATE_FILE,
        "object": FLEX_OBJECT,
        "encoding": FLEX_ENCODING,
        "joint_names": list(FLEX_JOINT_NAMES),
        "joint_count": len(FLEX_JOINT_NAMES),
        "state_dim": FLEX_STATE_DIM,
    }


def object_state_contract(objects):
    """Return (task_version, sidecar_contract) for rigid or the fixed 14-segment strip."""
    for name, info in objects.items():
        if name != FLEX_OBJECT and info.get("ball_qpos_adrs"):
            raise RuntimeError(f"unsupported articulated object: {name}")

    strip = objects[FLEX_OBJECT]
    addresses = tuple(strip.get("ball_qpos_adrs") or ())
    names = tuple(strip.get("ball_joint_names") or ())
    if not addresses:
        if names:
            raise RuntimeError("rigid strip unexpectedly has internal joint names")
        return RIGID_TASK_VERSION, None
    if len(addresses) != len(FLEX_JOINT_NAMES) or names != FLEX_JOINT_NAMES:
        raise RuntimeError(
            f"flex strip joints must be {FLEX_JOINT_NAMES}, got {names}"
        )
    return FLEX_TASK_VERSION, flex_contract()


def _quaternion_errors(state, expected_frames):
    errors = []
    state = np.asarray(state)
    expected_shape = (expected_frames, FLEX_STATE_DIM)
    if state.shape != expected_shape:
        return [f"internal quaternion shape {state.shape} != {expected_shape}"]
    if not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
        return ["internal quaternions contain non-numeric or NaN/Inf values"]
    norms = np.linalg.norm(
        state.reshape(expected_frames, len(FLEX_JOINT_NAMES), 4),
        axis=2,
    )
    if not np.allclose(norms, 1.0, atol=QUATERNION_NORM_ATOL, rtol=0.0):
        errors.append("internal ball-joint quaternions are not unit length")
    return errors


def capture_internal_state(data, strip_info):
    version, _ = object_state_contract({FLEX_OBJECT: strip_info})
    if version != FLEX_TASK_VERSION:
        raise RuntimeError("cannot capture internal state from a rigid strip")
    row = internal_ball_quaternions(data, strip_info)
    errors = _quaternion_errors(row.reshape(1, -1), 1)
    if errors:
        raise RuntimeError("; ".join(errors))
    return row


def save_internal_state(directory, rows):
    state = np.asarray(rows, dtype=np.float32)
    errors = _quaternion_errors(state, len(state))
    if errors:
        raise ValueError("; ".join(errors))
    np.savez_compressed(
        os.path.join(directory, INTERNAL_STATE_FILE),
        object=np.asarray(FLEX_OBJECT),
        encoding=np.asarray(FLEX_ENCODING),
        joint_names=np.asarray(FLEX_JOINT_NAMES),
        quaternion=state,
    )


def internal_state_errors(directory, episode_metadata, num_frames):
    """Validate version/metadata/archive consistency without loading a MuJoCo model."""
    task_version = episode_metadata.get("task_version")
    metadata_contract = episode_metadata.get("object_internal_state")
    path = os.path.join(directory, INTERNAL_STATE_FILE)

    if task_version == RIGID_TASK_VERSION:
        errors = []
        if metadata_contract is not None:
            errors.append("rigid task must not declare object_internal_state")
        if os.path.exists(path):
            errors.append("rigid task must not contain object_internal_state.npz")
        return errors
    if task_version != FLEX_TASK_VERSION:
        return [f"unsupported task_version: {task_version!r}"]

    errors = []
    if metadata_contract != flex_contract():
        errors.append("flex object_internal_state metadata does not match the contract")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"object", "encoding", "joint_names", "quaternion"}
            missing = sorted(required - set(archive.files))
            if missing:
                errors.append(f"object_internal_state.npz missing {missing}")
                return errors
            if archive["object"].shape != () or archive["object"].item() != FLEX_OBJECT:
                errors.append(f"internal state object must be {FLEX_OBJECT}")
            if archive["encoding"].shape != () or archive["encoding"].item() != FLEX_ENCODING:
                errors.append(f"internal state encoding must be {FLEX_ENCODING}")
            joint_names = archive["joint_names"]
            if joint_names.shape != (len(FLEX_JOINT_NAMES),) or list(joint_names) != list(FLEX_JOINT_NAMES):
                errors.append("internal state joint names/order do not match the flex contract")
            errors.extend(_quaternion_errors(archive["quaternion"], num_frames))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot read {INTERNAL_STATE_FILE}: {exc}")
    return errors


def load_internal_state(directory, episode_metadata, num_frames):
    errors = internal_state_errors(directory, episode_metadata, num_frames)
    if errors:
        raise ValueError("; ".join(errors))
    if episode_metadata.get("task_version") == RIGID_TASK_VERSION:
        return None
    with np.load(os.path.join(directory, INTERNAL_STATE_FILE), allow_pickle=False) as archive:
        return archive["quaternion"].astype(np.float64, copy=True)


def restore_internal_state(data, strip_info, row):
    version, _ = object_state_contract({FLEX_OBJECT: strip_info})
    if version != FLEX_TASK_VERSION:
        raise RuntimeError("cannot restore internal state into a rigid strip")
    row = np.asarray(row, dtype=np.float64)
    errors = _quaternion_errors(row.reshape(1, -1), 1)
    if errors:
        raise ValueError("; ".join(errors))
    for index, address in enumerate(strip_info["ball_qpos_adrs"]):
        data.qpos[address:address + 4] = row[4 * index:4 * index + 4]
