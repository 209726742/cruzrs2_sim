#!/usr/bin/env python3
"""Screen Sorting Roll policy-camera observability by replaying an episode."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from cruzr_s2_sdk_contract import ARM_JOINT_NAMES, SDK_CAMERAS
from sorting_roll_diversity import apply_model_diversity
from sorting_roll_realsense_profile import (
    CAMERA_ROLES as REALSENSE_CAMERA_ROLES,
    MODEL_CAMERA_SOURCES as REALSENSE_CAMERA_SOURCES,
    PROFILE_NAME as REALSENSE_PROFILE_NAME,
    apply_model_camera_overrides,
    profile_report as realsense_profile_report,
)


DEFAULT_SCENE = PACKAGE_ROOT / "assets" / "sorting_roll_scene.xml"
CAMERA_SOURCES = {
    "stereo_left": "stereo_left",
    "stereo_right": "stereo_right",
    "waist_front": "waist_front",
    "chassis_front": "chassis_front",
    **REALSENSE_CAMERA_SOURCES,
}
CAMERA_ROLES = {
    "stereo_left": "global",
    "stereo_right": "global",
    "waist_front": "context",
    "chassis_front": "navigation",
    **REALSENSE_CAMERA_ROLES,
}
CAMERA_SETS = {
    "current_policy": tuple(SDK_CAMERAS),
    "manipulation_stereo": ("stereo_left", "stereo_right", "waist_front"),
    "all_recorded": (
        "stereo_left",
        "stereo_right",
        "waist_front",
        "chassis_front",
    ),
    REALSENSE_PROFILE_NAME: tuple(REALSENSE_CAMERA_SOURCES),
}
STAGE_PHASES = {
    "pickup_observation": (
        "localize_roll_with_head_stereo",
        "coordinated_flat_pick_pregrasp_after_stereo_localization",
    ),
    "pickup_contact": (
        "horizontal_approach_and_grasp",
        "lift_flat_from_pickup_support",
    ),
    "transport": (
        "clear_table",
        "rotate_to_shelf",
        "navigate_to_shelf_stage",
    ),
    "placement_alignment": (
        "align_shelf_axis_above_front_lip",
        "realign_shelf_stage_after_axis",
        "level_release_support_surfaces",
        "fine_align_axis_before_entry",
        "lower_to_front_lip_clearance",
    ),
    "insertion": (
        "move_over_integrated_front_lip",
        "position_guarded_release_clearance",
    ),
    "release_confirmation": (
        "guarded_release_and_lift_open_hands",
        "verify_after_guarded_release",
        "retract_arms_after_release",
        "terminal_success_hold",
    ),
}
PLACEMENT_STAGES = {
    "placement_alignment",
    "insertion",
    "release_confirmation",
}
REQUIRED_ROLES_BY_STAGE = {
    "pickup_observation": {"global"},
    "pickup_contact": {"global", "left_wrist", "right_wrist"},
    "transport": {"global"},
    "placement_alignment": {"global", "left_wrist", "right_wrist"},
    "insertion": {"global", "left_wrist", "right_wrist"},
    "release_confirmation": {"global", "left_wrist", "right_wrist"},
}
TARGET_POINTS_M = np.asarray(
    [
        (0.950, 0.0, 0.912),
        (0.950, -0.250, 0.912),
        (0.950, 0.250, 0.912),
    ],
    dtype=np.float64,
)

# These are rejection-screen thresholds, not a claim that a policy will learn.
MIN_FULL_ROLL_PIXELS = 24
MIN_VISIBLE_FRACTION = 0.50
MIN_EXTENT_FRACTION = 0.70
MIN_TARGET_IN_FRAME_FRACTION = 1.0
MIN_WRIST_ROLL_PIXELS = 16
MIN_WRIST_HAND_PIXELS = 8


def phase_spans(phases):
    phases = np.asarray(phases)
    spans = {}
    if phases.size == 0:
        return spans
    start = 0
    for index in range(1, phases.size + 1):
        if index == phases.size or phases[index] != phases[start]:
            name = str(phases[start])
            if name in spans:
                raise ValueError(f"phase appears in multiple spans: {name}")
            spans[name] = (start, index - 1)
            start = index
    return spans


def sampled_indices(start, end, count):
    if count < 1:
        raise ValueError("sample count must be positive")
    if end < start:
        raise ValueError("phase end precedes start")
    return sorted({
        int(round(value))
        for value in np.linspace(start, end, min(count, end - start + 1))
    })


def stage_phase_gaps(spans, stage_phases=STAGE_PHASES):
    missing_stages = {
        stage: list(phases)
        for stage, phases in stage_phases.items()
        if not any(phase in spans for phase in phases)
    }
    skipped_noop_phases = sorted({
        phase
        for phases in stage_phases.values()
        for phase in phases
        if phase not in spans
    })
    return missing_stages, skipped_noop_phases


def principal_extent(mask):
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] < 2:
        return 0.0
    xy = coordinates[:, ::-1].astype(np.float64)
    centered = xy - np.mean(xy, axis=0)
    covariance = centered.T @ centered
    direction = np.linalg.eigh(covariance)[1][:, -1]
    projection = xy @ direction
    return float(np.max(projection) - np.min(projection) + 1.0)


def mask_metrics(visible_mask, full_mask):
    visible_pixels = int(np.count_nonzero(visible_mask))
    full_pixels = int(np.count_nonzero(full_mask))
    visible_extent = principal_extent(visible_mask)
    full_extent = principal_extent(full_mask)
    return {
        "visible_pixels": visible_pixels,
        "full_pixels": full_pixels,
        "visible_fraction": (
            float(visible_pixels / full_pixels) if full_pixels else 0.0
        ),
        "extent_fraction": (
            float(visible_extent / full_extent) if full_extent else 0.0
        ),
        "visible_extent_px": visible_extent,
        "full_extent_px": full_extent,
    }


def project_points(camera_position, camera_rotation, points, fovy_deg, width, height):
    camera_position = np.asarray(camera_position, dtype=np.float64)
    camera_rotation = np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3)
    points = np.asarray(points, dtype=np.float64)
    delta = points - camera_position
    right = camera_rotation[:, 0]
    up = camera_rotation[:, 1]
    forward = -camera_rotation[:, 2]
    depth = delta @ forward
    focal = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    pixel_x = width / 2.0 + focal * (delta @ right) / depth
    pixel_y = height / 2.0 - focal * (delta @ up) / depth
    pixels = np.column_stack((pixel_x, pixel_y))
    in_frame = (
        (depth > 0.0)
        & (pixel_x >= 0.0)
        & (pixel_x < width)
        & (pixel_y >= 0.0)
        & (pixel_y < height)
    )
    return pixels, depth, in_frame


def observation_is_usable(metrics, target_in_frame_fraction, placement_stage):
    return bool(
        metrics["full_pixels"] >= MIN_FULL_ROLL_PIXELS
        and metrics["visible_fraction"] >= MIN_VISIBLE_FRACTION
        and metrics["extent_fraction"] >= MIN_EXTENT_FRACTION
        and (
            not placement_stage
            or target_in_frame_fraction >= MIN_TARGET_IN_FRAME_FRACTION
        )
    )


def wrist_observation_is_usable(
    metrics,
    *,
    hand_visible_pixels,
    hand_reference_in_frame,
    roll_contact_in_frame,
    target_contact_in_frame,
    placement_stage,
):
    return bool(
        metrics["visible_pixels"] >= MIN_WRIST_ROLL_PIXELS
        and hand_visible_pixels >= MIN_WRIST_HAND_PIXELS
        and hand_reference_in_frame
        and roll_contact_in_frame
        and (not placement_stage or target_contact_in_frame)
    )


def aggregate_candidate(records, cameras):
    grouped = {}
    for record in records:
        key = (record["stage"], record["phase"], record["frame"])
        grouped.setdefault(key, {})[record["camera"]] = record

    stage_totals = {}
    for key, by_camera in grouped.items():
        stage = key[0]
        usable_count = sum(
            bool(by_camera[camera]["usable"])
            for camera in cameras
            if camera in by_camera
        )
        values = stage_totals.setdefault(
            stage,
            {
                "samples": 0,
                "covered": 0,
                "usable_view_sum": 0,
                "required_roles_covered": 0,
            },
        )
        usable_roles = {
            by_camera[camera].get("role")
            for camera in cameras
            if camera in by_camera and by_camera[camera]["usable"]
        }
        required_roles = REQUIRED_ROLES_BY_STAGE.get(stage, set())
        values["samples"] += 1
        values["covered"] += int(usable_count > 0)
        values["usable_view_sum"] += usable_count
        values["required_roles_covered"] += int(
            required_roles.issubset(usable_roles)
        )

    stage_report = {}
    total_samples = total_covered = total_usable_views = 0
    for stage, values in stage_totals.items():
        samples = values["samples"]
        stage_report[stage] = {
            **values,
            "coverage_fraction": values["covered"] / samples,
            "mean_usable_views": values["usable_view_sum"] / samples,
            "required_role_coverage_fraction": (
                values["required_roles_covered"] / samples
            ),
            "required_roles": sorted(REQUIRED_ROLES_BY_STAGE.get(stage, set())),
        }
        total_samples += samples
        total_covered += values["covered"]
        total_usable_views += values["usable_view_sum"]
    required_role_covered = sum(
        values["required_roles_covered"] for values in stage_totals.values()
    )
    return {
        "cameras": list(cameras),
        "samples": total_samples,
        "covered": total_covered,
        "coverage_fraction": total_covered / total_samples if total_samples else 0.0,
        "mean_usable_views": (
            total_usable_views / total_samples if total_samples else 0.0
        ),
        "required_role_covered": required_role_covered,
        "required_role_coverage_fraction": (
            required_role_covered / total_samples if total_samples else 0.0
        ),
        "stage_report": stage_report,
    }


def _subtree_geom_ids(model, root_body):
    body_ids = {int(root_body)}
    for body in range(model.nbody):
        if int(model.body_parentid[body]) in body_ids:
            body_ids.add(body)
    return {
        geom
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in body_ids
    }


def _wrist_reference_points(mujoco, model, data, roll_body):
    roll_center = data.xpos[roll_body].copy()
    roll_axis = data.xmat[roll_body].reshape(3, 3)[:, 0].copy()
    references = {}
    for role, side, target_y in (
        ("left_wrist", "L", 0.250),
        ("right_wrist", "R", -0.250),
    ):
        pad_ids = [
            _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad1"),
            _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad2"),
        ]
        hand_reference = np.mean(data.geom_xpos[pad_ids], axis=0)
        along = float(np.clip(
            (hand_reference - roll_center) @ roll_axis, -0.25, 0.25
        ))
        references[role] = {
            "hand_reference": hand_reference,
            "roll_contact": roll_center + along * roll_axis,
            "target_contact": np.asarray((0.950, target_y, 0.912)),
        }
    return references


def _named_id(mujoco, model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"scene is missing {name}")
    return object_id


def _restore_frame(mujoco, model, data, episode, meta, frame):
    for state_index, name in enumerate(ARM_JOINT_NAMES):
        joint = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = episode["state"][frame, state_index]

    gripper = meta["gripper_raw_convention"]
    open_value = float(gripper["open"])
    close_value = float(gripper["close"])
    for side, state_index in (("L", 14), ("R", 15)):
        fraction = float(episode["state"][frame, state_index])
        raw_value = close_value + fraction * (open_value - close_value)
        for suffix in ("finger1_joint", "finger2_joint"):
            joint = _named_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_{suffix}",
            )
            data.qpos[int(model.jnt_qposadr[joint])] = raw_value

    for value, name in zip(episode["base"][frame], ("base_x", "base_y", "base_yaw")):
        joint = _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = value

    roll_joint = _named_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "sorting_roll_free",
    )
    roll_qpos_address = int(model.jnt_qposadr[roll_joint])
    roll_dof_address = int(model.jnt_dofadr[roll_joint])
    data.qpos[roll_qpos_address:roll_qpos_address + 7] = episode["roll_qpos"][frame]
    data.qvel[:] = 0.0
    data.qvel[roll_dof_address:roll_dof_address + 6] = episode["roll_qvel"][frame]
    mujoco.mj_forward(model, data)


def run_audit(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    import mujoco

    episode_path = args.episode / "episode_data.npz"
    meta_path = args.episode / "meta.json"
    with np.load(episode_path, allow_pickle=False) as payload:
        episode = {name: payload[name] for name in payload.files}
    meta = json.loads(meta_path.read_text())
    spans = phase_spans(episode["phase"])
    missing_stages, skipped_noop_phases = stage_phase_gaps(spans)
    if missing_stages:
        raise RuntimeError(
            f"episode is missing critical stages: {missing_stages}"
        )

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    apply_model_camera_overrides(mujoco, model)
    data = mujoco.MjData(model)
    replay_diversity = None
    diversity = meta.get("diversity")
    if diversity is not None:
        if not isinstance(diversity, dict) or not isinstance(
            diversity.get("assignment"), dict
        ):
            raise RuntimeError("episode diversity metadata is invalid")
        replay_applied = apply_model_diversity(
            mujoco,
            model,
            data,
            diversity["assignment"],
        )
        if replay_applied != diversity.get("applied"):
            raise RuntimeError(
                "audit replay diversity does not match recorded metadata"
            )
        replay_diversity = {
            "assignment_id": diversity["assignment"]["assignment_id"],
            "applied": replay_applied,
            "matches_recorded": True,
        }
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.enable_segmentation_rendering()
    visible_option = mujoco.MjvOption()
    isolated_option = mujoco.MjvOption()
    isolated_option.geomgroup[:] = 0
    isolated_option.geomgroup[0] = 1

    roll_visual = _named_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "sorting_roll_visual",
    )
    roll_body = _named_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "sorting_roll",
    )
    hand_geom_ids = {
        "left_wrist": _subtree_geom_ids(
            model,
            _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "L_pgc140_mount"),
        ),
        "right_wrist": _subtree_geom_ids(
            model,
            _named_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "R_pgc140_mount"),
        ),
    }
    audit_cameras = tuple(dict.fromkeys(
        camera for cameras in CAMERA_SETS.values() for camera in cameras
    ))
    original_groups = model.geom_group.copy()
    records = []
    try:
        for stage, phases in STAGE_PHASES.items():
            for phase in phases:
                if phase not in spans:
                    continue
                start, end = spans[phase]
                for frame in sampled_indices(start, end, args.samples_per_phase):
                    _restore_frame(mujoco, model, data, episode, meta, frame)
                    wrist_references = _wrist_reference_points(
                        mujoco, model, data, roll_body
                    )
                    for camera in audit_cameras:
                        source_camera = CAMERA_SOURCES[camera]
                        role = CAMERA_ROLES[camera]
                        camera_id = _named_id(
                            mujoco,
                            model,
                            mujoco.mjtObj.mjOBJ_CAMERA,
                            source_camera,
                        )
                        model.geom_group[:] = original_groups
                        renderer.update_scene(
                            data,
                            camera=source_camera,
                            scene_option=visible_option,
                        )
                        segmentation = renderer.render()
                        visible_mask = (
                            (segmentation[:, :, 0] == roll_visual)
                            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                        )
                        hand_visible_pixels = 0
                        if role in hand_geom_ids:
                            hand_visible_pixels = int(np.count_nonzero(
                                np.isin(
                                    segmentation[:, :, 0],
                                    list(hand_geom_ids[role]),
                                )
                                & (
                                    segmentation[:, :, 1]
                                    == int(mujoco.mjtObj.mjOBJ_GEOM)
                                )
                            ))

                        model.geom_group[:] = 5
                        model.geom_group[roll_visual] = 0
                        renderer.update_scene(
                            data,
                            camera=source_camera,
                            scene_option=isolated_option,
                        )
                        segmentation = renderer.render()
                        full_mask = (
                            (segmentation[:, :, 0] == roll_visual)
                            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                        )
                        metrics = mask_metrics(visible_mask, full_mask)
                        _, _, target_in_frame = project_points(
                            data.cam_xpos[camera_id],
                            data.cam_xmat[camera_id],
                            TARGET_POINTS_M,
                            float(model.cam_fovy[camera_id]),
                            args.width,
                            args.height,
                        )
                        target_fraction = float(np.mean(target_in_frame))
                        hand_reference_in_frame = None
                        roll_contact_in_frame = None
                        target_contact_in_frame = None
                        if role in wrist_references:
                            references = wrist_references[role]
                            _, _, local_in_frame = project_points(
                                data.cam_xpos[camera_id],
                                data.cam_xmat[camera_id],
                                np.asarray([
                                    references["hand_reference"],
                                    references["roll_contact"],
                                    references["target_contact"],
                                ]),
                                float(model.cam_fovy[camera_id]),
                                args.width,
                                args.height,
                            )
                            (
                                hand_reference_in_frame,
                                roll_contact_in_frame,
                                target_contact_in_frame,
                            ) = (bool(value) for value in local_in_frame)
                            usable = wrist_observation_is_usable(
                                metrics,
                                hand_visible_pixels=hand_visible_pixels,
                                hand_reference_in_frame=hand_reference_in_frame,
                                roll_contact_in_frame=roll_contact_in_frame,
                                target_contact_in_frame=target_contact_in_frame,
                                placement_stage=stage in PLACEMENT_STAGES,
                            )
                        else:
                            usable = observation_is_usable(
                                metrics,
                                target_fraction,
                                stage in PLACEMENT_STAGES,
                            )
                        records.append({
                            "stage": stage,
                            "phase": phase,
                            "frame": int(frame),
                            "timestamp_s": float(episode["timestamp"][frame]),
                            "camera": camera,
                            "model_camera": source_camera,
                            "role": role,
                            **metrics,
                            "target_in_frame_fraction": target_fraction,
                            "hand_visible_pixels": hand_visible_pixels,
                            "hand_reference_in_frame": hand_reference_in_frame,
                            "roll_contact_in_frame": roll_contact_in_frame,
                            "target_contact_in_frame": target_contact_in_frame,
                            "usable": usable,
                        })
    finally:
        model.geom_group[:] = original_groups
        renderer.close()

    candidates = {
        name: aggregate_candidate(records, cameras)
        for name, cameras in CAMERA_SETS.items()
    }
    report = {
        "schema_version": 2,
        "screen_only_not_training_readiness": True,
        "episode": str(args.episode.resolve()),
        "scene": str(args.scene.resolve()),
        "resolution": [args.height, args.width],
        "samples_per_phase": args.samples_per_phase,
        "skipped_noop_phases": skipped_noop_phases,
        "task_version": meta.get("task_version"),
        "replay_diversity": replay_diversity,
        "realsense_profile": realsense_profile_report(),
        "thresholds": {
            "min_full_roll_pixels": MIN_FULL_ROLL_PIXELS,
            "min_visible_fraction": MIN_VISIBLE_FRACTION,
            "min_extent_fraction": MIN_EXTENT_FRACTION,
            "min_target_in_frame_fraction": MIN_TARGET_IN_FRAME_FRACTION,
            "min_wrist_roll_pixels": MIN_WRIST_ROLL_PIXELS,
            "min_wrist_hand_pixels": MIN_WRIST_HAND_PIXELS,
        },
        "limitations": [
            "SDK camera intrinsics and real CameraInfo remain unverified",
            "RealSense wrist views use unverified simulation proxy mounts",
            "left and right simulation proxy mounts are asymmetric",
            "screening visibility does not prove pi0.5 learnability",
            "simulator truth is used only to measure visibility, not as policy input",
        ],
        "candidates": candidates,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--samples-per-phase", type=int, default=3)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = args.episode / "camera_observability.json"
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.width < 64 or args.height < 64:
        parser.error("audit resolution must be at least 64x64")
    if args.samples_per_phase < 1:
        parser.error("--samples-per-phase must be positive")
    return args


def main():
    args = parse_args()
    report = run_audit(args)
    print(json.dumps(report["candidates"], ensure_ascii=False, indent=2))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
