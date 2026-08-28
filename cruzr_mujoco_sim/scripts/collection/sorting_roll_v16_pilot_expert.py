#!/usr/bin/env python3
"""Collect one H/T/R episode for the Sorting Roll v16 expansion pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path[:0] = [str(SCRIPT_DIR), str(CORE_DIR)]

import sorting_roll_expert as v15  # noqa: E402
from sorting_roll_v16_pilot_contract import (  # noqa: E402
    TASK_VERSION,
    assignment_for_seed,
    load_manifest,
)


PARTIAL_LIFT_M = 0.020
SUPPORT_SETTLE_FRAMES = 120
SUPPORT_STABILITY_WINDOW_FRAMES = 60
SUPPORT_MAX_CENTER_EXCURSION_M = 0.002
SUPPORT_MAX_AXIS_EXCURSION_DEG = 0.5
SUPPORT_GEOM_NAMES = tuple(
    f"roll_support_x_{side}_{part}_{kind}"
    for side in ("negative", "positive")
    for part in ("base", "robot_lip", "far_lip")
    for kind in ("visual", "col")
)


def support_stability_metrics(positions, axes):
    positions = np.asarray(positions, dtype=float)
    axes = np.asarray(axes, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("support stability positions must have shape (n, 3)")
    if axes.shape != positions.shape:
        raise ValueError("support stability axes must match positions")
    axis_norms = np.linalg.norm(axes, axis=1)
    if np.any(axis_norms <= 0.0):
        raise ValueError("support stability axes must be non-zero")
    normalized = axes / axis_norms[:, None]
    cosine = np.clip(np.abs(normalized @ normalized[-1]), -1.0, 1.0)
    return {
        "max_center_excursion_m": float(
            np.max(np.linalg.norm(positions - positions[-1], axis=1))
        ),
        "max_axis_excursion_deg": float(
            np.max(np.degrees(np.arccos(cosine)))
        ),
    }


class SortingRollV16PilotExpert(v15.SortingRollExpert):
    def __init__(
        self,
        args,
        ct,
        mujoco,
        scheduler,
        evaluate_placement,
        tracker_cls,
        assignment,
    ):
        super().__init__(
            args,
            ct,
            mujoco,
            scheduler,
            evaluate_placement,
            tracker_cls,
        )
        self.v16_assignment = assignment
        self.family = assignment["scenario_family"]
        self.variant = assignment["scenario_variant"]
        self.task_version = TASK_VERSION
        base_diversity = self.diversity
        self.support_transform_report = self.apply_pickup_support_transform(
            assignment["requested_transforms"]["pickup_support_and_roll"]
        )
        self.diversity = {
            "assignment": assignment,
            "base_diversity": base_diversity,
            "applied_transforms": {
                "pickup_support_and_roll": self.support_transform_report,
            },
            "manifest": str(args.manifest.resolve()),
        }
        self.recording_started = False
        self.recording_start_sim_seconds = None
        self.recovery_injected = False
        self.intervention_evidence = None
        self.ct.REC["on"] = False
        self.ct.REC["metadata"].update({
            "task_version": TASK_VERSION,
            "diversity": self.diversity,
            "scenario_family": self.family,
            "scenario_variant": self.variant,
            "scene_group_id": assignment["scene_group_id"],
            "counterfactual_pair_id": assignment["counterfactual_pair_id"],
            "start_phase": assignment["start_phase"],
            "terminal_phase": assignment["terminal_phase"],
            "intervention_type": assignment["intervention_type"],
            "intervention_frame": assignment["intervention_frame"],
            "recovery_start_frame": assignment["recovery_start_frame"],
            "target_object_id": assignment["target_object_id"],
            "target_color": assignment["target_color"],
            "distractor_object_ids": assignment["distractor_object_ids"],
            "requested_transforms": assignment["requested_transforms"],
            "applied_transforms": {
                "pickup_support_and_roll": self.support_transform_report,
            },
        })

    def apply_pickup_support_transform(self, requested):
        delta = np.array(
            [requested["x_m"], requested["y_m"], requested["z_m"]],
            dtype=float,
        )
        if abs(float(requested["yaw_rad"])) > 1e-12:
            raise ValueError("v16 pilot does not rotate pickup supports")
        applied = {}
        for name in SUPPORT_GEOM_NAMES:
            geom = self.ct.gid(name)
            before = self.model.geom_pos[geom].copy()
            self.model.geom_pos[geom] = before + delta
            applied[name] = {
                "before_m": np.round(before, 7).tolist(),
                "after_m": np.round(self.model.geom_pos[geom], 7).tolist(),
            }
        roll_before = self.data.qpos[
            self.roll_qpos_adr:self.roll_qpos_adr + 3
        ].copy()
        self.data.qpos[
            self.roll_qpos_adr:self.roll_qpos_adr + 3
        ] = roll_before + delta
        self.data.qvel[self.roll_dof_adr:self.roll_dof_adr + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return {
            "requested": dict(requested),
            "applied_delta_m": np.round(delta, 7).tolist(),
            "support_geoms": applied,
            "roll_before_m": np.round(roll_before, 7).tolist(),
            "roll_after_m": np.round(roll_before + delta, 7).tolist(),
            "visual_collision_and_roll_consistent": True,
        }

    def start_recording(self, phase):
        if self.recording_started or self.recorder.n:
            raise v15.ExpertFailure("v16 recording can only start once at frame zero")
        v15.SortingRollExpert.phase(self, phase)
        self.recording_started = True
        self.recording_start_sim_seconds = float(self.sim_seconds)
        self.ct.REC["metadata"]["recorded_start_phase"] = phase
        self.ct.REC["on"] = True

    def phase(self, name):
        if (
            self.family == "T"
            and name == self.v16_assignment["start_phase"]
            and not self.recording_started
        ):
            v15.SortingRollExpert.phase(self, name)
            self.start_recording(name)
            return
        if (
            self.variant == "double_hand_miss"
            and name == "horizontal_approach_and_grasp"
            and not self.recovery_injected
        ):
            self.ct.grip_cmd["l"] = self.ct.GRIP_CLOSE
            self.ct.grip_cmd["r"] = self.ct.GRIP_CLOSE
            self.frames(v15.GRASP_SETTLE_TICKS)
            left = self.grip_evidence("L")
            right = self.grip_evidence("R")
            self.gate(
                "v16_constructed_double_hand_miss",
                left["force_n"] <= 0.05
                and right["force_n"] <= 0.05
                and not left["pads"]
                and not right["pads"],
                f"left={left} right={right}",
            )
            self.intervention_evidence = {
                "type": self.variant,
                "completed_before_recording": True,
                "left": left,
                "right": right,
            }
            self.start_recording("recovery_double_hand_miss_open_and_retry")
            self.ct.grip_cmd["l"] = self.ct.GRIP_OPEN
            self.ct.grip_cmd["r"] = self.ct.GRIP_OPEN
            self.open_hand_until_released("l", "v16_double_miss")
            self.open_hand_until_released("r", "v16_double_miss")
            self.recovery_injected = True
            v15.SortingRollExpert.phase(self, name)
            return
        v15.SortingRollExpert.phase(self, name)

    def _open_hand_without_recording(self, hand, stage):
        self.ct.grip_cmd[hand] = self.ct.GRIP_OPEN
        self.open_hand_until_released(hand, stage)

    def _regrasp_from_supported_roll(self):
        self.ct.grip_cmd["l"] = self.ct.GRIP_OPEN
        self.ct.grip_cmd["r"] = self.ct.GRIP_OPEN
        roll = self.roll_position()
        grasp_positions, rotations = self.flat_pick_mount_poses(roll)
        pregrasp_positions = {
            hand: position
            + np.array([0.0, v15.FLAT_PICK_PREGRASP_CLEARANCE_Y_M, 0.0])
            for hand, position in grasp_positions.items()
        }
        v15.SortingRollExpert.phase(self, "recovery_backoff_and_relocalize")
        self.move_mounts(pregrasp_positions, rotations, iterations=1200)
        support_contact = self.contact_evidence(
            self.arm_geom_ids["l"] | self.arm_geom_ids["r"],
            self.pickup_support_geom_ids,
        )
        self.gate(
            "v16_recovery_pregrasp_support_clear",
            support_contact["force_n"] <= 0.2,
            f"evidence={support_contact}",
        )
        v15.SortingRollExpert.phase(self, "recovery_reapproach_and_bimanual_grasp")
        self.move_mounts(grasp_positions, rotations, iterations=1200)
        self.ct.grip_cmd["l"] = self.ct.GRIP_CLOSE
        self.ct.grip_cmd["r"] = self.ct.GRIP_CLOSE
        self.frames(v15.GRASP_SETTLE_TICKS)
        v15.SortingRollExpert.require_held(self, "v16_recovered_flat_pickup")

    def _recover_single_hand_contact(self, support_hand):
        released_hand = "r" if support_hand == "l" else "l"
        self._open_hand_without_recording(
            released_hand, "v16_construct_single_hand"
        )
        v15.SortingRollExpert.require_hand_held(
            self, support_hand, "v16_single_hand_failure_state"
        )
        evidence = {
            "type": self.variant,
            "completed_before_recording": True,
            "support_hand": support_hand,
            "released_hand": released_hand,
            "support": self.grip_evidence(support_hand.upper()),
            "released": self.grip_evidence(released_hand.upper()),
        }
        self.intervention_evidence = evidence
        self.start_recording("recovery_single_hand_safe_release_and_regrasp")
        self.ct.grip_cmd[support_hand] = self.ct.GRIP_OPEN
        self.open_hand_until_released(
            support_hand, "v16_single_hand_safe_release"
        )
        self._regrasp_from_supported_roll()

    def _recover_partial_lift(self, slip_hand):
        support_hand = "r" if slip_hand == "l" else "l"
        support_z = float(self.roll_position()[2])
        self.move_mount_commands_delta([0.0, 0.0, PARTIAL_LIFT_M])
        partial_z = float(self.roll_position()[2])
        self._open_hand_without_recording(slip_hand, "v16_construct_partial_slip")
        v15.SortingRollExpert.require_hand_held(
            self, support_hand, "v16_partial_lift_support"
        )
        self.intervention_evidence = {
            "type": self.variant,
            "completed_before_recording": True,
            "slip_hand": slip_hand,
            "support_hand": support_hand,
            "support_z_m": support_z,
            "partial_z_m": partial_z,
            "partial_lift_m": partial_z - support_z,
        }
        self.start_recording("recovery_partial_lift_safe_lower_and_regrasp")
        self.move_mount_commands_delta([0.0, 0.0, -PARTIAL_LIFT_M])
        self.frames(8)
        lowered_z = float(self.roll_position()[2])
        self.gate(
            "v16_partial_lift_lowered_to_support",
            lowered_z <= support_z + 0.010,
            f"support_z={support_z:.5f} partial_z={partial_z:.5f} "
            f"lowered_z={lowered_z:.5f}",
        )
        self.ct.grip_cmd[support_hand] = self.ct.GRIP_OPEN
        self.open_hand_until_released(
            support_hand, "v16_partial_lift_safe_release"
        )
        self._regrasp_from_supported_roll()

    def require_held(self, stage, minimum_force=v15.GRIP_FORCE_MIN_N):
        if stage != "flat_pickup":
            return v15.SortingRollExpert.require_held(
                self, stage, minimum_force=minimum_force
            )
        if self.recovery_injected:
            result = v15.SortingRollExpert.require_held(
                self, stage, minimum_force=minimum_force
            )
            if self.variant == "double_hand_miss":
                v15.SortingRollExpert.require_held(
                    self,
                    "v16_recovered_flat_pickup",
                    minimum_force=minimum_force,
                )
            return result
        if not (
            self.variant.startswith("single_hand_contact")
            or self.variant.startswith("partial_lift_slip")
        ):
            return v15.SortingRollExpert.require_held(
                self, stage, minimum_force=minimum_force
            )
        v15.SortingRollExpert.require_held(
            self, stage, minimum_force=minimum_force
        )
        if self.variant == "single_hand_contact_left":
            self._recover_single_hand_contact("l")
        elif self.variant == "single_hand_contact_right":
            self._recover_single_hand_contact("r")
        elif self.variant == "partial_lift_slip_left":
            self._recover_partial_lift("l")
        elif self.variant == "partial_lift_slip_right":
            self._recover_partial_lift("r")
        else:
            raise v15.ExpertFailure(f"unknown recovery variant: {self.variant}")
        self.recovery_injected = True

    def execute(self):
        if self.family == "H":
            self.ct.REC["on"] = False
            v15.SortingRollExpert.phase(self, "v16_pickup_support_settle")
            positions = []
            axes = []
            support_sides = set()
            support = None
            for index in range(SUPPORT_SETTLE_FRAMES):
                self.tick()
                if index < SUPPORT_SETTLE_FRAMES - SUPPORT_STABILITY_WINDOW_FRAMES:
                    continue
                positions.append(self.roll_position())
                axes.append(self.roll_axis())
                support = self.contact_evidence(
                    {self.roll_geom}, self.pickup_support_geom_ids
                )
                for pair in support["pairs"]:
                    for name in pair:
                        if "roll_support_x_negative" in name:
                            support_sides.add("negative")
                        elif "roll_support_x_positive" in name:
                            support_sides.add("positive")
            stability = support_stability_metrics(positions, axes)
            self.gate(
                "v16_pickup_support_physical_settle",
                support_sides == {"negative", "positive"}
                and stability["max_center_excursion_m"]
                <= SUPPORT_MAX_CENTER_EXCURSION_M
                and stability["max_axis_excursion_deg"]
                <= SUPPORT_MAX_AXIS_EXCURSION_DEG,
                f"support={support} support_sides={sorted(support_sides)} "
                f"stability={stability}",
            )
            self.start_recording("initial_hold")
        success = super().execute()
        if self.family == "R" and not self.recovery_injected:
            raise v15.ExpertFailure("recovery intervention was not injected")
        return success

    def finalize(self, success, error=None):
        if self.recording_started:
            self.ct.REC["metadata"]["recorded_terminal_phase"] = (
                self.ct.REC.get("phase")
            )
            self.ct.REC["metadata"]["recorded_frames"] = int(self.recorder.n)
            self.ct.REC["metadata"]["recorded_seconds"] = round(
                self.recorder.n / float(self.ct.REC_FPS), 4
            )
        self.ct.REC["metadata"]["intervention_evidence"] = (
            self.intervention_evidence
        )
        super().finalize(success, error=error)
        common = {
            "scenario_family": self.family,
            "scenario_variant": self.variant,
            "scene_group_id": self.v16_assignment["scene_group_id"],
            "counterfactual_pair_id": self.v16_assignment[
                "counterfactual_pair_id"
            ],
            "start_phase": self.v16_assignment["start_phase"],
            "terminal_phase": self.v16_assignment["terminal_phase"],
            "recorded_start_phase": self.ct.REC["metadata"].get(
                "recorded_start_phase"
            ),
            "recorded_terminal_phase": self.ct.REC["metadata"].get(
                "recorded_terminal_phase"
            ),
            "intervention_type": self.v16_assignment["intervention_type"],
            "intervention_frame": self.v16_assignment["intervention_frame"],
            "recovery_start_frame": self.v16_assignment[
                "recovery_start_frame"
            ],
            "intervention_evidence": self.intervention_evidence,
            "target_object_id": self.v16_assignment["target_object_id"],
            "target_color": self.v16_assignment["target_color"],
            "distractor_object_ids": self.v16_assignment[
                "distractor_object_ids"
            ],
            "requested_transforms": self.v16_assignment[
                "requested_transforms"
            ],
            "applied_transforms": {
                "pickup_support_and_roll": self.support_transform_report,
            },
            "recorded_frames": int(self.recorder.n),
            "recorded_seconds": round(
                self.recorder.n / float(self.ct.REC_FPS), 4
            ),
        }
        if (self.out / "meta.json").is_file():
            meta_path = self.out / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(common)
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if self.result_path.is_file():
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
            result.update(common)
            self.result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def main(argv=None):
    args = v15.parse_args(argv)
    if args.no_render:
        raise SystemExit("v16 training pilot requires rendered policy cameras")
    if args.manifest is None:
        raise SystemExit("v16 training pilot requires --manifest")
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    args.manifest = args.manifest.resolve()
    manifest = load_manifest(args.manifest)
    assignment = assignment_for_seed(manifest, args.seed)
    args.diversity_assignment = assignment["base_diversity_assignment"]
    prompt = assignment["prompt"]

    import sorting_roll_scene

    scene_path = sorting_roll_scene.materialize_scene()
    ct = v15.load_teleop(scene_path, args.gpu, args.seed, prompt=prompt)
    import mujoco

    v15.apply_model_camera_overrides(mujoco, ct.m)
    from sorting_roll_task import evaluate_placement, SortingRollSuccessTracker
    from teleop_timing import CumulativeSubstepScheduler

    scheduler = CumulativeSubstepScheduler(ct.TARGET_FPS, ct.m.opt.timestep)
    expert = SortingRollV16PilotExpert(
        args,
        ct,
        mujoco,
        scheduler,
        evaluate_placement,
        SortingRollSuccessTracker,
        assignment,
    )
    success = False
    error = None
    try:
        success = expert.execute()
    except v15.ExpertFailure as exc:
        error = str(exc)
        print(f"[v16 expert] FAIL {error}", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[v16 expert] ERROR {error}", flush=True)
        raise
    finally:
        expert.finalize(success, error=error)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
