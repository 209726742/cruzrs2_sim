#!/usr/bin/env python3
"""Collect one grouped H/T/R/C Sorting Roll v16 expansion episode."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
sys.path[:0] = [str(SCRIPT_DIR), str(CORE_DIR)]

import sorting_roll_expert as v15  # noqa: E402
from sorting_roll_v16_counterfactual_scene import (  # noqa: E402
    materialize_counterfactual_scene,
)
from sorting_roll_diversity import APPEARANCE_PROFILES  # noqa: E402
from sorting_roll_v16_expansion_contract import (  # noqa: E402
    TASK_VERSION,
    assignment_for_seed,
    load_manifest,
)
from sorting_roll_v16_pilot_expert import (  # noqa: E402
    SUPPORT_SETTLE_FRAMES,
    SortingRollV16PilotExpert,
)


DISTRACTOR_MAX_TRANSLATION_M = 0.010
DISTRACTOR_MAX_ROTATION_DEG = 2.0
DISTRACTOR_MAX_SPEED = 0.05
ORIGINAL_ROLL_X_M = 0.0
V16_SHELF_STAGE_OFFSET_X_M = -0.075
V16_RELEASE_CLEARANCE_ROLL_Z_M = 0.970
V16_RELEASE_LATERAL_STEP_M = 0.004
V16_RELEASE_LATERAL_MAX_STEPS = 3
C_RELEASE_INSERT_TARGET_X_M = float(v15.TARGET_CENTER[0]) + 0.0218
C_RELEASE_CLOSED_PREBACKOFF_M = 0.002
C_RELEASE_CLOSED_PRELIFT_M = 0.0015
C_RELEASE_OPEN_INITIAL_BACKOFF_M = 0.0
C_RELEASE_GUARDED_DROP_Z_M = 0.9470
C_RELEASE_OPEN_CLEARANCE_LIFT_MAX_M = 0.0
C_RELEASE_NEAR_SHELF_GRIP_TARGET_M = 0.0105
C_RELEASE_WRIST_LEVEL_DEG = 8.0
C_RELEASE_POST_OPEN_FORWARD_M = 0.0
C_RELEASE_PAD_GRAZE_MAX_PENETRATION_MM = 0.05
C_RELEASE_PAD_GRAZE_MAX_FORCE_N = 0.5
C_TABLE_OBSERVATION_MAX_SPEED_M_S = 0.28
C_TABLE_NAVIGATION_MAX_YAW_RATE_RAD_S = 0.65
V16_RELEASE_STAGE_CENTER_TOLERANCE_XZ_M = 0.0005
V16_RELEASE_CLEARANCE_LATERAL_STEP_MAX_M = 0.0015
V16_SHORT_BOUNDARY_LEFT_FLAT_PICK_TIP_BIAS_Y_M = 0.031
C_RELEASE_PAD_GRAZE_LABELS = frozenset({
    "guarded_release_and_lift_open_hands",
    "guarded_release_initial_clearance_settle",
    "guarded_release_clear_confirmation",
    "guarded_release_final_settle",
})


def c_release_contacts_are_incidental(contacts):
    if not contacts:
        return False
    pads = {"L_pad1", "L_pad2", "R_pad1", "R_pad2"}
    lip = "shelf_top_front_lip_col"
    for contact in contacts:
        pair = set(contact.get("pair", ()))
        if len(pair) != 2 or lip not in pair or not pair & pads:
            return False
        if (
            float(contact.get("penetration_mm", math.inf))
            > C_RELEASE_PAD_GRAZE_MAX_PENETRATION_MM
            or float(contact.get("normal_force_n", math.inf))
            > C_RELEASE_PAD_GRAZE_MAX_FORCE_N
        ):
            return False
    return True


def shelf_stage_offset_x_m(family):
    if family == "C":
        return float(v15.SHELF_STAGE_OFFSET_X)
    return V16_SHELF_STAGE_OFFSET_X_M


def rigid_motion_metrics(initial_qpos, final_qpos, final_qvel):
    initial_qpos = np.asarray(initial_qpos, dtype=float)
    final_qpos = np.asarray(final_qpos, dtype=float)
    final_qvel = np.asarray(final_qvel, dtype=float)
    if initial_qpos.shape != (7,) or final_qpos.shape != (7,):
        raise ValueError("free-joint qpos must have shape (7,)")
    if final_qvel.shape != (6,):
        raise ValueError("free-joint qvel must have shape (6,)")
    initial_quat = initial_qpos[3:] / np.linalg.norm(initial_qpos[3:])
    final_quat = final_qpos[3:] / np.linalg.norm(final_qpos[3:])
    cosine = float(np.clip(abs(initial_quat @ final_quat), 0.0, 1.0))
    initial_axis = np.array([
        1.0 - 2.0 * (initial_quat[2] ** 2 + initial_quat[3] ** 2),
        2.0 * (initial_quat[1] * initial_quat[2]
               + initial_quat[0] * initial_quat[3]),
        2.0 * (initial_quat[1] * initial_quat[3]
               - initial_quat[0] * initial_quat[2]),
    ])
    final_axis = np.array([
        1.0 - 2.0 * (final_quat[2] ** 2 + final_quat[3] ** 2),
        2.0 * (final_quat[1] * final_quat[2]
               + final_quat[0] * final_quat[3]),
        2.0 * (final_quat[1] * final_quat[3]
               - final_quat[0] * final_quat[2]),
    ])
    axis_cosine = float(np.clip(abs(initial_axis @ final_axis), 0.0, 1.0))
    return {
        "translation_m": float(np.linalg.norm(final_qpos[:3] - initial_qpos[:3])),
        "rotation_deg": float(math.degrees(math.acos(axis_cosine))),
        "raw_rotation_deg": float(2.0 * math.degrees(math.acos(cosine))),
        "speed": float(np.linalg.norm(final_qvel)),
        "initial_position_m": np.round(initial_qpos[:3], 7).tolist(),
        "final_position_m": np.round(final_qpos[:3], 7).tolist(),
    }


class SortingRollV16ExpansionExpert(SortingRollV16PilotExpert):
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
        self.expansion_assignment = assignment
        super().__init__(
            args,
            ct,
            mujoco,
            scheduler,
            evaluate_placement,
            tracker_cls,
            assignment,
        )
        self.task_version = TASK_VERSION
        self.shelf_stage_offset_x_m = shelf_stage_offset_x_m(self.family)
        if self.family != "C":
            self.release_clearance_roll_z_m = (
                V16_RELEASE_CLEARANCE_ROLL_Z_M
            )
            self.release_lateral_step_m = V16_RELEASE_LATERAL_STEP_M
            self.release_lateral_max_steps = V16_RELEASE_LATERAL_MAX_STEPS
            self.release_stage_center_tolerance_m = [
                V16_RELEASE_STAGE_CENTER_TOLERANCE_XZ_M,
                v15.PRE_RELEASE_Y_TOLERANCE_M,
                V16_RELEASE_STAGE_CENTER_TOLERANCE_XZ_M,
            ]
            self.release_clearance_axis_step_limits_m = [
                V16_RELEASE_CLEARANCE_LATERAL_STEP_MAX_M,
                V16_RELEASE_CLEARANCE_LATERAL_STEP_MAX_M,
                0.020,
            ]
            base_assignment = assignment["base_diversity_assignment"]
            if (
                self.family == "H"
                and base_assignment["object_profile"]["name"] == "short_slim"
                and base_assignment["pose_bin"] == "boundary"
            ):
                self.flat_pick_tip_bias_y_m_by_hand = {
                    "l": V16_SHORT_BOUNDARY_LEFT_FLAT_PICK_TIP_BIAS_Y_M
                }
        self.counterfactual_scene_report = getattr(
            args, "counterfactual_scene_report", None
        )
        self.counterfactual_evidence = None
        self.distractor_initial_qpos = None
        if self.family == "C":
            self.table_observation_max_speed = (
                C_TABLE_OBSERVATION_MAX_SPEED_M_S
            )
            self.table_navigation_max_yaw_rate = (
                C_TABLE_NAVIGATION_MAX_YAW_RATE_RAD_S
            )
        self.ct.REC["metadata"].update({
            "task_version": TASK_VERSION,
            "v16_shelf_stage_offset_x_m": self.shelf_stage_offset_x_m,
            "v16_release_clearance_roll_z_m": getattr(
                self,
                "release_clearance_roll_z_m",
                v15.RELEASE_CLEARANCE_ROLL_Z,
            ),
            "v16_release_lateral_step_m": getattr(
                self,
                "release_lateral_step_m",
                0.0,
            ),
            "v16_release_lateral_max_steps": getattr(
                self,
                "release_lateral_max_steps",
                0,
            ),
            "v16_release_stage_center_tolerance_m": getattr(
                self,
                "release_stage_center_tolerance_m",
                [
                    0.002,
                    v15.PRE_RELEASE_Y_TOLERANCE_M,
                    0.002,
                ],
            ),
            "v16_release_clearance_axis_step_limits_m": getattr(
                self,
                "release_clearance_axis_step_limits_m",
                None,
            ),
            "v16_flat_pick_tip_bias_y_m_by_hand": getattr(
                self,
                "flat_pick_tip_bias_y_m_by_hand",
                {},
            ),
            "target_lane": assignment["target_lane"],
            "distractor_color": assignment["distractor_color"],
            "counterfactual_scene": assignment["counterfactual_scene"],
            "counterfactual_scene_report": self.counterfactual_scene_report,
            "counterfactual_evidence": None,
            "table_observation_max_speed_m_s": getattr(
                self, "table_observation_max_speed", 0.26
            ),
            "table_navigation_max_yaw_rate_rad_s": getattr(
                self,
                "table_navigation_max_yaw_rate",
                v15.BASE_MAX_YAW_RATE,
            ),
        })
        if self.family == "C":
            self.distractor_body = ct.bid("sorting_roll_distractor")
            self.distractor_joint = ct.jid("sorting_roll_distractor_free")
            self.distractor_geom = ct.gid("sorting_roll_distractor_col")
            self.distractor_qpos_adr = int(
                self.model.jnt_qposadr[self.distractor_joint]
            )
            self.distractor_dof_adr = int(
                self.model.jnt_dofadr[self.distractor_joint]
            )
            self.distractor_support_geom_ids = {
                ct.gid(f"distractor_roll_support_x_{side}_{part}_col")
                for side in ("negative", "positive")
                for part in ("base", "robot_lip", "far_lip")
            }

    def apply_scene_randomization(self):
        assignment = getattr(self, "expansion_assignment", {})
        if assignment.get("scenario_family") != "C":
            return super().apply_scene_randomization()
        scene = assignment["counterfactual_scene"]
        episode_seed = self.args.seed
        self.args.seed = int(scene["scene_randomization_seed"])
        try:
            super().apply_scene_randomization()
        finally:
            self.args.seed = episode_seed

        distractor_joint = self.ct.jid("sorting_roll_distractor_free")
        qpos_adr = int(self.model.jnt_qposadr[distractor_joint])
        dof_adr = int(self.model.jnt_dofadr[distractor_joint])
        delta_xy = np.asarray(
            self.scene_randomization["roll_delta_xy_m"], dtype=float
        )
        self.data.qpos[qpos_adr:qpos_adr + 2] += delta_xy
        half_yaw = 0.5 * float(self.scene_randomization["roll_yaw_rad"])
        yaw_quaternion = np.asarray([
            math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)
        ])
        rotated = np.empty(4, dtype=float)
        self.mujoco.mju_mulQuat(
            rotated,
            yaw_quaternion,
            self.data.qpos[qpos_adr + 3:qpos_adr + 7],
        )
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = rotated
        self.data.qvel[dof_adr:dof_adr + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.scene_randomization.update({
            "episode_seed": int(episode_seed),
            "counterfactual_scene_seed": int(
                scene["scene_randomization_seed"]
            ),
            "paired_roll_randomization": True,
        })
        self.ct.REC["metadata"]["scene_randomization"] = (
            self.scene_randomization
        )

    def _target_binding_evidence(self):
        expected_target = np.asarray(
            self.expansion_assignment["base_diversity_assignment"]
            ["appearance_profile"]["rgba"],
            dtype=float,
        )
        expected_distractor = np.asarray(
            APPEARANCE_PROFILES[
                self.expansion_assignment["distractor_color"]
            ]["rgba"],
            dtype=float,
        )
        target_rgba = self.model.geom_rgba[self.roll_geom].copy()
        distractor_rgba = self.model.geom_rgba[self.distractor_geom].copy()
        return {
            "target_object_id": "sorting_roll",
            "distractor_object_id": "sorting_roll_distractor",
            "target_lane": self.expansion_assignment["target_lane"],
            "target_color": self.expansion_assignment["target_color"],
            "distractor_color": self.expansion_assignment["distractor_color"],
            "target_rgba": np.round(target_rgba, 5).tolist(),
            "distractor_rgba": np.round(distractor_rgba, 5).tolist(),
            "max_rgba_error": float(max(
                np.max(np.abs(target_rgba - expected_target)),
                np.max(np.abs(distractor_rgba - expected_distractor)),
            )),
        }

    def _prepare_counterfactual_recording(self):
        self.ct.REC["on"] = False
        v15.SortingRollExpert.phase(self, "v16_c_pair_physical_settle")
        for _ in range(SUPPORT_SETTLE_FRAMES):
            self.tick()
        target_support = self.contact_evidence(
            {self.roll_geom}, self.pickup_support_geom_ids
        )
        distractor_support = self.contact_evidence(
            {self.distractor_geom}, self.distractor_support_geom_ids
        )
        target_speed = float(np.linalg.norm(
            self.data.qvel[self.roll_dof_adr:self.roll_dof_adr + 6]
        ))
        distractor_speed = float(np.linalg.norm(
            self.data.qvel[
                self.distractor_dof_adr:self.distractor_dof_adr + 6
            ]
        ))
        self.gate(
            "v16_c_pair_physical_settle",
            bool(target_support["pairs"])
            and bool(distractor_support["pairs"])
            and target_speed <= DISTRACTOR_MAX_SPEED
            and distractor_speed <= DISTRACTOR_MAX_SPEED,
            f"target_support={target_support} distractor_support="
            f"{distractor_support} target_speed={target_speed:.5f} "
            f"distractor_speed={distractor_speed:.5f}",
        )
        binding = self._target_binding_evidence()
        self.gate(
            "v16_c_target_binding",
            binding["max_rgba_error"] <= 1e-6,
            json.dumps(binding, ensure_ascii=False),
        )
        self.distractor_initial_qpos = self.data.qpos[
            self.distractor_qpos_adr:self.distractor_qpos_adr + 7
        ].copy()
        self.counterfactual_evidence = {
            "target_binding": binding,
            "initial_target_support": target_support,
            "initial_distractor_support": distractor_support,
        }
        self.ct.REC["metadata"]["counterfactual_preflight_sim_seconds"] = round(
            float(self.sim_seconds), 4
        )
        self.sim_seconds = 0.0
        self.start_recording("initial_hold")

    def _finish_counterfactual_evidence(self):
        final_qpos = self.data.qpos[
            self.distractor_qpos_adr:self.distractor_qpos_adr + 7
        ].copy()
        final_qvel = self.data.qvel[
            self.distractor_dof_adr:self.distractor_dof_adr + 6
        ].copy()
        motion = rigid_motion_metrics(
            self.distractor_initial_qpos, final_qpos, final_qvel
        )
        self.gate(
            "v16_c_distractor_stationary",
            motion["translation_m"] <= DISTRACTOR_MAX_TRANSLATION_M
            and motion["rotation_deg"] <= DISTRACTOR_MAX_ROTATION_DEG
            and motion["speed"] <= DISTRACTOR_MAX_SPEED,
            json.dumps(motion, ensure_ascii=False),
        )
        final = self.final_evidence or {}
        self.gate(
            "v16_c_target_in_integrated_slot",
            final.get("instantaneous_success") is True
            and final.get("stable_seconds", 0.0) >= 2.0,
            json.dumps(final, ensure_ascii=False),
        )
        self.counterfactual_evidence.update({
            "distractor_motion": motion,
            "target_final_evidence": final,
            "target_selected_by_prompt": True,
        })

    def require_arms_clear_shelf(self, label):
        if self.family != "C" or label not in C_RELEASE_PAD_GRAZE_LABELS:
            return super().require_arms_clear_shelf(label)
        contacts = self.arm_shelf_contacts()
        if not contacts:
            return
        if not c_release_contacts_are_incidental(contacts):
            raise v15.ExpertFailure(
                f"arm-shelf collision phase={label} contacts={contacts}"
            )
        evidence = self.counterfactual_evidence.setdefault(
            "release_pad_front_lip_soft_contact",
            {
                "event_count": 0,
                "max_penetration_mm": 0.0,
                "max_normal_force_n": 0.0,
                "penetration_limit_mm": (
                    C_RELEASE_PAD_GRAZE_MAX_PENETRATION_MM
                ),
                "normal_force_limit_n": C_RELEASE_PAD_GRAZE_MAX_FORCE_N,
            },
        )
        evidence["event_count"] += len(contacts)
        evidence["max_penetration_mm"] = max(
            evidence["max_penetration_mm"],
            *(contact["penetration_mm"] for contact in contacts),
        )
        evidence["max_normal_force_n"] = max(
            evidence["max_normal_force_n"],
            *(contact["normal_force_n"] for contact in contacts),
        )
        print(
            "[gate:v16_c_release_pad_front_lip_soft_contact] PASS "
            f"phase={label} contacts={contacts}",
            flush=True,
        )

    def clear_released_hands_before_lift(self):
        if self.family != "C":
            return super().clear_released_hands_before_lift()
        placement = self.evaluate_placement(self.model, self.data)
        contacts = self.arm_shelf_contacts()
        if contacts:
            self.require_arms_clear_shelf(
                "guarded_release_and_lift_open_hands"
            )
        self.gates["v16_c_pre_lift_evidence"] = {
            "placement": placement,
            "arm_shelf_contacts": contacts,
        }
        self.gate(
            "v16_c_released_hands_ready_for_lift",
            placement.get("instantaneous_success") is True
            and placement["checks"]["released_from_both_grippers"]
            and (
                not contacts
                or c_release_contacts_are_incidental(contacts)
            ),
            f"contacts={contacts} placement="
            f"{json.dumps(placement, ensure_ascii=False)}",
        )

    def release_into_integrated_top_tier(self):
        if self.family != "C":
            return super().release_into_integrated_top_tier()
        self.move_mount_commands_delta(
            [
                -C_RELEASE_CLOSED_PREBACKOFF_M,
                0.0,
                C_RELEASE_CLOSED_PRELIFT_M,
            ],
            shelf_safe=True,
        )
        self.require_held("v16_c_closed_prebackoff")
        self.require_arms_clear_shelf("v16_c_closed_prebackoff")
        depth_margin = self.current_integrated_depth_margin()
        self.gate(
            "v16_c_closed_prebackoff",
            depth_margin >= 0.005,
            f"backoff_mm={1000.0 * C_RELEASE_CLOSED_PREBACKOFF_M:.1f} "
            f"prelift_mm={1000.0 * C_RELEASE_CLOSED_PRELIFT_M:.1f} "
            f"depth_margin_mm={1000.0 * depth_margin:.2f}",
        )
        return super().release_into_integrated_top_tier()

    def execute(self):
        if self.family != "C":
            return super().execute()
        self._prepare_counterfactual_recording()
        counterfactual_scene = self.expansion_assignment[
            "counterfactual_scene"
        ]
        target_x = counterfactual_scene["lane_x_m"][
            self.expansion_assignment["target_lane"]
        ]
        grasp_offset = float(target_x - ORIGINAL_ROLL_X_M)
        observation_offset = grasp_offset
        self.ct.REC["metadata"]["counterfactual_base_approach"] = {
            "target_x_m": float(target_x),
            "observation_offset_x_m": observation_offset,
            "grasp_base_offset_x_m": grasp_offset,
            "release_insert_target_x_m": C_RELEASE_INSERT_TARGET_X_M,
            "release_closed_prebackoff_m": C_RELEASE_CLOSED_PREBACKOFF_M,
            "release_closed_prelift_m": C_RELEASE_CLOSED_PRELIFT_M,
            "release_open_initial_backoff_m": C_RELEASE_OPEN_INITIAL_BACKOFF_M,
            "release_guarded_drop_z_m": C_RELEASE_GUARDED_DROP_Z_M,
            "release_open_clearance_lift_max_m": (
                C_RELEASE_OPEN_CLEARANCE_LIFT_MAX_M
            ),
            "release_near_shelf_grip_target_m": (
                C_RELEASE_NEAR_SHELF_GRIP_TARGET_M
            ),
            "release_wrist_level_deg": C_RELEASE_WRIST_LEVEL_DEG,
            "release_post_open_forward_m": C_RELEASE_POST_OPEN_FORWARD_M,
        }
        old_observation = v15.TABLE_OBSERVATION_XY.copy()
        old_grasp = v15.TABLE_GRASP_XY.copy()
        old_release_insert = v15.RELEASE_INSERT_TARGET_X_M
        old_release_backoff = v15.RELEASE_OPEN_INITIAL_BACKOFF_M
        old_release_drop_z = v15.RELEASE_GUARDED_DROP_Z_M
        old_release_open_lift = v15.RELEASE_OPEN_CLEARANCE_LIFT_MAX_M
        old_release_near_grip = v15.RELEASE_NEAR_SHELF_GRIP_TARGET_M
        old_release_wrist_level = v15.RELEASE_WRIST_LEVEL_DEG
        v15.TABLE_OBSERVATION_XY = (
            old_observation + [observation_offset, 0.0]
        )
        v15.TABLE_GRASP_XY = old_grasp + [grasp_offset, 0.0]
        v15.RELEASE_INSERT_TARGET_X_M = C_RELEASE_INSERT_TARGET_X_M
        v15.RELEASE_OPEN_INITIAL_BACKOFF_M = C_RELEASE_OPEN_INITIAL_BACKOFF_M
        v15.RELEASE_GUARDED_DROP_Z_M = C_RELEASE_GUARDED_DROP_Z_M
        v15.RELEASE_OPEN_CLEARANCE_LIFT_MAX_M = (
            C_RELEASE_OPEN_CLEARANCE_LIFT_MAX_M
        )
        v15.RELEASE_NEAR_SHELF_GRIP_TARGET_M = (
            C_RELEASE_NEAR_SHELF_GRIP_TARGET_M
        )
        v15.RELEASE_WRIST_LEVEL_DEG = C_RELEASE_WRIST_LEVEL_DEG
        try:
            success = super().execute()
        finally:
            v15.TABLE_OBSERVATION_XY = old_observation
            v15.TABLE_GRASP_XY = old_grasp
            v15.RELEASE_INSERT_TARGET_X_M = old_release_insert
            v15.RELEASE_OPEN_INITIAL_BACKOFF_M = old_release_backoff
            v15.RELEASE_GUARDED_DROP_Z_M = old_release_drop_z
            v15.RELEASE_OPEN_CLEARANCE_LIFT_MAX_M = old_release_open_lift
            v15.RELEASE_NEAR_SHELF_GRIP_TARGET_M = old_release_near_grip
            v15.RELEASE_WRIST_LEVEL_DEG = old_release_wrist_level
        self._finish_counterfactual_evidence()
        return success

    def finalize(self, success, error=None):
        self.ct.REC["metadata"]["counterfactual_evidence"] = (
            self.counterfactual_evidence
        )
        super().finalize(success, error=error)
        common = {
            "target_lane": self.expansion_assignment["target_lane"],
            "distractor_color": self.expansion_assignment[
                "distractor_color"
            ],
            "counterfactual_scene": self.expansion_assignment[
                "counterfactual_scene"
            ],
            "counterfactual_scene_report": self.counterfactual_scene_report,
            "counterfactual_evidence": self.counterfactual_evidence,
        }
        meta_path = self.out / "meta.json"
        if meta_path.is_file():
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
        raise SystemExit("v16 expansion requires rendered policy cameras")
    if args.manifest is None:
        raise SystemExit("v16 expansion requires --manifest")
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    args.manifest = args.manifest.resolve()
    manifest = load_manifest(args.manifest)
    assignment = assignment_for_seed(manifest, args.seed)
    args.diversity_assignment = assignment["base_diversity_assignment"]
    prompt = assignment["prompt"]

    import sorting_roll_scene

    base_scene = sorting_roll_scene.materialize_scene()
    derived_scene = None
    scene_path = base_scene
    if assignment["scenario_family"] == "C":
        derived_scene = base_scene.with_name(
            f"sorting_roll_v16_c_{args.seed}_{os.getpid()}.xml"
        )
        args.counterfactual_scene_report = materialize_counterfactual_scene(
            base_scene, derived_scene, assignment
        )
        scene_path = derived_scene
    try:
        ct = v15.load_teleop(scene_path, args.gpu, args.seed, prompt=prompt)
        import mujoco

        v15.apply_model_camera_overrides(mujoco, ct.m)
        from sorting_roll_task import (
            SortingRollSuccessTracker,
            evaluate_placement,
        )
        from teleop_timing import CumulativeSubstepScheduler

        scheduler = CumulativeSubstepScheduler(
            ct.TARGET_FPS, ct.m.opt.timestep
        )
        expert = SortingRollV16ExpansionExpert(
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
        old_shelf_stage_offset = v15.SHELF_STAGE_OFFSET_X
        try:
            v15.SHELF_STAGE_OFFSET_X = expert.shelf_stage_offset_x_m
            success = expert.execute()
        except v15.ExpertFailure as exc:
            error = str(exc)
            print(f"[v16 expansion] FAIL {error}", flush=True)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[v16 expansion] ERROR {error}", flush=True)
            raise
        finally:
            v15.SHELF_STAGE_OFFSET_X = old_shelf_stage_offset
            expert.finalize(success, error=error)
    finally:
        if derived_scene is not None:
            derived_scene.unlink(missing_ok=True)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
