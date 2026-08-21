#!/usr/bin/env python3
"""Single-episode Sorting Roll expert with physical success gating and review video."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
from sorting_roll_scene import (
    TARGET_AXIS as SCENE_TARGET_AXIS,
    TARGET_CENTER as SCENE_TARGET_CENTER,
)


TASK_VERSION = "sorting_roll_v2"
POLICY_CAMERAS = ("stereo_left", "stereo_right", "waist_front")
FLAT_REGRASP_ORDER = (("r", "l", -1.0), ("l", "r", 1.0))
TARGET_CENTER = np.array(SCENE_TARGET_CENTER, dtype=float, copy=True)
TARGET_AXIS = np.array(SCENE_TARGET_AXIS, dtype=float, copy=True)
ROLL_HALF_LENGTH = 0.25
ROLL_RADIUS = 0.012
SLOT_HALF_WIDTH = 0.015
TABLE_OBSERVATION_XY = np.array([0.0, -0.45])
TABLE_GRASP_XY = np.array([0.0, -0.70])
SHELF_STAGE_OFFSET_X = -0.065
PRE_RELEASE_Y_TOLERANCE_M = 0.003
PRE_RELEASE_ENDPOINT_MARGIN_M = 0.020
ARM_RETRACT_M = 0.082
HAND_FLAT_ROLL_Z = 1.240
RELEASE_ROLL_Z = 1.128
RELEASE_APPROACH_Y_BIAS_M = 0.008
RELEASE_PRE_TOUCH_X_M = 0.7850
RELEASE_TOUCH_STEP_M = 0.0001
RELEASE_TOUCH_MAX_STEPS = 40
RELEASE_TOUCH_MIN_FORCE_N = 0.02
RELEASE_TOUCH_MAX_FORCE_N = 5.0
RELEASE_WRIST_LEVEL_DEG = 4.0
RELEASE_TIP_REGRASP_X_M = {"l": -0.035, "r": -0.034}
RELEASE_TIP_REGRASP_STAGE_X_M = 0.750
RELEASE_INSERT_STEP_M = 0.002
RELEASE_OPEN_RAISE_M = 0.004
RELEASE_AXIS_COARSE_STEP_M = 0.004
RELEASE_AXIS_FINE_STEP_M = 0.0001
RELEASE_AXIS_COARSE_STEPS = 12
RELEASE_AXIS_MAX_STEPS = 80
RELEASE_PAD_SLIDING_FRICTION = 1.0
RELEASE_FRICTION_SETTLE_TICKS = 12
GRASP_YAW_DEG = 14.0
FLAT_REGRASP_ANGLE_DEG = 94.0
FLAT_REGRASP_TARGET_ALONG_M = 0.160
FLAT_REGRASP_COUPLED_START_M = 0.180
FLAT_REGRASP_NEAR_END_M = 0.270
FLAT_REGRASP_FAR_END_M = 0.290
FLAT_REGRASP_CLEARANCE = np.array([0.0, 0.043, 0.020])
FLAT_REGRASP_CLEARANCE_ONSET = 0.45
FLAT_REGRASP_ROTATION_EXPONENT = 2.0
FLAT_REGRASP_COUPLED_MIN_STEPS = 60
FLAT_REGRASP_ABSOLUTE_ROTATION_STEP_DEG = 1.0
FLAT_REGRASP_CART_STEP_M = 0.003
FLAT_REGRASP_COLLISION_STEP_RAD = 0.005
FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M = 0.005
FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M = 0.003
FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M = 0.008
FLAT_REGRASP_ANCHOR_CORRECTION_ATTEMPTS = 3
FLAT_REGRASP_HEIGHT_RESTORE_MAX_STEP_M = 0.015
FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M = 0.003
FLAT_REGRASP_HEIGHT_RESTORE_ATTEMPTS = 4
FLAT_REGRASP_LEVEL_MAX_STEP_M = 0.004
FLAT_REGRASP_LEVEL_TARGET_AXIS_Z = 0.010
FLAT_REGRASP_LEVEL_ATTEMPTS = 4
SLOT_AXIS_ARM_MAX_STEP_M = 0.004
SLOT_AXIS_ARM_ATTEMPTS = 4
INSERT_AXIS_X_SAFETY_LIMIT = 0.0012
INSERT_AXIS_Z_SAFETY_LIMIT = 0.02
INSERT_AXIS_CORRECTION_MAX_STEP_M = 0.001
INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M = 0.008
EMPTY_HAND_SERVO_MAX_STEP_RAD = 0.012
ONE_HAND_SUPPORT_DROP_TOLERANCE_M = 0.025
IK_ROTATION_TOLERANCE_DEG = 5.0
SLOT_X_COMMAND_BIAS = -0.0003
RELEASE_GRIP_RATE = 0.022
BASE_ACCEL = 0.5
BASE_YAW_ACCEL = 0.2
CONTROL_FPS = 60.0
GRASP_X = 0.13
GRASP_Y_BIAS = 0.009
GRASP_MOUNT_Z = 1.135
GRIP_FORCE_MIN_N = 0.2
HOLD_CONTACT_RECOVERY_TICKS = 30
ARM_TRACK_TOL_RAD = 0.03
ARM_TRACK_STABLE_TICKS = 12
ARM_TRACK_MAX_TICKS = 900


class ExpertFailure(RuntimeError):
    pass


def angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def cosine_steps(distance, max_step, minimum=1):
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    return max(
        int(minimum), int(math.ceil(math.pi * float(distance) / (2.0 * max_step)))
    )


def bounded_vector(vector, max_norm):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if norm <= max_norm:
        return vector.copy()
    return vector * (max_norm / norm)


def anchor_feedback_mount_position(
    command_mount_position,
    target_anchor_position,
    actual_anchor_position,
    max_correction,
):
    return np.asarray(command_mount_position, dtype=float) + bounded_vector(
        np.asarray(target_anchor_position, dtype=float)
        - np.asarray(actual_anchor_position, dtype=float),
        max_correction,
    )


def symmetric_level_correction(
    roll_axis,
    left_anchor,
    right_anchor,
    max_step,
):
    roll_axis = np.asarray(roll_axis, dtype=float)
    left_anchor = np.asarray(left_anchor, dtype=float)
    right_anchor = np.asarray(right_anchor, dtype=float)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    axis_norm = float(np.linalg.norm(roll_axis))
    if axis_norm <= 1e-9:
        raise ValueError("roll_axis must be non-zero")
    roll_axis = roll_axis / axis_norm
    hand_axis = left_anchor - right_anchor
    if float(np.dot(roll_axis, hand_axis)) < 0.0:
        roll_axis = -roll_axis
    horizontal_axis_norm = float(np.linalg.norm(roll_axis[:2]))
    horizontal_separation = float(np.linalg.norm(hand_axis[:2]))
    if horizontal_axis_norm <= 1e-9 or horizontal_separation <= 1e-9:
        raise ValueError("grasp axis must have horizontal separation")
    correction = (
        -0.5
        * float(roll_axis[2])
        / horizontal_axis_norm
        * horizontal_separation
    )
    return float(np.clip(correction, -max_step, max_step))


def symmetric_axis_correction(
    roll_axis,
    left_anchor,
    right_anchor,
    target_axis,
    max_step,
):
    roll_axis = np.asarray(roll_axis, dtype=float)
    left_anchor = np.asarray(left_anchor, dtype=float)
    right_anchor = np.asarray(right_anchor, dtype=float)
    target_axis = np.asarray(target_axis, dtype=float)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    roll_norm = float(np.linalg.norm(roll_axis))
    target_norm = float(np.linalg.norm(target_axis))
    if roll_norm <= 1e-9 or target_norm <= 1e-9:
        raise ValueError("roll_axis and target_axis must be non-zero")
    roll_axis = roll_axis / roll_norm
    target_axis = target_axis / target_norm
    hand_axis = left_anchor - right_anchor
    if float(np.dot(roll_axis, hand_axis)) < 0.0:
        roll_axis = -roll_axis
    if float(np.dot(roll_axis, target_axis)) < 0.0:
        target_axis = -target_axis
    target_component = float(np.dot(roll_axis, target_axis))
    target_separation = abs(float(np.dot(hand_axis, target_axis)))
    if target_component <= 1e-6 or target_separation <= 1e-9:
        raise ValueError("grasp axis must have target-axis separation")
    perpendicular_axis = roll_axis - target_component * target_axis
    correction = (
        -0.5
        * target_separation
        * perpendicular_axis
        / target_component
    )
    return bounded_vector(correction, max_step)


def rotation_x(radians):
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ])


def rotation_z(radians):
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_axis_angle(axis, radians):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("rotation axis must be non-zero")
    axis = axis / norm
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    radians = float(radians)
    return (
        np.eye(3)
        + math.sin(radians) * skew
        + (1.0 - math.cos(radians)) * (skew @ skew)
    )


def grasp_target_rotation(base_rotation, direction):
    return (
        rotation_z(float(direction) * math.radians(GRASP_YAW_DEG))
        @ np.asarray(base_rotation, dtype=float)
    )


def flatten_target_rotation(initial_rotation, progress):
    return (
        rotation_x(
            -float(progress) * math.radians(FLAT_REGRASP_ANGLE_DEG)
        )
        @ np.asarray(initial_rotation, dtype=float)
    )


def coupled_regrasp_progress(progress):
    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    clearance_progress = 0.0
    if progress > FLAT_REGRASP_CLEARANCE_ONSET:
        clearance_progress = (
            progress - FLAT_REGRASP_CLEARANCE_ONSET
        ) / (1.0 - FLAT_REGRASP_CLEARANCE_ONSET)
    return progress**FLAT_REGRASP_ROTATION_EXPONENT, clearance_progress


def cartesian_waypoints(start, target, max_step):
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(target - start))
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    steps = max(1, int(math.ceil(distance / float(max_step))))
    return [
        start + (target - start) * (index / steps)
        for index in range(1, steps + 1)
    ]


def flat_regrasp_anchors(roll_position, roll_axis, current_anchor, direction):
    roll_position = np.asarray(roll_position, dtype=float)
    roll_axis = np.asarray(roll_axis, dtype=float)
    current_anchor = np.asarray(current_anchor, dtype=float)
    norm = float(np.linalg.norm(roll_axis))
    if norm <= 1e-9:
        raise ValueError("roll_axis must be non-zero")
    roll_axis = roll_axis / norm
    direction = math.copysign(1.0, float(direction))
    radial = (
        current_anchor
        - roll_position
        - np.dot(current_anchor - roll_position, roll_axis) * roll_axis
    )
    far_end = (
        roll_position
        + direction * FLAT_REGRASP_FAR_END_M * roll_axis
        + radial
    )
    axis_far = (
        roll_position
        + direction * FLAT_REGRASP_FAR_END_M * roll_axis
    )
    target = (
        roll_position
        + direction * FLAT_REGRASP_TARGET_ALONG_M * roll_axis
    )
    return {
        "far_end": far_end,
        "axis_far": axis_far,
        "target": target,
    }


def anchored_mount_position(
    mount_position,
    mount_rotation,
    anchor_position,
    target_rotation,
    target_anchor_position=None,
):
    mount_position = np.asarray(mount_position, dtype=float)
    mount_rotation = np.asarray(mount_rotation, dtype=float)
    anchor_position = np.asarray(anchor_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if target_anchor_position is None:
        target_anchor_position = anchor_position
    target_anchor_position = np.asarray(
        target_anchor_position, dtype=float
    )
    anchor_in_mount = mount_rotation.T @ (anchor_position - mount_position)
    return target_anchor_position - target_rotation @ anchor_in_mount


def roll_half_extent_x(axis_x):
    axis_x = abs(float(axis_x))
    return (
        ROLL_HALF_LENGTH * axis_x
        + ROLL_RADIUS * math.sqrt(max(0.0, 1.0 - axis_x * axis_x))
    )


def cylinder_slot_fit_margin(center_x, axis_x):
    half_x = roll_half_extent_x(axis_x)
    center_error = abs(float(center_x) - float(TARGET_CENTER[0]))
    return SLOT_HALF_WIDTH - half_x - center_error


def insertion_axis_is_safe(roll_axis):
    roll_axis = np.asarray(roll_axis, dtype=float)
    return bool(
        abs(float(roll_axis[0])) <= INSERT_AXIS_X_SAFETY_LIMIT
        and abs(float(roll_axis[2])) <= INSERT_AXIS_Z_SAFETY_LIMIT
    )


def insertion_axis_correction_has_clearance(
    roll_clearance,
    pad_clearance,
):
    return bool(
        min(float(roll_clearance), float(pad_clearance))
        >= INSERT_AXIS_CORRECTION_MIN_CLEARANCE_M
    )


def release_axis_slide_distance(step_index):
    step_index = int(step_index)
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if step_index < RELEASE_AXIS_COARSE_STEPS:
        return RELEASE_AXIS_COARSE_STEP_M
    return RELEASE_AXIS_FINE_STEP_M


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new episode output directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args(argv)
    if args.seed < 1:
        parser.error("--seed must be positive")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.width < 224 or args.height < 224:
        parser.error("record dimensions must both be at least 224")
    return args


def load_teleop(scene_path, gpu, seed):
    os.environ["TELEOP_SCENE_XML"] = str(scene_path)
    os.environ["TELEOP_VIEWER"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["TELEOP_RECORD_GPU"] = str(gpu)
    os.environ["CRUZR_GRIP_CLOSE"] = "0.025"
    os.environ["CRUZR_EP_SEED"] = str(seed)
    os.environ["REC_CAMS"] = ",".join(POLICY_CAMERAS)
    os.environ["REC_SAVE_RAW_TIMESTAMPS"] = "1"
    os.environ["REC_PROMPT"] = (
        "Pick up the roll from the table and place it stably in the top shelf slot"
    )
    spec = importlib.util.spec_from_file_location(
        "cruzr_teleop", CORE_DIR / "cruzr_teleop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_third_person(recorder, mujoco, model, data, camera, output_path):
    from PIL import Image

    recorder._ensure_gl()
    recorder._gl.make_current()
    mujoco.mjv_updateScene(
        model,
        data,
        recorder._opt,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL.value,
        recorder._scn,
    )
    mujoco.mjr_render(recorder._vp, recorder._scn, recorder._con)
    width, height = recorder._vp.width, recorder._vp.height
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, recorder._vp, recorder._con)
    Image.fromarray(np.flipud(rgb)).save(output_path, quality=90)


class SortingRollExpert:
    def __init__(self, args, ct, mujoco, scheduler, evaluate_placement, tracker_cls):
        self.args = args
        self.ct = ct
        self.mujoco = mujoco
        self.scheduler = scheduler
        self.evaluate_placement = evaluate_placement
        self.tracker_cls = tracker_cls
        self.model = ct.m
        self.data = ct.d
        self.out = Path(args.out).resolve()
        self.review_dir = self.out / "diagnostics" / "third_person"
        self.review_video = self.out / "sorting_roll_review.mp4"
        self.result_path = self.out / "result.json"
        self.roll_body = ct.bid("sorting_roll")
        self.roll_geom = ct.gid("sorting_roll_col")
        self.roll_joint = ct.jid("sorting_roll_free")
        self.roll_qpos_adr = int(self.model.jnt_qposadr[self.roll_joint])
        self.roll_dof_adr = int(self.model.jnt_dofadr[self.roll_joint])
        self.pad_ids = {
            ct.gid(name)
            for name in ("L_pad1", "L_pad2", "R_pad1", "R_pad2")
        }
        self.release_touch_geom_ids = {
            ct.gid("target_middle_bar_col")
        }
        self.shelf_geom_ids = {
            ct.gid(name)
            for name in (
                "shelf_post_front_left_col",
                "shelf_post_front_right_col",
                "shelf_post_rear_left_col",
                "shelf_post_rear_right_col",
                "target_slot_floor_col",
                "target_slot_front_guard_col",
                "target_slot_back_guard_col",
                "target_middle_bar_col",
                "target_top_bar_col",
            )
        }
        self.arm_geom_ids = {}
        for hand, root_name in (
            ("l", "L_shoulder_pitch_link"),
            ("r", "R_shoulder_pitch_link"),
        ):
            root_body = ct.bid(root_name)
            body_ids = {root_body}
            for body in range(self.model.nbody):
                parent = int(self.model.body_parentid[body])
                if parent in body_ids:
                    body_ids.add(body)
            self.arm_geom_ids[hand] = {
                geom
                for geom in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom]) in body_ids
            }
        self.recorded_roll_qpos = []
        self.recorded_roll_qvel = []
        self.gates = {}
        self.final_evidence = None
        self.sim_seconds = 0.0
        self.release_contact_monitor = None

        self.out.mkdir(parents=True)
        self.review_dir.mkdir(parents=True)
        ct.REC_WH = (args.width, args.height)
        self.recorder = ct.EpisodeRecorder(str(self.out))
        ct.REC.update({
            "rec": self.recorder,
            "on": True,
            "count": 0,
            "phase": "initial_hold",
            "metadata": {
                "task_version": TASK_VERSION,
                "seed": args.seed,
                "collection_profile": "sorting_roll_canary_v2",
                "training_eligible": False,
                "success_source": "sorting_roll_task.SortingRollSuccessTracker",
                "policy_cameras": list(POLICY_CAMERAS),
                "review_camera": "free_camera_not_policy_input",
                "release_pad_sliding_friction": (
                    RELEASE_PAD_SLIDING_FRICTION
                ),
            },
        })

        self.review_camera = mujoco.MjvCamera()
        self.review_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.review_camera.lookat[:] = [0.45, -0.72, 0.72]
        self.review_camera.distance = 3.35
        self.review_camera.azimuth = 42.0
        self.review_camera.elevation = -24.0

    def phase(self, name):
        self.ct.REC["phase"] = name
        print(f"[phase] {name}", flush=True)

    def tick(self):
        previous = self.recorder.n
        substeps = self.scheduler.next_substeps()
        self.ct.control_step(substeps)
        dt = float(substeps) * float(self.model.opt.timestep)
        self.sim_seconds += dt
        if self.release_contact_monitor is not None:
            pad_contact = self.contact_evidence(
                self.pad_ids, self.shelf_geom_ids
            )
            monitor = self.release_contact_monitor
            if pad_contact["force_n"] > monitor["maximum_pad_force"]:
                monitor["maximum_pad_force"] = pad_contact["force_n"]
                monitor["maximum_pad_pairs"] = pad_contact["pairs"]
            roll_x = float(self.roll_position()[0])
            monitor["minimum_roll_x"] = min(
                monitor["minimum_roll_x"], roll_x
            )
            monitor["maximum_roll_x"] = max(
                monitor["maximum_roll_x"], roll_x
            )
        if self.recorder.n != previous:
            self.recorded_roll_qpos.append(
                self.data.qpos[self.roll_qpos_adr:self.roll_qpos_adr + 7].copy()
            )
            self.recorded_roll_qvel.append(
                self.data.qvel[self.roll_dof_adr:self.roll_dof_adr + 6].copy()
            )
            render_third_person(
                self.recorder,
                self.mujoco,
                self.model,
                self.data,
                self.review_camera,
                self.review_dir / f"frame_{self.recorder.n - 1:06d}.jpg",
            )
        return dt

    def frames(self, count):
        for _ in range(int(count)):
            self.tick()

    def gate(self, name, passed, detail):
        passed = bool(passed)
        self.gates[name] = {"passed": passed, "detail": detail}
        print(f"[gate:{name}] {'PASS' if passed else 'FAIL'} {detail}", flush=True)
        if not passed:
            raise ExpertFailure(f"{name}: {detail}")

    def roll_position(self):
        return self.data.xpos[self.roll_body].copy()

    def arm_joint_positions(self):
        return {
            hand: np.array(
                [self.data.qpos[address] for address in arm.qadr],
                dtype=float,
            )
            for hand, arm in (("l", self.ct.L), ("r", self.ct.R))
        }

    def contact_evidence(self, first_ids, second_ids):
        first_ids = set(first_ids)
        second_ids = set(second_ids)
        force = np.zeros(6, dtype=float)
        total = 0.0
        pairs = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            pair = {geom1, geom2}
            if not pair & first_ids or not pair & second_ids:
                continue
            self.mujoco.mj_contactForce(
                self.model, self.data, index, force
            )
            total += abs(float(force[0]))
            pairs.append([
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom1
                ),
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom2
                ),
            ])
        return {"force_n": total, "pairs": pairs}

    def geom_label(self, geom):
        name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom)
        )
        if name:
            return name
        body = int(self.model.geom_bodyid[int(geom)])
        body_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body
        )
        return body_name or f"geom_{int(geom)}"

    def moving_arm_contacts(self, hand):
        moving = self.arm_geom_ids[hand]
        contacts = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if geom1 in moving and geom2 in moving:
                continue
            if geom1 not in moving and geom2 not in moving:
                continue
            contacts.append({
                "pair": [self.geom_label(geom1), self.geom_label(geom2)],
                "penetration_mm": round(
                    -1000.0 * min(0.0, float(self.data.contact[index].dist)),
                    3,
                ),
            })
        return contacts

    def pad_vertical_span(self):
        spans = {}
        for geom in self.pad_ids:
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            half_span = float(
                np.abs(rotation[2]) @ self.model.geom_size[geom, :3]
            )
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom
            )
            spans[name] = 2.0 * half_span
        return spans

    def grip_evidence(self, hand):
        pad_ids = {
            self.ct.gid(f"{hand}_pad1"),
            self.ct.gid(f"{hand}_pad2"),
        }
        force = np.zeros(6, dtype=float)
        total = 0.0
        contacts = set()
        for index in range(self.data.ncon):
            pair = {
                int(self.data.contact[index].geom1),
                int(self.data.contact[index].geom2),
            }
            if self.roll_geom in pair and pair & pad_ids:
                self.mujoco.mj_contactForce(
                    self.model, self.data, index, force
                )
                total += abs(float(force[0]))
                contacts.update(pair & pad_ids)
        return {
            "force_n": total,
            "pads": sorted(
                self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom
                )
                for geom in contacts
            ),
        }

    def set_base_velocity(self, forward, yaw_rate):
        dt = (1.0 / CONTROL_FPS) * self.ct.REC_DECIM
        self.ct.base_vel[0] += float(np.clip(
            forward - self.ct.base_vel[0],
            -BASE_ACCEL * dt,
            BASE_ACCEL * dt,
        ))
        self.ct.base_vel[1] += float(np.clip(
            yaw_rate - self.ct.base_vel[1],
            -BASE_YAW_ACCEL * dt,
            BASE_YAW_ACCEL * dt,
        ))

    def stop_base(self):
        for _ in range(90):
            if max(abs(float(v)) for v in self.ct.base_vel) < 1e-3:
                break
            self.set_base_velocity(0.0, 0.0)
            self.frames(self.ct.REC_DECIM)
        self.ct.base_vel[:] = 0.0
        self.frames(12)

    @staticmethod
    def brake_cap(remaining, acceleration):
        return math.sqrt(max(0.0, 2.0 * acceleration * abs(float(remaining))))

    def turn_in_place(self, target_yaw, max_rate=0.22, tolerance=0.012):
        self.stop_base()
        for _ in range(2400):
            error = angle(target_yaw - self.ct.base_pose()[2])
            if abs(error) <= tolerance and abs(self.ct.base_vel[1]) <= 0.015:
                break
            rate_cap = min(
                max_rate,
                self.brake_cap(error, BASE_YAW_ACCEL),
            )
            self.set_base_velocity(
                0.0,
                float(np.clip(1.8 * error, -rate_cap, rate_cap)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure(f"turn timeout target={target_yaw:.3f}")
        self.stop_base()

    def go_to(self, target_xy, target_yaw, max_speed=0.16, tolerance=0.008):
        target_xy = np.asarray(target_xy, dtype=float)
        pose = self.ct.base_pose()
        if float(np.linalg.norm(target_xy - pose[:2])) > tolerance:
            heading = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
            self.turn_in_place(heading)
        for _ in range(5000):
            x, y, yaw = self.ct.base_pose()
            delta = target_xy - np.array([x, y])
            distance = float(np.linalg.norm(delta))
            if distance <= tolerance:
                break
            heading = math.atan2(delta[1], delta[0])
            heading_error = angle(heading - yaw)
            if abs(heading_error) > 0.25:
                self.turn_in_place(heading)
                continue
            speed_cap = min(
                max_speed,
                self.brake_cap(distance - tolerance, BASE_ACCEL),
            )
            speed = min(max(0.025, distance), speed_cap)
            yaw_cap = min(
                0.18,
                self.brake_cap(heading_error, BASE_YAW_ACCEL),
            )
            self.set_base_velocity(
                speed,
                float(np.clip(1.5 * heading_error, -yaw_cap, yaw_cap)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure(f"navigation timeout target={target_xy.tolist()}")
        self.stop_base()
        self.turn_in_place(target_yaw)

    def reverse(self, distance, max_speed=0.08):
        start = self.ct.base_pose()
        yaw = float(start[2])
        target = start[:2] - float(distance) * np.array(
            [math.cos(yaw), math.sin(yaw)]
        )
        for _ in range(2500):
            x, y, measured_yaw = self.ct.base_pose()
            remaining = float(np.linalg.norm(target - np.array([x, y])))
            if remaining <= 0.008:
                break
            reverse_heading = angle(
                math.atan2(target[1] - y, target[0] - x) + math.pi
            )
            yaw_error = angle(reverse_heading - measured_yaw)
            speed_cap = min(
                max_speed,
                self.brake_cap(remaining - 0.008, BASE_ACCEL),
            )
            self.set_base_velocity(
                -min(max(0.02, remaining), speed_cap),
                float(np.clip(1.2 * yaw_error, -0.10, 0.10)),
            )
            self.frames(self.ct.REC_DECIM)
        else:
            raise ExpertFailure("reverse clearance timeout")
        self.stop_base()

    def solve_mounts(self, positions, rotations, iterations=240, base_pose=None):
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        targets = {}
        residuals = {}
        rotation_residuals = {}
        if base_pose is not None:
            for address, value in zip(self.ct.BQ, base_pose):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
        for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
            for address, value in zip(arm.qadr, self.ct.qtgt[hand]):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
            self.ct.ik(
                arm,
                np.asarray(positions[hand], dtype=float),
                np.asarray(rotations[hand], dtype=float),
                iters=iterations,
                w=0.6,
            )
            targets[hand] = np.array(
                [self.data.qpos[address] for address in arm.qadr],
                dtype=float,
            )
            residuals[hand] = float(np.linalg.norm(
                self.data.xpos[arm.mount] - positions[hand]
            ))
            rotation_residuals[hand] = math.degrees(float(np.linalg.norm(
                self.ct.rot_err(
                    rotations[hand],
                    self.data.xmat[arm.mount].reshape(3, 3),
                )
            )))
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return targets, residuals, rotation_residuals

    def servo_arms(self, targets, max_step=0.006, minimum=30):
        distance = max(
            float(np.max(np.abs(targets[hand] - self.ct.qtgt[hand])))
            for hand in ("l", "r")
        )
        steps = cosine_steps(distance, max_step, minimum=minimum)
        starts = {hand: self.ct.qtgt[hand].copy() for hand in ("l", "r")}
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(math.pi * (index + 1) / steps)
            for hand in ("l", "r"):
                self.ct.qtgt[hand][:] = (
                    starts[hand] + (targets[hand] - starts[hand]) * blend
                )
            self.ct.base_vel[:] = 0.0
            self.frames(1)
        self.frames(20)

    def wait_arm_tracking(
        self,
        label,
        tolerance=ARM_TRACK_TOL_RAD,
        collision_free_hand=None,
    ):
        stable_ticks = 0
        error = float("inf")
        for tick in range(1, ARM_TRACK_MAX_TICKS + 1):
            if collision_free_hand is not None:
                contacts = self.moving_arm_contacts(collision_free_hand)
                if contacts:
                    raise ExpertFailure(
                        f"moving arm collision phase={label} "
                        f"hand={collision_free_hand} contacts={contacts}"
                    )
            error = max(
                abs(float(self.data.qpos[address]) - float(target))
                for hand, arm in (("l", self.ct.L), ("r", self.ct.R))
                for address, target in zip(arm.qadr, self.ct.qtgt[hand])
            )
            stable_ticks = stable_ticks + 1 if error <= tolerance else 0
            if stable_ticks >= ARM_TRACK_STABLE_TICKS:
                print(
                    f"[track:{label}] PASS error={error:.4f}rad ticks={tick}",
                    flush=True,
                )
                return
            self.frames(1)
        raise ExpertFailure(
            f"arm tracking timeout phase={label} error={error:.4f}rad"
        )

    def move_mounts(
        self,
        positions,
        rotations,
        iterations=240,
        solve_base_pose=None,
        tracking_tolerance=ARM_TRACK_TOL_RAD,
    ):
        targets, residuals, rotation_residuals = self.solve_mounts(
            positions,
            rotations,
            iterations=iterations,
            base_pose=solve_base_pose,
        )
        self.gate(
            "ik_reachable",
            max(residuals.values()) <= 0.012
            and max(rotation_residuals.values())
            <= IK_ROTATION_TOLERANCE_DEG,
            "residual_mm="
            + ",".join(
                f"{hand}:{1000.0 * residuals[hand]:.1f}"
                for hand in ("l", "r")
            )
            + " rotation_deg="
            + ",".join(
                f"{hand}:{rotation_residuals[hand]:.2f}"
                for hand in ("l", "r")
            ),
        )
        self.servo_arms(targets)
        self.wait_arm_tracking(
            self.ct.REC["phase"],
            tolerance=tracking_tolerance,
        )

    def solve_one_mount_target(
        self, hand, seed, position, rotation, iterations=400
    ):
        arms = {"l": self.ct.L, "r": self.ct.R}
        arm = arms[hand]
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        for address, value in zip(arm.qadr, seed):
            self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        self.ct.ik(
            arm,
            np.asarray(position, dtype=float),
            np.asarray(rotation, dtype=float),
            iters=iterations,
            w=0.6,
        )
        target = np.array(
            [self.data.qpos[address] for address in arm.qadr],
            dtype=float,
        )
        residual = float(np.linalg.norm(
            self.data.xpos[arm.mount] - position
        ))
        rotation_residual = math.degrees(float(np.linalg.norm(
            self.ct.rot_err(
                rotation,
                self.data.xmat[arm.mount].reshape(3, 3),
            )
        )))
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return target, residual, rotation_residual

    def validate_one_arm_segment(self, hand, start, target, label):
        arm = self.ct.L if hand == "l" else self.ct.R
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        distance = float(np.max(np.abs(target - start)))
        steps = cosine_steps(
            distance,
            FLAT_REGRASP_COLLISION_STEP_RAD,
            minimum=2,
        )
        collision = None
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * (index + 1) / steps
            )
            configuration = start + (target - start) * blend
            for address, value in zip(arm.qadr, configuration):
                self.data.qpos[address] = value
            self.mujoco.mj_forward(self.model, self.data)
            contacts = self.moving_arm_contacts(hand)
            if contacts:
                collision = {
                    "sample": index + 1,
                    "samples": steps,
                    "contacts": contacts,
                }
                break
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        if collision is not None:
            raise ExpertFailure(
                f"collision-free path failed phase={label} "
                f"hand={hand} evidence={collision}"
            )

    def servo_one_arm_target(self, hand, target, label):
        start = self.ct.qtgt[hand].copy()
        distance = float(np.max(np.abs(target - start)))
        steps = cosine_steps(
            distance,
            EMPTY_HAND_SERVO_MAX_STEP_RAD,
            minimum=1,
        )
        for index in range(steps):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * (index + 1) / steps
            )
            self.ct.qtgt[hand][:] = start + (target - start) * blend
            self.ct.base_vel[:] = 0.0
            self.frames(1)
            contacts = self.moving_arm_contacts(hand)
            if contacts:
                raise ExpertFailure(
                    f"moving arm collision phase={label} "
                    f"hand={hand} contacts={contacts}"
                )

    def follow_empty_hand_stage(self, hand, stage, targets):
        max_position_residual = 0.0
        max_rotation_residual = 0.0
        waypoint_count = 0
        for position, rotation in targets:
            waypoint_count += 1
            seed = self.ct.qtgt[hand].copy()
            target, residual, rotation_residual = (
                self.solve_one_mount_target(
                    hand,
                    seed,
                    position,
                    rotation,
                )
            )
            if (
                residual > 0.012
                or rotation_residual > IK_ROTATION_TOLERANCE_DEG
            ):
                self.gate(
                    f"flat_regrasp_path_{hand}",
                    False,
                    f"stage={stage} residual_mm={1000.0 * residual:.1f} "
                    f"rotation_deg={rotation_residual:.2f}",
                )
            self.validate_one_arm_segment(
                hand,
                seed,
                target,
                f"{hand}_{stage}",
            )
            self.servo_one_arm_target(
                hand,
                target,
                f"{hand}_{stage}",
            )
            max_position_residual = max(max_position_residual, residual)
            max_rotation_residual = max(
                max_rotation_residual, rotation_residual
            )
        self.wait_arm_tracking(
            f"{hand}_{stage}",
            collision_free_hand=hand,
        )
        contacts = self.moving_arm_contacts(hand)
        self.gate(
            f"collision_free_{hand}_{stage}",
            not contacts,
            f"waypoints={waypoint_count} "
            f"max_residual_mm={1000.0 * max_position_residual:.2f} "
            f"max_rotation_deg={max_rotation_residual:.3f} "
            f"contacts={contacts}",
        )

    def move_mounts_delta(self, delta):
        delta = np.asarray(delta, dtype=float)
        steps = max(1, int(math.ceil(float(np.linalg.norm(delta)) / 0.02)))
        for _ in range(steps):
            step = delta / steps
            positions = {
                "l": self.data.xpos[self.ct.L.mount].copy() + step,
                "r": self.data.xpos[self.ct.R.mount].copy() + step,
            }
            rotations = {
                "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
                "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
            }
            self.move_mounts(positions, rotations, iterations=300)

    def commanded_mount_poses(self):
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
            for address, value in zip(arm.qadr, self.ct.qtgt[hand]):
                self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        positions = {
            "l": self.data.xpos[self.ct.L.mount].copy(),
            "r": self.data.xpos[self.ct.R.mount].copy(),
        }
        rotations = {
            "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
            "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
        }
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return positions, rotations

    def move_mount_commands_delta(self, delta):
        delta = np.asarray(delta, dtype=float)
        self.move_mount_command_deltas({"l": delta, "r": delta})

    def move_mount_command_deltas(self, deltas):
        positions, rotations = self.commanded_mount_poses()
        for hand in ("l", "r"):
            positions[hand] += np.asarray(deltas[hand], dtype=float)
        self.move_mounts(positions, rotations, iterations=300)

    def require_held(self, stage, minimum_force=GRIP_FORCE_MIN_N):
        recovery_ticks = 0
        while True:
            left = self.grip_evidence("L")
            right = self.grip_evidence("R")
            position = self.roll_position()
            held = (
                left["force_n"] >= minimum_force
                and right["force_n"] >= minimum_force
                and len(left["pads"]) == 2
                and len(right["pads"]) == 2
                and position[2] >= 1.055
            )
            if held or recovery_ticks >= HOLD_CONTACT_RECOVERY_TICKS:
                break
            self.frames(1)
            recovery_ticks += 1
        self.gate(
            f"held_{stage}",
            held,
            f"position={np.round(position, 4).tolist()} "
            f"left={left['force_n']:.2f}N/{left['pads']} "
            f"right={right['force_n']:.2f}N/{right['pads']} "
            f"recovery_ticks={recovery_ticks}",
        )

    def require_hand_held(self, hand, stage, minimum_force=GRIP_FORCE_MIN_N):
        recovery_ticks = 0
        while True:
            evidence = self.grip_evidence(hand.upper())
            held = (
                evidence["force_n"] >= minimum_force
                and len(evidence["pads"]) == 2
                and self.roll_position()[2] >= 1.055
            )
            if held or recovery_ticks >= HOLD_CONTACT_RECOVERY_TICKS:
                break
            self.frames(1)
            recovery_ticks += 1
        self.gate(
            f"held_{hand}_{stage}",
            held,
            f"position={np.round(self.roll_position(), 4).tolist()} "
            f"force={evidence['force_n']:.2f}N/{evidence['pads']} "
            f"recovery_ticks={recovery_ticks}",
        )

    def require_hand_released(self, hand, stage):
        evidence = self.grip_evidence(hand.upper())
        self.gate(
            f"released_{hand}_{stage}",
            evidence["force_n"] <= 0.05 and not evidence["pads"],
            f"force={evidence['force_n']:.3f}N/{evidence['pads']}",
        )

    def open_hand_until_released(
        self, hand, stage, max_ticks=120, stable_ticks=6
    ):
        arm = self.ct.L if hand == "l" else self.ct.R
        stable = 0
        evidence = None
        open_fraction = 0.0
        for tick in range(1, int(max_ticks) + 1):
            self.frames(1)
            raw = float(np.mean([
                self.data.qpos[address] for address in arm.grip_qadr
            ]))
            span = self.ct.GRIP_OPEN - self.ct.GRIP_CLOSE
            open_fraction = float(np.clip(
                (raw - self.ct.GRIP_CLOSE) / span,
                0.0,
                1.0,
            ))
            evidence = self.grip_evidence(hand.upper())
            released = (
                open_fraction >= 0.95
                and evidence["force_n"] <= 0.05
                and not evidence["pads"]
            )
            stable = stable + 1 if released else 0
            if stable >= int(stable_ticks):
                self.gate(
                    f"released_{hand}_{stage}",
                    True,
                    f"open_fraction={open_fraction:.3f} "
                    f"force={evidence['force_n']:.3f}N/{evidence['pads']} "
                    f"ticks={tick} stable_ticks={stable}",
                )
                return
        self.gate(
            f"released_{hand}_{stage}",
            False,
            f"open_fraction={open_fraction:.3f} "
            f"force={evidence['force_n']:.3f}N/{evidence['pads']} "
            f"ticks={max_ticks} stable_ticks={stable}",
        )

    def flatten_hands(self):
        starting_height = float(self.roll_position()[2])
        arms = {"l": self.ct.L, "r": self.ct.R}
        for hand, support, direction in FLAT_REGRASP_ORDER:
            self.phase(f"release_{hand}_for_flat_regrasp")
            self.require_hand_held(support, f"supporting_{hand}_release")
            self.ct.grip_cmd[hand] = self.ct.GRIP_OPEN
            self.open_hand_until_released(hand, "before_flattening")
            self.require_hand_held(support, f"supporting_{hand}_flattening")
            arm = arms[hand]
            measured = self.arm_joint_positions()[hand]
            self.ct.qtgt[hand][:] = measured
            self.mujoco.mj_forward(self.model, self.data)

            def live_axis(reference=None):
                axis = (
                    self.data.xmat[self.roll_body]
                    .reshape(3, 3)[:, 0]
                    .copy()
                )
                if reference is None:
                    if axis[0] < 0.0:
                        axis = -axis
                elif float(np.dot(axis, reference)) < 0.0:
                    axis = -axis
                return axis / float(np.linalg.norm(axis))

            def verify_empty_stage(stage):
                self.require_hand_released(hand, f"after_{stage}")
                self.require_hand_held(
                    support,
                    f"supporting_{hand}_{stage}",
                )
                self.gate(
                    f"one_hand_support_height_{hand}_{stage}",
                    self.roll_position()[2]
                    >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                    f"starting_z={starting_height:.4f} "
                    f"actual_z={self.roll_position()[2]:.4f}",
                )

            slide_anchor = arm.padmid().copy()
            slide_mount = self.data.xpos[arm.mount].copy()
            slide_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            slide_roll = self.roll_position()
            slide_axis = live_axis()
            slide_along = float(np.dot(
                slide_anchor - slide_roll,
                slide_axis,
            ))
            slide_target_along = (
                float(direction) * FLAT_REGRASP_COUPLED_START_M
            )
            slide_radial = (
                slide_anchor
                - slide_roll
                - slide_along * slide_axis
            )
            slide_steps = max(1, int(math.ceil(
                abs(slide_target_along - slide_along)
                / FLAT_REGRASP_CART_STEP_M
            )))
            slide_alongs = np.linspace(
                slide_along,
                slide_target_along,
                slide_steps + 1,
            )[1:]

            def adaptive_slide_targets():
                for along in slide_alongs:
                    axis = live_axis(slide_axis)
                    radial = (
                        slide_radial
                        - np.dot(slide_radial, axis) * axis
                    )
                    anchor = self.roll_position() + along * axis + radial
                    yield (
                        anchored_mount_position(
                            slide_mount,
                            slide_rotation,
                            slide_anchor,
                            slide_rotation,
                            target_anchor_position=anchor,
                        ),
                        slide_rotation,
                    )

            self.phase(f"flatten_{hand}_slide_out")
            self.follow_empty_hand_stage(
                hand,
                "slide_out",
                adaptive_slide_targets(),
            )
            verify_empty_stage("slide_out")

            coupled_anchor = arm.padmid().copy()
            coupled_mount = self.data.xpos[arm.mount].copy()
            coupled_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            reference_axis = live_axis(slide_axis)
            coupled_roll = self.roll_position()
            coupled_along = float(np.dot(
                coupled_anchor - coupled_roll,
                reference_axis,
            ))
            coupled_target_along = (
                float(direction) * FLAT_REGRASP_NEAR_END_M
            )
            coupled_radial = (
                coupled_anchor
                - coupled_roll
                - coupled_along * reference_axis
            )
            coupled_steps = max(
                FLAT_REGRASP_COUPLED_MIN_STEPS,
                int(math.ceil(
                    abs(coupled_target_along - coupled_along)
                    / FLAT_REGRASP_CART_STEP_M
                )),
            )
            coupled_alongs = np.linspace(
                coupled_along,
                coupled_target_along,
                coupled_steps + 1,
            )[1:]

            def adaptive_coupled_targets():
                for index, along in enumerate(coupled_alongs, start=1):
                    progress = index / coupled_steps
                    rotation_progress, clearance_progress = (
                        coupled_regrasp_progress(progress)
                    )
                    axis = live_axis(reference_axis)
                    radial = (
                        coupled_radial
                        - np.dot(coupled_radial, axis) * axis
                    )
                    anchor = (
                        self.roll_position()
                        + along * axis
                        + radial
                        + clearance_progress * FLAT_REGRASP_CLEARANCE
                    )
                    rotation = flatten_target_rotation(
                        coupled_rotation,
                        rotation_progress,
                    )
                    yield (
                        anchored_mount_position(
                            coupled_mount,
                            coupled_rotation,
                            coupled_anchor,
                            rotation,
                            target_anchor_position=anchor,
                        ),
                        rotation,
                    )

            self.phase(f"flatten_{hand}_coupled_exit_and_flatten")
            self.follow_empty_hand_stage(
                hand,
                "coupled_exit_and_flatten",
                adaptive_coupled_targets(),
            )
            verify_empty_stage("coupled_exit_and_flatten")

            current_anchor = arm.padmid().copy()
            mount_position = self.data.xpos[arm.mount].copy()
            initial_rotation = (
                self.data.xmat[arm.mount].reshape(3, 3).copy()
            )
            roll_position = self.roll_position()
            reference_axis = live_axis(reference_axis)
            anchors = flat_regrasp_anchors(
                roll_position,
                reference_axis,
                current_anchor,
                direction,
            )
            target_rotation = flatten_target_rotation(
                grasp_target_rotation(self.ct.R_DES, direction),
                1.0,
            )

            def mount_targets(anchor_targets, rotations):
                return [
                    (
                        anchored_mount_position(
                            mount_position,
                            initial_rotation,
                            current_anchor,
                            rotation,
                            target_anchor_position=anchor,
                        ),
                        rotation,
                    )
                    for anchor, rotation in zip(anchor_targets, rotations)
                ]

            stages = []
            rotation_vector = self.ct.rot_err(
                target_rotation,
                initial_rotation,
            )
            rotation_angle = float(np.linalg.norm(rotation_vector))
            if rotation_angle > 1e-9:
                rotation_axis = rotation_vector / rotation_angle
                rotation_steps = max(1, int(math.ceil(
                    math.degrees(rotation_angle)
                    / FLAT_REGRASP_ABSOLUTE_ROTATION_STEP_DEG
                )))
                rotations = [
                    rotation_axis_angle(
                        rotation_axis,
                        rotation_angle * index / rotation_steps,
                    )
                    @ initial_rotation
                    for index in range(1, rotation_steps + 1)
                ]
                stages.append((
                    "finish_flat_rotation",
                    mount_targets(
                        [current_anchor] * len(rotations),
                        rotations,
                    ),
                ))
            for name, start, target in (
                (
                    "extend_flat",
                    current_anchor,
                    anchors["far_end"],
                ),
                (
                    "align_far",
                    anchors["far_end"],
                    anchors["axis_far"],
                ),
            ):
                anchor_targets = cartesian_waypoints(
                    start,
                    target,
                    FLAT_REGRASP_CART_STEP_M,
                )
                stages.append((
                    name,
                    mount_targets(
                        anchor_targets,
                        [target_rotation] * len(anchor_targets),
                    ),
                ))

            for stage, targets in stages:
                self.phase(f"flatten_{hand}_{stage}")
                self.follow_empty_hand_stage(hand, stage, targets)
                verify_empty_stage(stage)

            insert_start = float(direction) * FLAT_REGRASP_FAR_END_M
            insert_target = (
                float(direction) * FLAT_REGRASP_TARGET_ALONG_M
            )
            insert_steps = max(1, int(math.ceil(
                abs(insert_target - insert_start)
                / FLAT_REGRASP_CART_STEP_M
            )))
            insert_alongs = np.linspace(
                insert_start,
                insert_target,
                insert_steps + 1,
            )

            def adaptive_insert_targets():
                for along in insert_alongs:
                    axis = live_axis(reference_axis)
                    anchor = self.roll_position() + along * axis
                    yield (
                        anchored_mount_position(
                            mount_position,
                            initial_rotation,
                            current_anchor,
                            target_rotation,
                            target_anchor_position=anchor,
                        ),
                        target_rotation,
                    )

            self.phase(f"flatten_{hand}_insert_from_end")
            self.follow_empty_hand_stage(
                hand,
                "insert_from_end",
                adaptive_insert_targets(),
            )
            verify_empty_stage("insert_from_end")

            for attempt in range(
                1, FLAT_REGRASP_ANCHOR_CORRECTION_ATTEMPTS + 1
            ):
                axis = live_axis(reference_axis)
                target_anchor = (
                    self.roll_position()
                    + float(direction)
                    * FLAT_REGRASP_TARGET_ALONG_M
                    * axis
                )
                actual_anchor = arm.padmid().copy()
                anchor_error = target_anchor - actual_anchor
                if (
                    float(np.linalg.norm(anchor_error))
                    <= FLAT_REGRASP_ANCHOR_CORRECTION_TARGET_M
                ):
                    break

                command_positions, command_rotations = (
                    self.commanded_mount_poses()
                )
                command_mount = command_positions[hand]
                command_rotation = command_rotations[hand]

                correction_position = anchor_feedback_mount_position(
                    command_mount,
                    target_anchor,
                    actual_anchor,
                    FLAT_REGRASP_ANCHOR_CORRECTION_MAX_M,
                )
                correction_stage = f"anchor_correction_{attempt}"
                self.phase(f"flatten_{hand}_{correction_stage}")
                self.follow_empty_hand_stage(
                    hand,
                    correction_stage,
                    [(correction_position, command_rotation)],
                )
                verify_empty_stage(correction_stage)

            axis = (
                self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
            )
            if float(np.dot(axis, reference_axis)) < 0.0:
                axis = -axis
            target_anchor = (
                self.roll_position()
                + float(direction) * FLAT_REGRASP_TARGET_ALONG_M * axis
            )
            anchor_error = target_anchor - arm.padmid()
            self.gate(
                f"flat_regrasp_anchor_{hand}",
                float(np.linalg.norm(anchor_error))
                <= FLAT_REGRASP_ANCHOR_GATE_TOLERANCE_M,
                f"error_mm={np.round(1000.0 * anchor_error, 1).tolist()} "
                f"roll={np.round(self.roll_position(), 4).tolist()}",
            )
            self.gate(
                f"one_hand_support_height_before_{hand}_regrasp",
                self.roll_position()[2]
                >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                f"starting_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

            self.phase(f"regrasp_{hand}_flat")
            self.ct.grip_cmd[hand] = self.ct.GRIP_CLOSE
            self.frames(120)
            self.require_held(f"{hand}_flat_regrasp")
            self.gate(
                f"one_hand_support_height_{hand}",
                self.roll_position()[2]
                >= starting_height - ONE_HAND_SUPPORT_DROP_TOLERANCE_M,
                f"starting_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

            self.phase(f"restore_height_after_{hand}_flat_regrasp")
            for _ in range(FLAT_REGRASP_HEIGHT_RESTORE_ATTEMPTS):
                height_error = starting_height - float(
                    self.roll_position()[2]
                )
                if (
                    height_error
                    <= FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M
                ):
                    break
                self.move_mount_commands_delta([
                    0.0,
                    0.0,
                    min(
                        FLAT_REGRASP_HEIGHT_RESTORE_MAX_STEP_M,
                        height_error + 0.002,
                    ),
                ])
                self.require_held(f"restoring_height_after_{hand}")
            self.gate(
                f"flat_regrasp_height_restored_{hand}",
                self.roll_position()[2]
                >= starting_height
                - FLAT_REGRASP_HEIGHT_RESTORE_TOLERANCE_M,
                f"target_z={starting_height:.4f} "
                f"actual_z={self.roll_position()[2]:.4f}",
            )

        self.phase("level_roll_after_flat_regrasp")
        level_axis = None
        for _ in range(FLAT_REGRASP_LEVEL_ATTEMPTS):
            level_axis = (
                self.data.xmat[self.roll_body]
                .reshape(3, 3)[:, 0]
                .copy()
            )
            left_anchor = self.ct.L.padmid().copy()
            right_anchor = self.ct.R.padmid().copy()
            if float(np.dot(
                level_axis,
                left_anchor - right_anchor,
            )) < 0.0:
                level_axis = -level_axis
            if (
                abs(float(level_axis[2]))
                <= FLAT_REGRASP_LEVEL_TARGET_AXIS_Z
            ):
                break
            left_delta_z = symmetric_level_correction(
                level_axis,
                left_anchor,
                right_anchor,
                FLAT_REGRASP_LEVEL_MAX_STEP_M,
            )
            self.move_mount_command_deltas({
                "l": np.array([0.0, 0.0, left_delta_z]),
                "r": np.array([0.0, 0.0, -left_delta_z]),
            })
            self.require_held("leveling_flat_roll")
        level_axis = (
            self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        )
        self.gate(
            "flat_roll_levelled",
            abs(float(level_axis[2]))
            <= FLAT_REGRASP_LEVEL_TARGET_AXIS_Z,
            f"axis={np.round(level_axis, 6).tolist()}",
        )

        spans = self.pad_vertical_span()
        axis = self.roll_axis()
        self.gate(
            "hands_flat",
            max(spans.values()) <= 0.025
            and abs(float(axis[2])) <= 0.02
            and self.roll_position()[2] >= HAND_FLAT_ROLL_Z - 0.015,
            f"pad_vertical_span_mm="
            f"{json.dumps({name: round(1000.0 * value, 1) for name, value in spans.items()})} "
            f"roll_axis={np.round(axis, 6).tolist()} "
            f"roll={np.round(self.roll_position(), 4).tolist()}",
        )

    def align_roll_center(
        self,
        target,
        gate_name,
        tolerance,
        *,
        command_bias=None,
        command_space=False,
        max_step=0.02,
        attempts=12,
    ):
        target = np.asarray(target, dtype=float)
        tolerance = np.asarray(tolerance, dtype=float)
        command_target = target.copy()
        if command_bias is not None:
            command_target += np.asarray(command_bias, dtype=float)
        for _ in range(attempts):
            error = target - self.roll_position()
            if np.all(np.abs(error) <= tolerance):
                break
            command_error = command_target - self.roll_position()
            move_delta = (
                self.move_mount_commands_delta
                if command_space
                else self.move_mounts_delta
            )
            move_delta(bounded_vector(command_error, max_step))
        error = target - self.roll_position()
        self.gate(
            gate_name,
            np.all(np.abs(error) <= tolerance),
            f"target={np.round(target, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * error, 1).tolist()}",
        )

    def roll_axis(self):
        axis = self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        return -axis if axis[1] < 0.0 else axis

    def slot_fit_margin(self):
        return cylinder_slot_fit_margin(
            self.roll_position()[0], self.roll_axis()[0]
        )

    def insertion_axis_correction_clearances(self):
        obstacle_min_x = min(
            float(self.ct.geom_aabb(geom)[0][0])
            for geom in self.release_touch_geom_ids
        )
        roll_max_x = float(
            self.roll_position()[0]
            + roll_half_extent_x(self.roll_axis()[0])
        )
        pad_max_x = max(
            float(self.ct.geom_aabb(geom)[1][0])
            for geom in self.pad_ids
        )
        return obstacle_min_x - roll_max_x, obstacle_min_x - pad_max_x

    def align_roll_axis(self, held_stage, gate_name):
        for _ in range(5):
            axis = self.roll_axis()
            if abs(float(axis[0])) <= 0.0008:
                break
            yaw_correction = math.atan2(float(axis[0]), float(axis[1]))
            target_yaw = angle(self.ct.base_pose()[2] + yaw_correction)
            self.turn_in_place(
                target_yaw, max_rate=0.08, tolerance=0.0002
            )
            self.require_held(held_stage)
        axis = self.roll_axis()
        self.gate(
            gate_name,
            abs(float(axis[0])) <= 0.0008,
            f"axis={np.round(axis, 6).tolist()}",
        )

    def align_roll_axis_with_arms(
        self,
        held_stage,
        gate_name,
        max_step=SLOT_AXIS_ARM_MAX_STEP_M,
    ):
        for _ in range(SLOT_AXIS_ARM_ATTEMPTS):
            axis = self.roll_axis()
            if (
                abs(float(axis[0])) <= 0.0008
                and abs(float(axis[2])) <= 0.02
            ):
                break
            left_delta = symmetric_axis_correction(
                axis,
                self.ct.L.padmid(),
                self.ct.R.padmid(),
                TARGET_AXIS,
                max_step,
            )
            self.move_mount_command_deltas({
                "l": left_delta,
                "r": -left_delta,
            })
            self.require_held(held_stage)
        axis = self.roll_axis()
        self.gate(
            gate_name,
            abs(float(axis[0])) <= 0.0008
            and abs(float(axis[2])) <= 0.02,
            f"axis={np.round(axis, 6).tolist()}",
        )

    def slowly_insert_roll(
        self,
        target,
        max_step=0.004,
    ):
        target = np.asarray(target, dtype=float)
        if max_step <= 0.0:
            raise ValueError("max_step must be positive")
        tolerance = np.array([
            0.0006,
            PRE_RELEASE_Y_TOLERANCE_M,
            0.002,
        ])
        command_target = target + np.array([
            SLOT_X_COMMAND_BIAS,
            0.0,
            0.0,
        ])
        steps_taken = 0
        max_steps = max(
            24,
            int(math.ceil(
                float(np.linalg.norm(
                    command_target - self.roll_position()
                )) / max_step
            )) + 4,
        )
        for _ in range(max_steps):
            error = target - self.roll_position()
            if np.all(np.abs(error) <= tolerance):
                break
            self.move_mount_commands_delta(
                bounded_vector(
                    command_target - self.roll_position(),
                    max_step,
                )
            )
            steps_taken += 1
            self.require_held("slow_insert_step")
            pad_contact = self.contact_evidence(
                self.pad_ids, self.shelf_geom_ids
            )
            self.gate(
                "grippers_clear_during_insert",
                pad_contact["force_n"] <= 0.2,
                f"force_n={pad_contact['force_n']:.4f} "
                f"pairs={pad_contact['pairs']}",
            )
            axis = self.roll_axis()
            if not insertion_axis_is_safe(axis):
                roll_clearance, pad_clearance = (
                    self.insertion_axis_correction_clearances()
                )
                self.gate(
                    "slot_axis_correction_clearance",
                    insertion_axis_correction_has_clearance(
                        roll_clearance, pad_clearance
                    ),
                    f"roll_clearance_mm={1000.0 * roll_clearance:.2f} "
                    f"pad_clearance_mm={1000.0 * pad_clearance:.2f}",
                )
                self.align_roll_axis_with_arms(
                    "correcting_slot_axis_during_insert",
                    "slot_axis_during_insert",
                    max_step=INSERT_AXIS_CORRECTION_MAX_STEP_M,
                )
                pad_contact = self.contact_evidence(
                    self.pad_ids, self.shelf_geom_ids
                )
                self.gate(
                    "grippers_clear_after_insert_axis_correction",
                    pad_contact["force_n"] <= 0.2,
                    f"force_n={pad_contact['force_n']:.4f} "
                    f"pairs={pad_contact['pairs']}",
                )
                axis = self.roll_axis()
            self.gate(
                "slot_axis_safe_during_insert",
                insertion_axis_is_safe(axis),
                f"axis={np.round(axis, 6).tolist()}",
            )
        error = target - self.roll_position()
        self.gate(
            "slow_insert_alignment",
            np.all(np.abs(error) <= tolerance),
            f"steps={steps_taken} "
            f"target={np.round(target, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * error, 1).tolist()}",
        )

    def level_release_support_surfaces(self):
        remaining_deg = abs(RELEASE_WRIST_LEVEL_DEG)
        direction = math.copysign(1.0, RELEASE_WRIST_LEVEL_DEG)
        while remaining_deg > 1e-9:
            step_deg = direction * min(1.0, remaining_deg)
            axis = self.roll_axis()
            turn = rotation_axis_angle(axis, math.radians(step_deg))
            positions = {}
            rotations = {}
            for hand, arm in (("l", self.ct.L), ("r", self.ct.R)):
                mount_position = self.data.xpos[arm.mount].copy()
                mount_rotation = (
                    self.data.xmat[arm.mount].reshape(3, 3).copy()
                )
                anchor = arm.padmid().copy()
                target_rotation = turn @ mount_rotation
                positions[hand] = anchored_mount_position(
                    mount_position,
                    mount_rotation,
                    anchor,
                    target_rotation,
                )
                rotations[hand] = target_rotation
            self.move_mounts(positions, rotations, iterations=400)
            self.require_held("levelling_release_support_surfaces")
            remaining_deg -= abs(step_deg)
        spans = self.pad_vertical_span()
        self.gate(
            "release_support_surfaces_levelled",
            max(spans.values()) <= 0.025,
            "pad_vertical_span_mm="
            + json.dumps({
                name: round(1000.0 * span, 1)
                for name, span in spans.items()
            }),
        )

    def regrasp_at_release_tips(self):
        for hand, support in (("r", "l"), ("l", "r")):
            self.phase(f"release_{hand}_for_tip_regrasp")
            self.ct.grip_cmd[hand] = self.ct.GRIP_OPEN
            self.frames(120)
            self.require_hand_held(
                support, f"supporting_{hand}_tip_regrasp"
            )

            self.phase(f"shift_{hand}_to_release_tip")
            deltas = {"l": np.zeros(3), "r": np.zeros(3)}
            deltas[hand][0] = RELEASE_TIP_REGRASP_X_M[hand]
            self.move_mount_command_deltas(deltas)

            self.phase(f"regrasp_{hand}_at_release_tip")
            self.ct.grip_cmd[hand] = self.ct.GRIP_CLOSE
            self.frames(120)
            self.require_held(f"{hand}_release_tip_regrasp")

    def advance_until_gentle_touch(self):
        roll_contact = {"force_n": 0.0, "pairs": []}
        steps_taken = 0
        for _ in range(RELEASE_TOUCH_MAX_STEPS):
            roll_contact = self.contact_evidence(
                {self.roll_geom}, self.release_touch_geom_ids
            )
            if roll_contact["force_n"] >= RELEASE_TOUCH_MIN_FORCE_N:
                break
            self.move_mount_commands_delta([
                RELEASE_TOUCH_STEP_M,
                0.0,
                0.0,
            ])
            steps_taken += 1
            self.require_held("gentle_touch_advance")
            pad_contact = self.contact_evidence(
                self.pad_ids, self.shelf_geom_ids
            )
            self.gate(
                "grippers_clear_during_gentle_touch",
                pad_contact["force_n"] <= 0.2,
                f"force_n={pad_contact['force_n']:.4f} "
                f"pairs={pad_contact['pairs']}",
            )
        roll_contact = self.contact_evidence(
            {self.roll_geom}, self.release_touch_geom_ids
        )
        self.gate(
            "gentle_roll_touch",
            RELEASE_TOUCH_MIN_FORCE_N
            <= roll_contact["force_n"]
            <= RELEASE_TOUCH_MAX_FORCE_N,
            f"steps={steps_taken} force_n={roll_contact['force_n']:.4f} "
            f"pairs={roll_contact['pairs']}",
        )
        return roll_contact

    def release_with_axis_withdrawal(self):
        self.release_contact_monitor = {
            "maximum_pad_force": 0.0,
            "maximum_pad_pairs": [],
            "minimum_roll_x": float("inf"),
            "maximum_roll_x": -float("inf"),
        }
        for geom in self.pad_ids:
            self.model.geom_friction[geom, 0] = (
                RELEASE_PAD_SLIDING_FRICTION
            )
        print(
            "[release_setup] pad_sliding_friction="
            f"{RELEASE_PAD_SLIDING_FRICTION:.3f}",
            flush=True,
        )
        self.frames(RELEASE_FRICTION_SETTLE_TICKS)
        self.require_held("release_friction_transition")
        pad_contact = self.contact_evidence(
            self.pad_ids, self.shelf_geom_ids
        )
        self.gate(
            "grippers_clear_after_release_friction_transition",
            pad_contact["force_n"] <= 0.2,
            f"force_n={pad_contact['force_n']:.4f} "
            f"pairs={pad_contact['pairs']}",
        )
        roll_contact = self.contact_evidence(
            {self.roll_geom}, self.release_touch_geom_ids
        )
        self.gate(
            "gentle_touch_preserved_for_release",
            RELEASE_TOUCH_MIN_FORCE_N
            <= roll_contact["force_n"]
            <= RELEASE_TOUCH_MAX_FORCE_N,
            f"force_n={roll_contact['force_n']:.4f} "
            f"pairs={roll_contact['pairs']}",
        )
        self.ct.GRIP_RATE = RELEASE_GRIP_RATE
        self.ct.grip_cmd["l"] = self.ct.GRIP_OPEN
        self.ct.grip_cmd["r"] = self.ct.GRIP_OPEN
        self.frames(120)

        self.move_mount_commands_delta([
            0.0,
            0.0,
            RELEASE_OPEN_RAISE_M,
        ])

        steps_taken = 0
        for step_index in range(RELEASE_AXIS_MAX_STEPS):
            step_m = release_axis_slide_distance(step_index)
            center_x_correction = float(np.clip(
                TARGET_CENTER[0] - self.roll_position()[0],
                -0.015,
                0.015,
            ))
            self.move_mount_command_deltas({
                "l": np.array([
                    center_x_correction,
                    step_m,
                    0.0,
                ]),
                "r": np.array([
                    center_x_correction,
                    -step_m,
                    0.0,
                ]),
            })
            steps_taken += 1
            left = self.grip_evidence("L")
            right = self.grip_evidence("R")
            if left["pads"] and right["pads"]:
                for _ in range(3):
                    axis = self.roll_axis()
                    if (
                        abs(float(axis[0])) <= 0.0008
                        and abs(float(axis[2])) <= 0.02
                    ):
                        break
                    left_delta = symmetric_axis_correction(
                        axis,
                        self.ct.L.padmid(),
                        self.ct.R.padmid(),
                        TARGET_AXIS,
                        0.003,
                    )
                    self.move_mount_command_deltas({
                        "l": left_delta,
                        "r": -left_delta,
                    })
                left = self.grip_evidence("L")
                right = self.grip_evidence("R")
            print(
                f"[release_axis_withdrawal] step={steps_taken} "
                f"roll={np.round(self.roll_position(), 6).tolist()} "
                f"left={left['force_n']:.3f}N/{left['pads']} "
                f"right={right['force_n']:.3f}N/{right['pads']}",
                flush=True,
            )
            if not left["pads"] and not right["pads"]:
                break

        left = self.grip_evidence("L")
        right = self.grip_evidence("R")
        monitor = self.release_contact_monitor
        self.release_contact_monitor = None
        self.gate(
            "released_by_axis_withdrawal",
            not left["pads"]
            and not right["pads"]
            and left["force_n"] <= 0.05
            and right["force_n"] <= 0.05,
            f"steps={steps_taken} "
            f"left={left['force_n']:.3f}N/{left['pads']} "
            f"right={right['force_n']:.3f}N/{right['pads']}",
        )
        self.gate(
            "grippers_clear_during_release",
            monitor["maximum_pad_force"] <= 0.2,
            f"max_force_n={monitor['maximum_pad_force']:.4f} "
            f"pairs={monitor['maximum_pad_pairs']} "
            f"roll_x_range=[{monitor['minimum_roll_x']:.4f}, "
            f"{monitor['maximum_roll_x']:.4f}]",
        )

    def track_success(self, ticks=240):
        tracker = self.tracker_cls()
        evidence = None
        for _ in range(int(ticks)):
            dt = self.tick()
            evidence = self.evaluate_placement(self.model, self.data)
            if tracker.update(evidence, dt):
                evidence = dict(evidence)
                evidence["stable_seconds"] = round(
                    tracker.stable_seconds, 4
                )
                return evidence
        return evidence

    def execute(self):
        self.frames(30)

        initial_targets = {
            hand: self.ct.qtgt[hand].copy()
            for hand in ("l", "r")
        }
        initial_joints = self.arm_joint_positions()

        self.phase("navigate_to_table_observation")
        self.go_to(
            TABLE_OBSERVATION_XY,
            -math.pi / 2.0,
            max_speed=0.12,
        )
        target_motion = max(
            float(np.max(np.abs(self.ct.qtgt[hand] - initial_targets[hand])))
            for hand in ("l", "r")
        )
        measured_joints = self.arm_joint_positions()
        measured_motion = max(
            float(np.max(np.abs(
                measured_joints[hand] - initial_joints[hand]
            )))
            for hand in ("l", "r")
        )
        self.gate(
            "arms_unchanged_before_observation",
            target_motion <= 1e-12 and measured_motion <= 0.03,
            f"target_motion_rad={target_motion:.6f} "
            f"measured_motion_rad={measured_motion:.4f}",
        )
        base = self.ct.base_pose()
        self.gate(
            "table_observation_park",
            float(np.linalg.norm(
                base[:2] - TABLE_OBSERVATION_XY
            )) <= 0.02
            and abs(angle(base[2] + math.pi / 2.0)) <= 0.02,
            f"base={np.round(base, 4).tolist()}",
        )

        self.phase("observe_roll")
        self.frames(45)
        roll = self.roll_position()

        self.phase("raise_arms_after_observation")
        pregrasp_positions = {
            "l": np.array([GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z + 0.10]),
            "r": np.array([-GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z + 0.10]),
        }
        rotations = {
            "l": grasp_target_rotation(self.ct.R_DES, 1.0),
            "r": grasp_target_rotation(self.ct.R_DES, -1.0),
        }
        self.move_mounts(
            pregrasp_positions,
            rotations,
            iterations=1200,
            solve_base_pose=[
                TABLE_GRASP_XY[0],
                TABLE_GRASP_XY[1],
                -math.pi / 2.0,
            ],
        )

        self.phase("approach_table_for_grasp")
        self.go_to(TABLE_GRASP_XY, -math.pi / 2.0, max_speed=0.08)
        base = self.ct.base_pose()
        self.gate(
            "table_park",
            float(np.linalg.norm(base[:2] - TABLE_GRASP_XY)) <= 0.02
            and abs(angle(base[2] + math.pi / 2.0)) <= 0.02,
            f"base={np.round(base, 4).tolist()}",
        )

        self.phase("pregrasp")
        self.move_mounts(pregrasp_positions, rotations, iterations=1200)

        self.phase("lower_and_grasp")
        grasp_positions = {
            "l": np.array([GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z]),
            "r": np.array([-GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z]),
        }
        self.move_mounts(grasp_positions, rotations, iterations=1200)
        self.ct.grip_cmd["l"] = self.ct.GRIP_CLOSE
        self.ct.grip_cmd["r"] = self.ct.GRIP_CLOSE
        self.frames(120)

        self.phase("lift")
        table_height = float(self.roll_position()[2])
        lift_positions = {
            "l": self.data.xpos[self.ct.L.mount].copy() + [0.0, 0.0, 0.10],
            "r": self.data.xpos[self.ct.R.mount].copy() + [0.0, 0.0, 0.10],
        }
        lift_rotations = {
            "l": self.data.xmat[self.ct.L.mount].reshape(3, 3).copy(),
            "r": self.data.xmat[self.ct.R.mount].reshape(3, 3).copy(),
        }
        self.move_mounts(lift_positions, lift_rotations, iterations=800)
        self.frames(30)
        lifted = float(self.roll_position()[2] - table_height)
        self.require_held("lift")
        self.gate("lift_height", lifted >= 0.075, f"lifted={lifted:.4f}m")

        self.phase("raise_for_full_hand_flattening")
        for _ in range(10):
            height_error = HAND_FLAT_ROLL_Z - float(self.roll_position()[2])
            if height_error <= 0.003:
                break
            self.move_mounts_delta(
                [0.0, 0.0, min(0.03, height_error + 0.002)]
            )
            self.require_held("raising_for_flattening")
        self.gate(
            "full_flattening_height",
            self.roll_position()[2] >= HAND_FLAT_ROLL_Z - 0.005,
            f"target_z={HAND_FLAT_ROLL_Z:.4f} "
            f"actual_z={self.roll_position()[2]:.4f}",
        )

        self.phase("flatten_hands")
        self.flatten_hands()
        self.require_held("hands_flat")

        self.phase("clear_table")
        self.reverse(0.30)
        self.require_held("table_clear")

        self.phase("rotate_to_shelf")
        self.turn_in_place(0.0)
        self.require_held("rotated")

        self.phase("navigate_to_shelf_stage")
        base = self.ct.base_pose()
        roll = self.roll_position()
        carried_offset = roll[:2] - base[:2]
        stage_center = TARGET_CENTER.copy()
        stage_center[0] += SHELF_STAGE_OFFSET_X
        stage_center[1] = RELEASE_APPROACH_Y_BIAS_M
        stage_center[2] = roll[2]
        shelf_base = stage_center[:2] - carried_offset
        print(
            f"[transport_plan] carried_offset={np.round(carried_offset, 4).tolist()} "
            f"base_target={np.round(shelf_base, 4).tolist()}",
            flush=True,
        )
        self.go_to(shelf_base, 0.0, max_speed=0.12, tolerance=0.006)
        self.require_held("shelf_park")

        self.phase("align_slot_axis_above_shelf")
        self.align_roll_axis(
            "aligning_slot_axis_above_shelf",
            "slot_axis_x_above_shelf",
        )

        self.phase("realign_shelf_stage_after_axis")
        self.align_roll_center(
            stage_center,
            "high_stage_realignment",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.02,
            attempts=12,
        )
        self.require_held("axis_aligned_high_stage")

        self.phase("lower_near_top_slot")
        low_stage = stage_center.copy()
        low_stage[2] = RELEASE_ROLL_Z
        self.align_roll_center(
            low_stage,
            "low_stage_alignment",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.02,
            attempts=16,
        )
        self.require_held("low_stage")

        self.phase("level_release_support_surfaces")
        self.level_release_support_surfaces()
        self.align_roll_center(
            low_stage,
            "after_release_surface_leveling",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=8,
        )
        self.require_held("release_surfaces_levelled")

        insert_target = low_stage.copy()
        insert_target[0] = RELEASE_PRE_TOUCH_X_M
        tip_regrasp_stage = low_stage.copy()
        tip_regrasp_stage[0] = RELEASE_TIP_REGRASP_STAGE_X_M

        self.phase("align_slot_axis_before_tip_regrasp_stage")
        self.align_roll_axis_with_arms(
            "aligning_slot_axis_before_tip_regrasp_stage",
            "slot_axis_before_tip_regrasp_stage",
        )

        self.phase("recenter_before_tip_regrasp_stage")
        self.align_roll_center(
            low_stage,
            "center_before_tip_regrasp_stage",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=6,
        )
        self.require_held("centered_before_tip_regrasp_stage")

        self.phase("verify_slot_axis_before_tip_regrasp_stage")
        stage_axis = self.roll_axis()
        self.gate(
            "slot_axis_ready_for_tip_regrasp_stage",
            abs(float(stage_axis[0])) <= 0.0008
            and abs(float(stage_axis[2])) <= 0.02,
            f"axis={np.round(stage_axis, 6).tolist()}",
        )

        self.phase("slow_forward_to_tip_regrasp_stage")
        self.slowly_insert_roll(
            tip_regrasp_stage,
            max_step=RELEASE_INSERT_STEP_M,
        )
        self.require_held("tip_regrasp_stage")

        self.phase("regrasp_at_release_tips")
        self.regrasp_at_release_tips()
        self.align_roll_center(
            tip_regrasp_stage,
            "after_release_tip_regrasp",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=8,
        )
        self.require_held("release_tip_regrasp")

        self.phase("fine_align_slot_axis_with_arms")
        self.align_roll_axis_with_arms(
            "fine_aligning_slot_axis",
            "slot_axis_fine_alignment",
        )

        self.phase("recenter_low_stage_after_axis_alignment")
        self.align_roll_center(
            tip_regrasp_stage,
            "low_stage_recentered",
            [0.002, PRE_RELEASE_Y_TOLERANCE_M, 0.002],
            command_space=True,
            max_step=0.004,
            attempts=6,
        )
        self.require_held("low_stage_recentered")

        self.phase("verify_slot_axis_before_insert")
        insert_axis = self.roll_axis()
        self.gate(
            "slot_axis_ready_for_insert",
            abs(float(insert_axis[0])) <= 0.0008
            and abs(float(insert_axis[2])) <= 0.02,
            f"axis={np.round(insert_axis, 6).tolist()}",
        )

        self.phase("slow_forward_insert")
        self.slowly_insert_roll(
            insert_target,
            max_step=RELEASE_INSERT_STEP_M,
        )
        self.require_held("inserted_over_slot")
        fit_margin = self.slot_fit_margin()
        self.gate(
            "pre_touch_slot_fit",
            fit_margin >= 0.00005,
            f"fit_margin_mm={1000.0 * fit_margin:.3f}",
        )
        final_axis = self.roll_axis()
        self.gate(
            "final_slot_axis_x",
            insertion_axis_is_safe(final_axis),
            f"axis={np.round(final_axis, 6).tolist()}",
        )

        self.phase("gentle_touch_shelf")
        self.advance_until_gentle_touch()

        self.phase("position_above_top_slot_for_release")
        self.gate(
            "release_height_ready",
            abs(float(self.roll_position()[2]) - RELEASE_ROLL_Z) <= 0.002,
            f"target_roll_z={RELEASE_ROLL_Z:.4f} "
            f"actual_roll_z={self.roll_position()[2]:.4f}",
        )

        before_release = self.evaluate_placement(self.model, self.data)
        self.gates["pre_release_evidence"] = before_release
        self.gate(
            "pre_release_y_centered",
            abs(
                float(self.roll_position()[1])
                - RELEASE_APPROACH_Y_BIAS_M
            )
            <= PRE_RELEASE_Y_TOLERANCE_M,
            f"target_y_mm={1000.0 * RELEASE_APPROACH_Y_BIAS_M:.2f} "
            f"actual_y_mm={1000.0 * self.roll_position()[1]:.2f}",
        )
        endpoint_margins = before_release["endpoint_margin_m"]
        self.gate(
            "pre_release_endpoint_margins",
            min(endpoint_margins.values())
            >= PRE_RELEASE_ENDPOINT_MARGIN_M,
            f"margins_mm={json.dumps({side: round(1000.0 * margin, 2) for side, margin in endpoint_margins.items()})}",
        )
        self.gate(
            "pre_release_axis_alignment",
            before_release["axis_error_deg"] <= 5.0,
            f"axis_error_deg={before_release['axis_error_deg']}",
        )

        self.phase("release")
        print(
            f"[release] gripper_rate={RELEASE_GRIP_RATE:.3f}m/s "
            f"open_raise_mm={1000.0 * RELEASE_OPEN_RAISE_M:.1f} "
            "then withdraw along roll axis",
            flush=True,
        )
        self.release_with_axis_withdrawal()

        self.phase("verify_before_retract")
        pre_retract_evidence = self.track_success()
        self.gates["pre_retract_evidence"] = pre_retract_evidence
        self.gate(
            "placed_before_retract",
            pre_retract_evidence is not None
            and pre_retract_evidence.get("instantaneous_success") is True
            and pre_retract_evidence.get("stable_seconds", 0.0) >= 0.5,
            json.dumps(
                pre_retract_evidence,
                ensure_ascii=False,
            ),
        )

        self.phase("retract_arms_after_release")
        start_mount_x = float(np.mean([
            self.data.xpos[self.ct.L.mount, 0],
            self.data.xpos[self.ct.R.mount, 0],
        ]))
        self.move_mounts_delta([-ARM_RETRACT_M, 0.0, 0.0])
        end_mount_x = float(np.mean([
            self.data.xpos[self.ct.L.mount, 0],
            self.data.xpos[self.ct.R.mount, 0],
        ]))
        pad_contact = self.contact_evidence(
            self.pad_ids, self.shelf_geom_ids
        )
        self.gate(
            "arms_retracted",
            start_mount_x - end_mount_x >= 0.08
            and pad_contact["force_n"] <= 0.2,
            f"retracted_mm={1000.0 * (start_mount_x - end_mount_x):.1f} "
            f"shelf_contact_n={pad_contact['force_n']:.4f} "
            f"pairs={pad_contact['pairs']}",
        )

        self.phase("terminal_success_hold")
        self.final_evidence = self.track_success()
        self.gate(
            "sorting_roll_success_after_retract",
            self.final_evidence is not None
            and self.final_evidence.get("instantaneous_success") is True
            and self.final_evidence.get("stable_seconds", 0.0) >= 0.5,
            json.dumps(
                self.final_evidence
                or self.evaluate_placement(self.model, self.data),
                ensure_ascii=False,
            ),
        )
        self.frames(60)
        return True

    def encode_review_video(self):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(self.ct.REC_FPS),
            "-i",
            str(self.review_dir / "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(self.review_video),
        ]
        subprocess.run(command, check=True)

    def finalize(self, success, error=None):
        self.ct.REC["on"] = False
        self.ct.REC["metadata"]["gates"] = self.gates
        self.ct.REC["metadata"]["final_evidence"] = self.final_evidence
        self.recorder.finalize(success=bool(success))

        if self.recorder.n:
            episode_path = self.out / "episode_data.npz"
            with np.load(episode_path, allow_pickle=False) as data:
                payload = {name: np.asarray(data[name]) for name in data.files}
            if len(self.recorded_roll_qpos) != self.recorder.n:
                raise RuntimeError(
                    "roll-state log and recorded camera frame counts differ"
                )
            payload["roll_qpos"] = np.asarray(
                self.recorded_roll_qpos, dtype=np.float32
            )
            payload["roll_qvel"] = np.asarray(
                self.recorded_roll_qvel, dtype=np.float32
            )
            np.savez(episode_path, **payload)

            meta_path = self.out / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({
                "task": "sorting_roll_cruzr",
                "task_version": TASK_VERSION,
                "prompt": os.environ["REC_PROMPT"],
                "success": bool(success),
                "success_source": (
                    "sorting_roll_task.SortingRollSuccessTracker"
                ),
                "training_eligible": False,
                "review_video": self.review_video.name,
            })
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self.recorder.close()
        if self.recorder.n:
            self.encode_review_video()

        result = {
            "task": "sorting_roll_cruzr",
            "task_version": TASK_VERSION,
            "seed": self.args.seed,
            "success": bool(success),
            "error": error,
            "num_frames": int(self.recorder.n),
            "sim_seconds": round(float(self.sim_seconds), 3),
            "gates": self.gates,
            "final_evidence": self.final_evidence,
            "review_video": str(self.review_video),
        }
        self.result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")

    sys.path.insert(0, str(CORE_DIR))
    import sorting_roll_scene

    scene_path = sorting_roll_scene.materialize_scene()
    ct = load_teleop(scene_path, args.gpu, args.seed)
    import mujoco
    from sorting_roll_task import evaluate_placement, SortingRollSuccessTracker
    from teleop_timing import CumulativeSubstepScheduler

    scheduler = CumulativeSubstepScheduler(
        ct.TARGET_FPS, ct.m.opt.timestep
    )
    expert = SortingRollExpert(
        args,
        ct,
        mujoco,
        scheduler,
        evaluate_placement,
        SortingRollSuccessTracker,
    )
    success = False
    error = None
    try:
        success = expert.execute()
    except ExpertFailure as exc:
        error = str(exc)
        print(f"[expert] FAIL {error}", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[expert] ERROR {error}", flush=True)
        raise
    finally:
        expert.finalize(success, error=error)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
