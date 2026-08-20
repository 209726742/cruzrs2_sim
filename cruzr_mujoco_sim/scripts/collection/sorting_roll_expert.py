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
TASK_VERSION = "sorting_roll_v1"
POLICY_CAMERAS = ("stereo_left", "stereo_right", "waist_front")
TARGET_CENTER = np.array([0.9825, 0.0, 1.0125], dtype=float)
SHELF_APPROACH_Z = TARGET_CENTER[2] + 0.22
ROLL_HALF_LENGTH = 0.25
ROLL_RADIUS = 0.012
SLOT_HALF_WIDTH = 0.0125
SLOT_X_COMMAND_BIAS = -0.0003
RELEASE_GRIP_RATE = 0.022
BASE_ACCEL = 0.5
BASE_YAW_ACCEL = 0.2
CONTROL_FPS = 60.0
GRASP_X = 0.13
GRASP_Y_BIAS = 0.009
GRASP_MOUNT_Z = 1.135
GRIP_FORCE_MIN_N = 0.2
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


def cylinder_slot_fit_margin(center_x, axis_x):
    axis_x = abs(float(axis_x))
    half_x = (
        ROLL_HALF_LENGTH * axis_x
        + ROLL_RADIUS * math.sqrt(max(0.0, 1.0 - axis_x * axis_x))
    )
    center_error = abs(float(center_x) - float(TARGET_CENTER[0]))
    return SLOT_HALF_WIDTH - half_x - center_error

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
        self.recorded_roll_qpos = []
        self.recorded_roll_qvel = []
        self.gates = {}
        self.final_evidence = None
        self.sim_seconds = 0.0

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
                "collection_profile": "sorting_roll_canary_v1",
                "training_eligible": False,
                "success_source": "sorting_roll_task.SortingRollSuccessTracker",
                "policy_cameras": list(POLICY_CAMERAS),
                "review_camera": "free_camera_not_policy_input",
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
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.mujoco.mj_forward(self.model, self.data)
        return targets, residuals

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

    def wait_arm_tracking(self, label):
        stable_ticks = 0
        error = float("inf")
        for tick in range(1, ARM_TRACK_MAX_TICKS + 1):
            error = max(
                abs(float(self.data.qpos[address]) - float(target))
                for hand, arm in (("l", self.ct.L), ("r", self.ct.R))
                for address, target in zip(arm.qadr, self.ct.qtgt[hand])
            )
            stable_ticks = stable_ticks + 1 if error <= ARM_TRACK_TOL_RAD else 0
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

    def move_mounts(self, positions, rotations, iterations=240, solve_base_pose=None):
        targets, residuals = self.solve_mounts(
            positions,
            rotations,
            iterations=iterations,
            base_pose=solve_base_pose,
        )
        self.gate(
            "ik_reachable",
            max(residuals.values()) <= 0.012,
            "residual_mm="
            + ",".join(
                f"{hand}:{1000.0 * residuals[hand]:.1f}"
                for hand in ("l", "r")
            ),
        )
        self.servo_arms(targets)
        self.wait_arm_tracking(self.ct.REC["phase"])

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

    def require_held(self, stage, minimum_force=GRIP_FORCE_MIN_N):
        left = self.grip_evidence("L")
        right = self.grip_evidence("R")
        position = self.roll_position()
        self.gate(
            f"held_{stage}",
            left["force_n"] >= minimum_force
            and right["force_n"] >= minimum_force
            and position[2] >= 1.055,
            f"position={np.round(position, 4).tolist()} "
            f"left={left['force_n']:.2f}N/{left['pads']} "
            f"right={right['force_n']:.2f}N/{right['pads']}",
        )

    def align_roll_hover(self):
        hover = TARGET_CENTER.copy()
        hover[2] = SHELF_APPROACH_Z
        command_hover = hover.copy()
        command_hover[0] += SLOT_X_COMMAND_BIAS
        for _ in range(12):
            error = hover - self.roll_position()
            if (
                abs(error[0]) <= 0.0002
                and abs(error[1]) <= 0.008
                and abs(error[2]) <= 0.008
            ):
                break
            command_error = command_hover - self.roll_position()
            self.move_mounts_delta(bounded_vector(command_error, 0.025))
        error = hover - self.roll_position()
        self.gate(
            "hover_alignment",
            abs(error[0]) <= 0.0002
            and abs(error[1]) <= 0.008
            and abs(error[2]) <= 0.008,
            f"target={np.round(hover, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * error, 1).tolist()}",
        )

    def shelf_bar_clearance(self):
        top_bar = self.ct.gid("target_top_bar_col")
        top_half = (
            np.abs(self.data.geom_xmat[top_bar].reshape(3, 3))
            @ self.model.geom_size[top_bar, :3]
        )
        top_z = float(self.data.geom_xpos[top_bar, 2] + top_half[2])
        pad_min_z = float("inf")
        for name in ("L_pad1", "L_pad2", "R_pad1", "R_pad2"):
            pad = self.ct.gid(name)
            half = (
                np.abs(self.data.geom_xmat[pad].reshape(3, 3))
                @ self.model.geom_size[pad, :3]
            )
            pad_min_z = min(
                pad_min_z,
                float(self.data.geom_xpos[pad, 2] - half[2]),
            )
        return pad_min_z - top_z

    def roll_axis(self):
        axis = self.data.xmat[self.roll_body].reshape(3, 3)[:, 0].copy()
        return -axis if axis[1] < 0.0 else axis

    def slot_fit_margin(self):
        return cylinder_slot_fit_margin(
            self.roll_position()[0], self.roll_axis()[0]
        )

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

    def execute(self):
        self.frames(30)

        self.phase("raise_arms_clear")
        roll = self.roll_position()
        pregrasp_positions = {
            "l": np.array([GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z + 0.10]),
            "r": np.array([-GRASP_X, roll[1] + GRASP_Y_BIAS, GRASP_MOUNT_Z + 0.10]),
        }
        rotations = {"l": self.ct.R_DES.copy(), "r": self.ct.R_DES.copy()}
        self.move_mounts(
            pregrasp_positions,
            rotations,
            iterations=1200,
            solve_base_pose=[0.0, -1.0, -math.pi / 2.0],
        )

        self.phase("navigate_to_table")
        self.go_to([0.0, -1.0], -math.pi / 2.0)
        base = self.ct.base_pose()
        self.gate(
            "table_park",
            float(np.linalg.norm(base[:2] - [0.0, -1.0])) <= 0.02
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

        self.phase("clear_table")
        self.reverse(0.40)
        self.require_held("table_clear")

        self.phase("rotate_to_shelf")
        self.turn_in_place(0.0)
        self.require_held("rotated")

        self.phase("raise_over_shelf")
        for _ in range(8):
            height_error = SHELF_APPROACH_Z - float(self.roll_position()[2])
            if height_error <= 0.003:
                break
            self.move_mounts_delta(
                [0.0, 0.0, min(0.04, height_error + 0.005)]
            )
            self.require_held("raising_over_shelf")
        position = self.roll_position()
        self.gate(
            "over_shelf_height",
            position[2] >= SHELF_APPROACH_Z - 0.003,
            f"target_z={SHELF_APPROACH_Z:.4f} actual_z={position[2]:.4f}",
        )
        self.require_held("over_shelf_clearance")

        self.phase("align_slot_axis")
        self.align_roll_axis("aligning_slot_axis", "slot_axis_x")
        shelf_yaw = float(self.ct.base_pose()[2])

        self.phase("navigate_to_shelf")
        base = self.ct.base_pose()
        roll = self.roll_position()
        carried_offset = roll[:2] - base[:2]
        shelf_base = TARGET_CENTER[:2] - carried_offset
        print(
            f"[transport_plan] carried_offset={np.round(carried_offset, 4).tolist()} "
            f"base_target={np.round(shelf_base, 4).tolist()}",
            flush=True,
        )
        self.go_to(shelf_base, shelf_yaw, max_speed=0.12, tolerance=0.006)
        self.require_held("shelf_park")

        self.phase("align_above_slot")
        self.align_roll_hover()
        self.phase("final_axis_correction")
        self.align_roll_axis("final_axis_correction", "final_slot_axis_x")
        final_hover = TARGET_CENTER.copy()
        final_hover[2] = SHELF_APPROACH_Z
        final_error = final_hover - self.roll_position()
        self.gate(
            "final_hover_alignment",
            abs(final_error[0]) <= 0.0006
            and abs(final_error[1]) <= 0.008
            and abs(final_error[2]) <= 0.008,
            f"target={np.round(final_hover, 4).tolist()} "
            f"actual={np.round(self.roll_position(), 4).tolist()} "
            f"error_mm={np.round(1000.0 * final_error, 1).tolist()}",
        )
        clearance = self.shelf_bar_clearance()
        self.gate(
            "shelf_bar_clearance",
            clearance >= 0.005,
            f"minimum_clearance_mm={1000.0 * clearance:.1f}",
        )
        self.require_held("slot_aligned")
        before_release = self.evaluate_placement(self.model, self.data)
        self.gates["pre_release_evidence"] = before_release
        fit_margin = self.slot_fit_margin()
        self.gate(
            "pre_release_slot_fit",
            fit_margin >= 0.00005,
            f"fit_margin_mm={1000.0 * fit_margin:.3f}",
        )
        self.gate(
            "axis_alignment",
            before_release["axis_error_deg"] <= 5.0,
            f"axis_error_deg={before_release['axis_error_deg']}",
        )

        self.phase("release")
        self.ct.GRIP_RATE = RELEASE_GRIP_RATE
        print(
            f"[release] gripper_rate={RELEASE_GRIP_RATE:.3f}m/s "
            "settle_ticks=180",
            flush=True,
        )
        self.ct.grip_cmd["l"] = self.ct.GRIP_OPEN
        self.ct.grip_cmd["r"] = self.ct.GRIP_OPEN
        self.frames(180)

        self.phase("terminal_success_hold")
        tracker = self.tracker_cls()
        success = False
        for _ in range(240):
            dt = self.tick()
            evidence = self.evaluate_placement(self.model, self.data)
            if tracker.update(evidence, dt):
                success = True
                self.final_evidence = dict(evidence)
                self.final_evidence["stable_seconds"] = round(
                    tracker.stable_seconds, 4
                )
                break
        self.gate(
            "sorting_roll_success",
            success,
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
