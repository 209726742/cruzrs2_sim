#!/usr/bin/env python3
"""Run one fixed-seed Sorting Roll closed-loop evaluation."""

import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
CORE_DIR = PACKAGE_ROOT / "scripts/core"
COLLECTION_DIR = PACKAGE_ROOT / "scripts/collection"
MANIFEST = Path(os.environ["ROLLOUT_MANIFEST"]).resolve()
MANIFEST_KIND = os.environ.get("ROLLOUT_MANIFEST_KIND", "v15")
SEED = int(os.environ["ROLLOUT_SEED"])
MAX_STEPS = int(os.environ.get("ROLLOUT_STEPS", "1800"))
REPLAN = int(os.environ.get("ROLLOUT_REPLAN", "20"))
POLICY_SEED = int(os.environ.get("POLICY_SAMPLE_SEED", "28000"))
CHECKPOINT_LABEL = os.environ["CHECKPOINT_LABEL"]
EXPECTED_CHECKPOINT = Path(os.environ["EXPECTED_CHECKPOINT"]).resolve()
OUTPUT = Path(os.environ["ROLLOUT_OUTPUT"]).resolve()
POLICY_PORT = int(os.environ.get("POLICY_PORT", "8742"))
PREPARE_ONLY = os.environ.get("ROLLOUT_PREPARE_ONLY") == "1"
LOGICAL_CAMERAS = (
    "stereo_left",
    "left_wrist_realsense",
    "right_wrist_realsense",
)


for path in (CORE_DIR, COLLECTION_DIR):
    sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "true")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from openpi_client import websocket_client_policy
from sorting_roll_diversity import (
    assignment_for_seed as assignment_for_seed_v15,
    load_manifest as load_manifest_v15,
)
from sorting_roll_v16_pilot_contract import (
    assignment_for_seed as assignment_for_seed_v16,
    load_manifest as load_manifest_v16,
)
from sorting_roll_realsense_profile import MODEL_CAMERA_SOURCES
import sorting_roll_scene


def load_teleop(scene_path, prompt):
    os.environ["TELEOP_SCENE_XML"] = str(scene_path)
    os.environ["TELEOP_VIEWER"] = "passive"
    os.environ["TELEOP_FPS"] = "60"
    os.environ["CRUZR_GRIP_CLOSE"] = "0.025"
    os.environ["CRUZR_EP_SEED"] = str(SEED)
    os.environ["REC_PROMPT"] = prompt
    spec = importlib.util.spec_from_file_location(
        "cruzr_teleop", CORE_DIR / "cruzr_teleop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_image(raw, jpeg_quality):
    buffer = io.BytesIO()
    Image.fromarray(raw).save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    resized = decoded.resize((224, 126), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), "black")
    canvas.paste(resized, (0, 49))
    return np.asarray(canvas, dtype=np.uint8)


def compose_video_frame(third_person, policy_images, step, evidence):
    canvas = Image.new("RGB", (1280, 540), "black")
    canvas.paste(Image.fromarray(third_person), (0, 0))
    draw = ImageDraw.Draw(canvas)
    stable = float(evidence.get("stable_seconds", 0.0))
    draw.rectangle((0, 0, 960, 30), fill=(0, 0, 0))
    draw.text(
        (10, 8),
        f"{CHECKPOINT_LABEL} official PI0.5 | {MANIFEST_KIND} seed {SEED} | "
        f"replan={REPLAN} | sample={POLICY_SEED} | "
        f"t={step / 30.0:.1f}s | stable={stable:.2f}s",
        fill="white",
    )
    for index, (name, image) in enumerate(zip(LOGICAL_CAMERAS, policy_images)):
        view = Image.fromarray(image).resize((180, 180), Image.Resampling.NEAREST)
        y = index * 180
        canvas.paste(view, (1030, y))
        draw.text((965, y + 8), name.replace("_realsense", ""), fill="white")
    return np.asarray(canvas)


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")

    if MANIFEST_KIND == "v15":
        manifest = load_manifest_v15(MANIFEST)
        assignment = assignment_for_seed_v15(manifest, SEED)
        base_assignment = assignment
    elif MANIFEST_KIND == "v16":
        manifest = load_manifest_v16(MANIFEST)
        assignment = assignment_for_seed_v16(manifest, SEED)
        base_assignment = assignment["base_diversity_assignment"]
    else:
        raise RuntimeError(f"unsupported ROLLOUT_MANIFEST_KIND={MANIFEST_KIND!r}")
    prompt = assignment["prompt"]

    scene_path = sorting_roll_scene.materialize_scene()
    ct = load_teleop(scene_path, prompt)
    import mujoco
    from teleop_timing import CumulativeSubstepScheduler
    from sorting_roll_task import SortingRollSuccessTracker, evaluate_placement
    import sorting_roll_expert as sorting_expert

    sorting_expert.apply_model_camera_overrides(mujoco, ct.m)
    scheduler = CumulativeSubstepScheduler(ct.TARGET_FPS, ct.m.opt.timestep)
    args = SimpleNamespace(
        out=str(OUTPUT),
        seed=SEED,
        gpu=0,
        width=640,
        height=360,
        no_render=False,
        review_videos=False,
        randomize=True,
        manifest=MANIFEST,
        diversity_assignment=base_assignment,
    )
    prepared_start_phase = "initial_hold"
    if MANIFEST_KIND == "v15":
        expert = sorting_expert.SortingRollExpert(
            args, ct, mujoco, scheduler, evaluate_placement,
            SortingRollSuccessTracker,
        )
    else:
        from sorting_roll_v16_pilot_expert import SortingRollV16PilotExpert

        class StartStateReady(Exception):
            pass

        class FixedSuiteExpert(SortingRollV16PilotExpert):
            def start_recording(self, phase):
                sorting_expert.SortingRollExpert.phase(self, phase)
                self.recording_started = True
                self.recording_start_sim_seconds = float(self.sim_seconds)
                self.ct.REC["metadata"]["recorded_start_phase"] = phase
                self.ct.REC["on"] = False
                raise StartStateReady(phase)

        expert = FixedSuiteExpert(
            args, ct, mujoco, scheduler, evaluate_placement,
            SortingRollSuccessTracker, assignment,
        )
        try:
            expert.execute()
        except StartStateReady as ready:
            prepared_start_phase = str(ready)
        else:
            raise RuntimeError("v16 expert did not stop at its declared start phase")
    ct.REC["on"] = False
    ct.REC["rec"] = None

    if PREPARE_ONLY:
        payload = {
            "manifest_kind": MANIFEST_KIND,
            "seed": SEED,
            "scenario_family": assignment.get("scenario_family", "nominal"),
            "scenario_variant": assignment.get(
                "scenario_variant", "v15_nominal"
            ),
            "prepared_start_phase": prepared_start_phase,
            "state_shape": list(expert.recorder._state16().shape),
            "finite_state": bool(
                np.isfinite(expert.recorder._state16()).all()
            ),
            "roll_position_m": expert.roll_position().tolist(),
            "scene_randomization": expert.scene_randomization,
        }
        (OUTPUT / "prepared_start_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expert.recorder.close()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model, data = ct.m, ct.d
    camera_renderer = mujoco.Renderer(model, height=360, width=640)
    camera_option = mujoco.MjvOption()
    camera_ids = {
        logical: mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            MODEL_CAMERA_SOURCES[logical],
        )
        for logical in LOGICAL_CAMERAS
    }
    if min(camera_ids.values()) < 0:
        raise RuntimeError(f"missing policy camera: {camera_ids}")

    third_renderer = mujoco.Renderer(model, height=540, width=960)
    third_camera = expert.review_camera
    third_option = expert.visual_review_option
    jpeg_quality = int(base_assignment["image_profile"]["jpeg_quality"])

    def render_policy_images():
        images = []
        for logical in LOGICAL_CAMERAS:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_ids[logical]
            camera_renderer.update_scene(data, camera, camera_option)
            images.append(model_image(camera_renderer.render().copy(), jpeg_quality))
        return images

    def state18():
        state16 = expert.recorder._state16()
        base_velocity = ct.base_velocity().astype(np.float32)
        state = np.concatenate((state16, base_velocity)).astype(np.float32)
        if state.shape != (18,) or not np.isfinite(state).all():
            raise RuntimeError(f"invalid policy state {state}")
        return state

    def apply_action(action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid policy action shape {action.shape}")
        ct.qtgt["l"][:] = np.clip(action[:7], ct.L.lo, ct.L.hi)
        ct.qtgt["r"][:] = np.clip(action[7:14], ct.R.lo, ct.R.hi)
        left_open = float(np.clip(action[14], 0.0, 1.0))
        right_open = float(np.clip(action[15], 0.0, 1.0))
        grip_span = ct.GRIP_CLOSE - ct.GRIP_OPEN
        ct.grip_cmd["l"] = ct.GRIP_OPEN + (1.0 - left_open) * grip_span
        ct.grip_cmd["r"] = ct.GRIP_OPEN + (1.0 - right_open) * grip_span
        ct.base_vel[:] = [
            float(np.clip(action[16], -ct.VMAX, ct.VMAX)),
            float(np.clip(action[17], -ct.WZMAX, ct.WZMAX)),
        ]

    def control_action_step():
        ct.control_step(scheduler.next_substeps())
        ct.control_step(scheduler.next_substeps())

    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1", port=POLICY_PORT
    )
    metadata = client.get_server_metadata()
    served_checkpoint = Path(metadata.get("checkpoint", "")).resolve()
    if served_checkpoint != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"wrong policy server checkpoint: {metadata}")
    if metadata.get("inference_api") != "PI05Policy.predict_action_chunk":
        raise RuntimeError(f"wrong official inference API: {metadata}")

    video_path = OUTPUT / f"sorting_roll_{CHECKPOINT_LABEL}_seed{SEED}_rollout.mp4"
    video = imageio.get_writer(
        video_path,
        fps=15,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    tracker = SortingRollSuccessTracker()
    inference_seconds = []
    action_chunk = None
    action_index = 0
    policy_images = None
    success = False
    executed_steps = 0
    evidence = evaluate_placement(model, data)
    initial_roll_position = data.xpos[expert.roll_body].copy()
    max_roll_z = float(initial_roll_position[2])
    first_bimanual_grasp_step = None
    peak_left_grip_force_n = 0.0
    peak_right_grip_force_n = 0.0
    min_pad_roll_clearance_m = float("inf")
    reached_grasp_workzone = False
    stable_lift_frames = 0
    max_stable_lift_frames = 0
    unsafe_collision_steps = 0
    max_unsafe_penetration_mm = 0.0
    initial_base_yaw = float(ct.base_pose()[2])
    max_abs_yaw_displacement_rad = 0.0
    started = time.perf_counter()
    try:
        for step in range(MAX_STEPS):
            if action_chunk is None or action_index >= REPLAN:
                policy_images = render_policy_images()
                observation = {
                    "observation/state": state18(),
                    "observation/image": policy_images[0],
                    "observation/left_wrist_image": policy_images[1],
                    "observation/right_wrist_image": policy_images[2],
                    "prompt": prompt,
                    "policy_seed": POLICY_SEED,
                }
                infer_started = time.perf_counter()
                action_chunk = np.asarray(client.infer(observation)["actions"])
                inference_seconds.append(time.perf_counter() - infer_started)
                if action_chunk.shape != (50, 18) or not np.isfinite(action_chunk).all():
                    raise RuntimeError(f"invalid action chunk {action_chunk.shape}")
                action_index = 0
                if len(inference_seconds) <= 3 or len(inference_seconds) % 10 == 0:
                    print(
                        f"[rollout] official chunk {len(inference_seconds):03d} "
                        f"at t={step / 30.0:.1f}s "
                        f"latency={inference_seconds[-1]:.3f}s",
                        flush=True,
                    )

            apply_action(action_chunk[action_index])
            action_index += 1
            control_action_step()
            executed_steps = step + 1
            evidence = evaluate_placement(model, data)
            success = tracker.update(evidence, 1.0 / 30.0)
            evidence = dict(evidence)
            evidence["stable_seconds"] = round(tracker.stable_seconds, 4)
            max_roll_z = max(max_roll_z, float(data.xpos[expert.roll_body][2]))
            left_grip = expert.grip_evidence("L")
            right_grip = expert.grip_evidence("R")
            peak_left_grip_force_n = max(
                peak_left_grip_force_n, float(left_grip["force_n"])
            )
            peak_right_grip_force_n = max(
                peak_right_grip_force_n, float(right_grip["force_n"])
            )
            clearance = expert.minimum_geom_clearance(
                expert.pad_ids, {expert.roll_geom}
            )["distance_m"]
            min_pad_roll_clearance_m = min(min_pad_roll_clearance_m, clearance)
            reached_grasp_workzone = reached_grasp_workzone or clearance <= 0.10
            strict_bimanual = (
                float(left_grip["force_n"]) >= 1.0
                and float(right_grip["force_n"]) >= 1.0
                and len(left_grip["pads"]) == 2
                and len(right_grip["pads"]) == 2
            )
            lift_m = float(data.xpos[expert.roll_body][2] - initial_roll_position[2])
            stable_lift_frames = (
                stable_lift_frames + 1
                if strict_bimanual and lift_m >= 0.070
                else 0
            )
            max_stable_lift_frames = max(
                max_stable_lift_frames, stable_lift_frames
            )
            unsafe_contacts = [
                contact
                for contact in expert.early_unintended_arm_contacts(
                    allow_pad_roll=True
                )
                if float(contact["penetration_mm"]) >= 2.0
            ]
            if unsafe_contacts:
                unsafe_collision_steps += 1
                max_unsafe_penetration_mm = max(
                    max_unsafe_penetration_mm,
                    max(float(item["penetration_mm"]) for item in unsafe_contacts),
                )
            yaw_displacement = abs(float(ct.base_pose()[2]) - initial_base_yaw)
            max_abs_yaw_displacement_rad = max(
                max_abs_yaw_displacement_rad, yaw_displacement
            )
            if (
                first_bimanual_grasp_step is None
                and strict_bimanual
            ):
                first_bimanual_grasp_step = executed_steps

            if step % 2 == 0:
                third_renderer.update_scene(data, third_camera, third_option)
                video.append_data(
                    compose_video_frame(
                        third_renderer.render().copy(),
                        policy_images,
                        step,
                        evidence,
                    )
                )
            if step % 150 == 0:
                base = ct.base_pose()
                roll = data.xpos[expert.roll_body]
                print(
                    f"[rollout] t={step / 30.0:4.1f}s "
                    f"base={np.round(base, 3).tolist()} "
                    f"roll={np.round(roll, 3).tolist()} "
                    f"instant={evidence['instantaneous_success']} "
                    f"stable={tracker.stable_seconds:.2f}s",
                    flush=True,
                )
            if success:
                print(f"[rollout] SUCCESS at t={executed_steps / 30.0:.2f}s", flush=True)
                break
    finally:
        video.close()
        camera_renderer.close()
        third_renderer.close()
        expert.recorder.close()

    elapsed = time.perf_counter() - started
    result = {
        "success": bool(success),
        "scope": "full closed-loop MuJoCo rollout",
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_step": metadata.get("checkpoint_step"),
        "policy_class": metadata.get("policy_class"),
        "inference_api": metadata.get("inference_api"),
        "device": metadata.get("device"),
        "policy_sample_seed": POLICY_SEED,
        "checkpoint_label": CHECKPOINT_LABEL,
        "manifest_kind": MANIFEST_KIND,
        "seed": SEED,
        "split": assignment["split"],
        "prompt": prompt,
        "scenario_family": assignment.get("scenario_family", "nominal"),
        "scenario_variant": assignment.get("scenario_variant", "v15_nominal"),
        "prepared_start_phase": prepared_start_phase,
        "diversity_assignment": assignment,
        "scene_randomization": expert.scene_randomization,
        "executed_steps": executed_steps,
        "simulated_seconds": executed_steps / 30.0,
        "wall_seconds": elapsed,
        "replan_steps": REPLAN,
        "predicted_chunk_steps": 50,
        "inference_count": len(inference_seconds),
        "inference_seconds": inference_seconds,
        "mean_inference_seconds": float(np.mean(inference_seconds)),
        "initial_roll_position": initial_roll_position.tolist(),
        "max_roll_z": max_roll_z,
        "max_lift_from_initial_m": max_roll_z - float(initial_roll_position[2]),
        "reached_grasp_workzone": bool(reached_grasp_workzone),
        "min_pad_roll_clearance_m": min_pad_roll_clearance_m,
        "stable_lift_at_least_70mm": bool(max_stable_lift_frames >= 15),
        "max_stable_lift_seconds": max_stable_lift_frames / 30.0,
        "unsafe_collision": bool(unsafe_collision_steps > 0),
        "unsafe_collision_steps": unsafe_collision_steps,
        "max_unsafe_penetration_mm": max_unsafe_penetration_mm,
        "continuous_rotation": bool(max_abs_yaw_displacement_rad > 6.283185),
        "max_abs_yaw_displacement_rad": max_abs_yaw_displacement_rad,
        "lifted_at_least_6cm": bool(
            max_roll_z - float(initial_roll_position[2]) >= 0.06
        ),
        "first_bimanual_grasp_step": first_bimanual_grasp_step,
        "first_bimanual_grasp_seconds": (
            None
            if first_bimanual_grasp_step is None
            else first_bimanual_grasp_step / 30.0
        ),
        "peak_left_grip_force_n": peak_left_grip_force_n,
        "peak_right_grip_force_n": peak_right_grip_force_n,
        "final_roll_position": data.xpos[expert.roll_body].tolist(),
        "final_base_pose": ct.base_pose().tolist(),
        "final_evidence": evidence,
        "video": str(video_path),
        "render_backend": "Mesa llvmpipe software EGL",
    }
    (OUTPUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
