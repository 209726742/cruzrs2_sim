#!/usr/bin/env python3
"""Record and audit a short, non-training SDK camera/timestamp smoke episode."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from cruzr_s2_sdk_contract import (  # noqa: E402
    SDK_CAMERAS,
    SDK_COLLECTION_PROFILE,
    SDK_DOC_REVISION,
    SDK_TASK_HEAD_POSE_RAD,
    audit_sdk_episode,
    load_sdk_timestamp_sidecar,
)
from teleop_timing import CumulativeSubstepScheduler  # noqa: E402


def _materialize_scene_for_asset_relative_includes(scene: str) -> tuple[str, str | None]:
    """Place an e2e template beside robot assets so its relative include resolves."""
    assets_dir = os.path.join(ROOT, "assets")
    if os.path.dirname(scene) == assets_dir:
        return scene, None
    with open(scene, encoding="utf-8") as source:
        scene_text = source.read()
    fd, temporary = tempfile.mkstemp(
        prefix=".sdk_capture_smoke_", suffix=".xml", dir=assets_dir, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            destination.write(scene_text)
    except BaseException:
        os.remove(temporary)
        raise
    return temporary, temporary


def _load_teleop(scene: str, gpu: int):
    os.environ["TELEOP_SCENE_XML"] = scene
    os.environ["TELEOP_HOME"] = "droop"
    os.environ["TELEOP_FPS"] = "60"
    os.environ.pop("TELEOP_SUBSTEPS", None)
    os.environ["REC_CAMS"] = ",".join(SDK_CAMERAS)
    os.environ["REC_SAVE_RAW_TIMESTAMPS"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["TELEOP_RECORD_GPU"] = str(gpu)
    os.environ["CRUZR_EP_SEED"] = "0"

    spec = importlib.util.spec_from_file_location(
        "cruzr_teleop", os.path.join(CORE_DIR, "cruzr_teleop.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_third_person(recorder, mujoco, model, data, camera, out_path: str) -> None:
    """Reuse the recorder's EGL context without adding a fourth policy camera."""
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
    Image.fromarray(np.flipud(rgb)).save(out_path, quality=90)


def _audit(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with np.load(
        os.path.join(out_dir, "episode_data.npz"), allow_pickle=False
    ) as data:
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        base_action = np.asarray(data["base_action"])
        timestamp = np.asarray(data["timestamp"])
    sdk_state_timestamp, camera_timestamps = load_sdk_timestamp_sidecar(out_dir)
    return audit_sdk_episode(
        state,
        action,
        base_action,
        fps=float(meta["fps"]),
        joint_names=meta["action_names"],
        cameras=tuple(meta["cameras"]),
        timestamp=timestamp,
        sdk_state_timestamp=sdk_state_timestamp,
        camera_timestamps=camera_timestamps,
        require_camera_timestamps=True,
        enforce_rated_speed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new smoke output directory")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scene",
        default=os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml"),
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    scene = os.path.abspath(args.scene)
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if os.path.exists(out_dir):
        raise SystemExit(f"refusing to overwrite existing smoke output: {out_dir}")
    if not os.path.isfile(scene):
        raise SystemExit(f"scene does not exist: {scene}")
    loaded_scene, temporary_scene = _materialize_scene_for_asset_relative_includes(scene)
    try:
        ct = _load_teleop(loaded_scene, args.gpu)
    finally:
        if temporary_scene is not None:
            os.remove(temporary_scene)
    import mujoco

    os.makedirs(out_dir)
    ct.REC_WH = (224, 224)
    recorder = ct.EpisodeRecorder(out_dir)
    ct.REC.update({
        "rec": recorder,
        "on": True,
        "count": 0,
        "phase": "sdk_capture_smoke",
        "metadata": {
            "collection_profile": SDK_COLLECTION_PROFILE,
            "sdk_document_revision": SDK_DOC_REVISION,
            "sdk_timestamp_source": "mujoco_sim_time_synchronous_render",
            "sdk_task_head_pose_rad": dict(SDK_TASK_HEAD_POSE_RAD),
            "capture_smoke_only": True,
            "training_eligible": False,
        },
    })

    diagnostic_dir = os.path.join(out_dir, "diagnostics", "third_person")
    os.makedirs(diagnostic_dir, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.8, 0.0, 1.0]
    camera.distance = 5.2
    camera.azimuth = 55.0
    camera.elevation = -18.0

    scheduler = CumulativeSubstepScheduler(ct.TARGET_FPS, ct.m.opt.timestep)
    control_ticks = args.frames * ct.REC_DECIM
    for _ in range(control_ticks):
        previous_frames = recorder.n
        ct.control_step(scheduler.next_substeps())
        if recorder.n != previous_frames:
            frame_path = os.path.join(
                diagnostic_dir, f"frame_{recorder.n - 1:06d}.jpg"
            )
            _render_third_person(recorder, mujoco, ct.m, ct.d, camera, frame_path)

    ct.REC["on"] = False
    recorder.finalize(success=False)
    recorder.close()
    audit = _audit(out_dir)
    frame_counts = {
        camera_name: len(glob.glob(os.path.join(
            out_dir, "frames", camera_name, "frame_*.jpg"
        )))
        for camera_name in SDK_CAMERAS
    }
    diagnostic_count = len(glob.glob(os.path.join(
        diagnostic_dir, "frame_*.jpg"
    )))
    counts_ok = (
        set(frame_counts.values()) == {args.frames}
        and diagnostic_count == args.frames
    )
    result = {
        "passed": bool(audit["passed"] and counts_ok),
        "purpose": "capture_smoke_only_not_training_data",
        "collection_profile": SDK_COLLECTION_PROFILE,
        "sdk_document_revision": SDK_DOC_REVISION,
        "scene": scene,
        "requested_frames": args.frames,
        "control_ticks": control_ticks,
        "physics_steps": scheduler.physics_steps,
        "frame_counts": frame_counts,
        "diagnostic_third_person_frames": diagnostic_count,
        "audit": audit,
    }
    if not counts_ok:
        result["frame_count_error"] = (
            f"policy={frame_counts}, third_person={diagnostic_count}, "
            f"expected={args.frames}"
        )
    with open(
        os.path.join(out_dir, "capture_smoke_result.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
