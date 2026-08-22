#!/usr/bin/env python3
"""Replay a Sorting Roll episode through the provisional RealSense views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PACKAGE_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from sorting_roll_realsense_profile import (
    MODEL_CAMERA_SOURCES,
    apply_model_camera_overrides,
    profile_report,
)
from sorting_roll_camera_audit import _restore_frame


DEFAULT_SCENE = PACKAGE_ROOT / "assets" / "sorting_roll_scene.xml"


def _review_camera(mujoco):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.40, -0.45, 0.82]
    camera.distance = 2.55
    camera.azimuth = 45.0
    camera.elevation = -22.0
    return camera


def _label(image, text):
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((6, 6, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((11, 11), text, fill=(255, 255, 255), font=font)
    return np.asarray(result)


def _video_command(args, fps):
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{args.width * 2}x{args.height * 2}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(args.out),
    ]


def replay(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    import mujoco

    with np.load(args.episode / "episode_data.npz", allow_pickle=False) as payload:
        episode = {name: payload[name] for name in payload.files}
    meta = json.loads((args.episode / "meta.json").read_text())
    frame_count = int(episode["state"].shape[0])
    fps = int(meta["fps"])

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    apply_model_camera_overrides(mujoco, model)
    data = mujoco.MjData(model)
    missing = [
        source
        for source in MODEL_CAMERA_SOURCES.values()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, source) < 0
    ]
    if missing:
        raise RuntimeError(f"scene is missing candidate camera sources: {missing}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    review_camera = _review_camera(mujoco)
    process = subprocess.Popen(_video_command(args, fps), stdin=subprocess.PIPE)
    try:
        for frame in range(frame_count):
            _restore_frame(mujoco, model, data, episode, meta, frame)
            panels = []
            for logical, source in MODEL_CAMERA_SOURCES.items():
                renderer.update_scene(data, camera=source)
                panels.append(_label(renderer.render(), logical))
            renderer.update_scene(data, camera=review_camera)
            panels.append(_label(renderer.render(), "third_person_review_only"))
            montage = np.concatenate(
                (
                    np.concatenate(panels[:2], axis=1),
                    np.concatenate(panels[2:], axis=1),
                ),
                axis=0,
            )
            process.stdin.write(np.ascontiguousarray(montage).tobytes())
            if frame % max(fps * 10, 1) == 0:
                print(f"rendered {frame}/{frame_count}", flush=True)
    finally:
        renderer.close()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    return {
        "episode": str(args.episode.resolve()),
        "video": str(args.out.resolve()),
        "frames": frame_count,
        "fps": fps,
        "profile": profile_report(),
        "review_panel_is_policy_input": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.width < 64 or args.height < 64:
        parser.error("panel resolution must be at least 64x64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    result = replay(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
