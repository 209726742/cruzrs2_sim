#!/usr/bin/env python3
"""Turn shelf_e2e teleop episode frames/ into watchable mp4 previews.

Input layout (from shelf_e2e_dual_expert.py recording):
  <episode>/frames/<camera>/frame_%06d.jpg

Outputs (written beside meta.json):
  preview_<camera>.mp4          one mp4 per camera
  preview_camera_grid.mp4       metadata-ordered horizontal camera grid

Usage:
  python scripts/collection/shelf_e2e_make_videos.py out/teleop/shelf_e2e_dual/shelf_e2e_dual_000013
  python scripts/collection/shelf_e2e_make_videos.py out/teleop/shelf_e2e_dual/shelf_e2e_dual_*
"""
import glob
import json
import os
import subprocess
import sys

DEFAULT_FPS = 30


def encode_one(cam_dir: str, out_path: str, fps: int) -> None:
    pat = os.path.join(cam_dir, "frame_%06d.jpg")
    if not glob.glob(os.path.join(cam_dir, "frame_*.jpg")):
        raise FileNotFoundError(f"no frames in {cam_dir}")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", pat, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_path],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(r.stderr[:300] or f"ffmpeg failed for {cam_dir}")


def encode_grid(ep_dir: str, cameras: tuple[str, ...], out_path: str, fps: int) -> None:
    ins = [os.path.join(ep_dir, "frames", c, "frame_%06d.jpg") for c in cameras]
    for c in cameras:
        if not glob.glob(os.path.join(ep_dir, "frames", c, "frame_*.jpg")):
            raise FileNotFoundError(f"missing camera {c} under {ep_dir}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in ins:
        cmd += ["-framerate", str(fps), "-i", p]
    cmd += [
        "-filter_complex",
        "".join(f"[{index}:v]" for index in range(len(cameras)))
        + f"hstack=inputs={len(cameras)}[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(r.stderr[:400] or "grid ffmpeg failed")


def episode_fps(ep_dir: str) -> int:
    meta_path = os.path.join(ep_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return int(meta.get("fps", DEFAULT_FPS))
    return DEFAULT_FPS


def episode_cameras(ep_dir: str) -> tuple[str, ...]:
    meta_path = os.path.join(ep_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    cameras = tuple((meta.get("cameras") or {}).keys())
    if not cameras:
        raise ValueError(f"meta.json has no cameras: {ep_dir}")
    return cameras


def validate_frame_counts(ep_dir: str, cameras: tuple[str, ...]) -> int:
    counts = {
        camera: len(glob.glob(os.path.join(ep_dir, "frames", camera, "frame_*.jpg")))
        for camera in cameras
    }
    if not counts or len(set(counts.values())) != 1 or next(iter(counts.values())) == 0:
        raise ValueError(f"camera frame counts are missing or unequal: {counts}")
    return next(iter(counts.values()))


def make_videos(ep_dir: str) -> list[str]:
    ep_dir = os.path.abspath(ep_dir)
    if not os.path.isfile(os.path.join(ep_dir, "meta.json")):
        raise FileNotFoundError(f"not an episode dir: {ep_dir}")
    fps = episode_fps(ep_dir)
    cameras = episode_cameras(ep_dir)
    validate_frame_counts(ep_dir, cameras)
    outs = []
    for cam in cameras:
        cam_dir = os.path.join(ep_dir, "frames", cam)
        if not os.path.isdir(cam_dir):
            continue
        out = os.path.join(ep_dir, f"preview_{cam}.mp4")
        encode_one(cam_dir, out, fps)
        outs.append(out)
    grid_out = os.path.join(ep_dir, "preview_camera_grid.mp4")
    encode_grid(ep_dir, cameras, grid_out, fps)
    outs.append(grid_out)
    return outs


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    paths = []
    for arg in argv[1:]:
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        print("no episodes matched", file=sys.stderr)
        sys.exit(1)
    for ep in paths:
        if not os.path.isfile(os.path.join(ep, "meta.json")):
            print(f"[skip] incomplete episode (no meta.json): {ep}", flush=True)
            continue
        print(f"[video] {ep}", flush=True)
        for p in make_videos(ep):
            mb = os.path.getsize(p) / (1024 * 1024)
            print(f"  -> {os.path.basename(p)} ({mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv)
