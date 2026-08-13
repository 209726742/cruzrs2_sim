#!/usr/bin/env python3
"""Replay a recorded shelf_e2e episode and render a third-person overview mp4.

Writes preview_3rd.mp4, reconstructed from meta.json + episode_data.npz. For the four
onboard cameras use scripts/collection/shelf_e2e_make_videos.py, which encodes the recorded JPEGs
directly (per-camera mp4 plus a 2x2 grid).

Object poses come from object_poses.npz when the collector wrote one. Without it the
parts can only be pinned to the gripper pads and freeze where they were released, which
makes a correct placement look like it is floating -- do not judge placement from a
video that lacks that file.

Usage:
  MUJOCO_GL=egl python scripts/collection/shelf_e2e_replay_3rd.py out/teleop/shelf_e2e_dual/shelf_e2e_dual_000001
  MUJOCO_GL=egl python scripts/collection/shelf_e2e_replay_3rd.py out/teleop/shelf_e2e_dual/shelf_e2e_dual_*
Env:
  VCAM=lookat_x,y,z,distance,azimuth,elevation   (default tracks robot; set FOLLOW=0 for fixed)
  FOLLOW=1 (default) camera lookat follows base
  FPS=15  STRIDE=2  (skip frames for faster encode)
"""
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from shelf_e2e_flex_state import (  # noqa: E402
    FLEX_TASK_VERSION,
    RIGID_TASK_VERSION,
    load_internal_state,
    object_state_contract,
    restore_internal_state,
)
from shelf_e2e_objects import object_info  # noqa: E402

OBJECT_ORDER = ("pillar", "strip")
FOLLOW = os.environ.get("FOLLOW", "1") == "1"
STRIDE = max(1, int(os.environ.get("STRIDE", "2")))
FPS = int(os.environ.get("FPS", "15"))
GRIP_CLOSED_THRESHOLD = 0.5
# default: workshop overview; FOLLOW overrides lookat to robot each frame
_VCAM = [float(v) for v in os.environ.get("VCAM", "-0.9,0.0,1.0,5.2,55,-18").split(",")]


def _load_ct(scene_xml: str):
    os.environ["TELEOP_SCENE_XML"] = scene_xml
    os.environ.setdefault("TELEOP_HOME", "droop")
    os.environ.setdefault("MUJOCO_GL", "egl")
    # fresh import each episode so scene reload works
    for name in list(sys.modules):
        if name == "cruzr_teleop" or name.startswith("cruzr_teleop."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "cruzr_teleop", os.path.join(CORE_DIR, "cruzr_teleop.py")
    )
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    import mujoco
    return ct, mujoco


def _build_scene(cart_xy: np.ndarray, seed: int, task_version: str) -> str:
    scene_dir = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
    template_names = {
        RIGID_TASK_VERSION: "template_pillar_v1.xml",
        FLEX_TASK_VERSION: "template_strip_cable_v1.xml",
    }
    template_path = os.path.join(ROOT, "assets", "e2e", template_names[task_version])
    if not os.path.isfile(template_path):
        raise RuntimeError(
            f"replay template for {task_version} is unavailable: {template_path}; "
            "do not replay a flexible episode with the rigid template"
        )
    tmpl = open(template_path).read()
    tmpl = re.sub(
        r'(<body name="shelf_cart" pos=")[^"]*(")',
        lambda m: f'{m.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{m.group(2)}',
        tmpl,
    )
    path = os.path.join(scene_dir, f"e2e_replay_scene_{seed}.xml")
    open(path, "w").write(tmpl)
    return path


def _frac_to_q(frac: float, open_q: float, close_q: float) -> float:
    f = float(np.clip(frac, 0.0, 1.0))
    return open_q + (1.0 - f) * (close_q - open_q)


def replay_one(ep_dir: str) -> str:
    ep_dir = os.path.abspath(ep_dir)
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    em = meta.get("episode_metadata") or {}
    task_version = em.get("task_version")
    if task_version not in (RIGID_TASK_VERSION, FLEX_TASK_VERSION):
        raise RuntimeError(f"not a dual shelf_e2e episode: {ep_dir}")
    z = np.load(os.path.join(ep_dir, "episode_data.npz"))
    state, action = z["state"].astype(np.float32), z["action"].astype(np.float32)
    base = z["base"].astype(np.float32)
    n = len(state)
    cart_xy = np.asarray(em["cart_xy"], np.float32)
    object_xy = {
        "pillar": np.asarray(em["pillar_xy"], np.float32),
        "strip": np.asarray(em["strip_xy"], np.float32),
    }
    nominal = {"pillar": np.array([0.58, 0.0005]), "strip": np.array([1.05, 0.0])}
    seed = int(em.get("seed", 0))

    scene = _build_scene(cart_xy, seed, task_version)
    ct, mujoco = _load_ct(scene)
    m, d = ct.m, ct.d
    body, qadr, object_infos = {}, {}, {}
    for name in OBJECT_ORDER:
        object_infos[name] = object_info(m, name)
        body[name] = object_infos[name]["body"]
        qadr[name] = object_infos[name]["free_qpos_adr"]
        d.qpos[qadr[name]] += object_xy[name][0] - nominal[name][0]
        d.qpos[qadr[name] + 1] += object_xy[name][1] - nominal[name][1]
    model_task_version, _ = object_state_contract(object_infos)
    if model_task_version != task_version:
        raise RuntimeError(
            f"episode/model task version mismatch: {task_version} != {model_task_version}"
        )
    for i, adr in enumerate(ct.BQ):
        d.qpos[adr] = float(base[0, i])
    mujoco.mj_forward(m, d)

    pad_ids = {
        "r": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("R_pad1", "R_pad2")],
        "l": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("L_pad1", "L_pad2")],
    }
    active = 0
    held = False
    held_hands = frozenset()
    hold_off = None

    # Real object poses if the collector logged them; otherwise fall back to pinning the
    # part to the pads, which cannot show what happens after release.
    pose_log = None
    pose_path = os.path.join(ep_dir, "object_poses.npz")
    if os.path.isfile(pose_path):
        pose_log = np.load(pose_path)["pose"].astype(np.float32)
        if len(pose_log) < n:
            print(f"  [warn] object_poses has {len(pose_log)} of {n} frames", flush=True)
    if task_version == FLEX_TASK_VERSION and (
        pose_log is None or pose_log.shape != (n, 7 * len(OBJECT_ORDER))
    ):
        raise RuntimeError(f"flex replay requires object_poses.npz with shape ({n}, 14)")
    internal_log = load_internal_state(ep_dir, em, n)

    import imageio.v2 as imageio
    out_mp4 = os.path.join(ep_dir, "preview_3rd.mp4")
    writer = imageio.get_writer(out_mp4, fps=FPS)
    ren = mujoco.Renderer(m, 544, 960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = _VCAM[0:3]
    cam.distance, cam.azimuth, cam.elevation = _VCAM[3], _VCAM[4], _VCAM[5]
    opt = mujoco.MjvOption()

    for i in range(0, n, STRIDE):
        st, b = state[i], base[i]
        for j, adr in enumerate(ct.L.qadr):
            d.qpos[adr] = float(st[j])
        for j, adr in enumerate(ct.R.qadr):
            d.qpos[adr] = float(st[7 + j])
        ql = _frac_to_q(st[14], ct.GRIP_OPEN, ct.GRIP_CLOSE)
        qr = _frac_to_q(st[15], ct.GRIP_OPEN, ct.GRIP_CLOSE)
        for adr in ct.L.grip_qadr:
            d.qpos[adr] = ql
        for adr in ct.R.grip_qadr:
            d.qpos[adr] = qr
        for j, adr in enumerate(ct.BQ):
            d.qpos[adr] = float(b[j])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        if pose_log is not None:
            row = pose_log[min(i, len(pose_log) - 1)]
            for k, name in enumerate(OBJECT_ORDER):
                d.qpos[qadr[name]:qadr[name] + 7] = row[7 * k:7 * k + 7]
            if internal_log is not None:
                restore_internal_state(d, object_infos["strip"], internal_log[i])
            mujoco.mj_forward(m, d)
            if FOLLOW:
                cam.lookat[:] = [float(b[0]), float(b[1]), 0.9]
            ren.update_scene(d, cam, opt)
            writer.append_data(ren.render())
            continue

        closed_hands = frozenset(
            hand for hand, col in (("l", 14), ("r", 15))
            if float(action[i, col]) < GRIP_CLOSED_THRESHOLD
        )
        if closed_hands and active < len(OBJECT_ORDER):
            name = OBJECT_ORDER[active]
            mid = np.mean([
                np.mean([d.geom_xpos[g] for g in pad_ids[hand]], axis=0)
                for hand in closed_hands
            ], axis=0)
            if not held and np.linalg.norm(d.xpos[body[name]] - mid) < 0.45:
                held = True
                held_hands = closed_hands
                hold_off = d.xpos[body[name]].copy() - mid
            elif held and closed_hands != held_hands:
                held_hands = closed_hands
                hold_off = d.xpos[body[name]].copy() - mid
            if held:
                d.qpos[qadr[name]:qadr[name] + 3] = mid + hold_off
                mujoco.mj_forward(m, d)
        elif not closed_hands and held:
            held = False
            held_hands = frozenset()
            hold_off = None
            active += 1

        if FOLLOW:
            cam.lookat[:] = [float(b[0]), float(b[1]), 0.9]
        ren.update_scene(d, cam, opt)
        writer.append_data(ren.render())

    writer.close()
    ren.close()
    try:
        os.remove(scene)
    except OSError:
        pass
    return out_mp4


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
            print(f"[skip] {ep}", flush=True)
            continue
        print(f"[3rd] {ep}", flush=True)
        out = replay_one(ep)
        mb = os.path.getsize(out) / (1024 * 1024)
        print(f"  -> {os.path.basename(out)} ({mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv)
