#!/usr/bin/env python3
"""Closed-loop rollout: the fine-tuned pi0.5 policy drives the CRUZR MuJoCo sim.

The policy server (openpi serve_policy.py, config pi05_cruzr_ecu_lora) receives the same
observation schema the model was trained on (3 cams + 21D state) and returns an (16,18)
absolute action chunk (16 joint targets + diff-drive v/wz). Actions are applied through
cruzr_teleop's OWN runtime interface (qtgt / grip_cmd / base_vel / control_step) - the
exact code path used by both the human demo and the IK expert, base channel included.

Privileged info (ECU pose) is used ONLY for diagnostics/metrics, never sent to the policy.

Env:
  POLICY_HOST/POLICY_PORT   server address        (default 127.0.0.1:8731)
  ROLLOUT_STEPS             max recorded frames   (default 3600 = 120s @30fps)
  ROLLOUT_REPLAN            frames per re-plan    (default 8; 1 = replan every frame)
  ROLLOUT_OUT               output dir            (default out/policy_rollout)
  ROLLOUT_GPU               EGL render device     (default 3)

Run:
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 ROLLOUT_OUT=out/policy_rollout_try1 \
    /data1/hsr/tools/miniconda3/envs/mjx/bin/python scripts/ecu_policy_rollout.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get(  # override when openpi lives elsewhere
    "OPENPI_CLIENT_SRC", "/data1/hsr/openpi-main/packages/openpi-client/src"))

os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CRUZR_GRIP_CLOSE", "0.025")   # same gripper range as the expert data

_spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

import mujoco  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

m, d = ct.m, ct.d
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "8731"))
STEPS = int(os.environ.get("ROLLOUT_STEPS", "3600"))
REPLAN = int(os.environ.get("ROLLOUT_REPLAN", "8"))
SKIP = int(os.environ.get("ROLLOUT_SKIP", "0"))     # execute chunk[k+SKIP]: servo lead-ahead
OUT = os.path.join(HERE, "..", os.environ.get("ROLLOUT_OUT", "out/policy_rollout"))
PROMPT = "pick up the parts and place it down"
# ROLLOUT_STAGED=1: sub-goal FSM - switch the prompt when the (privileged) stage
# completion condition is met. Stage logic mirrors scripts/stage_split.py prompts.
STAGED = os.environ.get("ROLLOUT_STAGED", "0") == "1"
STAGE_PROMPTS = [
    "pick up the ECU from the stand",
    "carry the ECU and place it on the fixture",
    "pick up the ECU from the fixture",
    "carry the ECU and insert it into the rack bay",
]

JIG = ct.bid("jig")
JQ = m.jnt_qposadr[ct.jid("jig_free")]
CAMS = ["top_head", "hand_left", "hand_right"]

# ROLLOUT_SPAWN_SEED>0: randomize the ECU spawn with the SAME window the v3 training
# data used - the eval distribution must match the training distribution.
_sseed = int(os.environ.get("ROLLOUT_SPAWN_SEED", "0"))
if _sseed > 0:
    _rng = np.random.default_rng(_sseed)
    _jx = float(_rng.uniform(-0.009, 0.012))
    _jy = float(_rng.uniform(-0.070, 0.070))
    _jyaw = float(np.deg2rad(_rng.uniform(-12.0, 12.0)))
    d.qpos[JQ:JQ + 3] = [0.459 + _jx, -0.006 + _jy, 0.9228 + 0.003]
    d.qpos[JQ + 3:JQ + 7] = [np.cos(_jyaw / 2), 0.0, 0.0, np.sin(_jyaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    print(f"[rollout] spawn seed={_sseed} dx={_jx*1000:.0f}mm dy={_jy*1000:.0f}mm "
          f"dyaw={np.degrees(_jyaw):.1f}deg", flush=True)


def grip_frac(arm):
    """gripper open fraction from qpos (1=open, 0=closed), same as the recorder."""
    q = float(np.mean([d.qpos[a] for a in arm.grip_qadr]))
    return float(np.clip(1.0 - (q - ct.GRIP_OPEN) / (ct.GRIP_CLOSE - ct.GRIP_OPEN), 0.0, 1.0))


def state21():
    qL = [float(d.qpos[a]) for a in ct.L.qadr]
    qR = [float(d.qpos[a]) for a in ct.R.qadr]
    base = ct.base_pose()
    bvel = ct.base_velocity()
    return np.array(qL + qR + [grip_frac(ct.L), grip_frac(ct.R)]
                    + list(base) + list(bvel), dtype=np.float32)


class CamRig:
    def __init__(self):
        self.renderer = mujoco.Renderer(m, 480, 640)
        self.opt = mujoco.MjvOption()
        self.ids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}

    def shot(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.ids[name]
        self.renderer.update_scene(d, cam, self.opt)
        return self.renderer.render().copy()


def ecu_metrics():
    p = d.xpos[JIG]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, d.qpos[JQ + 3:JQ + 7])
    up = mat.reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return [float(p[0]), float(p[1]), float(p[2]), tilt]


def apply_action(a):
    a = np.asarray(a, dtype=float)
    ct.qtgt["l"][:] = np.clip(a[0:7], ct.L.lo, ct.L.hi)
    ct.qtgt["r"][:] = np.clip(a[7:14], ct.R.lo, ct.R.hi)
    # open-frac -> gripper ctrl (1=open at GRIP_OPEN, 0=closed at GRIP_CLOSE)
    ct.grip_cmd["l"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[14], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    ct.grip_cmd["r"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[15], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    ct.base_vel[:] = [float(np.clip(a[16], -0.4, 0.4)), float(np.clip(a[17], -0.6, 0.6))]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    rig = CamRig()
    print(f"[rollout] connecting to policy server ws://{HOST}:{PORT} ...", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)
    print("[rollout] connected; starting closed loop "
          f"(steps={STEPS}, replan={REPLAN})", flush=True)

    metrics = []
    chunk = None
    k = 0
    stage = 0
    stage_t0 = 0

    def stage_done(step):
        p = d.xpos[JIG]
        held = any(True for i in range(d.ncon)
                   if {d.contact[i].geom1, d.contact[i].geom2} &
                      {ct.gid("R_pad1"), ct.gid("R_pad2")} and
                      {d.contact[i].geom1, d.contact[i].geom2} &
                      {g for g in range(m.ngeom) if m.geom_bodyid[g] == JIG})
        if stage == 0:                        # lifted off the stand
            return held and p[2] > 0.955
        if stage == 1:                        # resting on the fixture bridge, released
            return (not held) and p[2] < 0.90 and abs(p[0] - 0.50) < 0.12 and abs(p[1] + 0.77) < 0.12
        if stage == 2:                        # lifted again
            return held and p[2] > 0.87
        return False
    import imageio.v2 as imageio
    video = imageio.get_writer(os.path.join(OUT, "rollout_top_head.mp4"), fps=30)
    try:
        for step in range(STEPS):
            top = rig.shot("top_head")
            if step % REPLAN == 0 or chunk is None or k >= len(chunk):
                obs = {
                    "observation/state": state21(),
                    "observation/image": top,
                    "observation/left_wrist_image": rig.shot("hand_left"),
                    "observation/right_wrist_image": rig.shot("hand_right"),
                    "prompt": STAGE_PROMPTS[stage] if STAGED else PROMPT,
                }
                chunk = np.asarray(client.infer(obs)["actions"])
                k = 0
            apply_action(chunk[min(k + SKIP, len(chunk) - 1)])
            k += 1
            for _ in range(2):                      # 1 recorded-frame period = 2 control steps
                ct.control_step(SUB)
            if STAGED and stage < 3 and stage_done(step):
                stage += 1
                stage_t0 = step
                chunk = None                   # force replan with the new prompt
                print(f"[stage] -> {stage+1}: '{STAGE_PROMPTS[stage]}' at t={step/30:.1f}s", flush=True)
            video.append_data(top)
            metrics.append([step] + ecu_metrics() + list(ct.base_pose())
                           + [float(ct.base_vel[0]), float(ct.base_vel[1])])
            if step % 150 == 0:
                e = metrics[-1]
                print(f"[t={step/30:5.1f}s] ecu=({e[1]:.3f},{e[2]:.3f},{e[3]:.3f}) tilt={e[4]:.1f} "
                      f"base=({e[5]:.2f},{e[6]:.2f},{e[7]:.2f}) cmd=({e[8]:+.2f},{e[9]:+.2f})", flush=True)
    finally:
        video.close()
        np.save(os.path.join(OUT, "metrics.npy"), np.asarray(metrics, dtype=np.float32))

    mt = np.asarray(metrics)
    print("\n=== ROLLOUT DIAGNOSTICS (privileged, not seen by policy) ===", flush=True)
    print(f"frames: {len(mt)} ({len(mt)/30:.0f}s)")
    print(f"ECU start ({mt[0,1]:.3f},{mt[0,2]:.3f},{mt[0,3]:.3f})  "
          f"end ({mt[-1,1]:.3f},{mt[-1,2]:.3f},{mt[-1,3]:.3f})")
    print(f"ECU max z: {mt[:,3].max():.3f} (spawn 0.923; >0.94 = lifted)")
    print(f"ECU net displacement: {np.linalg.norm(mt[-1,1:3]-mt[0,1:3]):.3f} m")
    print(f"base travelled: {np.sum(np.linalg.norm(np.diff(mt[:,5:7],axis=0),axis=1)):.2f} m, "
          f"final ({mt[-1,5]:.2f},{mt[-1,6]:.2f},{mt[-1,7]:.2f})")
    print(f"video: {os.path.abspath(os.path.join(OUT,'rollout_top_head.mp4'))}", flush=True)


if __name__ == "__main__":
    main()
