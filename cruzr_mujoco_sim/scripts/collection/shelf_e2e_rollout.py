#!/usr/bin/env python3
"""END-TO-END rollout: one pi0.5 policy drives EVERYTHING (base velocity + both arms + grippers).

Mirrors the selected e2e collection profile exactly (strict_v1 by default;
sdk_recovery_v1 uses stereo_left, waist_front, chassis_front):
  state18 : 16 arm/gripper + base velocity (2), with no MuJoCo-only object/cart truth
  action18: 14 arm joint targets + 2 gripper cmds + base (v_fwd, wz) -- base channel EXECUTED.
Scene: same per-seed generation as shelf_e2e_dual_expert.py (template + randomized cart/objects/start),
so eval seeds reproduce the exact training-style layouts (use unseen seeds for eval).

Env: SEED (required; controls layout), CKPT via policy server POLICY_HOST/PORT (serve_ckpt.sh),
     ROLLOUT_OUT, ROLLOUT_STEPS (default 7200 = 240s).
"""
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
sys.path.insert(0, os.environ.get(  # override when openpi lives elsewhere
    "OPENPI_CLIENT_SRC", "/data1/hsr/openpi-main/packages/openpi-client/src"))

SEED = int(os.environ["SEED"])
rng = np.random.default_rng(SEED)

from cruzr_s2_sdk_contract import (  # noqa: E402
    SDK_BASE_V_FWD_RANGE_M_S,
    SDK_BASE_WZ_RANGE_RAD_S,
    SDK_COLLECTION_PROFILE,
    SDK_DOC_REVISION,
    clip_arm_target_to_operational_limits,
    rate_limit_arm_target,
)
from shelf_e2e_profiles import (  # noqa: E402
    collection_cameras,
    normalize_collection_profile,
)

COLLECTION_PROFILE = normalize_collection_profile(
    os.environ.get("E2E_COLLECTION_PROFILE")
)
SDK_RECOVERY = COLLECTION_PROFILE == SDK_COLLECTION_PROFILE
CAMS = collection_cameras(COLLECTION_PROFILE)

# ---- identical layout randomization to the expert (keep in sync!) ----
CART_NOM = np.array([-2.40, 0.0])
PILLAR_NOM = np.array([0.58, 0.0005])
STRIP_NOM = np.array([1.05, 0.0])
cart_xy = CART_NOM + np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.30, 0.30)])
rack_y = rng.uniform(-0.24, 0.24)
pillar_xy = PILLAR_NOM + np.array([rng.uniform(-0.04, 0.04), rack_y + rng.uniform(-0.035, 0.035)])
strip_xy = STRIP_NOM + np.array([rng.uniform(-0.03, 0.03), rack_y + rng.uniform(-0.035, 0.035)])
robot0 = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])

SCENE_DIR = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
_tmpl = open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")).read()
_tmpl = re.sub(r'(<body name="shelf_cart" pos=")[^"]*(")',
               lambda m: f'{m.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{m.group(2)}', _tmpl)
SCENE = os.path.join(SCENE_DIR, f"e2e_eval_scene_{SEED}.xml")
open(SCENE, "w").write(_tmpl)
os.environ["TELEOP_SCENE_XML"] = SCENE
os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")

_spec = importlib.util.spec_from_file_location(
    "cruzr_teleop", os.path.join(CORE_DIR, "cruzr_teleop.py")
)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)
import mujoco  # noqa: E402
from teleop_timing import CumulativeSubstepScheduler  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from shelf_e2e_contract import (  # noqa: E402
    make_state,
    validate_action_chunk,
    validate_policy_observation,
)
from shelf_e2e_objects import object_info  # noqa: E402
m, d = ct.m, ct.d
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
SUBSTEP_SCHEDULER = (
    CumulativeSubstepScheduler(ct.TARGET_FPS, m.opt.timestep)
    if SDK_RECOVERY else None
)
OUT = os.path.join(ROOT, os.environ.get("ROLLOUT_OUT", f"out/rollout/shelf_e2e_dual/e2e_dual_roll_{SEED}"))
STEPS = int(os.environ.get("ROLLOUT_STEPS", "7200"))
REPLAN = int(os.environ.get("ROLLOUT_REPLAN", "8"))
PROMPT = "move the steel pillar to the middle shelf of the cart, then move the rubber strip to the top shelf"

OBJECTS = {}
for name in ("pillar", "strip"):
    OBJECTS[name] = object_info(m, name)
SHELF = {
    "pillar": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cart_shelf1"),
    "strip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cart_shelf3"),
}
PADG = {"r": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("R_pad1", "R_pad2")],
        "l": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("L_pad1", "L_pad2")]}
_FT = np.zeros(6)


def control_once():
    substeps = (
        SUBSTEP_SCHEDULER.next_substeps()
        if SUBSTEP_SCHEDULER is not None else SUB
    )
    ct.control_step(substeps)


def obj_pos(name):
    return d.xpos[OBJECTS[name]["body"]].copy()


def pair_force(ga, gb):
    total = 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 in ga and g2 in gb) or (g2 in ga and g1 in gb):
            mujoco.mj_contactForce(m, d, i, _FT)
            total += abs(_FT[0])
    return total


def grip_force(hand, name):
    return pair_force(set(PADG[hand]), OBJECTS[name]["geoms"])


def in_assigned_layer(name):
    center = obj_pos(name)
    bounds = [ct.geom_aabb(g) for g in OBJECTS[name]["geoms"]]
    bottom = min(x[0][2] for x in bounds)
    slo, shi = ct.geom_aabb(SHELF[name])
    support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
    released = grip_force("l", name) < 0.5 and grip_force("r", name) < 0.5
    return (slo[0] - 0.04 <= center[0] <= shi[0] + 0.04
            and slo[1] - 0.04 <= center[1] <= shi[1] + 0.04
            and abs(bottom - shi[2]) < 0.08 and support >= 1.0 and released)


# ---- randomized start state (same recipe as the expert) ----
for name, xy, nominal in (("pillar", pillar_xy, PILLAR_NOM), ("strip", strip_xy, STRIP_NOM)):
    qadr = OBJECTS[name]["free_qpos_adr"]
    d.qpos[qadr] += xy[0] - nominal[0]
    d.qpos[qadr + 1] += xy[1] - nominal[1]
for i, adr in enumerate(ct.BQ):
    d.qpos[adr] = robot0[i]
ct.base_tgt[:] = robot0
mujoco.mj_forward(m, d)
for _ in range(30):
    control_once()
for li in range(m.nlight):
    m.light_pos[li] = m.light_pos[li] + rng.uniform(-0.4, 0.4, 3)
    m.light_diffuse[li] = np.clip(m.light_diffuse[li] * rng.uniform(0.7, 1.25), 0.05, 1.0)


def grip_frac(arm):
    q = float(np.mean([d.qpos[a] for a in arm.grip_qadr]))
    return float(np.clip(1.0 - (q - ct.GRIP_OPEN) / (ct.GRIP_CLOSE - ct.GRIP_OPEN), 0.0, 1.0))


def state18():
    qL = [float(d.qpos[a]) for a in ct.L.qadr]
    qR = [float(d.qpos[a]) for a in ct.R.qadr]
    v = ct.base_velocity()
    return make_state(
        qL + qR + [grip_frac(ct.L), grip_frac(ct.R)],
        [float(v[0]), float(v[1])],
    )


class CamRig:
    def __init__(self):
        self.renderer = mujoco.Renderer(m, 224, 224)   # MATCH TRAINING: recorder saved 224x224 square; 480x640 skews aspect -> OOD images
        self.opt = mujoco.MjvOption()
        self.ids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}

    def shot(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.ids[name]
        self.renderer.update_scene(d, cam, self.opt)
        return self.renderer.render().copy()


def apply_action(a):
    a = np.asarray(a, dtype=float)
    target_l = np.clip(a[0:7], ct.L.lo, ct.L.hi)
    target_r = np.clip(a[7:14], ct.R.lo, ct.R.hi)
    if SDK_RECOVERY:
        target_l = rate_limit_arm_target(ct.qtgt["l"], target_l)
        target_r = rate_limit_arm_target(ct.qtgt["r"], target_r)
        operational_target = clip_arm_target_to_operational_limits(
            np.concatenate((target_l, target_r))
        )
        target_l = operational_target[:7]
        target_r = operational_target[7:14]
    ct.qtgt["l"][:] = target_l
    ct.qtgt["r"][:] = target_r
    ct.grip_cmd["l"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[14], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    ct.grip_cmd["r"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[15], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    v_range = SDK_BASE_V_FWD_RANGE_M_S if SDK_RECOVERY else (-0.4, 0.4)
    wz_range = SDK_BASE_WZ_RANGE_RAD_S if SDK_RECOVERY else (-0.6, 0.6)
    ct.base_vel[:] = [
        float(np.clip(a[16], *v_range)),
        float(np.clip(a[17], *wz_range)),
    ]


def main():
    os.makedirs(OUT, exist_ok=True)
    rig = CamRig()
    import imageio.v2 as imageio
    tp = imageio.get_writer(os.path.join(OUT, "e2e_3rd.mp4"), fps=20)
    tp_ren = mujoco.Renderer(m, 540, 960)
    tp_cam = mujoco.MjvCamera()
    tp_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _vc = [float(v) for v in os.environ.get("VCAM", "-0.9,0,1.0,4.8,52,-14").split(",")]
    tp_cam.lookat[:] = _vc[0:3]
    tp_cam.distance = _vc[3]
    tp_cam.azimuth = _vc[4]
    tp_cam.elevation = _vc[5]
    tp_opt = mujoco.MjvOption()
    client = websocket_client_policy.WebsocketClientPolicy(
        host=os.environ.get("POLICY_HOST", "127.0.0.1"),
        port=int(os.environ.get("POLICY_PORT", "8731")))
    print(f"[e2e-roll] seed={SEED} pillar=({pillar_xy[0]:+.2f},{pillar_xy[1]:+.2f}) "
          f"strip=({strip_xy[0]:+.2f},{strip_xy[1]:+.2f}) cart=({cart_xy[0]:+.2f},{cart_xy[1]:+.2f})",
          flush=True)
    z0 = {name: float(obj_pos(name)[2]) for name in OBJECTS}
    placed = {"pillar": False, "strip": False}
    grasped = {"pillar": False, "strip": False}
    ok = False
    chunk, k = None, 0
    try:
        for step in range(STEPS):
            if step % REPLAN == 0 or chunk is None or k >= len(chunk):
                obs = {"observation/state": state18(),
                       "observation/image": rig.shot(CAMS[0]),
                       "observation/left_wrist_image": rig.shot(CAMS[1]),
                       "observation/right_wrist_image": rig.shot(CAMS[2]),
                       "prompt": PROMPT}
                validate_policy_observation(obs)
                chunk = validate_action_chunk(client.infer(obs)["actions"])
                k = 0
            apply_action(chunk[k])
            k += 1
            for _ in range(2):
                control_once()
            if step % 3 == 0:
                tp_ren.update_scene(d, tp_cam, tp_opt)
                tp.append_data(tp_ren.render())
            for name in OBJECTS:
                if (not grasped[name] and grip_force("r", name) >= 1.0
                        and grip_force("l", name) >= 1.0 and obj_pos(name)[2] > z0[name] + 0.06):
                    grasped[name] = True
                    print(f"[e2e-roll] {name} grasped+lifted at t={step/30:.1f}s", flush=True)
                if not placed[name] and in_assigned_layer(name):
                    placed[name] = True
                    print(f"[e2e-roll] {name} PLACED at t={step/30:.1f}s", flush=True)
            if all(placed.values()):
                for _ in range(60):
                    control_once()
                ok = all(in_assigned_layer(name) for name in OBJECTS)
                if ok:
                    break
            if step % 300 == 0:
                pp, sp, bp = obj_pos("pillar"), obj_pos("strip"), ct.base_pose()
                print(f"  t={step/30:5.1f}s pillar=({pp[0]:+.2f},{pp[1]:+.2f},{pp[2]:.2f}) "
                      f"strip=({sp[0]:+.2f},{sp[1]:+.2f},{sp[2]:.2f}) "
                      f"base=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f}) placed={placed}", flush=True)
    finally:
        tp.close()
        try:
            os.remove(SCENE)
        except OSError:
            pass
    final_pos = {name: [round(float(x), 3) for x in obj_pos(name)] for name in OBJECTS}
    json.dump({"seed": SEED, "placed": placed, "success": bool(ok), "grasped": grasped,
               "collection_profile": COLLECTION_PROFILE, "cameras": list(CAMS),
               "sdk_document_revision": SDK_DOC_REVISION if SDK_RECOVERY else None,
               "pillar_xy": pillar_xy.tolist(), "strip_xy": strip_xy.tolist(),
               "cart_xy": cart_xy.tolist(), "final_pos": final_pos},
              open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(f"[e2e-roll] RESULT {'SUCCESS' if ok else 'FAIL'} placed={placed} grasped={grasped}", flush=True)



if __name__ == "__main__":
    main()
