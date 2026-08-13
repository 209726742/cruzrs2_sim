#!/usr/bin/env python3
"""DAgger correction for the e2e grasp failure — NO cheating: corrections are REAL-PHYSICS,
force-gated grasps (identical mechanism to the training-data expert; no weld/teleport/attach),
recorded only as TRAINING labels. Eval stays pure-vision closed-loop (this script never runs at
eval time). Corrections that don't physically succeed are DROPPED, never fabricated.

Flow per seed:
  1. Randomized layout (same recipe as shelf_e2e_rollout).
  2. Run the v2 policy for LEAD seconds (it navigates to the rack + reaches, but doesn't grasp).
  3. INTERVENE: if the base is within REACH of the object's grasp standoff, the scripted expert
     (a) drives the base to the exact object-relative park pose, (b) blends the arm to home,
     (c) real-physics SE(3)-replay grasp, (d) checks grip_firm (both pads + >=1N/hand).
  4. MODE=smoke: just report whether the expert grasped. MODE=collect: also record the whole
     intervention (obs images + state22 + expert action) as an episode IF the grasp succeeded,
     for aggregation into the DAgger dataset.

Env: SEED, POLICY_PORT, MODE=smoke|collect, LEAD=45, OUT (collect only).
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get(  # override when openpi lives elsewhere
    "OPENPI_CLIENT_SRC", "/data1/hsr/openpi-main/packages/openpi-client/src"))

SEED = int(os.environ["SEED"])
rng = np.random.default_rng(SEED)
OBJ = "pillar"
MODE = os.environ.get("MODE", "smoke")
LEAD = float(os.environ.get("LEAD", "45"))

CART_NOM = np.array([-2.40, 0.0])
cart_xy = CART_NOM + np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.30, 0.30)])
obj_nom = np.array([0.58, 0.0])
obj_xy = obj_nom + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.30, 0.30)])
robot0 = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])

_tmpl = open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")).read()
_tmpl = re.sub(r'(<body name="shelf_cart" pos=")[^"]*(")',
               lambda mm: f'{mm.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{mm.group(2)}', _tmpl)
SCENE = os.path.join(ROOT, "assets", f"e2e_dagger_scene_{SEED}.xml")
open(SCENE, "w").write(_tmpl)
os.environ["TELEOP_SCENE_XML"] = SCENE
os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")

_spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)
import mujoco  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

m, d = ct.m, ct.d
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
DECIM = int(getattr(ct, "REC_DECIM", 2))
PROMPT = "pick up the steel pillar from the rack in front and place it on the second shelf of the cart"
REC_CAMS = ["head_stereo_l_shelf", "hand_left_shelf", "hand_right_shelf", "chassis_front"]  # 录4相机匹配v2数据集
POLICY_CAMS = ["head_stereo_l_shelf", "chassis_front", "hand_right_shelf"]  # 策略obs用3相机(训练契约)

BODIES = [i for i in range(m.nbody)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("_")[0] == OBJ]
OB = BODIES[0]
OBJ_GEOMS = {g for g in range(m.ngeom) if m.geom_bodyid[g] in BODIES}
PADG = {"r": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("R_pad1", "R_pad2")],
        "l": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("L_pad1", "L_pad2")]}
_FT = np.zeros(6)


def obj_pos():
    return np.mean([d.xpos[b] for b in BODIES], axis=0)


def grip_force(k):
    tot = 0.0
    for i in range(d.ncon):
        gs = {d.contact[i].geom1, d.contact[i].geom2}
        if gs & set(PADG[k]) and gs & OBJ_GEOMS:
            mujoco.mj_contactForce(m, d, i, _FT)
            tot += abs(_FT[0])
    return tot


def pad_contacts_both(k):
    seen = set()
    for i in range(d.ncon):
        gs = {d.contact[i].geom1, d.contact[i].geom2}
        if gs & OBJ_GEOMS:
            seen |= (gs & set(PADG[k]))
    return len(seen) == 2


def frames(n):
    for _ in range(n):
        ct.control_step(SUB)


# ---- nearest variant demo + FK grasp mount trajectory (same as expert) ----
VAR = {2: dict(obj=(0.66, 0.35), dz=0.00), 3: dict(obj=(0.60, -0.30), dz=0.00),
       4: dict(obj=(0.62, 0.15), dz=0.10), 5: dict(obj=(0.64, -0.20), dz=-0.10),
       6: dict(obj=(0.60, 0.20), dz=0.10)}
vsel = min(VAR, key=lambda v: abs(VAR[v]["obj"][1] - obj_xy[1]))
DEMO = os.path.join(ROOT, "out", "teleop", "demos", f"pillar_v{vsel}_refined")
dd = np.load(os.path.join(DEMO, "episode_data.npz"))
ge = json.load(open(os.path.join(DEMO, "refine.json")))["grasp_end"]
dact, dbase = dd["action"], dd["base"]
bmov = np.abs(dd["base_action"][:ge]).max(1)
gs = next((f for f in range(ge) if bmov[f:ge].max() < 0.02), 0)
demo_obj = np.array(VAR[vsel]["obj"])
d_obj = np.array([obj_xy[0] - demo_obj[0], obj_xy[1] - demo_obj[1], -VAR[vsel]["dz"]])

_q0, _v0 = d.qpos.copy(), d.qvel.copy()
MNT = {"l": [], "r": []}
for f in range(gs, ge):
    for i, adr in enumerate(ct.BQ):
        d.qpos[adr] = dbase[f, i]
    for k, A in (("l", ct.L), ("r", ct.R)):
        sl = slice(0, 7) if k == "l" else slice(7, 14)
        for j, a in enumerate(A.qadr):
            d.qpos[a] = dact[f, sl][j]
    mujoco.mj_kinematics(m, d)
    for k, A in (("l", ct.L), ("r", ct.R)):
        MNT[k].append((d.xpos[A.mount].copy(), d.xmat[A.mount].reshape(3, 3).copy()))
d.qpos[:], d.qvel[:] = _q0, _v0
mujoco.mj_forward(m, d)
park_grasp = dbase[gs].copy()
park_grasp[:2] += d_obj[:2]

# ---- randomize start ----
_oq = m.jnt_qposadr[m.body_jntadr[OB]]
d.qpos[_oq + 0] += obj_xy[0] - obj_nom[0]
d.qpos[_oq + 1] += obj_xy[1] - obj_nom[1]
for i, adr in enumerate(ct.BQ):
    d.qpos[adr] = robot0[i]
ct.base_tgt[:] = robot0
mujoco.mj_forward(m, d)
frames(30)
for li in range(m.nlight):
    m.light_pos[li] = m.light_pos[li] + rng.uniform(-0.4, 0.4, 3)
    m.light_diffuse[li] = np.clip(m.light_diffuse[li] * rng.uniform(0.7, 1.25), 0.05, 1.0)

Q_HOME_L = ct.qtgt["l"].copy()
Q_HOME_R = ct.qtgt["r"].copy()


def grip_frac(arm):
    q = float(np.mean([d.qpos[a] for a in arm.grip_qadr]))
    return float(np.clip(1.0 - (q - ct.GRIP_OPEN) / (ct.GRIP_CLOSE - ct.GRIP_OPEN), 0.0, 1.0))


class CamRig:
    def __init__(self):
        self.r = mujoco.Renderer(m, 224, 224)
        self.opt = mujoco.MjvOption()
        self.ids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in REC_CAMS}

    def shot(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.ids[name]
        self.r.update_scene(d, cam, self.opt)
        return self.r.render().copy()


RIG = CamRig()


def state22():
    qL = [float(d.qpos[a]) for a in ct.L.qadr]
    qR = [float(d.qpos[a]) for a in ct.R.qadr]
    x, y, yaw = ct.base_pose()
    c, s = np.cos(yaw), np.sin(yaw)

    def rel(t):
        dx, dy = t[0] - x, t[1] - y
        return [c * dx + s * dy, -s * dx + c * dy]
    v = ct.base_velocity()
    return np.array(qL + qR + [grip_frac(ct.L), grip_frac(ct.R)]
                    + rel(obj_xy) + rel(cart_xy) + [float(v[0]), float(v[1])], dtype=np.float32)


def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


REC = []          # (imgs dict, state22, action18) recorded during the expert intervention


def rec_step():
    if MODE == "collect":
        REC.append(({c: RIG.shot(c) for c in REC_CAMS}, state22(),
                    _last_action.copy()))


_last_action = np.zeros(18)


def go_to(tx, ty, tyaw, vmax=0.20, wmax=0.5, tol=0.02, max_f=3000):
    global _last_action
    for _ in range(max_f):
        x, y, yaw = ct.base_pose()
        dist = float(np.hypot(tx - x, ty - y))
        if dist < tol:
            break
        hdg = np.arctan2(ty - y, tx - x)
        e = _ang(hdg - yaw)
        back = abs(e) > np.pi / 2
        if back:
            e = _ang(hdg + np.pi - yaw)
        if abs(e) > 0.35:
            vfb, wz = 0.0, float(np.clip(2.0 * e, -wmax, wmax))
        else:
            vfb = float(np.clip(1.2 * dist, 0.05, vmax)) * (-1 if back else 1)
            wz = float(np.clip(1.8 * e, -wmax, wmax))
        ct.base_vel[:] = [vfb, wz]
        _last_action = np.concatenate([ct.qtgt["l"], ct.qtgt["r"],
                                       [grip_frac(ct.L), grip_frac(ct.R)], [vfb, wz]])
        for _ in range(DECIM):
            ct.control_step(SUB)
            rec_step()
    for _ in range(900):
        e = _ang(tyaw - ct.base_pose()[2])
        if abs(e) < 0.03:
            break
        wz = float(np.clip(2.0 * e, -0.45, 0.45))
        ct.base_vel[:] = [0.0, wz]
        _last_action = np.concatenate([ct.qtgt["l"], ct.qtgt["r"],
                                       [grip_frac(ct.L), grip_frac(ct.R)], [0.0, wz]])
        for _ in range(DECIM):
            ct.control_step(SUB)
            rec_step()
    ct.base_vel[:] = 0.0


def arm_home_blend(nf=40):
    global _last_action
    cl, cr = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
    for i in range(nf):
        ss = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / nf)
        ct.qtgt["l"][:] = cl + (Q_HOME_L - cl) * ss
        ct.qtgt["r"][:] = cr + (Q_HOME_R - cr) * ss
        ct.grip_cmd["l"] = ct.GRIP_OPEN
        ct.grip_cmd["r"] = ct.GRIP_OPEN
        ct.base_vel[:] = 0.0
        _last_action = np.concatenate([ct.qtgt["l"], ct.qtgt["r"], [0.0, 0.0], [0.0, 0.0]])
        frames(DECIM)
        rec_step()


def expert_grasp():
    """Real-physics SE(3)-replay grasp from home at park_grasp. Records (obs, action)."""
    global _last_action
    for idx, f in enumerate(range(gs, ge)):
        q0s, v0s = d.qpos.copy(), d.qvel.copy()
        tq = {}
        for k, A in (("l", ct.L), ("r", ct.R)):
            pm, Rm = MNT[k][idx]
            for j, a in enumerate(A.qadr):
                d.qpos[a] = ct.qtgt[k][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(A, pm + d_obj, Rm, iters=15, w=0.6)
            tq[k] = np.array([d.qpos[a] for a in A.qadr])
        d.qpos[:], d.qvel[:] = q0s, v0s
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["l"][:] = tq["l"]
        ct.qtgt["r"][:] = tq["r"]
        gl = float((1 - dact[f, 14]) * 0.025)
        gr = float((1 - dact[f, 15]) * 0.025)
        ct.grip_cmd["l"] = gl
        ct.grip_cmd["r"] = gr
        ct.base_vel[:] = 0.0
        _last_action = np.concatenate([tq["l"], tq["r"], [dact[f, 14], dact[f, 15]], [0.0, 0.0]])
        frames(DECIM)
        rec_step()


def main():
    global _last_action
    client = websocket_client_policy.WebsocketClientPolicy(
        host="127.0.0.1", port=int(os.environ.get("POLICY_PORT", "8731")))
    # 1) run policy for LEAD seconds
    chunk, k = None, 0
    for step in range(int(LEAD * 30)):
        if step % 8 == 0 or chunk is None or k >= len(chunk):
            obs = {"observation/state": state22(), "observation/image": RIG.shot(POLICY_CAMS[0]),
                   "observation/left_wrist_image": RIG.shot(POLICY_CAMS[1]),
                   "observation/right_wrist_image": RIG.shot(POLICY_CAMS[2]), "prompt": PROMPT}
            chunk = np.asarray(client.infer(obs)["actions"])
            k = 0
        a = np.asarray(chunk[k], float)
        k += 1
        ct.qtgt["l"][:] = np.clip(a[0:7], ct.L.lo, ct.L.hi)
        ct.qtgt["r"][:] = np.clip(a[7:14], ct.R.lo, ct.R.hi)
        ct.grip_cmd["l"] = ct.GRIP_OPEN + (1 - np.clip(a[14], 0, 1)) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
        ct.grip_cmd["r"] = ct.GRIP_OPEN + (1 - np.clip(a[15], 0, 1)) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
        ct.base_vel[:] = [float(np.clip(a[16], -0.4, 0.4)), float(np.clip(a[17], -0.6, 0.6))]
        for _ in range(2):
            ct.control_step(SUB)
    bx, by, byaw = ct.base_pose()
    dist_park = float(np.hypot(park_grasp[0] - bx, park_grasp[1] - by))
    reachable = dist_park < 0.5 and abs(_ang(byaw)) < 0.6 and obj_pos()[2] > 0.5
    print(f"[dagger] seed={SEED} 策略后 base=({bx:+.2f},{by:+.2f},{byaw:+.2f}) "
          f"park=({park_grasp[0]:+.2f},{park_grasp[1]:+.2f}) 距park={dist_park:.2f} reachable={reachable}", flush=True)
    if not reachable:
        print(f"[dagger] seed={SEED} RESULT SKIP (策略漂移过远/失控, 专家不硬凑)", flush=True)
        os.remove(SCENE)
        return
    # 2) expert intervention (recorded from here)
    go_to(park_grasp[0], park_grasp[1], park_grasp[2])
    arm_home_blend()
    expert_grasp()
    frames(10)
    fR, fL = grip_force("r"), grip_force("l")
    ok = pad_contacts_both("r") and pad_contacts_both("l") and fR >= 1.0 and fL >= 1.0
    print(f"[dagger] seed={SEED} RESULT {'GRASP_OK' if ok else 'GRASP_FAIL'} "
          f"fR={fR:.1f}N fL={fL:.1f}N recframes={len(REC)}", flush=True)
    if MODE == "collect" and ok:
        save_correction()
    os.remove(SCENE)


def save_correction():
    out = os.environ["OUT"]
    os.makedirs(out, exist_ok=True)
    n = len(REC)
    st = np.stack([r[1] for r in REC]).astype(np.float32)
    ac = np.stack([r[2] for r in REC]).astype(np.float32)
    np.savez(os.path.join(out, f"corr_{SEED:06d}.npz"),
             state22=st, action18=ac, obj_xy=obj_xy, cart_xy=cart_xy)
    import imageio.v2 as imageio
    for c in REC_CAMS:
        vd = os.path.join(out, f"corr_{SEED:06d}_{c}")
        os.makedirs(vd, exist_ok=True)
        for i, r in enumerate(REC):
            imageio.imwrite(os.path.join(vd, f"frame_{i:06d}.jpg"), r[0][c])
    print(f"[dagger] seed={SEED} 纠正已存 {n} 帧 -> {out}", flush=True)


if __name__ == "__main__":
    main()
