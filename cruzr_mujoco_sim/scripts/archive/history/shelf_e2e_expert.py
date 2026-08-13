#!/usr/bin/env python3
"""END-TO-END data expert for the shelf task (isolated line, prefix shelf_e2e_*).

Fixes the three foundations that sank every end-to-end BC attempt (see
docs/VLA路线复盘与方向决策_2026-07-22.md §7.6):
  ① base view    -> records chassis_front (the robot model always had it; policy can now SEE the road)
  ② base data    -> the base path is COMPUTED per episode from the actual object/cart pose (continuous,
                    not 5 replayed scripts) by a CLOSED-LOOP go_to controller, with random mid-drive
                    "kicks" (pose+servo-target shifted together = wheel slip) that the controller then
                    corrects -> the data contains "drifted -> steer back" demonstrations.
  ③ relative task-> robot start pose, object x/y and cart x/y are ALL randomized, so absolute workshop
                    coordinates carry no signal; the prompt describes the task relative to the robot.

Arms: world-pose SE(3) transform of the nearest variant demo -- offline-FK the demo's mount
trajectory (demo joints @ demo base pose), add the object/cart displacement, IK under the CURRENT
base pose each frame. The arm therefore absorbs both the object offset AND the parking error left
by the closed-loop base. Gates: grip_firm (both pads + >=1N/hand) and strict in-region placement.

Isolation: touches NOTHING shared. Scene = per-seed XML generated from assets/e2e/template_pillar_v1.xml
(cart body repositioned; include/mesh paths absolutized). New output prefix shelf_e2e_*.
The final placement gate requires the pillar to stay in the cart, released by both hands and
supported by the cart for 0.5 s; entering the cart region while still held is not a success.

Env: SEED (required), EXPERT_OUT, OBJ=pillar, E2E_KICKS=2, E2E_SCENE_DIR (default scratch).
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

SEED = int(os.environ["SEED"])
rng = np.random.default_rng(SEED)
OBJ = os.environ.get("OBJ", "pillar")
assert OBJ == "pillar", "e2e line starts pillar-only"

# ---------------- per-seed scene from the ISOLATED template (never the shared xml) ----------------
CART_NOM = np.array([-2.40, 0.0])
cart_xy = CART_NOM + np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.30, 0.30)])
obj_nom = np.array([0.58, 0.0])
obj_xy = obj_nom + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.30, 0.30)])
robot0 = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])

# per-seed scene MUST live in assets/ (the robot include resolves its STL meshes relative to the
# MAIN xml's directory). Own filename per seed -> no shared file touched, no worker race.
SCENE_DIR = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
os.makedirs(SCENE_DIR, exist_ok=True)
_tmpl = open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")).read()
_tmpl = re.sub(r'(<body name="shelf_cart" pos=")[^"]*(")',
               lambda m: f'{m.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{m.group(2)}', _tmpl)
SCENE = os.path.join(SCENE_DIR, f"e2e_scene_{SEED}.xml")
open(SCENE, "w").write(_tmpl)
os.environ["TELEOP_SCENE_XML"] = SCENE
os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("REC_CAMS", "head_stereo_l_shelf,hand_left_shelf,hand_right_shelf,chassis_front")
os.environ.setdefault("REC_PROMPT",
                      "pick up the steel pillar from the rack in front and place it on the second shelf of the cart")

_spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)
import mujoco  # noqa: E402

m, d = ct.m, ct.d
# The shared recorder currently saves 480x640 (another line's need) -> ~500MB/episode, which is
# what filled the disk mid-batch (5.3G left, 546 crashed episodes). Patch the MODULE VARIABLE at
# runtime (capture() reads ct.REC_WH each call) back to the 224x224 the training pipeline uses --
# no shared file edited. ~85MB/episode.
ct.REC_WH = (224, 224)
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
DECIM = int(getattr(ct, "REC_DECIM", 2))
OUT = os.path.join(ROOT, os.environ.get("EXPERT_OUT", f"out/teleop/shelf_e2e/shelf_e2e_{OBJ}_{SEED:06d}"))

BODIES = [i for i in range(m.nbody)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("_")[0] == OBJ]
OB = BODIES[0]
OBJ_GEOMS = {g for g in range(m.ngeom) if m.geom_bodyid[g] in BODIES}
CART_GEOMS = {g for g in range(m.ngeom)
              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
              == "shelf_cart"}
PADG = {"r": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("R_pad1", "R_pad2")],
        "l": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("L_pad1", "L_pad2")]}
_FT = np.zeros(6)
_CD = cart_xy - CART_NOM
REGION = dict(x=(-2.73 + _CD[0], -2.07 + _CD[0]), y=(-0.91 + _CD[1], 0.92 + _CD[1]), z=0.903)

FAIL = []


def obj_pos():
    return np.mean([d.xpos[b] for b in BODIES], axis=0)


def obj_extent():
    ps = np.array([d.geom_xpos[g] for g in OBJ_GEOMS])
    return ps.min(0), ps.max(0)


def in_region():
    lo, hi = obj_extent()
    return (REGION["x"][0] - 0.04 <= lo[0] and hi[0] <= REGION["x"][1] + 0.04 and
            REGION["y"][0] - 0.04 <= lo[1] and hi[1] <= REGION["y"][1] + 0.04 and
            abs(obj_pos()[2] - REGION["z"]) < 0.10)


def pair_force(ga, gb):
    tot = 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 in ga and g2 in gb) or (g2 in ga and g1 in gb):
            mujoco.mj_contactForce(m, d, i, _FT)
            tot += abs(_FT[0])
    return tot


def grip_force(k):
    return pair_force(set(PADG[k]), OBJ_GEOMS)


def placement_evidence():
    p = obj_pos()
    f_r, f_l = grip_force("r"), grip_force("l")
    support = pair_force(OBJ_GEOMS, CART_GEOMS)
    return {
        "version": 2,
        "in_region": bool(in_region()),
        "released": bool(f_r < 0.5 and f_l < 0.5),
        "supported": bool(support >= 1.0),
        "grip_force_right_n": round(float(f_r), 3),
        "grip_force_left_n": round(float(f_l), 3),
        "cart_support_force_n": round(float(support), 3),
        "object_position": [round(float(x), 4) for x in p],
    }


def pad_contacts_both(k):
    seen = set()
    for i in range(d.ncon):
        gs = {d.contact[i].geom1, d.contact[i].geom2}
        if gs & OBJ_GEOMS:
            seen |= (gs & set(PADG[k]))
    return len(seen) == 2


def gate(name, ok, detail=""):
    print(f"[gate] {name:12s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    if not ok:
        FAIL.append(f"{name}: {detail}")
    return ok


RLH = None          # set below once park poses exist; E2E_RLHOOK=1 arms it


def frames(n):
    for _ in range(n):
        ct.control_step(SUB)
        if RLH is not None:
            RLH.tick()


# ---------------- variant demos: pick the nearest-in-y arm reference --------------------------
VAR = {2: dict(cart=(-2.20, 0.25), obj=(0.66, 0.35), dz=0.00),
       3: dict(cart=(-2.60, -0.25), obj=(0.60, -0.30), dz=0.00),
       4: dict(cart=(-2.40, 0.30), obj=(0.62, 0.15), dz=0.10),
       5: dict(cart=(-2.30, -0.30), obj=(0.64, -0.20), dz=-0.10),
       6: dict(cart=(-2.55, 0.00), obj=(0.60, 0.20), dz=0.10)}
vsel = min(VAR, key=lambda v: abs(VAR[v]["obj"][1] - obj_xy[1]))
DEMO = os.path.join(ROOT, "out", "teleop", "demos", f"pillar_v{vsel}_refined")
dd = np.load(os.path.join(DEMO, "episode_data.npz"))
ge = json.load(open(os.path.join(DEMO, "refine.json")))["grasp_end"]
dact, dbact, dbase = dd["action"], dd["base_action"], dd["base"]
N = len(dact)

# stage anchors (same detection as the staged builder): gs = base settle before the arm grasp;
# ps = arrival at the cart
bmov = np.abs(dbact[:ge]).max(1)
gs = 0
for f in range(ge):
    if bmov[f:ge].max() < 0.02:
        gs = f
        break
bx = dbase[:, 0]
arr = np.where(bx <= bx.min() + 0.10)[0]
arr = arr[arr > ge]
ps = int(arr[0]) if len(arr) else int(ge + (N - ge) * 0.7)

demo_obj = np.array(VAR[vsel]["obj"])
demo_cart = np.array(VAR[vsel]["cart"])
# arm goal shifts are 3D: variant demos differ in RACK height (v4/v6 +0.10, v5 -0.10) while the
# e2e scene is fixed at RACK_DZ=0 -> without the z term the arm hovers 10cm above the object.
d_obj = np.array([obj_xy[0] - demo_obj[0], obj_xy[1] - demo_obj[1], -VAR[vsel]["dz"]])
d_cart = np.array([cart_xy[0] - demo_cart[0], cart_xy[1] - demo_cart[1], 0.0])  # cart height fixed

# ---------------- offline FK: demo mount world-pose trajectories ------------------------------
_q0, _v0 = d.qpos.copy(), d.qvel.copy()
MNT = {}
for phase, f0, f1 in (("grasp", gs, ge), ("place", ps, N)):
    P = {"l": [], "r": []}
    for f in range(f0, f1):
        for i, adr in enumerate(ct.BQ):
            d.qpos[adr] = dbase[f, i]
        for k, A in (("l", ct.L), ("r", ct.R)):
            sl = slice(0, 7) if k == "l" else slice(7, 14)
            for j, a in enumerate(A.qadr):
                d.qpos[a] = dact[f, sl][j]
        mujoco.mj_kinematics(m, d)
        for k, A in (("l", ct.L), ("r", ct.R)):
            P[k].append((d.xpos[A.mount].copy(), d.xmat[A.mount].reshape(3, 3).copy()))
    MNT[phase] = P
d.qpos[:], d.qvel[:] = _q0, _v0
mujoco.mj_forward(m, d)
park_grasp = dbase[gs].copy()      # demo base pose while the arm grasps (base static)
park_place = dbase[ps].copy()
park_grasp[:2] += d_obj[:2]        # our parking = demo parking shifted with the object/cart (xy)
park_place[:2] += d_cart[:2]

# ---------------- randomize world: object qpos + robot start pose -----------------------------
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

print(f"[e2e] seed={SEED} obj=({obj_xy[0]:+.2f},{obj_xy[1]:+.2f}) cart=({cart_xy[0]:+.2f},{cart_xy[1]:+.2f}) "
      f"robot0=({robot0[0]:+.2f},{robot0[1]:+.2f},{robot0[2]:+.2f}) demo=v{vsel}", flush=True)

# ---------------- recorder ---------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
rec = ct.EpisodeRecorder(OUT)
ct.REC["rec"] = rec
ct.REC["on"] = True
ct.REC["count"] = 0
ct.REC["metadata"] = {"e2e": True, "seed": SEED, "demo_variant": vsel,
                      "obj_xy": obj_xy.tolist(), "cart_xy": cart_xy.tolist(),
                      "robot0": robot0.tolist()}


def finish(ok):
    validation = ct.REC["metadata"].setdefault("validation", {"version": 2})
    validation["passed"] = bool(ok)
    validation["failed_gates"] = list(FAIL)
    ct.REC["on"] = False
    rec.finalize(success=bool(ok))
    if RLH is not None:
        RLH.finalize(bool(ok))
    print(f"=== EPISODE {'PASS' if ok else 'FAIL'} ===", flush=True)
    if FAIL:
        print("  failed:", "; ".join(FAIL), flush=True)
    sys.exit(0)


# ---------------- closed-loop diff-drive navigation with kick-recovery -------------------------
def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


N_KICKS = int(os.environ.get("E2E_KICKS", "2"))
_kick_budget = N_KICKS


def maybe_kick(p_frame, scale=1.0):
    """With small per-frame probability shift base pose AND servo target together (= wheel slip /
    external shove). The go_to loop then measurably steers back -- that correction, seen from
    chassis_front, is the recovery demonstration we need in the data."""
    global _kick_budget
    if _kick_budget <= 0 or rng.random() > p_frame:
        return
    dq = scale * np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), rng.uniform(-0.15, 0.15)])
    for i, adr in enumerate(ct.BQ):
        d.qpos[adr] += dq[i]
    ct.base_tgt[:] += dq
    dvadr = [m.jnt_dofadr[j] for j in ct.BJ]
    for adr in dvadr:
        d.qvel[adr] = 0.0
    mujoco.mj_forward(m, d)
    _kick_budget -= 1
    print(f"[kick] dq=({dq[0]:+.2f},{dq[1]:+.2f},{dq[2]:+.2f}) at base=({ct.base_pose()[0]:+.2f},"
          f"{ct.base_pose()[1]:+.2f})", flush=True)


def go_to(tx, ty, tyaw, vmax=0.25, wmax=0.5, tol=0.025, yaw_tol=0.03, kicks=True, kick_scale=1.0, max_f=4500):
    """Closed-loop drive to (tx,ty) then rotate to tyaw. Diff-drive (no crab): turn toward the
    goal point, drive with continuous heading correction, then align the final yaw."""
    for _ in range(max_f):
        x, y, yaw = ct.base_pose()
        dist = float(np.hypot(tx - x, ty - y))
        if dist < tol:
            break
        hdg = np.arctan2(ty - y, tx - x)
        e = _ang(hdg - yaw)
        back = abs(e) > np.pi / 2          # allow reversing (demo backs away from the rack)
        if back:
            e = _ang(hdg + np.pi - yaw)
        if abs(e) > 0.35:
            ct.base_vel[:] = [0.0, float(np.clip(2.0 * e, -wmax, wmax))]
        else:
            v = float(np.clip(1.2 * dist, 0.05, vmax)) * (-1.0 if back else 1.0)
            ct.base_vel[:] = [v, float(np.clip(1.8 * e, -wmax, wmax))]
        frames(DECIM)
        if kicks:
            maybe_kick(0.004, kick_scale)
    for _ in range(1800):
        e = _ang(tyaw - ct.base_pose()[2])
        if abs(e) < yaw_tol:
            break
        ct.base_vel[:] = [0.0, float(np.clip(2.0 * e, -0.45, 0.45))]
        frames(DECIM)
    ct.base_vel[:] = 0.0
    frames(8)


# ---------------- SE(3) world-pose arm replay ---------------------------------------------------
def arm_replay(phase, f0, f1, dxy, base_free=False):
    """Per-frame: target mount pose = demo mount world pose + planar shift; IK under the CURRENT
    base pose (absorbs both the object/cart offset and our parking error). Gripper follows demo."""
    P = MNT[phase]
    off = np.asarray(dxy, dtype=float)          # 3D shift (x, y, z)
    for idx, f in enumerate(range(f0, f1)):
        q0s, v0s = d.qpos.copy(), d.qvel.copy()
        tq = {}
        for k, A in (("l", ct.L), ("r", ct.R)):
            pm, Rm = P[k][idx]
            for j, a in enumerate(A.qadr):
                d.qpos[a] = ct.qtgt[k][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(A, pm + off, Rm, iters=15, w=0.6)
            tq[k] = np.array([d.qpos[a] for a in A.qadr])
        d.qpos[:], d.qvel[:] = q0s, v0s
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["l"][:] = tq["l"]
        ct.qtgt["r"][:] = tq["r"]
        ct.grip_cmd["l"] = float((1 - dact[f, 14]) * 0.025)
        ct.grip_cmd["r"] = float((1 - dact[f, 15]) * 0.025)
        ct.base_vel[:] = 0.0
        frames(DECIM)


Q_HOME_L = ct.qtgt["l"].copy()
Q_HOME_R = ct.qtgt["r"].copy()

# ---------------- optional RL hook: curriculum snapshots + live reward scoring -----------------
if os.environ.get("E2E_RLHOOK") == "1":
    import shelf_e2e_rlhook
    RLH = shelf_e2e_rlhook.Hook(
        ct, mujoco, m, d, BODIES, OBJ_GEOMS, PADG,
        region_center=[0.5 * (REGION["x"][0] + REGION["x"][1]),
                       0.5 * (REGION["y"][0] + REGION["y"][1]), REGION["z"]],
        park_grasp=park_grasp, park_place=park_place,
        out_npz=os.path.join(ROOT, os.environ.get("E2E_SNAP_DIR", "out/rl/snap"),
                             f"snap_{SEED:06d}.npz"))
    RLH.in_region = in_region
    if os.environ.get("E2E_NOREC") == "1":   # snapshot-only run: no 85MB/episode video dump
        ct.REC["on"] = False

# ================================ FSM ================================
# 1) drive to the grasp parking pose (computed FROM the object; closed loop; kicks -> recovery)
go_to(park_grasp[0], park_grasp[1], park_grasp[2])
print(f"[nav1] parked ({ct.base_pose()[0]:+.2f},{ct.base_pose()[1]:+.2f},{ct.base_pose()[2]:+.2f}) "
      f"target=({park_grasp[0]:+.2f},{park_grasp[1]:+.2f},{park_grasp[2]:+.2f})", flush=True)

# 2) arm grasp (base static; world-pose SE3 replay of the nearest demo)
if RLH is not None:
    RLH.snap("pre_grasp")
arm_replay("grasp", gs, ge, d_obj)
ct.base_vel[:] = 0.0
frames(10)
_fR, _fL = grip_force("r"), grip_force("l")
if not gate("grip_firm", pad_contacts_both("r") and pad_contacts_both("l") and _fR >= 1.0 and _fL >= 1.0,
            f"bothpadR={pad_contacts_both('r')} bothpadL={pad_contacts_both('l')} fR={_fR:.1f}N fL={_fL:.1f}N"):
    finish(False)

if RLH is not None:
    RLH.note_grasp_offset()

# 3) tuck to the demo carry pose, then drive to the cart parking pose (closed loop; kicks)
qgl, qgr = dact[ge, 0:7], dact[ge, 7:14]
cl, cr = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
for i in range(60):
    s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 60)
    ct.qtgt["l"][:] = cl + (qgl - cl) * s
    ct.qtgt["r"][:] = cr + (qgr - cr) * s
    ct.grip_cmd["l"] = float((1 - dact[ge, 14]) * 0.025)
    ct.grip_cmd["r"] = float((1 - dact[ge, 15]) * 0.025)
    frames(DECIM)
# carrying a heavy pillar: a full-size kick (0.1m teleport) shakes it out of the two-hand
# squeeze -> smaller slips while loaded (still enough to demonstrate steering back).
if RLH is not None:
    RLH.snap("post_lift")
go_to(park_place[0], park_place[1], park_place[2], vmax=0.20, kick_scale=0.35)
print(f"[nav2] at cart ({ct.base_pose()[0]:+.2f},{ct.base_pose()[1]:+.2f},{ct.base_pose()[2]:+.2f}) "
      f"held fR={grip_force('r'):.1f} fL={grip_force('l'):.1f}", flush=True)
if grip_force("r") < 0.5 or grip_force("l") < 0.5:
    gate("carry", False, "dropped during transport")
    finish(False)

if RLH is not None:
    RLH.snap("pre_place")

# 4) place (world-pose SE3 replay of the demo place segment, shifted with the cart)
arm_replay("place", ps, N, d_cart)
ct.base_vel[:] = 0.0
frames(40)

# 5) settle + release both + retreat + home (deterministic close-out)
ct.grip_cmd["l"] = ct.GRIP_OPEN
ct.grip_cmd["r"] = ct.GRIP_OPEN
frames(60)
yaw = ct.base_pose()[2]
try:
    startR = d.xpos[ct.R.mount].copy()
    # small up+back retreat so the open grippers clear the part before homing
    for delta in ([0, 0, 0.05], [-0.12 * np.cos(yaw), -0.12 * np.sin(yaw), 0.0]):
        q0s, v0s = d.qpos.copy(), d.qvel.copy()
        tq = {}
        for k, A in (("l", ct.L), ("r", ct.R)):
            pm = d.xpos[A.mount].copy() + np.asarray(delta)
            Rm = d.xmat[A.mount].reshape(3, 3).copy()
            for j, a in enumerate(A.qadr):
                d.qpos[a] = ct.qtgt[k][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(A, pm, Rm, iters=20, w=0.6)
            tq[k] = np.array([d.qpos[a] for a in A.qadr])
        d.qpos[:], d.qvel[:] = q0s, v0s
        mujoco.mj_fwdPosition(m, d)
        for i in range(14):
            pass
        ct.qtgt["l"][:] = tq["l"]
        ct.qtgt["r"][:] = tq["r"]
        frames(16)
except Exception as e:  # retreat is best-effort; the gate below is what counts
    print(f"[retreat] skipped: {e}", flush=True)
cl, cr = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
for i in range(50):
    s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 50)
    ct.qtgt["l"][:] = cl + (Q_HOME_L - cl) * s
    ct.qtgt["r"][:] = cr + (Q_HOME_R - cr) * s
    ct.grip_cmd["l"] = ct.GRIP_OPEN
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    frames(DECIM)
frames(30)

required = max(1, int(np.ceil(0.5 / (SUB * m.opt.timestep))))
streak = 0
evidence = placement_evidence()
for _ in range(required * 4):
    frames(1)
    evidence = placement_evidence()
    valid_now = evidence["in_region"] and evidence["released"] and evidence["supported"]
    streak = streak + 1 if valid_now else 0
    if streak >= required:
        break
evidence["stable_for_s"] = round(float(streak * SUB * m.opt.timestep), 3)
evidence["stable"] = bool(streak >= required)
ct.REC["metadata"]["validation"] = evidence

p = obj_pos()
lo, hi = obj_extent()
ok = gate("placed", evidence["stable"],
          f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) xr=({lo[0]:.2f},{hi[0]:.2f}) "
          f"yr=({lo[1]:.2f},{hi[1]:.2f}) released={evidence['released']} "
          f"support={evidence['cart_support_force_n']:.1f}N stable={evidence['stable_for_s']:.2f}s")
finish(ok)
