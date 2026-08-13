#!/usr/bin/env python3
"""Scripted IK expert that records CRUZR ECU-transport episodes in the EXACT
teleop schema (route B of the data plan, 2026-07-12).

Two-stage task mirroring the user's teleop demo (out/teleop/_mujoco_rec):
  stage 1: side-pinch the ECU off the pickup pedestal, diff-drive around the
           kitchen cabinet to its EAST side, place the ECU flat on the enlarged
           precision_machine_part front plate at (0.88, -0.90)   [drop-verified 0.8 deg]
  stage 2: regrasp, diff-drive to the server rack, insert the ECU into the
           BOTTOM bay (floor z=0.904, same arm posture as the pedestal grasp),
           release, retract.

Everything is driven through cruzr_teleop's own runtime interface
(qtgt / grip_cmd / base_vel / control_step), so recorded state/action/base rows
are produced by the SAME code path as human teleop, and the base motion obeys
the built-in diff-drive integration (no crab-walk by construction).

Privileged info (ECU pose, tilt, contacts) is used ONLY by the expert
controller and the acceptance gates - it is never written into the 16-dim
policy-facing schema.

Run (records to mujoco_teleop/out/teleop/<EXPERT_OUT>):
  cd mujoco_teleop
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TELEOP_RECORD_GPU=2 \
    EXPERT_OUT=ecu_expert_00 \
    /data1/hsr/tools/miniconda3/envs/mjx/bin/python scripts/ecu_expert_record.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("TELEOP_HOME", "droop")       # start like the demo: arms hanging
os.environ.setdefault("MUJOCO_GL", "egl")
# deeper pinch command (force-limited anyway): more normal force -> the held plate twists
# less in yaw during carrying/turns (9 deg at 0.021 was failing the nest yaw gate)
os.environ.setdefault("CRUZR_GRIP_CLOSE", "0.025")

_spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

import mujoco  # noqa: E402  (after ct so the GL backend is already settled)
from scipy.optimize import least_squares  # noqa: E402

m, d = ct.m, ct.d
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))

OUT_NAME = os.environ.get("EXPERT_OUT", "ecu_expert_00")
OUT_DIR = os.path.join(HERE, "..", "out", "teleop", OUT_NAME)
SEED = int(os.environ.get("EXPERT_SEED", "0"))
rng = np.random.default_rng(SEED)

# ----------------------------- task constants (all ray/drop-verified 2026-07-12) -----------------------------
ECU_HALF = np.array([0.0824, 0.0756, 0.0128])       # jig_plate box half extents
# BRIDGE target (user 2026-07-12 round 3): the ECU rests ON TOP of the U-notch mouth,
# spanning it like a lid - NOT inside it. precision part back at x1.0 so the notch
# (~125mm) is NARROWER than the 151mm ECU; drop-verified BRIDGED tilt 0.5-0.9 deg at
# (0.50,-0.77), rest z = plate top 0.822 + half = 0.8347. The ECU west edge sits 19mm
# proud of the plate west edge (0.437) with a 72mm drop below - free finger access.
BRIDGE_ECU = np.array([0.50, -0.77])
BRIDGE_Z_REST = 0.8347
# 3x3 bay grid (all ray-probed; all three levels IK-verified reachable). Select the
# target bay with EXPERT_BAY="row,col": row 0/1/2 = bottom/middle/top level, col 0/1/2 =
# south/centre/north column. Default = bottom-centre (the validated v3 target).
BAY_FLOORS = [0.904, 1.020, 1.137]
BAY_COLS_Y = [-0.35, -0.04, 0.27]
_bay = os.environ.get("EXPERT_BAY", "0,1").split(",")
BAY_ROW, BAY_COL = int(_bay[0]), int(_bay[1])
RACK_BAY_FLOOR = BAY_FLOORS[BAY_ROW]
RACK_BAY_ECU = np.array([-2.27, BAY_COLS_Y[BAY_COL]])   # FULLY inside: east edge 30mm past face
RACK_BAY_MOUTH_Y = (BAY_COLS_Y[BAY_COL] - 0.10, BAY_COLS_Y[BAY_COL] + 0.10)
RACK_STANDOFF_X = -1.55                              # pre-insertion base x (chassis clear of rack face)
CARRY_LIFT = 0.05                                    # lift off the pedestal
CLAMP_DEPTH = 0.018                                  # pedestal grasp: padmid this far inside the near edge
GRIP_HOLD = float(os.environ.get("EXPERT_GRIP", "0.025"))
VMAX, WZMAX = 0.32, 0.45

R = ct.R
PADS = [ct.gid("R_pad1"), ct.gid("R_pad2")]
JIG = ct.bid("jig")
JQ = m.jnt_qposadr[ct.jid("jig_free")]
JIG_GEOMS = {g for g in range(m.ngeom)
             if m.geom_bodyid[g] == JIG and (m.geom_contype[g] or m.geom_conaffinity[g])}

FAIL = []

# ----------------------------- initial-pose randomization (EXPERT_SEED>0) -----------------------------
# Jitter the ECU spawn on the pedestal; every downstream step is closed-loop on the
# MEASURED ECU pose, so no other constant needs to change. x jitter is asymmetric: the
# grasp needs the west-edge overhang of the pedestal front face, so going east reduces it.
if SEED > 0:
    # v3 (2026-07-13): WIDE randomization - the ±2cm/±4° of v2 was visually invisible and
    # cancelled by the object-relative closed loop -> all episodes were near-identical.
    # x window keeps the pedestal-front overhang needed for the side pinch (x∈[0.450,0.500]).
    # window tuned by smoke tests (2026-07-13): dx>+12mm starves the west overhang the
    # side pinch needs (fingertip hits the pedestal front), |dy|>70mm degrades right-arm
    # IK quality. Still 3-7x wider than v2, plus path/noise layers on top.
    _jx = float(rng.uniform(-0.009, 0.012))
    _jy = float(rng.uniform(-0.070, 0.070))
    _jyaw = float(np.deg2rad(rng.uniform(-12.0, 12.0)))
    d.qpos[JQ:JQ + 3] = [0.459 + _jx, -0.006 + _jy, 0.9228 + 0.003]
    d.qpos[JQ + 3:JQ + 7] = [np.cos(_jyaw / 2), 0.0, 0.0, np.sin(_jyaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    print(f"[rand] seed={SEED} spawn jitter dx={_jx*1000:.0f}mm dy={_jy*1000:.0f}mm "
          f"dyaw={np.degrees(_jyaw):.1f}deg -> ecu={np.round(d.xpos[JIG],3).tolist()}", flush=True)


def ecu_pos():
    return d.xpos[JIG].copy()


def ecu_tilt_deg():
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, d.qpos[JQ + 3:JQ + 7])
    up = mat.reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def ecu_yaw_deg():
    """In-plane rotation of the ECU (0 = spawn orientation, box axes world-aligned)."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, d.qpos[JQ + 3:JQ + 7])
    r = mat.reshape(3, 3)
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    # the plate is 180-deg symmetric in yaw for our purposes
    yaw = (yaw + 90.0) % 180.0 - 90.0
    return float(yaw)


def padmid():
    return (d.geom_xpos[PADS[0]] + d.geom_xpos[PADS[1]]) / 2.0


def pad_ecu_contacts():
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        gs = {c.geom1, c.geom2}
        if gs & set(PADS) and gs & JIG_GEOMS:
            n += 1
    return n


NOISE_STD = float(os.environ.get("EXPERT_NOISE", "0.0035" if SEED > 0 else "0"))
_ou = np.zeros(7)
# noise is scaled DOWN during long base-drive carries: sustained arm jitter while
# carrying works the pinch loose (smoke 8201: dropped mid-carry), and correction
# demos matter for manipulation phases, not driving.
_noise_scale = 1.0


def frames(n):
    """n recorded frames = 2n control steps (REC_DECIM=2).

    v3: DART-style action noise - temporally-correlated (OU) perturbation is added to the
    right-arm command each control step (steady-state ~0.008 rad). The recorded action IS
    the noisy executed command; the expert's closed loops pull the arm back, so the data
    demonstrates deviation->correction pairs instead of a single perfect trajectory."""
    global _ou
    for _ in range(2 * n):
        if NOISE_STD > 0:
            _ou = 0.9 * _ou + rng.normal(0.0, NOISE_STD * _noise_scale, 7)
            clean = ct.qtgt["r"].copy()
            ct.qtgt["r"][:] = np.clip(clean + _ou, R.lo, R.hi)
            ct.control_step(SUB)
            ct.qtgt["r"][:] = clean
        else:
            ct.control_step(SUB)


# ----------------------------- side-grasp solver (from the validated prototype) -----------------------------
def _fk_probe(q):
    """padmid / gap direction / insert direction for arm joints q at the CURRENT base pose."""
    qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
    for adr, val in zip(R.qadr, q):
        d.qpos[adr] = val
    mujoco.mj_fwdPosition(m, d)
    p1, p2 = d.geom_xpos[PADS[0]].copy(), d.geom_xpos[PADS[1]].copy()
    mnt = d.xpos[R.mount].copy()
    d.qpos[:], d.qvel[:] = qpos0, qvel0
    mujoco.mj_fwdPosition(m, d)
    pm = (p1 + p2) / 2.0
    gap = p2 - p1
    ins = pm - mnt
    return pm, gap / np.linalg.norm(gap), ins / np.linalg.norm(ins)


def solve_side_grasp(pad_target, insert_axis):
    """R-arm joints putting the pad midpoint at pad_target with a VERTICAL pad gap
    and the mount->padmid axis along insert_axis (horizontal side pinch)."""
    insert_axis = np.asarray(insert_axis, float)
    insert_axis = insert_axis / np.linalg.norm(insert_axis)
    seeds = [
        np.array(ct.SIDEGRASP_POSE["r"]),
        np.array([-0.4071, -0.1980, 0.2932, -1.4225, -1.0526, -0.3290, -1.1418]),
        np.array([-0.5198, -0.0696, 0.1555, -1.3054, -0.9862, -0.3548, -1.2535]),
        np.array([-1.0, -0.5, 0.5, -1.4, 0.5, 0.0, 0.0]),
        # mean v4 grasp-window pose (2026-07-16): known positive-gap branch that fits the
        # SDK-tightened limits - keeps the optimizer off the mirrored branch.
        np.array([-0.37, -0.13, 0.34, -1.58, -1.14, -0.08, -1.20]),
    ]

    def residual(q):
        pm, gap, ins = _fk_probe(q)
        # gap[2] pulled to +1 (not just vertical): with the SDK-tightened joint limits
        # (2026-07-16) the optimizer otherwise lands on the 180-deg mirrored branch
        # (gap_z=-1, fingers flipped -> zero pad contacts).
        return np.r_[25.0 * (pm - pad_target),
                     5.0 * gap[0], 5.0 * gap[1], 3.0 * (gap[2] - 1.0),
                     5.0 * (ins - insert_axis)]

    best = None
    for seed in seeds:
        res = least_squares(residual, np.clip(seed, R.lo, R.hi), bounds=(R.lo, R.hi),
                            max_nfev=300, xtol=1e-6, ftol=1e-6, gtol=1e-6)
        pm, gap, ins = _fk_probe(res.x)
        score = (np.linalg.norm(pm - pad_target)
                 + 0.05 * (1.0 - gap[2])
                 + 0.05 * np.linalg.norm(ins - insert_axis))
        if best is None or score < best[0]:
            best = (score, res.x.copy(), pm, gap, ins)
    return best


# ----------------------------- motion primitives (all through control_step) -----------------------------
def smoothstep(k, n):
    t = (k + 1) / n
    return t * t * (3.0 - 2.0 * t)


def move_arm_joints(q_to, nframes, tag=""):
    q_from = ct.qtgt["r"].copy()
    q_to = np.clip(np.asarray(q_to, float), R.lo, R.hi)
    dbg = os.environ.get("EXPERT_JAMDBG") == "1"
    for k in range(nframes):
        ct.qtgt["r"][:] = q_from + (q_to - q_from) * smoothstep(k, nframes)
        q_prev = np.array([d.qpos[a] for a in R.qadr])
        frames(1)
        if dbg:
            q_now = np.array([d.qpos[a] for a in R.qadr])
            err = float(np.abs(ct.qtgt["r"] - q_now).max())
            jump = float(np.abs(q_now - q_prev).max())
            if err > 0.25 or jump > 0.08:
                pairs = set()
                for i in range(d.ncon):
                    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1) or f"g{d.contact[i].geom1}"
                    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2) or f"g{d.contact[i].geom2}"
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]) or "?"
                    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]) or "?"
                    if any(s.startswith("R_") or s.startswith("r_") for s in (b1, b2)):
                        pairs.add(f"{b1}({g1})~{b2}({g2})")
                print(f"[jamdbg {tag} k={k}/{nframes}] err={err*1000:.0f}mrad jump={jump*1000:.0f}mrad "
                      f"contacts={sorted(pairs) if pairs else '无臂接触'}", flush=True)


def move_mount_cart(delta, nframes, rd=None):
    """Cartesian straight-line move of the mount by world-frame delta, orientation held."""
    start = d.xpos[R.mount].copy()
    if rd is None:
        rd = d.xmat[R.mount].reshape(3, 3).copy()
    delta = np.asarray(delta, float)
    for k in range(nframes):
        tgt = start + delta * smoothstep(k, nframes)
        qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
        for i, adr in enumerate(R.qadr):        # seed IK from the current command
            d.qpos[adr] = ct.qtgt["r"][i]
        ct.ik(R, tgt, rd, iters=40, w=0.6)
        q = np.array([d.qpos[a] for a in R.qadr])
        d.qpos[:], d.qvel[:] = qpos0, qvel0
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["r"][:] = q
        frames(1)


def set_grip(target, nframes):
    ct.grip_cmd["r"] = float(target)
    frames(nframes)


def servo_padmid_to(pad_target, max_iters=3):
    """Closed-loop correction of gravity/servo sag: after a joint move the measured pad
    midpoint typically sags/falls short by 1-2cm; nudge the mount by the measured error."""
    for _ in range(max_iters):
        err = np.asarray(pad_target, float) - padmid()
        if np.linalg.norm(err) < 0.004:
            break
        move_mount_cart(err, 12)
    print(f"[servo] padmid err {np.linalg.norm(np.asarray(pad_target)-padmid())*1000:.1f}mm", flush=True)


def _ang_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def base_turn_to(yaw_target, tol=0.02):
    global _noise_scale
    _prev, _noise_scale = _noise_scale, 0.3
    for _ in range(1200):
        err = _ang_wrap(yaw_target - ct.base_pose()[2])
        if abs(err) < tol:
            break
        ct.base_vel[:] = [0.0, float(np.clip(2.0 * err, -WZMAX, WZMAX))]
        frames(1)
    ct.base_vel[:] = 0.0
    frames(3)
    _noise_scale = _prev


def base_drive_to(dist_fn, tol=0.015, vmax=VMAX):
    """Drive straight along the CURRENT heading until dist_fn() (signed, +=forward) < tol."""
    global _noise_scale
    _prev, _noise_scale = _noise_scale, 0.3
    for _ in range(2400):
        rem = dist_fn()
        if abs(rem) < tol:
            break
        ct.base_vel[:] = [float(np.clip(1.5 * rem, -vmax, vmax)), 0.0]
        frames(1)
    ct.base_vel[:] = 0.0
    frames(3)
    _noise_scale = _prev


def drive_axis(axis, target, vmax=VMAX, tol=0.015):
    """Drive along the world axis ('x'|'y') the base is currently facing (or backing along)."""
    idx = 0 if axis == "x" else 1

    def rem():
        yaw = ct.base_pose()[2]
        head = np.array([np.cos(yaw), np.sin(yaw)])
        delta = target - ct.base_pose()[idx]
        return delta / head[idx] if abs(head[idx]) > 0.5 else 0.0

    base_drive_to(rem, tol=tol, vmax=vmax)


def gate(name, ok, detail=""):
    print(f"[gate] {name:26s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    if not ok:
        FAIL.append(f"{name}: {detail}")
    return ok


# ----------------------------- episode script -----------------------------
# EXPERT_STAGE=multi: one full-chain run emits the LAST THREE stages as separate
# sub-episodes (OUT_st2 place / OUT_st3 regrasp / OUT_st4 insert), rendering ONLY inside
# those stages. Transit legs run unrecorded (no camera render = ~4x faster wall clock),
# each stage opens with a 15-frame parked settle to match stage_split's parked_start.
MULTI = os.environ.get("EXPERT_STAGE", "") == "multi"
_stage_rec = {"rec": None, "idx": None}


def stage_rec(idx, success=True):
    """Close the current stage recorder (finalize) and open the one for stage idx."""
    if not MULTI:
        return
    if _stage_rec["rec"] is not None:
        ct.REC["on"] = False
        _stage_rec["rec"].finalize(success=bool(success))
        print(f"[multi] stage {_stage_rec['idx']} closed success={success} "
              f"({_stage_rec['rec'].n} frames)", flush=True)
        _stage_rec["rec"] = None
    if idx is None:
        return
    os.environ["REC_PROMPT"] = {
        2: "carry the ECU and place it on the fixture",
        3: "pick up the ECU from the fixture",
        4: "carry the ECU and insert it into the rack bay"}[idx]
    rec = ct.EpisodeRecorder(OUT_DIR + f"_st{idx}")
    ct.REC["rec"] = rec
    ct.REC["on"] = True
    ct.REC["count"] = 0
    _stage_rec["rec"], _stage_rec["idx"] = rec, idx
    # parked settle context at LOW noise: full-scale OU jitter on a stationary pinch
    # works the plate loose (st2 bridged-gate failures s41/s42) - not in the v3 recipe
    global _noise_scale
    _prev, _noise_scale = _noise_scale, 0.2
    frames(15)
    _noise_scale = _prev


NOMINAL_ECU = np.array([0.459, -0.006, 0.9228 + 0.003])  # 未抖动的名义 spawn（与随机化基准一致）


def goto_ready_pose():
    """就绪位起点契约（2026-07-18 根因修复）。

    droop->pre 的任何大转移（关节直线/外弧/笛卡尔高跨三种设计均实测失败）都会让
    夹爪手指钩上 SDK 对齐后的升降连杆(lifter_pitch_2_link)/腰(waist_yaw_link)，
    "卡死-弹通"尖峰污染了 v5 S1 全部 379 条数据。走廊被升降柱(中)+厨柜(外)堵死，
    透明修不掉，故改契约：把危险转移移到【开录之前】——
      就绪位 = 名义 ECU 前方 standoff+50mm、上方 80mm 处的侧抓姿态（固定常量，
      不用抖动后的真实 ECU 位置，留给策略 ~10-15cm 的视觉伺服余量）。
    数据从就绪位开始（含干净的最后接近+抓取）；eval 侧 M1 前由 FSM 走同一就绪位
    （ecu_hybrid_rollout reset_arm_ready）。与 S3 数据同构——S3 起点就是放置后
    姿态而非垂臂，其数据 83-100% 干净。
    就绪位关节角写入 out/smoke/_s1_ready_pose.json 供 rollout 读取，保证两侧一致。
    """
    ins0 = np.array([1.0, 0.0, 0.0])
    pad_ready = NOMINAL_ECU + ins0 * (-ECU_HALF[0] + CLAMP_DEPTH - 0.09 - 0.05)
    pad_ready[2] = NOMINAL_ECU[2] + 0.08
    _, q_ready, pm_r, _, _ = solve_side_grasp(pad_ready, ins0)
    # kinematic 就位：物理转移三种设计均实测被本体几何卡死（升降柱/腰/厨柜堵死走廊），
    # 故直接置位——episode 从静止有效状态开始，策略永远看不到就位过程（真机上这是
    # 底层控制器的规划职责，不属于数据分布）。置位后验证无本体接触再放行。
    q_ready = np.clip(q_ready, R.lo, R.hi)
    for i, adr in enumerate(R.qadr):
        d.qpos[adr] = q_ready[i]
    ct.qtgt["r"][:] = q_ready
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    bad = []
    for i in range(d.ncon):
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]) or "?"
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]) or "?"
        if (b1.startswith("R_") != b2.startswith("R_")) and not {b1, b2} & {"jig"}:
            bad.append(f"{b1}~{b2}")
    if bad:
        print(f"[ready] FATAL: 就绪位本身有本体接触 {bad}", flush=True)
        sys.exit(3)
    for _ in range(30):
        ct.control_step(SUB)                       # 置位后物理稳定
    err = float(np.abs(np.array([d.qpos[a] for a in R.qadr]) - q_ready).max())
    print(f"[ready] kinematic 就位完成 稳定后跟踪误差={err*1000:.0f}mrad", flush=True)
    import json as _json
    _ready_path = os.path.join(HERE, "..", "out", "smoke", "_s1_ready_pose.json")
    os.makedirs(os.path.dirname(_ready_path), exist_ok=True)
    with open(_ready_path, "w") as f:
        _json.dump({"q_ready": [float(x) for x in q_ready],
                    "note": "S1 ready-pose contract 2026-07-18; solved at nominal ECU; kinematic set"}, f)
    return q_ready


def main():
    # 就绪位转移在开录之前完成（危险转移不进数据）
    if os.environ.get("EXPERT_READY_START", "1") == "1":
        goto_ready_pose()

    if os.environ.get("EXPERT_NOREC") == "1" or MULTI:   # no full-episode recorder
        rec = None
        print("=== ECU expert", "(MULTI single-stage mode)" if MULTI else
              "(NOREC tuning mode, no cameras)", "===", flush=True)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        rec = ct.EpisodeRecorder(OUT_DIR)
        ct.REC["rec"] = rec
        ct.REC["on"] = True
        ct.REC["count"] = 0

    print(f"=== ECU expert episode -> {os.path.abspath(OUT_DIR)}  seed={SEED} ===", flush=True)
    frames(12)                                             # settle / leading context

    # ---------------- stage 1: grasp at pedestal ----------------
    ecu0 = ecu_pos()
    ins = ct.base_rotz() @ np.array([1.0, 0.0, 0.0])
    clamp = float(rng.uniform(0.016, 0.028)) if SEED > 0 else CLAMP_DEPTH
    pad_t = ecu0 + ins * (-ECU_HALF[0] + clamp)
    pad_t[2] = ecu0[2]
    score, q_grasp, pm, gap, _ = solve_side_grasp(pad_t, ins)
    print(f"[grasp] solve score={score:.4f} padmid_err={np.linalg.norm(pm-pad_t)*1000:.1f}mm gap_z={gap[2]:+.3f}", flush=True)
    if not gate("grasp_ik", np.linalg.norm(pm - pad_t) < 0.012 and gap[2] > 0.95):
        return finish(rec, False)

    standoff = float(rng.uniform(0.07, 0.12)) if SEED > 0 else 0.09
    _, q_pre, pm_pre, _, _ = solve_side_grasp(pad_t - ins * standoff, ins)
    # 外弧接近（2026-07-18 根因修复）：droop->pre 的关节空间直线插值会让夹爪手指
    # 钩上 SDK 对齐新增的升降连杆(lifter_pitch_2_link)——jamdbg 实测跟踪误差蓄到
    # 1800mrad 后弹通，v5 S1 数据因此 379/379 全部污染（"卡死-弹通"尖峰）。
    # 修复=先经"高+远"中转位走外侧弧线绕开升降连杆，再落进预插入位。
    # EXPERT_ARC_APPROACH=0 可回退旧直线路径（对照用）。
    if os.environ.get("EXPERT_READY_START", "1") == "1":
        # 就绪位起点契约（见 goto_ready_pose）：开录时手臂已在名义 ECU 前上方的就绪位，
        # 录制段只含"就绪位 -> 预插入"的短程前侧转移（远离升降柱/腰），干净可验。
        move_arm_joints(q_pre, int(rng.integers(35, 60)) if SEED > 0 else 45, tag="ready2pre")
    else:
        move_arm_joints(q_pre, int(rng.integers(60, 110)) if SEED > 0 else 80, tag="droop2pre")   # droop -> pre-insert
    if SEED > 0:
        frames(int(rng.integers(0, 15)))                   # random hesitation pause
    ok = gate("approach_undisturbed", np.linalg.norm(ecu_pos() - ecu0) < 0.02,
              f"ecu moved {np.linalg.norm(ecu_pos()-ecu0)*1000:.0f}mm")
    move_arm_joints(q_grasp, int(rng.integers(30, 60)) if SEED > 0 else 45, tag="pre2grasp")   # straddle the plate
    servo_padmid_to(pad_t)                                 # cancel servo sag before closing
    set_grip(GRIP_HOLD, int(rng.integers(45, 70)) if SEED > 0 else 55)   # rate-limited close
    ok &= gate("grasp_contacts", pad_ecu_contacts() >= 1, f"{pad_ecu_contacts()} pad contacts")
    z_before = ecu_pos()[2]
    lift_h = float(rng.uniform(0.05, 0.075)) if SEED > 0 else CARRY_LIFT
    move_mount_cart([0.0, 0.0, lift_h], int(rng.integers(35, 60)) if SEED > 0 else 45)
    lift = ecu_pos()[2] - z_before
    # tilt gate 8->10 deg (2026-07-16): SDK-tightened sh_roll (+0.08 rad cap) shallows the
    # edge pinch by ~1 deg; carry_held (15 deg) still guards transport downstream.
    ok &= gate("lift", lift > 0.03 and ecu_tilt_deg() < 10.0,
               f"lift={lift*1000:.0f}mm tilt={ecu_tilt_deg():.1f}deg")
    if not ok:
        return finish(rec, False)
    # EXPERT_STAGE=grasp_only: short grasp-densification episodes (droop -> approach ->
    # close -> lift -> hold). Used to raise the grasp-window sample density for BC - the
    # full-task episodes spend only ~6% of their frames on the grasp.
    if os.environ.get("EXPERT_STAGE", "") == "grasp_only":
        frames(10)
        return finish(rec, True)

    # ---------------- stage 1: short west route to the U-notch mouth ----------------
    # ANALYTIC route (no correction wiggles, user round 4): the gripper->ECU offset is a
    # rigid-body constant once grasped, so measure it ONCE and pre-compute every leg target
    # exactly: at yaw=0, ecu_y = base_y + o1_y.
    bp = ct.base_pose()
    c, s = np.cos(bp[2]), np.sin(bp[2])
    e = ecu_pos()[:2] - bp[:2]
    o1 = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])   # base-frame ECU offset
    print(f"[route] measured gripper->ECU offset o1=({o1[0]:.3f},{o1[1]:.3f})", flush=True)
    drive_axis("x", -0.55)                                  # reverse clear of the pedestal
    base_turn_to(-np.pi / 2)                                # face -Y
    drive_axis("y", BRIDGE_ECU[1] - o1[1], tol=0.003)       # exact: no correction loop needed
    base_turn_to(0.0)                                       # face +X toward the notch mouth
    # closed-loop on the OBJECT: advance until the ECU is centred over the U-notch mouth
    base_drive_to(lambda: BRIDGE_ECU[0] - ecu_pos()[0], tol=0.004, vmax=0.08)
    # single fallback correction ONLY if the turn wobble exceeded the bridge margin
    if abs(ecu_pos()[1] - BRIDGE_ECU[1]) > 0.008:
        err_y = ecu_pos()[1] - BRIDGE_ECU[1]
        by = ct.base_pose()[1] - err_y
        base_turn_to(-np.pi / 2)
        base_drive_to(lambda: ct.base_pose()[1] - by, tol=0.003, vmax=0.10)
        base_turn_to(0.0)
        base_drive_to(lambda: BRIDGE_ECU[0] - ecu_pos()[0], tol=0.004, vmax=0.06)
    stage_rec(2)   # multi: record the place stage from the PARKED pose (split-data match)
    p = ecu_pos()
    ok = gate("carry_held", pad_ecu_contacts() >= 1 and ecu_tilt_deg() < 15.0,
              f"tilt={ecu_tilt_deg():.1f}deg")
    ok &= gate("bridge_align",
               abs(p[0] - BRIDGE_ECU[0]) < 0.008 and abs(p[1] - BRIDGE_ECU[1]) < 0.010,
               f"ecu=({p[0]:.3f},{p[1]:.3f}) target=({BRIDGE_ECU[0]:.3f},{BRIDGE_ECU[1]:.3f})")
    if not ok:
        return finish(rec, False)

    # ---------------- stage 1: BRIDGE the ECU over the U-notch mouth ----------------
    # Plain flat-surface set-down (the v1 sequence that measured 0.8-1.5 deg): closed-loop
    # descend with horizontal lock until the plate carries the ECU, unload the pinch, open,
    # withdraw level toward the west (the fingers pass over the open notch mouth / past the
    # plate west edge, 72mm of free air below), then rise.
    def _dbg(tag):
        print(f"[bridge dbg {tag}] ecu={np.round(ecu_pos(),3).tolist()} tilt={ecu_tilt_deg():.1f} "
              f"yaw={ecu_yaw_deg():.1f}", flush=True)
    global _noise_scale
    _prev_ns, _noise_scale = _noise_scale, 0.2   # mm-level set-down: noise wrecks it
    pad_xy0 = padmid()[:2].copy()
    for _ in range(120):
        if ecu_pos()[2] <= BRIDGE_Z_REST + 0.002:
            break
        err = pad_xy0 - padmid()[:2]
        move_mount_cart([float(np.clip(err[0], -0.002, 0.002)),
                         float(np.clip(err[1], -0.002, 0.002)), -0.003], 1)
    frames(12)
    _dbg("set-down")
    set_grip(GRIP_HOLD - 0.001, 15)                          # unload the pinch
    set_grip(ct.GRIP_OPEN, 30)
    frames(20)
    _dbg("released")
    move_mount_cart(ct.base_rotz() @ np.array([-0.11, 0.0, 0.0]), 40)   # level withdrawal
    move_mount_cart([0.0, 0.0, 0.06], 20)
    frames(15)
    _noise_scale = _prev_ns
    p = ecu_pos()
    ok = gate("bridged",
              abs(p[0] - BRIDGE_ECU[0]) < 0.025 and abs(p[1] - BRIDGE_ECU[1]) < 0.013
              and abs(p[2] - BRIDGE_Z_REST) < 0.008
              and ecu_tilt_deg() < 7.0 and abs(ecu_yaw_deg()) < 10.0,
              f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) tilt={ecu_tilt_deg():.1f} yaw={ecu_yaw_deg():.1f}")
    if not ok:
        return finish(rec, False)
    move_mount_cart([0.0, 0.0, 0.05], 20)
    stage_rec(None)                                         # multi: st2 done (success)

    # ---------------- stage 2: regrasp off the bridge ----------------
    # reposition so the right shoulder faces the grasp point (reach probing, v2 lesson)
    base_turn_to(np.pi / 2)
    base_drive_to(lambda: -0.60 - ct.base_pose()[1], tol=0.005, vmax=0.15)
    base_turn_to(0.0)
    stage_rec(3)                                            # multi: record the regrasp stage
    drive_axis("x", 0.02, tol=0.005, vmax=0.08)
    # pedestal-style deep clamp on the ECU west edge: it sits 19mm proud of the plate west
    # edge with 72mm of clear air below - no jam constraints here.
    ecu1 = ecu_pos()
    ins2 = ct.base_rotz() @ np.array([1.0, 0.0, 0.0])
    pad_t2 = ecu1 + ins2 * (-ECU_HALF[0] + CLAMP_DEPTH)
    pad_t2[2] = ecu1[2]
    score, q_grasp2, pm2, gap2, _ = solve_side_grasp(pad_t2, ins2)
    print(f"[regrasp] solve score={score:.4f} padmid_err={np.linalg.norm(pm2-pad_t2)*1000:.1f}mm", flush=True)
    if not gate("regrasp_ik", np.linalg.norm(pm2 - pad_t2) < 0.012):
        return finish(rec, False)
    _, q_pre2, _, _, _ = solve_side_grasp(pad_t2 - ins2 * 0.09, ins2)
    move_arm_joints(q_pre2, 50)
    move_arm_joints(q_grasp2, 40)
    servo_padmid_to(pad_t2)                                 # cancel servo sag before closing
    set_grip(GRIP_HOLD, 55)
    move_mount_cart(-ins2 * 0.04, 25)                       # pull 4cm out of the mouth first
    z_before = ecu_pos()[2]
    move_mount_cart([0.0, 0.0, 0.125], 60)                  # lift clear of the plate top (0.851)
    ok = gate("regrasp_lift", ecu_pos()[2] > 0.868 and pad_ecu_contacts() >= 2
              and ecu_tilt_deg() < 16.0,
              f"z={ecu_pos()[2]:.3f} lift={(ecu_pos()[2]-z_before)*1000:.0f}mm "
              f"tilt={ecu_tilt_deg():.1f}deg yaw={ecu_yaw_deg():.1f}deg")
    if not ok:
        return finish(rec, False)
    stage_rec(None)                                         # multi: st3 done (success)

    # ---------------- stage 2: drive to the rack ----------------
    # ANALYTIC route: re-measure the (new, post-regrasp) rigid offset once; at yaw=pi,
    # ecu_y = base_y - o2_y, so the y-leg target is exact - no correction wiggles.
    bp = ct.base_pose()
    c, s = np.cos(bp[2]), np.sin(bp[2])
    e = ecu_pos()[:2] - bp[:2]
    o2 = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
    print(f"[route] measured post-regrasp offset o2=({o2[0]:.3f},{o2[1]:.3f})", flush=True)
    drive_axis("x", -0.60)                                  # reverse away from the cabinet
    base_turn_to(np.pi / 2)                                 # face +Y
    drive_axis("y", RACK_BAY_ECU[1] + o2[1], tol=0.004)     # at yaw=pi: ecu_y = base_y - o2_y
    base_turn_to(np.pi)                                     # face -X toward the rack
    drive_axis("x", RACK_STANDOFF_X)
    # single fallback correction ONLY if needed (analytic leg is normally within ~5mm)
    if abs(ecu_pos()[1] - RACK_BAY_ECU[1]) > 0.010:
        err_y = ecu_pos()[1] - RACK_BAY_ECU[1]
        by = ct.base_pose()[1] - err_y
        base_turn_to(np.pi / 2)
        base_drive_to(lambda: by - ct.base_pose()[1], tol=0.004, vmax=0.12)
        base_turn_to(np.pi)
        drive_axis("x", RACK_STANDOFF_X, tol=0.01, vmax=0.10)

    stage_rec(4)                                            # multi: record the insert stage

    # ---------------- stage 2: PRE-INSERTION VERIFICATION (user-required) ----------------
    # verify the held ECU is straight, level enough, and centred in the bay mouth BEFORE
    # approaching, so it cannot clip the mouth edges and get spun out of the pinch.
    p = ecu_pos()
    y_lo = RACK_BAY_MOUTH_Y[0] + ECU_HALF[1] + 0.010
    y_hi = RACK_BAY_MOUTH_Y[1] - ECU_HALF[1] - 0.010
    ok = gate("preinsert_check",
              pad_ecu_contacts() >= 1 and ecu_tilt_deg() < 18.0
              and abs(ecu_yaw_deg()) < 10.0 and y_lo < p[1] < y_hi,
              f"y={p[1]:.3f} in ({y_lo:.3f},{y_hi:.3f}) yaw={ecu_yaw_deg():.1f} tilt={ecu_tilt_deg():.1f}")
    if not ok:
        return finish(rec, False)

    # entry height: drooping far tip must clear the 28mm bay front rim
    z_entry = RACK_BAY_FLOOR + ECU_HALF[2] + 0.033
    dz = z_entry - ecu_pos()[2]
    move_mount_cart([0.0, 0.0, dz], max(30, int(abs(dz) / 0.004)))

    # ---------------- stage 2: watched base-drive insertion ----------------
    # closed-loop on the OBJECT with a collision WATCHDOG: if mouth contact starts twisting
    # the ECU (tilt/yaw jump), back off and retry once, slower.
    def insert_pass(vmax):
        for _ in range(2400):
            remx = ecu_pos()[0] - RACK_BAY_ECU[0]
            if remx < 0.008:
                ct.base_vel[:] = 0.0
                frames(3)
                return True
            if ecu_tilt_deg() > 26.0 or abs(ecu_yaw_deg()) > 12.0:
                ct.base_vel[:] = 0.0
                frames(3)
                return False
            ct.base_vel[:] = [float(np.clip(1.5 * remx, 0.0, vmax)), 0.0]
            frames(1)
        ct.base_vel[:] = 0.0
        return False

    if not insert_pass(0.05):
        print(f"[insert] watchdog tripped (tilt={ecu_tilt_deg():.1f} yaw={ecu_yaw_deg():.1f}) "
              "-> back off and retry slower", flush=True)
        drive_axis("x", RACK_STANDOFF_X, vmax=0.10)         # back out
        move_mount_cart([0.0, 0.0, z_entry + 0.006 - ecu_pos()[2]], 20)
        insert_pass(0.03)
    p = ecu_pos()
    ok = gate("inserted", p[0] < -2.25 and ecu_tilt_deg() < 22.0 and abs(ecu_yaw_deg()) < 12.0,
              f"ecu_x={p[0]:.3f} tilt={ecu_tilt_deg():.1f} yaw={ecu_yaw_deg():.1f}")
    move_mount_cart([0.0, 0.0, -(ecu_pos()[2] - (RACK_BAY_FLOOR + ECU_HALF[2] + 0.001))], 30)
    set_grip(ct.GRIP_OPEN, 35)
    frames(20)
    if MULTI:
        ct.REC["on"] = False   # st4 data ends at release; success still set by rack_final
    drive_axis("x", RACK_STANDOFF_X + 0.15, vmax=0.15)      # reverse out
    frames(25)

    p = ecu_pos()
    tail = [ecu_pos().copy()]
    for _ in range(19):
        frames(1)
        tail.append(ecu_pos().copy())
    drift = float(np.linalg.norm(np.array(tail) - tail[0], axis=1).max())
    ok &= gate("rack_final",
               -2.55 < p[0] < -2.19 and abs(p[1] - RACK_BAY_ECU[1]) < 0.06
               and RACK_BAY_FLOOR - 0.01 < p[2] - ECU_HALF[2] < RACK_BAY_FLOOR + 0.03
               and ecu_tilt_deg() < 8.0 and abs(ecu_yaw_deg()) < 15.0 and drift < 0.01,
               f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) tilt={ecu_tilt_deg():.1f} "
               f"yaw={ecu_yaw_deg():.1f} drift={drift*1000:.0f}mm")
    return finish(rec, ok)


def finish(rec, success):
    if MULTI and _stage_rec["rec"] is not None:              # gate failed mid-stage
        stage_rec(None, success=success)
    ct.REC["on"] = False
    if rec is not None:
        rec.finalize(success=bool(success))
    n = rec.n if rec is not None else "norec"
    print("\n=== EPISODE", "PASS" if success else "FAIL", f"({n} frames) ===", flush=True)
    for f in FAIL:
        print("  failed gate:", f, flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
