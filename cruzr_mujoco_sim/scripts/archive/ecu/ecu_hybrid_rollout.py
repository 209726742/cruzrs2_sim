#!/usr/bin/env python3
"""HYBRID closed-loop rollout: system navigation + VLA manipulation (2026-07-14).

Mirrors the real CRUZR S2 architecture (SDK NavTo A000002 handles driving; skills run
on top): scripted diff-drive navigation moves the base between stations, the fine-tuned
pi0.5 policy performs the parked manipulation stages with stage prompts:
  M1 "pick up the ECU from the stand"            @ spawn parking
  -> NavTo fixture parking (-0.063,-0.7765,0) then (-0.02,-0.60,0)-style repose per stage
  M2 "carry the ECU and place it on the fixture"
  M3 "pick up the ECU from the fixture"
  -> NavTo rack parking (-1.55, bay_y-0.006, pi) + micro y-align
  M4 "carry the ECU and insert it into the rack bay"

Privileged info: stage-completion checks + navigation targets only (the real robot gets
these from its SLAM map); never fed to the policy.

Env: POLICY_HOST/PORT, ROLLOUT_SPAWN_SEED, ROLLOUT_BAY="row,col", ROLLOUT_OUT,
     ROLLOUT_REPLAN(8), ROLLOUT_SKIP(3), stage timeouts ROLLOUT_STAGE_TIMEOUT(45s).
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
os.environ.setdefault("CRUZR_GRIP_CLOSE", "0.025")

_spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

import mujoco  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

m, d = ct.m, ct.d
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "8731"))
REPLAN = int(os.environ.get("ROLLOUT_REPLAN", "8"))
SKIP = int(os.environ.get("ROLLOUT_SKIP", "3"))
OUT = os.path.join(HERE, "..", os.environ.get("ROLLOUT_OUT", "out/rollout/ecu/hybrid_rollout"))
STAGE_TIMEOUT = float(os.environ.get("ROLLOUT_STAGE_TIMEOUT", "70"))
_bay = os.environ.get("ROLLOUT_BAY", "0,1").split(",")
BAY_ROW, BAY_COL = int(_bay[0]), int(_bay[1])
BAY_FLOORS = [0.904, 1.020, 1.137]
BAY_COLS_Y = [-0.35, -0.04, 0.27]
RACK_BAY_FLOOR = BAY_FLOORS[BAY_ROW]
RACK_BAY_Y = BAY_COLS_Y[BAY_COL]

JIG = ct.bid("jig")
JQ = m.jnt_qposadr[ct.jid("jig_free")]
PADS = [ct.gid("R_pad1"), ct.gid("R_pad2")]
JIG_GEOMS = {g for g in range(m.ngeom) if m.geom_bodyid[g] == JIG}
STAGE_PROMPTS = [
    "pick up the ECU from the stand",
    "carry the ECU and place it on the fixture",
    "pick up the ECU from the fixture",
    # M4 行条件化（2026-07-20）：目标格位的"行"此前无任何通道传达给策略（列由停靠
    # 侧向对齐传达）——v6full 取证: 策略按自身偏好高度插(~1.15), s63 要底层插成中层,
    # s48"首通"实为顶层撞对。数据侧 st4 prompt 已按行分化重训, 此处与之对齐。
    f"carry the ECU and insert it into the {['bottom', 'middle', 'top'][BAY_ROW]} rack bay",
]

_sseed = int(os.environ.get("ROLLOUT_SPAWN_SEED", "0"))
if _sseed > 0:
    _rng = np.random.default_rng(_sseed)
    d.qpos[JQ:JQ + 3] = [0.459 + float(_rng.uniform(-0.009, 0.012)),
                         -0.006 + float(_rng.uniform(-0.070, 0.070)), 0.9258]
    yaw = float(np.deg2rad(_rng.uniform(-12, 12)))
    d.qpos[JQ + 3:JQ + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    print(f"[hybrid] spawn seed={_sseed} ecu={np.round(d.xpos[JIG],3).tolist()}", flush=True)


# ---------------- shared helpers (policy-side identical to ecu_policy_rollout) ----------------
def grip_frac(arm):
    q = float(np.mean([d.qpos[a] for a in arm.grip_qadr]))
    return float(np.clip(1.0 - (q - ct.GRIP_OPEN) / (ct.GRIP_CLOSE - ct.GRIP_OPEN), 0.0, 1.0))


def state21():
    qL = [float(d.qpos[a]) for a in ct.L.qadr]
    qR = [float(d.qpos[a]) for a in ct.R.qadr]
    return np.array(qL + qR + [grip_frac(ct.L), grip_frac(ct.R)]
                    + list(ct.base_pose()) + list(ct.base_velocity()), dtype=np.float32)


# 送给 policy 的三路相机，须与训练 config 的 cams 顺序一致：
#   cams[0] -> observation/image (base_0_rgb), cams[1] -> left_wrist, cams[2] -> right_wrist
# v5 = 真机契约(无腕相机); v4 = legacy 头+双腕。对照实验用 ROLLOUT_CAMS 切换。
POLICY_CAMS = tuple(os.environ.get(
    "ROLLOUT_CAMS", "stereo_left,stereo_right,waist_front").split(","))
assert len(POLICY_CAMS) == 3, f"ROLLOUT_CAMS 需 3 路，得到 {POLICY_CAMS}"


class CamRig:
    def __init__(self):
        self.renderer = mujoco.Renderer(m, 480, 640)
        self.opt = mujoco.MjvOption()
        # top_head 恒需渲染（录像用），再并上本次策略要的三路
        self.ids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
                    for c in ("top_head",) + POLICY_CAMS}
        for c, i in self.ids.items():
            assert i >= 0, f"场景中无相机 '{c}'"

    def shot(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.ids[name]
        self.renderer.update_scene(d, cam, self.opt)
        return self.renderer.render().copy()


def held():
    for i in range(d.ncon):
        gs = {d.contact[i].geom1, d.contact[i].geom2}
        if gs & set(PADS) and gs & JIG_GEOMS:
            return True
    return False


def ecu():
    return d.xpos[JIG].copy()


def apply_action(a):
    a = np.asarray(a, dtype=float)
    ct.qtgt["l"][:] = np.clip(a[0:7], ct.L.lo, ct.L.hi)
    ct.qtgt["r"][:] = np.clip(a[7:14], ct.R.lo, ct.R.hi)
    ct.grip_cmd["l"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[14], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    ct.grip_cmd["r"] = ct.GRIP_OPEN + (1.0 - float(np.clip(a[15], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
    ct.base_vel[:] = [float(np.clip(a[16], -0.4, 0.4)), float(np.clip(a[17], -0.6, 0.6))]


# ---------------- NavTo (scripted diff-drive = simulation stand-in for SDK A000002) ----------------
VID = None


def frames(n):
    for _ in range(2 * n):
        ct.control_step(SUB)
        # video capture inside NavTo too
    if VID is not None:
        VID.append_data(RIG.shot("top_head"))


def _ang(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def turn_to(yaw_t, tol=0.02, wmax=0.35):
    for _ in range(1500):
        e = _ang(yaw_t - ct.base_pose()[2])
        if abs(e) < tol:
            break
        ct.base_vel[:] = [0.0, float(np.clip(2.0 * e, -wmax, wmax))]
        frames(1)
    ct.base_vel[:] = 0.0
    frames(3)


def drive_axis(axis, target, vmax=0.22, tol=0.015):
    idx = 0 if axis == "x" else 1
    for _ in range(2600):
        yaw = ct.base_pose()[2]
        head = np.array([np.cos(yaw), np.sin(yaw)])
        if abs(head[idx]) < 0.5:
            break
        rem = (target - ct.base_pose()[idx]) / head[idx]
        if abs(rem) < tol:
            break
        ct.base_vel[:] = [float(np.clip(1.5 * rem, -vmax, vmax)), 0.0]
        frames(1)
    ct.base_vel[:] = 0.0
    frames(3)


Q_CARRY_STD = np.array([-0.358, -0.156, 0.371, -1.721, -1.114, 0.024, -1.223])
# 上面是 v4 时代硬编码的搬运姿态。v5 SDK 收紧关节限位后新录的数据里，专家搬运姿态在
# elbow_yaw/wrist_roll 上与此差 2-3 弧度且符号相反（2026-07-17 审计）。secure_grip 强掰
# 到过时常量会经腕关节大翻转甩掉 ECU。ROLLOUT_CARRY_STD 用于验证/覆盖，默认保持原值。
_carry_override = os.environ.get("ROLLOUT_CARRY_STD")
if _carry_override:
    Q_CARRY_STD = np.array([float(x) for x in _carry_override.split(",")])
    assert Q_CARRY_STD.shape == (7,), f"ROLLOUT_CARRY_STD 需 7 个关节值，得到 {Q_CARRY_STD.shape}"
    print(f"[carry] Q_CARRY_STD 覆盖为 {np.round(Q_CARRY_STD,3).tolist()}", flush=True)
# S3 sub-episodes end in their OWN carry pose (audit 2026-07-15: mean over 175 eps,
# per-joint std <= 0.07) - different base pose and grasp than the stand-side carry.
Q_CARRY_S3 = np.array([-0.301, -0.668, 0.248, -1.797, -1.014, -0.438, -1.152])


def qR():
    return np.array([float(d.qpos[a]) for a in ct.R.qadr])


# A/B result (2026-07-15, 6 rollouts + probe sweep): requiring the POLICY to finish the
# arm tuck before stage-done dropped M1 4/4 -> 3/7 - the shallow pinch sheds the plate
# during the extended tuck. secure_grip's 60-frame smoothstep tuck is the safer handover
# (2x2 ablation). Alignment check stays available via ROLLOUT_ALIGN_TOL>0, default OFF.
ALIGN_TOL = float(os.environ.get("ROLLOUT_ALIGN_TOL", "0"))


def arm_near(qref, tol):
    if ALIGN_TOL <= 0:
        return True
    return float(np.abs(qR() - qref).max()) < tol * ALIGN_TOL / 0.45


def secure_grip(q_carry=None):
    """Handover: tuck the carrying arm to the expert standard carry pose (2x2 ablation
    2026-07-15: policy grasp + std carry pose survived transport 2/2 even with a shallow
    pinch; the policy's own raw posture was the drop cause). Grip command untouched.
    q_carry: 目标携带姿态, 默认 Q_CARRY_STD(S1->S2 接缝用)。M3->M4 接缝须用 S3 终点
    姿态(S4 数据起点), 掰向 S1 姿态会把 ECU 甩回夹具(2026-07-19 s103 实证)。"""
    tgt = Q_CARRY_STD if q_carry is None else q_carry
    q0 = ct.qtgt["r"].copy()
    for k in range(60):
        t = (k + 1) / 60
        ct.qtgt["r"][:] = q0 + (tgt - q0) * (t * t * (3 - 2 * t))
        frames(1)
    frames(10)


BRIDGE_ECU_XY = (0.50, -0.77)

# ================= 动态工位导航层 (2026-07-19, 参考真机方案 §3.1) =================
# 真机流程"RGB-D 检测工位→算相对位姿→生成本次停靠位"的仿真版：工位位姿按 body 名
# 实测查询（导航栈的感知代理——策略依然只看相机）。所有停靠点/放置目标/门判据坐标
# = 当前工位位姿 ∘ 名义布局下反解的相对常量。默认布局下与旧行为数学等价（零回归）；
# 工位平移/旋转后自动跟随。ROLLOUT_LAYOUT_RAND=1 启用布局随机化评估。
STATION_BODY = {"fixture": "fa2_precision_machine_part",
                "rack": "fa2_modular_server_drive_rack"}
# 名义工位位姿 (x, y, yaw) —— 取自默认场景实测 (2026-07-19)
STATION_NOM = {"fixture": (0.6471, -0.7739, np.pi),
               "rack": (-2.3694, -0.0399, 0.0)}


def station_pose(key):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, STATION_BODY[key])
    R = d.xmat[b].reshape(3, 3)
    return float(d.xpos[b][0]), float(d.xpos[b][1]), float(np.arctan2(R[1, 0], R[0, 0]))


def _st_rel(key, wx, wy):
    """世界点 -> 名义工位系（相对常量只算一次）。"""
    nx, ny, nyaw = STATION_NOM[key]
    c, s = np.cos(nyaw), np.sin(nyaw)
    dx, dy = wx - nx, wy - ny
    return (c * dx + s * dy, -s * dx + c * dy)


def st_world(key, rel):
    """工位自身系相对常量 -> 当前世界点。默认布局(当前=名义)下精确还原旧世界常量；
    工位平移/旋转后点随工位刚体变换。"""
    x, y, yaw = station_pose(key)
    c, s = np.cos(yaw), np.sin(yaw)
    return (x + c * rel[0] - s * rel[1], y + s * rel[0] + c * rel[1])


def st_heading(key, nominal_heading):
    """名义布局下的朝向 -> 当前布局（跟随工位 yaw 差，wrap 到 (-pi,pi]）。"""
    h = nominal_heading + (station_pose(key)[2] - STATION_NOM[key][2])
    return float((h + np.pi) % (2 * np.pi) - np.pi)


# 相对常量（由旧世界常量在名义布局下反解；数值上默认布局= 旧行为）
REL_BRIDGE = _st_rel("fixture", *BRIDGE_ECU_XY)          # 放置目标 in 夹具系
REL_REGRASP_PARK = _st_rel("fixture", 0.02, -0.60)       # 重抓停靠 in 夹具系
REL_BAY = _st_rel("rack", -2.27, RACK_BAY_Y)             # 目标格位点 in 货架系
RACK_STANDOFF = 0.72                                      # 名义: 格位 -2.27, 停靠 -1.55

# ---- 布局随机化 (ROLLOUT_LAYOUT_RAND=1): 随机平移/旋转夹具与货架, 检验动态导航 ----
# 幅度参考真机方案首轮随机化(±15cm/±15°)取保守子集, 保证现有走廊拓扑仍安全。
if os.environ.get("ROLLOUT_LAYOUT_RAND") == "1":
    _lrng = np.random.default_rng(int(os.environ.get("ROLLOUT_SPAWN_SEED", "0")) + 9973)

    def _yawq(dyaw, q0):
        w1, z1 = np.cos(dyaw / 2), np.sin(dyaw / 2)
        w2, x2, y2, z2 = q0
        return np.array([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2,
                         w1 * y2 + z1 * x2, w1 * z2 + z1 * w2])

    _lay = {}
    for _key, (_dxr, _dyr, _dwr) in {"fixture": (0.08, 0.08, np.deg2rad(8)),
                                     "rack": (0.06, 0.10, np.deg2rad(6))}.items():
        _b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, STATION_BODY[_key])
        _dx, _dy = float(_lrng.uniform(-_dxr, _dxr)), float(_lrng.uniform(-_dyr, _dyr))
        _dw = float(_lrng.uniform(-_dwr, _dwr))
        m.body_pos[_b][0] += _dx
        m.body_pos[_b][1] += _dy
        m.body_quat[_b] = _yawq(_dw, m.body_quat[_b].copy())
        _lay[_key] = (_dx, _dy, _dw)
    # jig2(货架内预置件, 自由体)随货架做同一刚体变换
    _dx, _dy, _dw = _lay["rack"]
    _jq2 = m.jnt_qposadr[ct.jid("jig2_free")] if hasattr(ct, "jid") else None
    try:
        _jq2 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jig2_free")]
    except Exception:
        _jq2 = None
    if _jq2 is not None and _jq2 >= 0:
        _rn = np.array(STATION_NOM["rack"][:2])
        _off = d.qpos[_jq2:_jq2 + 2] - _rn
        _cw, _sw = np.cos(_dw), np.sin(_dw)
        d.qpos[_jq2] = _rn[0] + _dx + _cw * _off[0] - _sw * _off[1]
        d.qpos[_jq2 + 1] = _rn[1] + _dy + _sw * _off[0] + _cw * _off[1]
        _q0 = d.qpos[_jq2 + 3:_jq2 + 7].copy()
        d.qpos[_jq2 + 3:_jq2 + 7] = _yawq(_dw, _q0)
        d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    print(f"[layout] fixture d=({_lay['fixture'][0]*100:+.0f},{_lay['fixture'][1]*100:+.0f})cm "
          f"{np.degrees(_lay['fixture'][2]):+.0f}deg | "
          f"rack d=({_dx*100:+.0f},{_dy*100:+.0f})cm {np.degrees(_dw):+.0f}deg", flush=True)


def bridge_xy():
    return st_world("fixture", REL_BRIDGE)


def drive_to_point(x, y, vmax=0.22, tol=0.01):
    """通用两段腿：转向目标点航向 -> 闭环直驱。替代 axis 对齐 drive 的布局假设。"""
    bp = ct.base_pose()
    hd = float(np.arctan2(y - bp[1], x - bp[0]))
    if np.hypot(x - bp[0], y - bp[1]) > tol:
        turn_to(hd)
        drive_axis("x" if abs(np.cos(hd)) > abs(np.sin(hd)) else "y",
                   x if abs(np.cos(hd)) > abs(np.sin(hd)) else y, tol=tol, vmax=vmax)


def park_at(x, y, yaw, vmax=0.22, tol=0.008):
    drive_to_point(x, y, vmax=vmax, tol=tol)
    turn_to(yaw)


def nav_to_fixture():
    """spawn parking -> fixture manipulation pose. OBJECT-RELATIVE park (baseline 0/6
    autopsy 2026-07-15): the training data parks with the held ECU over the notch mouth
    (expert closed-loop on the object), so a fixed base coordinate leaves the ECU up to
    10cm out of distribution when the policy's grasp offset differs - M2 then sets the
    plate down west of the plate edge, into free air. Same trick the rack leg already
    uses; the real robot would do this with visual servoing."""
    print("[NavTo] fixture", flush=True)
    secure_grip()
    # measure the rigid base->ECU offset once (grasp-dependent constant)
    bp = ct.base_pose()
    c, s = np.cos(bp[2]), np.sin(bp[2])
    e = ecu()[:2] - bp[:2]
    o1 = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
    # 动态版：放置目标/停靠航向按当前夹具实测位姿推导（默认布局=旧常量）。
    tgt = bridge_xy()
    hd = st_heading("fixture", 0.0)             # 名义朝向 +x，随夹具 yaw 跟随
    drive_axis("x", -0.55)                      # 倒离料架（相对当前航向后退）
    # 预停靠点：使携带的 ECU 恰好落在目标后方 0.30m 的接近线上
    ch, sh = np.cos(hd), np.sin(hd)
    pre = (tgt[0] - ch * 0.30 - (ch * o1[0] - sh * o1[1]),
           tgt[1] - sh * 0.30 - (sh * o1[0] + ch * o1[1]))
    drive_to_point(*pre, tol=0.006)
    turn_to(hd)
    # closed loop on the OBJECT: park when the ECU is over the notch mouth
    for vmax in (0.10, 0.06):
        exy = np.array(tgt) - np.array(ecu()[:2])
        rem = ch * exy[0] + sh * exy[1]         # 沿接近航向的剩余距离
        if abs(rem) < 0.008:
            break
        bp = ct.base_pose()
        drive_axis("x" if abs(ch) > abs(sh) else "y",
                   (bp[0] + ch * rem) if abs(ch) > abs(sh) else (bp[1] + sh * rem),
                   tol=0.005, vmax=vmax)


def nav_to_fixture_regrasp():
    print("[NavTo] fixture regrasp pose", flush=True)
    # 原版盲停固定位 (0.02,-0.60)，不看 ECU 实际在哪。但 S3 训练数据里 ECU 距标称
    # (0.50,-0.77) 偏差 ≤~1.3cm（专家门收紧），而策略 M2 放置可偏到 4-5cm——盲停后
    # ECU 在机器人系里的位置对 S3 策略是 OOD（实测 s21: y 偏 3.7cm → M3 抓空）。
    # ROLLOUT_REGRASP_OBJREL=1: 停靠位按 ECU 实测偏移平移，恢复策略眼中的标称几何。
    # 先例：nav_to_fixture 本就有 object 闭环（park when ECU over the notch mouth）；
    # 专家 regrasp 也是按 ecu_pos() 实测算的。此为导航层修复，不动 M 门判据。
    dx = dy = 0.0
    if os.environ.get("ROLLOUT_REGRASP_OBJREL"):
        e = ecu()
        b = bridge_xy()
        dx, dy = float(e[0] - b[0]), float(e[1] - b[1])
        print(f"[NavTo] objrel offset dx={dx*1000:.0f}mm dy={dy*1000:.0f}mm", flush=True)
    # 动态版：停靠位/航向按当前夹具位姿推导（默认布局 = 旧 (0.02,-0.60)+objrel）
    px, py = st_world("fixture", REL_REGRASP_PARK)
    hd = st_heading("fixture", 0.0)
    park_at(px + dx, py + dy, hd, vmax=0.12, tol=0.005)


def nav_to_rack():
    print("[NavTo] rack bay", flush=True)
    # M3->M4 接缝契约: S4 数据起点 = S3 段终点姿态(v5 实测均值, 逐关节 std≤0.05)。
    # 掰向 Q_CARRY_STD(S1 姿态)会把刚重抓的 ECU 甩回夹具(s103 实证: ECU 回落 z=0.834)。
    q_s4 = np.array([-0.117, -0.628, 0.462, -1.844, -1.089, -0.332, -1.048])
    _c = os.environ.get("ROLLOUT_RACK_CARRY_STD")
    if _c:
        q_s4 = np.array([float(x) for x in _c.split(",")])
    secure_grip(q_s4)
    # ANALYTIC route (2026-07-16, mirrors the expert): measure the rigid base->ECU
    # offset ONCE while carrying; at yaw=pi the ECU y = base_y - o2_y, so the y-leg
    # target is exact - no micro-align correction loop (was 4-8 extra turns).
    bp = ct.base_pose()
    c, s_ = np.cos(bp[2]), np.sin(bp[2])
    e = ecu()[:2] - bp[:2]
    o2 = np.array([c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1]])

    # 动态版：格位点/接近航向按当前货架实测位姿推导。停靠位 = 格位 - standoff·dir -
    # 侧向补偿(携带偏移 o2 的垂向分量)。默认布局下与旧 (-1.55, RACK_BAY_Y+o2y, yaw=pi)
    # 数学等价。
    def _rack_park():
        bay = np.array(st_world("rack", REL_BAY))
        hd = st_heading("rack", np.pi)
        di = np.array([np.cos(hd), np.sin(hd)])
        pp = np.array([-di[1], di[0]])
        v = np.array([np.cos(hd) * o2[0] - np.sin(hd) * o2[1],
                      np.sin(hd) * o2[0] + np.cos(hd) * o2[1]])
        base = bay - RACK_STANDOFF * di - float(pp @ v) * pp
        return bay, hd, di, pp, base

    bay, hd, di, pp, base = _rack_park()
    drive_axis("x", -0.60, vmax=0.2)            # 走廊 lane（世界常量，随机化范围内安全）
    drive_to_point(*(base - 0.35 * di), tol=0.006)
    turn_to(hd)
    drive_axis("x" if abs(di[0]) > abs(di[1]) else "y",
               base[0] if abs(di[0]) > abs(di[1]) else base[1], tol=0.008)
    # 侧向误差兜底：按实测 ECU 相对格位的垂向偏差重停一次
    lat = float(pp @ (np.array(ecu()[:2]) - bay))
    if abs(lat) > 0.015:
        bay, hd, di, pp, base = _rack_park()
        park_at(*(base - 0.25 * di), hd, vmax=0.12, tol=0.005)
        drive_axis("x" if abs(di[0]) > abs(di[1]) else "y",
                   base[0] if abs(di[0]) > abs(di[1]) else base[1], tol=0.008, vmax=0.10)


# ---------------- policy manipulation stage ----------------
def run_stage(client, stage, done_fn, timeout_s):
    print(f"[stage {stage+1}] '{STAGE_PROMPTS[stage]}'", flush=True)
    chunk, k = None, 0
    done_run = 0
    for step in range(int(timeout_s * 30)):
        top = RIG.shot("top_head")
        if step % REPLAN == 0 or chunk is None or k >= len(chunk):
            obs = {
                "observation/state": state21(),
                "observation/image": RIG.shot(POLICY_CAMS[0]),
                "observation/left_wrist_image": RIG.shot(POLICY_CAMS[1]),
                "observation/right_wrist_image": RIG.shot(POLICY_CAMS[2]),
                "prompt": STAGE_PROMPTS[stage],
            }
            chunk = np.asarray(client.infer(obs)["actions"])
            k = 0
        apply_action(chunk[min(k + SKIP, len(chunk) - 1)])
        k += 1
        for _ in range(2):
            ct.control_step(SUB)
        if VID is not None:
            VID.append_data(top)
        if step % 150 == 0:
            print(f"  [dbg t={step/30:.0f}s] ecu={np.round(ecu(),3).tolist()} held={held()} "
                  f"padmid_d={np.linalg.norm((padmid()-ecu())[:2]):.3f} ncon_pad="
                  f"{sum(1 for i in range(d.ncon) if {d.contact[i].geom1,d.contact[i].geom2} & set(PADS))}",
                  flush=True)
        if done_fn():
            done_run += 1
            if done_run >= 15:
                # freeze the arm AT the done posture (qtgt stays); running the policy past
                # its data boundary (post-lift) de-stabilizes the hold - measured.
                ct.base_vel[:] = 0.0
                print(f"[stage {stage+1}] DONE at {step/30:.1f}s  ecu={np.round(ecu(),3).tolist()}", flush=True)
                return True
        else:
            done_run = 0
    ct.base_vel[:] = 0.0
    print(f"[stage {stage+1}] TIMEOUT  ecu={np.round(ecu(),3).tolist()} held={held()}", flush=True)
    return False


def padmid():
    return (d.geom_xpos[PADS[0]] + d.geom_xpos[PADS[1]]) / 2.0


def retreat_up(dz=0.05, nframes=25):
    """镜像专家放置后的 +z 撤臂（ecu_expert_record.py: move_mount_cart([0,0,0.05])）。
    eval 原本缺这一步：M2 结束手臂以深放置姿态冻结，随后 regrasp 导航原地转 pi/2，
    伸展的手臂扫过夹具区域可能碰掉刚放好的 ECU；且数据里 st3 起点是撤离后的姿态，
    不撤离的起点对策略是 OOD。ROLLOUT_RETREAT=1 启用（A/B 验证用，默认关）。"""
    R = ct.R
    start = d.xpos[R.mount].copy()
    rd = d.xmat[R.mount].reshape(3, 3).copy()
    for k in range(nframes):
        t = (k + 1) / nframes
        tgt = start + np.array([0.0, 0.0, dz]) * (t * t * (3 - 2 * t))
        qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
        for i, adr in enumerate(R.qadr):        # seed IK from current command
            d.qpos[adr] = ct.qtgt["r"][i]
        ct.ik(R, tgt, rd, iters=40, w=0.6)
        q = np.array([d.qpos[a] for a in R.qadr])
        d.qpos[:], d.qvel[:] = qpos0, qvel0
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["r"][:] = q
        for _ in range(2):
            ct.control_step(SUB)
    print(f"[retreat] mount +{dz*100:.0f}cm done, ecu={np.round(ecu(),3).tolist()}", flush=True)


def grip_quality_ok():
    """deep enough pinch: pad midpoint within 6cm of the ECU centre horizontally."""
    return float(np.linalg.norm((padmid() - ecu())[:2])) < 0.075   # edge pinch design value ~0.064


def move_mount(delta, nframes):
    """cartesian mount move via ct.ik (scripted, expert-style)."""
    start = d.xpos[ct.R.mount].copy()
    rd = d.xmat[ct.R.mount].reshape(3, 3).copy()
    delta = np.asarray(delta, float)
    for k in range(nframes):
        t = (k + 1) / nframes
        tgt = start + delta * (t * t * (3 - 2 * t))
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        for i, adr in enumerate(ct.R.qadr):
            d.qpos[adr] = ct.qtgt["r"][i]
        ct.ik(ct.R, tgt, rd, iters=40, w=0.6)
        q = np.array([d.qpos[a] for a in ct.R.qadr])
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["r"][:] = q
        frames(1)


def reseat_grip():
    """Deepen a shallow pinch (real-robot re-seat): lower the ECU back to the stand,
    open, advance 30mm along the pinch axis, re-close, lift again. Fully scripted."""
    print(f"[reseat] shallow pinch d={np.linalg.norm((padmid()-ecu())[:2]):.3f}; re-seating", flush=True)
    drop = ecu()[2] - 0.9235                     # ECU truly seated on the stand
    if drop > 0:
        move_mount([0, 0, -drop], 30)
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    frames(15)
    horiz = (ecu() - padmid())[:2]
    horiz = horiz / (np.linalg.norm(horiz) + 1e-9)
    move_mount([-0.008 * horiz[0], -0.008 * horiz[1], 0.0], 8)   # detach first
    move_mount([0.043 * horiz[0], 0.043 * horiz[1], 0.0], 25)    # then advance deeper
    ct.grip_cmd["r"] = ct.GRIP_CLOSE
    frames(20)
    move_mount([0, 0, 0.05], 30)
    print(f"[reseat] after: d={np.linalg.norm((padmid()-ecu())[:2]):.3f} held={held()} "
          f"ecu={np.round(ecu(),3).tolist()}", flush=True)


def reset_arm_droop():
    droop = np.array([0.0, 0.0, 0.0, -0.35, 0.0, 0.0, 0.0])
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    q0 = ct.qtgt["r"].copy()
    for k in range(45):
        t = (k + 1) / 45
        ct.qtgt["r"][:] = q0 + (droop - q0) * (t * t * (3 - 2 * t))
        frames(1)
    frames(10)


# S1 就绪位契约（2026-07-18，与 ecu_expert_record.goto_ready_pose 配对）：
# v6 起 S1 数据从料架前方的固定就绪位开始（droop->pre 的物理转移会钩挂升降连杆，
# 已移出数据）。eval 侧 M1 前必须进入同一就绪位，否则策略起点 OOD。
# 就绪位由录制器写入 out/smoke/_s1_ready_pose.json，两侧共享同一常量。
_READY_JSON = os.path.join(HERE, "..", "out", "smoke", "_s1_ready_pose.json")


# S3 重抓起点契约（2026-07-19）：S3 数据起点 = 专家"放置+5cm撤臂"后的姿态
# （267 段实测均值，逐关节 std≤0.25）。eval 的 M3 原先从"放置深姿态冻结"(attempt1)
# 或"垂臂"(retry) 开始——数据从没见过这两种起点。v6 32 组 M3 全灭且 ECU 全部原位、
# 停靠已补偿 → 位置 OOD 已排除，起点姿态是首要嫌疑。与 M1 就绪位契约同构。
Q_S3_READY = np.array([0.077, -0.097, 0.489, -1.963, 1.745, -0.351, 1.104])


def reset_arm_s3ready():
    """kinematic 置位到 S3 数据起点姿态（含本体接触校验）。"""
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    for i, adr in enumerate(ct.R.qadr):
        d.qpos[adr] = Q_S3_READY[i]
    ct.qtgt["r"][:] = Q_S3_READY
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    bad = []
    for i in range(d.ncon):
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]) or "?"
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]) or "?"
        if (b1.startswith("R_") != b2.startswith("R_")) and "jig" not in (b1, b2):
            bad.append(f"{b1}~{b2}")
    if bad:
        print(f"[s3ready] WARN 置位后存在本体接触: {bad}", flush=True)
    for _ in range(30):
        ct.control_step(SUB)
    print(f"[s3ready] M3 起点置位完成 ecu={np.round(ecu(),3).tolist()}", flush=True)


def reset_arm_ready():
    """kinematic 置位到 S1 就绪位（镜像录制器：qpos 直写 + 物理稳定）。"""
    import json as _json
    q_ready = np.array(_json.load(open(_READY_JSON))["q_ready"])
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    for i, adr in enumerate(ct.R.qadr):
        d.qpos[adr] = q_ready[i]
    ct.qtgt["r"][:] = q_ready
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(30):
        ct.control_step(SUB)
    print(f"[ready] M1 就绪位置位完成 ecu={np.round(ecu(),3).tolist()}", flush=True)


def main():
    global RIG, VID
    import imageio.v2 as imageio
    os.makedirs(OUT, exist_ok=True)
    RIG = CamRig()
    VID = imageio.get_writer(os.path.join(OUT, "hybrid_top_head.mp4"), fps=30)
    # per-stage policy endpoints: POLICY_PORT_M1..M4 override the shared POLICY_PORT,
    # so each manipulation stage can use its own checkpoint (mixture-of-experts FSM)
    _clients = {}

    def stage_client(i):
        p = int(os.environ.get(f"POLICY_PORT_M{i+1}", PORT))
        if p not in _clients:
            _clients[p] = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=p)
        return _clients[p]

    client = stage_client(0)
    print(f"[hybrid] bay=({BAY_ROW},{BAY_COL})", flush=True)

    results = {}
    try:
        # M1: pick from stand - up to 3 tries, with a grip-quality gate (a shallow pinch
        # survives the lift but dies in transport; the real robot would retry too)
        results["M1"] = False
        # 就绪位契约(v6+): M1 每次尝试都从与 S1 数据一致的就绪位开始。
        # ROLLOUT_READY_START=0 回退 droop 起点（评估 v5 及更早的旧档用）。
        _ready = os.environ.get("ROLLOUT_READY_START", "1") == "1"
        for attempt in range(3):
            if _ready:
                reset_arm_ready()
            # done = data S1 end-state: held + lifted + arm tucked to the carry pose
            # (cutting at held+lift interrupted the policy's learned tuck - FSM fix)
            ok = run_stage(stage_client(0), 0,
                           lambda: held() and ecu()[2] > 0.955 and arm_near(Q_CARRY_STD, 0.45),
                           STAGE_TIMEOUT)
            if ok:
                results["M1"] = True
                break
            if ecu()[2] < 0.5:
                break                      # dropped to the floor: unrecoverable
            print(f"[retry] M1 attempt {attempt+1} failed; resetting arm", flush=True)
            if not _ready:
                reset_arm_droop()
        if results["M1"]:
            nav_to_fixture()
            # M2: place on fixture bridge
            results["M2"] = run_stage(
                stage_client(1), 1,
                lambda: (not held()) and 0.80 < ecu()[2] < 0.90
                and abs(ecu()[0] - bridge_xy()[0]) < 0.05
                and abs(ecu()[1] - bridge_xy()[1]) < 0.04,
                STAGE_TIMEOUT)
        if results.get("M2"):
            if os.environ.get("ROLLOUT_RETREAT"):
                retreat_up()
            nav_to_fixture_regrasp()
            # M3: pick from fixture - 2 tries
            results["M3"] = False
            # S3 起点契约 (ROLLOUT_S3READY=1, 默认开): 每次尝试从 S3 数据实测起点姿态开始
            _s3r = os.environ.get("ROLLOUT_S3READY", "1") == "1"
            for attempt in range(2):
                if _s3r:
                    reset_arm_s3ready()
                ok = run_stage(stage_client(2), 2,
                               lambda: held() and ecu()[2] > 0.87 and arm_near(Q_CARRY_S3, 0.60),
                               STAGE_TIMEOUT)
                if ok:
                    results["M3"] = True
                    break
                if ecu()[2] < 0.5:
                    break
                print(f"[retry] M3 attempt {attempt+1} failed; resetting arm", flush=True)
                reset_arm_droop()
        if results.get("M3"):
            nav_to_rack()
            # M4: insert into bay
            zlo = RACK_BAY_FLOOR + 0.0128 - 0.01

            def _m4_ok():
                # 动态门：沿货架接近方向的进深 + 垂向偏差（默认布局 = 旧 x<-2.16/|y-bay|<0.12）
                bay = np.array(st_world("rack", REL_BAY))
                hd = st_heading("rack", np.pi)
                di = np.array([np.cos(hd), np.sin(hd)])
                pp = np.array([-di[1], di[0]])
                rel = np.array(ecu()[:2]) - bay
                return (not held()) and float(rel @ di) > -0.11 \
                    and abs(float(rel @ pp)) < 0.12 and zlo < ecu()[2] < zlo + 0.05

            results["M4"] = run_stage(stage_client(3), 3, _m4_ok, STAGE_TIMEOUT + 15)
    finally:
        VID.close()

    print("\n=== HYBRID RESULT ===")
    for k in ("M1", "M2", "M3", "M4"):
        print(f"  {k}: {results.get(k, '-')}")
    p = ecu()
    print(f"  final ecu: {np.round(p,3).tolist()}  held={held()}")
    print(f"  video: {os.path.abspath(os.path.join(OUT,'hybrid_top_head.mp4'))}", flush=True)


if __name__ == "__main__":
    main()
