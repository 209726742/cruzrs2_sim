#!/usr/bin/env python3
"""Keyboard tele-operation of the CRUZR bimanual PGC-140 rig in MuJoCo.

逐关节遥操（键位与 Isaac 版 run_teleop_cruzr.py 完全一致）：每个手臂的 7 个关节由
qwerty 两排键逐关节 nudge（上排 + / 下排 −），夹爪是对 5mm 壁 tote 的真摩擦夹持。抓取初始
姿态仍由阻尼最小二乘 IK 摆到「双臂对称、指向下」的预抓取位（measure-after-settle），之后
运行时不再解 IK，纯逐关节位置伺服——和 Isaac 手感统一。

输入走【终端 tty】（和 Isaac 一样：焦点放在本终端，不是 MuJoCo 图形窗口）。这修掉了之前
「按键没反应」——旧版从 GLFW 窗口读键，需焦点在图形窗；现在焦点在终端即可。

Physics notes carried over (do not revert):
  * CRUZR lifter/waist/head 关节在 assets/cruzr_pgc140.xml 里已焊死刚体（URDF 中未驱动，
    否则 ~40kg 上身会在重力下塌陷）。身体键 i/k/o/l/p/;/n/m 仍无对应关节 → 忽略。
  * 底盘【已可动】(2026-07-07)：chassis body = slide-x + slide-y + hinge-yaw 平面机构 +
    position actuator（见 cruzr_pgc140.xml 注释）。驱动端做【差速运动学】：8/2 设前进速度
    （沿航向）、7/9 设 yaw 率、0 急停（冻结目标=位置保持零漂移）；4/6 平移对差速底盘是
    螃蟹步（物理上做不到）→ 忽略并打印提示（用 7/9 转向 + 8/2 前进组合到达）。
    臂 IK 目标 = 底盘系常量，每次 IK 时按底盘位姿变换到世界系（开走后仍可抓）。
  * tote 直接 BAKE 进 cruzr_newusd_scene.xml（由 build_newusd_scene.py 从 new.usd 抽出重建，
    真实 KLT 尺寸 0.278×0.3445×0.146）；只平移其 body。运行时改 geom_pos/size 会破坏
    broadphase 并让指腹穿薄壁。

================================  KEY MAP（= Isaac run_teleop_cruzr.py）  ==============
  臂逐关节（激活手，上排 + / 下排 −，步进 CRUZR_DARM 默认 0.02 rad）
    肩pitch q/a   肩roll w/s   肩yaw e/d   肘roll r/f   肘yaw t/g   腕pitch y/h   腕roll u/j
  末端位姿模式（按 v 在 joint/ee 间切换；激活手，步进 CRUZR_DEE_POS 默认 0.01m）
    末端平移 x q/a   y w/s   z e/d
    末端模式下 t/g y/h u/j 仍直接控制最后 3 个关节：肘yaw、腕pitch、腕roll
  夹爪（激活手）
    SPACE : 开/合切换       [ : 全张(OPEN)       ] : 全合(CLOSE)
  切手 / 杂项
    TAB : 切换激活手(左<->右)     m : 双臂同时控制开关
    = : 重置臂关节目标到实测     \\ : 打印状态
    c   : 打印当前 jig/pad/support/table 相关接触对（碰撞调试）
    b   : 身体锁开关（本平台身体已焊死，仅提示）
  录制（EGL 离屏 GPU，默认跟随 MUJOCO_EGL_DEVICE_ID；默认 OFF，不开则纯遥操）
    z : 开始/继续录制     x : 暂停录制（已录帧保留）
    ENTER : 保存 episode 并结束（TELEOP_RECORD=<name>；默认 outputs/teleop，不可写则 fallback 到 mujoco_teleop/out/teleop）
    BACKSPACE : 丢弃录制并结束（删除该目录）
    录到 3 相机 JPG(top_head/hand_left/hand_right) + 16 维双臂/夹爪 open_frac state/action
    + 底盘 base/base_velocity/base_action + meta.json；历史 scripts/archive/ecu/build_carton_lerobot.py 会
    转成固定基座 16D 或 mobile-base 21D/18D LeRobot schema。
  底盘（差速，= Isaac 步进/限幅：±0.05 m/s /键 上限 0.40；±0.10 rad/s /键 上限 0.60）
    8/2 前进/后退(沿航向)   7/9 左/右转   0 急停(目标冻结,零漂移)
    4/6 平移：差速底盘无侧移(禁螃蟹步)→忽略+提示
  身体（Isaac 有、MuJoCo 焊死→忽略，仅提示）
    i/k 升降  o/l 腰  p/; 头偏  n/m 头俯
======================================================================================

Run interactively（焦点放这个终端）:
  ── 默认(安全) = launch_passive viewer（该用户已验证可用；不含降 pan 灵敏度）──
    MUJOCO_GL=glfw /data1/hsr/tools/miniconda3/envs/mjx/bin/python scripts/core/cruzr_teleop.py
  ── 要降低鼠标平移(pan)灵敏度 = 自定义 GLFW viewer（TELEOP_VIEWER=glfw）──
    TELEOP_VIEWER=glfw MUJOCO_GL=glfw /data1/hsr/tools/miniconda3/envs/mjx/bin/python scripts/core/cruzr_teleop.py

Viewer 选择（env TELEOP_VIEWER）:
  * passive (DEFAULT) → mujoco.viewer.launch_passive（该用户实测可用，只是 FPS 偏低）。
  * glfw              → 自定义 GLFW 渲染循环，可用 PAN_SENS/ROT_SENS/ZOOM_SENS 调鼠标灵敏度。
                        已加固上下文创建(不请求 core-profile / forward-compat，mjr 要 legacy GL)，
                        且 Python 级异常会自动回退到 passive；但硬 segfault 无法在 Python 捕获，
                        所以默认走 passive。
  * egl               → GPU 加速（推荐给本用户）。本机 onscreen GLX 是 Mesa llvmpipe(软件/CPU)
                        所以 passive/glfw 的 3D 都在 CPU 上画=卡；但 EGL 离屏渲染直接命中
                        RTX 4090。此模式用 MUJOCO_EGL_DEVICE_ID/TELEOP_EGL_GPU 指定的 GPU
                        通过 mujoco.Renderer(EGL) 离屏渲染每帧，
                        再把 RGB 帧 blit 进一个 cv2 2D 窗口（2D 贴图对 llvmpipe 也很便宜）。
                        无需 VirtualGL / sudo / PRIME。键盘遥操仍走【终端 stdin】；cv2 窗口只给
                        鼠标控相机(左拖=旋转 右拖/Shift+左拖=平移 滚轮=缩放 ESC=退出)。分辨率
                        env EGL_W/EGL_H(默认 1280x720)。启动脚本内部自动设 MUJOCO_GL=egl —
                        【不要】自己再设 MUJOCO_GL=glfw / 任何 PRIME 变量。
    TELEOP_VIEWER=egl TELEOP_FPS=60 /data1/hsr/tools/miniconda3/envs/mjx/bin/python scripts/core/cruzr_teleop.py

【不要】默认加 PRIME 变量：__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia 在该用户
显示器上会让自定义 GLFW viewer 段错误(core dump)。仅作为「强制独显」的实验选项，可能 segfault。
"""
import os, sys, json, select, termios, tty, threading, numpy as np
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_CORE_DIR)
_PKG_ROOT = os.path.dirname(_SCRIPTS_DIR)
SAFE_VLA_ROOT = os.path.dirname(_PKG_ROOT)
_ARCHIVED_ECU_DIR = os.path.join(_SCRIPTS_DIR, "archive", "ecu")
# The package and repository roots keep standalone-package and source-tree imports stable.
# The archived ECU directory is retained only for the optional legacy factory-assets loader.
for _p in (SAFE_VLA_ROOT, _PKG_ROOT, _ARCHIVED_ECU_DIR, _CORE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sim.cruzr_grip_control import grip_open_fraction
# --- EGL backend MUST be selected BEFORE `import mujoco` ---
# MuJoCo resolves its GL backend the first time its render module is imported (importing
# mujoco.viewer pulls in glfw and can lock the platform), so for TELEOP_VIEWER=egl we set
# MUJOCO_GL=egl + device id into the environment here, ahead of the mujoco import below. This is
# the ONLY place early enough; a later os.environ set is silently too late (glfw wins first).
if os.environ.get("TELEOP_VIEWER", "passive").strip().lower() == "egl":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get(
        "MUJOCO_EGL_DEVICE_ID",
        os.environ.get("TELEOP_EGL_GPU", "0"),
    )
import mujoco, mujoco.viewer
from teleop_timing import FramePacer, control_substeps_for_fps

# ============================ VIEW MOUSE SENSITIVITY (edit me) ============================
# mujoco.viewer.launch_passive exposes NO API to scale mouse sensitivity, so this file uses
# a minimal custom GLFW viewer (standard mujoco glfw pattern: mjv_moveCamera in the cursor
# callback). These three constants scale the per-frame mouse motion fed to mjv_moveCamera:
#   PAN_SENS  scales ONLY the pan/translate drag (mjMOUSE_MOVE_H / mjMOUSE_MOVE_V) — the
#             right-button (or shift+left) drag that slides the view. Default 0.3 = ~1/3 of
#             MuJoCo's stock feel, so panning is calm instead of flying across the scene.
#   ROT_SENS  scales the orbit/rotate drag (left button). 1.0 = stock feel.
#   ZOOM_SENS scales the scroll-wheel zoom. 1.0 = stock feel.
PAN_SENS = float(os.environ.get("TELEOP_PAN_SENS", "0.3"))   # translate sensitivity (~1/3 default)
ROT_SENS = float(os.environ.get("TELEOP_ROT_SENS", "1.0"))   # rotate/orbit sensitivity
ZOOM_SENS = float(os.environ.get("TELEOP_ZOOM_SENS", "1.0")) # scroll-zoom sensitivity
# TELEOP_VIEWER selects the viewer backend. DEFAULT = "passive" (mujoco.viewer.launch_passive,
# the path this user has verified works on their display). "glfw" = the custom GLFW render loop
# above, which honours PAN_SENS/ROT_SENS/ZOOM_SENS. The custom viewer once SEGFAULTED at native
# glfw window/context creation on this user's display (uncatchable from Python), so passive is the
# safe default and the user is never blocked; opt into glfw only to lower pan sensitivity.
VIEWER = os.environ.get("TELEOP_VIEWER", "passive").strip().lower()
# TELEOP_VIEWER=egl → GPU-accelerated EGL offscreen render + lightweight 2D window (see
# run_viewer_egl). The egl+device2 backend selection is done at the very top of this file
# (before `import mujoco`) — MuJoCo locks its GL backend at import time, so setting it here
# would be too late. EGL_W/EGL_H size the offscreen framebuffer (GPU render resolution).
EGL_W = int(os.environ.get("EGL_W", "1280"))     # offscreen GPU render width
EGL_H = int(os.environ.get("EGL_H", "720"))      # offscreen GPU render height
EGL_FAST = os.environ.get("TELEOP_EGL_FAST", "0").strip() == "1"
TARGET_FPS = float(os.environ.get("TELEOP_FPS", "60"))
CONTROL_SUBSTEPS_ENV = os.environ.get("TELEOP_SUBSTEPS")
# =========================================================================================

# TELEOP_SCENE_XML 覆盖：直接加载指定场景 xml（跳过 factory_assets 注入）。用于自建场景
# （如 cruzr_shelf_scene.xml）。缺省时行为与旧版完全一致。
_SCENE_OVERRIDE = os.environ.get("TELEOP_SCENE_XML")
XML = _SCENE_OVERRIDE or os.path.join(_PKG_ROOT, "assets", "cruzr_newusd_scene.xml")
# 工厂道具（machinery / jig_base / server_rack）：assets/factory/manifest.json 存在时以
# 【静态 body】注入（位姿读 assets/factory/placement.json，缺省用内置默认位；摆位用
# scripts/place_assets.py）。manifest 缺失时回退到裸场景 —— 行为与旧版完全一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if _SCENE_OVERRIDE:
    m = mujoco.MjModel.from_xml_path(XML)
    print(f"[teleop] TELEOP_SCENE_XML -> {XML}", flush=True)
else:
    try:
        import factory_assets as _FA
        m = _FA.load_model(free=False)
    except Exception as _e:
        print(f"[teleop] factory assets unavailable ({_e}) -> bare scene", flush=True)
        m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
DEBUG_COLLISION = os.environ.get("TELEOP_DEBUG_COLLISION", "0") == "1"
DEBUG_CONTACTS = os.environ.get("TELEOP_DEBUG_CONTACTS", "0") == "1"
CONTROL_SUBSTEPS = (
    int(CONTROL_SUBSTEPS_ENV)
    if CONTROL_SUBSTEPS_ENV
    else control_substeps_for_fps(TARGET_FPS, m.opt.timestep)
)

def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
def aid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

def _new_scene_option():
    opt = mujoco.MjvOption()
    if DEBUG_COLLISION:
        opt.geomgroup[3] = 1
        if DEBUG_CONTACTS:
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    return opt

def _set_debug_geom_rgba(name, rgba):
    idx = gid(name)
    if idx >= 0:
        m.geom_rgba[idx] = rgba

def _hide_jig_visual_geoms():
    jig_body = bid("jig")
    if jig_body < 0:
        return
    for geom_id in range(m.ngeom):
        if m.geom_bodyid[geom_id] != jig_body:
            continue
        if m.geom_contype[geom_id] == 0 and m.geom_conaffinity[geom_id] == 0:
            m.geom_rgba[geom_id, 3] = 0.0

def _apply_debug_collision_visuals():
    if not DEBUG_COLLISION:
        return
    _hide_jig_visual_geoms()
    for name in ("L_pad1", "L_pad2", "R_pad1", "R_pad2"):
        _set_debug_geom_rgba(name, [1.0, 0.18, 0.02, 0.55])
    _set_debug_geom_rgba("jig_plate", [0.10, 0.45, 1.0, 0.35])
    _set_debug_geom_rgba("support_col0", [1.0, 0.85, 0.05, 0.30])
    contact_msg = " plus contact points/forces" if DEBUG_CONTACTS else ""
    print("[teleop] TELEOP_DEBUG_COLLISION=1: showing normal scene visuals + collision group 3"
          f"{contact_msg}; jig visual-only mesh is hidden.", flush=True)

def _geom_name(idx):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, idx) or f"geom{idx}"

def _print_relevant_contacts(limit=50):
    mujoco.mj_forward(m, d)
    keys = ("jig", "pad", "support", "table")
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = _geom_name(c.geom1), _geom_name(c.geom2)
        if not any(k in g1 or k in g2 for k in keys):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        rows.append((i, g1, g2, float(c.dist), c.pos.copy(), force.copy()))
    if not rows:
        print(f"[contact] no contacts matching {keys}; total ncon={d.ncon}", flush=True)
        return
    print(f"[contact] showing {min(len(rows), limit)}/{len(rows)} relevant contacts "
          f"(total ncon={d.ncon})", flush=True)
    for i, g1, g2, dist, pos, force in rows[:limit]:
        print(f"[contact] #{i:02d} {g1} <-> {g2} dist={dist:+.5f} "
              f"pos={np.round(pos,4).tolist()} normal_force={force[0]:.2f}", flush=True)

_apply_debug_collision_visuals()

class Arm:
    def __init__(self, jnames, mount, pads, grip_act, arm_acts):
        self.jid = [jid(n) for n in jnames]
        self.dof = [m.jnt_dofadr[j] for j in self.jid]
        self.qadr = [m.jnt_qposadr[j] for j in self.jid]
        self.lo = np.array([m.jnt_range[j, 0] for j in self.jid])
        self.hi = np.array([m.jnt_range[j, 1] for j in self.jid])
        self.mount = bid(mount)
        self.p1, self.p2 = gid(pads[0]), gid(pads[1])
        self.grip = aid(grip_act)
        side = pads[0].split("_", 1)[0]
        self.grip_qadr = [m.jnt_qposadr[jid(f"{side}_finger1_joint")],
                          m.jnt_qposadr[jid(f"{side}_finger2_joint")]]
        self.arm_acts = [aid(a) for a in arm_acts]
        self.tgt_pos = np.zeros(3)          # target mount position (world) — 仅初始 IK 用
        self.tgt_mat = np.eye(3)            # target mount orientation      — 仅初始 IK 用
    def padmid(self):
        return (d.geom_xpos[self.p1] + d.geom_xpos[self.p2]) / 2

R = Arm(["R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
         "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint"],
        "R_pgc140_mount", ("R_pad1", "R_pad2"), "a_grip",
        ["a_sp", "a_sr", "a_sy", "a_er", "a_ey", "a_wp", "a_wr"])
L = Arm(["L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
         "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint"],
        "L_pgc140_mount", ("L_pad1", "L_pad2"), "a_Lgrip",
        ["a_Lsp", "a_Lsr", "a_Lsy", "a_Ler", "a_Ley", "a_Lwp", "a_Lwr"])
ARMS = {"r": R, "l": L}

# manipuland: the "jig" body (precision_jig_base, replaces the old tote 2026-07-07).
# vars keep the historical names tote/tq for minimal churn in the tests that import
# this module (base_drive_test uses T.jig alias below).
tote = bid("jig"); tq = m.jnt_qposadr[jid("jig_free")]
jig = tote; jq = tq                                   # descriptive aliases
R_DES = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)   # fingers down, close along world Y

# ---------------- mobile base (diff-drive, 2026-07-07) ----------------
# chassis 平面机构 qpos = (base_x, base_y, base_yaw)，slide 轴恒为世界系（hinge 列最后）。
# 驱动 = Isaac run_teleop_cruzr.py 同款「速度积分目标 + 位置伺服」：base_vel=(v_fwd, wz)，
# 每控制拍 tgt.x += cos(yaw)*v*dt, tgt.y += sin(yaw)*v*dt, tgt.yaw += wz*dt（差速运动学，
# 平移只沿航向 → 禁螃蟹步），position actuator 出力经物理步进跟踪；0 键清速度=目标冻结，
# 停死无漂移（抓取反作用推不动 kp=6e4 的保持）。
BJ = [jid(n) for n in ("base_x", "base_y", "base_yaw")]
BQ = [m.jnt_qposadr[j] for j in BJ]
BV = [m.jnt_dofadr[j] for j in BJ]
BASE_ACTS = [aid(n) for n in ("a_base_x", "a_base_y", "a_base_yaw")]
DVEL = float(os.environ.get("TELEOP_DVEL", "0.05"))    # m/s /键   (= Isaac)
DWZ = float(os.environ.get("TELEOP_DWZ", "0.10"))      # rad/s /键 (= Isaac)
VMAX = float(os.environ.get("TELEOP_VMAX", "0.40"))    # (= Isaac)
WZMAX = float(os.environ.get("TELEOP_WZMAX", "0.60"))  # (= Isaac)
base_vel = np.zeros(2)                                 # (v_fwd, wz) 底盘系
base_tgt = np.array([d.qpos[a] for a in BQ], float)    # 积分目标 (x, y, yaw) 世界系

def base_pose():
    """measured (x, y, yaw) of the chassis in world frame."""
    return np.array([d.qpos[BQ[0]], d.qpos[BQ[1]], d.qpos[BQ[2]]], float)

def base_velocity():
    """Measured base velocity as policy-facing (forward speed, yaw rate)."""
    yaw = float(d.qpos[BQ[2]])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = float(d.qvel[BV[0]]), float(d.qvel[BV[1]])
    return np.array([c * vx + s * vy, float(d.qvel[BV[2]])], float)

def base_world(p_local):
    """chassis-frame point -> world (z passes through: no base z DOF)."""
    x, y, yaw = base_pose(); c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x + c*p_local[0] - s*p_local[1],
                     y + s*p_local[0] + c*p_local[1], p_local[2]])

def base_rotz():
    yaw = base_pose()[2]; c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

# 夹爪 ctrl：DH PGC-140-50 行程 50mm，总行程由两个手指各 25mm 贡献。
# 官方/Isaac 约定：GRIP_OPEN=0.0（张开），GRIP_CLOSE=0.025（闭合）。
_gr = m.actuator_ctrlrange[R.grip]
GRIP_OPEN, GRIP_CLOSE = float(_gr[0]), float(_gr[1])
if GRIP_CLOSE <= GRIP_OPEN:                       # 未限幅兜底
    GRIP_OPEN, GRIP_CLOSE = 0.0, 0.03
# Precision jig side-pinch should not drive the PGC-140-50 to the hard 0.025 m endpoint:
# the rigid thin plate over-compresses in MuJoCo and slips during carrying. 0.021 m is the
# validated jig-safe default; the full actuator ctrlrange remains 0..0.025 for explicit sweeps.
GRIP_CLOSE = min(GRIP_CLOSE, float(os.environ.get("CRUZR_GRIP_CLOSE", "0.021")))
GRIP_MID = 0.5 * (GRIP_OPEN + GRIP_CLOSE)
DARM = float(os.environ.get("CRUZR_DARM", "0.02"))   # rad/键 关节步进（= Isaac 默认）
DEE_POS = float(os.environ.get("CRUZR_DEE_POS", "0.01"))  # m/键 末端平移步进
DEE_JOINT_MAX = float(os.environ.get("CRUZR_DEE_JOINT_MAX", "0.04"))  # rad/键 EE IK 关节限幅
DEE_DAMPING = float(os.environ.get("CRUZR_DEE_DAMPING", "0.06"))      # 阻尼越大越不甩
# 夹爪限速：真 PGC-140 全程 ~0.4s。旧版 ctrl 直接跳变 → kp3000 舵机 0.8m/s 拍上 5mm 壁，
# 瞬态穿透 ~3.9mm（用户可见）；限速后瞬态 ≤~1mm。
GRIP_RATE = float(os.environ.get("CRUZR_GRIP_RATE", "0.075"))   # ctrl 单位/秒

def set_gripper_state(arm, value):
    """Synchronize gripper actuator command and both finger qpos.

    MuJoCo position actuators do not move qpos during mj_forward().  Sidegrasp
    starts with the fingers already around the jig, so ctrl=OPEN but qpos=CLOSE
    creates a first-step collision impulse that launches the manipuland.
    """
    value = float(np.clip(value, GRIP_OPEN, GRIP_CLOSE))
    d.ctrl[arm.grip] = value
    for adr in arm.grip_qadr:
        d.qpos[adr] = value

def rot_err(Rd, Rc):
    Re = Rd @ Rc.T; q = np.zeros(4); mujoco.mju_mat2Quat(q, Re.flatten())
    ang = 2*np.arccos(np.clip(q[0], -1, 1)); v = q[1:]; n = np.linalg.norm(v)
    return v/n*ang if n > 1e-9 else np.zeros(3)

def ik(arm, p_des, Rd, iters=200, w=0.6):
    # 前 30% 只解位置再叠姿态（staged，同 probe_reach）——新 tote 更窄（±0.172 vs 旧 ±0.30），
    # 一步到位的 6D IK 会卡局部极小（左臂差 0.33m）；staged + 新 seed 双臂 0.0mm 收敛。
    for k in range(iters):
        mujoco.mj_fwdPosition(m, d)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, arm.mount)
        ww = 0.0 if k < iters * 0.3 else w
        J = np.vstack([jp[:, arm.dof], ww*jr[:, arm.dof]])
        err = np.concatenate([p_des - d.xpos[arm.mount], ww*rot_err(Rd, d.xmat[arm.mount].reshape(3, 3))])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(6), err)
        dq = np.clip(dq, -0.05, 0.05)
        for i, a in enumerate(arm.qadr):
            d.qpos[a] = np.clip(d.qpos[a] + dq[i], arm.lo[i], arm.hi[i])
    mujoco.mj_fwdPosition(m, d)

def geom_aabb(g):
    typ = m.geom_type[g]
    pos = d.geom_xpos[g]
    mat = d.geom_xmat[g].reshape(3, 3)
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.abs(mat) @ m.geom_size[g, :3]
        return pos - half, pos + half
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        vw = v @ mat.T + pos
        return vw.min(axis=0), vw.max(axis=0)
    return pos.copy(), pos.copy()

def feature_mount_target(feature_name):
    g = gid(feature_name)
    if g < 0:
        raise RuntimeError(f"missing jig grasp feature geom: {feature_name}")
    mn, mx = geom_aabb(g)
    center = (mn + mx) / 2.0
    return np.array([center[0], center[1], center[2] + PAD_TO_MOUNT_Z])

# ---------- initialise: symmetric fingers-down pre-grasp, grippers OPEN ----------
#   忠实场景 measure-after-settle（同 scripts/cruzr_bimanual.py）：seed 前伸姿态 → IK 双臂到
#   对称指向下的 mount → settle 承载 → 实测指腹中点 → tote XY 贴到指腹中心。tote Z 固定，底部
#   落在固定 rack_support 立柱上（永不平移）。
# new.usd 实测（build_newusd_scene.py 打印的 magenta KLT AABB）：tote 中心
# model(0.5606, -0.0203, 0.9882)，外尺寸 0.278(x)×0.3445(y)×0.146(z)，顶沿 z=1.061，
# 底 z=0.915；rack_support 顶 0.910 → TOTE_CZ=0.9837 直接落座（旧 0.9882 有 5mm 掉落
# 弹跳，撞出瞬态穿透）。MOUNT_Z=1.166：IK 收敛后承载下垂 ~13mm → 实测 mount z≈1.153，
# PGC 基座底 = 1.153-0.0915 ≈ 1.0615，高于落座后顶沿 1.0564（+5mm）——旧 1.131 让基座底
# (1.033)低于顶沿 23mm，箱沿从实心基座里穿出来 = 用户看到的「夹爪穿模」（视觉性，
# pad-壁物理穿透实测 <0.3mm）。指腹 pad 实测竖跨 ≈[0.995,1.065]，仍咬住壁顶 ~61mm。
# 【底盘可动后】这些常量是【底盘系】(chassis frame)：底盘 spawn 在原点、yaw=0 时数值与
# 旧世界系标定完全一致（不回归）；IK 时经 base_world()/base_rotz() 变换到世界系，开走后照抓。
# v6 (2026-07-08) table-edge jig: no riser.  The jig rests on support_col0
# (top z=0.910) with its near-robot -x edge overhanging the table for single-gripper
# side-pinch access.  Keep these constants in sync with the MJCF body pose.
# v2 (2026-07-11) ecu_module manipuland at x0.80 of original (footprint 0.168x0.155x0.0240),
# paired with precision_machine_part enlarged x1.4. centre (0.459,-0.006,0.9223) on the pedestal;
# half-width y=0.0728.
MOUNT_X, MOUNT_Z, WALL_Y, TOTE_CZ = 0.459, 1.048, 0.0756, 0.9228
JIG_Y = -0.006
JIG_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
PAD_TO_MOUNT_Z = 0.123
SIDE_FEATURE = {"r": "jig_plate", "l": "jig_plate"}
ARM_SEED = np.array([-1.0, -0.5, 0.5, -1.4, 0.5, 0.0, 0.0])   # grasp-ready IK seed (kept for IK convergence)
# 自然下垂 home（2026-07-07）：teleop 启动时双臂沿躯干自然垂下（不再是抬起的预抓取位）。
# 关节序 = [肩pitch, 肩roll, 肩yaw, 肘roll, 肘yaw, 腕pitch, 腕roll]。全零已是上臂竖直下垂、
# 夹爪朝下的自然姿态；肘roll=-0.35(≈20°)给一点放松的微屈。用户开机看到手臂垂在身侧，
# 要抓取时用 q/w/e... 逐关节把臂抬起来（自然流程）。此姿态同时作为 ctrl/qtgt 初值，避免开机弹跳。
DROOP_POSE = np.array([0.0, 0.0, 0.0, -0.35, 0.0, 0.0, 0.0])   # natural-hang home
SIDEGRASP_POSE = {
    # Computed by scripts/jig_single_side_smoke.py for the current jig pose.
    # Pad midpoint = [0.330, 0.005, 0.930], pad gap = vertical, insertion axis
    # mount->padmid = world +X.  This is a safe pre-insertion home just outside the
    # robot-side table edge; the operator inserts straight along +X before closing.
    "l": np.array([0.1031, -0.1377, -0.1968, -1.7698, 1.4715, -0.2688, -1.3824]),
    "r": np.array([0.0923, -0.0298, 0.3844, -1.5082, -1.4207, -0.6198, -1.1137]),
}

# 逐关节运行时状态：每臂 7 关节目标 + 每手夹爪命令 + 激活手
qtgt = {"r": np.zeros(7), "l": np.zeros(7)}
grip_cmd = {"r": GRIP_OPEN, "l": GRIP_OPEN}
active = {"s": "r"}
control_mode = {"v": "joint"}
dual_control = {"v": False}

def init_pregrasp():
    """(保留，不再是启动默认) 阻尼最小二乘 IK 把双臂摆到对称预抓取位并把 jig 停到指腹中点——
    供需要「一键回抓取位」或抓取回归测试(base_drive_test)调用。启动默认 = 自然下垂 init_droop。"""
    d.qpos[tq:tq+3] = base_world(np.array([MOUNT_X, JIG_Y, TOTE_CZ]))
    d.qpos[tq+3:tq+7] = JIG_QUAT
    mujoco.mj_forward(m, d)
    targetR = feature_mount_target(SIDE_FEATURE["r"])
    targetL = feature_mount_target(SIDE_FEATURE["l"])
    d.qpos[tq:tq+3] = [3, 3, 3]                        # park tote off-scene during IK
    for a, v in zip(R.qadr, np.clip(ARM_SEED, R.lo, R.hi)): d.qpos[a] = v
    for a, v in zip(L.qadr, np.clip(ARM_SEED, L.lo, L.hi)): d.qpos[a] = v
    mujoco.mj_forward(m, d)
    Rz = base_rotz()
    ik(R, targetR, Rz @ R_DES, iters=1200)
    ik(L, targetL, Rz @ R_DES, iters=1200)
    for i, a in enumerate(R.arm_acts): d.ctrl[a] = d.qpos[R.qadr[i]]
    for i, a in enumerate(L.arm_acts): d.ctrl[a] = d.qpos[L.qadr[i]]
    set_gripper_state(R, GRIP_OPEN); set_gripper_state(L, GRIP_OPEN)
    for i, a in enumerate(BASE_ACTS): d.ctrl[a] = base_tgt[i]
    for _ in range(500): mujoco.mj_step(m, d)
    d.qpos[tq:tq+3] = base_world(np.array([MOUNT_X, JIG_Y, TOTE_CZ]))
    d.qpos[tq+3:tq+7] = JIG_QUAT; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for s, arm in ARMS.items():
        qtgt[s] = np.array([d.qpos[a] for a in arm.qadr], float)
        grip_cmd[s] = GRIP_OPEN

def init_droop():
    """启动 home = 双臂自然下垂。臂关节、ctrl 目标、逐关节 qtgt 全部初始化到 DROOP_POSE，
    夹爪 OPEN；jig 保持场景默认位（落在取件台上，不再 re-park 到指腹）。这样开机手臂垂在
    身侧不弹跳，用户要抓取时逐关节把臂抬起来。"""
    for arm in (R, L):
        for i, a in enumerate(arm.qadr):
            d.qpos[a] = np.clip(DROOP_POSE[i], arm.lo[i], arm.hi[i])
        for i, a in enumerate(arm.arm_acts):
            d.ctrl[a] = np.clip(DROOP_POSE[i], arm.lo[i], arm.hi[i])
    set_gripper_state(R, GRIP_OPEN); set_gripper_state(L, GRIP_OPEN)
    for i, a in enumerate(BASE_ACTS): d.ctrl[a] = base_tgt[i]   # base 位置保持
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(200): mujoco.mj_step(m, d)          # 让 jig 在取件台上落稳、臂在 ctrl 下稳住
    for s, arm in ARMS.items():
        qtgt[s] = np.array([d.qpos[a] for a in arm.qadr], float)
        grip_cmd[s] = GRIP_OPEN

def init_sidegrasp():
    """Start directly at the computed single-gripper side-grasp posture.

    Use TELEOP_SIDEGRASP_ARM=l/r to choose the arm. The other arm stays in the
    natural droop pose so it does not collide with the table while the operator
    closes the selected gripper.
    """
    chosen = os.environ.get("TELEOP_SIDEGRASP_ARM", "r").lower()
    if chosen not in SIDEGRASP_POSE:
        chosen = "l"
    d.qpos[tq:tq+3] = base_world(np.array([MOUNT_X, JIG_Y, TOTE_CZ]))
    d.qpos[tq+3:tq+7] = JIG_QUAT
    for s, arm in ARMS.items():
        pose = SIDEGRASP_POSE[s] if s == chosen else DROOP_POSE
        for i, a in enumerate(arm.qadr):
            d.qpos[a] = np.clip(pose[i], arm.lo[i], arm.hi[i])
        for i, a in enumerate(arm.arm_acts):
            d.ctrl[a] = np.clip(pose[i], arm.lo[i], arm.hi[i])
    set_gripper_state(R, GRIP_OPEN); set_gripper_state(L, GRIP_OPEN)
    for i, a in enumerate(BASE_ACTS): d.ctrl[a] = base_tgt[i]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(200): mujoco.mj_step(m, d)
    for s, arm in ARMS.items():
        qtgt[s] = np.array([d.qpos[a] for a in arm.qadr], float)
        grip_cmd[s] = GRIP_OPEN
    active["s"] = chosen

# 启动 home：默认=自然下垂；TELEOP_HOME=pregrasp 为旧双臂特征预抓取；
# TELEOP_HOME=sidegrasp 直接摆到当前 jig 的单夹爪侧夹姿态。
home = os.environ.get("TELEOP_HOME", "droop").lower()
if home == "droop":
    init_droop()
elif home == "sidegrasp":
    init_sidegrasp()
else:
    init_pregrasp()

# ---------------------------- 终端 tty 读键（非阻塞，daemon 线程）----------------------------
class TTYReader:
    """Raw-stdin cbreak reader（= Isaac sim/keyboard_teleop.KeyReader 的精简自包含版，
    不跨包引用以保持 mujoco_teleop 隔离）。非 tty（管道喂键）时退化为逐字节读。"""
    def __init__(self):
        self._buf, self._lock, self._stop = [], threading.Lock(), False
        try: self._is_tty = sys.stdin.isatty()
        except Exception: self._is_tty = False
        self._fd = sys.stdin.fileno() if self._is_tty else None
        self._old = None
    def start(self):
        if self._is_tty:
            self._old = termios.tcgetattr(self._fd); tty.setcbreak(self._fd)
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while not self._stop:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r: continue
                ch = sys.stdin.read(1)
                if ch == "\x1b":                    # 丢弃方向键等 ESC 序列
                    select.select([sys.stdin], [], [], 0.002);
                    try: sys.stdin.read(2)
                    except Exception: pass
                    continue
                if ch:
                    with self._lock: self._buf.append(ch)
            except Exception:
                break
    def drain(self):
        with self._lock:
            out = self._buf[:]; self._buf.clear()
        return out
    def restore(self):
        self._stop = True
        if self._is_tty and self._old is not None:
            try: termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception: pass

# 臂逐关节键（激活手）：上排 q w e r t y u = +，下排 a s d f g h j = −
_ARMKEY = {"q": (0, +1), "a": (0, -1), "w": (1, +1), "s": (1, -1),
           "e": (2, +1), "d": (2, -1), "r": (3, +1), "f": (3, -1),
           "t": (4, +1), "g": (4, -1), "y": (5, +1), "h": (5, -1),
           "u": (6, +1), "j": (6, -1)}
_EEKEY = {"q": [1.0, 0.0, 0.0], "a": [-1.0, 0.0, 0.0],
          "w": [0.0, 1.0, 0.0], "s": [0.0, -1.0, 0.0],
          "e": [0.0, 0.0, 1.0], "d": [0.0, 0.0, -1.0]}
_EE_WRISTKEY = {"t": (4, +1), "g": (4, -1), "y": (5, +1), "h": (5, -1),
                "u": (6, +1), "j": (6, -1)}
_BODYKEYS = set("ikol p;n".replace(" ", ""))   # 身体键：Isaac 有、本平台焊死→忽略；m 留给双臂模式
done = {"v": False}

def _selected_sides():
    return ("r", "l") if dual_control["v"] else (active["s"],)

def _selected_label():
    return "both" if dual_control["v"] else active["s"]

def ee_translate_target(s, delta_world):
    """Nudge active end-effector position through one damped Jacobian step.

    Full iterative IK reaches the requested Cartesian target, but near the natural
    droop pose it can use large elbow/wrist null-space motion for a 1 cm key press.
    Teleop should feel like velocity control, so each key applies one bounded
    joint increment and the normal joint position servo executes it smoothly.
    """
    arm = ARMS[s]
    sq, sv = d.qpos.copy(), d.qvel.copy()
    for i, a in enumerate(arm.qadr):
        d.qpos[a] = qtgt[s][i]
    mujoco.mj_forward(m, d)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, arm.mount)
    j = jp[:, arm.dof]
    dq = j.T @ np.linalg.solve(j @ j.T + DEE_DAMPING * DEE_DAMPING * np.eye(3), delta_world)
    dq = np.clip(dq, -DEE_JOINT_MAX, DEE_JOINT_MAX)
    solved = qtgt[s] + dq
    d.qpos[:], d.qvel[:] = sq, sv
    mujoco.mj_forward(m, d)
    qtgt[s] = np.clip(solved, arm.lo, arm.hi)
    print(
        f"[teleop] {s} ee dpos={np.round(delta_world,3).tolist()} "
        f"dq={np.round(dq,3).tolist()} -> qtgt={np.round(qtgt[s],3).tolist()}",
        flush=True,
    )

def handle_key(ch):
    s = active["s"]; arm = ARMS[s]
    _disp = {" ": "SPACE", "\t": "TAB", "\r": "ENTER", "\n": "ENTER",
             "\x7f": "BACKSPACE", "\b": "BACKSPACE"}.get(ch, repr(ch))
    print(f"[KEY] {_disp}", flush=True)               # 每键回显（据此判断键真送进来了）
    if ch == "v":
        control_mode["v"] = "ee" if control_mode["v"] == "joint" else "joint"
        print(f"[teleop] control mode -> {control_mode['v']}", flush=True)
    elif ch == "m":
        dual_control["v"] = not dual_control["v"]
        print(f"[teleop] dual-arm control -> {'ON' if dual_control['v'] else 'OFF'} "
              f"(active={active['s']})", flush=True)
    elif control_mode["v"] == "ee" and ch in _EEKEY:
        for ss in _selected_sides():
            ee_translate_target(ss, DEE_POS * np.array(_EEKEY[ch], float))
    elif control_mode["v"] == "ee" and ch in _EE_WRISTKEY:
        ji, sgn = _EE_WRISTKEY[ch]
        for ss in _selected_sides():
            aa = ARMS[ss]
            qtgt[ss][ji] = float(np.clip(qtgt[ss][ji] + sgn * DARM, aa.lo[ji], aa.hi[ji]))
            print(f"[teleop] {ss} joint{ji} -> {qtgt[ss][ji]:+.3f} rad (ee wrist)", flush=True)
    elif ch in _ARMKEY:
        ji, sgn = _ARMKEY[ch]
        for ss in _selected_sides():
            aa = ARMS[ss]
            qtgt[ss][ji] = float(np.clip(qtgt[ss][ji] + sgn * DARM, aa.lo[ji], aa.hi[ji]))
            print(f"[teleop] {ss} joint{ji} -> {qtgt[ss][ji]:+.3f} rad", flush=True)
    elif ch == " ":
        close = any(grip_cmd[ss] <= GRIP_MID for ss in _selected_sides())
        value = GRIP_CLOSE if close else GRIP_OPEN
        for ss in _selected_sides():
            grip_cmd[ss] = value
        print(f"[teleop] {_selected_label()} gripper -> {'CLOSE' if value > GRIP_MID else 'OPEN'}", flush=True)
    elif ch == "[":
        for ss in _selected_sides():
            grip_cmd[ss] = GRIP_OPEN
        print(f"[teleop] {_selected_label()} gripper -> OPEN", flush=True)
    elif ch == "]":
        for ss in _selected_sides():
            grip_cmd[ss] = GRIP_CLOSE
        print(f"[teleop] {_selected_label()} gripper -> CLOSE", flush=True)
    elif ch == "\t":
        active["s"] = "l" if s == "r" else "r"
        print(f"[teleop] active hand -> {active['s']} "
              f"({'dual ON' if dual_control['v'] else 'single'})", flush=True)
    elif ch == "=":
        for ss, aa in ARMS.items():
            qtgt[ss] = np.array([d.qpos[a] for a in aa.qadr], float)
        print("[teleop] re-seeded arm joint targets to measured", flush=True)
    elif ch == "8":
        base_vel[0] = float(np.clip(base_vel[0] + DVEL, -VMAX, VMAX))
        print(f"[BASE] v_fwd={base_vel[0]:+.2f} m/s  wz={base_vel[1]:+.2f} rad/s", flush=True)
    elif ch == "2":
        base_vel[0] = float(np.clip(base_vel[0] - DVEL, -VMAX, VMAX))
        print(f"[BASE] v_fwd={base_vel[0]:+.2f} m/s  wz={base_vel[1]:+.2f} rad/s", flush=True)
    elif ch == "7":
        base_vel[1] = float(np.clip(base_vel[1] + DWZ, -WZMAX, WZMAX))
        print(f"[BASE] v_fwd={base_vel[0]:+.2f} m/s  wz={base_vel[1]:+.2f} rad/s", flush=True)
    elif ch == "9":
        base_vel[1] = float(np.clip(base_vel[1] - DWZ, -WZMAX, WZMAX))
        print(f"[BASE] v_fwd={base_vel[0]:+.2f} m/s  wz={base_vel[1]:+.2f} rad/s", flush=True)
    elif ch == "0":
        base_vel[:] = 0.0
        base_tgt[:] = base_pose()      # 目标钉在当前实测位姿：停死、抓取反作用零漂移
        print("[BASE] STOP (目标冻结,位置保持)", flush=True)
    elif ch in ("4", "6"):
        print("[BASE] 差速底盘无侧移(禁螃蟹步)——4/6 忽略；请用 7/9 转向 + 8/2 前进组合", flush=True)
    elif ch == "\\":
        bx, by, byaw = base_pose()
        print(f"[teleop] active={s} dual={dual_control['v']} mode={control_mode['v']} "
              f"qtgt={np.round(qtgt[s],3).tolist()} "
              f"grip={ {k: round(v,3) for k,v in grip_cmd.items()} } "
              f"tote_z={d.qpos[tq+2]:.3f} "
              f"base=({bx:+.3f},{by:+.3f},{np.degrees(byaw):+.1f}deg) "
              f"base_vel=(v={base_vel[0]:+.2f},wz={base_vel[1]:+.2f})", flush=True)
    elif ch == "c":
        _print_relevant_contacts()
    elif ch == "b":
        print("[teleop] 身体锁: MuJoCo 升降柱/腰/头已焊死为刚体，本就固定，无需(键忽略)", flush=True)
    elif ch == "z":                                    # START / RESUME recording
        _rec_start()
    elif ch == "x":                                    # STOP / PAUSE recording
        if REC["rec"] is not None:
            _rec_stop()
        else:
            print("[teleop] (recording not started; press z to start)", flush=True)
    elif ch in _BODYKEYS:
        print("[teleop] 身体键: MuJoCo 已焊死/固定，此平台无对应关节(忽略)", flush=True)
    elif ch in ("\r", "\n"):                           # ENTER = SAVE episode + exit
        if REC["rec"] is not None:
            REC["rec"].finalize(success=True)
        else:
            print("[teleop] (no recording active)", flush=True)
        done["v"] = True; print("[teleop] ENTER -> save & 结束", flush=True)
    elif ch in ("\x7f", "\b"):                          # BACKSPACE = DISCARD episode + exit
        if REC["rec"] is not None:
            REC["rec"].discard()
        done["v"] = True; print("[teleop] BACKSPACE -> discard & 结束", flush=True)

# ------------------------- one control-loop iteration（纯逐关节位置伺服，不解 IK）-------------
def control_step(substeps=8):
    dg = GRIP_RATE * m.opt.timestep * substeps       # 每控制拍夹爪 ctrl 最大变化
    # 底盘差速目标积分（平移只沿航向 → 禁螃蟹步），力经 position actuator 物理跟踪
    dtc = m.opt.timestep * substeps
    v, wz = float(base_vel[0]), float(base_vel[1])
    if v != 0.0 or wz != 0.0:
        c, s_ = np.cos(base_tgt[2]), np.sin(base_tgt[2])
        base_tgt[0] += c * v * dtc
        base_tgt[1] += s_ * v * dtc
        base_tgt[2] += wz * dtc
    for i, a in enumerate(BASE_ACTS):
        d.ctrl[a] = base_tgt[i]
    for s, arm in ARMS.items():
        for i, a in enumerate(arm.arm_acts):
            d.ctrl[a] = qtgt[s][i]
        tgt = float(np.clip(grip_cmd[s], GRIP_OPEN, GRIP_CLOSE))
        cur = float(d.ctrl[arm.grip])
        d.ctrl[arm.grip] = cur + np.clip(tgt - cur, -dg, dg)   # 限速逼近（防拍壁瞬态穿透）
    for _ in range(substeps):
        mujoco.mj_step(m, d)
    # ---- episode recording hook (OFF until the operator presses START; base_drive_test /
    # jig_smoke never start it, so this is a no-op for them). Sample every REC_DECIM-th control
    # step so frames land at the schema's ~30fps regardless of the render/target FPS. ----
    if REC["rec"] is not None and REC["on"]:
        REC["count"] += 1
        if REC["count"] % REC_DECIM == 0:
                REC["rec"].capture(REC.get("phase", "teleop"))

# ============================ EPISODE RECORDING (LeRobot CRUZR schema) ============================
# Teleop data capture matching the EXACT schema the Isaac CRUZR recorder writes
# (scripts/run_teleop_cruzr.py) so recordings drop straight into the existing LeRobot converter
# scripts/build_carton_lerobot.py — NO new converter. Per saved episode <out>/:
#   frames/{top_head,hand_left,hand_right}/frame_%06d.jpg   (JPEG q90, 480x640x3)
#   episode_data.npz    timestamp / state(n,16) / action(n,16) / action_real / base / phase
#   meta.json           task/seed/prompt/fps/success/num_frames/resolution_hw/cameras/
#                       state_joint_names/robot
# state16  = [L arm 7, R arm 7, L/R gripper_open_frac]  (== Isaac L-then-R order)
# action16 = commanded [L qtgt 7, R qtgt 7, L/R gripper_open_frac_cmd]  (action_real = True).
# mobile-base arrays are saved separately for π0.5 mobile manipulation conversion:
#   base(n,3) = measured x/y/yaw, base_velocity(n,2) = measured v_fwd/wz,
#   base_action(n,2) = commanded v_fwd/wz.
# The 3 record cameras ALWAYS render offscreen via EGL on GPU device REC_GPU. By default recording
# follows the interactive EGL device, so it does not silently fall back to a busy hard-coded GPU.
REC_FPS = 30
REC_WH = (640, 480)                       # (width, height) -> saved frames are 480x640x3
REC_JPEG_Q = 90
# v5 observation contract (2026-07-16, direction B/plan-A): record FIVE cameras -
# stereo_left/right + waist_front are the real-robot training trio (SDK extrinsics),
# hand_left/right kept as archive (future real-robot wrist-cam retrofit) so switching
# combos never needs re-recording. Override via REC_CAMS env (comma-separated).
REC_CAMS = os.environ.get(
    "REC_CAMS", "stereo_left,stereo_right,waist_front,hand_left,hand_right").split(",")
REC_SAVE_RAW_TIMESTAMPS = os.environ.get("REC_SAVE_RAW_TIMESTAMPS", "0") == "1"
REC_GPU = os.environ.get("TELEOP_RECORD_GPU", os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
REPO_ROOT = SAFE_VLA_ROOT
RECORD_ROOT = os.path.join(REPO_ROOT, "outputs", "teleop")
RECORD_FALLBACK_ROOT = os.path.join(_PKG_ROOT, "out", "teleop")
REC_DECIM = max(1, int(round(TARGET_FPS / REC_FPS)))   # control steps per recorded frame -> ~30fps
_L_ARM_NAMES = ["L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
                "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint"]
_R_ARM_NAMES = ["R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
                "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint"]
STATE_JOINT_NAMES = _L_ARM_NAMES + _R_ARM_NAMES + ["l_gripper_open_frac", "r_gripper_open_frac"]
ACTION_NAMES = _L_ARM_NAMES + _R_ARM_NAMES + ["l_gripper_open_frac_cmd", "r_gripper_open_frac_cmd"]
BASE_STATE_NAMES = ["base_x", "base_y", "base_yaw", "base_v_fwd", "base_wz"]
BASE_ACTION_NAMES = ["base_cmd_v_fwd", "base_cmd_wz"]


def _rec_seed():
    try:
        return int(os.environ.get("CRUZR_EP_SEED", "0"))
    except Exception:
        return 0


def _record_root_writable(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK | os.X_OK)


def _record_dir_for_name(name):
    for root in (RECORD_ROOT, RECORD_FALLBACK_ROOT):
        if _record_root_writable(root):
            if root != RECORD_ROOT:
                print(f"[REC] default record root not writable; using fallback {root}", flush=True)
            return os.path.join(root, name)
    return os.path.join(RECORD_ROOT, name)


def resolve_record_dir():
    """Record dir from env (the gui launcher passes a name). TELEOP_RECORD / TELEOP_RECORD_NAME /
    CRUZR_RECORD_DIR: a bare name -> writable record root/<name>; a path (abs or with '/') -> used as-is."""
    v = (os.environ.get("TELEOP_RECORD") or os.environ.get("TELEOP_RECORD_NAME")
         or os.environ.get("CRUZR_RECORD_DIR"))
    if not v:
        return None
    if os.path.isabs(v) or "/" in v:
        return v
    return _record_dir_for_name(v)


class EpisodeRecorder:
    """Offscreen EGL (GPU REC_GPU) recorder writing the Isaac CRUZR / LeRobot schema (module header)."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.n = 0
        self.rows = {"state": [], "action": [], "phase": [], "base": [],
                     "base_velocity": [], "base_action": [], "capture_timestamp": []}
        self._gl = self._con = self._scn = self._opt = self._vp = None
        self._camids = None
        self._Lf = [m.jnt_qposadr[jid("L_finger1_joint")], m.jnt_qposadr[jid("L_finger2_joint")]]
        self._Rf = [m.jnt_qposadr[jid("R_finger1_joint")], m.jnt_qposadr[jid("R_finger2_joint")]]
        for c in REC_CAMS:
            os.makedirs(os.path.join(out_dir, "frames", c), exist_ok=True)
        print(f"[REC] armed -> {out_dir} (16-dim, {REC_CAMS}, {REC_WH[1]}x{REC_WH[0]}, "
              f"{REC_FPS}fps, decim={REC_DECIM})", flush=True)

    def _ensure_gl(self):
        if self._gl is not None:
            return
        os.environ["MUJOCO_EGL_DEVICE_ID"] = REC_GPU     # force record render onto GPU REC_GPU
        from mujoco.egl import GLContext
        W, H = REC_WH
        self._gl = GLContext(W, H); self._gl.make_current()
        self._con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
        self._scn = mujoco.MjvScene(m, maxgeom=20000)
        self._opt = mujoco.MjvOption()
        self._vp = mujoco.MjrRect(0, 0, W, H)
        self._camids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in REC_CAMS}
        missing = [c for c, i in self._camids.items() if i < 0]
        assert not missing, f"record cameras missing in model: {missing}"
        print(f"[REC] EGL offscreen render on GPU device {REC_GPU} ({W}x{H}) ready {REC_CAMS}",
              flush=True)

    def _state16(self):
        arm_l = [float(d.qpos[a]) for a in L.qadr]
        arm_r = [float(d.qpos[a]) for a in R.qadr]
        gl = 0.5 * (float(d.qpos[self._Lf[0]]) + float(d.qpos[self._Lf[1]]))
        gr = 0.5 * (float(d.qpos[self._Rf[0]]) + float(d.qpos[self._Rf[1]]))
        return np.array(arm_l + arm_r + [
            grip_open_fraction(gl, GRIP_OPEN, GRIP_CLOSE),
            grip_open_fraction(gr, GRIP_OPEN, GRIP_CLOSE),
        ], np.float32)

    def _action16(self):
        return np.array(list(qtgt["l"]) + list(qtgt["r"])
                        + [grip_open_fraction(grip_cmd["l"], GRIP_OPEN, GRIP_CLOSE),
                           grip_open_fraction(grip_cmd["r"], GRIP_OPEN, GRIP_CLOSE)], np.float32)

    def capture(self, phase="teleop"):
        self._ensure_gl()
        self._gl.make_current()          # bind our EGL ctx (viewer render re-binds its own after)
        from PIL import Image
        W, H = REC_WH
        for c in REC_CAMS:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = self._camids[c]
            mujoco.mjv_updateScene(m, d, self._opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL.value, self._scn)
            mujoco.mjr_render(self._vp, self._scn, self._con)
            rgb = np.zeros((H, W, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, self._vp, self._con)
            Image.fromarray(np.flipud(rgb)).save(
                os.path.join(self.out_dir, "frames", c, f"frame_{self.n:06d}.jpg"), quality=REC_JPEG_Q)
        self.rows["state"].append(self._state16())
        self.rows["action"].append(self._action16())
        self.rows["phase"].append(phase)
        bx, by, byaw = base_pose()
        bv, bwz = base_velocity()
        self.rows["base"].append([float(bx), float(by), float(byaw)])
        self.rows["base_velocity"].append([float(bv), float(bwz)])
        self.rows["base_action"].append([float(base_vel[0]), float(base_vel[1])])
        # Rendering is synchronous and does not advance MuJoCo, so state and all
        # camera frames captured above share this exact simulation timestamp.
        self.rows["capture_timestamp"].append(float(d.time))
        self.n += 1

    def finalize(self, success):
        n = self.n
        if n == 0:
            with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
                json.dump({"task": "transport_carton_cruzr", "seed": _rec_seed(),
                           "aborted": True, "success": False, "num_frames": 0},
                          f, ensure_ascii=False, indent=2)
            print(f"[REC] finalize(empty): 0 frames -> sentinel meta {self.out_dir}", flush=True)
            return
        state = np.stack(self.rows["state"]); action = np.stack(self.rows["action"])
        ts = ((np.arange(n) + 1) / REC_FPS).astype(np.float32)
        np.savez(os.path.join(self.out_dir, "episode_data.npz"),
                 timestamp=ts, state=state, action=action,
                 action_real=np.ones(n, dtype=bool),
                 base=np.array(self.rows["base"], np.float32),
                 base_velocity=np.array(self.rows["base_velocity"], np.float32),
                 base_action=np.array(self.rows["base_action"], np.float32),
                 phase=np.array(self.rows["phase"]))
        if REC_SAVE_RAW_TIMESTAMPS:
            raw_timestamp = np.asarray(self.rows["capture_timestamp"], dtype=np.float64)
            if raw_timestamp.shape != (n,) or not np.isfinite(raw_timestamp).all():
                raise RuntimeError("raw capture timestamps are missing or non-finite")
            timestamp_payload = {"state_timestamp": raw_timestamp}
            timestamp_payload.update({
                f"camera_{camera}_timestamp": raw_timestamp.copy()
                for camera in REC_CAMS
            })
            np.savez_compressed(
                os.path.join(self.out_dir, "sdk_timestamps.npz"),
                **timestamp_payload,
            )
        meta = {"task": "transport_carton_cruzr", "seed": _rec_seed(),
                "prompt": os.environ.get("REC_PROMPT", "pick up the parts and place it down"),
                "fps": REC_FPS, "success": bool(success), "num_frames": n,
                "resolution_hw": [REC_WH[1], REC_WH[0]],
                "cameras": {c: f"{c} (mujoco fixed camera, EGL/GPU{REC_GPU})" for c in REC_CAMS},
                "state_joint_names": STATE_JOINT_NAMES,
                "action_names": ACTION_NAMES,
                "base_state_names": BASE_STATE_NAMES,
                "base_action_names": BASE_ACTION_NAMES,
                "gripper_state": "state/action gripper fields are open_frac: 1=open, 0=closed",
                "gripper_raw_convention": {
                    "open": GRIP_OPEN,
                    "close": GRIP_CLOSE,
                    "units": "m",
                    "note": "PGC raw finger q is negative when open and 0 when closed",
                },
                "robot": "CRUZR_S2_pgc_mujoco",
                "episode_metadata": REC.get("metadata", {})}
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[REC] finalize: {n} frames + episode_data.npz + meta.json -> {self.out_dir}",
              flush=True)

    def close(self):
        """Release the offscreen render context in dependency order."""
        if self._gl is None:
            return
        self._gl.make_current()
        if self._con is not None:
            self._con.free()
            self._con = None
        self._gl.free()
        self._gl = None

    def discard(self):
        import shutil
        try:
            shutil.rmtree(self.out_dir)
        except Exception as e:
            print(f"[REC] discard rmtree failed: {e}", flush=True)
        print(f"[REC] DISCARDED (no data saved): {self.out_dir}", flush=True)


# recording state (OFF until the operator presses START).
REC = {"rec": None, "on": False, "count": 0, "phase": "teleop", "metadata": {}}


def _rec_start():
    if REC["rec"] is None:
        out = resolve_record_dir()
        if out is None:
            out = _record_dir_for_name("_mujoco_rec")
            print(f"[REC] no TELEOP_RECORD name/dir set -> default {out}", flush=True)
        try:
            REC["rec"] = EpisodeRecorder(out)
        except OSError as e:
            REC["rec"] = None
            REC["on"] = False
            print(f"[REC] ERROR: cannot start recording at {out}: {e}", flush=True)
            print("[REC] viewer kept running; set TELEOP_RECORD to a writable path or fix directory permissions",
                  flush=True)
            return
    REC["on"] = True
    print(f"[REC] ● RECORDING ON ({REC['rec'].out_dir}) — ENTER=save  BACKSPACE=discard  x=pause",
          flush=True)


def _rec_stop():
    REC["on"] = False
    print(f"[REC] ‖ paused (n={REC['rec'].n if REC['rec'] else 0} frames kept; z=resume)", flush=True)

# -------------------- custom GLFW viewer (pan-sensitivity scalable) --------------------
# launch_passive has no hook to lower pan sensitivity, so we run the standard mujoco glfw
# render loop ourselves: left-drag = orbit (ROT_SENS), right-drag / shift+left = pan
# (PAN_SENS, the whole point of this rewrite), scroll = zoom (ZOOM_SENS). The stdin tty key
# teleop (TTYReader + handle_key + control_step) is UNCHANGED and still drives the sim — the
# render loop just polls reader.drain() each frame exactly like the old launch_passive loop.
def _init_camera(cam):
    """same view init as the old launch_passive viewer."""
    cam.lookat[:] = [0.22, 0, 0.55]; cam.distance = 1.9
    cam.azimuth = 35; cam.elevation = -20

def _print_gl_renderer():
    """Print GL_RENDERER / GL_VENDOR of the live onscreen context. This is the GPU
    diagnostic: 'NVIDIA ... RTX 4090' = hardware (smooth); 'llvmpipe' / 'softpipe' /
    'Mesa' software = the classic GLX software fallback = the choppy teleop the user sees.
    Fix by forcing NVIDIA GLX for the window (see the launch-command banner below)."""
    try:
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
        rnd = glGetString(GL_RENDERER); ven = glGetString(GL_VENDOR); ver = glGetString(GL_VERSION)
        rnd = rnd.decode() if isinstance(rnd, bytes) else str(rnd)
        ven = ven.decode() if isinstance(ven, bytes) else str(ven)
        ver = ver.decode() if isinstance(ver, bytes) else str(ver)
        soft = any(k in rnd.lower() for k in ("llvmpipe", "softpipe", "swrast", "software"))
        print("=" * 78, flush=True)
        print(f"[GL] GL_RENDERER: {rnd}", flush=True)
        print(f"[GL] GL_VENDOR  : {ven}", flush=True)
        print(f"[GL] GL_VERSION : {ver}", flush=True)
        if soft:
            print("[GL] >>> SOFTWARE renderer (Mesa/llvmpipe): this is why teleop is choppy.", flush=True)
            print("[GL] >>> Re-launch forcing the NVIDIA GPU (see command below).", flush=True)
        else:
            print("[GL] >>> Hardware GPU context — rendering is GPU-accelerated.", flush=True)
        print("=" * 78, flush=True)
    except Exception as e:
        print(f"[GL] could not query GL_RENDERER ({e})", flush=True)

def run_viewer_glfw():
    """Open the custom GLFW window and run the render+teleop loop on the main thread.
    Honours PAN_SENS/ROT_SENS/ZOOM_SENS. Hardened so Python-level failures raise (main
    catches and falls back to launch_passive); a hard native segfault can't be caught here,
    which is why passive is the default viewer."""
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw.init() failed (no display?). Set MUJOCO_GL/DISPLAY or run a headless test.")
    # --- context hints: MuJoCo's mjr renderer needs LEGACY (fixed-function) OpenGL. Requesting a
    # core profile or forward-compatible context makes mjr_render crash, and pinning a specific
    # CONTEXT_VERSION_MAJOR/MINOR is fragile across drivers. So reset to glfw defaults and only
    # assert "no core profile / no forward-compat" — i.e. an ANY(compat/legacy) context, matching
    # what mujoco.viewer.launch_passive creates. Do NOT set CONTEXT_VERSION hints.
    glfw.default_window_hints()
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.FALSE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(1200, 900, "CRUZR teleop (custom GLFW viewer)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window failed (no display?).")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera(); opt = _new_scene_option()
    _init_camera(cam)
    scene = mujoco.MjvScene(m, maxgeom=20000)
    context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    state = {"lx": 0.0, "ly": 0.0, "bl": False, "bm": False, "br": False}

    def on_mouse_button(win, button, act, mods):
        state["bl"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        state["bm"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        state["br"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        x, y = glfw.get_cursor_pos(win); state["lx"], state["ly"] = x, y

    def on_mouse_move(win, xpos, ypos):
        dx = xpos - state["lx"]; dy = ypos - state["ly"]
        state["lx"], state["ly"] = xpos, ypos
        if not (state["bl"] or state["bm"] or state["br"]):
            return
        w, h = glfw.get_window_size(win); h = max(h, 1)
        shift = (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if state["br"] or (state["bl"] and shift):          # ---- PAN / translate ----
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if abs(dx) > abs(dy) else mujoco.mjtMouse.mjMOUSE_MOVE_V
            sens = PAN_SENS
        elif state["bl"]:                                   # ---- rotate / orbit ----
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if abs(dx) > abs(dy) else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            sens = ROT_SENS
        else:                                               # middle drag = zoom
            action = mujoco.mjtMouse.mjMOUSE_ZOOM; sens = ZOOM_SENS
        mujoco.mjv_moveCamera(m, action, sens * dx / h, sens * dy / h, scene, cam)

    def on_scroll(win, xoff, yoff):
        mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * ZOOM_SENS * yoff, scene, cam)

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_scroll_callback(window, on_scroll)

    first_render = True
    pacer = FramePacer(TARGET_FPS)
    while not glfw.window_should_close(window) and not done["v"]:
        for ch in reader.drain():
            handle_key(ch)
        control_step(substeps=CONTROL_SUBSTEPS)
        # re-bind THIS window's GL context before rendering: the recorder's offscreen EGL context
        # (used inside control_step when recording) leaves its own context current, so restore the
        # onscreen glfw context here or mjr_render would draw into the offscreen buffer.
        glfw.make_context_current(window)
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        vp = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjr_render(vp, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        if first_render:                       # <<< GPU diagnostic, only after mjr_render survives
            first_render = False               #     (safer than querying right after context create)
            _print_gl_renderer()
        pacer.sleep_until_next_frame()
    glfw.terminate()


def run_viewer_passive():
    """Original mujoco.viewer.launch_passive path — the SAFE DEFAULT, verified working on this
    user's display. Same view init (lookat[0.22,0,0.55]/dist1.9/az35/el-20) and the same
    drain-keys -> control_step(substeps=8) -> viewer.sync() loop as before the glfw rewrite.
    No mouse-sensitivity scaling (launch_passive exposes no such hook)."""
    with mujoco.viewer.launch_passive(m, d) as viewer:
        _init_camera(viewer.cam)
        if DEBUG_COLLISION and hasattr(viewer, "opt"):
            viewer.opt = _new_scene_option()
        pacer = FramePacer(TARGET_FPS)
        while viewer.is_running() and not done["v"]:
            for ch in reader.drain():
                handle_key(ch)
            control_step(substeps=CONTROL_SUBSTEPS)
            viewer.sync()
            pacer.sleep_until_next_frame()


# -------------------- EGL GPU-offscreen viewer (render on GPU, blit in a 2D window) --------------------
# WHY: this user's onscreen GLX is Mesa llvmpipe (SOFTWARE/CPU) -> the native GLFW/passive
# window renders the 3D scene on the CPU and is choppy. But EGL OFFSCREEN rendering hits the
# RTX 4090 (MUJOCO_GL=egl + MUJOCO_EGL_DEVICE_ID=<free gpu>, ~376 fps when idle).
# VirtualGL isn't installed and
# there's no sudo. So we render the 3D scene on the GPU via EGL offscreen (mujoco.Renderer),
# then just BLIT the finished RGB frame into a lightweight 2D window (cv2.imshow) — a 2D image
# blit is trivial even for llvmpipe. Net: real GPU frame-rate with zero admin changes.
#
# DISPLAY BACKEND = cv2 (opencv-python) — chosen because it was the cleanest GUI option actually
# installable in the mjx env (cv2.imshow needs only the software X display for a 2D window, no
# OpenGL context of its own -> avoids the native GLX/PRIME segfault that bit the custom GLFW
# viewer). pygame was absent; the glfw-textured-quad path would re-create an onscreen GL context
# (the very thing we're avoiding). cv2 install: opencv-python from the tsinghua mirror.
#
# INPUT SPLIT (identical intent to passive/glfw): keyboard TELEOP still comes from the TERMINAL
# stdin (TTYReader.drain()->handle_key) — NOT from the display window. The cv2 window only
# provides MOUSE camera control (setMouseCallback) + ESC-to-quit. So the user keeps focus on the
# terminal to drive the robot, and uses the mouse over the image window to orbit/pan/zoom.
def _egl_mouse_cb_factory(m, cam, scene_holder):
    """Build a cv2 setMouseCallback handler that drives the free MjvCamera:
    left-drag = orbit (ROT_SENS), right-drag OR shift+left-drag = pan (PAN_SENS),
    wheel = zoom (ZOOM_SENS). Uses mjv_moveCamera exactly like the glfw viewer, so feel matches."""
    import cv2
    st = {"lx": 0.0, "ly": 0.0}
    def cb(event, x, y, flags, param):
        scene = scene_holder[0]
        if scene is None:
            st["lx"], st["ly"] = x, y; return
        h = max(EGL_H, 1)
        if event == cv2.EVENT_MOUSEWHEEL:                       # scroll = zoom
            yoff = 1.0 if flags > 0 else -1.0
            mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, -0.05 * ZOOM_SENS * yoff, scene, cam)
            return
        dx = x - st["lx"]; dy = y - st["ly"]; st["lx"], st["ly"] = x, y
        lbtn = bool(flags & cv2.EVENT_FLAG_LBUTTON)
        rbtn = bool(flags & cv2.EVENT_FLAG_RBUTTON)
        mbtn = bool(flags & cv2.EVENT_FLAG_MBUTTON)
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if not (lbtn or rbtn or mbtn):
            return
        if rbtn or (lbtn and shift):                            # ---- PAN / translate ----
            action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if abs(dx) > abs(dy)
                      else mujoco.mjtMouse.mjMOUSE_MOVE_V); sens = PAN_SENS
        elif lbtn:                                              # ---- rotate / orbit ----
            action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if abs(dx) > abs(dy)
                      else mujoco.mjtMouse.mjMOUSE_ROTATE_V); sens = ROT_SENS
        else:                                                   # middle drag = zoom
            action = mujoco.mjtMouse.mjMOUSE_ZOOM; sens = ZOOM_SENS
        mujoco.mjv_moveCamera(m, action, sens * dx / h, sens * dy / h, scene, cam)
    return cb


def _apply_egl_render_flags(scene):
    if EGL_FAST:
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def _print_egl_diag(renderer):
    """GPU diagnostic for the EGL path: confirm the selected EGL device and, if
    the EGL context's GL_RENDERER is queryable, print it (should be NVIDIA ... RTX 4090)."""
    dev = os.environ.get("MUJOCO_EGL_DEVICE_ID", "?")
    print("=" * 78, flush=True)
    print(f"[GL] Rendering via EGL OFFSCREEN on GPU device {dev} (MUJOCO_GL=egl).", flush=True)
    try:
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
        def _s(x): return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
        rnd = _s(glGetString(GL_RENDERER)); ven = _s(glGetString(GL_VENDOR)); ver = _s(glGetString(GL_VERSION))
        print(f"[GL] GL_RENDERER: {rnd}", flush=True)
        print(f"[GL] GL_VENDOR  : {ven}", flush=True)
        print(f"[GL] GL_VERSION : {ver}", flush=True)
        soft = any(k in rnd.lower() for k in ("llvmpipe", "softpipe", "swrast", "software"))
        print("[GL] >>> " + ("SOFTWARE renderer?! EGL should be on the GPU — check MUJOCO_EGL_DEVICE_ID."
                             if soft else "Hardware GPU EGL context — 3D is GPU-accelerated."), flush=True)
    except Exception as e:
        print(f"[GL] (EGL GL_RENDERER not queryable from this context: {e}) — render fps below is the proof.", flush=True)
    if EGL_FAST:
        print("[GL] Fast interactive mode: shadows/reflections disabled.", flush=True)
    print("=" * 78, flush=True)


def run_viewer_egl():
    """GPU-offscreen viewer: render the 3D scene on the GPU via EGL (mujoco.Renderer) and blit
    the RGB frame into a cv2 window. Keyboard teleop stays on terminal stdin; the window gives
    mouse camera control + ESC quit. Prints a rolling-average render fps to the terminal ~every 2s."""
    import time, cv2
    renderer = mujoco.Renderer(m, EGL_H, EGL_W)   # EGL offscreen, GPU device selected at import
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); _init_camera(cam)
    opt = _new_scene_option()
    _print_egl_diag(renderer)

    scene_holder = [None]
    win = "CRUZR teleop (EGL/GPU render)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, _egl_mouse_cb_factory(m, cam, scene_holder))

    n = 0; t0 = time.time(); tlast = t0
    pacer = FramePacer(TARGET_FPS)
    while not done["v"]:
        for ch in reader.drain():
            handle_key(ch)
        control_step(substeps=CONTROL_SUBSTEPS)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        _apply_egl_render_flags(renderer.scene)
        scene_holder[0] = renderer.scene           # expose live scene to mouse cb for mjv_moveCamera
        rgb = renderer.render()                     # HxWx3 uint8 RGB, rendered on the GPU
        cv2.imshow(win, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if (cv2.waitKey(1) & 0xFF) == 27:           # ESC in the window quits
            done["v"] = True
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed with the X button
            done["v"] = True
        n += 1
        now = time.time()
        if now - tlast >= 2.0:                       # rolling render fps to terminal every ~2s
            print(f"[EGL] render fps ~{n / (now - t0):.1f} (avg over {n} frames)", flush=True)
            tlast = now
        pacer.sleep_until_next_frame()
    renderer.close()
    cv2.destroyAllWindows()


def egl_selftest(nframes=100, out_png=None):
    """Headless self-test of the GPU render PATH (no display needed): build the EGL Renderer,
    render `nframes` while stepping physics, measure fps, and save a sample frame. Proves the
    whole GPU-render half of run_viewer_egl works even where the 2D window can't be opened."""
    import time
    renderer = mujoco.Renderer(m, EGL_H, EGL_W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); _init_camera(cam)
    opt = _new_scene_option()
    _print_egl_diag(renderer)
    last = None; t0 = time.time()
    for _ in range(nframes):
        control_step(substeps=CONTROL_SUBSTEPS)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        _apply_egl_render_flags(renderer.scene)
        last = renderer.render()
    fps = nframes / (time.time() - t0)
    print(f"[EGL-selftest] rendered {nframes} frames @ {EGL_W}x{EGL_H}: {fps:.1f} render+physics fps", flush=True)
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        try:
            import cv2
            cv2.imwrite(out_png, cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
        except Exception:
            from PIL import Image; Image.fromarray(last).save(out_png)
        print(f"[EGL-selftest] sample frame -> {out_png}", flush=True)
    renderer.close()
    return fps


def run_viewer():
    """Dispatch to the selected viewer. TELEOP_VIEWER=glfw uses the custom pan-sensitivity
    viewer, wrapped so ANY Python-level failure prints a clear message and auto-falls-back to
    launch_passive (so teleop still runs). Default = passive."""
    if VIEWER == "egl":
        run_viewer_egl()
    elif VIEWER == "glfw":
        try:
            run_viewer_glfw()
        except BaseException as e:             # segfault can't reach here; catch everything else
            print(f"[teleop] custom GLFW viewer failed ({type(e).__name__}: {e}) "
                  f"-> falling back to launch_passive", flush=True)
            run_viewer_passive()
    else:
        run_viewer_passive()

def smoke_headless():
    """No-display smoke: verify model load + one control_step + camera init, no window.
    Run with TELEOP_SMOKE=1 (or auto-fallback when glfw/display is unavailable)."""
    print(f"[smoke] model loaded: nq={m.nq} nu={m.nu} nbody={m.nbody}", flush=True)
    cam = mujoco.MjvCamera(); _init_camera(cam)
    print(f"[smoke] camera init lookat={list(np.round(cam.lookat,3))} dist={cam.distance} "
          f"az={cam.azimuth} el={cam.elevation}", flush=True)
    control_step(substeps=CONTROL_SUBSTEPS)
    ok = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    print(f"[smoke] one control_step OK, states finite={ok}  PAN_SENS={PAN_SENS} "
          f"ROT_SENS={ROT_SENS} ZOOM_SENS={ZOOM_SENS} "
          f"TELEOP_FPS={TARGET_FPS} TELEOP_SUBSTEPS={CONTROL_SUBSTEPS}", flush=True)
    return ok

# ------------------------------- main -------------------------------
if __name__ == "__main__":
    print(__doc__)
    if os.environ.get("TELEOP_SMOKE") == "1":
        sys.exit(0 if smoke_headless() else 1)
    if os.environ.get("TELEOP_EGL_SELFTEST") == "1":   # headless GPU-render self-test (no window)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "egl_viewer", "sample.png")
        fps = egl_selftest(nframes=int(os.environ.get("EGL_SELFTEST_FRAMES", "100")), out_png=out)
        sys.exit(0 if fps > 0 else 1)
    if VIEWER == "egl":
        print("Launching EGL GPU-OFFSCREEN viewer (TELEOP_VIEWER=egl): 3D rendered on GPU device "
              f"{os.environ.get('MUJOCO_EGL_DEVICE_ID')} via EGL, blitted into a cv2 2D window. "
              f"Target FPS={TARGET_FPS}, control substeps/frame={CONTROL_SUBSTEPS}. "
              "焦点放【本终端】(不是图形窗口) 按键遥操；鼠标在图形窗口里控制相机 "
              "(左拖=旋转  右拖/Shift+左拖=平移  滚轮=缩放  ESC=退出)。", flush=True)
    elif VIEWER == "glfw":
        print("Launching CUSTOM GLFW viewer (TELEOP_VIEWER=glfw)... "
              "焦点放【本终端】(不是 MuJoCo 窗口)，按上面的键。", flush=True)
        print(f"鼠标：左键拖=旋转  右键拖(或Shift+左键)=平移(PAN_SENS={PAN_SENS})  滚轮=缩放。 "
              "若此 viewer 在你的显示器上 segfault，去掉 TELEOP_VIEWER=glfw 即回到默认 passive。", flush=True)
    else:
        print("Launching PASSIVE viewer (mujoco.viewer.launch_passive, 默认/安全). "
              "焦点放【本终端】(不是 MuJoCo 窗口)，按上面的键。 "
              "要降低鼠标平移灵敏度请加 TELEOP_VIEWER=glfw。", flush=True)
    reader = TTYReader(); reader.start()
    try:
        run_viewer()
    except Exception as e:
        print(f"[teleop] viewer unavailable ({e}) -> headless smoke instead", flush=True)
        smoke_headless()
    finally:
        reader.restore()
        print("\n[teleop] finished", flush=True)
