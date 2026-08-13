#!/usr/bin/env python3
"""Two-trip shelf expert: pillar -> cart middle, strip -> cart top.

The task succeeds only when both released objects remain supported by their
assigned cart shelves for 0.5 seconds.  The old single-object collector and
its data are intentionally left untouched.

Env: SEED (required), EXPERT_OUT, E2E_KICKS=4, E2E_NOREC=1.
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

# Correlate the two rack parts in y: they belong to the same visible rack bay,
# while small independent offsets prevent memorising one fixed alignment.
CART_NOM = np.array([-2.40, 0.0])
PILLAR_NOM = np.array([0.58, 0.0005])
STRIP_NOM = np.array([1.05, 0.0])
cart_xy = CART_NOM + np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.30, 0.30)])
rack_y = rng.uniform(-0.24, 0.24)
pillar_xy = PILLAR_NOM + np.array([rng.uniform(-0.04, 0.04), rack_y + rng.uniform(-0.035, 0.035)])
strip_xy = STRIP_NOM + np.array([rng.uniform(-0.03, 0.03), rack_y + rng.uniform(-0.035, 0.035)])
robot0 = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])

# Per-seed XML lives in assets because included robot mesh paths resolve from
# the main XML directory.  Every worker gets a unique file.
SCENE_DIR = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
os.makedirs(SCENE_DIR, exist_ok=True)
with open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")) as f:
    scene_text = f.read()
scene_text = re.sub(
    r'(<body name="shelf_cart" pos=")[^"]*(")',
    lambda x: f'{x.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{x.group(2)}',
    scene_text,
)
SCENE = os.path.join(SCENE_DIR, f"e2e_dual_scene_{SEED}.xml")
with open(SCENE, "w") as f:
    f.write(scene_text)

os.environ["TELEOP_SCENE_XML"] = SCENE
os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("REC_CAMS", "head_stereo_l_shelf,hand_left_shelf,hand_right_shelf,chassis_front")
os.environ.setdefault(
    "REC_PROMPT",
    "move the steel pillar to the middle shelf of the cart, then move the rubber strip to the top shelf",
)

spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)
import mujoco  # noqa: E402

m, d = ct.m, ct.d
ct.REC_WH = (224, 224)
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
DECIM = int(getattr(ct, "REC_DECIM", 2))
OUT = os.path.join(
    ROOT,
    os.environ.get("EXPERT_OUT", f"out/teleop/shelf_e2e_dual/shelf_e2e_dual_{SEED:06d}"),
)


def body_id(name):
    value = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"scene is missing body {name}")
    return value


OBJECTS = {}
for name in ("pillar", "strip"):
    bid = body_id(name)
    OBJECTS[name] = {
        "body": bid,
        "geoms": {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid},
    }

SHELF = {
    "pillar": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cart_shelf1"),
    "strip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cart_shelf3"),
}
PADG = {
    "r": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, x) for x in ("R_pad1", "R_pad2")],
    "l": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, x) for x in ("L_pad1", "L_pad2")],
}
FT = np.zeros(6)
FAIL = []
VALIDATION = {"version": 3, "task": "pillar_middle_then_strip_top", "objects": {}}


TICKS = 0


def frames(n):
    global TICKS
    TICKS += n
    for _ in range(n):
        ct.control_step(SUB)


def obj_pos(name):
    return d.xpos[OBJECTS[name]["body"]].copy()


def obj_extent(name):
    bounds = [ct.geom_aabb(g) for g in OBJECTS[name]["geoms"]]
    return np.min([x[0] for x in bounds], axis=0), np.max([x[1] for x in bounds], axis=0)


def pair_force(ga, gb):
    total = 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 in ga and g2 in gb) or (g2 in ga and g1 in gb):
            mujoco.mj_contactForce(m, d, i, FT)
            total += abs(FT[0])
    return total


def grip_force(hand, name):
    return pair_force(set(PADG[hand]), OBJECTS[name]["geoms"])


def pad_contacts_both(hand, name):
    seen = set()
    for i in range(d.ncon):
        gs = {d.contact[i].geom1, d.contact[i].geom2}
        if gs & OBJECTS[name]["geoms"]:
            seen |= gs & set(PADG[hand])
    return len(seen) == 2


def in_assigned_layer(name):
    lo, hi = obj_extent(name)
    slo, shi = ct.geom_aabb(SHELF[name])
    center = obj_pos(name)
    return (
        slo[0] - 0.04 <= center[0] <= shi[0] + 0.04
        and slo[1] - 0.04 <= center[1] <= shi[1] + 0.04
        and abs(lo[2] - shi[2]) < 0.08
    )


def placement_evidence(name):
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
    return {
        "assigned_shelf": "middle" if name == "pillar" else "top",
        "in_assigned_layer": bool(in_assigned_layer(name)),
        "released": bool(f_r < 0.5 and f_l < 0.5),
        "supported": bool(support >= 1.0),
        "grip_force_right_n": round(float(f_r), 3),
        "grip_force_left_n": round(float(f_l), 3),
        "cart_support_force_n": round(float(support), 3),
        "object_position": [round(float(x), 4) for x in obj_pos(name)],
    }


def gate(name, ok, detail=""):
    print(f"[gate] {name:16s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    if not ok:
        FAIL.append(f"{name}: {detail}")
    return ok


# Pillar demonstrations are also a suitable two-hand centre pinch for the
# longer strip.  v3 has zero rack-height offset and sufficient x/y reach.
VAR = {
    2: {"cart": (-2.20, 0.25), "obj": (0.66, 0.35), "dz": 0.00},
    3: {"cart": (-2.60, -0.25), "obj": (0.60, -0.30), "dz": 0.00},
    4: {"cart": (-2.40, 0.30), "obj": (0.62, 0.15), "dz": 0.10},
    5: {"cart": (-2.30, -0.30), "obj": (0.64, -0.20), "dz": -0.10},
    6: {"cart": (-2.55, 0.00), "obj": (0.60, 0.20), "dz": 0.10},
}


def load_reference(variant):
    demo = os.path.join(ROOT, "out", "teleop", "demos", f"pillar_v{variant}_refined")
    dd = np.load(os.path.join(demo, "episode_data.npz"))
    with open(os.path.join(demo, "refine.json")) as f:
        ge = int(json.load(f)["grasp_end"])
    action, base_action, base = dd["action"], dd["base_action"], dd["base"]
    n = len(action)
    moving = np.abs(base_action[:ge]).max(1)
    gs = 0
    for frame in range(ge):
        if moving[frame:ge].max() < 0.02:
            gs = frame
            break
    bx = base[:, 0]
    arrivals = np.where(bx <= bx.min() + 0.10)[0]
    arrivals = arrivals[arrivals > ge]
    ps = int(arrivals[0]) if len(arrivals) else int(ge + (n - ge) * 0.7)
    releases = np.where((action[:, 14] > 0.90) & (action[:, 15] > 0.90))[0]
    releases = releases[releases > ps]
    release = int(releases[0]) if len(releases) else n - 1
    place_end = min(n, release + 55)

    q0, v0 = d.qpos.copy(), d.qvel.copy()
    mount = {}
    for phase, f0, f1 in (("grasp", gs, ge), ("place", ps, place_end)):
        paths = {"l": [], "r": []}
        for frame in range(f0, f1):
            for i, adr in enumerate(ct.BQ):
                d.qpos[adr] = base[frame, i]
            for hand, arm in (("l", ct.L), ("r", ct.R)):
                sl = slice(0, 7) if hand == "l" else slice(7, 14)
                for j, adr in enumerate(arm.qadr):
                    d.qpos[adr] = action[frame, sl][j]
            mujoco.mj_kinematics(m, d)
            for hand, arm in (("l", ct.L), ("r", ct.R)):
                paths[hand].append(
                    (d.xpos[arm.mount].copy(), d.xmat[arm.mount].reshape(3, 3).copy())
                )
        mount[phase] = paths
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_forward(m, d)

    grasp_mid = 0.5 * (mount["grasp"]["l"][-1][0] + mount["grasp"]["r"][-1][0])
    ridx = min(release - ps, len(mount["place"]["l"]) - 1)
    release_mid = 0.5 * (
        mount["place"]["l"][ridx][0] + mount["place"]["r"][ridx][0]
    )
    ref_center = np.array([*VAR[variant]["obj"], 0.9277 + VAR[variant]["dz"]])
    release_obj = release_mid + (ref_center - grasp_mid)
    return {
        "variant": variant,
        "action": action,
        "base": base,
        "gs": gs,
        "ge": ge,
        "ps": ps,
        "release": release,
        "place_end": place_end,
        "mount": mount,
        "ref_center": ref_center,
        "release_obj": release_obj,
    }


pillar_variant = min(VAR, key=lambda x: abs(VAR[x]["obj"][1] - pillar_xy[1]))
REF = {"pillar": load_reference(pillar_variant), "strip": load_reference(3)}

# Randomise both free bodies and the robot after the offline reference FK.
for name, xy, nominal in (
    ("pillar", pillar_xy, PILLAR_NOM),
    ("strip", strip_xy, STRIP_NOM),
):
    bid = OBJECTS[name]["body"]
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]
    d.qpos[qadr] += xy[0] - nominal[0]
    d.qpos[qadr + 1] += xy[1] - nominal[1]
for i, adr in enumerate(ct.BQ):
    d.qpos[adr] = robot0[i]
ct.base_tgt[:] = robot0
mujoco.mj_forward(m, d)
frames(30)
for light in range(m.nlight):
    m.light_pos[light] += rng.uniform(-0.4, 0.4, 3)
    m.light_diffuse[light] = np.clip(m.light_diffuse[light] * rng.uniform(0.7, 1.25), 0.05, 1.0)

print(
    f"[dual] seed={SEED} pillar=({pillar_xy[0]:+.2f},{pillar_xy[1]:+.2f}) "
    f"strip=({strip_xy[0]:+.2f},{strip_xy[1]:+.2f}) "
    f"cart=({cart_xy[0]:+.2f},{cart_xy[1]:+.2f}) demos=pillar_v{pillar_variant}/strip_v3",
    flush=True,
)

os.makedirs(OUT, exist_ok=True)
rec = ct.EpisodeRecorder(OUT)
ct.REC["rec"] = rec
ct.REC["on"] = os.environ.get("E2E_NOREC") != "1"
ct.REC["count"] = 0
ct.REC["metadata"] = {
    "e2e": True,
    "task_version": "dual_two_trip_v1",
    "seed": SEED,
    "demo_variants": {"pillar": pillar_variant, "strip": 3},
    "pillar_xy": pillar_xy.tolist(),
    "strip_xy": strip_xy.tolist(),
    "cart_xy": cart_xy.tolist(),
    "robot0": robot0.tolist(),
    "trip_order": ["pillar_to_middle", "strip_to_top"],
    "validation": VALIDATION,
}


def finish(ok):
    VALIDATION["passed"] = bool(ok)
    VALIDATION["failed_gates"] = list(FAIL)
    ct.REC["on"] = False
    rec.finalize(success=bool(ok))
    print(f"[duration] ticks={TICKS} sim={TICKS * SUB * m.opt.timestep:.1f}s", flush=True)
    print(f"=== DUAL EPISODE {'PASS' if ok else 'FAIL'} ===", flush=True)
    if FAIL:
        print("  failed:", "; ".join(FAIL), flush=True)
    sys.exit(0)


def angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


N_KICKS = int(os.environ.get("E2E_KICKS", "4"))
kick_budget = N_KICKS


def maybe_kick(probability, scale=1.0):
    global kick_budget
    if kick_budget <= 0 or rng.random() > probability:
        return
    shift = scale * np.array(
        [rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), rng.uniform(-0.15, 0.15)]
    )
    for i, adr in enumerate(ct.BQ):
        d.qpos[adr] += shift[i]
    ct.base_tgt[:] += shift
    for joint in ct.BJ:
        d.qvel[m.jnt_dofadr[joint]] = 0.0
    mujoco.mj_forward(m, d)
    kick_budget -= 1
    print(f"[kick] remaining={kick_budget} shift=({shift[0]:+.2f},{shift[1]:+.2f},{shift[2]:+.2f})", flush=True)


def go_to(tx, ty, tyaw, vmax=0.25, wmax=0.5, kicks=True, kick_scale=1.0, max_frames=5000):
    for _ in range(max_frames):
        x, y, yaw = ct.base_pose()
        distance = float(np.hypot(tx - x, ty - y))
        if distance < 0.025:
            break
        heading = np.arctan2(ty - y, tx - x)
        error = angle(heading - yaw)
        reverse = abs(error) > np.pi / 2
        if reverse:
            error = angle(heading + np.pi - yaw)
        if abs(error) > 0.35:
            ct.base_vel[:] = [0.0, float(np.clip(2.0 * error, -wmax, wmax))]
        else:
            speed = float(np.clip(1.2 * distance, 0.05, vmax)) * (-1.0 if reverse else 1.0)
            ct.base_vel[:] = [speed, float(np.clip(1.8 * error, -wmax, wmax))]
        frames(DECIM)
        if kicks:
            maybe_kick(0.004, kick_scale)
    for _ in range(1800):
        error = angle(tyaw - ct.base_pose()[2])
        if abs(error) < 0.03:
            break
        ct.base_vel[:] = [0.0, float(np.clip(2.0 * error, -min(wmax, 0.45), min(wmax, 0.45)))]
        frames(DECIM)
    ct.base_vel[:] = 0.0
    frames(8)


def transform_mount(position, rotation, offset, yaw_rotation, pivot):
    c, s = np.cos(yaw_rotation), np.sin(yaw_rotation)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pivot + rz @ (position - pivot) + offset, rz @ rotation


def arm_replay(ref, phase, offset, ik_weight=0.6, yaw_rotation=0.0):
    f0 = ref["gs"] if phase == "grasp" else ref["ps"]
    f1 = ref["ge"] if phase == "grasp" else ref["place_end"]
    paths = ref["mount"][phase]
    pivot = ref["ref_center"]
    for idx, frame in enumerate(range(f0, f1)):
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets = {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            pm, rm = paths[hand][idx]
            target_pm, target_rm = transform_mount(pm, rm, offset, yaw_rotation, pivot)
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = ct.qtgt[hand][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(arm, target_pm, target_rm, iters=18, w=ik_weight)
            targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["l"][:] = targets["l"]
        ct.qtgt["r"][:] = targets["r"]
        ct.grip_cmd["l"] = float((1 - ref["action"][frame, 14]) * 0.025)
        ct.grip_cmd["r"] = float((1 - ref["action"][frame, 15]) * 0.025)
        ct.base_vel[:] = 0.0
        frames(DECIM)


HOME_L = ct.qtgt["l"].copy()
HOME_R = ct.qtgt["r"].copy()


def tuck(ref):
    frame = ref["ge"]
    goal_l, goal_r = ref["action"][frame, 0:7], ref["action"][frame, 7:14]
    start_l, start_r = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
    for i in range(60):
        blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 60)
        ct.qtgt["l"][:] = start_l + (goal_l - start_l) * blend
        ct.qtgt["r"][:] = start_r + (goal_r - start_r) * blend
        ct.grip_cmd["l"] = ct.GRIP_CLOSE
        ct.grip_cmd["r"] = ct.GRIP_CLOSE
        frames(DECIM)


def lift_joint_targets(delta):
    """Solve the common end pose, then use the stable demonstrated joint interpolation."""
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    starts = {"l": ct.qtgt["l"].copy(), "r": ct.qtgt["r"].copy()}
    targets = {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        for j, adr in enumerate(arm.qadr):
            d.qpos[adr] = ct.qtgt[hand][j]
        mujoco.mj_fwdPosition(m, d)
        position = d.xpos[arm.mount].copy() + np.asarray(delta, dtype=float)
        rotation = d.xmat[arm.mount].reshape(3, 3).copy()
        ct.ik(arm, position, rotation, iters=45, w=0.35)
        targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_fwdPosition(m, d)
    for i in range(110):
        blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 110)
        for hand in ("l", "r"):
            ct.qtgt[hand][:] = starts[hand] + (targets[hand] - starts[hand]) * blend
            ct.grip_cmd[hand] = ct.GRIP_CLOSE
        frames(DECIM)


def retreat_home():
    ct.grip_cmd["l"] = ct.GRIP_OPEN
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    frames(70)
    yaw = ct.base_pose()[2]
    for delta in (
        np.array([0.0, 0.0, 0.04]),
        np.array([-0.12 * np.cos(yaw), -0.12 * np.sin(yaw), 0.0]),
    ):
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets = {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            pm = d.xpos[arm.mount].copy() + delta
            rm = d.xmat[arm.mount].reshape(3, 3).copy()
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = ct.qtgt[hand][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(arm, pm, rm, iters=20, w=0.5)
            targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        ct.qtgt["l"][:] = targets["l"]
        ct.qtgt["r"][:] = targets["r"]
        frames(18)
    start_l, start_r = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
    for i in range(55):
        blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 55)
        ct.qtgt["l"][:] = start_l + (HOME_L - start_l) * blend
        ct.qtgt["r"][:] = start_r + (HOME_R - start_r) * blend
        ct.grip_cmd["l"] = ct.GRIP_OPEN
        ct.grip_cmd["r"] = ct.GRIP_OPEN
        frames(DECIM)
    frames(25)


def verify_placement(name):
    required = max(1, int(np.ceil(0.5 / (SUB * m.opt.timestep))))
    streak = 0
    evidence = placement_evidence(name)
    for _ in range(required * 4):
        frames(1)
        evidence = placement_evidence(name)
        valid = evidence["in_assigned_layer"] and evidence["released"] and evidence["supported"]
        streak = streak + 1 if valid else 0
        if streak >= required:
            break
    evidence["stable_for_s"] = round(float(streak * SUB * m.opt.timestep), 3)
    evidence["stable"] = bool(streak >= required)
    VALIDATION["objects"][name] = evidence
    return evidence


def plan_trip(name):
    ref = REF[name]
    xy = pillar_xy if name == "pillar" else strip_xy
    demo_obj = np.asarray(VAR[ref["variant"]]["obj"])
    grasp_offset = np.array(
        [xy[0] - demo_obj[0], xy[1] - demo_obj[1], -VAR[ref["variant"]]["dz"]]
    )
    park_grasp = ref["base"][ref["gs"]].copy()
    if name == "pillar":
        park_grasp[:2] += grasp_offset[:2]
        grasp_yaw_rotation = 0.0
        demo_cart = np.asarray(VAR[ref["variant"]]["cart"])
        place_offset = np.array([*(cart_xy - demo_cart), 0.0])
    else:
        # Mirror the reference about the object. This preserves the exact
        # demonstrated robot/object geometry while approaching from behind.
        park_grasp[:2] = xy - (ref["base"][ref["gs"], :2] - demo_obj)
        park_grasp[2] = angle(ref["base"][ref["gs"], 2] + np.pi)
        grasp_yaw_rotation = np.pi
        # The strip is 4 cm narrower than the pillar along the approach axis;
        # centre the pads instead of letting their back faces push it away.
        grasp_offset[0] -= 0.04
        shelf_top = ct.geom_aabb(SHELF["strip"])[1][2]
        desired_center = np.array([cart_xy[0], cart_xy[1] - 0.05, shelf_top + 0.040])
        place_offset = desired_center - ref["release_obj"]
    park_place = ref["base"][ref["ps"]].copy()
    park_place[:2] += place_offset[:2]
    if name == "strip":
        park_place[1] += 0.06
    return grasp_offset, place_offset, park_grasp, park_place, grasp_yaw_rotation


def run_trip(name):
    ref = REF[name]
    grasp_offset, place_offset, park_grasp, park_place, grasp_yaw_rotation = plan_trip(name)
    print(
        f"[trip:{name}] rack_target=({park_grasp[0]:+.2f},{park_grasp[1]:+.2f}) "
        f"cart_target=({park_place[0]:+.2f},{park_place[1]:+.2f}) place_dz={place_offset[2]:+.3f}",
        flush=True,
    )
    side = 1.0 if strip_xy[1] >= 0 else -1.0
    front_side = np.array([-0.35, side * 2.05, 0.0])
    back_side = np.array([2.08, side * 2.05, 0.0])
    if name == "strip":
        go_to(*front_side)
        go_to(*back_side)
    go_to(*park_grasp)
    arm_replay(ref, "grasp", grasp_offset, ik_weight=0.6, yaw_rotation=grasp_yaw_rotation)
    frames(10)
    if name == "strip":
        target_l, _ = transform_mount(*ref["mount"]["grasp"]["l"][-1], grasp_offset, grasp_yaw_rotation, ref["ref_center"])
        target_r, _ = transform_mount(*ref["mount"]["grasp"]["r"][-1], grasp_offset, grasp_yaw_rotation, ref["ref_center"])
        pad_pos = {
            hand: [[round(float(v), 3) for v in d.geom_xpos[g]] for g in PADG[hand]]
            for hand in ("l", "r")
        }
        print(
            f"[strip_probe] obj={np.round(obj_pos(name), 3).tolist()} "
            f"mount_err_mm=({1000*np.linalg.norm(d.xpos[ct.L.mount]-target_l):.1f},"
            f"{1000*np.linalg.norm(d.xpos[ct.R.mount]-target_r):.1f}) pads={pad_pos}",
            flush=True,
        )
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    if not gate(
        f"{name}_grip",
        pad_contacts_both("r", name) and pad_contacts_both("l", name) and f_r >= 1.0 and f_l >= 1.0,
        f"bothR={pad_contacts_both('r', name)} bothL={pad_contacts_both('l', name)} fR={f_r:.1f}N fL={f_l:.1f}N",
    ):
        finish(False)

    tuck(ref)
    carry_hands = ["r", "l"]

    def require_held(stage):
        force_r, force_l = grip_force("r", name), grip_force("l", name)
        print(f"[carry_probe:{name}] {stage} fR={force_r:.1f}N fL={force_l:.1f}N", flush=True)
        forces = {"r": force_r, "l": force_l}
        if any(forces[hand] < 0.5 for hand in carry_hands):
            gate(f"{name}_carry", False, f"dropped at {stage}: fR={force_r:.1f}N fL={force_l:.1f}N")
            finish(False)

    require_held("tucked")
    if name == "strip":
        # Pull straight back, rotate only while stationary and drive each leg
        # nearly straight. The long part cannot tolerate normal chassis yaw.
        clear_back = np.array([2.08, park_grasp[1], park_grasp[2]])
        go_to(*clear_back, vmax=0.12, wmax=0.06, kick_scale=0.20)
        require_held("clear_rack")
        heading_side = np.arctan2(back_side[1] - clear_back[1], back_side[0] - clear_back[0])
        go_to(clear_back[0], clear_back[1], heading_side, wmax=0.06)
        require_held("turned_to_side")
        go_to(back_side[0], back_side[1], heading_side, vmax=0.08, wmax=0.06, kick_scale=0.20)
        require_held("back_side")
        go_to(back_side[0], back_side[1], np.pi, wmax=0.06)
        go_to(front_side[0], front_side[1], np.pi, vmax=0.08, wmax=0.06, kick_scale=0.20)
        require_held("front_side")
        lift_joint_targets([0.0, 0.0, place_offset[2]])
        force_r, force_l = grip_force("r", name), grip_force("l", name)
        if force_l >= 0.5:
            carry_hands[:] = ["l"]
            released_hand = "r"
        elif force_r >= 0.5:
            carry_hands[:] = ["r"]
            released_hand = "l"
        else:
            gate(f"{name}_carry", False, "both hands lost during top lift")
            finish(False)
        print(f"[top_handoff] keep={carry_hands[0]} fR={force_r:.1f}N fL={force_l:.1f}N", flush=True)
        ct.grip_cmd[released_hand] = ct.GRIP_OPEN
        frames(20)
        require_held("raised_for_top")
        print(f"[strip_lift] object_z={obj_pos(name)[2]:.3f}", flush=True)
        align_cart = np.array([-0.75, park_place[1], 0.0])
        heading_align = np.arctan2(align_cart[1] - front_side[1], align_cart[0] - front_side[0])
        go_to(front_side[0], front_side[1], heading_align, wmax=0.06)
        go_to(align_cart[0], align_cart[1], heading_align, vmax=0.10, wmax=0.06, kick_scale=0.20)
        require_held("cart_alignment_point")
        go_to(align_cart[0], align_cart[1], park_place[2], wmax=0.06)
        require_held("aligned_with_cart")
        go_to(*park_place, vmax=0.08, wmax=0.06, kick_scale=0.20)
    else:
        go_to(*park_place, vmax=0.20, kick_scale=0.35)
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    carry_forces = {"r": f_r, "l": f_l}
    if not gate(f"{name}_carry", all(carry_forces[hand] >= 0.5 for hand in carry_hands),
                f"hands={carry_hands} fR={f_r:.1f}N fL={f_l:.1f}N"):
        finish(False)

    if name != "strip":
        arm_replay(ref, "place", place_offset, ik_weight=0.6)
    ct.base_vel[:] = 0.0
    retreat_home()
    evidence = verify_placement(name)
    p = obj_pos(name)
    if not gate(
        f"{name}_placed",
        evidence["stable"],
        f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) released={evidence['released']} "
        f"support={evidence['cart_support_force_n']:.1f}N stable={evidence['stable_for_s']:.2f}s",
    ):
        finish(False)


# Fixed semantic order: red circled pillar first, green circled strip second.
run_trip("pillar")
run_trip("strip")

# Recheck both after the second trip: contact from the top-layer operation must
# not have displaced the already completed middle-layer placement.
pillar_final = verify_placement("pillar")
strip_final = verify_placement("strip")
both = pillar_final["stable"] and strip_final["stable"]
gate(
    "placed_both",
    both,
    f"pillar={pillar_final['stable']}({pillar_final['cart_support_force_n']:.1f}N) "
    f"strip={strip_final['stable']}({strip_final['cart_support_force_n']:.1f}N)",
)
finish(both)
