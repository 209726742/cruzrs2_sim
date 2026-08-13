#!/usr/bin/env python3
"""Two-trip shelf expert: pillar -> cart middle, strip -> cart top.

The task succeeds only when both released objects remain supported by their
assigned cart shelves for 0.5 seconds.  The old single-object collector and
its data are intentionally left untouched.

Env: SEED (required), EXPERT_OUT, E2E_DIVERSITY_MODE=clean|recovery,
E2E_KICKS=0 (clean default; recovery requires 1), E2E_NOREC=1,
E2E_LAYOUT_MODE=random|boundary, E2E_TOUCHDOWN=1.
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

from action_quality import analyze_motion  # noqa: E402
from cruzr_s2_sdk_contract import (  # noqa: E402
    ARM_RATED_DELTA_RAD_AT_DATASET_FPS,
    SDK_BASE_V_FWD_RANGE_M_S,
    SDK_BASE_WZ_RANGE_RAD_S,
    SDK_CAMERA_STATE_MAX_SKEW_S,
    SDK_COLLECTION_PROFILE,
    SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS,
    SDK_DOC_REVISION,
    SDK_TASK_HEAD_POSE_RAD,
    audit_sdk_episode,
    clip_arm_target_to_operational_limits,
)
from shelf_e2e_flex_state import (  # noqa: E402
    capture_internal_state,
    object_state_contract,
    save_internal_state,
)
from shelf_e2e_objects import object_info, root_pose  # noqa: E402
from shelf_e2e_profiles import (  # noqa: E402
    STRICT_COLLECTION_PROFILE,
    collection_cameras,
    normalize_collection_profile,
)

SEED = int(os.environ["SEED"])
rng = np.random.default_rng(SEED)
COLLECTION_PROFILE = normalize_collection_profile(
    os.environ.get("E2E_COLLECTION_PROFILE")
)
SDK_RECOVERY = COLLECTION_PROFILE == SDK_COLLECTION_PROFILE
RECORD_CAMERAS = collection_cameras(COLLECTION_PROFILE)
N_KICKS = int(os.environ.get("E2E_KICKS", "0"))
DIVERSITY_MODE = os.environ.get("E2E_DIVERSITY_MODE", "clean").strip()
if DIVERSITY_MODE not in ("clean", "recovery"):
    raise ValueError(
        f"unsupported E2E_DIVERSITY_MODE {DIVERSITY_MODE!r}; expected clean or recovery"
    )
if DIVERSITY_MODE == "clean" and N_KICKS != 0:
    raise ValueError("clean diversity mode requires E2E_KICKS=0")
if DIVERSITY_MODE == "recovery" and N_KICKS != 1:
    raise ValueError("recovery diversity mode requires E2E_KICKS=1")
PERTURBATION_EVENTS = []
LAYOUT_MODE = os.environ.get("E2E_LAYOUT_MODE", "random").strip()
if LAYOUT_MODE not in ("random", "boundary"):
    raise ValueError(
        f"unsupported E2E_LAYOUT_MODE {LAYOUT_MODE!r}; expected random or boundary"
    )
BOUNDARY_AXES = (
    "cart_x", "cart_y", "rack_y", "robot_x", "robot_y", "robot_yaw"
)
boundary_axis = str(rng.choice(BOUNDARY_AXES)) if LAYOUT_MODE == "boundary" else None


def sample_layout_axis(name, low, high):
    """Sample normally, or put exactly one selected axis in the outer 20%."""
    if name != boundary_axis:
        return rng.uniform(low, high)
    edge_width = 0.20 * (high - low)
    if rng.integers(0, 2) == 0:
        return rng.uniform(low, low + edge_width)
    return rng.uniform(high - edge_width, high)

# Correlate the two rack parts in y: they belong to the same visible rack bay,
# while small independent offsets prevent memorising one fixed alignment.
CART_NOM = np.array([-2.40, 0.0])
PILLAR_NOM = np.array([0.58, 0.0005])
STRIP_NOM = np.array([1.05, 0.0])
cart_xy = CART_NOM + np.array([
    sample_layout_axis("cart_x", -0.20, 0.20),
    sample_layout_axis("cart_y", -0.30, 0.30),
])
rack_y = sample_layout_axis("rack_y", -0.24, 0.24)
pillar_xy = PILLAR_NOM + np.array([rng.uniform(-0.04, 0.04), rack_y + rng.uniform(-0.035, 0.035)])
strip_xy = STRIP_NOM + np.array([rng.uniform(-0.03, 0.03), rack_y + rng.uniform(-0.035, 0.035)])
robot0 = np.array([
    sample_layout_axis("robot_x", -0.08, 0.08),
    sample_layout_axis("robot_y", -0.08, 0.08),
    sample_layout_axis("robot_yaw", -0.12, 0.12),
])

# Per-seed XML lives in assets because included robot mesh paths resolve from
# the main XML directory. A run-id prevents different GPU shards from sharing
# a filename even if their seed ranges were configured incorrectly.
SCENE_DIR = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
SCENE_RUN_ID = os.environ.get("E2E_RUN_ID", "")
if SCENE_RUN_ID and not re.fullmatch(r"[A-Za-z0-9._-]+", SCENE_RUN_ID):
    raise ValueError(f"unsupported E2E_RUN_ID: {SCENE_RUN_ID!r}")
scene_token = f"{SCENE_RUN_ID}_{SEED}" if SCENE_RUN_ID else str(SEED)
os.makedirs(SCENE_DIR, exist_ok=True)
with open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")) as f:
    scene_text = f.read()
scene_text = re.sub(
    r'(<body name="shelf_cart" pos=")[^"]*(")',
    lambda x: f'{x.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{x.group(2)}',
    scene_text,
)
SCENE = os.path.join(SCENE_DIR, f"e2e_dual_scene_{scene_token}.xml")
with open(SCENE, "w") as f:
    f.write(scene_text)

os.environ["TELEOP_SCENE_XML"] = SCENE
os.environ["CRUZR_EP_SEED"] = str(SEED)
os.environ.setdefault("TELEOP_HOME", "droop")
os.environ.setdefault("MUJOCO_GL", "egl")
configured_cameras = tuple(
    camera for camera in os.environ.get("REC_CAMS", "").split(",") if camera
)
if configured_cameras and configured_cameras != RECORD_CAMERAS:
    raise ValueError(
        f"REC_CAMS {configured_cameras} conflicts with {COLLECTION_PROFILE} "
        f"camera contract {RECORD_CAMERAS}"
    )
os.environ["REC_CAMS"] = ",".join(RECORD_CAMERAS)
if SDK_RECOVERY:
    if os.environ.get("REC_SAVE_RAW_TIMESTAMPS", "1") != "1":
        raise ValueError("sdk_recovery_v1 requires REC_SAVE_RAW_TIMESTAMPS=1")
    os.environ["REC_SAVE_RAW_TIMESTAMPS"] = "1"
os.environ.setdefault(
    "REC_PROMPT",
    "move the steel pillar to the middle shelf of the cart, then move the rubber strip to the top shelf",
)

spec = importlib.util.spec_from_file_location(
    "cruzr_teleop", os.path.join(CORE_DIR, "cruzr_teleop.py")
)
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)
import mujoco  # noqa: E402
from teleop_timing import CumulativeSubstepScheduler  # noqa: E402

m, d = ct.m, ct.d
ct.REC_WH = (224, 224)
SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
DECIM = int(getattr(ct, "REC_DECIM", 2))
SUBSTEP_SCHEDULER = (
    CumulativeSubstepScheduler(ct.TARGET_FPS, m.opt.timestep)
    if SDK_RECOVERY else None
)
CONTROL_DT = (
    1.0 / float(ct.TARGET_FPS)
    if SDK_RECOVERY else SUB * m.opt.timestep
)
OUT = os.path.join(
    ROOT,
    os.environ.get("EXPERT_OUT", f"out/teleop/shelf_e2e_dual/shelf_e2e_dual_{SEED:06d}"),
)
# A NOREC self-check that forgets to set EXPERT_OUT lands on the official episode name
# and, on abort, leaves a 0-frame meta.json behind. That is how the 429.9 s seed 13
# baseline lost its metadata and twelve empty shells appeared under official names.
if os.environ.get("E2E_NOREC") == "1" and re.fullmatch(
        r"shelf_e2e_dual_\d{6}", os.path.basename(OUT.rstrip("/"))):
    raise SystemExit(
        f"refusing to run a NOREC self-check into the official episode name "
        f"{os.path.basename(OUT)!r}. Set EXPERT_OUT to a scratch path, e.g. "
        f"EXPERT_OUT=out/teleop/shelf_e2e_dual/_ab_<candidate>_{SEED}"
    )

# ---- chassis motion shaping -------------------------------------------------
# What swings the 1.6 m strip out of the grippers is yaw *acceleration*, not the
# steady-state rate. Ramping the command lets the strip legs run ~4x faster.
BASE_ACC = float(os.environ.get("E2E_BASE_ACC", "0.5"))    # m/s^2 on commanded v
BASE_YACC = float(os.environ.get("E2E_BASE_YACC", "0.2"))  # rad/s^2 on commanded w
STRIP_VMAX = float(os.environ.get("E2E_STRIP_VMAX", "0.20"))
STRIP_WMAX = float(os.environ.get("E2E_STRIP_WMAX", "0.25"))

# ---- placement -------------------------------------------------------------
# Grasp and place are otherwise fully open loop: a parking difference well inside
# the 0.025 m go_to tolerance changes where the pads close on the part, and nothing
# downstream corrects for it. PLACE_FIX measures the in-hand offset and feeds it back.
PLACE_FIX = int(os.environ.get("E2E_PLACE_FIX", "1"))
PLACE_CORR_MAX = float(os.environ.get("E2E_PLACE_CORR_MAX", "0.25"))  # m, safety clamp
QDOT_MAX = float(os.environ.get("E2E_QDOT_MAX", "0.020"))  # rad/control tick
# The reference distribution has only three discontinuities above 0.15 rad
# (0.316/0.397/0.915); interpolating smaller, normal IK motion changes contact timing.
default_action_delta = (
    ARM_RATED_DELTA_RAD_AT_DATASET_FPS if SDK_RECOVERY else 0.15
)
ACTION_DELTA_MAX = float(
    os.environ.get("E2E_ACTION_DELTA_MAX", str(default_action_delta))
)
COMMAND_DELTA_MAX = (
    min(ACTION_DELTA_MAX, SDK_COMMAND_DELTA_RAD_AT_DATASET_FPS)
    if SDK_RECOVERY else ACTION_DELTA_MAX
)
SDK_CONTROL_TICK_DELTA_MAX = COMMAND_DELTA_MAX / DECIM
HOME_COMMAND_STEP = SDK_CONTROL_TICK_DELTA_MAX if SDK_RECOVERY else 0.06
HOME_FEEDBACK_LEAD = HOME_COMMAND_STEP if SDK_RECOVERY else 0.06
if SDK_RECOVERY and ACTION_DELTA_MAX > ARM_RATED_DELTA_RAD_AT_DATASET_FPS + 1e-9:
    raise ValueError(
        f"sdk_recovery_v1 E2E_ACTION_DELTA_MAX={ACTION_DELTA_MAX} exceeds "
        f"the SDK rated 30 FPS delta {ARM_RATED_DELTA_RAD_AT_DATASET_FPS}"
    )
if SDK_RECOVERY and QDOT_MAX > SDK_CONTROL_TICK_DELTA_MAX + 1e-9:
    raise ValueError(
        f"sdk_recovery_v1 E2E_QDOT_MAX={QDOT_MAX} exceeds per-control-tick "
        f"rated delta {SDK_CONTROL_TICK_DELTA_MAX}"
    )
TRACKING_P95_MAX = float(os.environ.get("E2E_TRACKING_P95_MAX", "0.03"))
TRACKING_MAX = float(os.environ.get("E2E_TRACKING_MAX", "0.15"))
TERMINAL_TRACKING_MAX = float(os.environ.get("E2E_TERMINAL_TRACKING_MAX", "0.05"))
TRACK_SETTLE_TOL = float(os.environ.get("E2E_TRACK_SETTLE_TOL", "0.03"))
TRACK_SETTLE_TICKS = int(os.environ.get("E2E_TRACK_SETTLE_TICKS", "180"))
# Kept as the backward-compatible on/off switch for the demo-seeded grasp entry.
# 0 disables it; positive values enable it.
IK_HOP_M = float(os.environ.get("E2E_IK_HOP_M", "0.05"))
TOUCHDOWN = int(os.environ.get("E2E_TOUCHDOWN", "1"))
# Separate knob from TOUCHDOWN because the pillar reaches the shelf held in two hands via
# a demo place replay, while the strip arrives from a hand-over lift, so the two need to
# be regressed independently.
PILLAR_TOUCHDOWN = int(os.environ.get("E2E_PILLAR_TOUCHDOWN", "0"))
# Arc length between replayed demo frames, rad. 0 replays the demo one frame at a time,
# which is what makes the arms hesitate 64 times and vary between 0 and 2.6 rad/s.
# 0.028 was chosen on peak contact force and on a held-out seed set, not on the seeds that
# had been iterated over: 8/13 on seeds 1-13 and 5/13 on seeds 14-26, against 7/13 and
# 1/13 for 1:1 replay. Slower is not safer -- 0.008 and 0.024 both spike past 500 N
# because a saturated IK command is held for longer and the pad springs keep loading.
MAX_STEP = float(os.environ.get("E2E_MAX_STEP", "0.028"))
# Frames held after the gripper finishes closing or opening, so the pads can seat before
# the arm moves on. This is the one thing the demo's idle frames genuinely provided.
GRIP_SETTLE = int(os.environ.get("E2E_GRIP_SETTLE", "10"))
# Which trips to time-warp: none, pillar, or all. Warping the strip trip is not currently
# safe -- it reaches the rack with the strip already nudged ~10 mm by the faster pillar
# trip, and one pad then misses the 32 mm bar.
WARP_SCOPE = os.environ.get("E2E_WARP", "pillar")
GRASP_DUMP = int(os.environ.get("E2E_GRASP_DUMP", "0"))
# Align each hand on the section it closes on, instead of the demo's blanket -40 mm shift.
# On held-out seeds 14-26 this plus PARK_FROM_HAND takes 3/13 to 10/13, and it clears
# strip_grip and strip_carry outright; the two belong together, because a firmly held bar
# is what makes an accurate place target matter in the first place.
GRASP_FIX = os.environ.get("E2E_GRASP_FIX", "strip")
GRASP_CORR_MAX = float(os.environ.get("E2E_GRASP_CORR_MAX", "0.08"))
# Park the chassis from the measured in-hand pose rather than the demo's release frame.
PARK_FROM_HAND = int(os.environ.get("E2E_PARK_FROM_HAND", "1"))
PLACE_REACH = int(os.environ.get("E2E_PLACE_REACH", "0"))
PILLAR_PLACE_BIAS_X = float(os.environ.get("E2E_PILLAR_PLACE_BIAS_X", "0.020"))
# Minimum x clearance, m, between the bar and anything standing proud of the shelf surface.
POST_CLEAR = float(os.environ.get("E2E_POST_CLEAR", "0"))
RETREAT = os.environ.get("E2E_RETREAT", "release")
HOME_ORDER = os.environ.get("E2E_HOME_ORDER", "rl")
SEAT = os.environ.get("E2E_SEAT", "0") == "1"
LEVEL = os.environ.get("E2E_LEVEL", "1") == "1"
CLEAR_POSTS = os.environ.get("E2E_CLEAR_POSTS", "1") == "1"
UNGRASP_M = float(os.environ.get("E2E_UNGRASP_MM", "50")) / 1000.0
STRIP_LIFT = float(os.environ.get("E2E_STRIP_LIFT", "0.04"))
STRIP_LIFT_STAGE_BACKOFF = 0.40
STRIP_TOP_CLEARANCE = 0.080

OBJECTS = {}
for name in ("pillar", "strip"):
    OBJECTS[name] = object_info(m, name)
TASK_VERSION, OBJECT_INTERNAL_CONTRACT = object_state_contract(OBJECTS)

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
# Every gate in order, so a run can be judged from result.json instead of by reading a
# terminal log that may be truncated. Peak contact force is tracked because a candidate
# that "passes" while spiking hundreds of newtons is a docking collision, not a success.
GATES = []
PEAK_FORCE = {"value": 0.0, "where": ""}


TRIP = "pillar"
TICKS = 0
PHYSICS_STEPS = 0
AUDIT_TICKS = 0
MOTION_LOG = None
POLICY_LOG_ON = False
# Object poses are not part of the LeRobot schema, so the third-person replay used to
# fake them by pinning the part to the pads and freezing it on release. Log the real
# poses at the recorder's cadence so the replay can show what actually happened.
OBJ_LOG = []
OBJ_INTERNAL_LOG = []


def frames(n):
    global TICKS, PHYSICS_STEPS, AUDIT_TICKS
    for _ in range(n):
        TICKS += 1
        if SDK_RECOVERY:
            operational_target = clip_arm_target_to_operational_limits(
                np.concatenate((ct.qtgt["l"], ct.qtgt["r"]))
            )
            ct.qtgt["l"][:] = operational_target[:7]
            ct.qtgt["r"][:] = operational_target[7:14]
        substeps = (
            SUBSTEP_SCHEDULER.next_substeps()
            if SUBSTEP_SCHEDULER is not None else SUB
        )
        ct.control_step(substeps)
        PHYSICS_STEPS += substeps
        if MOTION_LOG is not None and POLICY_LOG_ON:
            AUDIT_TICKS += 1
            if AUDIT_TICKS % DECIM == 0:
                MOTION_LOG["state"].append(rec._state16())
                MOTION_LOG["action"].append(rec._action16())
                MOTION_LOG["base_action"].append(np.asarray(ct.base_vel, dtype=np.float32).copy())
                MOTION_LOG["phase"].append(ct.REC.get("phase", "unknown"))
        if ct.REC["on"] and ct.REC["count"] % DECIM == 0:
            OBJ_LOG.append(np.concatenate([
                root_pose(d, OBJECTS[nm])
                for nm in ("pillar", "strip")
            ]))
            if OBJECT_INTERNAL_CONTRACT is not None:
                OBJ_INTERNAL_LOG.append(capture_internal_state(d, OBJECTS["strip"]))


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
    f = pair_force(set(PADG[hand]), OBJECTS[name]["geoms"])
    if f > PEAK_FORCE["value"]:
        PEAK_FORCE.update(value=float(f), where=f"{hand}/{name}")
    return f


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

def fully_on_shelf(name, tol=0.002):
    slo, shi = ct.geom_aabb(SHELF[name])
    lo, hi = obj_extent(name)
    return (
        lo[0] >= slo[0] - tol and hi[0] <= shi[0] + tol
        and lo[1] >= slo[1] - tol and hi[1] <= shi[1] + tol
    )


def shelf_geometry(name):
    """The numbers behind the containment gate, in millimetres.

    overhang is positive when the part sticks out past that shelf edge, so a valid
    placement has all four negative. margin_mm is the tightest of the four.
    """
    slo, shi = ct.geom_aabb(SHELF[name])
    lo, hi = obj_extent(name)
    centre = obj_pos(name)
    shelf_c = 0.5 * (slo + shi)
    over = {
        "-x": 1000.0 * float(slo[0] - lo[0]), "+x": 1000.0 * float(hi[0] - shi[0]),
        "-y": 1000.0 * float(slo[1] - lo[1]), "+y": 1000.0 * float(hi[1] - shi[1]),
    }
    return {
        "centre": [round(float(v), 4) for v in centre],
        "shelf_centre": [round(float(shelf_c[0]), 4), round(float(shelf_c[1]), 4)],
        "off_centre_m": [round(float(centre[0] - shelf_c[0]), 4),
                         round(float(centre[1] - shelf_c[1]), 4)],
        "gap_to_surface_mm": round(1000.0 * float(lo[2] - shi[2]), 1),
        "overhang_mm": {k: round(v, 1) for k, v in over.items()},
        "margin_mm": round(-max(over.values()), 1),
        "fully_on_shelf": bool(fully_on_shelf(name)),
        "support_n": round(float(pair_force(OBJECTS[name]["geoms"], {SHELF[name]})), 2),
    }


def nearest_section(name, y):
    """AABB of the part's segment lying closest to y along the bar."""
    best, best_dy = None, None
    for g in OBJECTS[name]["geoms"]:
        glo, ghi = ct.geom_aabb(g)
        dy = abs(0.5 * (glo[1] + ghi[1]) - y)
        if best_dy is None or dy < best_dy:
            best, best_dy = (glo, ghi), dy
    return best


def hand_section_clearance(name, hand):
    """Smallest AABB gap from either pad to the nearest part section."""
    clearance = float("inf")
    for pad in PADG[hand]:
        plo, phi = ct.geom_aabb(pad)
        glo, ghi = nearest_section(name, float(d.geom_xpos[pad][1]))
        axis_gap = np.maximum(np.maximum(glo - phi, plo - ghi), 0.0)
        clearance = min(clearance, float(np.linalg.norm(axis_gap)))
    return clearance


def grasp_correction_per_hand(name, ref, offset, yaw_rotation):
    """Align each hand to the cross-section it will actually close on.

    The bar is arched -- 15 segments whose local z spans 53 mm -- so its surface sits at a
    different height under each hand, and the two grasp points are ~340 mm apart. The pads
    close across the ~20 mm thickness with roughly 1.5 mm of clearance a side, so one offset
    cannot serve both hands.

    Each hand is corrected at its own closest approach rather than at the end of the grasp
    segment, because that segment ends already lifted, ~136 mm above where the bar rests; a
    correction computed there drove the arm into the rack at 216 N. The two hands do not
    arrive together either -- in the recorded demo the right reaches the bar around frame
    832 and the left not until 1769 -- so a single frame cannot serve both.

    On the recorded demo this finds the right hand a steady 16.5 mm off the section it
    closes on while the left lands within 2.1 mm, which is why the right shoves the bar out
    from under the left rather than the other way round.
    """
    paths = ref["mount"]["grasp"]
    pivot = ref["ref_center"]
    hand_offset = {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        mount_pos = d.xpos[arm.mount]
        mount_rot = d.xmat[arm.mount].reshape(3, 3)
        pad_local = [mount_rot.T @ (d.geom_xpos[g] - mount_pos) for g in PADG[hand]]
        best = None
        for idx in range(len(paths[hand])):
            tp, tr = transform_mount(*paths[hand][idx], offset, yaw_rotation, pivot)
            pads = np.mean([tp + tr @ loc for loc in pad_local], axis=0)
            glo, ghi = nearest_section(name, float(pads[1]))
            target = 0.5 * (glo + ghi)
            delta = np.array([target[0] - pads[0], 0.0, target[2] - pads[2]])
            dist = float(np.linalg.norm(delta))
            if best is None or dist < best[0]:
                best = (dist, idx, delta, float(ghi[2] - glo[2]))
        dist, idx, delta, thickness = best
        clamped = dist > GRASP_CORR_MAX
        if clamped:
            delta = delta * (GRASP_CORR_MAX / dist)
        hand_offset[hand] = delta
        print(f"[grasp_fix:{name}:{hand}] closest at frame {idx} miss={1000*dist:.1f}mm "
              f"dx={1000*delta[0]:+.1f}mm dy={1000*delta[1]:+.1f}mm "
              f"dz={1000*delta[2]:+.1f}mm"
              f"{' CLAMPED' if clamped else ''} section_thickness_mm={1000*thickness:.1f}",
              flush=True)
    return hand_offset


def gname(g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


def grasp_dump(name):
    """Everything about the grasp pose, to find why one pad misses the part.

    Reports what each pad is actually touching, how far each pad sits from the part's
    surface along the pinch axis, and how close every arm joint is to its limit -- a
    saturated joint means the commanded pose was outside the reachable set even though
    the mount residual looks small.
    """
    lo, hi = obj_extent(name)
    print(f"[dump:{name}] part_aabb lo={np.round(lo, 4).tolist()} hi={np.round(hi, 4).tolist()} "
          f"size_mm={np.round(1000 * (hi - lo), 1).tolist()}", flush=True)
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        for g in PADG[hand]:
            p = d.geom_xpos[g]
            inside = [bool(lo[k] - 1e-4 <= p[k] <= hi[k] + 1e-4) for k in range(3)]
            outside_mm = [round(1000 * float(max(lo[k] - p[k], p[k] - hi[k], 0.0)), 1)
                          for k in range(3)]
            print(f"[dump:{name}] pad {gname(g):<8} pos={np.round(p, 4).tolist()} "
                  f"inside_xyz={inside} outside_mm={outside_mm}", flush=True)
        touching = {}
        for i in range(d.ncon):
            gs = {d.contact[i].geom1, d.contact[i].geom2}
            hit = gs & set(PADG[hand])
            if hit:
                other = (gs - set(PADG[hand])) or hit
                for o in other:
                    mujoco.mj_contactForce(m, d, i, FT)
                    touching[gname(o)] = touching.get(gname(o), 0.0) + abs(float(FT[0]))
        print(f"[dump:{name}] {hand} pad contacts: "
              f"{ {k: round(v, 2) for k, v in touching.items()} or 'none'}", flush=True)
        near = []
        for j, adr in enumerate(arm.qadr):
            q = float(d.qpos[adr])
            slack = min(q - arm.lo[j], arm.hi[j] - q)
            if slack < 0.05:
                near.append(f"j{j}={q:+.3f}(lo={arm.lo[j]:+.2f},hi={arm.hi[j]:+.2f})")
        print(f"[dump:{name}] {hand} joints at limit: {near or 'none'}", flush=True)


def final_contacts(name):
    """What is holding the part up, and how high each of its sections sits.

    The bar weighs 3.93 N but the shelf was only reading 2.61 N of it, which is the kind of
    gap that means something else is taking the load. Its sections are also worth listing
    separately: the part is arched by 53 mm, so a correctly seated bar still shows a raised
    middle and that is not a placement fault.
    """
    _, shi = ct.geom_aabb(SHELF[name])
    loads = {}
    for i in range(d.ncon):
        pair = {d.contact[i].geom1, d.contact[i].geom2}
        mine = pair & OBJECTS[name]["geoms"]
        if not mine:
            continue
        mujoco.mj_contactForce(m, d, i, FT)
        for other in (pair - OBJECTS[name]["geoms"]) or mine:
            loads[gname(other)] = loads.get(gname(other), 0.0) + abs(float(FT[0]))
    print(f"[final:{name}] weight={OBJECTS[name]['weight_n']:.2f}N "
          f"shelf_top_z={shi[2]:.4f} touching={ {k: round(v, 2) for k, v in loads.items()} or 'nothing'}",
          flush=True)
    rows = []
    for g in sorted(OBJECTS[name]["geoms"]):
        glo, ghi = ct.geom_aabb(g)
        rows.append((0.5 * (glo[1] + ghi[1]), gname(g), glo[2] - shi[2]))
    for y, nm, above in sorted(rows):
        print(f"[final:{name}]   {nm:<14} y={y:+.3f} bottom_above_surface={1000 * above:+6.1f}mm",
              flush=True)


def placement_audit(name):
    g = shelf_geometry(name)
    over = g["overhang_mm"]
    print(
        f"[audit:{name}] centre={g['centre'][:3]} "
        f"shelf_centre=({g['shelf_centre'][0]:.3f},{g['shelf_centre'][1]:.3f}) "
        f"off_centre=({g['off_centre_m'][0]:+.3f},{g['off_centre_m'][1]:+.3f}) "
        f"gap_to_surface={g['gap_to_surface_mm']:+.0f}mm "
        f"overhang_mm={{'-x':{over['-x']:+.0f},'+x':{over['+x']:+.0f},"
        f"'-y':{over['-y']:+.0f},'+y':{over['+y']:+.0f}}} "
        f"margin={g['margin_mm']:+.0f}mm inside={g['fully_on_shelf']} "
        f"support={g['support_n']:.1f}N",
        flush=True,
    )


def placement_evidence(name):
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
    return {
        "assigned_shelf": "middle" if name == "pillar" else "top",
        "in_assigned_layer": bool(in_assigned_layer(name)),
        "fully_on_shelf": bool(fully_on_shelf(name)),
        "released": bool(f_r < 0.5 and f_l < 0.5),
        "supported": bool(support >= 1.0),
        "grip_force_right_n": round(float(f_r), 3),
        "grip_force_left_n": round(float(f_l), 3),
        "cart_support_force_n": round(float(support), 3),
        "object_position": [round(float(x), 4) for x in obj_pos(name)],
    }


def gate(name, ok, detail=""):
    print(f"[gate] {name:16s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    GATES.append({"gate": name, "passed": bool(ok), "detail": detail})
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


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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
    # The part is rigid in the grippers, so its offset from the hand midpoint is fixed
    # in the *base* frame, not the world frame. The demo turns ~180 deg between grasp
    # and release, so carrying the world-frame offset across mis-predicts the release
    # point by up to twice the offset (~0.3 m).
    inhand_base = rz(-base[ge, 2]) @ (ref_center - grasp_mid)
    release_obj = release_mid + rz(base[release, 2]) @ inhand_base
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
        "inhand_base": inhand_base,
    }


pillar_variant = int(os.environ.get("E2E_PILLAR_VARIANT", "3"))
REF = {"pillar": load_reference(pillar_variant), "strip": load_reference(3)}

# Randomise both free bodies and the robot after the offline reference FK.
for name, xy, nominal in (
    ("pillar", pillar_xy, PILLAR_NOM),
    ("strip", strip_xy, STRIP_NOM),
):
    qadr = OBJECTS[name]["free_qpos_adr"]
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
    f"cart=({cart_xy[0]:+.2f},{cart_xy[1]:+.2f}) "
    f"layout={LAYOUT_MODE}:{boundary_axis or 'none'} "
    f"demos=pillar_v{pillar_variant}/strip_v3",
    flush=True,
)

os.makedirs(OUT, exist_ok=True)
rec = ct.EpisodeRecorder(OUT)
ct.REC["rec"] = rec
ct.REC["on"] = os.environ.get("E2E_NOREC") != "1"
ct.REC["count"] = 0
ct.REC["phase"] = "setup"
ct.REC["metadata"] = {
    "e2e": True,
    "task_version": TASK_VERSION,
    "seed": SEED,
    "demo_variants": {"pillar": pillar_variant, "strip": 3},
    "pillar_xy": pillar_xy.tolist(),
    "strip_xy": strip_xy.tolist(),
    "cart_xy": cart_xy.tolist(),
    "robot0": robot0.tolist(),
    "trip_order": ["pillar_to_middle", "strip_to_top"],
    "diversity": {
        "schema_version": 1,
        "mode": DIVERSITY_MODE,
        "scene_randomization": {
            "layout_mode": LAYOUT_MODE,
            "boundary_axis": boundary_axis,
            "cart_offset_xy_m": (cart_xy - CART_NOM).tolist(),
            "rack_y_offset_m": float(rack_y),
            "pillar_offset_xy_m": (pillar_xy - PILLAR_NOM).tolist(),
            "strip_offset_xy_m": (strip_xy - STRIP_NOM).tolist(),
            "robot_initial_xyyaw": robot0.tolist(),
            "light_position_jitter_range_m": [-0.4, 0.4],
            "light_diffuse_scale_range": [0.7, 1.25],
        },
        "perturbation_type": (
            "controlled_empty_navigation_base_pose_shift"
            if DIVERSITY_MODE == "recovery" else "none"
        ),
        "requested_event_count": N_KICKS,
        "actual_event_count": 0,
        "events": PERTURBATION_EVENTS,
    },
    "validation": VALIDATION,
}
if SDK_RECOVERY:
    ct.REC["metadata"].update({
        "collection_profile": COLLECTION_PROFILE,
        "sdk_document_revision": SDK_DOC_REVISION,
        "sdk_timestamp_source": "mujoco_sim_time_synchronous_render",
        "sdk_camera_state_max_skew_s": SDK_CAMERA_STATE_MAX_SKEW_S,
        "sdk_task_head_pose_rad": dict(SDK_TASK_HEAD_POSE_RAD),
    })
if OBJECT_INTERNAL_CONTRACT is not None:
    ct.REC["metadata"]["object_internal_state"] = OBJECT_INTERNAL_CONTRACT
MOTION_LOG = {"state": [], "action": [], "base_action": [], "phase": []}
AUDIT_TICKS = 0
POLICY_LOG_ON = True


def set_phase(value):
    ct.REC["phase"] = value
def close_policy_episode(reason):
    """Stop policy data/audit capture while allowing post-task safety control."""
    global POLICY_LOG_ON
    if not POLICY_LOG_ON:
        return
    POLICY_LOG_ON = False
    ct.REC["on"] = False
    endpoint = {
        "reason": reason,
        "phase": ct.REC.get("phase", "unknown"),
        "recorded_frames": int(rec.n),
        "audit_frames": int(len(MOTION_LOG["action"])),
    }
    ct.REC["metadata"]["policy_episode_end"] = endpoint
    print(
        f"[policy_end] reason={reason} recorded={rec.n} audit={len(MOTION_LOG['action'])}",
        flush=True,
    )



def motion_quality():
    return analyze_motion(
        np.asarray(MOTION_LOG["state"]),
        np.asarray(MOTION_LOG["action"]),
        np.asarray(MOTION_LOG["base_action"]),
        fps=ct.REC_FPS,
        phases=np.asarray(MOTION_LOG["phase"]),
        joint_names=ct.ACTION_NAMES,
        action_delta_limit=ACTION_DELTA_MAX,
        tracking_p95_limit=TRACKING_P95_MAX,
        tracking_max_limit=TRACKING_MAX,
        terminal_tracking_limit=TERMINAL_TRACKING_MAX,
        enforce_tracking=True,
    )


def sdk_alignment_quality():
    """Audit the policy-cadence command/state log before publishing an SDK episode."""
    n = len(MOTION_LOG["action"])
    training_timestamp = (
        (np.arange(n, dtype=np.float64) + 1.0) / ct.REC_FPS
    ).astype(np.float32)
    raw_timestamp = None
    camera_timestamps = None
    if rec.n:
        raw_timestamp = np.asarray(rec.rows["capture_timestamp"], dtype=np.float64)
        camera_timestamps = {
            camera: raw_timestamp.copy() for camera in RECORD_CAMERAS
        }
    return audit_sdk_episode(
        np.asarray(MOTION_LOG["state"]),
        np.asarray(MOTION_LOG["action"]),
        np.asarray(MOTION_LOG["base_action"]),
        fps=ct.REC_FPS,
        joint_names=ct.ACTION_NAMES,
        cameras=RECORD_CAMERAS,
        timestamp=training_timestamp,
        sdk_state_timestamp=raw_timestamp,
        camera_timestamps=camera_timestamps,
        require_camera_timestamps=bool(rec.n),
        enforce_rated_speed=True,
    )


def finish(ok):
    global POLICY_LOG_ON
    diversity = ct.REC["metadata"]["diversity"]
    diversity["actual_event_count"] = len(PERTURBATION_EVENTS)
    quality = motion_quality()
    VALIDATION["motion_quality"] = quality
    sdk_quality = sdk_alignment_quality() if SDK_RECOVERY else None
    if sdk_quality is not None:
        VALIDATION["sdk_alignment"] = sdk_quality
    if ok:
        delta = quality.get("action_delta_rad", {})
        tracking = quality.get("tracking_error_rad", {})
        quality_ok = bool(quality.get("passed"))
        gate(
            "motion_quality",
            quality_ok,
            f"dmax={delta.get('max', float('nan')):.4f}rad "
            f"track_p95={tracking.get('p95', float('nan')):.4f}rad "
            f"track_max={tracking.get('max', float('nan')):.4f}rad "
            f"terminal={tracking.get('terminal_max', float('nan')):.4f}rad "
            f"tracking={'PASS' if quality.get('tracking_passed') else 'FAIL'}",
        )
        ok = bool(ok and quality_ok)
        if sdk_quality is not None:
            sdk_speed = sdk_quality.get("joint_command_speed", {})
            sdk_time = sdk_quality.get("camera_state_timestamp", {})
            sdk_ok = bool(sdk_quality.get("passed"))
            sdk_skew = sdk_time.get("max_skew_s")
            skew_text = "n/a" if sdk_skew is None else f"{sdk_skew:.6f}s"
            gate(
                "sdk_alignment",
                sdk_ok,
                f"vmax={sdk_speed.get('max_rad_s', float('nan')):.4f}rad/s "
                f"skew={skew_text}",
            )
            ok = bool(ok and sdk_ok)
    VALIDATION["passed"] = bool(ok)
    VALIDATION["failed_gates"] = list(FAIL)
    ct.REC["on"] = False
    POLICY_LOG_ON = False
    rec.finalize(success=bool(ok))
    if OBJ_LOG:
        np.savez_compressed(
            os.path.join(OUT, "object_poses.npz"),
            names=np.array(["pillar", "strip"]),
            pose=np.asarray(OBJ_LOG, dtype=np.float32),  # (n, 2*7) pos+quat per object
        )
        if OBJECT_INTERNAL_CONTRACT is not None:
            if len(OBJ_INTERNAL_LOG) != len(OBJ_LOG):
                raise RuntimeError("object root/internal state logs have different frame counts")
            save_internal_state(OUT, OBJ_INTERNAL_LOG)
    sim_s = PHYSICS_STEPS * m.opt.timestep
    print(f"[duration] ticks={TICKS} sim={sim_s:.1f}s", flush=True)
    result = {
        "seed": SEED,
        "passed": bool(ok),
        "first_failed_gate": FAIL[0].split(":")[0] if FAIL else None,
        "failed_gates": list(FAIL),
        "gates": GATES,
        "sim_seconds": round(float(sim_s), 1),
        "peak_grip_force_n": round(PEAK_FORCE["value"], 1),
        "peak_grip_force_where": PEAK_FORCE["where"],
        "objects": VALIDATION["objects"],
        "geometry": {name: shelf_geometry(name) for name in ("pillar", "strip")},
        "motion_quality": quality,
        "sdk_alignment": sdk_quality,
        "collection_profile": COLLECTION_PROFILE,
        "diversity": diversity,
        "policy_episode_end": ct.REC["metadata"].get("policy_episode_end"),
        "safety_home": VALIDATION.get("safety_home"),
        "config": {
            key: os.environ.get(key)
            for key in ("E2E_PLACE_FIX", "E2E_TOUCHDOWN", "E2E_PILLAR_TOUCHDOWN",
                        "E2E_IK_HOP_M", "E2E_DIVERSITY_MODE", "E2E_KICKS",
                        "E2E_PILLAR_VARIANT", "E2E_STRIP_VMAX", "E2E_STRIP_WMAX")
            if os.environ.get(key) is not None
        },
    }
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"=== DUAL EPISODE {'PASS' if ok else 'FAIL'} ===", flush=True)
    if FAIL:
        print("  failed:", "; ".join(FAIL), flush=True)
    sys.exit(0)


def angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


kick_budget = N_KICKS


def maybe_kick(probability, scale=1.0, trigger="stochastic_navigation"):
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
    PERTURBATION_EVENTS.append({
        "trigger": trigger,
        "phase": str(ct.REC.get("phase", "unknown")),
        "recorded_frame": int(rec.n),
        "sim_time_s": round(float(d.time), 6),
        "base_pose_delta": {
            "x_m": float(shift[0]),
            "y_m": float(shift[1]),
            "yaw_rad": float(shift[2]),
        },
    })
    print(f"[kick] remaining={kick_budget} shift=({shift[0]:+.2f},{shift[1]:+.2f},{shift[2]:+.2f})", flush=True)


def set_base_vel(v, w):
    """Ramp the commanded chassis velocity instead of stepping it."""
    if SDK_RECOVERY and not (
        SDK_BASE_V_FWD_RANGE_M_S[0] <= float(v) <= SDK_BASE_V_FWD_RANGE_M_S[1]
        and SDK_BASE_WZ_RANGE_RAD_S[0] <= float(w) <= SDK_BASE_WZ_RANGE_RAD_S[1]
    ):
        raise ValueError(f"SDK base target outside documented range: v={v}, wz={w}")
    dt = CONTROL_DT * DECIM
    ct.base_vel[0] += float(np.clip(v - ct.base_vel[0], -BASE_ACC * dt, BASE_ACC * dt))
    ct.base_vel[1] += float(np.clip(w - ct.base_vel[1], -BASE_YACC * dt, BASE_YACC * dt))


def stop_base():
    for _ in range(60):
        if abs(ct.base_vel[0]) < 1e-3 and abs(ct.base_vel[1]) < 1e-3:
            break
        set_base_vel(0.0, 0.0)
        frames(DECIM)
    ct.base_vel[:] = 0.0
    frames(8)


def brake_cap(remaining, accel):
    """Fastest speed from which `accel` can still stop within `remaining`.

    Without this the loops break while the ramp is still spinning the base down,
    and it coasts past the target -- up to 9.6 deg of parking yaw error, which is
    enough to misalign the grasp and drop the part off the shelf.
    """
    return float(np.sqrt(max(0.0, 2.0 * accel * abs(remaining))))


NAVDBG = int(os.environ.get("E2E_NAVDBG", "0"))


SHORT_REVERSE_MAX = 0.45
SHORT_REVERSE_VMAX = 0.08


def turn_in_place(target_yaw, wmax=0.5, max_frames=1800):
    """Stop, turn to one heading, and stop again before translating."""
    stop_base()
    for _ in range(max_frames):
        error = angle(target_yaw - ct.base_pose()[2])
        if abs(error) < 0.03 and abs(ct.base_vel[1]) < 0.02:
            break
        w_cap = min(wmax, 0.45, brake_cap(error, BASE_YACC))
        set_base_vel(0.0, float(np.clip(2.0 * error, -w_cap, w_cap)))
        frames(DECIM)
    stop_base()


def go_to(tx, ty, tyaw, vmax=0.25, wmax=0.5, kicks=True, kick_scale=1.0, max_frames=5000):
    """Forward-preferred transit: face the path, drive forward, then face the goal."""
    used = max_frames
    x, y, yaw = ct.base_pose()
    if np.hypot(tx - x, ty - y) >= 0.025:
        turn_in_place(np.arctan2(ty - y, tx - x), wmax=wmax)
    for used in range(max_frames):
        x, y, yaw = ct.base_pose()
        distance = float(np.hypot(tx - x, ty - y))
        if distance < 0.025:
            break
        heading = np.arctan2(ty - y, tx - x)
        error = angle(heading - yaw)
        if abs(error) > 0.35:
            turn_in_place(heading, wmax=wmax)
            continue
        w_cap = min(wmax, brake_cap(error, BASE_YACC))
        v_cap = min(vmax, brake_cap(distance - 0.025, BASE_ACC))
        speed = min(float(np.clip(1.2 * distance, 0.04, vmax)), v_cap)
        set_base_vel(speed, float(np.clip(1.8 * error, -w_cap, w_cap)))
        frames(DECIM)
        if kicks:
            maybe_kick(0.004, kick_scale)
    stop_base()
    turn_in_place(tyaw, wmax=wmax)
    if NAVDBG:
        x, y, yaw = ct.base_pose()
        print(f"[nav:transit] tgt=({tx:+.2f},{ty:+.2f},{tyaw:+.2f}) "
              f"got=({x:+.2f},{y:+.2f},{yaw:+.2f}) "
              f"derr={np.hypot(tx-x,ty-y):.3f} yerr={angle(tyaw-yaw):+.3f} it={used+1}"
              f"{' MAXFRAMES' if used + 1 >= max_frames else ''}", flush=True)


def clearance_reverse(distance=0.42, vmax=SHORT_REVERSE_VMAX, max_frames=1200):
    """Back straight out of a rack, with a hard distance and speed limit."""
    distance = float(distance)
    if not 0.0 < distance <= SHORT_REVERSE_MAX:
        raise ValueError(f"clearance reverse must be in (0, {SHORT_REVERSE_MAX}] m")
    vmax = min(float(vmax), SHORT_REVERSE_VMAX)
    stop_base()
    start = ct.base_pose()
    hold_yaw = float(start[2])
    target = start[:2] - distance * np.array([np.cos(hold_yaw), np.sin(hold_yaw)])
    for _ in range(max_frames):
        x, y, yaw = ct.base_pose()
        remaining = float(np.hypot(*(target - np.array([x, y]))))
        if remaining < 0.015:
            break
        reverse_heading = angle(np.arctan2(target[1] - y, target[0] - x) + np.pi)
        error = angle(reverse_heading - yaw)
        v_cap = min(vmax, brake_cap(remaining - 0.015, BASE_ACC))
        set_base_vel(-min(float(np.clip(1.0 * remaining, 0.025, vmax)), v_cap),
                     float(np.clip(1.2 * error, -0.15, 0.15)))
        frames(DECIM)
    stop_base()
    x, y, yaw = ct.base_pose()
    moved = float(np.hypot(x - start[0], y - start[1]))
    print(f"[nav:clearance_reverse] asked={distance:.3f}m moved={moved:.3f}m "
          f"yaw_change={np.degrees(angle(yaw-hold_yaw)):+.1f}deg", flush=True)


def transform_mount(position, rotation, offset, yaw_rotation, pivot):
    c, s = np.cos(yaw_rotation), np.sin(yaw_rotation)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pivot + rz @ (position - pivot) + offset, rz @ rotation


def approach_first(ref, paths, offset, yaw_rotation, pivot, ik_weight):
    """Enter grasp replay without allowing IK to flip arm branches.

    The pillar solve starts from the reference joint pose, corrects only the small
    position error, and then moves there smoothly in joint space. The mirrored strip
    cannot reuse that joint branch, so it retains the legacy Cartesian hops.
    """
    goals = {h: transform_mount(*paths[h][0], offset, yaw_rotation, pivot) for h in ("l", "r")}
    if ref is not REF["pillar"]:
        starts = {h: d.xpos[arm.mount].copy() for h, arm in (("l", ct.L), ("r", ct.R))}
        span = max(float(np.linalg.norm(goals[h][0] - starts[h])) for h in ("l", "r"))
        hops = max(1, int(np.ceil(span / IK_HOP_M)))
        for k in range(1, hops + 1):
            q0, v0 = d.qpos.copy(), d.qvel.copy()
            targets = {}
            for hand, arm in (("l", ct.L), ("r", ct.R)):
                position = starts[hand] + (goals[hand][0] - starts[hand]) * (k / hops)
                for j, adr in enumerate(arm.qadr):
                    d.qpos[adr] = ct.qtgt[hand][j]
                mujoco.mj_fwdPosition(m, d)
                ct.ik(arm, position, goals[hand][1], iters=18, w=ik_weight)
                targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
            d.qpos[:], d.qvel[:] = q0, v0
            mujoco.mj_fwdPosition(m, d)
            command_arm_targets(
                targets["l"], targets["r"], label=f"{TRIP}/approach_hop_{k}"
            )
        return

    seeds = {
        "l": ref["action"][ref["gs"], 0:7],
        "r": ref["action"][ref["gs"], 7:14],
    }
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    targets = {}
    residuals = {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        for j, adr in enumerate(arm.qadr):
            d.qpos[adr] = seeds[hand][j]
        mujoco.mj_fwdPosition(m, d)
        ct.ik(arm, goals[hand][0], goals[hand][1], iters=18, w=0.0)
        targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        residuals[hand] = 1000.0 * float(np.linalg.norm(goals[hand][0] - d.xpos[arm.mount]))
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_fwdPosition(m, d)

    print(
        f"[approach] demo_seed residual_mm=({residuals['l']:.1f},{residuals['r']:.1f}) "
        f"target_jump={max(float(np.abs(targets[h] - ct.qtgt[h]).max()) for h in ('l', 'r')):.3f}rad",
        flush=True,
    )
    command_arm_targets(targets["l"], targets["r"], label=f"{TRIP}/approach")


def warp_frames(ref, f0, f1, max_step, settle=10, grip_settle=GRIP_SETTLE):
    """Which demo frames to replay, spaced evenly in joint-space arc length.

    The demo is an operator's recording: 34% of the grasp segment and 37% of the place
    segment command no joint change at all, spread over 64 separate hesitations, and the
    frames that do move range from 0 to 0.044 rad. Replaying it frame for frame
    reproduces both the dead time and the uneven speed. Emitting a frame once the
    accumulated arc reaches max_step makes the speed uniform and drops the pauses.

    Gripper transitions are never skipped, and closing or opening is followed by repeats
    of the same target so contact can settle before the arm moves on -- that settling is
    the one thing the idle frames were genuinely providing.
    """
    q = ref["action"][f0:f1, :14]
    grip = ref["action"][f0:f1, 14:16]
    step = np.abs(np.diff(q, axis=0)).max(1)
    grip_move = np.abs(np.diff(grip, axis=0)).max(1)
    order = [0]
    arc = 0.0
    for i, ds in enumerate(step):
        arc += float(ds)
        moving_grip = float(grip_move[i]) > 1e-6
        if arc >= max_step or moving_grip:
            order.append(i + 1)
            arc = 0.0
            # Settle only where the grip has just finished moving, so it costs frames at
            # the two contact events rather than throughout. The demo leaves ~195 mostly
            # idle frames after the left gripper closes; collapsing that to a few frames
            # catches the 32 mm strip mid-closure at 2.0 N with one pad off.
            if moving_grip and (i + 1 >= len(grip_move) or float(grip_move[i + 1]) <= 1e-6):
                order.extend([i + 1] * grip_settle)
    if order[-1] != len(q) - 1:
        order.append(len(q) - 1)
    order.extend([len(q) - 1] * settle)
    return order


def arm_replay(ref, phase, offset, ik_weight=0.6, yaw_rotation=0.0, stop_at=None,
               hand_offset=None):
    f0 = ref["gs"] if phase == "grasp" else ref["ps"]
    f1 = ref["ge"] if phase == "grasp" else ref["place_end"]
    if stop_at is not None:
        f1 = max(f0 + 1, min(f1, stop_at))
    warping = MAX_STEP > 0 and WARP_SCOPE in ("all", TRIP)
    order = warp_frames(ref, f0, f1, MAX_STEP) if warping else list(range(f1 - f0))
    paths = ref["mount"][phase]
    pivot = ref["ref_center"]
    max_residuals = {"l": 0.0, "r": 0.0}
    release_residuals = {"pre": {"l": 0.0, "r": 0.0}, "post": {"l": 0.0, "r": 0.0}}
    # Only on the grasp segment: that is where the arms start from home and have to
    # cover a long way. At place time they are already near the demo's first pose.
    if IK_HOP_M > 0 and phase == "grasp":
        approach_first(ref, paths, offset, yaw_rotation, pivot, ik_weight)
    for idx in order:
        frame = f0 + idx
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets = {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            pm, rm = paths[hand][idx]
            hand_shift = np.zeros(3) if hand_offset is None else hand_offset[hand]
            target_pm, target_rm = transform_mount(
                pm, rm, offset + hand_shift, yaw_rotation, pivot)
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = ct.qtgt[hand][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(arm, target_pm, target_rm, iters=18, w=ik_weight)
            residual = float(np.linalg.norm(target_pm - d.xpos[arm.mount]))
            max_residuals[hand] = max(max_residuals[hand], residual)
            if phase == "place":
                period = "pre" if frame < ref["release"] else "post"
                release_residuals[period][hand] = max(
                    release_residuals[period][hand], residual)
            targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        grip_l = float((1 - ref["action"][frame, 14]) * 0.025)
        grip_r = float((1 - ref["action"][frame, 15]) * 0.025)
        command_arm_targets(
            targets["l"],
            targets["r"],
            grip_l=grip_l,
            grip_r=grip_r,
            label=f"{TRIP}/{phase}/frame_{frame}",
        )
    if warping:
        saved_s = (f1 - f0 - len(order)) * DECIM * CONTROL_DT
        print(f"[warp:{TRIP}/{phase}] frames {f1 - f0} -> {len(order)} "
              f"(saved {saved_s:.1f}s)", flush=True)
    if phase == "place":
        print(
            f"[place_ik] max_residual_mm=({1000*max_residuals['l']:.1f},"
            f"{1000*max_residuals['r']:.1f}) "
            f"pre=({1000*release_residuals['pre']['l']:.1f},{1000*release_residuals['pre']['r']:.1f}) "
            f"post=({1000*release_residuals['post']['l']:.1f},{1000*release_residuals['post']['r']:.1f})mm",
            flush=True,
        )


HOME_L = ct.qtgt["l"].copy()
HOME_R = ct.qtgt["r"].copy()


def tuck(ref):
    frame = ref["ge"]
    goal_l = ref["action"][frame, 0:7].copy()
    goal_r = ref["action"][frame, 7:14].copy()
    start_l, start_r = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
    need = max(float(np.abs(goal_l - start_l).max()), float(np.abs(goal_r - start_r).max()))
    steps = cosine_steps(need, COMMAND_DELTA_MAX, minimum=60)
    for i in range(steps):
        blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / steps)
        ct.qtgt["l"][:] = start_l + (goal_l - start_l) * blend
        ct.qtgt["r"][:] = start_r + (goal_r - start_r) * blend
        ct.grip_cmd["l"] = ct.GRIP_CLOSE
        ct.grip_cmd["r"] = ct.GRIP_CLOSE
        frames(DECIM)


def lift_joint_targets(delta):
    """Raise both mounts along the same segmented Cartesian path."""
    delta = np.asarray(delta, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return
    steps = int(np.ceil(distance / 0.04))
    step_delta = delta / steps
    print(f"[top_lift] cartesian_steps={steps} step={np.round(step_delta, 3).tolist()}", flush=True)
    for _ in range(steps):
        move_hands(step_delta)


def shelf_target(name):
    """Where the part should end up: centred on its shelf, resting on the surface."""
    slo, shi = ct.geom_aabb(SHELF[name])
    lo, hi = obj_extent(name)
    return np.array([
        0.5 * (slo[0] + shi[0]),
        0.5 * (slo[1] + shi[1]),
        shi[2] + 0.5 * (hi[2] - lo[2]) + 0.004,
    ])


def shelf_free_x_band(name):
    """The widest x interval on the shelf with nothing standing proud of its surface.

    The cart's corner posts top out 8 mm above the top shelf, and the bar is 1595 mm against
    a 1580 mm clear span between the near pair, so it always overlaps a post in y and can
    only avoid resting on one by staying clear in x. Discovered from the model rather than
    by naming the posts, so it still holds if the cart changes.
    """
    slo, shi = ct.geom_aabb(SHELF[name])
    blocked = []
    for g in range(m.ngeom):
        # Only the shelf's own structure, and only geometry that can actually be hit. The
        # cart's visual mesh spans the whole shelf and tops out above it, so counting it
        # leaves no free band at all; the arms hovering over the shelf do the same.
        if g == SHELF[name] or m.geom_bodyid[g] != m.geom_bodyid[SHELF[name]]:
            continue
        if m.geom_contype[g] == 0:
            continue
        glo, ghi = ct.geom_aabb(g)
        proud = ghi[2] > shi[2] + 0.001 and glo[2] < shi[2] + 0.10
        overlaps = (ghi[0] > slo[0] and glo[0] < shi[0]
                    and ghi[1] > slo[1] and glo[1] < shi[1])
        if proud and overlaps:
            blocked.append((float(glo[0]), float(ghi[0])))
    lo, hi = float(slo[0]), float(shi[0])
    free = [(lo, hi)]
    for blo, bhi in blocked:
        nxt = []
        for flo, fhi in free:
            if bhi <= flo or blo >= fhi:
                nxt.append((flo, fhi))
                continue
            if blo > flo:
                nxt.append((flo, blo))
            if bhi < fhi:
                nxt.append((bhi, fhi))
        free = nxt
    if not free:
        return lo, hi
    return max(free, key=lambda ab: ab[1] - ab[0])


def park_for_carried(name, yaw, demo_xy, margin=0.030):
    """Base xy that gets the carried part onto its shelf, moving as little as possible.

    The strip never replays a place pose -- the arms hold the carry pose and the chassis
    does the positioning -- so park_place was derived from where the demo's *release* frame
    would have left the part. The arms are never in that configuration, so the prediction is
    off by 416 mm in x. Measuring the part in the base frame instead is exact for any carry
    pose, and stays valid through the remaining navigation because the arms no longer move.

    Aiming at the shelf centre is the wrong target though. The bar is 32 mm across a 800 mm
    shelf, so it has +/-384 mm of freedom in x and only +/-77 mm in y, and the demo already
    lands inside the y window. Centring in x means driving 207 mm further in, which wedges
    the bar against the cart's upper structure: the left hand reads 99 N, and touchdown then
    commands 196 mm of descent while the part moves 14 mm and its centre rises 176 mm. So
    shift only as far as containment needs, with a margin over the gate's 2 mm tolerance.
    """
    base = ct.base_pose()
    rel = rz(-base[2]) @ (obj_pos(name) - np.array([base[0], base[1], 0.0]))
    lo, hi = obj_extent(name)
    half = 0.5 * (hi - lo)
    slo, shi = ct.geom_aabb(SHELF[name])
    demo_obj = np.asarray(demo_xy) + (rz(yaw) @ rel)[:2]
    want = demo_obj.copy()
    for k in (0, 1):
        low = slo[k] + half[k] + margin
        high = shi[k] - half[k] - margin
        if low > high:  # shelf too small on this axis; centre it and let the gate judge
            want[k] = 0.5 * (slo[k] + shi[k])
        else:
            want[k] = float(np.clip(demo_obj[k], low, high))
    if POST_CLEAR > 0.0:
        # Keep the ends off the posts that stand proud of the surface. The demo lands 71 mm
        # from the near pair, which is only ~5 deg of yaw before an end swings on top of
        # one; the free band's centre gives ~290 mm. Move no further than asked, because
        # driving all the way to the centre wedges the bar and touchdown then fails.
        blo, bhi = shelf_free_x_band(name)
        target = float(np.clip(want[0], blo + half[0] + POST_CLEAR, bhi - half[0] - POST_CLEAR)
                       if blo + half[0] + POST_CLEAR <= bhi - half[0] - POST_CLEAR
                       else 0.5 * (blo + bhi))
        want[0] = target
        print(f"[post_clear:{name}] free_x_band=[{blo:.3f},{bhi:.3f}] "
              f"aim_x={want[0]:.3f} shifted={1000*(want[0] - demo_obj[0]):+.0f}mm", flush=True)
    if name == "strip" and cart_xy[0] < CART_NOM[0] - 0.16:
        shelf_centre_x = 0.5 * (slo[0] + shi[0])
        inward = float(np.sign(shelf_centre_x - want[0]))
        low_x = slo[0] + half[0] + margin
        high_x = shi[0] - half[0] - margin
        want[0] = float(np.clip(want[0] + 0.030 * inward, low_x, high_x))
        print(f"[park_inset:{name}] left_workspace_edge toward_centre=30mm "
              f"aim_x={want[0]:.3f}", flush=True)
    xy = want - (rz(yaw) @ rel)[:2]
    print(f"[park_fix:{name}] carried_rel_base={np.round(rel, 3).tolist()} "
          f"demo_would_land={np.round(demo_obj, 3).tolist()} "
          f"aim={np.round(want, 3).tolist()} "
          f"moved={np.round(xy - np.asarray(demo_xy), 3).tolist()}", flush=True)
    return xy


def cosine_steps(need, max_step, minimum=1):
    """Number of cosine-ramp samples whose largest increment is <= max_step."""
    if max_step <= 0.0:
        raise ValueError("joint step limit must be positive")
    return max(int(minimum), int(np.ceil(np.pi * float(need) / (2.0 * max_step))))


def servo_to(target_l, target_r, min_steps=None, grip_l=None, grip_r=None, label=None):
    """Ramp joint targets to a goal at a bounded rate.

    The cosine ramp has zero endpoint velocity. Its sample count includes the pi/2
    peak-slope factor, so QDOT_MAX is an actual per-control-tick bound rather than
    an average-rate estimate.
    """
    min_steps = DECIM if min_steps is None else min_steps
    target_l = np.clip(np.asarray(target_l, dtype=float), ct.L.lo, ct.L.hi)
    target_r = np.clip(np.asarray(target_r, dtype=float), ct.R.lo, ct.R.hi)
    need = max(float(np.abs(target_l - ct.qtgt["l"]).max()),
               float(np.abs(target_r - ct.qtgt["r"]).max()))
    n = cosine_steps(need, QDOT_MAX, minimum=min_steps)
    n = int(np.ceil(n / DECIM) * DECIM)
    start_l, start_r = ct.qtgt["l"].copy(), ct.qtgt["r"].copy()
    if label and need > COMMAND_DELTA_MAX:
        print(f"[joint_smooth:{label}] jump={need:.4f}rad ticks={n}", flush=True)
    for i in range(n):
        blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)
        ct.qtgt["l"][:] = start_l + (target_l - start_l) * blend
        ct.qtgt["r"][:] = start_r + (target_r - start_r) * blend
        if grip_l is not None:
            ct.grip_cmd["l"] = grip_l
        if grip_r is not None:
            ct.grip_cmd["r"] = grip_r
        ct.base_vel[:] = 0.0
        frames(1)


def command_arm_targets(target_l, target_r, grip_l=None, grip_r=None, label="motion"):
    """Reach one IK target without exposing a branch jump to the dataset."""
    target_l = np.clip(np.asarray(target_l, dtype=float), ct.L.lo, ct.L.hi)
    target_r = np.clip(np.asarray(target_r, dtype=float), ct.R.lo, ct.R.hi)
    need = max(float(np.abs(target_l - ct.qtgt["l"]).max()),
               float(np.abs(target_r - ct.qtgt["r"]).max()))
    if need > COMMAND_DELTA_MAX:
        servo_to(target_l, target_r, grip_l=grip_l, grip_r=grip_r,
                 label=label if GRASP_DUMP else None)
        return
    ct.qtgt["l"][:] = target_l
    ct.qtgt["r"][:] = target_r
    if grip_l is not None:
        ct.grip_cmd["l"] = grip_l
    if grip_r is not None:
        ct.grip_cmd["r"] = grip_r
    ct.base_vel[:] = 0.0
    frames(DECIM)


def measured_arm_targets():
    return (
        np.array([d.qpos[adr] for adr in ct.L.qadr], dtype=float),
        np.array([d.qpos[adr] for adr in ct.R.qadr], dtype=float),
    )


def arm_tracking_error(hands=("l", "r")):
    errors = []
    for hand in hands:
        arm = ct.L if hand == "l" else ct.R
        measured = np.array([d.qpos[adr] for adr in arm.qadr], dtype=float)
        errors.append(float(np.abs(measured - ct.qtgt[hand]).max()))
    return max(errors, default=0.0)


def wait_for_arm_tracking(label, hands=("l", "r"), tol=None):
    tolerance = TRACK_SETTLE_TOL if tol is None else float(tol)
    stable = 0
    error = arm_tracking_error(hands)
    for tick in range(TRACK_SETTLE_TICKS):
        error = arm_tracking_error(hands)
        stable = stable + 1 if error <= tolerance else 0
        if stable >= DECIM:
            print(f"[track:{label}] settled err={error:.4f}rad tol={tolerance:.4f} "
                  f"ticks={tick + 1}", flush=True)
            return True
        frames(1)
    print(f"[track:{label}] TIMEOUT err={error:.4f}rad tol={tolerance:.4f} "
          f"ticks={TRACK_SETTLE_TICKS}", flush=True)
    return False


def sync_command_to_state(label):
    """Unload position-servo error after release without stepping the command."""
    measured_l, measured_r = measured_arm_targets()
    servo_to(
        measured_l,
        measured_r,
        grip_l=ct.GRIP_OPEN,
        grip_r=ct.GRIP_OPEN,
        label=f"{label}/sync",
    )
    return wait_for_arm_tracking(f"{label}/sync")


def move_hands(delta, allow_reorient=True, hands=("l", "r")):
    """Shift selected mounts by a world-frame delta, holding orientation and grip.

    allow_reorient lets IK trade wrist orientation for reaching the position when the full pose
    is out of range. That is what touchdown wants, a few millimetres at a time. It is wrong for
    a longer sideways move: giving up orientation there rotates the carried part and lifted the
    bar 68 mm, out of the arms' reach, so the descent that followed never moved it.
    """
    delta = np.asarray(delta, float)
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    targets = {}
    errors = {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        seed = ct.qtgt[hand].copy()
        if hand not in hands:
            targets[hand] = seed
            errors[hand] = 0.0
            continue
        for j, adr in enumerate(arm.qadr):
            d.qpos[adr] = seed[j]
        mujoco.mj_fwdPosition(m, d)
        target_position = d.xpos[arm.mount].copy() + delta
        target_rotation = d.xmat[arm.mount].reshape(3, 3).copy()
        ct.ik(arm, target_position, target_rotation, iters=45, w=0.35)
        best_q = np.array([d.qpos[adr] for adr in arm.qadr])
        best_error = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
        if best_error > 0.010 and allow_reorient:
            full_error = best_error
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = seed[j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(arm, target_position, target_rotation, iters=45, w=0.0)
            position_q = np.array([d.qpos[adr] for adr in arm.qadr])
            position_error = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
            if position_error < best_error:
                best_q = position_q
                best_error = position_error
            print(
                f"[move_hands:{hand}] full_err={1000*full_error:.1f}mm "
                f"chosen_err={1000*best_error:.1f}mm",
                flush=True,
            )
        targets[hand] = best_q
        errors[hand] = best_error
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_fwdPosition(m, d)
    if max(errors.values()) > 0.010:
        print(
            f"[move_hands] rejected residual_mm=({1000*errors['l']:.1f},{1000*errors['r']:.1f})",
            flush=True,
        )
        return False
    servo_to(targets["l"], targets["r"],
             min_steps=max(DECIM, int(np.ceil(float(np.linalg.norm(delta)) / 0.002))))

    return True


def inhand_correction(name, ref):
    """How far the part sits from where the demo's grasp would have put it.

    Both offsets are taken in the base frame so the comparison is independent of how
    far the chassis has turned. The result is rotated back into world coordinates.
    """
    yaw = ct.base_pose()[2]
    mid = 0.5 * (d.xpos[ct.L.mount] + d.xpos[ct.R.mount])
    actual = rz(-yaw) @ (obj_pos(name) - mid)
    delta = ref["inhand_base"] - actual
    # ref_center's height is a hardcoded nominal, not a measured grasp height, so the
    # vertical term is a modelling offset rather than a grasp error. Correcting on it
    # drives the hands ~9 cm low and the part misses the shelf. Height is handled by
    # place_offset[2] and, properly, by touchdown detection.
    delta[2] = 0.0
    corr = rz(yaw) @ delta
    norm = float(np.linalg.norm(corr))
    if norm > PLACE_CORR_MAX:
        corr = corr * (PLACE_CORR_MAX / norm)
    print(f"[inhand:{name}] demo_base={np.round(ref['inhand_base'], 3).tolist()} "
          f"actual_base={np.round(actual, 3).tolist()} corr={np.round(corr, 3).tolist()} "
          f"|corr|={norm:.3f}m{' CLAMPED' if norm > PLACE_CORR_MAX else ''}", flush=True)
    return corr


def clear_posts(name, margin=0.030, limit=0.100, minimum=0.049):
    """Slide the part sideways with the arms until it clears the shelf's proud posts.

    The cart's four corner posts stand 8 mm above the top shelf. The bar is 1595 mm against a
    1580 mm clear span between them in y, so it always overlaps a post in y and can only miss
    them by staying inside the 580 mm free band in x -- easy for something 32 mm wide, and the
    demo lands only 46-54 mm outside it. Until it does, a levelled bar rests on a post 8 mm
    above the surface and no downward force will seat it: touchdown spends 120 mm of descent
    while cart_post_n2_1 takes 42 N and the pads are pressed to 88 N.

    Done with the arms, not the chassis. Driving the base those 50 mm is what wedges the bar:
    it takes the pass rate from 7/13 to 1/13 and makes levelling diverge from -3.3 to -5.2 deg.
    """
    low, high = shelf_free_x_band(name)
    lo, hi = obj_extent(name)
    want_low, want_high = low + margin, high - margin
    if float(hi[0] - lo[0]) > want_high - want_low:
        print(f"[clear_posts:{name}] no band wide enough for a "
              f"{1000*(hi[0]-lo[0]):.0f}mm footprint", flush=True)
        return False
    shift = 0.0
    if lo[0] < want_low:
        shift = want_low - float(lo[0])
    elif hi[0] > want_high:
        shift = want_high - float(hi[0])
    if abs(shift) < 0.001:
        print(f"[clear_posts:{name}] already clear of the posts", flush=True)
        return True
    if abs(shift) < minimum:
        shift = float(np.copysign(minimum, shift))
    shift = float(np.clip(shift, -limit, limit))
    steps = max(1, int(np.ceil(abs(shift) / 0.005)))
    moved = 0.0
    for _ in range(steps):
        if not move_hands([shift / steps, 0.0, 0.0], allow_reorient=False):
            break
        moved += shift / steps
    print(f"[clear_posts:{name}] free_x_band=[{low:.3f},{high:.3f}] "
          f"wanted={1000*shift:+.0f}mm moved={1000*moved:+.0f}mm", flush=True)
    return abs(moved - shift) < 0.005


def part_long_axis(name):
    """Unit vector along the carried part, from its own section centres."""
    centres = np.array([0.5 * (ct.geom_aabb(g)[0] + ct.geom_aabb(g)[1])
                        for g in OBJECTS[name]["geoms"]])
    _, _, basis = np.linalg.svd(centres - centres.mean(axis=0))
    axis = basis[0]
    return axis if axis[1] >= 0.0 else -axis


def move_hands_rigid(rotation, pivot, tol=0.010):
    """Rotate both mounts together about a world pivot, holding the part rigidly between them.

    A part clamped in two grippers has no freedom left: the hands cannot be levelled relative
    to it, so the only way to change how it sits is to carry both hands through the same rigid
    motion. Anything less rotates the part against the jaws.
    """
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    targets, errors = {}, {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        seed = ct.qtgt[hand].copy()
        for j, adr in enumerate(arm.qadr):
            d.qpos[adr] = seed[j]
        mujoco.mj_fwdPosition(m, d)
        target_position = pivot + rotation @ (d.xpos[arm.mount] - pivot)
        target_rotation = rotation @ d.xmat[arm.mount].reshape(3, 3)
        ct.ik(arm, target_position, target_rotation, iters=45, w=0.6)
        targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        errors[hand] = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_fwdPosition(m, d)
    if max(errors.values()) > tol:
        print(f"[level] rejected residual_mm="
              f"({1000*errors['l']:.1f},{1000*errors['r']:.1f})", flush=True)
        return False
    servo_to(targets["l"], targets["r"])
    return True


def level_part(name, tol_deg=1.0, step_deg=1.0):
    """Bring the carried part's long axis horizontal before lowering it.

    The bar arrives tilted 4.2 deg along its length -- the two hands sit 25 mm apart in height
    -- so touchdown puts one end on the shelf while the section between the pads is still 94 mm
    up. Seating it from there is what entangles the jaws with the arch. Level it first and both
    ends come down together, which is also the only pose where the arch's own 50 mm of
    clearance sits under the pads, giving them somewhere to slide out to.

    Applied in small increments about the part's centre, because the two arms have to follow one
    rigid motion and a single large rotation is beyond what the IK will accept.
    """
    for i in range(12):
        axis = part_long_axis(name)
        tilt = float(np.arcsin(np.clip(axis[2], -1.0, 1.0)))
        if i == 0:
            print(f"[level:{name}] arrives at {np.rad2deg(tilt):+.2f}deg", flush=True)
        if abs(tilt) <= np.deg2rad(tol_deg):
            print(f"[level:{name}] level at {np.rad2deg(tilt):+.2f}deg", flush=True)
            return True
        # cross(axis, horizontal) already selects the correction direction;
        # a signed angle applies that sign twice and increases negative tilt.
        take = min(abs(tilt), float(np.deg2rad(step_deg)))
        horizontal = np.array([axis[0], axis[1], 0.0])
        norm = float(np.linalg.norm(horizontal))
        if norm < 1e-6:
            return False
        turn = np.cross(axis, horizontal / norm)
        turn /= float(np.linalg.norm(turn))
        c, s = np.cos(take), np.sin(take)
        cross = np.array([[0.0, -turn[2], turn[1]],
                          [turn[2], 0.0, -turn[0]],
                          [-turn[1], turn[0], 0.0]])
        rotation = np.eye(3) + s * cross + (1.0 - c) * (cross @ cross)
        if not move_hands_rigid(rotation, obj_pos(name)):
            print(f"[level:{name}] stopped at {np.rad2deg(tilt):+.2f}deg", flush=True)
            return False
    print(f"[level:{name}] still {np.rad2deg(tilt):+.2f}deg after 12 steps", flush=True)
    return False


def seat_down(name, step=0.004, budget=0.090):
    """Keep lowering, jaws open, until the shelf carries the part's whole weight.

    Touchdown stops when the part's lowest point meets the surface, which for the arched
    rubber bar is one end: the section between the pads is then still 94 mm up, resting on
    the lower pads, and letting go is a 94 mm drop. That is the floating seen on video, and
    the reason the gate reports gap = 0 while the bar is not seated -- the gate measures the
    whole-body minimum, i.e. that one touching end.

    The bar is a single rigid body with 53 mm of arch, so it cannot be pressed flat, and with
    the jaws shut it cannot rotate either; pressing harder only drives the touching end in.
    Opening the jaws first lets it pivot, and the lower pads then lower it the rest of the way
    onto both ends. Full weight on the shelf is the signal that the pads carry nothing, and it
    needs no tuning: the bar weighs 3.93 N and every seed that passes today reads 3.9 N, while
    every seed that fails reads 2.0-2.6 N with a cart post taking the rest.
    """
    want = 0.9 * OBJECTS[name]["weight_n"]
    if name == "strip":
        # Keep the right mount above the bar while the left jaw clears from underneath.
        ct.grip_cmd["l"] = ct.GRIP_OPEN
        ct.grip_cmd["r"] = ct.GRIP_CLOSE
        seat_hands = ("r",)
    else:
        ct.grip_cmd["l"] = ct.GRIP_OPEN
        ct.grip_cmd["r"] = ct.GRIP_OPEN
        seat_hands = ("l", "r")
    frames(40)
    lowered = 0.0
    left_cleared = 0.0
    left_preclear_direction = None
    seat_anchor_xy = obj_pos(name)[:2].copy()
    while lowered < budget - 1e-9:
        if name == "strip":
            support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
            while (grip_force("l", name) > 0.2
                   or (left_cleared < 0.015 and support < want)):
                clear_step = 0.005
                if left_cleared + clear_step > 0.030 + 1e-9:
                    print(f"[seat:{name}] left preclear exhausted "
                          f"force={grip_force('l', name):.2f}N", flush=True)
                    return False
                weighted = np.zeros(3)
                total_weight = 0.0
                pads = set(PADG["l"])
                for i in range(d.ncon):
                    contact = d.contact[i]
                    pad = None
                    if contact.geom1 in pads and contact.geom2 in OBJECTS[name]["geoms"]:
                        pad = contact.geom1
                    elif contact.geom2 in pads and contact.geom1 in OBJECTS[name]["geoms"]:
                        pad = contact.geom2
                    if pad is None:
                        continue
                    normal = np.asarray(contact.frame[:3], dtype=float).copy()
                    if float(np.dot(normal, d.geom_xpos[pad] - contact.pos)) < 0.0:
                        normal *= -1.0
                    mujoco.mj_contactForce(m, d, i, FT)
                    weight = max(abs(float(FT[0])), 1e-6)
                    weighted += weight * normal
                    total_weight += weight
                if total_weight:
                    coherence = float(np.linalg.norm(weighted) / total_weight)
                    if coherence < 0.25:
                        print(f"[seat:{name}] left preclear normal incoherent "
                              f"coherence={coherence:.3f}", flush=True)
                        return False
                    horizontal = weighted.copy()
                    horizontal[2] = 0.0
                    horizontal_norm = float(np.linalg.norm(horizontal))
                    if horizontal_norm < 0.25 * total_weight:
                        print(f"[seat:{name}] left preclear horizontal normal too small",
                              flush=True)
                        return False
                    left_preclear_direction = horizontal / horizontal_norm
                elif left_preclear_direction is None:
                    print(f"[seat:{name}] left preclear has no measured direction",
                          flush=True)
                    return False
                direction = left_preclear_direction
                if not move_hands(clear_step * direction, hands=("l",)):
                    print(f"[seat:{name}] left preclear IK rejected", flush=True)
                    return False
                left_cleared += clear_step
                support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
                shift = float(np.linalg.norm(obj_pos(name)[:2] - seat_anchor_xy))
                print(f"[seat:{name}] left_preclear={1000*left_cleared:.0f}mm "
                      f"axis={np.round(direction, 2).tolist()} "
                      f"force={grip_force('l', name):.2f}N "
                      f"support={support:.2f}N "
                      f"object_xy_shift={1000*shift:.1f}mm", flush=True)
                if shift > 0.015:
                    print(f"[seat:{name}] left preclear object shift exceeded", flush=True)
                    return False
        support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
        if support >= want:
            print(f"[seat:{name}] seated after {1000*lowered:.0f}mm "
                  f"support={support:.2f}N of {OBJECTS[name]['weight_n']:.2f}N "
                  f"centre_z={obj_pos(name)[2]:.3f}", flush=True)
            return True
        if not move_hands([0.0, 0.0, -step], hands=seat_hands):
            print(f"[seat:{name}] IK rejected at {1000*lowered:.0f}mm "
                  f"support={support:.2f}N", flush=True)
            return False
        lowered += step
    support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
    print(f"[seat:{name}] not seated in {1000*budget:.0f}mm "
          f"support={support:.2f}N of {OBJECTS[name]['weight_n']:.2f}N "
          f"centre_z={obj_pos(name)[2]:.3f}", flush=True)
    return False


def touch_down(name, step=0.004, max_steps=None, contact_gap=0.004, seat=False):
    """Lower the part until the shelf carries it, then let go.

    Releasing while the part is still in the air lets it drop and skate: the strip was
    let go ~12 cm above the top shelf and slid 0.46 m, ending up 0.42 m off centre with
    34 cm hanging over the edge.

    The descent budget is sized from the measured gap rather than fixed, because the
    pillar arrives 136-174 mm up and a flat 40 x 4 mm budget stopped 0-2 mm short of the
    surface and reported failure. The support test alone is also not enough: while the
    pads still hold the part at a commanded height the shelf carries no load, so contact
    has to be recognised geometrically as well.
    """
    _, shi = ct.geom_aabb(SHELF[name])
    max_center_z = shi[2] + 0.070
    if max_steps is None:
        lo, _ = obj_extent(name)
        max_steps = int(np.ceil((max(0.0, float(lo[2] - shi[2])) + 0.040) / step))
    for i in range(max_steps):
        support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
        centre_z = float(obj_pos(name)[2])
        # The whole-body minimum, deliberately: for the arched bar this is whichever end is
        # lowest, and stopping when that end meets the surface is what lets it settle flat.
        # Using a median section instead never registers contact and fails all 13 seeds.
        lo, _ = obj_extent(name)
        gap = float(lo[2] - shi[2])
        if (support >= 1.0 and centre_z <= max_center_z) or gap <= contact_gap:
            print(
                f"[touchdown:{name}] contact after {i} steps ({1000*i*step:.0f}mm) "
                f"support={support:.1f}N centre_z={centre_z:.3f} gap={1000*gap:+.0f}mm",
                flush=True,
            )
            for hand, arm in (("l", ct.L), ("r", ct.R)):
                ct.qtgt[hand][:] = np.array([d.qpos[adr] for adr in arm.qadr])
                ct.grip_cmd[hand] = ct.GRIP_CLOSE
            frames(20)
            support = pair_force(OBJECTS[name]["geoms"], {SHELF[name]})
            print(
                f"[touchdown:{name}] unloaded support={support:.1f}N "
                f"centre_z={obj_pos(name)[2]:.3f}",
                flush=True,
            )
            if (seat or SEAT) and not seat_down(name, step=step):
                return False
            return True
        if not move_hands([0.0, 0.0, -step]):
            print(f"[touchdown:{name}] IK rejected at step {i}", flush=True)
            return False
    print(f"[touchdown:{name}] no contact in {1000*max_steps*step:.0f}mm, "
          f"gap={1000*(obj_extent(name)[0][2] - shi[2]):+.0f}mm "
          f"centre_z={obj_pos(name)[2]:.3f}",
          flush=True)
    if GRASP_DUMP:
        loads = {}
        for i in range(d.ncon):
            pair = {d.contact[i].geom1, d.contact[i].geom2}
            if not pair & OBJECTS[name]["geoms"]:
                continue
            mujoco.mj_contactForce(m, d, i, FT)
            for other in pair - OBJECTS[name]["geoms"]:
                loads[gname(other)] = loads.get(gname(other), 0.0) + abs(float(FT[0]))
        print(f"[touchdown:{name}] stuck against "
              f"{ {k: round(v, 2) for k, v in loads.items()} or 'nothing (arms not descending)'}",
              flush=True)
    return False


def escape_dir(name, hand):
    """Horizontal direction that takes a hand off the part rather than through it.

    Measured from the section the hand is actually next to, not from the part's centre.
    For the bar those differ completely: the hands sit ~170 mm either side of its middle,
    so a direction taken from the middle points along the bar and slides the pads down it
    instead of off it. That is what pushed a correctly released bar 160 mm back off the
    shelf and then loaded the pads to 377 N on the way home.
    """
    pads = np.mean([d.geom_xpos[g] for g in PADG[hand]], axis=0)
    glo, ghi = nearest_section(name, float(pads[1]))
    away = pads[:2] - 0.5 * (glo[:2] + ghi[:2])
    norm = float(np.linalg.norm(away))
    if norm < 1e-3:
        # Degenerate: pads centred on the section. Back off along the chassis heading.
        yaw = ct.base_pose()[2]
        return np.array([-np.cos(yaw), -np.sin(yaw)])
    return away / norm


def move_home(steps=55, order="both", wait=True, stop_when=None):
    """Return the arms home together or one arm at a time."""
    homes = {"l": HOME_L, "r": HOME_R}

    def segment(targets):
        label = "".join(targets)
        need = max(
            float(np.abs(target - measured_arm_targets()[0 if hand == "l" else 1]).max())
            for hand, target in targets.items()
        )
        max_ticks = max(TRACK_SETTLE_TICKS * 5, steps * 8) if wait else max(1, steps)
        print(f"[move_home:{label}] jump={need:.3f}rad "
              f"feedback_lead={HOME_FEEDBACK_LEAD:.4f}rad "
              f"command_step={HOME_COMMAND_STEP:.4f}rad", flush=True)
        stable = 0
        remaining = need
        for tick in range(max_ticks):
            remaining = 0.0
            measured = measured_arm_targets()
            for hand, target in targets.items():
                index = 0 if hand == "l" else 1
                delta = target - measured[index]
                remaining = max(remaining, float(np.abs(delta).max()))
                leader = measured[index] + np.clip(
                    delta, -HOME_FEEDBACK_LEAD, HOME_FEEDBACK_LEAD
                )
                ct.qtgt[hand] += np.clip(
                    leader - ct.qtgt[hand],
                    -HOME_COMMAND_STEP,
                    HOME_COMMAND_STEP,
                )
            ct.grip_cmd["l"] = ct.GRIP_OPEN
            ct.grip_cmd["r"] = ct.GRIP_OPEN
            stable = stable + 1 if remaining <= TRACK_SETTLE_TOL else 0
            frames(1)
            if stop_when is not None and stop_when(label):
                print(f"[move_home:{label}] clearance goal reached ticks={tick + 1}",
                      flush=True)
                return True
            if stable >= DECIM:
                print(f"[move_home:{label}] reached home ticks={tick + 1}", flush=True)
                return True
        if not wait:
            print(f"[move_home:{label}] clearance ticks={max_ticks} "
                  f"remaining={remaining:.4f}rad", flush=True)
            return True
        print(f"[move_home:{label}] feedback TIMEOUT remaining={remaining:.4f}rad",
              flush=True)
        return False

    print(f"[move_home] order={order}", flush=True)
    tracked = True
    if order == "both":
        tracked = segment(homes)
    elif order in ("lr", "rl"):
        for hand in order:
            tracked = segment({hand: homes[hand]}) and tracked
            if GRASP_DUMP:
                print(f"[move_home:{hand}] strip={np.round(obj_pos('strip'), 3).tolist()} "
                      f"fR={grip_force('r', 'strip'):.1f}N "
                      f"fL={grip_force('l', 'strip'):.1f}N", flush=True)
    else:
        raise ValueError(f"unsupported home order: {order}")
    frames(25)
    return tracked


def release_hands(name, quiet=0.2, step=0.005, max_escape=0.030,
                  max_object_shift=0.015, quiet_s=0.5, tol=0.010):
    """Withdraw contacting open grippers along the current pad-contact normal.

    The contact normal is recomputed after every small move. Recontact during the
    quiet hold restarts withdrawal; object displacement and travel remain bounded.
    """
    anchor_xy = obj_pos(name)[:2].copy()
    distances = {"l": 0.0, "r": 0.0}
    object_geoms = set(OBJECTS[name]["geoms"])
    quiet_required = max(1, int(np.ceil(quiet_s / CONTROL_DT)))
    quiet_streak = 0

    last_directions = {"l": None, "r": None}
    while True:
        active = {hand for hand in ("l", "r") if grip_force(hand, name) > quiet}
        object_shift = float(np.linalg.norm(obj_pos(name)[:2] - anchor_xy))
        if object_shift > max_object_shift:
            print(f"[release:{name}] ABORT object_xy_shift={1000*object_shift:.1f}mm "
                  f"limit={1000*max_object_shift:.0f}mm",
                  flush=True)
            return False
        left_escape_exhausted = (
            name == "strip" and active == {"l"}
            and distances["l"] + step > max_escape + 1e-9
        )
        if active and (quiet_streak or left_escape_exhausted):
            if quiet_streak:
                print(f"[release:{name}] recontact after "
                      f"{quiet_streak * CONTROL_DT:.2f}s", flush=True)
            else:
                print(f"[release:{name}] left normal escape exhausted at "
                      f"{1000*distances['l']:.0f}mm; clearance=cartesian", flush=True)
            quiet_streak = 0
            if name == "strip" and active == {"l"}:
                set_phase("strip_clearance_entry")
                if not sync_command_to_state("strip/clearance_entry"):
                    print(f"[release:{name}] ABORT clearance entry tracking", flush=True)
                    return False
                set_phase("strip_clearance_home")
                print(f"[release:{name}] left pad is trapped below part; "
                      f"clearance=cartesian_right_then_left",
                      flush=True)

                right_direction = last_directions["r"]
                right_requires_gap = right_direction is not None
                if right_direction is None:
                    print(f"[release:{name}] right clearance=contact_monitor_only",
                          flush=True)
                right_extra = 0.0
                while (right_requires_gap
                       and hand_section_clearance(name, "r") < 0.020):
                    if right_extra + step > 0.080 + 1e-9:
                        print(f"[release:{name}] ABORT right normal clearance "
                              f"gap={1000*hand_section_clearance(name, 'r'):.1f}mm",
                              flush=True)
                        return False
                    q0, v0 = d.qpos.copy(), d.qvel.copy()
                    seed = measured_arm_targets()[1]
                    for j, adr in enumerate(ct.R.qadr):
                        d.qpos[adr] = seed[j]
                    mujoco.mj_fwdPosition(m, d)
                    rotation = d.xmat[ct.R.mount].reshape(3, 3).copy()
                    target_position = d.xpos[ct.R.mount].copy() + step * right_direction
                    # Once contact is quiet, position clearance is the only physical
                    # objective. Keeping the old wrist orientation at the joint limit
                    # makes the commanded Cartesian path fold back through the strip.
                    ct.ik(ct.R, target_position, rotation, iters=45, w=0.0)
                    target = np.array([d.qpos[adr] for adr in ct.R.qadr])
                    residual = float(np.linalg.norm(target_position - d.xpos[ct.R.mount]))
                    d.qpos[:], d.qvel[:] = q0, v0
                    mujoco.mj_forward(m, d)
                    if residual > tol:
                        print(f"[release:{name}] ABORT right clearance IK "
                              f"residual={1000*residual:.1f}mm", flush=True)
                        return False
                    servo_to(ct.qtgt["l"].copy(), target,
                             min_steps=max(DECIM, int(np.ceil(step / 0.002))))
                    right_extra += step
                    object_shift = float(np.linalg.norm(obj_pos(name)[:2] - anchor_xy))
                    gap = hand_section_clearance(name, "r")
                    print(f"[release:{name}:r_clear] extra={1000*right_extra:.0f}mm "
                          f"gap={1000*gap:.1f}mm force={grip_force('r', name):.2f}N "
                          f"track={arm_tracking_error(('r',)):.4f}rad",
                          flush=True)
                    if object_shift > max_object_shift:
                        print(f"[release:{name}] ABORT right clearance object_xy_shift="
                              f"{1000*object_shift:.1f}mm", flush=True)
                        return False
                set_phase("strip_clearance_right_sync")
                if not sync_command_to_state("strip/right_clearance"):
                    print(f"[release:{name}] ABORT right clearance sync", flush=True)
                    return False
                right_gap = hand_section_clearance(name, "r")
                if (grip_force("r", name) > quiet
                        or (right_requires_gap and right_gap < 0.020)):
                    print(f"[release:{name}] ABORT right clearance lost after sync "
                          f"gap={1000*right_gap:.1f}mm", flush=True)
                    return False

                left_extra = 0.0
                while True:
                    set_phase("strip_clearance_left")
                    while max(grip_force("l", name), grip_force("r", name)) > quiet:
                        if grip_force("r", name) > quiet:
                            print(f"[release:{name}] ABORT right recontact during left clear",
                                  flush=True)
                            return False
                        if left_extra + step > 0.080 + 1e-9:
                            print(f"[release:{name}] ABORT left normal clearance "
                                  f"gap={1000*hand_section_clearance(name, 'l'):.1f}mm",
                                  flush=True)
                            return False
                        weighted = np.zeros(3)
                        total_weight = 0.0
                        pads = set(PADG["l"])
                        for i in range(d.ncon):
                            contact = d.contact[i]
                            pad = None
                            if contact.geom1 in pads and contact.geom2 in object_geoms:
                                pad = contact.geom1
                            elif contact.geom2 in pads and contact.geom1 in object_geoms:
                                pad = contact.geom2
                            if pad is None:
                                continue
                            normal = np.asarray(contact.frame[:3], dtype=float).copy()
                            if float(np.dot(normal, d.geom_xpos[pad] - contact.pos)) < 0.0:
                                normal *= -1.0
                            mujoco.mj_contactForce(m, d, i, FT)
                            weight = max(abs(float(FT[0])), 1e-6)
                            weighted += weight * normal
                            total_weight += weight
                        coherence = (float(np.linalg.norm(weighted) / total_weight)
                                     if total_weight else 0.0)
                        left_direction = weighted.copy()
                        left_norm = float(np.linalg.norm(left_direction))
                        if coherence < 0.25:
                            print(f"[release:{name}] ABORT left contact normal "
                                  f"coherence={coherence:.3f}", flush=True)
                            return False
                        left_direction /= left_norm
                        q0, v0 = d.qpos.copy(), d.qvel.copy()
                        seed = measured_arm_targets()[0]
                        for j, adr in enumerate(ct.L.qadr):
                            d.qpos[adr] = seed[j]
                        mujoco.mj_fwdPosition(m, d)
                        rotation = d.xmat[ct.L.mount].reshape(3, 3).copy()
                        target_position = d.xpos[ct.L.mount].copy() + step * left_direction
                        ct.ik(ct.L, target_position, rotation, iters=45, w=0.0)
                        target = np.array([d.qpos[adr] for adr in ct.L.qadr])
                        residual = float(np.linalg.norm(target_position - d.xpos[ct.L.mount]))
                        d.qpos[:], d.qvel[:] = q0, v0
                        mujoco.mj_forward(m, d)
                        if residual > tol:
                            print(f"[release:{name}] ABORT left clearance IK "
                                  f"residual={1000*residual:.1f}mm", flush=True)
                            return False
                        servo_to(target, ct.qtgt["r"].copy(),
                                 min_steps=max(DECIM, int(np.ceil(step / 0.002))))
                        left_extra += step
                        object_shift = float(np.linalg.norm(obj_pos(name)[:2] - anchor_xy))
                        left_gap = hand_section_clearance(name, "l")
                        right_gap = hand_section_clearance(name, "r")
                        print(f"[release:{name}:l_clear] extra={1000*left_extra:.0f}mm "
                              f"axis={np.round(left_direction, 2).tolist()} "
                              f"gap=({1000*left_gap:.1f},{1000*right_gap:.1f})mm "
                              f"force=({grip_force('l', name):.2f},"
                              f"{grip_force('r', name):.2f})N "
                              f"track={arm_tracking_error(('l',)):.4f}rad",
                              flush=True)
                        if (object_shift > max_object_shift
                                or (right_requires_gap and right_gap < 0.020)):
                            print(f"[release:{name}] ABORT left clearance safety "
                                  f"object_xy_shift={1000*object_shift:.1f}mm "
                                  f"right_gap={1000*right_gap:.1f}mm", flush=True)
                            return False
                    set_phase("strip_clearance_exit")
                    if not sync_command_to_state("strip/clearance_exit"):
                        print(f"[release:{name}] ABORT clearance exit tracking", flush=True)
                        return False
                    final_gaps = [hand_section_clearance(name, hand)
                                  for hand in ("l", "r")]
                    if right_requires_gap and final_gaps[1] < 0.020:
                        print(f"[release:{name}] ABORT clearance lost at exit "
                              f"gaps=({1000*final_gaps[0]:.1f},"
                              f"{1000*final_gaps[1]:.1f})mm", flush=True)
                        return False
                    set_phase("strip_clearance_hold")
                    stable = 0
                    evidence = placement_evidence(name)
                    recontact = False
                    for _ in range(quiet_required * 4):
                        evidence = placement_evidence(name)
                        force = max(grip_force("l", name), grip_force("r", name))
                        object_shift = float(np.linalg.norm(obj_pos(name)[:2] - anchor_xy))
                        right_gap = hand_section_clearance(name, "r")
                        if (object_shift > max_object_shift
                                or (right_requires_gap and right_gap < 0.020)):
                            print(f"[release:{name}] ABORT clearance hold safety "
                                  f"object_xy_shift={1000*object_shift:.1f}mm "
                                  f"right_gap={1000*right_gap:.1f}mm", flush=True)
                            return False
                        valid = (force <= quiet and evidence["fully_on_shelf"]
                                 and evidence["supported"])
                        stable = stable + 1 if valid else 0
                        if stable >= quiet_required:
                            print(f"[release:{name}] left clearance quiet_for={quiet_s:.2f}s "
                                  f"object_xy_shift={1000*object_shift:.1f}mm "
                                  f"support={evidence['cart_support_force_n']:.2f}N",
                                  flush=True)
                            return True
                        if force > quiet:
                            print(f"[release:{name}] left recontact during hold "
                                  f"force={force:.2f}N", flush=True)
                            recontact = True
                            break
                        frames(1)
                    if not recontact:
                        print(f"[release:{name}] ABORT left clearance unstable "
                              f"force={force:.2f}N inside={evidence['fully_on_shelf']}",
                              flush=True)
                        return False
        if not active:
            quiet_streak += 1
            if quiet_streak >= quiet_required:
                print(f"[release:{name}] quiet_for={quiet_s:.2f}s out_mm="
                      f"({1000*distances['l']:.0f},{1000*distances['r']:.0f}) "
                      f"object_xy_shift={1000*object_shift:.1f}mm", flush=True)
                return True
            frames(1)
            continue
        if any(distances[hand] + step > max_escape + 1e-9 for hand in active):
            print(f"[release:{name}] ABORT max_escape_mm={1000*max_escape:.0f} "
                  f"force=({grip_force('l', name):.2f},{grip_force('r', name):.2f})N",
                  flush=True)
            return False

        directions = {}
        for hand in active:
            weighted = np.zeros(3)
            total_weight = 0.0
            pads = set(PADG[hand])
            for i in range(d.ncon):
                contact = d.contact[i]
                pad = None
                if contact.geom1 in pads and contact.geom2 in object_geoms:
                    pad = contact.geom1
                elif contact.geom2 in pads and contact.geom1 in object_geoms:
                    pad = contact.geom2
                if pad is None:
                    continue
                normal = np.asarray(contact.frame[:3], dtype=float).copy()
                if float(np.dot(normal, d.geom_xpos[pad] - contact.pos)) < 0.0:
                    normal *= -1.0
                mujoco.mj_contactForce(m, d, i, FT)
                weight = max(abs(float(FT[0])), 1e-6)
                weighted += weight * normal
                total_weight += weight
            coherence = float(np.linalg.norm(weighted) / total_weight) if total_weight else 0.0
            if coherence < 0.25:
                print(f"[release:{name}] ABORT contact_normal_coherence={coherence:.3f}",
                      flush=True)
                return False
            directions[hand] = weighted / np.linalg.norm(weighted)
            last_directions[hand] = directions[hand].copy()

        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets, errors = {}, {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            seed = ct.qtgt[hand].copy()
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = seed[j]
            mujoco.mj_fwdPosition(m, d)
            if hand not in active:
                targets[hand] = seed
                errors[hand] = 0.0
                continue
            rotation = d.xmat[arm.mount].reshape(3, 3).copy()
            target_position = d.xpos[arm.mount].copy() + step * directions[hand]
            ct.ik(arm, target_position, rotation, iters=45, w=0.35)
            targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
            errors[hand] = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)
        if max(errors.values()) > tol:
            print(f"[release:{name}] ABORT escape_ik_residual_mm="
                  f"({1000*errors['l']:.1f},{1000*errors['r']:.1f})", flush=True)
            return False
        servo_to(targets["l"], targets["r"],
                 min_steps=max(DECIM, int(np.ceil(step / 0.002))))
        for hand in active:
            distances[hand] += step
            print(f"[release:{name}:{hand}] out={1000*distances[hand]:.0f}mm "
                  f"axis={np.round(directions[hand], 2).tolist()} "
                  f"force={grip_force(hand, name):.2f}N", flush=True)


def ungrasp_hands(name, total=0.050, step=0.010, tol=0.010):
    """Back each hand off the part along its own approach axis, in small steps.

    The pads sit 95.5 mm out along the mount's local z. At release the measured strip section
    is only 82-83 mm out, behind the pad centres. Negative local z drives the pads into the
    part; positive local z decreases the section's mount-frame z and is the escape direction.

    Every alternative tried moves the part instead. Sliding sideways off the nearest section
    sends the right hand deeper into the cart, where the arm is at full extension and delivers
    1 mm of a 20 mm request, and going home from there rakes the bar 90-220 mm along its own
    length at 216-302 N. Backing straight off along the chassis heading is worse still, 0/13
    at up to 913 N, because that is across the jaws rather than out of them. A 40 mm vertical
    lift half-works only because it partly aligns with this axis, and it is a knife edge:
    0 mm gives 0/13, 40 mm gives 8/13, 60 mm gives 2/13.

    Returns the distance actually withdrawn, which is less than asked when the arm runs out
    of reach -- the caller's move home is rate-limited and can finish the job.
    """
    done = 0.0
    while done < total - 1e-9:
        span = min(step, total - done)
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets, errors = {}, {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            seed = ct.qtgt[hand].copy()
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = seed[j]
            mujoco.mj_fwdPosition(m, d)
            target_rotation = d.xmat[arm.mount].reshape(3, 3).copy()
            target_position = d.xpos[arm.mount] + span * target_rotation[:, 2]
            if GRASP_DUMP:
                print(f"[ungrasp:{name}:{hand}] axis="
                      f"{np.round(target_rotation[:, 2], 2).tolist()}", flush=True)
            ct.ik(arm, target_position, target_rotation, iters=45, w=0.35)
            best_q = np.array([d.qpos[adr] for adr in arm.qadr])
            best_error = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
            if best_error > tol:
                for j, adr in enumerate(arm.qadr):
                    d.qpos[adr] = seed[j]
                mujoco.mj_fwdPosition(m, d)
                ct.ik(arm, target_position, target_rotation, iters=45, w=0.0)
                free_q = np.array([d.qpos[adr] for adr in arm.qadr])
                free_error = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
                if free_error < best_error:
                    best_q, best_error = free_q, free_error
            targets[hand], errors[hand] = best_q, best_error
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        if max(errors.values()) > tol:
            print(f"[ungrasp:{name}] stopped at {1000*done:.0f}mm of {1000*total:.0f} "
                  f"residual_mm=({1000*errors['l']:.1f},{1000*errors['r']:.1f})", flush=True)
            return done
        servo_to(targets["l"], targets["r"],
                 min_steps=max(DECIM, int(np.ceil(span / 0.002))))
        done += span
    print(f"[ungrasp:{name}] withdrew {1000*done:.0f}mm", flush=True)
    return done


def retreat_home(name, mode="lift", lift=0.04, home_steps=55):
    """Let go of the part and get the arms home without moving what was just placed.

    mode is how the pads come off: "lift" raises then backs off along the chassis heading,
    "liftcare" does the same through the checked IK channel, "ungrasp" backs out along the
    approach axis, "spread" pushes sideways off the nearest section. See ungrasp_hands for the
    measurements that rank the directions.
    """
    set_phase(f"{name}_release_entry")
    # First unload stale IK error through a rate-limited command. Otherwise the opening
    # hold records a large state/action mismatch before the old post-hold sync can run.
    if not sync_command_to_state(f"{name}/release_entry"):
        gate(f"{name}_release_entry_tracking", False, "failed to settle before opening")
        finish(False)
    set_phase(f"{name}_release_open")
    ct.grip_cmd["l"] = ct.GRIP_OPEN
    ct.grip_cmd["r"] = ct.GRIP_OPEN
    frames(70)
    yaw = ct.base_pose()[2]

    def stage(label):
        print(f"[retreat:{name}] {label:12s} obj={np.round(obj_pos(name), 3).tolist()} "
              f"fR={grip_force('r', name):.1f}N fL={grip_force('l', name):.1f}N "
              f"yaw={yaw:+.2f}", flush=True)

    stage("opened")
    if GRASP_DUMP:
        grasp_dump(name)
        # Where does the part actually sit inside the open jaws? That decides which way the
        # hand can leave without taking it along, and it is not guessable: the jaw axis is the
        # mount's local y, the approach axis its local z, and the wrist puts both at an angle.
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            rot = d.xmat[arm.mount].reshape(3, 3)
            origin = d.xpos[arm.mount]
            pads = np.mean([d.geom_xpos[g] for g in PADG[hand]], axis=0)
            glo, ghi = nearest_section(name, float(pads[1]))
            local = rot.T @ (0.5 * (glo + ghi) - origin)
            _, shelf_top = ct.geom_aabb(SHELF[name])
            print(f"[jaw:{name}:{hand}] section_in_mount_mm={np.round(1000*local, 1).tolist()} "
                  f"local_axes_z={np.round(rot[:, 2], 2).tolist()} "
                  f"y={np.round(rot[:, 1], 2).tolist()}", flush=True)
            for g in sorted(PADG[hand], key=lambda g: d.geom_xpos[g][2]):
                plo, phi = ct.geom_aabb(g)
                print(f"[jaw:{name}:{hand}] {gname(g)} z_span_mm="
                      f"[{1000*plo[2]:.0f},{1000*phi[2]:.0f}] "
                      f"vs shelf_top={1000*shelf_top[2]:.0f} "
                      f"part_bottom={1000*obj_extent(name)[0][2]:.0f}", flush=True)
    if mode == "liftcare":
        # Same path as "lift", executed through the checked channel. The single-shot version
        # below asks for the whole move with 20 iterations, no residual check, and writes the
        # answer in as a step change run off in 18 frames. That is why more lift comes out
        # worse than less: 0 mm gives 0/13, 40 mm gives 8/13, 60 mm gives 2/13, which cannot
        # be geometry. move_hands refuses what IK cannot reach and ramps what it accepts.
        raised = 0.0
        while raised < lift - 1e-9:
            span = min(0.010, lift - raised)
            if not move_hands([0.0, 0.0, span]):
                break
            raised += span
        print(f"[liftcare:{name}] raised {1000*raised:.0f}mm of {1000*lift:.0f}", flush=True)
        stage("lifted")
        step = np.array([-0.020 * np.cos(yaw), -0.020 * np.sin(yaw), 0.0])
        moved = 0.0
        for _ in range(6):
            if not move_hands(step):
                break
            moved += 0.020
        print(f"[liftcare:{name}] backed off {1000*moved:.0f}mm of 120", flush=True)
        stage("withdraw")
        move_home()
        stage("home")
        return
    if mode == "release":
        if name == "strip":
            set_phase("strip_release_withdraw")
            released = release_hands(name)
            stage("released")
            if not released:
                gate("strip_policy_release", False, "pad force remained above 0.2 N")
                finish(False)
            set_phase("strip_policy_terminal")
            if not wait_for_arm_tracking("strip/policy_terminal", tol=TERMINAL_TRACKING_MAX):
                gate("strip_policy_terminal_tracking", False, "failed to settle before policy end")
                finish(False)
            if max(grip_force("r", name), grip_force("l", name)) > 0.2:
                gate("strip_policy_terminal_release", False, "pad recontact after tracking settle")
                finish(False)
            stage("policy_release")
            return
        # Joint-space home from inside the shelf is collision-prone: the old path
        # often changed the command by 2.2 rad while the measured joints did not
        # move. Pull the whole robot straight back first, keeping the open hands
        # at their measured pose, then return each arm home in free space.
        set_phase("pillar_clearance")
        clearance_reverse(0.10, vmax=0.04)
        stage("base_clear")
        set_phase("pillar_home")
        if not move_home(steps=home_steps, order=HOME_ORDER):
            gate("pillar_home_tracking", False, "closed-loop home governor timed out")
            finish(False)
        stage("home")
        return
    if mode == "ungrasp":
        ungrasp_hands(name, total=UNGRASP_M)
        stage("ungrasp")
        move_home()
        stage("home")
        return
    if mode == "spread":
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets = {}
        errors = {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            seed = ct.qtgt[hand].copy()
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = seed[j]
            mujoco.mj_fwdPosition(m, d)
            position = d.xpos[arm.mount].copy()
            rotation = d.xmat[arm.mount].reshape(3, 3).copy()
            outward = escape_dir(name, hand)
            target_position = position.copy()
            target_position[:2] += 0.08 * outward
            ct.ik(arm, target_position, rotation, iters=45, w=0.0)
            errors[hand] = float(np.linalg.norm(target_position - d.xpos[arm.mount]))
            targets[hand] = (
                np.array([d.qpos[adr] for adr in arm.qadr])
                if errors[hand] <= 0.020 else seed
            )
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        print(
            f"[{name}_release] spread_err_mm=({1000*errors['l']:.1f},{1000*errors['r']:.1f})",
            flush=True,
        )
        starts = {h: ct.qtgt[h].copy() for h in ("l", "r")}
        for i in range(45):
            blend = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / 45)
            for hand in ("l", "r"):
                ct.qtgt[hand][:] = starts[hand] + (targets[hand] - starts[hand]) * blend
                ct.grip_cmd[hand] = ct.GRIP_OPEN
            frames(1)
        stage("spread")
        ungrasp_hands(name, total=0.12, step=0.02)
        stage("withdraw")
        move_home()
        stage("home")
        return

    back = np.array([-0.12 * np.cos(yaw), -0.12 * np.sin(yaw), 0.0])
    steps = [np.array([0.0, 0.0, lift]), back] if lift > 0.0 else [back]
    for delta in steps:
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        targets = {}
        errors = {}
        for hand, arm in (("l", ct.L), ("r", ct.R)):
            pm = d.xpos[arm.mount].copy() + delta
            rm = d.xmat[arm.mount].reshape(3, 3).copy()
            for j, adr in enumerate(arm.qadr):
                d.qpos[adr] = ct.qtgt[hand][j]
            mujoco.mj_fwdPosition(m, d)
            ct.ik(arm, pm, rm, iters=20, w=0.5)
            errors[hand] = float(np.linalg.norm(pm - d.xpos[arm.mount]))
            targets[hand] = np.array([d.qpos[adr] for adr in arm.qadr])
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_fwdPosition(m, d)
        print(f"[{name}_withdraw] err_mm=({1000*errors['l']:.1f},{1000*errors['r']:.1f})",
              flush=True)
        command_arm_targets(
            targets["l"],
            targets["r"],
            grip_l=ct.GRIP_OPEN,
            grip_r=ct.GRIP_OPEN,
            label=f"{name}/withdraw",
        )
        frames(18)
        stage("withdraw")
    move_home()
    stage("home")


def verify_placement(name):
    required = max(1, int(np.ceil(0.5 / CONTROL_DT)))
    streak = 0
    evidence = placement_evidence(name)
    for _ in range(required * 4):
        frames(1)
        evidence = placement_evidence(name)
        valid = (evidence["in_assigned_layer"] and evidence["fully_on_shelf"]
                 and evidence["released"] and evidence["supported"])
        streak = streak + 1 if valid else 0
        if streak >= required:
            break
    evidence["stable_for_s"] = round(float(streak * CONTROL_DT), 3)
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
    # The pillar keeps its demo-relative place_offset on purpose: retargeting it to
    # the shelf centre would shift park_place ~0.39 m towards the cart and collide.
    park_place = ref["base"][ref["ps"]].copy()
    park_place[:2] += place_offset[:2]
    if name == "strip":
        park_place[1] += 0.03
    return grasp_offset, place_offset, park_grasp, park_place, grasp_yaw_rotation


def run_trip(name):
    global TRIP
    TRIP = name
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
    set_phase(f"{name}_navigate_to_grasp")
    # Formal recovery data gets one guaranteed, bounded perturbation while the
    # grippers are empty. The existing controller then supplies the recovery label.
    # Clean data never enters this branch. The formal batch requests one event,
    # so its later stochastic calls see a zero budget.
    if name == "pillar" and DIVERSITY_MODE == "recovery":
        maybe_kick(1.0, scale=0.35, trigger="controlled_empty_navigation_entry")
    if name == "strip":
        go_to(*front_side)
        go_to(*back_side)
    go_to(*park_grasp)
    hand_offset = None
    if GRASP_FIX in ("all", name):
        hand_offset = grasp_correction_per_hand(
            name, ref, grasp_offset, grasp_yaw_rotation)
    set_phase(f"{name}_grasp")
    arm_replay(ref, "grasp", grasp_offset, ik_weight=0.6, yaw_rotation=grasp_yaw_rotation,
               hand_offset=hand_offset)
    frames(10)
    shift_l = np.zeros(3) if hand_offset is None else hand_offset["l"]
    shift_r = np.zeros(3) if hand_offset is None else hand_offset["r"]
    target_l, _ = transform_mount(*ref["mount"]["grasp"]["l"][-1], grasp_offset + shift_l, grasp_yaw_rotation, ref["ref_center"])
    target_r, _ = transform_mount(*ref["mount"]["grasp"]["r"][-1], grasp_offset + shift_r, grasp_yaw_rotation, ref["ref_center"])
    pad_pos = {
        hand: [[round(float(v), 3) for v in d.geom_xpos[g]] for g in PADG[hand]]
        for hand in ("l", "r")
    }
    print(
        f"[grasp_probe:{name}] obj={np.round(obj_pos(name), 3).tolist()} "
        f"mount_err_mm=({1000*np.linalg.norm(d.xpos[ct.L.mount]-target_l):.1f},"
        f"{1000*np.linalg.norm(d.xpos[ct.R.mount]-target_r):.1f}) pads={pad_pos}",
        flush=True,
    )
    if GRASP_DUMP:
        grasp_dump(name)
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    if not gate(
        f"{name}_grip",
        pad_contacts_both("r", name) and pad_contacts_both("l", name) and f_r >= 1.0 and f_l >= 1.0,
        f"bothR={pad_contacts_both('r', name)} bothL={pad_contacts_both('l', name)} fR={f_r:.1f}N fL={f_l:.1f}N",
    ):
        finish(False)

    set_phase(f"{name}_tuck")
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
    set_phase(f"{name}_transport")
    if name == "strip":
        # Pull straight back, rotate only while stationary and drive each leg
        # nearly straight. The long part cannot tolerate normal chassis yaw.
        clearance_reverse(0.42)
        require_held("clear_rack")
        clear_back = np.array(ct.base_pose())
        heading_side = np.arctan2(back_side[1] - clear_back[1], back_side[0] - clear_back[0])
        go_to(clear_back[0], clear_back[1], heading_side, wmax=STRIP_WMAX)
        require_held("turned_to_side")
        go_to(back_side[0], back_side[1], heading_side, vmax=STRIP_VMAX, wmax=STRIP_WMAX, kick_scale=0.20)
        require_held("back_side")
        go_to(back_side[0], back_side[1], np.pi, wmax=STRIP_WMAX)
        go_to(front_side[0], front_side[1], np.pi, vmax=STRIP_VMAX, wmax=STRIP_WMAX, kick_scale=0.20)
        require_held("front_side")
        if PARK_FROM_HAND:
            park_place = np.array([
                *park_for_carried(name, park_place[2], park_place[:2]), park_place[2]])
        align_cart = np.array([-0.75, park_place[1], 0.0])
        heading_align = np.arctan2(align_cart[1] - front_side[1], align_cart[0] - front_side[0])
        go_to(front_side[0], front_side[1], heading_align, wmax=STRIP_WMAX)
        go_to(align_cart[0], align_cart[1], heading_align, vmax=STRIP_VMAX,
              wmax=STRIP_WMAX, kick_scale=0.20)
        require_held("cart_alignment_point")
        go_to(align_cart[0], align_cart[1], park_place[2], wmax=STRIP_WMAX)
        require_held("aligned_with_cart")

        forward = np.array([np.cos(park_place[2]), np.sin(park_place[2])])
        lift_stage = park_place[:2] - STRIP_LIFT_STAGE_BACKOFF * forward
        go_to(lift_stage[0], lift_stage[1], park_place[2], vmax=STRIP_VMAX,
              wmax=STRIP_WMAX, kick_scale=0.20)
        require_held("lift_stage")
        bx, by, byaw = ct.base_pose()
        bv, bw = ct.base_velocity()
        stage_error = float(np.hypot(lift_stage[0] - bx, lift_stage[1] - by))
        yaw_error = abs(angle(park_place[2] - byaw))
        stage_ok = stage_error <= 0.030 and yaw_error <= 0.040 and abs(bv) <= 0.020 and abs(bw) <= 0.020
        if not gate(
            "strip_lift_stage",
            stage_ok,
            f"derr={stage_error:.3f}m yerr={np.degrees(yaw_error):.1f}deg "
            f"v={bv:.3f}m/s w={bw:.3f}rad/s",
        ):
            finish(False)

        set_phase("strip_lift")
        target_z = float(shelf_target(name)[2] + STRIP_TOP_CLEARANCE)
        lift_delta = max(0.0, target_z - float(obj_pos(name)[2]))
        print(
            f"[strip_height] shelf_top={ct.geom_aabb(SHELF[name])[1][2]:.3f} "
            f"target_z={target_z:.3f} current_z={obj_pos(name)[2]:.3f} "
            f"lift={lift_delta:.3f}m",
            flush=True,
        )
        lift_joint_targets([0.0, 0.0, lift_delta])
        force_r, force_l = grip_force("r", name), grip_force("l", name)
        released_hand = None
        if force_r >= 0.5 and force_l >= 0.5:
            carry_hands[:] = ["r", "l"]
        elif force_l >= 0.5:
            carry_hands[:] = ["l"]
            released_hand = "r"
        elif force_r >= 0.5:
            carry_hands[:] = ["r"]
            released_hand = "l"
        else:
            gate(f"{name}_carry", False, "both hands lost during top lift")
            finish(False)
        print(f"[top_handoff] keep={'+'.join(carry_hands)} fR={force_r:.1f}N fL={force_l:.1f}N", flush=True)
        if released_hand is not None:
            ct.grip_cmd[released_hand] = ct.GRIP_OPEN
        frames(20)
        require_held("raised_for_top")
        actual_z = float(obj_pos(name)[2])
        if not gate(
            "strip_lift_height",
            abs(actual_z - target_z) <= 0.030,
            f"target_z={target_z:.3f} actual_z={actual_z:.3f} "
            f"error={1000*abs(actual_z-target_z):.0f}mm",
        ):
            finish(False)
        print(f"[strip_lift] object_z={actual_z:.3f}", flush=True)
        set_phase("strip_final_approach")
        go_to(*park_place, vmax=STRIP_VMAX, wmax=STRIP_WMAX, kick_scale=0.20)
    else:
        go_to(*park_place, vmax=0.20, wmax=0.20, kick_scale=0.35)
    f_r, f_l = grip_force("r", name), grip_force("l", name)
    carry_forces = {"r": f_r, "l": f_l}
    if not gate(f"{name}_carry", all(carry_forces[hand] >= 0.5 for hand in carry_hands),
                f"hands={carry_hands} fR={f_r:.1f}N fL={f_l:.1f}N"):
        finish(False)

    set_phase(f"{name}_place")
    if PLACE_FIX:
        correction = inhand_correction(name, ref)
        if name == "pillar" and PILLAR_PLACE_BIAS_X:
            correction[0] += PILLAR_PLACE_BIAS_X
            print(f"[pillar_place_bias] dx={1000*PILLAR_PLACE_BIAS_X:+.0f}mm", flush=True)
        place_offset = place_offset + correction
    if name != "strip":
        # The demo opens the gripper at ref["release"], and the place IK typically lands
        # 60-95 mm short, so replaying that frame drops the pillar 11-12 cm onto the shelf
        # and whether it stays is luck: seeds 1, 8 and 11 all reach release with ~95 mm
        # residual, ~100 N pad force and a +113..121 mm gap, and only 11 survives the fall.
        # Holding through the release frame and touching down first is what the strip
        # already does.
        arm_replay(ref, "place", place_offset, ik_weight=0.6,
                   stop_at=ref["release"] if PILLAR_TOUCHDOWN else None)
        probe = placement_evidence(name)
        lo, _ = obj_extent(name)
        _, shi = ct.geom_aabb(SHELF[name])
        print(
            f"[place_probe:{name}] pos={np.round(obj_pos(name), 3).tolist()} "
            f"grip=({probe['grip_force_right_n']:.1f},{probe['grip_force_left_n']:.1f})N "
            f"support={probe['cart_support_force_n']:.1f}N gap={1000*(lo[2]-shi[2]):+.0f}mm",
            flush=True,
        )
    elif PLACE_FIX:
        # The strip has no place replay: the arms hold the carry pose and the chassis does
        # the positioning. But the cart blocks the chassis about 200 mm short of where the
        # bar would be centred, so that last stretch has to come from the arms. Touchdown
        # used to zero this correction entirely, which left the bar 416 mm off the shelf.
        residual = shelf_target(name) - obj_pos(name)
        norm = float(np.linalg.norm(residual))
        if norm > PLACE_CORR_MAX:
            residual = residual * (PLACE_CORR_MAX / norm)
        # PLACE_REACH is off by default: reaching for the remaining 200 mm only completes
        # 3 of 11 steps before the IK gives out, and abandoning it part-way leaves the bar
        # 175 mm higher and further from the shelf than not trying at all. Closing that gap
        # needs a real place trajectory rather than a correction on top of the carry pose.
        correction = residual.copy() if PLACE_REACH else np.zeros(3)
        if TOUCHDOWN:
            correction[2] = 0.0  # height is what touchdown is for
        steps = max(1, int(np.ceil(float(np.linalg.norm(correction)) / 0.02)))
        done = 0
        for _ in range(steps):
            if not np.any(correction) or not move_hands(correction / steps):
                break
            done += 1
        after = shelf_target(name) - obj_pos(name)
        print(
            f"[place:{name}] residual={np.round(residual, 3).tolist()} |r|={norm:.3f}m "
            f"steps={done}/{steps} remaining={np.round(after, 3).tolist()} "
            f"|rem|={float(np.linalg.norm(after)):.3f}m",
            flush=True,
        )
    if CLEAR_POSTS and name == "strip":
        clear_posts(name)
    if LEVEL and name == "strip":
        level_part(name)
    release_home_steps = 55
    seat_for_touchdown = False
    if name == "strip":
        axis = part_long_axis(name)
        release_tilt_deg = abs(float(np.degrees(np.arcsin(np.clip(axis[2], -1.0, 1.0)))))
        release_home_steps = 35 if release_tilt_deg < 0.65 else 55
        # The rigid strip is arched: first contact only seats one end, regardless of
        # its small measured tilt. Always transfer its full weight to the shelf.
        seat_for_touchdown = True
        print(f"[release_profile:{name}] tilt={release_tilt_deg:.2f}deg "
              f"seat={seat_for_touchdown} home_steps={release_home_steps}", flush=True)
    wants_touchdown = (TOUCHDOWN and name == "strip") or (PILLAR_TOUCHDOWN and name == "pillar")
    if wants_touchdown and not touch_down(name, seat=seat_for_touchdown):
        gate(f"{name}_placed", False, "touchdown not reached")
        finish(False)
    ct.base_vel[:] = 0.0
    set_phase(f"{name}_release_and_home")
    retreat_home(name, mode=RETREAT,
                 lift=STRIP_LIFT if name == "strip" else 0.04,
                 home_steps=release_home_steps)
    evidence = verify_placement(name)
    if GRASP_DUMP:
        final_contacts(name)
    p = obj_pos(name)
    if not gate(
        f"{name}_placed",
        evidence["stable"],
        f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) released={evidence['released']} "
        f"inside={evidence['fully_on_shelf']} support={evidence['cart_support_force_n']:.1f}N "
    ):
        finish(False)
    if name == "strip":
        set_phase("policy_terminal_verify")
        pillar_endpoint = verify_placement("pillar")
        strip_endpoint = verify_placement("strip")
        both_endpoint = pillar_endpoint["stable"] and strip_endpoint["stable"]
        if not gate(
            "placed_both_policy_end",
            both_endpoint,
            f"pillar={pillar_endpoint['stable']} strip={strip_endpoint['stable']}",
        ):
            finish(False)

        close_policy_episode("both_objects_released_and_stable")
        set_phase("safety_home")
        move_home(steps=release_home_steps, order=HOME_ORDER, wait=False)
        home_error = arm_tracking_error()
        VALIDATION["safety_home"] = {
            "tracking_error_rad": round(float(home_error), 6),
            "tracking_passed": bool(home_error <= TRACK_SETTLE_TOL),
            "recorded_in_policy_episode": False,
        }
        print(
            f"[safety_home] tracking_error={home_error:.4f}rad "
            f"tracked={home_error <= TRACK_SETTLE_TOL}",
            flush=True,
        )



# Fixed semantic order: red circled pillar first, green circled strip second.
run_trip("pillar")
run_trip("strip")

# Recheck both after the second trip: contact from the top-layer operation must
# not have displaced the already completed middle-layer placement.
set_phase("final_verify")
pillar_final = verify_placement("pillar")
strip_final = verify_placement("strip")
placement_audit("pillar")
placement_audit("strip")
both = pillar_final["stable"] and strip_final["stable"]
if "safety_home" in VALIDATION:
    VALIDATION["safety_home"]["objects_stable"] = bool(both)
gate(
    "placed_both",
    both,
    f"pillar={pillar_final['stable']}({pillar_final['cart_support_force_n']:.1f}N) "
    f"strip={strip_final['stable']}({strip_final['cart_support_force_n']:.1f}N)",
)
finish(both)
