#!/usr/bin/env python3
"""Reward function for the CRUZR S2 pillar pick-and-place online-RL run.

Design doc: docs/RL_REWARD_DESIGN_pillar_e2e.md

Structure (deliberate): the reward LOGIC is a pure function of a plain `Sense` struct plus
latch state -- no MuJoCo import needed -- so every anti-hacking property can be unit tested
without a simulator (test_shelf_e2e_reward.py). `MujocoSensor` is the thin adapter that fills
a `Sense` from (m, d).

Core properties the tests pin down:
  * dense terms are potential-based (r = gamma*Phi' - Phi) -> any CLOSED loop in state space
    accumulates exactly zero shaping. Cycling cannot be farmed.
  * sparse event bonuses are latched (paid once) and strictly ordered
    touch -> grasp -> lift -> arrive -> place; skipping a stage pays nothing.
  * yaw budget is the one deliberately path-dependent penalty: spinning must cost real money,
    "spinning back" must not refund it.

Shaping at termination: we keep the ordinary gamma*Phi(s') - Phi(s) form and do NOT force
Phi(terminal)=0. Ng et al.'s policy-invariance proof wants the zeroing, but zeroing would
subtract the whole accumulated progress at a failure terminal, i.e. punish a policy for having
made progress before it dropped the pillar. The property we actually rely on -- zero shaping
around any cycle -- holds either way, and failure terminals already carry explicit penalties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- stages
APPROACH, REACH, GRASP, LIFT, TRANSPORT, PLACE = range(6)
STAGE_NAMES = ["APPROACH", "REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE"]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Sense:
    """Everything the reward needs, measured from the sim. Privileged: reward-only, never observed."""
    t: int = 0                      # decision-step index
    dt: float = 0.272               # decision = 8 actions x 2 control steps x 17 x 1 ms
    hand_mid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasp_pt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obj_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obj_tilt: float = 0.0           # rad between object z-axis and world z
    obj_vz: float = 0.0             # vertical velocity [m/s]; distinguishes a set-down from a drop
    base_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    base_yaw: float = 0.0
    d_yaw: float = 0.0              # |delta yaw| accumulated over this decision step
    v_fwd: float = 0.0
    wz: float = 0.0
    f_l: float = 0.0                # gripper-pad <-> object normal force, per hand [N]
    f_r: float = 0.0
    touch_l: bool = False           # both pads of that hand in contact with the object
    touch_r: bool = False
    obj_rack_force: float = 0.0     # object <-> rack contact force [N]
    obj_shelf_force: float = 0.0    # object <-> cart shelf contact force [N]
    in_region: bool = False
    body_env_force: float = 0.0     # worst non-pad robot<->world contact force [N]
    n_joint_clip: int = 0
    a_arm: np.ndarray = field(default_factory=lambda: np.zeros(14))
    park_grasp: np.ndarray = field(default_factory=lambda: np.zeros(3))   # x, y, yaw
    park_place: np.ndarray = field(default_factory=lambda: np.zeros(3))
    region_center: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Cfg:
    gamma: float = 0.995        # RL discount (returns / GAE)
    gamma_shape: float = 1.0    # discount INSIDE the shaping term. Deliberately 1.0, not gamma:
                                # with gamma<1 the (gamma-1)*sum(Phi) leakage costs ~-12 over an
                                # 800-step episode -- more than the whole event budget pays, so a
                                # competent expert scored +5.9. gamma_shape=1 makes shaping exactly
                                # telescoping (a closed loop sums to exactly 0, still unfarmable).
    lam_shape: float = 1.0
    # --- potentials
    d_nav: float = 0.5
    d_carry: float = 1.0
    d_reach: float = 0.25
    reach_offset: float = 0.0   # hand-mount<->object distance at a firm grip (measured from the
                                # expert): subtracting it lets phi_reach actually reach 1.0
    d_place: float = 0.4
    f_hold: float = 3.0
    lift_h: float = 0.10            # potential saturates at +10 cm
    # --- event bonuses
    r_touch: float = 2.0
    r_grasp: float = 5.0
    r_lift: float = 10.0
    r_arrive: float = 5.0
    r_place: float = 40.0
    # --- gates
    park_tol_xy: float = 0.15
    park_tol_yaw: float = 0.15
    arrive_tol_xy: float = 0.25
    f_firm: float = 1.0
    f_release: float = 0.5
    lift_dz: float = 0.06
    free_of_rack_f: float = 1.0     # lift only counts if the pillar is really off the rack
    hold_grasp_s: float = 0.5
    hold_place_s: float = 2.0
    # --- penalties
    k_yaw: float = 0.15
    yaw_budget: float = 4 * math.pi
    p_spin: float = 5.0
    p_drop: float = 10.0
    drop_dz: float = 0.05
    drop_grace_s: float = 1.0   # the expert releases ~11 cm above the shelf and lets the pillar
                                # settle; judging a drop mid-flight flagged a SUCCESSFUL place as a
                                # drop. Only call it a drop after it has stayed lost this long.
    drop_floor_dz: float = 0.25 # ...or immediately once it is clearly heading for the floor
    drop_vz: float = 0.02       # |m/s| below which the object counts as resting, not falling
    p_knock: float = 5.0
    knock_tilt: float = math.radians(45)
    f_collide: float = 50.0
    f_collide_hard: float = 200.0
    p_collide: float = 0.5
    p_collide_hard: float = 10.0
    k_smooth: float = 0.02
    k_time: float = 0.002
    k_jclip: float = 0.02
    k_energy: float = 0.01
    max_steps: int = 300


class PillarReward:
    """Stateful reward. One instance per episode; call reset() then step(sense) each decision step."""

    def __init__(self, cfg: Cfg | None = None):
        self.cfg = cfg or Cfg()
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self, stage: int = APPROACH):
        c = self.cfg
        self.stage = stage
        # latches -- monotone, never cleared inside an episode
        self.lat = {"touch": False, "grasp": False, "lift": False, "arrive": False, "place": False}
        # pre-set latches when we warm-start mid-task (curriculum Phase A/B resets)
        for name, k in (("touch", GRASP), ("grasp", LIFT), ("lift", TRANSPORT), ("arrive", PLACE)):
            if stage >= k:
                self.lat[name] = True
        self._hold_grasp = 0.0
        self._hold_place = 0.0
        self._yaw_used = 0.0
        self._lost = 0.0
        self._z_peak = -1e9          # highest object z seen while firmly held
        self._z0 = None              # object rest height at reset
        self._prev_phi = None
        self._prev_a = None
        self.done = False
        self.term = ""
        self.ret = 0.0

    # ------------------------------------------------------------ potential
    def _stage_of(self, s: Sense) -> int:
        """Latched, monotone stage index."""
        k = self.stage
        if k <= APPROACH:
            dxy = float(np.linalg.norm(s.base_xy - s.park_grasp[:2]))
            if dxy < self.cfg.park_tol_xy and abs(wrap(s.base_yaw - s.park_grasp[2])) < self.cfg.park_tol_yaw:
                k = REACH
        if k <= REACH and (s.touch_l and s.touch_r):
            k = GRASP
        if k <= GRASP and self.lat["grasp"]:
            k = LIFT
        if k <= LIFT and self.lat["lift"]:
            k = TRANSPORT
        if k <= TRANSPORT and self.lat["arrive"]:
            k = PLACE
        return max(k, self.stage)

    def _phi(self, s: Sense, k: int) -> float:
        c = self.cfg
        if k == APPROACH:
            dxy = float(np.linalg.norm(s.base_xy - s.park_grasp[:2]))
            ang = abs(wrap(s.base_yaw - s.park_grasp[2])) / math.pi
            p = 0.7 * (1 - math.tanh(dxy / c.d_nav)) + 0.3 * (1 - ang)
        elif k == REACH:
            dh = max(0.0, float(np.linalg.norm(s.hand_mid - s.grasp_pt)) - c.reach_offset)
            p = 1 - math.tanh(dh / c.d_reach)
        elif k == GRASP:
            p = min(1.0, (s.f_l + s.f_r) / (2 * c.f_hold))
        elif k == LIFT:
            z0 = self._z0 if self._z0 is not None else s.obj_pos[2]
            p = float(np.clip((s.obj_pos[2] - z0) / c.lift_h, 0.0, 1.0))
        elif k == TRANSPORT:
            dxy = float(np.linalg.norm(s.base_xy - s.park_place[:2]))
            ang = abs(wrap(s.base_yaw - s.park_place[2])) / math.pi
            p = 0.7 * (1 - math.tanh(dxy / c.d_carry)) + 0.3 * (1 - ang)
        else:
            p = 1 - math.tanh(float(np.linalg.norm(s.obj_pos - s.region_center)) / c.d_place)
        return k + float(np.clip(p, 0.0, 1.0))

    # ------------------------------------------------------------ step
    def step(self, s: Sense):
        c = self.cfg
        if self._z0 is None:
            self._z0 = float(s.obj_pos[2])
            # Warm-started mid-task (curriculum phase B/C): the carried height was reached before
            # this episode began, so seed the peak now -- otherwise _z_peak stays -inf and the drop
            # detector is dead for the whole episode.
            if self.lat["lift"]:
                self._z_peak = float(s.obj_pos[2])
        info = {"events": [], "stage": self.stage}
        r_evt = 0.0
        r_pen = 0.0

        firm = s.f_l >= c.f_firm and s.f_r >= c.f_firm
        held = s.f_l >= c.f_release and s.f_r >= c.f_release
        off_rack = s.obj_rack_force < c.free_of_rack_f
        lifted_now = firm and off_rack and s.obj_pos[2] > self._z0 + c.lift_dz

        # ---- sustain timers
        self._hold_grasp = self._hold_grasp + s.dt if firm else 0.0
        released = (s.f_l < c.f_release and s.f_r < c.f_release)
        supported = s.obj_shelf_force > c.free_of_rack_f
        place_ok = self.lat["lift"] and s.in_region and released and supported
        self._hold_place = self._hold_place + s.dt if place_ok else 0.0

        # "lost" must mean UNSUPPORTED AND FALLING, not merely "out of hand".  A real set-down
        # releases one hand first and lets the pillar slide down the shelf edge for ~1.6 s with
        # intermittent shelf contact -- measured on seed 17, where a placement the expert's own
        # gate called PASS was scored as a drop.  Contact with the shelf/rack, or a stationary z,
        # both mean it is resting on something, not falling.
        supported_now = (s.obj_shelf_force + s.obj_rack_force) > c.free_of_rack_f
        falling = s.obj_vz < -c.drop_vz
        lost = (self.lat["lift"] and not held and not s.in_region
                and s.obj_pos[2] < self._z_peak - c.drop_dz
                and not supported_now and falling)
        self._lost = self._lost + s.dt if lost else 0.0

        if held and s.obj_pos[2] > self._z_peak:
            self._z_peak = float(s.obj_pos[2])

        # ---- ordered, latched event bonuses
        if not self.lat["touch"] and s.touch_l and s.touch_r:
            self.lat["touch"] = True; r_evt += c.r_touch; info["events"].append("touch")
        if self.lat["touch"] and not self.lat["grasp"] and self._hold_grasp >= c.hold_grasp_s:
            self.lat["grasp"] = True; r_evt += c.r_grasp; info["events"].append("grasp")
        if self.lat["grasp"] and not self.lat["lift"] and lifted_now:
            self.lat["lift"] = True; r_evt += c.r_lift; info["events"].append("lift")
            self._z_peak = float(s.obj_pos[2])
        if self.lat["lift"] and not self.lat["arrive"] and held \
                and float(np.linalg.norm(s.base_xy - s.park_place[:2])) < c.arrive_tol_xy:
            self.lat["arrive"] = True; r_evt += c.r_arrive; info["events"].append("arrive")
        if self.lat["arrive"] and not self.lat["place"] and self._hold_place >= c.hold_place_s:
            self.lat["place"] = True; r_evt += c.r_place; info["events"].append("place")
            self.done = True; self.term = "success"

        # ---- penalties
        self._yaw_used += abs(s.d_yaw)
        r_pen -= c.k_yaw * abs(s.d_yaw)
        r_pen -= c.k_time
        r_pen -= c.k_jclip * s.n_joint_clip
        r_pen -= c.k_energy * (s.v_fwd ** 2 + s.wz ** 2)
        if self._prev_a is not None:
            r_pen -= c.k_smooth * float(np.sum((np.asarray(s.a_arm) - self._prev_a) ** 2))
        self._prev_a = np.asarray(s.a_arm, dtype=float).copy()
        if s.body_env_force > c.f_collide:
            r_pen -= c.p_collide

        # ---- terminations (checked after success so success wins ties)
        if not self.done:
            if s.body_env_force > c.f_collide_hard:
                r_pen -= c.p_collide_hard; self.done = True; self.term = "hard_collision"
            elif self._yaw_used > c.yaw_budget:
                r_pen -= c.p_spin; self.done = True; self.term = "spin"
            # a controlled set-down also lowers the object below its carried peak, so "drop"
            # means losing it ANYWHERE BUT the goal region -- otherwise placing self-terminates.
            elif self.lat["lift"] and (
                    self._lost >= c.drop_grace_s
                    or (self._lost > 0 and s.obj_pos[2] < s.region_center[2] - c.drop_floor_dz)):
                r_pen -= c.p_drop; self.done = True; self.term = "drop"
            elif not self.lat["grasp"] and (s.obj_tilt > c.knock_tilt
                                            or s.obj_pos[2] < self._z0 - c.drop_dz):
                r_pen -= c.p_knock; self.done = True; self.term = "knock"
            elif s.t + 1 >= c.max_steps:
                self.done = True; self.term = "timeout"

        # ---- potential-based shaping (computed last: stage may have advanced this step)
        self.stage = self._stage_of(s)
        phi = self._phi(s, self.stage)
        r_shape = 0.0 if self._prev_phi is None else c.lam_shape * (c.gamma_shape * phi - self._prev_phi)
        self._prev_phi = phi

        r = r_shape + r_evt + r_pen
        self.ret += r
        info.update(stage=self.stage, stage_name=STAGE_NAMES[self.stage], phi=phi,
                    r_shape=r_shape, r_event=r_evt, r_pen=r_pen, r=r,
                    yaw_used=self._yaw_used, latches=dict(self.lat),
                    done=self.done, term=self.term, ret=self.ret)
        return r, info


# ---------------------------------------------------------------- MuJoCo adapter
class MujocoSensor:
    """Fills a Sense from a live MuJoCo (m, d). Kept separate so the logic stays testable."""

    def __init__(self, m, d, ct, obj_bodies, obj_geoms, padg, region_center,
                 park_grasp, park_place, h_grip=0.0,
                 rack_prefix="rack", shelf_prefix="cart"):
        import mujoco
        self.mj = mujoco
        self.m, self.d, self.ct = m, d, ct
        self.bodies = list(obj_bodies)
        self.objg = set(obj_geoms)
        self.padg = {k: set(v) for k, v in padg.items()}
        self.allpad = self.padg["l"] | self.padg["r"]
        self.region_center = np.asarray(region_center, float)
        self.park_grasp = np.asarray(park_grasp, float)
        self.park_place = np.asarray(park_place, float)
        self.h_grip = h_grip
        self._ft = np.zeros(6)
        self.rackg, self.shelfg = set(), set()
        for g in range(m.ngeom):
            n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
            b = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").lower()
            tag = n + "|" + b
            if rack_prefix in tag:
                self.rackg.add(g)
            if shelf_prefix in tag:
                self.shelfg.add(g)
        self.robotg = {g for g in range(m.ngeom)
                       if m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[ct.L.mount]}
        self._prev_yaw = None
        self._prev_z = None
        self.t = 0

    def _pair_force(self, ga: set, gb: set) -> float:
        tot = 0.0
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact[i].geom1, self.d.contact[i].geom2
            if (g1 in ga and g2 in gb) or (g2 in ga and g1 in gb):
                self.mj.mj_contactForce(self.m, self.d, i, self._ft)
                tot += abs(self._ft[0])
        return tot

    def _touch(self, k: str) -> bool:
        pads = set()
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact[i].geom1, self.d.contact[i].geom2
            if g1 in self.padg[k] and g2 in self.objg:
                pads.add(g1)
            elif g2 in self.padg[k] and g1 in self.objg:
                pads.add(g2)
        return len(pads) >= 2

    def reset(self):
        self._prev_yaw = None
        self._prev_z = None
        self.t = 0

    def read(self, a_arm=None, n_joint_clip=0, dt=0.272, max_body_env_force=None) -> Sense:
        d, m, ct = self.d, self.m, self.ct
        x, y, yaw = ct.base_pose()
        dyaw = 0.0 if self._prev_yaw is None else abs(wrap(yaw - self._prev_yaw))
        self._prev_yaw = yaw
        op = np.mean([d.xpos[b] for b in self.bodies], axis=0)
        vz = 0.0 if self._prev_z is None else float((op[2] - self._prev_z) / max(dt, 1e-6))
        self._prev_z = float(op[2])
        zax = d.xmat[self.bodies[0]].reshape(3, 3)[:, 2]
        tilt = float(np.arccos(np.clip(abs(zax[2]), -1, 1)))
        hm = 0.5 * (d.xpos[ct.L.mount] + d.xpos[ct.R.mount])
        v = ct.base_velocity()
        nonpad = self.robotg - self.allpad
        s = Sense(
            t=self.t, dt=dt,
            hand_mid=np.asarray(hm, float).copy(),
            grasp_pt=op + np.array([0.0, 0.0, self.h_grip]),
            obj_pos=np.asarray(op, float).copy(), obj_tilt=tilt, obj_vz=vz,
            base_xy=np.array([x, y]), base_yaw=float(yaw), d_yaw=dyaw,
            v_fwd=float(v[0]), wz=float(v[1]),
            f_l=self._pair_force(self.padg["l"], self.objg),
            f_r=self._pair_force(self.padg["r"], self.objg),
            touch_l=self._touch("l"), touch_r=self._touch("r"),
            obj_rack_force=self._pair_force(self.objg, self.rackg),
            obj_shelf_force=self._pair_force(self.objg, self.shelfg),
            in_region=False,
            # collision force must be the MAX over the decision interval: point-sampling at the
            # boundary missed a >200 N spike on seed 13 and flipped its verdict when the sampling
            # cadence changed.
            body_env_force=(self._pair_force(nonpad, set(range(m.ngeom)) - self.robotg)
                            if max_body_env_force is None else float(max_body_env_force)),
            n_joint_clip=int(n_joint_clip),
            a_arm=np.zeros(14) if a_arm is None else np.asarray(a_arm, float),
            park_grasp=self.park_grasp, park_place=self.park_place,
            region_center=self.region_center,
        )
        self.t += 1
        return s
