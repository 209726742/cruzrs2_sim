#!/usr/bin/env python3
"""Offline validation of the pillar RL reward (doc section 6).

Runs without MuJoCo: every scenario is a synthetic `Sense` stream, so each anti-hacking claim
in the design doc is pinned by an executable assertion. Run: python3 test_shelf_e2e_reward.py
"""
import math
import sys

import numpy as np

from shelf_e2e_reward import APPROACH, PLACE, Cfg, PillarReward, Sense

PG = np.array([1.0, 0.0, 0.0])       # grasp station
PP = np.array([-2.4, 0.0, math.pi])  # cart station
RC = np.array([-2.4, 0.0, 0.903])    # region centre
Z0 = 0.90                            # pillar rest height on the rack


def mk(**kw):
    s = Sense(park_grasp=PG, park_place=PP, region_center=RC)
    s.obj_pos = np.array([1.6, 0.0, Z0])
    s.grasp_pt = s.obj_pos.copy()
    s.hand_mid = np.array([3.0, 0.0, Z0])
    s.base_xy = np.array([3.0, 0.0])
    s.obj_rack_force = 20.0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def run(rw, senses):
    infos = []
    for t, s in enumerate(senses):
        s.t = t
        _, i = rw.step(s)
        infos.append(i)
        if rw.done:
            break
    return infos


def expert_stream(n_nav=40, n_reach=15, n_grip=6, n_lift=10, n_carry=60, n_place=20):
    """A competent, non-hacking rollout: drive up, reach, squeeze, lift, carry, set down, release."""
    out = []
    for i in range(n_nav):                                   # APPROACH: 3.0 -> 1.0 m
        u = (i + 1) / n_nav
        out.append(mk(base_xy=np.array([3.0 - 2.0 * u, 0.0]), d_yaw=0.02,
                      hand_mid=np.array([3.0 - 2.0 * u + 0.5, 0.0, Z0]), v_fwd=0.3))
    for i in range(n_reach):                                 # REACH: hand -> pillar
        u = (i + 1) / n_reach
        out.append(mk(base_xy=PG[:2].copy(), hand_mid=np.array([1.6 + 0.4 * (1 - u), 0.0, Z0])))
    for i in range(n_grip):                                  # GRASP: close, build force
        f = 3.0 * (i + 1) / n_grip
        out.append(mk(base_xy=PG[:2].copy(), hand_mid=np.array([1.6, 0.0, Z0]),
                      f_l=f, f_r=f, touch_l=True, touch_r=True))
    for i in range(n_lift):                                  # LIFT: off the rack
        u = (i + 1) / n_lift
        out.append(mk(base_xy=PG[:2].copy(), obj_pos=np.array([1.6, 0.0, Z0 + 0.12 * u]),
                      hand_mid=np.array([1.6, 0.0, Z0 + 0.12 * u]),
                      f_l=3.0, f_r=3.0, touch_l=True, touch_r=True, obj_rack_force=0.0))
    for i in range(n_carry):                                 # TRANSPORT
        u = (i + 1) / n_carry
        bx = 1.0 + (PP[0] - 1.0) * u
        out.append(mk(base_xy=np.array([bx, 0.0]), base_yaw=math.pi * u, d_yaw=math.pi / n_carry,
                      obj_pos=np.array([bx - 0.6, 0.0, Z0 + 0.12]),
                      f_l=3.0, f_r=3.0, touch_l=True, touch_r=True, obj_rack_force=0.0, v_fwd=0.3))
    for i in range(n_place):                                 # PLACE: set down, release, hold
        out.append(mk(base_xy=PP[:2].copy(), base_yaw=PP[2], obj_pos=RC.copy(),
                      obj_rack_force=0.0, obj_shelf_force=15.0, in_region=True,
                      f_l=0.0, f_r=0.0))
    return out


# ------------------------------------------------------------------ scenarios
def t_expert_high():
    rw = PillarReward()
    infos = run(rw, expert_stream())
    assert rw.term == "success", rw.term
    assert infos[-1]["latches"] == {k: True for k in infos[-1]["latches"]}
    assert rw.ret > 50, rw.ret
    return f"return={rw.ret:+.2f} term={rw.term} steps={len(infos)}"


def t_spin_strongly_negative():
    """The observed BC failure mode: yaw 6-55 rad of spinning, no task progress."""
    rw = PillarReward()
    infos = run(rw, [mk(d_yaw=0.15, wz=0.55) for _ in range(400)])
    assert rw.term == "spin", rw.term
    assert rw.ret < -2.0, rw.ret
    assert infos[-1]["stage"] == APPROACH
    assert infos[-1]["yaw_used"] > 4 * math.pi
    return f"return={rw.ret:+.2f} term={rw.term} steps={len(infos)} yaw={infos[-1]['yaw_used']:.1f}"


def t_idle_negative():
    rw = PillarReward()
    run(rw, [mk() for _ in range(400)])
    assert rw.term == "timeout"
    assert -3.0 < rw.ret < 0.0, rw.ret
    return f"return={rw.ret:+.2f} term={rw.term}"


def t_push_into_region_pays_nothing():
    """Hack: never grasp -- shove the pillar off the rack so it ends up inside the region."""
    rw = PillarReward()
    st = [mk() for _ in range(2)]                       # at rest on the rack (fixes z0 baseline)
    st += [mk(obj_pos=np.array([1.6, 0.0, Z0 - 0.2]), obj_tilt=0.0) for _ in range(3)]
    st += [mk(obj_pos=RC.copy(), in_region=True, obj_shelf_force=15.0, obj_rack_force=0.0)
           for _ in range(40)]
    run(rw, st)
    assert not rw.lat["lift"] and not rw.lat["place"], rw.lat
    assert rw.term == "knock", rw.term
    assert rw.ret < 0, rw.ret
    return f"return={rw.ret:+.2f} term={rw.term} place_latch={rw.lat['place']}"


def t_wedge_on_rack_is_not_a_lift():
    """Hack: pry the pillar upward while it still rests on the rack -> no lift bonus."""
    rw = PillarReward()
    st = [mk(f_l=3.0, f_r=3.0, touch_l=True, touch_r=True,
             obj_pos=np.array([1.6, 0.0, Z0 + 0.15]), obj_rack_force=30.0) for _ in range(40)]
    run(rw, st)
    assert rw.lat["grasp"] and not rw.lat["lift"], rw.lat
    return f"grasp={rw.lat['grasp']} lift={rw.lat['lift']} return={rw.ret:+.2f}"


def t_hover_in_region_without_release():
    """Hack: carry the pillar into the region and just hold it there -> no place bonus."""
    rw = PillarReward()
    st = expert_stream(n_place=0)
    st += [mk(base_xy=PP[:2].copy(), base_yaw=PP[2], obj_pos=RC.copy(), in_region=True,
              obj_rack_force=0.0, obj_shelf_force=0.0,
              f_l=3.0, f_r=3.0, touch_l=True, touch_r=True) for _ in range(60)]
    run(rw, st)
    assert rw.lat["arrive"] and not rw.lat["place"], rw.lat
    return f"arrive={rw.lat['arrive']} place={rw.lat['place']} return={rw.ret:+.2f}"


def t_cycle_shaping_is_not_positive():
    """Core anti-hacking property: a closed loop in state space accumulates <= 0 shaping."""
    rw = PillarReward()
    loop = []
    for _ in range(20):
        for x in list(np.linspace(3.0, 1.4, 25)) + list(np.linspace(1.4, 3.0, 25)):
            loop.append(mk(base_xy=np.array([x, 0.0]), hand_mid=np.array([x + 0.5, 0.0, Z0])))
    rw.cfg.max_steps = 10 ** 6
    infos = run(rw, loop)
    tot = sum(i["r_shape"] for i in infos)
    assert tot <= 1e-6, tot
    return f"shaping over 20 closed loops = {tot:+.4f} (<= 0 required)"


def t_yaw_penalty_not_refunded():
    """Turning back must not refund the spin penalty (deliberately path-dependent)."""
    rw = PillarReward()
    infos = run(rw, [mk(d_yaw=0.1) for _ in range(20)] + [mk(d_yaw=-0.1) for _ in range(20)])
    assert abs(infos[-1]["yaw_used"] - 4.0) < 1e-6, infos[-1]["yaw_used"]
    pen = sum(i["r_pen"] for i in infos)
    assert pen < -0.15 * 4.0 * 0.99, pen
    return f"yaw_used={infos[-1]['yaw_used']:.2f} rad, penalty={pen:+.3f}"


def t_grasp_needs_sustain():
    """A one-step force spike is not a grasp."""
    rw = PillarReward()
    run(rw, [mk(f_l=5, f_r=5, touch_l=True, touch_r=True), mk(), mk(), mk()])
    assert rw.lat["touch"] and not rw.lat["grasp"], rw.lat
    return f"touch={rw.lat['touch']} grasp={rw.lat['grasp']}"


def t_drop_terminates():
    rw = PillarReward()
    st = expert_stream(n_carry=10, n_place=0)
    st += [mk(base_xy=np.array([0.0, 0.0]), obj_pos=np.array([0.0, 0.0, Z0 - 0.5]),
              obj_rack_force=0.0, obj_vz=-1.2) for _ in range(5)]
    run(rw, st)
    assert rw.term == "drop", rw.term
    return f"term={rw.term} return={rw.ret:+.2f}"


def t_stage_monotone():
    rw = PillarReward()
    infos = run(rw, expert_stream())
    ks = [i["stage"] for i in infos]
    assert all(b >= a for a, b in zip(ks, ks[1:])), ks
    assert max(ks) == PLACE
    return f"stages {ks[0]} -> {ks[-1]}, monotone over {len(ks)} steps"


def t_curriculum_warm_start():
    """Phase A resets mid-task: earlier latches are pre-set so their bonuses cannot be re-farmed."""
    rw = PillarReward()
    rw.reset(stage=2)
    assert rw.lat["touch"] and not rw.lat["grasp"]
    run(rw, expert_stream(n_nav=0, n_reach=0))
    assert rw.term == "success"
    assert rw.ret < PillarReward().ret + 100
    return f"warm-start@GRASP -> {rw.term}, return={rw.ret:+.2f} (no touch bonus re-paid)"


def t_warm_start_drop_detector_alive():
    """Regression: warm-starting at TRANSPORT must still be able to detect a drop.

    _z_peak used to be seeded only while the object was held, so a phase-B episode that released
    on step 1 never had a peak and the drop detector stayed dead for the whole episode."""
    rw = PillarReward()
    rw.reset(stage=4)                                   # TRANSPORT: lift already latched
    carried = Z0 + 0.12
    st = [mk(obj_pos=np.array([0.5, 0.0, carried]), f_l=3.0, f_r=3.0,
             touch_l=True, touch_r=True, obj_rack_force=0.0)]
    st += [mk(obj_pos=np.array([0.5, 0.0, carried - 0.4]), obj_rack_force=0.0, obj_vz=-1.5)
           for _ in range(20)]
    run(rw, st)
    assert rw.term == "drop", rw.term
    return f"term={rw.term} return={rw.ret:+.2f}"


def t_controlled_setdown_is_not_a_drop():
    """Regression from seed 17: a real set-down releases one hand first and lets the pillar slide
    down the shelf edge for ~1.6 s, out of region, below its carried peak, with only intermittent
    shelf contact. The expert's own `placed` gate called it PASS; the reward called it a drop."""
    rw = PillarReward()
    rw.reset(stage=4)                                     # TRANSPORT, lift latched
    peak = Z0 + 0.21
    st = [mk(base_xy=PP[:2].copy(), base_yaw=PP[2], obj_pos=np.array([-2.4, 0.0, peak]),
             f_l=13.0, f_r=13.0, touch_l=True, touch_r=True, obj_rack_force=0.0)]
    # left hand releases; pillar rests on the shelf edge, z almost static, contact chattering
    for i, (fs, z) in enumerate([(0.0, 1.029), (0.0, 1.016), (12.1, 1.016),
                                 (9.1, 1.015), (0.0, 1.015), (0.0, 1.014)]):
        st.append(mk(base_xy=PP[:2].copy(), base_yaw=PP[2], obj_pos=np.array([-2.4, 0.0, z]),
                     f_l=0.0, f_r=6.0, obj_vz=-0.005, obj_shelf_force=fs, obj_rack_force=0.0))
    st += [mk(base_xy=PP[:2].copy(), base_yaw=PP[2], obj_pos=RC.copy(), in_region=True,
              obj_shelf_force=7.8, obj_rack_force=0.0, f_l=0.0, f_r=0.0, obj_vz=0.0)
           for _ in range(20)]
    run(rw, st)
    assert rw.term == "success", rw.term
    return f"term={rw.term} return={rw.ret:+.2f} (set-down survives, place awarded)"


TESTS = [t_expert_high, t_spin_strongly_negative, t_idle_negative,
         t_push_into_region_pays_nothing, t_wedge_on_rack_is_not_a_lift,
         t_hover_in_region_without_release, t_cycle_shaping_is_not_positive,
         t_yaw_penalty_not_refunded, t_grasp_needs_sustain, t_drop_terminates,
         t_stage_monotone, t_curriculum_warm_start, t_warm_start_drop_detector_alive,
         t_controlled_setdown_is_not_a_drop]

if __name__ == "__main__":
    bad = 0
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__:34s} {fn()}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {fn.__name__:34s} {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
