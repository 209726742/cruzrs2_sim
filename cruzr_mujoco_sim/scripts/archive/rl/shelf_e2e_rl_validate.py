#!/usr/bin/env python3
"""In-sim validation of the RL reward (design doc section 6, cases B/C/D) on the REAL environment.

Validation A (a competent expert must score high) runs inside the expert itself via E2E_RLHOOK=1.
Here we drive PillarRLEnv with scripted degenerate agents and check the reward reacts correctly:

  spin      : the exact BC failure mode -- rotate in place forever      -> term=spin, return << 0
  idle      : hold still                                                -> term=timeout, small < 0
  charge    : drive straight at the rack at full speed                  -> must not pay any bonus
  hold_open : from post_lift, open the grippers immediately (throw it)  -> term=drop

Env: SEED (default 1), needs out/rl/snap/snap_<seed>.npz from an E2E_RLHOOK=1 expert run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

import shelf_e2e_rlenv as E  # noqa: E402

SEED = int(os.environ.get("SEED", "1"))


def hold_action(env):
    """No-op action = keep the current servo targets, zero base velocity."""
    ct = env.ct
    a = np.zeros(18)
    a[0:7] = ct.qtgt["l"]
    a[7:14] = ct.qtgt["r"]
    gf = lambda k, A: 1.0 - (ct.grip_cmd[k] - ct.GRIP_OPEN) / (ct.GRIP_CLOSE - ct.GRIP_OPEN)  # noqa: E731
    a[14] = np.clip(gf("l", ct.L), 0, 1)
    a[15] = np.clip(gf("r", ct.R), 0, 1)
    return a


def run(env, phase, policy, cap=None):
    env.reset(phase)
    n = cap or E.PHASE_STEPS[phase]
    for t in range(n):
        a = policy(env, t)
        chunk = np.tile(np.asarray(a, float), (E.ACT_PER_DECISION, 1))   # hold across the decision
        _, r, done, info = env.step(chunk)
        if done:
            break
    return env.reward


def main():
    env = E.PillarRLEnv(seed=SEED, render=False)   # no rendering: reward validation only
    bad = 0

    def check(name, rw, want_term, cond, detail=""):
        nonlocal bad
        ok = (rw.term == want_term) and cond
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:10s} term={rw.term:14s} return={rw.ret:+8.2f} "
              f"latches={''.join(k[0].upper() if v else '-' for k, v in rw.lat.items())} {detail}",
              flush=True)

    rw = run(env, "C", lambda e, t: np.concatenate([hold_action(e)[:16], [0.0, 0.6]]))
    check("spin", rw, "spin", rw.ret < -2.0)

    rw = run(env, "C", lambda e, t: hold_action(e))
    check("idle", rw, "timeout", -5.0 < rw.ret < 0.0)

    rw = run(env, "C", lambda e, t: np.concatenate([hold_action(e)[:16], [0.4, 0.0]]))
    check("charge", rw, rw.term, not rw.lat["grasp"] and not rw.lat["place"],
          "(no grasp/place bonus for ramming the rack)")

    def throw(e, t):
        a = hold_action(e)
        a[14] = a[15] = 1.0        # gripper channel is OPEN-fraction: 1 = wide open, 0 = closed
        return a
    rw = run(env, "B", throw)
    check("hold_open", rw, "drop", rw.ret < 0)

    print(f"\n{4 - bad}/4 in-sim checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
