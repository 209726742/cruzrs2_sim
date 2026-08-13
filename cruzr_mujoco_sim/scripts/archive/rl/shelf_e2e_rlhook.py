#!/usr/bin/env python3
"""Opt-in hook that rides along inside shelf_e2e_expert.py (env E2E_RLHOOK=1).

Two jobs, both needed before any RL run can start:
  1. SNAPSHOT  -- dump full sim state (qpos/qvel/servo targets) at the curriculum reset anchors
     pre_grasp / post_lift / pre_place, plus the per-seed constants the reward needs
     (park poses, region centre, rest height, grasp-time hand<->object offset).
  2. SCORE     -- run the reward function live on the expert's own trajectory. This is design-doc
     section 6 validation A: a competent expert must score high in the REAL sim, not just in unit tests.

The expert is untouched except for a guarded import + four one-line calls.
"""
import os

import numpy as np

import shelf_e2e_reward as R

ANCHORS = ("pre_grasp", "post_lift", "pre_place")


class Hook:
    def __init__(self, ct, mujoco, m, d, bodies, obj_geoms, padg, region_center,
                 park_grasp, park_place, out_npz, replan=16, dt=0.272):
        self.ct, self.mj, self.m, self.d = ct, mujoco, m, d
        self.out = out_npz
        self.replan = replan
        self.snaps = {}
        self.const = dict(park_grasp=np.asarray(park_grasp, float),
                          park_place=np.asarray(park_place, float),
                          region_center=np.asarray(region_center, float))
        self.sensor = R.MujocoSensor(m, d, ct, bodies, obj_geoms, padg, region_center,
                                     park_grasp, park_place)
        self.reward = R.PillarReward(R.Cfg(max_steps=10 ** 9))   # scoring only: never time out
        self._n = 0
        self.log = []
        self.trace = []
        print(f"[rlhook] geom sets: obj={len(self.sensor.objg)} rack={len(self.sensor.rackg)} "
              f"shelf={len(self.sensor.shelfg)} robot={len(self.sensor.robotg)} "
              f"pads={len(self.sensor.allpad)}", flush=True)

    # -------------------------------------------------- 1. snapshots
    def snap(self, name):
        d, ct = self.d, self.ct
        self.snaps[name] = dict(
            qpos=d.qpos.copy(), qvel=d.qvel.copy(),
            qtgt_l=ct.qtgt["l"].copy(), qtgt_r=ct.qtgt["r"].copy(),
            grip_l=np.array([ct.grip_cmd["l"]]), grip_r=np.array([ct.grip_cmd["r"]]),
            base_tgt=np.asarray(ct.base_tgt, float).copy(),
            t=np.array([self._n // self.replan]),   # decision step -> sizes the curriculum horizons
        )
        print(f"[rlhook] snapshot {name} @ base={ct.base_pose()}", flush=True)

    def note_grasp_offset(self):
        """Hand<->object distance at the moment the expert has a firm grip -> reach potential offset."""
        d, ct = self.d, self.ct
        hm = 0.5 * (d.xpos[ct.L.mount] + d.xpos[ct.R.mount])
        op = np.mean([d.xpos[b] for b in self.sensor.bodies], axis=0)
        self.const["reach_offset"] = np.array([float(np.linalg.norm(hm - op))])
        print(f"[rlhook] reach_offset={self.const['reach_offset'][0]:.3f} m", flush=True)

    # -------------------------------------------------- 2. live scoring
    def tick(self):
        """Called once per control step from the expert's frames(); scores every `replan` steps."""
        self._n += 1
        if self._n % self.replan:
            return
        s = self.sensor.read()
        s.in_region = self.in_region()
        self.trace.append([s.t, s.f_l, s.f_r, s.obj_pos[2], float(s.in_region),
                           s.obj_shelf_force, s.obj_rack_force, s.body_env_force, s.d_yaw])
        if self.reward.done:      # keep tracing past termination -- needed to debug the reward itself
            return
        _, info = self.reward.step(s)
        if info["events"]:
            print(f"[rlhook] t={s.t} {info['events']} stage={info['stage_name']} "
                  f"ret={info['ret']:+.2f}", flush=True)
        self.log.append((s.t, info["stage"], info["r"], info["ret"]))

    in_region = staticmethod(lambda: False)   # replaced by the expert's own in_region

    # -------------------------------------------------- finish
    def finalize(self, success):
        if "reach_offset" not in self.const:
            self.const["reach_offset"] = np.array([0.20])
        z0 = self.reward._z0 if self.reward._z0 is not None else 0.0
        payload = {f"{a}.{k}": v for a in self.snaps for k, v in self.snaps[a].items()}
        payload.update(self.const)
        payload["z0"] = np.array([z0])
        payload["expert_return"] = np.array([self.reward.ret])
        payload["expert_success"] = np.array([1.0 if success else 0.0])
        payload["anchors"] = np.array(sorted(self.snaps), dtype=object)
        payload["trace"] = np.array(self.trace, dtype=np.float32)  # t,fL,fR,z,inreg,fshelf,frack,fenv,dyaw
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        np.savez(self.out, **payload)
        print(f"[rlhook] SCORE expert_return={self.reward.ret:+.2f} "
              f"term={self.reward.term or 'n/a'} latches={self.reward.lat} "
              f"snaps={sorted(self.snaps)} -> {self.out}", flush=True)
