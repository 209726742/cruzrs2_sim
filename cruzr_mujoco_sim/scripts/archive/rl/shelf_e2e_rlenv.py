#!/usr/bin/env python3
"""Online-RL environment for the CRUZR S2 pillar pick-and-place.

Same observation/action contract as shelf_e2e_rollout.py (so a BC checkpoint can be dropped in
as the initial policy), plus the reward of shelf_e2e_reward.py and curriculum resets from the
expert snapshots produced by shelf_e2e_rlhook (E2E_RLHOOK=1 run of shelf_e2e_expert.py).

  obs     : {"image": head_stereo_l_shelf, "left_wrist": chassis_front,
             "right_wrist": hand_right_shelf, "state": state22}   -- 224x224, pure vision
  action  : 18 = 14 arm joint targets + 2 gripper cmds + (v_fwd, wz)
  reward  : PillarReward (privileged measurements, reward-only)
  reset   : phase="A" -> from the pre_grasp snapshot (reach+grasp+lift only)
            phase="B" -> from the post_lift snapshot (transport+place)
            phase="C" -> from the randomized start state (full task)

One MuJoCo model per PROCESS (cruzr_teleop is a module-level singleton keyed by TELEOP_SCENE_XML),
so vectorized training must use subprocesses -- one env each.

Env vars: SEED (required), RL_SNAP_DIR (default out/rl/snap), E2E_SCENE_DIR.
"""
import importlib.util
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PROMPT = "pick up the steel pillar from the rack in front and place it on the second shelf of the cart"
CAMS = ["head_stereo_l_shelf", "chassis_front", "hand_right_shelf"]
# BC contract (shelf_e2e_rollout.py): ONE chunk action per 2 control steps, re-infer every 8
# actions. So a decision = 8 actions x 2 control steps = 16 control steps = 0.272 s. Holding a
# single action for 8 control steps instead would be a different (coarser) controller than the one
# the BC checkpoint was trained for.
ACT_PER_DECISION = int(os.environ.get("RL_REPLAN", "8"))
SUBSTEPS_PER_ACT = 2
PHASE_STAGE = {"A": "pre_grasp", "B": "post_lift", "C": None}
PHASE_STEPS = {"A": 200, "B": 250, "C": 600}     # expert full task = 419 decisions


def _build_scene(seed):
    """Byte-identical layout randomization to shelf_e2e_expert.py -- keep in sync."""
    rng = np.random.default_rng(seed)
    cart_nom, obj_nom = np.array([-2.40, 0.0]), np.array([0.58, 0.0])
    cart_xy = cart_nom + np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.30, 0.30)])
    obj_xy = obj_nom + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.30, 0.30)])
    robot0 = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])
    scene_dir = os.environ.get("E2E_SCENE_DIR", os.path.join(ROOT, "assets"))
    os.makedirs(scene_dir, exist_ok=True)
    tmpl = open(os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")).read()
    tmpl = re.sub(r'(<body name="shelf_cart" pos=")[^"]*(")',
                  lambda mo: f'{mo.group(1)}{cart_xy[0]:.6f} {cart_xy[1]:.6f} 0.800000{mo.group(2)}', tmpl)
    path = os.path.join(scene_dir, f"e2e_rl_scene_{seed}.xml")
    open(path, "w").write(tmpl)
    return path, cart_xy, obj_xy, robot0, rng


class PillarRLEnv:
    def __init__(self, seed=None, snap_dir=None, render=True):
        self.seed = int(seed if seed is not None else os.environ["SEED"])
        scene, self.cart_xy, self.obj_xy, self.robot0, self.rng = _build_scene(self.seed)
        os.environ["TELEOP_SCENE_XML"] = scene
        os.environ.setdefault("TELEOP_HOME", "droop")
        os.environ.setdefault("MUJOCO_GL", "egl")
        spec = importlib.util.spec_from_file_location("cruzr_teleop", os.path.join(HERE, "cruzr_teleop.py"))
        self.ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ct)
        import mujoco
        import shelf_e2e_reward as R
        self.mj, self.R = mujoco, R
        ct = self.ct
        self.m, self.d = ct.m, ct.d
        m, d = self.m, self.d
        self.SUB = int(getattr(ct, "CONTROL_SUBSTEPS", 17))
        self.dt = ACT_PER_DECISION * SUBSTEPS_PER_ACT * self.SUB * float(m.opt.timestep)

        self.bodies = [i for i in range(m.nbody)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("_")[0] == "pillar"]
        self.OB = self.bodies[0]
        self.objg = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bodies}
        self.padg = {k: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in gs]
                     for k, gs in (("r", ("R_pad1", "R_pad2")), ("l", ("L_pad1", "L_pad2")))}
        cd = self.cart_xy - np.array([-2.40, 0.0])
        self.REGION = dict(x=(-2.73 + cd[0], -2.07 + cd[0]), y=(-0.91 + cd[1], 0.92 + cd[1]), z=0.903)
        region_center = [0.5 * sum(self.REGION["x"]), 0.5 * sum(self.REGION["y"]), self.REGION["z"]]

        sd = snap_dir or os.environ.get("RL_SNAP_DIR", os.path.join(ROOT, "out", "rl", "snap"))
        self.snap = np.load(os.path.join(sd, f"snap_{self.seed:06d}.npz"), allow_pickle=True)
        self.cfg = R.Cfg(reach_offset=float(self.snap["reach_offset"][0]))
        self.sensor = R.MujocoSensor(m, d, ct, self.bodies, self.objg, self.padg, region_center,
                                     self.snap["park_grasp"], self.snap["park_place"])
        self.reward = R.PillarReward(self.cfg)
        self.rig = _CamRig(mujoco, m, d) if render else None
        self._q0 = None

    # ------------------------------------------------------------------ helpers
    def _obj_extent(self):
        ps = np.array([self.d.geom_xpos[g] for g in self.objg])
        return ps.min(0), ps.max(0)

    def in_region(self):
        lo, hi = self._obj_extent()
        op = np.mean([self.d.xpos[b] for b in self.bodies], axis=0)
        Rg = self.REGION
        return (Rg["x"][0] - 0.04 <= lo[0] and hi[0] <= Rg["x"][1] + 0.04 and
                Rg["y"][0] - 0.04 <= lo[1] and hi[1] <= Rg["y"][1] + 0.04 and
                abs(op[2] - Rg["z"]) < 0.10)

    def _grip_frac(self, arm):
        q = float(np.mean([self.d.qpos[a] for a in arm.grip_qadr]))
        return float(np.clip(1.0 - (q - self.ct.GRIP_OPEN) / (self.ct.GRIP_CLOSE - self.ct.GRIP_OPEN), 0, 1))

    def state22(self):
        ct, d = self.ct, self.d
        qL = [float(d.qpos[a]) for a in ct.L.qadr]
        qR = [float(d.qpos[a]) for a in ct.R.qadr]
        x, y, yaw = ct.base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        rel = lambda t: [c * (t[0] - x) + s * (t[1] - y), -s * (t[0] - x) + c * (t[1] - y)]  # noqa: E731
        v = ct.base_velocity()
        return np.array(qL + qR + [self._grip_frac(ct.L), self._grip_frac(ct.R)]
                        + rel(self.obj_xy) + rel(self.cart_xy) + [float(v[0]), float(v[1])], np.float32)

    def obs(self):
        o = {"state": self.state22(), "prompt": PROMPT}
        if self.rig is not None:
            o["image"] = self.rig.shot("head_stereo_l_shelf")
            o["left_wrist"] = self.rig.shot("chassis_front")
            o["right_wrist"] = self.rig.shot("hand_right_shelf")
        return o

    # ------------------------------------------------------------------ reset
    def _fresh_start(self):
        m, d, ct = self.m, self.d, self.ct
        self.mj.mj_resetData(m, d)
        oq = m.jnt_qposadr[m.body_jntadr[self.OB]]
        d.qpos[oq + 0] += self.obj_xy[0] - 0.58
        d.qpos[oq + 1] += self.obj_xy[1] - 0.0
        for i, adr in enumerate(ct.BQ):
            d.qpos[adr] = self.robot0[i]
        ct.base_tgt[:] = self.robot0
        self.mj.mj_forward(m, d)
        for _ in range(30):
            ct.control_step(self.SUB)

    def _restore(self, anchor):
        m, d, ct = self.m, self.d, self.ct
        d.qpos[:] = self.snap[f"{anchor}.qpos"]
        d.qvel[:] = self.snap[f"{anchor}.qvel"]
        ct.qtgt["l"][:] = self.snap[f"{anchor}.qtgt_l"]
        ct.qtgt["r"][:] = self.snap[f"{anchor}.qtgt_r"]
        ct.grip_cmd["l"] = float(self.snap[f"{anchor}.grip_l"][0])
        ct.grip_cmd["r"] = float(self.snap[f"{anchor}.grip_r"][0])
        ct.base_tgt[:] = self.snap[f"{anchor}.base_tgt"]
        ct.base_vel[:] = 0.0
        self.mj.mj_forward(m, d)

    def reset(self, phase="C", jitter=0.0):
        anchor = PHASE_STAGE[phase]
        if anchor is None:
            self._fresh_start()
            stage = self.R.APPROACH
        else:
            self._restore(anchor)
            stage = {"pre_grasp": self.R.REACH, "post_lift": self.R.TRANSPORT}[anchor]
            if jitter > 0:                      # widen the reset distribution around the anchor
                for adr in self.ct.BQ:
                    self.d.qpos[adr] += self.rng.normal(0, jitter)
                self.mj.mj_forward(self.m, self.d)
        for li in range(self.m.nlight):         # per-episode visual domain randomization
            self.m.light_diffuse[li] = np.clip(
                self.m.light_diffuse[li] * self.rng.uniform(0.8, 1.2), 0.05, 1.0)
        self.cfg.max_steps = PHASE_STEPS[phase]
        self.sensor.reset()
        self.reward.reset(stage=stage)
        self._prev_a = None
        return self.obs()

    # ------------------------------------------------------------------ step
    def apply(self, a):
        ct = self.ct
        a = np.asarray(a, float)
        lo = np.concatenate([ct.L.lo, ct.R.lo]); hi = np.concatenate([ct.L.hi, ct.R.hi])
        n_clip = int(np.sum((a[:14] < lo) | (a[:14] > hi)))
        ct.qtgt["l"][:] = np.clip(a[0:7], ct.L.lo, ct.L.hi)
        ct.qtgt["r"][:] = np.clip(a[7:14], ct.R.lo, ct.R.hi)
        ct.grip_cmd["l"] = ct.GRIP_OPEN + (1 - float(np.clip(a[14], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
        ct.grip_cmd["r"] = ct.GRIP_OPEN + (1 - float(np.clip(a[15], 0, 1))) * (ct.GRIP_CLOSE - ct.GRIP_OPEN)
        ct.base_vel[:] = [float(np.clip(a[16], -0.4, 0.4)), float(np.clip(a[17], -0.6, 0.6))]
        return n_clip

    def step(self, chunk):
        """chunk: (>=ACT_PER_DECISION, 18). Applied one action per 2 control steps, BC-style."""
        chunk = np.asarray(chunk, float)
        assert chunk.ndim == 2 and chunk.shape[0] >= ACT_PER_DECISION, chunk.shape
        n_clip, fmax = 0, 0.0
        nonpad = self.sensor.robotg - self.sensor.allpad
        world = set(range(self.m.ngeom)) - self.sensor.robotg
        for j in range(ACT_PER_DECISION):
            n_clip += self.apply(chunk[j])
            for _ in range(SUBSTEPS_PER_ACT):
                self.ct.control_step(self.SUB)
            fmax = max(fmax, self.sensor._pair_force(nonpad, world))
        s = self.sensor.read(a_arm=chunk[ACT_PER_DECISION - 1, :14], n_joint_clip=n_clip,
                             dt=self.dt, max_body_env_force=fmax)
        s.in_region = self.in_region()
        r, info = self.reward.step(s)
        return self.obs(), r, self.reward.done, info


class _CamRig:
    """224x224 square, exactly as the BC data was recorded (480x640 skews aspect -> OOD images)."""

    def __init__(self, mujoco, m, d):
        self.mj, self.m, self.d = mujoco, m, d
        self.renderer = mujoco.Renderer(m, 224, 224)
        self.opt = mujoco.MjvOption()
        self.ids = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMS}

    def shot(self, name):
        cam = self.mj.MjvCamera()
        cam.type = self.mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.ids[name]
        self.renderer.update_scene(self.d, cam, self.opt)
        return self.renderer.render().copy()
