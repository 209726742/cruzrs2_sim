#!/usr/bin/env python3
"""PPO (Flow-SDE) online RL for pi0.5 on the CRUZR S2 pillar pick-and-place.

Design: docs/archive/立柱线Online_RL奖励设计_0820.md  (reward validated before this file was written)

Layout
  * env workers      -- one subprocess per env (MuJoCo + cruzr_teleop are per-process singletons).
                        Workers only step physics and render; they hold NO model weights.
  * central actor    -- one pi0.5 on the training GPU. Observations from all workers are batched
                        into a single Flow-SDE sample per decision step.
  * asymmetric critic-- small MLP over state22 + PRIVILEGED reward features. Legitimate in sim and
                        far cheaper than a vision critic; the actor stays pure-vision.
  * PPO              -- ratio from Flow-SDE path log-probs, GAE(0.95), clip 0.2, grads only through
                        the trainable (LoRA) leaves with per-denoising-step rematerialization.
  * BC anchor        -- the BC policy is base(frozen, shared) + LoRA_BC, so keeping a copy of just
                        the trainable leaves is enough to score the BC drift exactly, for a few MB.

Curriculum (phase gates are evaluation criteria -- see doc section 7):
  A: reset at pre_grasp  -> learn reach+grasp+lift        gate: grasp success >= 60%
  B: reset at post_lift  -> learn transport+place         gate: place success >= 50%
  C: full task, expert-state reset probability annealed 0.5 -> 0

Env: RL_PHASE=A|B|C, RL_ENVS, RL_ITERS, RL_OUT, RL_CKPT, RL_CONFIG, RL_SEEDS (csv or 'auto').
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHASE = os.environ.get("RL_PHASE", "A")
N_ENVS = int(os.environ.get("RL_ENVS", "8"))
N_ITERS = int(os.environ.get("RL_ITERS", "500"))
T_ROLL = int(os.environ.get("RL_T", "32"))          # decision steps per env per iteration
OUT = os.environ.get("RL_OUT", os.path.join(os.path.dirname(HERE), "out", f"rl_phase{PHASE}"))
CKPT = os.environ.get(
    "RL_CKPT", "/data1/hsr/openpi-main/checkpoints/pi05_cruzr_e2e_v2/cruzr_shelf_e2e_v2full/159999")
CFG_NAME = os.environ.get("RL_CONFIG", "pi05_cruzr_e2e_v2")
SNAP_DIR = os.environ.get("RL_SNAP_DIR", os.path.join(os.path.dirname(HERE), "out", "rl", "snap"))
EPISODES_PER_SEED = int(os.environ.get("RL_EPS_PER_SEED", "5"))


@dataclasses.dataclass
class PPOCfg:
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    minibatch: int = 4          # transitions per grad step (each is K transformer passes)
    lr_actor: float = 1e-5
    lr_critic: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0       # Flow-SDE entropy is a constant of sigma -> no gradient signal
    bc_coef: float = 0.02       # KL-to-BC anchor
    max_grad_norm: float = 1.0
    target_kl: float = 0.02     # early-stop an epoch if the policy moves too far
    sigma: float = 0.15
    denoise_steps: int = 5


# ===================================================================== env workers
# Envs run in a SEPARATE interpreter (mjx conda env): the scene needs MuJoCo 3.9, the openpi venv
# has an older one that rejects `actuatorfrcrange`. Rather than mutate the shared openpi venv, each
# env is its own OS process talking over a UNIX socket. See shelf_e2e_rl_worker.py.
MJX_PY = os.environ.get("RL_MJX_PY", "/data1/hsr/tools/miniconda3/envs/mjx/bin/python")


# ndarray codec: the two interpreters have different numpy majors (mjx=2.x, openpi venv=1.x) and
# numpy's own pickles are not portable across that boundary ("No module named numpy._core").
# Encode arrays as (tag, dtype-str, shape, raw bytes) instead.
_ND = "__nd__"


def enc(o):
    if isinstance(o, np.ndarray):
        return (_ND, o.dtype.str, o.shape, o.tobytes())
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(enc(v) for v in o)
    if isinstance(o, np.generic):
        return o.item()
    return o


def dec(o):
    if isinstance(o, tuple) and len(o) == 4 and o[0] == _ND:
        return np.frombuffer(o[3], dtype=np.dtype(o[1])).reshape(o[2])
    if isinstance(o, dict):
        return {k: dec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(dec(v) for v in o)
    return o


def _send(sock, obj):
    b = pickle.dumps(enc(obj), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(b)) + b)


def _recv(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            raise EOFError("env worker died -- see its stderr in RL_OUT/worker_*.log")
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise EOFError("env worker died mid-message")
        buf += c
    return dec(pickle.loads(bytes(buf)))


class VecEnv:
    def __init__(self, n, seed_pool, phase, out_dir):
        self.path = os.path.join(out_dir, f"envs_{os.getpid()}.sock")
        if os.path.exists(self.path):
            os.remove(self.path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.listen(n)
        chunks = np.array_split(np.asarray(seed_pool), n)
        self.procs, self.socks = [], []
        for i in range(n):
            log = open(os.path.join(out_dir, f"worker_{i}.log"), "w")
            self.procs.append(subprocess.Popen(
                [MJX_PY, os.path.join(HERE, "shelf_e2e_rl_worker.py"),
                 "--sock", self.path, "--phase", phase,
                 "--seeds", ",".join(str(int(x)) for x in chunks[i % len(chunks)]),
                 "--eps-per-seed", str(EPISODES_PER_SEED), "--snap-dir", SNAP_DIR],
                stdout=log, stderr=log, env={**os.environ, "MUJOCO_GL": "egl"}))
        for _ in range(n):
            c, _ = srv.accept()
            self.socks.append(c)
        srv.close()
        print(f"[rl] {n} env workers up ({MJX_PY})", flush=True)

    def reset(self, idx=None):
        idx = list(range(len(self.socks))) if idx is None else list(idx)
        for i in idx:
            _send(self.socks[i], ("reset", None))
        return {i: _recv(self.socks[i]) for i in idx}

    def step(self, actions):
        for i, c in enumerate(self.socks):
            _send(c, ("step", actions[i]))
        return [_recv(c) for c in self.socks]

    def close(self):
        for c in self.socks:
            try:
                _send(c, ("close", None))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        if os.path.exists(self.path):
            os.remove(self.path)


# ===================================================================== critic
def make_critic(rngs, in_dim, width=256):
    from flax import nnx

    class Critic(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(in_dim, width, rngs=rngs)
            self.l2 = nnx.Linear(width, width, rngs=rngs)
            self.l3 = nnx.Linear(width, 1, rngs=rngs)

        def __call__(self, x):
            import jax.numpy as jnp
            h = jnp.tanh(self.l1(x))
            h = jnp.tanh(self.l2(h))
            return self.l3(h)[..., 0]

    return Critic(rngs)


def critic_features(state22, info):
    """Privileged, reward-side features. The ACTOR never sees these."""
    lat = info["latches"]
    one_hot = np.zeros(6, np.float32)
    one_hot[int(info["stage"])] = 1.0
    return np.concatenate([
        np.asarray(state22, np.float32),
        one_hot,
        np.array([float(lat[k]) for k in ("touch", "grasp", "lift", "arrive", "place")], np.float32),
    ])


# ===================================================================== main
def main():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from openpi.models import model as _model
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config

    import pi05_flow_sde as F

    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "train.jsonl")
    cfg = PPOCfg()
    sde = F.SDEConfig(num_steps=cfg.denoise_steps, sigma=cfg.sigma)

    seeds = sorted(int(f.split("_")[1].split(".")[0])
                   for f in os.listdir(SNAP_DIR) if f.startswith("snap_"))
    assert seeds, f"no curriculum snapshots in {SNAP_DIR} -- run rl_snap_batch.sh first"
    print(f"[rl] phase={PHASE} envs={N_ENVS} seeds={len(seeds)} out={OUT}", flush=True)

    tcfg = _config.get_config(CFG_NAME)
    policy = _pc.create_trained_policy(tcfg, CKPT)
    model = policy._model
    bc_params = jax.tree.map(jnp.copy, nnx.state(model, tcfg.trainable_filter))

    vec = VecEnv(N_ENVS, seeds, PHASE, OUT)
    first = vec.reset()
    obs_list = [first[i][0] for i in range(N_ENVS)]

    rngs = nnx.Rngs(0)
    dummy = critic_features(obs_list[0]["state"], {"stage": 0, "latches": dict.fromkeys(
        ("touch", "grasp", "lift", "arrive", "place"), False)})
    critic = make_critic(rngs, len(dummy))
    ctx = nnx.Optimizer(critic, optax.adam(cfg.lr_critic), wrt=nnx.Param)
    atx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr_actor))
    aopt = atx.init(nnx.state(model, tcfg.trainable_filter))

    def to_obs(obs_list):
        batch = [policy._input_transform(
            {"observation/state": o["state"],
             "observation/image": o["image"],
             "observation/left_wrist_image": o["left_wrist"],
             "observation/right_wrist_image": o["right_wrist"],
             "prompt": o["prompt"]}) for o in obs_list]
        stacked = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *batch)
        return _model.Observation.from_dict(stacked), batch

    def to_env_actions(batch_inputs, acts):
        out = []
        for i, b in enumerate(batch_inputs):
            o = policy._output_transform({"state": b["state"], "actions": np.asarray(acts[i])})
            out.append(np.asarray(o["actions"], np.float64))
        return out

    DEFAULT_INFO = {"stage": 0, "latches": dict.fromkeys(
        ("touch", "grasp", "lift", "arrive", "place"), False)}
    # info describing the CURRENT state of each env (stage/latches). Reset -> defaults; after each
    # step the returned info describes the state we just landed in, i.e. the next V() input.
    cur_info = [dict(DEFAULT_INFO) for _ in range(N_ENVS)]

    key = jax.random.key(0)
    ep_ret = np.zeros(N_ENVS)
    ep_len = np.zeros(N_ENVS, int)
    finished = []

    for it in range(N_ITERS):
        t0 = time.time()
        buf = {k: [] for k in ("obs", "path", "logp", "rew", "done", "val", "feat")}
        for _ in range(T_ROLL):
            obs, batch_inputs = to_obs(obs_list)
            key, k1 = jax.random.split(key)
            acts, path, logp = F.sample_path(model, obs, k1, sde)
            env_acts = to_env_actions(batch_inputs, acts)
            feats = np.stack([critic_features(obs_list[i]["state"], cur_info[i])
                              for i in range(N_ENVS)])
            val = np.asarray(critic(jnp.asarray(feats)))
            res = vec.step(env_acts)

            buf["obs"].append(batch_inputs)
            buf["path"].append(np.asarray(path))
            buf["logp"].append(np.asarray(logp))
            buf["feat"].append(feats)
            buf["val"].append(val)
            buf["rew"].append(np.array([r for _, r, _, _ in res]))
            buf["done"].append(np.array([d for _, _, d, _ in res]))

            obs_list = [o for o, _, _, _ in res]
            cur_info = [inf for _, _, _, inf in res]
            ep_ret += buf["rew"][-1]
            ep_len += 1
            need = [i for i, (_, _, d, _) in enumerate(res) if d]
            for i in need:
                finished.append({"ret": float(ep_ret[i]), "len": int(ep_len[i]),
                                 "term": res[i][3]["term"], "lat": res[i][3]["latches"]})
                ep_ret[i], ep_len[i] = 0.0, 0
            if need:
                for i, (o, _) in vec.reset(need).items():
                    obs_list[i] = o
                    cur_info[i] = dict(DEFAULT_INFO)

        # ---- GAE
        rew = np.stack(buf["rew"]); done = np.stack(buf["done"]); val = np.stack(buf["val"])
        # bootstrap from V(s_T) -- the state we are in NOW, not the last one we stored
        last_feat = np.stack([critic_features(obs_list[i]["state"], cur_info[i])
                              for i in range(N_ENVS)])
        last_v = np.asarray(critic(jnp.asarray(last_feat)))
        adv = np.zeros_like(rew); gae = np.zeros(N_ENVS)
        for t in reversed(range(T_ROLL)):
            nv = last_v if t == T_ROLL - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + cfg.gamma * nv * nonterm - val[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO update
        flat = [(t, i) for t in range(T_ROLL) for i in range(N_ENVS)]
        stats = {"pg": [], "kl": [], "vf": [], "bc": []}
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            np.random.shuffle(flat)
            for s in range(0, len(flat), cfg.minibatch):
                mb = flat[s:s + cfg.minibatch]
                mb_inputs = [buf["obs"][t][i] for t, i in mb]
                mb_obs = _model.Observation.from_dict(
                    jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *mb_inputs))
                mb_path = jnp.asarray(np.stack([buf["path"][t][:, i] for t, i in mb], axis=1))
                mb_logp = jnp.asarray(np.array([buf["logp"][t][i] for t, i in mb]))
                mb_adv = jnp.asarray(np.array([advn[t, i] for t, i in mb]))

                def actor_loss(mdl):
                    lp = F.path_logprob(mdl, mb_obs, mb_path, sde)
                    # The "action" is a 5-step denoising path over 16x32 dims, so log-probs are
                    # ~3e3. A raw (logp_old - logp) threshold is therefore meaningless -- bf16 noise
                    # alone exceeded target_kl and early-stopped every update. Use the scale-free
                    # Schulman k3 estimator on the ratio instead; it is >= 0 and ~0 at ratio 1.
                    logr = jnp.clip(lp - mb_logp, -20.0, 20.0)
                    ratio = jnp.exp(logr)
                    pg = -jnp.mean(jnp.minimum(ratio * mb_adv,
                                               jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv))
                    approx_kl = jnp.mean(ratio - 1.0 - logr)
                    return pg, (approx_kl, ratio)

                (pg, (kl, _)), grads = nnx.value_and_grad(
                    actor_loss, argnums=nnx.DiffState(0, tcfg.trainable_filter), has_aux=True)(model)
                if float(kl) > cfg.target_kl * 2:
                    stop = True
                    break
                params = nnx.state(model, tcfg.trainable_filter)
                # KL-to-BC anchor: pull the trainable leaves back toward the BC checkpoint
                anchor = jax.tree.map(lambda p, b: cfg.bc_coef * (p - b), params, bc_params)
                grads = jax.tree.map(lambda g, a: g + a, grads, anchor)
                upd, aopt = atx.update(grads, aopt, params)
                nnx.update(model, optax.apply_updates(params, upd))
                stats["pg"].append(float(pg)); stats["kl"].append(float(kl))

                mb_feat = jnp.asarray(np.stack([buf["feat"][t][i] for t, i in mb]))
                mb_ret = jnp.asarray(np.array([ret[t, i] for t, i in mb]))

                def critic_loss(c):
                    return jnp.mean((c(mb_feat) - mb_ret) ** 2)
                vl, cg = nnx.value_and_grad(critic_loss)(critic)
                ctx.update(critic, cg)
                stats["vf"].append(float(vl))

        # ---- log
        rec = {"iter": it, "steps": (it + 1) * T_ROLL * N_ENVS,
               "rew_mean": float(rew.mean()), "sec": round(time.time() - t0, 1),
               "pg": float(np.mean(stats["pg"] or [0])), "kl": float(np.mean(stats["kl"] or [0])),
               "vf": float(np.mean(stats["vf"] or [0])), "early_stop": stop,
               "episodes": len(finished)}
        if finished:
            last = finished[-50:]
            rec["ep_ret"] = float(np.mean([e["ret"] for e in last]))
            rec["ep_len"] = float(np.mean([e["len"] for e in last]))
            for k in ("grasp", "lift", "place"):
                rec[f"sr_{k}"] = float(np.mean([e["lat"][k] for e in last]))
            rec["terms"] = {t: sum(e["term"] == t for e in last) for t in {e["term"] for e in last}}
        print("[rl] " + json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if (it + 1) % 25 == 0:
            import orbax.checkpoint as ocp
            p = os.path.join(OUT, f"ckpt_{it + 1:05d}")
            ocp.PyTreeCheckpointer().save(p, nnx.state(model, tcfg.trainable_filter).to_pure_dict())
            print(f"[rl] saved {p}", flush=True)

    vec.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
