#!/usr/bin/env python3
"""Correctness checks for the Flow-SDE wrapper, on the real BC checkpoint.

  1. re-scoring a stored path with UNCHANGED params reproduces the rollout log-prob
     (this is what makes the PPO ratio exactly 1.0 at the start of each update)
  2. sigma -> 0 collapses the SDE onto the deterministic flow that BC serving uses
  3. log-prob is sensitive to the path (a perturbed path must score lower)
  4. gradients w.r.t. the LoRA params of log-prob are finite and non-zero

Env: RL_CKPT (default the v2full BC checkpoint), RL_CONFIG (default pi05_cruzr_e2e_v2).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flax import nnx  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config as _pc  # noqa: E402
from openpi.training import config as _config  # noqa: E402

import pi05_flow_sde as F  # noqa: E402

CKPT = os.environ.get(
    "RL_CKPT", "/data1/hsr/openpi-main/checkpoints/pi05_cruzr_e2e_v2/cruzr_shelf_e2e_v2full/159999")
CFG = os.environ.get("RL_CONFIG", "pi05_cruzr_e2e_v2")
PROMPT = "pick up the steel pillar from the rack in front and place it on the second shelf of the cart"


def make_obs(policy, rng):
    """A syntactically valid observation; content is random -- these are numerical identities."""
    r = np.random.default_rng(0)
    raw = {
        "observation/image": r.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": r.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": r.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": r.normal(size=22).astype(np.float32),
        "prompt": PROMPT,
    }
    inputs = policy._input_transform(raw)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


def main():
    print(f"[test] loading {CFG} <- {CKPT}", flush=True)
    policy = _pc.create_trained_policy(_config.get_config(CFG), CKPT)
    model = policy._model
    obs = make_obs(policy, None)
    cfg = F.SDEConfig(num_steps=5, sigma=0.15)
    bad = 0

    def check(name, ok, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {detail}", flush=True)

    # 1 -- re-scoring identity
    act, path, logp = F.sample_path(model, obs, jax.random.key(0), cfg)
    logp2 = F.path_logprob(model, obs, path, cfg)
    rel = float(jnp.max(jnp.abs(logp - logp2) / (jnp.abs(logp) + 1e-6)))
    check("rescore == rollout logp", rel < 1e-4, f"logp={float(logp[0]):.3f} rel_err={rel:.2e}")

    # 2 -- sigma -> 0 recovers the deterministic flow
    det_cfg = F.SDEConfig(num_steps=5, sigma=0.0, min_sigma=1e-12)
    z = jax.random.normal(jax.random.key(1), (1, model.action_horizon, model.action_dim))
    a0, _, _ = F.sample_path(model, obs, jax.random.key(1), det_cfg, noise=z)
    a1 = model.sample_actions(jax.random.key(1), obs, num_steps=5, noise=z)
    d = float(jnp.max(jnp.abs(a0 - a1)))
    # Tolerance is set by the model's own precision, not by wishful thinking: openpi keeps the
    # frozen backbone in bfloat16 (eps = 2^-8 = 3.9e-3), so two mathematically identical Euler
    # chains that XLA fuses differently cannot agree to better than a few ulp.
    dts = {str(np.asarray(getattr(v, "value", v)).dtype)
           for _, v in nnx.state(model, nnx.Param).flat_state().items()}
    tol = 8 * 3.9e-3 if "bfloat16" in dts else 1e-4
    check("sigma->0 == BC flow ODE", d < tol,
          f"max|dA|={d:.2e} tol={tol:.1e} param dtypes={sorted(dts)}")

    # 3 -- density discriminates paths
    bump = path.at[-1].add(0.5)
    logp3 = F.path_logprob(model, obs, bump, cfg)
    check("perturbed path scores lower", bool(logp3[0] < logp[0]),
          f"{float(logp[0]):.1f} -> {float(logp3[0]):.1f}")

    # 4 -- differentiable w.r.t. the TRAINABLE (LoRA) params only.
    # Full-param jax.grad through the denoising chain OOMs a 24 GB 4090 -- differentiate only what
    # the optimizer will actually update, exactly as openpi's train_step does.
    tcfg = _config.get_config(CFG)
    n_train = len(nnx.state(model, tcfg.trainable_filter).flat_state())
    print(f"  ..    trainable leaves: {n_train}", flush=True)

    def loss(mdl):
        return jnp.sum(F.path_logprob(mdl, obs, path, cfg))
    g = nnx.grad(loss, argnums=nnx.DiffState(0, tcfg.trainable_filter))(model)
    gl = [np.asarray(getattr(v, "value", v)) for _, v in g.flat_state().items()]
    finite = all(np.all(np.isfinite(x)) for x in gl)
    nonzero = any(np.any(x != 0) for x in gl)
    check("trainable grads ok", bool(finite and nonzero and len(gl) > 0),
          f"n_grads={len(gl)} finite={finite} nonzero={nonzero}")

    print(f"\n{4 - bad}/4 flow-SDE checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
