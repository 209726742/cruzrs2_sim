#!/usr/bin/env python3
"""Flow-SDE wrapper around openpi's pi0.5 -- the piece online RL needs that BC does not.

pi0.5 samples an action chunk by integrating a deterministic flow ODE
    x_{k+1} = x_k + dt * v_theta(x_k, t_k),      dt = -1/K,  t: 1 -> 0
which is a Dirac policy: no density, so no PPO ratio. Following piRL/RLinf we turn each Euler
step into a Gaussian transition (Flow-SDE)
    x_{k+1} ~ N( x_k + dt * v_theta(x_k, t_k),  (sigma*sqrt|dt|)^2 I )
so the whole denoising path has a tractable density
    log pi(path | obs) = sum_k log N( x_{k+1} ; mu_k, sigma^2|dt| )
The RL "action" is the PATH; storing it during rollout lets the PPO ratio be recomputed exactly
under updated parameters. The environment still executes only the final x_0 chunk.

Honest deviation: exact Flow-SDE preserves the ODE's marginals by adding a (sigma^2/2)*score
term to the drift. We inject noise WITHOUT the score correction, so the sampling distribution is
not identical to the BC flow's marginals. PPO does not need marginal preservation -- it needs a
well-defined density it can sample and score -- and the drift away from BC behaviour is bounded
by sigma and by the KL-to-BC anchor in the loss. Documented rather than silently assumed.

Nothing in /data1/hsr/openpi-main is modified: this module only reads the model.
"""
from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


@dataclasses.dataclass(frozen=True)
class SDEConfig:
    num_steps: int = 5      # denoising steps used for RL (BC serving uses 10; fewer = cheaper
                            # rollouts and a shorter path to differentiate through)
    sigma: float = 0.15     # exploration noise scale, in velocity units
    min_sigma: float = 1e-3


def _prefix_cache(model, obs):
    tokens, mask, ar = model.embed_prefix(obs)
    attn = make_attn_mask(mask, ar)
    pos = jnp.cumsum(mask, axis=1) - 1
    _, kv = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
    return tokens, mask, kv


def _velocity(model, obs, prefix, x_t, time):
    tokens, mask, kv = prefix
    b = x_t.shape[0]
    s_tok, s_mask, s_ar, adarms = model.embed_suffix(obs, x_t, jnp.broadcast_to(time, b))
    s_attn = make_attn_mask(s_mask, s_ar)
    p_attn = jnp.repeat(mask[:, None, :], s_tok.shape[1], axis=1)
    full = jnp.concatenate([p_attn, s_attn], axis=-1)
    pos = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(s_mask, axis=-1) - 1
    (_, s_out), _ = model.PaliGemma.llm([None, s_tok], mask=full, positions=pos,
                                        kv_cache=kv, adarms_cond=[None, adarms])
    return model.action_out_proj(s_out[:, -model.action_horizon:])


def _gauss_logp(x, mu, std):
    """Diagonal Gaussian log-density, summed over the (action_horizon, action_dim) axes."""
    z = (x - mu) / std
    return jnp.sum(-0.5 * z ** 2 - jnp.log(std) - 0.5 * math.log(2 * math.pi), axis=(-1, -2))


def sample_path(model, obs, rng, cfg: SDEConfig, noise=None):
    """Roll the SDE forward. Returns (actions, path, logp).

    path: (K+1, b, ah, ad) -- path[0] is the initial noise, path[K] the executed chunk.
    logp: (b,) log-density of the whole path under the current parameters.
    """
    obs = _model.preprocess_observation(None, obs, train=False)
    b = obs.state.shape[0]
    K = cfg.num_steps
    dt = -1.0 / K
    std = max(cfg.sigma * math.sqrt(abs(dt)), cfg.min_sigma)
    prefix = _prefix_cache(model, obs)

    if noise is None:
        rng, k0 = jax.random.split(rng)
        noise = jax.random.normal(k0, (b, model.action_horizon, model.action_dim))
    x = noise
    path = [x]
    logp = jnp.zeros((b,))
    t = 1.0
    for _ in range(K):
        mu = x + dt * _velocity(model, obs, prefix, x, t)
        rng, kk = jax.random.split(rng)
        x = mu + std * jax.random.normal(kk, mu.shape)
        logp = logp + _gauss_logp(x, mu, std)
        path.append(x)
        t = t + dt
    return x, jnp.stack(path), logp


def path_logprob(model, obs, path, cfg: SDEConfig, remat: bool = True):
    """Re-score a stored path under (possibly updated) parameters. Differentiable -> PPO ratio.

    remat=True checkpoints each denoising step: without it, backprop keeps the activations of all
    K transformer passes alive at once and a 3B pi0.5 OOMs on a 24 GB card (measured, not feared).
    """
    obs = _model.preprocess_observation(None, obs, train=False)
    K = cfg.num_steps
    dt = -1.0 / K
    std = max(cfg.sigma * math.sqrt(abs(dt)), cfg.min_sigma)
    prefix = _prefix_cache(model, obs)

    def vel(x, t):
        return _velocity(model, obs, prefix, x, t)
    if remat:
        vel = jax.checkpoint(vel)

    logp = jnp.zeros((path.shape[1],))
    t = 1.0
    for k in range(K):
        mu = path[k] + dt * vel(path[k], jnp.asarray(t))
        logp = logp + _gauss_logp(path[k + 1], mu, std)
        t = t + dt
    return logp


def bc_mean_path(model, obs, path, cfg: SDEConfig):
    """Deterministic-flow means along a stored path -- the KL-to-BC anchor compares against these."""
    obs = _model.preprocess_observation(None, obs, train=False)
    K = cfg.num_steps
    dt = -1.0 / K
    prefix = _prefix_cache(model, obs)
    out, t = [], 1.0
    for k in range(K):
        out.append(path[k] + dt * _velocity(model, obs, prefix, path[k], t))
        t = t + dt
    return jnp.stack(out)


# ----------------------------------------------------------------------------- functional form
def split(model):
    """(graphdef, params) so PPO can differentiate w.r.t. params functionally."""
    return nnx.split(model)


def apply_fn(graphdef, params, fn, *args, **kw):
    return fn(nnx.merge(graphdef, params), *args, **kw)


def as_actions(actions, norm_stats_unnorm_fn=None):
    a = np.asarray(actions)
    return a if norm_stats_unnorm_fn is None else norm_stats_unnorm_fn(a)
