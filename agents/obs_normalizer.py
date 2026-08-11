"""
obs_normalizer.py

Running (online) per-dimension mean/std used to normalize observations
before they reach either agent's networks. Both PPOAgent and SACAgent carry
one of these inside their `train_state` and fold each rollout batch into it
in `update()`, so the estimate keeps accumulating over the whole training
run -- each iteration's `act()` normalizes with the estimate as of the end
of the *previous* iteration (the only stats available at act-time), and
`update()` then folds the new batch in for the next one.

Uses Chan et al.'s parallel-variance formula (the same running-mean/std
trick behind, e.g., OpenAI baselines' VecNormalize) to merge a new batch's
mean/var into the running estimate exactly, rather than an exponential
moving average, which would need a smoothing-constant to tune and would
permanently under-weight the earliest observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from flax import struct


@struct.dataclass
class RunningNorm:
    mean: jax.Array
    var: jax.Array
    count: jax.Array  # total number of observations folded in so far


def init_running_norm(obs_size: int, dtype=jp.float32) -> RunningNorm:
    return RunningNorm(
        mean=jp.zeros((obs_size,), dtype=dtype),
        var=jp.ones((obs_size,), dtype=dtype),
        # Small positive count (rather than 0) avoids a div-by-zero in the
        # merge formula below on the very first update.
        count=jp.array(1e-4, dtype=dtype),
    )


def update_running_norm(norm: RunningNorm, batch_obs: jax.Array) -> RunningNorm:
    """Fold a (batch, obs_size) batch of freshly-collected observations into
    the running per-dimension mean/var estimate."""
    batch_mean = jp.mean(batch_obs, axis=0)
    batch_var = jp.var(batch_obs, axis=0)
    batch_count = batch_obs.shape[0]

    delta = batch_mean - norm.mean
    tot_count = norm.count + batch_count

    new_mean = norm.mean + delta * batch_count / tot_count
    m_a = norm.var * norm.count
    m_b = batch_var * batch_count
    new_m2 = m_a + m_b + jp.square(delta) * norm.count * batch_count / tot_count
    new_var = new_m2 / tot_count

    return norm.replace(mean=new_mean, var=new_var, count=tot_count)


def normalize_obs(norm: RunningNorm, obs: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Scale `obs` to zero-mean/unit-variance per dimension using the
    running estimate. `eps` guards dimensions whose variance is ~0."""
    return (obs - norm.mean) / jp.sqrt(norm.var + eps)
