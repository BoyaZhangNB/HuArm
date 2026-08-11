
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import optax
from flax import linen as nn

from training_interface import Agent, Transition
from agents.obs_normalizer import (
    RunningNorm,
    init_running_norm,
    normalize_obs,
    update_running_norm,
)

class _PolicyNet(nn.Module):
    action_size: int
    hidden: int = 64

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden)(obs)
        x = nn.tanh(x)
        # Raw (unbounded) pre-tanh mean -- the actual [-1, 1] squashing
        # happens at sample time in `act()`/`update()`, together with the
        # matching log-prob correction below. Squashing this directly
        # (as a plain `nn.tanh(...)`) was the bug: it saturates and kills
        # the mean's gradient near +-1 while the log-prob math still
        # assumed an ordinary unbounded Gaussian, so PPO's cheapest way to
        # "improve" was to collapse `log_std` instead of moving the mean --
        # a fast path to a frozen, degenerate policy.
        mean = nn.Dense(self.action_size)(x)
        log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_size,)
        )
        value = jp.squeeze(nn.Dense(1)(x), axis=-1)
        return mean, log_std, value


def _atanh(x, eps: float = 1e-6):
    x = jp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jp.log((1.0 + x) / (1.0 - x))


def _squashed_gaussian_log_prob(action, mean, log_std):
    """log-prob of a tanh-squashed action under a Gaussian with the given
    (pre-tanh) mean/log_std, i.e. log pi(a) = log N(atanh(a); mean, std) -
    sum(log(1 - a^2)) -- the tanh-Jacobian correction that a squashed
    Gaussian policy needs and a plain Gaussian log-prob is missing.

    `action` is the already-squashed action (what's actually stored in a
    Transition/replay batch); we invert tanh to recover the pre-tanh value
    the Gaussian density applies to.
    """
    pre_tanh = _atanh(action)
    std = jp.exp(log_std)
    log_prob = -0.5 * jp.sum(
        ((pre_tanh - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
    )
    log_prob -= jp.sum(jp.log(1.0 - jp.square(action) + 1e-6), axis=-1)
    return log_prob


def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    termination: jax.Array,
    truncation: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jax.Array, jax.Array]:
    """Generalized Advantage Estimation that distinguishes true termination
    from episode-length truncation under autoreset.

    `termination` (real done, no bootstrap) zeroes out the next-value term as
    usual. `truncation` additionally masks the *delta itself* and cuts the
    backward recursion at that step, since under AutoResetWrapper the
    `next_obs`/value one step past a truncation boundary belongs to the next
    (reset) episode, not the true continuation of this one.
    """
    truncation_mask = 1.0 - truncation
    values_t_plus_1 = jp.concatenate([values[1:], bootstrap_value[None]], axis=0)
    deltas = rewards + gamma * (1.0 - termination) * values_t_plus_1 - values
    deltas = deltas * truncation_mask

    def scan_fn(acc, xs):
        truncation_mask_t, termination_t, delta_t = xs
        acc = delta_t + gamma * (1.0 - termination_t) * truncation_mask_t * gae_lambda * acc
        return acc, acc

    init_acc = jp.zeros_like(bootstrap_value)
    _, advantages = jax.lax.scan(
        scan_fn, init_acc, (truncation_mask, termination, deltas), reverse=True
    )
    value_targets = advantages + values
    return advantages, value_targets


class PPOAgent(Agent):
    """Proximal Policy Optimization (PPO) agent. obs_size/action_size come from the
    env; nothing here depends on which MJCF model the env wraps."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        num_epochs: int = 4,
        num_minibatches: int = 8,
        max_grad_norm: float = 0.5,
    ):
        self.net = _PolicyNet(action_size=action_size)
        self.obs_size = obs_size
        self.action_size = action_size
        self.max_grad_norm = max_grad_norm
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

    def init(self, rng: jax.Array) -> Any:
        rng, params_rng = jax.random.split(rng)
        dummy_obs = jp.zeros((self.obs_size,), dtype=jp.float32)
        params = self.net.init(params_rng, dummy_obs)
        opt_state = self.optimizer.init(params)
        obs_norm = init_running_norm(self.obs_size)
        return {"params": params, "opt_state": opt_state, "rng": rng, "obs_norm": obs_norm}

    def act(
        self,
        train_state: Any,
        obs: jax.Array,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Normalized with the running mean/std as of the *previous*
        # iteration's update -- see agents/obs_normalizer.py.
        obs = normalize_obs(train_state["obs_norm"], obs)
        mean, log_std, value = self.net.apply(train_state["params"], obs)
        if deterministic:
            action = jp.tanh(mean)
        else:
            std = jp.exp(log_std)
            noise = jax.random.normal(rng, mean.shape)
            pre_tanh = mean + noise * std
            action = jp.tanh(pre_tanh)
        log_prob = _squashed_gaussian_log_prob(action, mean, log_std)
        return action, {"log_prob": log_prob, "value": value}

    def update(
        self, train_state: Any, batch: Transition
    ) -> Tuple[Any, Dict[str, jax.Array]]:
        params = train_state["params"]
        opt_state = train_state["opt_state"]
        rng = train_state["rng"]
        obs_norm = train_state["obs_norm"]

        # `batch.done` is set on both true termination and episode-length
        # truncation (EpisodeWrapper); split them so we only skip bootstrapping
        # on real termination, and cut the GAE recursion (without zeroing the
        # step's own reward) on truncation -- see `_compute_gae`.
        truncation = batch.truncation
        termination = batch.done * (1.0 - truncation)

        values = batch.extra["value"]
        # Same running stats `act()` used to normalize `batch.obs` during
        # this rollout -- `obs_norm` isn't folded in until below, so this
        # keeps the bootstrap value on the same scale as `values`.
        norm_next_obs_last = normalize_obs(obs_norm, batch.next_obs[-1])
        _, _, bootstrap_value = self.net.apply(params, norm_next_obs_last)

        advantages, value_targets = _compute_gae(
            rewards=batch.reward,
            values=values,
            bootstrap_value=bootstrap_value,
            termination=termination,
            truncation=truncation,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        num_steps, num_envs = batch.reward.shape
        total = num_steps * num_envs
        flatten = lambda x: x.reshape((total,) + x.shape[2:])

        raw_obs = flatten(batch.obs)
        # Same running stats used above for the bootstrap value / used by
        # `act()` to collect this batch -- updated (below) only after the
        # loss uses them, so this iteration's optimization stays internally
        # consistent.
        obs = normalize_obs(obs_norm, raw_obs)
        action = flatten(batch.action)
        old_log_prob = flatten(batch.extra["log_prob"])
        advantages = flatten(advantages)
        value_targets = flatten(value_targets)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        minibatch_size = total // self.num_minibatches
        usable = minibatch_size * self.num_minibatches

        def loss_fn(p, mb_obs, mb_action, mb_old_log_prob, mb_advantages, mb_value_targets):
            mean, log_std, value = self.net.apply(p, mb_obs)
            log_prob = _squashed_gaussian_log_prob(mb_action, mean, log_std)

            ratio = jp.exp(log_prob - mb_old_log_prob)
            unclipped = ratio * mb_advantages
            clipped = jp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
            policy_loss = -jp.mean(jp.minimum(unclipped, clipped))

            value_loss = jp.mean((value - mb_value_targets) ** 2)

            entropy = jp.mean(jp.sum(log_std + 0.5 * jp.log(2 * jp.pi * jp.e), axis=-1))

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            return loss, (policy_loss, value_loss, entropy)

        def minibatch_step(carry, idx):
            params, opt_state = carry
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, obs[idx], action[idx], old_log_prob[idx], advantages[idx], value_targets[idx]
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, *aux)

        def epoch_step(carry, epoch_rng):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_rng, total)[:usable]
            mb_indices = perm.reshape(self.num_minibatches, minibatch_size)
            (params, opt_state), metrics = jax.lax.scan(
                minibatch_step, (params, opt_state), mb_indices
            )
            return (params, opt_state), metrics

        rng, epochs_rng = jax.random.split(rng)
        epoch_rngs = jax.random.split(epochs_rng, self.num_epochs)
        (params, opt_state), (loss, policy_loss, value_loss, entropy) = jax.lax.scan(
            epoch_step, (params, opt_state), epoch_rngs
        )

        # Fold this iteration's raw observations into the running estimate
        # *after* they've been used above, so the next iteration's act()/
        # update() see it, not this one.
        new_obs_norm = update_running_norm(obs_norm, raw_obs)

        new_train_state = {
            "params": params,
            "opt_state": opt_state,
            "rng": rng,
            "obs_norm": new_obs_norm,
        }
        param_leaves = jax.tree_util.tree_leaves(params)
        param_norm = optax.global_norm(params)
        params_isnan = jp.any(
            jp.array([jp.isnan(leaf).any() for leaf in param_leaves])
        )
        metrics = {
            "loss": jp.mean(loss),
            "policy_loss": jp.mean(policy_loss),
            "value_loss": jp.mean(value_loss),
            "entropy": jp.mean(entropy),
            "param_norm": param_norm,
            "params_isnan": params_isnan,
            # Running obs-normalization constants, carried through the
            # metrics dict so callers (e.g. train.py) can persist them
            # alongside the model without reaching into train_state.
            "obs_norm": {
                "mean": new_obs_norm.mean,
                "var": new_obs_norm.var,
                "count": new_obs_norm.count,
            },
        }
        return new_train_state, metrics