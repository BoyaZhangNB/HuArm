
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import optax
from flax import linen as nn

from training_interface import Agent, Transition

class _PolicyNet(nn.Module):
    action_size: int
    hidden: int = 64

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden)(obs)
        x = nn.tanh(x)
        mean = nn.Dense(self.action_size)(x)
        log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_size,)
        )
        value = jp.squeeze(nn.Dense(1)(x), axis=-1)
        return mean, log_std, value


def _gaussian_log_prob(action, mean, log_std):
    std = jp.exp(log_std)
    return -0.5 * jp.sum(
        ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
    )


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
        entropy_coef: float = 0.0,
        num_epochs: int = 4,
        num_minibatches: int = 4,
    ):
        self.net = _PolicyNet(action_size=action_size)
        self.obs_size = obs_size
        self.action_size = action_size
        self.optimizer = optax.adam(lr)
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
        return {"params": params, "opt_state": opt_state, "rng": rng}

    def act(
        self, train_state: Any, obs: jax.Array, rng: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        mean, log_std, value = self.net.apply(train_state["params"], obs)
        std = jp.exp(log_std)
        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = _gaussian_log_prob(action, mean, log_std)
        return action, {"log_prob": log_prob, "value": value}

    def update(
        self, train_state: Any, batch: Transition
    ) -> Tuple[Any, Dict[str, jax.Array]]:
        params = train_state["params"]
        opt_state = train_state["opt_state"]
        rng = train_state["rng"]

        # `batch.done` is set on both true termination and episode-length
        # truncation (EpisodeWrapper); split them so we only skip bootstrapping
        # on real termination, and cut the GAE recursion (without zeroing the
        # step's own reward) on truncation -- see `_compute_gae`.
        truncation = batch.truncation
        termination = batch.done * (1.0 - truncation)

        values = batch.extra["value"]
        _, _, bootstrap_value = self.net.apply(params, batch.next_obs[-1])

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

        obs = flatten(batch.obs)
        action = flatten(batch.action)
        old_log_prob = flatten(batch.extra["log_prob"])
        advantages = flatten(advantages)
        value_targets = flatten(value_targets)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        minibatch_size = total // self.num_minibatches
        usable = minibatch_size * self.num_minibatches

        def loss_fn(p, mb_obs, mb_action, mb_old_log_prob, mb_advantages, mb_value_targets):
            mean, log_std, value = self.net.apply(p, mb_obs)
            log_prob = _gaussian_log_prob(mb_action, mean, log_std)

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

        new_train_state = {"params": params, "opt_state": opt_state, "rng": rng}
        metrics = {
            "loss": jp.mean(loss),
            "policy_loss": jp.mean(policy_loss),
            "value_loss": jp.mean(value_loss),
            "entropy": jp.mean(entropy),
        }
        return new_train_state, metrics