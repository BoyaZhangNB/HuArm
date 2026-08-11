"""
sac_agent.py

Soft Actor-Critic (SAC), implementing the same `Agent` protocol as
`agent.py`'s PPOAgent (see `training_interface.py`), so it's a drop-in
alternative selected via `configs/*.yaml`'s `algo` key -- nothing in
`training_interface.py` or `agent.py` needs to change.

Motivation: PPO's policy (agent.py) squashes its Gaussian *mean* through
`nn.tanh` but computes log-prob/entropy as if the distribution were an
ordinary unbounded Gaussian -- no tanh-Jacobian correction. That lets the
mean saturate on the tanh plateau (vanishing gradient) while `log_std`
keeps shrinking, freezing the policy into a near-static action within a
handful of iterations. SAC's squashed-Gaussian policy below applies the
correction properly, and its automatically-tuned entropy coefficient
(`log_alpha`, toward `target_entropy = -action_size`) actively resists
exactly this kind of premature exploration collapse.

Off-policy training loop: `SACAgent.update()` receives the same
on-policy-shaped batch that `training_interface.rollout()` already
produces each iteration, pushes those transitions into a fixed-capacity
replay buffer carried inside `train_state`, then runs its own sampled
gradient steps against that buffer. `training_interface.py`'s generic
`rollout()`/`train()`/`eval()` never need to know the difference.
"""

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


class _ActorNet(nn.Module):
    action_size: int
    hidden: int = 256

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_size)(x)
        log_std = nn.Dense(self.action_size)(x)
        # Floor raised from -20 to -5: -20 permits a near-deterministic
        # policy (std ~ exp(-20)) whose log-probs blow up to pathological
        # magnitudes, which is what was driving the alpha/actor/critic
        # divergence (see agents/sac_agent.py module docstring investigation).
        log_std = jp.clip(log_std, -5.0, 2.0)
        return mean, log_std


class _QNet(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, obs, action):
        x = jp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        q = nn.Dense(1)(x)
        return jp.squeeze(q, axis=-1)


def _gaussian_log_prob(pre_tanh, mean, log_std):
    std = jp.exp(log_std)
    return -0.5 * jp.sum(
        ((pre_tanh - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
    )


def _sample_squashed(actor_net, params, obs, rng):
    """Reparameterized sample from the squashed (tanh) Gaussian policy,
    with the log-prob correctly including the tanh-Jacobian term -- the
    piece PPO's `_gaussian_log_prob` (agent.py) is missing."""
    mean, log_std = actor_net.apply(params, obs)
    std = jp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    pre_tanh = mean + noise * std
    action = jp.tanh(pre_tanh)
    log_prob = _gaussian_log_prob(pre_tanh, mean, log_std) - jp.sum(
        jp.log(1.0 - jp.square(action) + 1e-6), axis=-1
    )
    return action, log_prob


class SACAgent(Agent):
    """Soft Actor-Critic with twin Q-critics, target networks, and
    automatic entropy-coefficient tuning (Haarnoja et al.)."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_capacity: int = 500_000,
        gradient_steps: int = 64,
        min_buffer_size: int = 2_000,
        hidden_dim: int = 256,
        target_entropy: float = None,
        max_grad_norm: float = 1.0,
        alpha_max_grad_norm: float = 1.0,
        log_alpha_min: float = -10.0,
        log_alpha_max: float = 2.0,
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.actor_net = _ActorNet(action_size=action_size, hidden=hidden_dim)
        self.q_net = _QNet(hidden=hidden_dim)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.gradient_steps = gradient_steps
        self.min_buffer_size = min_buffer_size
        self.target_entropy = (
            -float(action_size) if target_entropy is None else target_entropy
        )
        # Hard backstop on the entropy temperature: independent of gradient
        # clipping, this caps how large `alpha = exp(log_alpha)` can ever
        # get, which is what was letting alpha (and, downstream, actor/critic
        # loss) run away unboundedly once log_prob diverged from target_entropy.
        self.log_alpha_min = log_alpha_min
        self.log_alpha_max = log_alpha_max

        # All four optimizers previously had no gradient clipping at all
        # (unlike PPOAgent's clip_by_global_norm, see agents/ppo_agent.py) --
        # a handful of outlier samples could otherwise produce unbounded
        # gradient norms with nothing to arrest them.
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(actor_lr)
        )
        self.q1_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(critic_lr)
        )
        self.q2_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(critic_lr)
        )
        self.alpha_optimizer = optax.chain(
            optax.clip_by_global_norm(alpha_max_grad_norm), optax.adam(alpha_lr)
        )

    def init(self, rng: jax.Array) -> Any:
        rng, actor_rng, q1_rng, q2_rng = jax.random.split(rng, 4)
        dummy_obs = jp.zeros((self.obs_size,), dtype=jp.float32)
        dummy_action = jp.zeros((self.action_size,), dtype=jp.float32)

        actor_params = self.actor_net.init(actor_rng, dummy_obs)
        q1_params = self.q_net.init(q1_rng, dummy_obs, dummy_action)
        q2_params = self.q_net.init(q2_rng, dummy_obs, dummy_action)
        q1_target = jax.tree_util.tree_map(lambda x: x, q1_params)
        q2_target = jax.tree_util.tree_map(lambda x: x, q2_params)
        log_alpha = jp.zeros(())

        capacity = self.buffer_capacity
        return {
            "actor_params": actor_params,
            "q1_params": q1_params,
            "q2_params": q2_params,
            "q1_target": q1_target,
            "q2_target": q2_target,
            "log_alpha": log_alpha,
            "actor_opt_state": self.actor_optimizer.init(actor_params),
            "q1_opt_state": self.q1_optimizer.init(q1_params),
            "q2_opt_state": self.q2_optimizer.init(q2_params),
            "alpha_opt_state": self.alpha_optimizer.init(log_alpha),
            # Fixed-capacity ring buffer of past transitions.
            "buf_obs": jp.zeros((capacity, self.obs_size), dtype=jp.float32),
            "buf_action": jp.zeros((capacity, self.action_size), dtype=jp.float32),
            "buf_reward": jp.zeros((capacity,), dtype=jp.float32),
            "buf_next_obs": jp.zeros((capacity, self.obs_size), dtype=jp.float32),
            # "done" flag used to mask Q-target bootstrapping (see update()).
            "buf_done": jp.zeros((capacity,), dtype=jp.float32),
            "write_ptr": jp.zeros((), dtype=jp.int32),
            "buffer_size": jp.zeros((), dtype=jp.int32),
            "rng": rng,
            "obs_norm": init_running_norm(self.obs_size),
        }

    def act(
        self,
        train_state: Any,
        obs: jax.Array,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        params = train_state["actor_params"]
        # Normalized with the running mean/std as of the *previous*
        # iteration's update -- see agents/obs_normalizer.py.
        obs = normalize_obs(train_state["obs_norm"], obs)
        if deterministic:
            mean, _ = self.actor_net.apply(params, obs)
            action = jp.tanh(mean)
            log_prob = jp.zeros(action.shape[:-1])
        else:
            action, log_prob = _sample_squashed(self.actor_net, params, obs, rng)
        return action, {"log_prob": log_prob}

    def update(
        self, train_state: Any, batch: Transition
    ) -> Tuple[Any, Dict[str, jax.Array]]:
        num_steps, num_envs = batch.reward.shape
        total = num_steps * num_envs
        flatten = lambda x: x.reshape((total,) + x.shape[2:])

        obs = flatten(batch.obs)
        action = flatten(batch.action)
        reward = flatten(batch.reward)
        next_obs = flatten(batch.next_obs)
        # `batch.done` is set on both true termination and episode-length
        # truncation (EpisodeWrapper). For the Q-target's bootstrap mask we
        # want to zero out *both* cases: under AutoResetWrapper, `next_obs`
        # one step past a truncation boundary belongs to the next (reset)
        # episode, not the true continuation of this one (same reasoning as
        # `_compute_gae`'s termination/truncation split in agent.py), so we
        # must not bootstrap through it either way. Algebraically this is
        # `max(termination, truncation)` where `termination = done * (1 -
        # truncation)`, which simplifies to exactly `done`.
        done_mask = flatten(batch.done)

        # 1. Push this iteration's transitions into the ring buffer.
        capacity = self.buffer_capacity
        write_ptr = train_state["write_ptr"]
        idx = (write_ptr + jp.arange(total)) % capacity
        buf_obs = train_state["buf_obs"].at[idx].set(obs)
        buf_action = train_state["buf_action"].at[idx].set(action)
        buf_reward = train_state["buf_reward"].at[idx].set(reward)
        buf_next_obs = train_state["buf_next_obs"].at[idx].set(next_obs)
        buf_done = train_state["buf_done"].at[idx].set(done_mask)
        new_write_ptr = (write_ptr + total) % capacity
        new_buffer_size = jp.minimum(train_state["buffer_size"] + total, capacity)

        # Fold this iteration's raw observations into the running estimate
        # before sampling minibatches below, so every gradient step in this
        # call (old buffer entries included) normalizes with the freshest
        # stats -- unlike PPO, SAC's off-policy updates don't need the
        # normalization to match what `act()` used when a sample was
        # collected, since log_probs/values are always recomputed fresh.
        new_obs_norm = update_running_norm(train_state["obs_norm"], obs)

        rng = train_state["rng"]
        init_carry = (
            train_state["actor_params"],
            train_state["q1_params"],
            train_state["q2_params"],
            train_state["q1_target"],
            train_state["q2_target"],
            train_state["log_alpha"],
            train_state["actor_opt_state"],
            train_state["q1_opt_state"],
            train_state["q2_opt_state"],
            train_state["alpha_opt_state"],
        )

        def grad_step(carry, step_rng):
            (
                actor_params, q1_params, q2_params, q1_target, q2_target,
                log_alpha, actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
            ) = carry

            idx_rng, next_a_rng, a_rng = jax.random.split(step_rng, 3)
            sample_idx = jax.random.randint(
                idx_rng, (self.batch_size,), 0, new_buffer_size
            )
            # Buffers hold raw observations regardless of when they were
            # collected; normalize with the freshest running stats here.
            mb_obs = normalize_obs(new_obs_norm, buf_obs[sample_idx])
            mb_action = buf_action[sample_idx]
            mb_reward = buf_reward[sample_idx]
            mb_next_obs = normalize_obs(new_obs_norm, buf_next_obs[sample_idx])
            mb_done = buf_done[sample_idx]

            alpha = jp.exp(log_alpha)

            # --- critic update ---
            next_action, next_log_prob = _sample_squashed(
                self.actor_net, actor_params, mb_next_obs, next_a_rng
            )
            q1_next = self.q_net.apply(q1_target, mb_next_obs, next_action)
            q2_next = self.q_net.apply(q2_target, mb_next_obs, next_action)
            min_q_next = jp.minimum(q1_next, q2_next) - alpha * next_log_prob
            target = mb_reward + self.gamma * (1.0 - mb_done) * min_q_next
            target = jax.lax.stop_gradient(target)

            def critic_loss_fn(q1_p, q2_p):
                q1 = self.q_net.apply(q1_p, mb_obs, mb_action)
                q2 = self.q_net.apply(q2_p, mb_obs, mb_action)
                q1_loss = jp.mean((q1 - target) ** 2)
                q2_loss = jp.mean((q2 - target) ** 2)
                return q1_loss + q2_loss, (q1_loss, q2_loss)

            (critic_loss, (q1_loss, q2_loss)), (q1_grads, q2_grads) = jax.value_and_grad(
                critic_loss_fn, argnums=(0, 1), has_aux=True
            )(q1_params, q2_params)

            q1_updates, q1_opt_state = self.q1_optimizer.update(
                q1_grads, q1_opt_state, q1_params
            )
            q1_params = optax.apply_updates(q1_params, q1_updates)
            q2_updates, q2_opt_state = self.q2_optimizer.update(
                q2_grads, q2_opt_state, q2_params
            )
            q2_params = optax.apply_updates(q2_params, q2_updates)

            # --- actor update ---
            def actor_loss_fn(a_params):
                action_s, log_prob_s = _sample_squashed(
                    self.actor_net, a_params, mb_obs, a_rng
                )
                q1_pi = self.q_net.apply(q1_params, mb_obs, action_s)
                q2_pi = self.q_net.apply(q2_params, mb_obs, action_s)
                min_q_pi = jp.minimum(q1_pi, q2_pi)
                loss = jp.mean(alpha * log_prob_s - min_q_pi)
                return loss, log_prob_s

            (actor_loss, log_prob_s), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_params)
            actor_updates, actor_opt_state = self.actor_optimizer.update(
                actor_grads, actor_opt_state, actor_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)

            # --- alpha (entropy coefficient) update ---
            def alpha_loss_fn(log_alpha_):
                return -jp.mean(
                    log_alpha_ * jax.lax.stop_gradient(log_prob_s + self.target_entropy)
                )

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
            alpha_updates, alpha_opt_state = self.alpha_optimizer.update(
                alpha_grads, alpha_opt_state, log_alpha
            )
            log_alpha = optax.apply_updates(log_alpha, alpha_updates)
            log_alpha = jp.clip(log_alpha, self.log_alpha_min, self.log_alpha_max)

            # --- Polyak-average target critics ---
            q1_target = jax.tree_util.tree_map(
                lambda t, o: self.tau * o + (1.0 - self.tau) * t, q1_target, q1_params
            )
            q2_target = jax.tree_util.tree_map(
                lambda t, o: self.tau * o + (1.0 - self.tau) * t, q2_target, q2_params
            )

            new_carry = (
                actor_params, q1_params, q2_params, q1_target, q2_target,
                log_alpha, actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
            )
            step_metrics = {
                "critic_loss": critic_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": alpha,
                "entropy": -jp.mean(log_prob_s),
            }
            return new_carry, step_metrics

        def do_updates(args):
            carry, loop_rng = args
            loop_rng, steps_rng = jax.random.split(loop_rng)
            step_rngs = jax.random.split(steps_rng, self.gradient_steps)
            carry, metrics_seq = jax.lax.scan(grad_step, carry, step_rngs)
            step_metrics = jax.tree_util.tree_map(jp.mean, metrics_seq)
            return carry, step_metrics, loop_rng

        def skip_updates(args):
            carry, loop_rng = args
            log_alpha = carry[5]
            zero = jp.zeros(())
            step_metrics = {
                "critic_loss": zero, "q1_loss": zero, "q2_loss": zero,
                "actor_loss": zero, "alpha_loss": zero,
                "alpha": jp.exp(log_alpha), "entropy": zero,
            }
            return carry, step_metrics, loop_rng

        do_update = new_buffer_size >= self.min_buffer_size
        (
            (
                actor_params, q1_params, q2_params, q1_target, q2_target,
                log_alpha, actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
            ),
            metrics,
            rng,
        ) = jax.lax.cond(do_update, do_updates, skip_updates, (init_carry, rng))

        new_train_state = {
            "actor_params": actor_params,
            "q1_params": q1_params,
            "q2_params": q2_params,
            "q1_target": q1_target,
            "q2_target": q2_target,
            "log_alpha": log_alpha,
            "actor_opt_state": actor_opt_state,
            "q1_opt_state": q1_opt_state,
            "q2_opt_state": q2_opt_state,
            "alpha_opt_state": alpha_opt_state,
            "buf_obs": buf_obs,
            "buf_action": buf_action,
            "buf_reward": buf_reward,
            "buf_next_obs": buf_next_obs,
            "buf_done": buf_done,
            "write_ptr": new_write_ptr,
            "buffer_size": new_buffer_size,
            "rng": rng,
            "obs_norm": new_obs_norm,
        }

        param_leaves = (
            jax.tree_util.tree_leaves(actor_params)
            + jax.tree_util.tree_leaves(q1_params)
            + jax.tree_util.tree_leaves(q2_params)
        )
        param_norm = (
            optax.global_norm(actor_params)
            + optax.global_norm(q1_params)
            + optax.global_norm(q2_params)
        )
        params_isnan = jp.any(
            jp.array([jp.isnan(leaf).any() for leaf in param_leaves])
        )

        metrics = dict(metrics)
        metrics["loss"] = metrics["critic_loss"] + metrics["actor_loss"]
        metrics["param_norm"] = param_norm
        metrics["params_isnan"] = params_isnan
        # Running obs-normalization constants, carried through the metrics
        # dict so callers (e.g. train.py) can persist them alongside the
        # model without reaching into train_state.
        metrics["obs_norm"] = {
            "mean": new_obs_norm.mean,
            "var": new_obs_norm.var,
            "count": new_obs_norm.count,
        }

        return new_train_state, metrics
