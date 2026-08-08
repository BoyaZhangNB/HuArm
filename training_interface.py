"""
training_interface.py

Defines the contract between an MjxEnv and ANY training algorithm, so you
can swap in PPO, SAC, evolutionary strategies, or a hand-written controller
without touching the environment code.

The contract is intentionally tiny:

    class Agent(Protocol):
        def init(self, rng) -> params
        def act(self, params, obs, rng) -> action, extra   # extra = e.g. log_prob, value
        def update(self, params, opt_state, batch) -> params, opt_state, metrics

A `Transition` is the standard unit of data passed from env -> agent.
`rollout()` collects a batch of Transitions from a (possibly vmapped) env
using jax.lax.scan -- this part never needs to change per-algorithm.
`train()` is a thin loop that alternates rollout() and agent.update();
swapping algorithms means swapping the `agent` object, nothing else.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Tuple

import jax
import jax.numpy as jp
from flax import struct
from tqdm import tqdm
import time

from envs.mjx_env import MjxEnv, State


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array
    # 1./0. flag: was `done` caused by hitting the episode-length limit
    # (EpisodeWrapper) rather than true termination? Needed to avoid treating
    # the post-autoreset `next_obs` as a real terminal observation.
    truncation: jax.Array
    extra: Dict[str, jax.Array] = struct.field(default_factory=dict)


class Agent(Protocol):
    """Minimal interface any RL algorithm must satisfy to train on an
    MjxEnv. Implement this once per algorithm (PPO, SAC, ...) -- the env
    and rollout code never need to know which one is plugged in."""

    def init(self, rng: jax.Array) -> Any:
        """Return initial (params, opt_state) or similar training state."""
        ...

    def act(
        self,
        train_state: Any,
        obs: jax.Array,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Return (action, extra_info) given a batch of observations.
        `deterministic=True` should return the policy's greedy/mean action
        (no exploration noise), for evaluation."""
        ...

    def update(
        self, train_state: Any, batch: Transition
    ) -> Tuple[Any, Dict[str, jax.Array]]:
        """Consume a batch of Transitions, return (new_train_state, metrics)."""
        ...


def rollout(
    env: MjxEnv,
    agent: Agent,
    train_state: Any,
    state: State,
    rng: jax.Array,
    n_steps: int,
) -> Tuple[State, Transition, jax.Array]:
    """Collect `n_steps` of (vectorized) transitions by alternating
    agent.act() and env.step(). Works for any env/agent satisfying the
    interfaces above -- this function never needs to change."""

    def scan_fn(carry, _):
        state, rng = carry
        rng, act_rng = jax.random.split(rng)
        action, extra = agent.act(train_state, state.obs, act_rng)
        next_state = env.step(state, action)
        transition = Transition(
            obs=state.obs,
            action=action,
            reward=next_state.reward,
            done=next_state.done,
            next_obs=next_state.obs,
            truncation=next_state.truncation,
            extra=extra,
        )
        return (next_state, rng), transition

    (state, rng), transitions = jax.lax.scan(
        scan_fn, (state, rng), None, length=n_steps
    )
    return state, transitions, rng

def eval(env, agent, train_state, rng, n_episodes, max_steps=2000):
    """Evaluate the agent on a vectorized, autoresetting env until at least
    `n_episodes` episodes have completed (across all parallel envs combined),
    and return the mean episode return, along with the mean per-episode
    reward terms (env-specific breakdown from state.metrics["reward_terms"],
    e.g. velocity/pressure/contact_duration/...), as
    {"reward": scalar, "reward_terms": {name: scalar, ...}}.

    Only episodes that actually finish (done==1) are counted; any episode
    still in progress when `max_steps` is hit is dropped so it can't bias
    the average with a truncated, partial return.
    """

    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    num_envs = state.done.shape[0]

    zero_terms = jax.tree_util.tree_map(jp.zeros_like, state.metrics["reward_terms"])

    ep_return = jp.zeros(num_envs, dtype=jp.float32)
    ep_reward_terms = zero_terms
    ep_len = jp.zeros(num_envs, dtype=jp.int32)
    return_sum = jp.zeros((), dtype=jp.float32)
    reward_terms_sum = jax.tree_util.tree_map(lambda _: jp.zeros((), dtype=jp.float32), zero_terms)
    length_sum = jp.zeros((), dtype=jp.int32)
    episode_count = jp.zeros((), dtype=jp.int32)
    step = jp.zeros((), dtype=jp.int32)

    def cond_fn(carry):
        *_, episode_count, step = carry
        return jp.logical_and(episode_count < n_episodes, step < max_steps)

    def body_fn(carry):
        (state, rng, ep_return, ep_reward_terms, ep_len, return_sum, reward_terms_sum,
         length_sum, episode_count, step) = carry

        rng, act_rng = jax.random.split(rng)
        action, _ = agent.act(train_state, state.obs, act_rng, deterministic=True)
        state = env.step(state, action)

        ep_return = ep_return + state.reward
        ep_reward_terms = jax.tree_util.tree_map(
            lambda acc, term: acc + term, ep_reward_terms, state.metrics["reward_terms"]
        )
        ep_len = ep_len + 1
        done = state.done > 0

        return_sum = return_sum + jp.sum(jp.where(done, ep_return, 0.0))
        reward_terms_sum = jax.tree_util.tree_map(
            lambda acc, ep_term: acc + jp.sum(jp.where(done, ep_term, 0.0)),
            reward_terms_sum, ep_reward_terms,
        )
        length_sum = length_sum + jp.sum(jp.where(done, ep_len, 0), dtype=jp.int32)
        episode_count = episode_count + jp.sum(done, dtype=jp.int32)
        ep_return = jp.where(done, 0.0, ep_return)
        ep_reward_terms = jax.tree_util.tree_map(lambda t: jp.where(done, 0.0, t), ep_reward_terms)
        ep_len = jp.where(done, 0, ep_len)

        return (state, rng, ep_return, ep_reward_terms, ep_len, return_sum, reward_terms_sum,
                length_sum, episode_count, step + 1)

    (state, rng, ep_return, ep_reward_terms, ep_len, return_sum, reward_terms_sum,
     length_sum, episode_count, step) = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (state, rng, ep_return, ep_reward_terms, ep_len, return_sum, reward_terms_sum,
         length_sum, episode_count, step),
    )

    denom = jp.maximum(episode_count, 1)
    mean_reward_terms = jax.tree_util.tree_map(lambda s: s / denom, reward_terms_sum)
    return {
        "reward": return_sum / denom,
        "reward_terms": mean_reward_terms,
        "episode_length": length_sum / denom,
    }

def train(
    env: MjxEnv,
    agent: Agent,
    rng: jax.Array,
    num_iterations: int,
    steps_per_iteration: int,
    eval_interval: int,
    log_fn: Callable[[int, Dict[str, jax.Array]], None] = lambda i, m: None,
) -> Any:
    """Generic train loop: rollout -> update -> repeat. Swap `agent` to
    swap the entire learning algorithm; nothing else here changes."""

    rng, init_rng, reset_rng = jax.random.split(rng, 3)
    train_state = agent.init(init_rng)
    state = env.reset(reset_rng)

    def _train_step(train_state, state, rng):
        state, batch, rng = rollout(
                    env, agent, train_state, state, rng, steps_per_iteration
                )
        train_state, metrics = agent.update(train_state, batch)
        return train_state, state, rng, metrics

    _train_step = jax.jit(_train_step)

    def _eval_fn(train_state, rng):
        return eval(env, agent, train_state, rng, n_episodes=30)

    _eval = jax.jit(_eval_fn)

    try:
        for it in tqdm(range(num_iterations), desc="Training", unit="iter"):
            train_state, state, rng, metrics = _train_step(train_state, state, rng)

            if it % eval_interval == 0:
                eval_metrics = _eval(train_state, rng)
                metrics["eval_reward"] = eval_metrics["reward"]
                metrics["eval_episode_length"] = eval_metrics["episode_length"]
                for name, val in eval_metrics["reward_terms"].items():
                    metrics[f"eval_reward_terms/{name}"] = val

            log_fn(it, metrics)
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user.")

    return train_state