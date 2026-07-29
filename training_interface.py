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
    extra: Dict[str, jax.Array] = struct.field(default_factory=dict)


class Agent(Protocol):
    """Minimal interface any RL algorithm must satisfy to train on an
    MjxEnv. Implement this once per algorithm (PPO, SAC, ...) -- the env
    and rollout code never need to know which one is plugged in."""

    def init(self, rng: jax.Array) -> Any:
        """Return initial (params, opt_state) or similar training state."""
        ...

    def act(
        self, train_state: Any, obs: jax.Array, rng: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Return (action, extra_info) given a batch of observations."""
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
            extra=extra,
        )
        return (next_state, rng), transition

    (state, rng), transitions = jax.lax.scan(
        scan_fn, (state, rng), None, length=n_steps
    )
    return state, transitions, rng


def train(
    env: MjxEnv,
    agent: Agent,
    rng: jax.Array,
    num_iterations: int,
    steps_per_iteration: int,
    log_fn: Callable[[int, Dict[str, jax.Array]], None] = lambda i, m: None,
) -> Any:
    """Generic train loop: rollout -> update -> repeat. Swap `agent` to
    swap the entire learning algorithm; nothing else here changes."""

    rng, init_rng, reset_rng = jax.random.split(rng, 3)
    train_state = agent.init(init_rng)
    state = env.reset(reset_rng)

    for it in tqdm(range(num_iterations), desc="Training", unit="iter"):
        t0 = time.perf_counter()
        state, batch, rng = rollout(
            env, agent, train_state, state, rng, steps_per_iteration
        )
        train_state, metrics = agent.update(train_state, batch)

        # Force sync to get accurate wall-clock execution time
        jax.block_until_ready(metrics)
        step_time = time.perf_counter() - t0

        if it == 0:
            print(f"\n[Iteration 0] Compilation + Execution time: {step_time:.4f}s")
        log_fn(it, metrics)

    return train_state
