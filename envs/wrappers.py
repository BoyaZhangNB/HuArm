"""
wrappers.py

Reusable, algorithm-agnostic env wrappers. These compose around any
MjxEnv subclass without needing to know task-specific details, and
without needing to know what training algorithm will consume them.

    EpisodeWrapper   - adds a max episode length + truncation flag
    AutoResetWrapper  - auto-resets sub-episodes inside a long jax.lax.scan
                        rollout (standard trick for on-policy RL in JAX)
    VmapWrapper       - vectorizes reset/step over a batch of parallel envs

Typical stack for on-policy training (e.g. PPO):

    env = MyTask()
    env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
    env = AutoResetWrapper(env)
    env = VmapWrapper(env, batch_size=2048)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp

from .mjx_env import MjxEnv, State


class Wrapper(MjxEnv):
    """Base wrapper: forwards everything to the wrapped env by default."""

    def __init__(self, env: MjxEnv):
        # deliberately skip MjxEnv.__init__ (no model to load here)
        self.env = env

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    def _get_obs(self, data, info):
        return self.env._get_obs(data, info)

    @property
    def dt(self):
        return self.env.dt

    @property
    def action_size(self):
        return self.env.action_size

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def unwrapped(self) -> MjxEnv:
        e = self.env
        while isinstance(e, Wrapper):
            e = e.env
        return e


class EpisodeWrapper(Wrapper):
    """Adds fixed-length episodes with a `truncation` signal, and optional
    action repeat (apply the same action for k physics-control steps)."""

    def __init__(self, env: MjxEnv, episode_length: int, action_repeat: int = 1):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["steps"] = jp.asarray(0.0, dtype=jp.int32)
        return state.replace(truncation=jp.zeros_like(state.done))

    def step(self, state: State, action: jax.Array) -> State:
        def repeat_step(s, _):
            return self.env.step(s, action), None

        state, _ = jax.lax.scan(repeat_step, state, None, length=self.action_repeat)

        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done, dtype=jp.float32)
        zero = jp.zeros_like(state.done, dtype=jp.float32)
        episode_over = steps >= self.episode_length
        not_already_done = state.done == 0
        truncation = jp.where(episode_over & not_already_done, one, zero)
        done = jp.where(episode_over, one, state.done)

        state.info["steps"] = steps
        return state.replace(done=done, truncation=truncation)


class AutoResetWrapper(Wrapper):
    """Automatically resets to a fresh episode when `done` is set, without
    ending the outer jax.lax.scan rollout. This lets a training loop collect
    a fixed-length trajectory of transitions that may span episode
    boundaries, which is standard practice for JAX-native on-policy RL.

    Unlike the common brax-style AutoResetWrapper -- which splices back a
    single "first" state captured once at the outer `reset()` and replays it
    for every in-rollout auto-reset -- this one calls the wrapped env's real
    `reset()` again (with a fresh rng split off a key carried in
    `info["rng"]`) at each auto-reset boundary. That matters for any env
    whose `reset()` does per-episode domain randomization (e.g. ErhuEnv's
    `dr_params` -- friction/mass/damping/erhu placement, drawn fresh each
    `reset()` call): with the replay-the-first-state approach, that
    randomization would only actually change once per *outer* reset (e.g.
    once per `num_resets_per_eval` cycle), not once per episode. Calling
    `reset()` again here re-draws it every episode, matching the "per
    episode" behavior the randomization is meant to have.
    """

    def reset(self, rng: jax.Array) -> State:
        rng, reset_rng = jax.random.split(rng)
        state = self.env.reset(reset_rng)
        state.info["rng"] = rng
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # clear `done` before stepping so the wrapped env doesn't see a
        # stale terminal flag from the previous transition, and reset the
        # episode step counter for envs that just finished so EpisodeWrapper
        # doesn't see a stale (already over-threshold) step count and mark
        # every subsequent step done again.
        if "steps" in state.info:
            steps = jp.where(state.done, jp.zeros_like(state.info["steps"]), state.info["steps"])
            state.info["steps"] = steps
        carry_rng = state.info["rng"]
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # Fresh reset (including fresh domain randomization) for whichever
        # envs just finished, rather than replaying a single state captured
        # once at the outer reset.
        carry_rng, reset_rng = jax.random.split(carry_rng)
        reset_state = self.env.reset(reset_rng)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        pipeline_state = jax.tree_util.tree_map(
            where_done, reset_state.pipeline_state, state.pipeline_state
        )
        obs = where_done(reset_state.obs, state.obs)
        # Splice every other per-episode info field (dr_params, action/force
        # history, contact_steps, ...) the same way, so they stay in sync
        # with pipeline_state/obs instead of bleeding stale values from the
        # just-finished episode across the auto-reset boundary.
        info = dict(state.info)
        for key, reset_value in reset_state.info.items():
            info[key] = jax.tree_util.tree_map(where_done, reset_value, state.info[key])
        info["rng"] = carry_rng

        return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)


class VmapWrapper(Wrapper):
    """Vectorizes reset/step across `batch_size` parallel environments using
    jax.vmap. This is what lets MJX simulate thousands of envs on a GPU/TPU
    in lockstep, independent of whatever algorithm consumes the batch."""

    def __init__(self, env: MjxEnv, num_envs: int):
        super().__init__(env)
        self.batch_size = num_envs

    def reset(self, rng: jax.Array) -> State:
        rngs = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rngs)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)
