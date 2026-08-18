"""
demonstrations/demo_buffer.py

Loads teleop.py's `DemoRecorder` episodes (`demo_*.npz` under
`demonstrations/`) as (obs, action, reward, next_obs, done) transitions,
for seeding SACAgent's off-policy replay buffer -- see
agents/sac_agent.py's `seed_buffer_overrides` and train.py's `--demo-dir`.

This is the alternative to bc/train_bc.py's supervised-imitation warm
start: rather than pretraining the policy to *mimic* expert actions (no
reward signal, can only ever match, never improve on, the demos), the
transitions here are dropped straight into SAC's replay buffer as real
experience. Off-policy RL then bootstraps genuine Q-values from them and
keeps training exactly as usual -- the demo transitions are just older
entries alongside every rollout transition collected afterwards, so nothing
downstream (training_interface.py's train()/rollout(), SACAgent.update())
needs to know they didn't come from the current policy.

Loads plain (obs, action) pairs; bc/train_bc.py's `load_episodes` is the
same idea but doesn't need reward/done, so the two aren't shared -- keeping
them separate avoids coupling BC's format to the replay-buffer one.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class DemoTransitions(NamedTuple):
    obs: np.ndarray       # (N, obs_size)
    action: np.ndarray    # (N, action_size)
    reward: np.ndarray    # (N,)
    next_obs: np.ndarray  # (N, obs_size)
    done: np.ndarray      # (N,) float32 0./1. -- see SACAgent.update()'s done_mask


def load_demo_transitions(demo_dir, min_len: int = 2) -> DemoTransitions:
    """Concatenate every `demo_*.npz` under `demo_dir` into flat transition
    arrays, mirroring demonstrations/clean_demo.py's file glob.

    Each recorded episode of length T contributes T-1 transitions
    `(obs[t], action[t], reward[t], obs[t+1], done[t])` for t in
    [0, T-2] -- the last recorded step has no logged successor observation,
    so it's dropped rather than faked (e.g. duplicating obs[-1] into its own
    next_obs would teach the critic a spurious zero-motion transition).

    `min_len` (default 2, i.e. at least one transition) mirrors
    bc/train_bc.py's `--min-demo-len` safety net; demonstrations/clean_demo.py
    already filters short recordings out at collection time.
    """
    demo_dir = Path(demo_dir)
    paths = sorted(demo_dir.glob("demo_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No demo_*.npz files found under {demo_dir}")

    obs_chunks, action_chunks, reward_chunks, next_obs_chunks, done_chunks = [], [], [], [], []
    for p in paths:
        with np.load(p) as d:
            obs, action, reward, done = d["obs"], d["action"], d["reward"], d["done"]
        if len(obs) < min_len:
            print(f"[skip] {p.name}: {len(obs)} steps < min_len {min_len}")
            continue
        obs_chunks.append(obs[:-1].astype(np.float32))
        action_chunks.append(action[:-1].astype(np.float32))
        reward_chunks.append(reward[:-1].astype(np.float32))
        next_obs_chunks.append(obs[1:].astype(np.float32))
        done_chunks.append(done[:-1].astype(np.float32))

    if not obs_chunks:
        raise ValueError(f"All demos under {demo_dir} were shorter than min_len {min_len}")

    return DemoTransitions(
        obs=np.concatenate(obs_chunks, axis=0),
        action=np.concatenate(action_chunks, axis=0),
        reward=np.concatenate(reward_chunks, axis=0),
        next_obs=np.concatenate(next_obs_chunks, axis=0),
        done=np.concatenate(done_chunks, axis=0),
    )
