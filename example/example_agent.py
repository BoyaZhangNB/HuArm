"""
example_agent.py

A minimal Gaussian-policy REINFORCE agent implementing the `Agent`
protocol from training_interface.py. This is deliberately simple (a
one-hidden-layer MLP policy trained with vanilla policy gradient) -- the
point is to show the SHAPE of a custom algorithm plugging into the env,
not to be a strong baseline. Swap this file out for a PPO/SAC
implementation and nothing in mjx_env.py, wrappers.py, or
training_interface.py needs to change.
"""

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
        return mean, log_std


class ReinforceAgent(Agent):
    """Vanilla policy-gradient agent. obs_size/action_size come from the
    env; nothing here depends on which MJCF model the env wraps."""

    def __init__(self, obs_size: int, action_size: int, lr: float = 3e-4):
        self.net = _PolicyNet(action_size=action_size)
        self.obs_size = obs_size
        self.action_size = action_size
        self.optimizer = optax.adam(lr)

    def init(self, rng: jax.Array) -> Any:
        dummy_obs = jp.zeros((self.obs_size,))
        params = self.net.init(rng, dummy_obs)
        opt_state = self.optimizer.init(params)
        return {"params": params, "opt_state": opt_state}

    def act(
        self, train_state: Any, obs: jax.Array, rng: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        mean, log_std = self.net.apply(train_state["params"], obs)
        std = jp.exp(log_std)
        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = -0.5 * jp.sum(
            ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
        )
        return action, {"log_prob": log_prob}

    def update(
        self, train_state: Any, batch: Transition
    ) -> Tuple[Any, Dict[str, jax.Array]]:
        # returns-to-go per rollout step (batch dims: [time, ...env_batch])
        returns = jp.cumsum(batch.reward[::-1], axis=0)[::-1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        def loss_fn(params):
            def logp(obs, action):
                mean, log_std = self.net.apply(params, obs)
                std = jp.exp(log_std)
                return -0.5 * jp.sum(
                    ((action - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi),
                    axis=-1,
                )

            log_probs = jax.vmap(jax.vmap(logp))(batch.obs, batch.action)
            return -jp.mean(log_probs * returns)

        loss, grads = jax.value_and_grad(loss_fn)(train_state["params"])
        updates, opt_state = self.optimizer.update(
            grads, train_state["opt_state"], train_state["params"]
        )
        params = optax.apply_updates(train_state["params"], updates)
        return {"params": params, "opt_state": opt_state}, {"loss": loss}
