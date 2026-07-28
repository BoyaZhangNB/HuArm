"""
example_env.py

A concrete MjxEnv subclass showing how to plug in a custom MJCF model +
task-specific reward/termination logic. Swap the XML and _get_obs/step
logic to define a totally different task while reusing all the plumbing
in mjx_env.py.

This example: cartpole swing-up-and-balance.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jp

from envs.mjx_env import MjxEnv, State

# A self-contained MJCF model -- replace with your own robot/task XML.
_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="top" pos="0 0 2"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 0.3">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="cart_geom" type="box" size="0.1 0.1 0.05" rgba="0.2 0.4 0.8 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.02" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slider" gear="50" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class CartpoleSwingUp(MjxEnv):
    """Swing the pole upright and balance it by pushing the cart."""

    def __init__(self, n_frames: int = 4, **kwargs: Any):
        super().__init__(xml_string=_CARTPOLE_XML, n_frames=n_frames, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)
        # start hanging down with small random perturbation
        qpos = jp.array([0.0, jp.pi]) + jax.random.uniform(
            qpos_rng, (2,), minval=-0.05, maxval=0.05
        )
        qvel = jax.random.uniform(qvel_rng, (2,), minval=-0.05, maxval=0.05)

        data = self.pipeline_init(qpos, qvel)
        info: Dict[str, Any] = {}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(()), jp.zeros(())
        metrics = {"pole_height": jp.zeros(())}
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        data = self.pipeline_step(state.pipeline_state, action)

        cart_x = data.qpos[0]
        pole_angle = data.qpos[1]
        # height of pole tip: 0 at hinge, pi = hanging down, 0/2pi = upright
        pole_height = jp.cos(pole_angle)

        # reward: pole up, cart centered, small control cost
        reward = pole_height - 0.01 * cart_x**2 - 0.001 * jp.sum(action**2)

        out_of_bounds = jp.abs(cart_x) > 1.8
        done = out_of_bounds.astype(jp.float32)

        obs = self._get_obs(data, state.info)
        state.metrics.update(pole_height=pole_height)
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, data, info) -> jax.Array:
        cart_x = data.qpos[0]
        pole_angle = data.qpos[1]
        return jp.array(
            [
                cart_x,
                jp.sin(pole_angle),
                jp.cos(pole_angle),
                data.qvel[0],
                data.qvel[1],
            ]
        )


if __name__ == "__main__":
    # Quick smoke test: random rollout, no training.
    env = CartpoleSwingUp()
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    jit_step = jax.jit(env.step)

    print(f"obs_size={env.observation_size} action_size={env.action_size} dt={env.dt}")
    for i in range(5):
        rng, act_rng = jax.random.split(rng)
        action = jax.random.uniform(act_rng, (env.action_size,), minval=-1, maxval=1)
        state = jit_step(state, action)
        print(f"step {i}: reward={float(state.reward):.3f} done={bool(state.done)}")
