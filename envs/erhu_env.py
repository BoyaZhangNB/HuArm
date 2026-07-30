import jax
import jax.numpy as jp

import mujoco
from mujoco import mjx
from .mjx_env import MjxEnv, State
from .utils_envs import init_huarm

from typing import Any, Dict

class ErhuEnv(MjxEnv):
    """A simple environment for controlling an Erhu robot. n_frames is frame_skip"""
    def __init__(self, n_frames: int = 4, **kwargs):
        super().__init__(xml_path="huarm/arm.xml", n_frames=n_frames, **kwargs)
        self.id_dict = None
        self.mj_model, self.mj_data = init_huarm(self.mj_model, self.mj_data)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        self.terminal_condition = 30 # 30 seconds

    def reset(self, rng: jax.Array) -> State:
        rng, env_rng= jax.random.split(rng, 2)
        
        data = self.mjx_data
        # TODO implement domain randomization function
        # self.mjx_model, in_axes = domain_randomization(self.mjx_model, env_rng)

        info: Dict[str, Any] = {}
        obs = self._get_obs(data, info)
        reward, done = jp.asarray(0.0, dtype=jp.float32), jp.asarray(0.0, dtype=jp.float32)
        qpos = data.qpos
        metrics = {"joint_positions": qpos}
        return State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        # For simplicity, we return the joint positions and velocities as the observation
        qpos = data.qpos
        qvel = data.qvel
        obs = jp.concatenate([qpos, qvel])
        return obs

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        data = self.pipeline_step(state.pipeline_state, action)
        
        info: Dict[str, Any] = state.info.copy()
        obs = self._get_obs(data, info)
        # Define a simple reward function based on joint positions
        reward = -jp.sum(jp.square(data.qpos))  # Encourage staying near zero position
        done = jp.where(data.time > self.terminal_condition, 1.0, 0.0)
        
        metrics = {"joint_positions": data.qpos}
        
        return State(data, obs, reward, done, metrics, info)