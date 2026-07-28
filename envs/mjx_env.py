"""
mjx_env.py

Base abstract environment for MuJoCo-XLA (MJX) reinforcement learning.

Design goals
------------
1. MODEL-AGNOSTIC: subclass MjxEnv and load ANY MJCF (xml_path or xml_string).
   The base class only handles model loading / physics stepping plumbing.
2. ALGORITHM-AGNOSTIC: the env exposes a plain functional API (reset, step,
   observation_size, action_size) with no assumptions about how actions are
   produced or how updates happen. Any training loop (PPO, SAC, evolution,
   random search, ...) can drive it -- see training_interface.py.
3. JAX-NATIVE: State is a flax pytree so reset/step are jit-able and
   vmap-able out of the box (required for MJX's batched physics).

A concrete environment only needs to implement three methods:
    - reset(rng) -> State
    - step(state, action) -> State
    - _get_obs(data, info) -> jax.Array
Everything else (model loading, physics substeps, dt, sizes) is handled here.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    """Environment state carried between steps. Registered as a JAX pytree,
    so it can safely pass through jit/vmap/scan."""

    pipeline_state: mjx.Data          # full MJX physics state (qpos, qvel, ...)
    obs: jax.Array                    # observation vector
    reward: jax.Array                 # scalar reward
    done: jax.Array                   # 0./1. float flag (float, not bool, so
                                       # it plays nicely with jax.lax.select)
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MjxEnv(abc.ABC):
    """Base class for all MJX environments.

    Subclass this for every new task/model. Only reset(), step() and
    _get_obs() are required; everything else is reusable infrastructure.
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        xml_string: Optional[str] = None,
        n_frames: int = 1,
        **kwargs: Any,
    ):
        """
        Args:
            xml_path: path to a MJCF (.xml) file describing the model.
            xml_string: MJCF contents as a string (alternative to xml_path).
            n_frames: number of physics substeps per env.step() call
                (i.e. control is applied at model.opt.timestep * n_frames).
            kwargs: forwarded to subclasses for task-specific config.
        """
        if xml_string is not None:
            self.mj_model = mujoco.MjModel.from_xml_string(xml_string)
        elif xml_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            raise ValueError("MjxEnv requires either xml_path or xml_string.")

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_frames = n_frames

    # ------------------------------------------------------------------
    # Required overrides (the only thing a new task needs to implement)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Return a fresh State for a new episode."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Advance physics by one control step and return the new State."""

    @abc.abstractmethod
    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Build the observation vector from raw physics state."""

    # ------------------------------------------------------------------
    # Reusable physics plumbing -- rarely needs overriding
    # ------------------------------------------------------------------
    def pipeline_init(self) -> mjx.Data:
        """Create an MJX Data at a given (qpos, qvel) and run forward
        kinematics so derived quantities (xpos, sensordata, ...) are valid."""
        data = mjx.make_data(self.mjx_model)
        data = mjx.forward(self.mjx_model, data)
        return data

    def pipeline_step(self, data: mjx.Data, action: jax.Array) -> mjx.Data:
        """Apply `action` as ctrl and integrate physics for n_frames steps."""

        def substep(d, _):
            d = d.replace(ctrl=action)
            return mjx.step(self.mjx_model, d), None

        data, _ = jax.lax.scan(substep, data, None, length=self.n_frames)
        return data

    # ------------------------------------------------------------------
    # Convenience properties used by training code
    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return float(self.mj_model.opt.timestep) * self.n_frames

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        state = self.reset(rng)
        return int(state.obs.shape[-1])

    @property
    def backend(self) -> str:
        return "mjx"
