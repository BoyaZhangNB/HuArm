import jax
import jax.numpy as jp

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from .mjx_env import MjxEnv, State
from .utils_envs import build_erhu_pose_pool, domain_randomize_jax, arm_joint_indices

from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Small self-contained quaternion helpers (jax-native, jit/vmap safe).
# Not pulled from mujoco.mjx._src.math since that module is private API.
# ---------------------------------------------------------------------------

def _mat_to_quat(mat: jax.Array) -> jax.Array:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m00, m01, m02 = mat[0, 0], mat[0, 1], mat[0, 2]
    m10, m11, m12 = mat[1, 0], mat[1, 1], mat[1, 2]
    m20, m21, m22 = mat[2, 0], mat[2, 1], mat[2, 2]
    tr = m00 + m11 + m22
    qw = jp.sqrt(jp.maximum(0.0, 1.0 + tr)) / 2.0
    qx = jp.sqrt(jp.maximum(0.0, 1.0 + m00 - m11 - m22)) / 2.0
    qy = jp.sqrt(jp.maximum(0.0, 1.0 - m00 + m11 - m22)) / 2.0
    qz = jp.sqrt(jp.maximum(0.0, 1.0 - m00 - m11 + m22)) / 2.0
    qx = jp.copysign(qx, m21 - m12)
    qy = jp.copysign(qy, m02 - m20)
    qz = jp.copysign(qz, m10 - m01)
    return jp.array([qw, qx, qy, qz])


def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q: jax.Array) -> jax.Array:
    w, x, y, z = q
    return jp.array([w, -x, -y, -z])


def _get_contact(data: mjx.Data):
    """`Data.contact` moved under `Data._impl` in newer mujoco.mjx; support both."""
    impl = getattr(data, "_impl", None)
    return impl.contact if impl is not None else data.contact


# ---------------------------------------------------------------------------
# Placeholders -- goal generation and forbidden-volume query.
#
# TODO: at inference time these are meant to be replaced by real sensing:
#   - desired_velocity_and_pressure(): motion-capture + pressure-sensor
#     readings of a reference bow stroke.
#   - forbidden_area_distance(): a computer-vision estimate of the erhu /
#     keep-out region location, converted to a signed distance.
# For now both are pure functions of (model, data) that ignore their inputs
# and return fixed placeholder values, so the surrounding reward/termination/
# obs code has a stable interface to build against.
# ---------------------------------------------------------------------------

def desired_velocity_and_pressure(
    mjx_model: mjx.Model, data: mjx.Data
) -> Tuple[jax.Array, jax.Array]:
    """Returns (desired lateral bow velocity, target pressure in N).

    `velocity` is a signed scalar (m/s), not a world-frame direction vector:
    it's the desired speed of the bow along the erhu's own left/right
    (lateral) axis -- see `ErhuEnv._lateral_axis_local` -- positive/negative
    indicating stroke direction (e.g. push vs. pull) relative to the erhu,
    independent of the erhu's world pose.
    """
    velocity = jp.sin(2 * jp.pi * data.time / 5) * 0.05  # oscillate between -0.05 and 0.05 m/s
    pressure = jp.asarray(2.0)
    return velocity, pressure


def forbidden_area_distance(mjx_model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Signed distance to the forbidden volume: positive = outside/safe,
    negative = inside the forbidden region. Placeholder is a large positive
    constant so the term stays inert until a real query is wired in."""
    return jp.asarray(1e3)


class ErhuEnv(MjxEnv):
    """Erhu bowing task for the HuArm robot. n_frames is frame_skip.

    Action space: normalized delta on the 5 arm position actuators
    (joint1, joint2, joint5, joint3, joint4 -- actuator order), i.e.
    action in [-1, 1]^5 scaled by `max_ctrl_delta`
    (radians) and added to the previous actuator target each step.

    ctrl target is clamped to `mjx_model.actuator_ctrlrange` each step.
    """

    def __init__(
        self,
        n_frames: int = 20, # timestep 0.002 * 20 = 0.04s per step, 25Hz
        n_stack: int = 3,
        enable_forbidden_zone: bool = True,
        max_ctrl_delta: float = 0.05,
        episode_time_limit: float = 100.0,
        f_max: float = 30.0,
        f_safe: float = 3.0,
        clip_penetration_limit: float = 0.01,
        string_tolerance: float = 0.15,
        forbidden_margin: float = 0.1,
        contact_duration_scale: float = 20.0, # 20 steps (0.8s) give you reward of 0.76, 40 steps give you reward of 0.96, 60 steps give you reward of 0.99.
        contact_force_ema_alpha: float = 0.2, # low-pass factor on bow_a_force (per RL step, 25Hz); see `_step_kinematics`.
        contact_steps_decay: float = 5.0, # steps of `contact_steps` lost per non-contact step, instead of an instant reset to 0.
        velocity_kernel_scale: float = 20.0, # corresopnd to accpetable error of 0.05 [m/s].
        pressure_kernel_scale: float = 1.0, # corresopnd to accpetable error of 1 [N].
        reward_weights: Dict[str, float] = None,
        dr_pool_size: int = 1024,
        dr_pool_seed: int = 0,
        **kwargs,
    ):
        super().__init__(xml_path="huarm/arm.xml", n_frames=n_frames, **kwargs)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        self._erhu_pose_pool = build_erhu_pose_pool(
            self.mj_model, self.mjx_model,
            jax.random.PRNGKey(dr_pool_seed), dr_pool_size, seed=dr_pool_seed,
        )

        # Static arm-joint index arrays for domain_randomize_jax /
        # _set_joint_ctrl_jax, computed once here instead of on every
        # reset() to cut python-side mj_name2id lookups off the hot path.
        (self._arm_qpos_idxs,
         self._arm_ctrl_aids,
         self._arm_ctrl_qpos_idxs,
         self._arm_act_idxs) = arm_joint_indices(self.mj_model)

        self.n_stack = n_stack
        self.enable_forbidden_zone = enable_forbidden_zone
        self.max_ctrl_delta = max_ctrl_delta
        self.episode_time_limit = episode_time_limit
        self.f_max = f_max
        self.f_safe = f_safe
        self.clip_penetration_limit = clip_penetration_limit
        self.string_tolerance = string_tolerance
        self.forbidden_margin = forbidden_margin
        self.contact_duration_scale = contact_duration_scale
        self.contact_force_ema_alpha = contact_force_ema_alpha
        self.contact_steps_decay = contact_steps_decay
        self.velocity_kernel_scale = velocity_kernel_scale
        self.pressure_kernel_scale = pressure_kernel_scale

        self.reward_weights = dict(
            velocity=1.0,
            pressure=1.0,
            contact_duration=0.5,
            action_smoothness=0.5,
            bow_flat=0.2,
            bow_touch=0.2,
            string_center=0.5,
            contact_penalty=1.0,
            forbidden=1.0,
            termination=1.0,
            survival=1.0,
        )
        if reward_weights:
            self.reward_weights.update(reward_weights)

        m = self.mj_model

        def bid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        def gid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        def sid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

        self._base_id = bid("base")
        self._erhu_root_id = bid("erhu_root")
        self._sound_box_id = bid("sound_box")
        self._string_d_id = bid("string_D")
        self._string_a_id = bid("string_A")
        self._bow_link1_id = bid("bow_link_1")

        self._sound_box_geom_id = gid("sound_box_geom")
        self._bow_hair_geom_id = gid("bow_hair_geom")
        self._string_d_geom_id = gid("string_D_geom")
        self._string_a_geom_id = gid("string_A_geom")

        self._bow_hair_frog_site = sid("bow_hair_frog_site")
        self._bow_hair_tip_site = sid("bow_hair_tip_site")

        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "bow_arm_contact")
        self._force_adr = int(m.sensor_adr[sensor_id])
        self._force_dim = int(m.sensor_dim[sensor_id])

        # Number of contact slots mjx allocates per step -- static (fixed by
        # the model), so it's safe to unroll a Python loop over it inside
        # jit when reading per-contact forces (mjx.contact_force takes a
        # concrete/static contact index, not a traced one).
        self._ncon = int(_get_contact(self.mjx_data).geom.shape[0])

        # Bow-hair/string contact-detection tolerance -- both pairs share
        # the "bow_hair_string_contact" class in arm.xml, so read it once
        # from either pair rather than hard-coding it here.
        pair_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, "bow_hair_string_D_pair")
        self._bow_string_margin = float(m.pair_margin[pair_id])
        self._bow_string_gap = float(m.pair_gap[pair_id])

        self._sound_box_radius = float(m.geom_size[self._sound_box_geom_id][0])
        self._ctrl_lo = self.mjx_model.actuator_ctrlrange[:, 0]
        self._ctrl_hi = self.mjx_model.actuator_ctrlrange[:, 1]

        # The erhu's local left/right axis, expressed in the erhu root frame.
        self._lateral_axis_local = jp.array([1.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def _effective_model(self, dr_params: Dict[str, jax.Array]) -> mjx.Model:
        """Merges this episode's randomized fields onto the closure-captured
        base `self.mjx_model`."""
        return self.mjx_model.replace(**dr_params)

    # ------------------------------------------------------------------
    # Geometry helpers (static ids captured by closure, dynamic data args)
    # ------------------------------------------------------------------
    def _relative(self, world_pos: jax.Array, frame_id: int, data: mjx.Data) -> jax.Array:
        """`world_pos` expressed in the frame of body `frame_id`."""
        origin = data.xpos[frame_id]
        rot = data.xmat[frame_id].reshape(3, 3)
        return rot.T @ (world_pos - origin)

    def _bow_geometry(self, data: mjx.Data):
        """Returns (frog, tip, mid, axis_unit) in world frame."""
        frog = data.site_xpos[self._bow_hair_frog_site]
        tip = data.site_xpos[self._bow_hair_tip_site]
        mid = 0.5 * (frog + tip)
        axis = tip - frog
        axis_unit = axis / (jp.linalg.norm(axis) + 1e-8)
        return frog, tip, mid, axis_unit

    def _between_strings_target(self, data: mjx.Data) -> jax.Array:
        local_tip = jp.array([0.0, 0.0, -0.6])
        d_tip = data.xpos[self._string_d_id] + data.xmat[self._string_d_id].reshape(3, 3) @ local_tip
        a_tip = data.xpos[self._string_a_id] + data.xmat[self._string_a_id].reshape(3, 3) @ local_tip
        midpoint = 0.5 * (d_tip + a_tip)
        top_z = data.xpos[self._sound_box_id][2] + self._sound_box_radius
        midpoint = midpoint.at[2].set(jp.maximum(midpoint[2], top_z) + 0.01)
        return midpoint

    def _sound_box_normal(self, data: mjx.Data) -> jax.Array:
        """Unit normal vector of the sound box's top surface, in world frame. Proxied by the string's normal."""
        return data.xmat[self._string_a_id].reshape(3, 3)[:, 2]

    def _bow_string_contact(self, data: mjx.Data):
        """Returns (touching, min_dist) for the bow hair vs. either string. Not taking into account for contact margin"""
        margin = self._bow_string_margin
        gap = self._bow_string_gap
        contact = _get_contact(data)
        g1, g2, dist = contact.geom[:, 0], contact.geom[:, 1], contact.dist
        is_pair = (
            (g1 == self._bow_hair_geom_id)
            & ((g2 == self._string_a_geom_id) | (g2 == self._string_d_geom_id))
        ) | (
            (g2 == self._bow_hair_geom_id)
            & ((g1 == self._string_a_geom_id) | (g1 == self._string_d_geom_id))
        )
        
        active = is_pair & (dist < margin + gap)
        touching = jp.any(active)
        min_dist = jp.min(jp.where(active, dist, jp.inf))
        return touching, min_dist

    def _contact_force_mags(self, data: mjx.Data, model: mjx.Model) -> jax.Array:
        """Per-contact-slot linear contact force magnitude, shape (ncon,).
        
        `model` should be this episode's effective (possibly randomized)
        model -- see `_effective_model` -- not necessarily `self.mjx_model`.
        """
        forces = [
            mjx_support.contact_force(model, data, i)[:3]
            for i in range(self._ncon)
        ]
        return jp.linalg.norm(jp.stack(forces), axis=-1)

    def _max_contact_force(self, data: mjx.Data, model: mjx.Model) -> jax.Array:
        """Largest contact force magnitude among ALL active contacts."""
        return jp.max(self._contact_force_mags(data, model), initial=0.0)

    def _bow_hair_a_string_force(self, data: mjx.Data, model: mjx.Model) -> jax.Array:
        """Contact force magnitude between the bow hair and the A-string
        specifically"""
        contact = _get_contact(data)
        g1, g2 = contact.geom[:, 0], contact.geom[:, 1]
        is_pair = (
            (g1 == self._bow_hair_geom_id) & (g2 == self._string_a_geom_id)
        ) | (
            (g2 == self._bow_hair_geom_id) & (g1 == self._string_a_geom_id)
        )
        mags = jp.where(is_pair, self._contact_force_mags(data, model), 0.0)
        return jp.max(mags, initial=0.0)

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> State:
        rng, env_rng = jax.random.split(rng, 2)

        dr_params, data = domain_randomize_jax(
            self.mjx_model, self.mjx_data, env_rng, self._erhu_pose_pool,
            self._arm_qpos_idxs, self._arm_ctrl_aids, self._arm_ctrl_qpos_idxs,
            self._arm_act_idxs,
        )
        model = self._effective_model(dr_params)

        _, _, mid, _ = self._bow_geometry(data)
        info: Dict[str, Any] = {
            "dr_params": dr_params,
            "action_history": jp.zeros((self.n_stack, self.action_size)),
            "force_history": jp.zeros((self.n_stack, self._force_dim)),
            "prev_bow_mid": mid,
            "contact_steps": jp.asarray(0.0),
            "bow_a_force_ema": jp.asarray(0.0),
        }
        obs = self._get_obs(data, info, model)
        reward, done = jp.asarray(0.0, dtype=jp.float32), jp.asarray(0.0, dtype=jp.float32)
        # metrics dict must match step()'s structure so wrappers that scan
        # over state (e.g. EpisodeWrapper's action_repeat) see a consistent
        # pytree between the initial carry and the scanned output.
        metrics = {
            "joint_positions": data.qpos,
            "force_mag": jp.asarray(0.0, dtype=jp.float32),
            "max_contact_force": jp.asarray(0.0, dtype=jp.float32),
            "bow_a_force": jp.asarray(0.0, dtype=jp.float32),
            "bow_a_force_ema": jp.asarray(0.0, dtype=jp.float32),
            "velocity_error": jp.asarray(0.0, dtype=jp.float32),
            "pressure_error": jp.asarray(0.0, dtype=jp.float32),
            "contact": jp.asarray(0.0, dtype=jp.float32),
            "reward_terms": {name: jp.asarray(0.0, dtype=jp.float32) for name in self.reward_weights},
        }
        return State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], model: mjx.Model = None) -> jax.Array:
        model = self.mjx_model if model is None else model
        qpos, qvel = data.qpos, data.qvel

        sound_box_rel = self._relative(data.xpos[self._sound_box_id], self._base_id, data)
        string_d_rel = self._relative(data.xpos[self._string_d_id], self._base_id, data)
        string_a_rel = self._relative(data.xpos[self._string_a_id], self._base_id, data)

        frog, tip, mid, _ = self._bow_geometry(data)
        frog_rel = self._relative(frog, self._erhu_root_id, data)
        tip_rel = self._relative(tip, self._erhu_root_id, data)
        mid_rel = self._relative(mid, self._erhu_root_id, data)

        bow_quat = _mat_to_quat(data.xmat[self._bow_link1_id].reshape(3, 3))
        erhu_quat = _mat_to_quat(data.xmat[self._erhu_root_id].reshape(3, 3))
        rel_quat = _quat_mul(_quat_conj(erhu_quat), bow_quat)

        force = data.sensordata[self._force_adr:self._force_adr + self._force_dim]

        desired_velocity, desired_pressure = desired_velocity_and_pressure(model, data)
        forbidden_dist = forbidden_area_distance(model, data)

        obs = jp.concatenate([
            qpos, qvel,
            sound_box_rel, string_d_rel, string_a_rel,
            rel_quat, frog_rel, tip_rel, mid_rel,
            force,
            jp.reshape(desired_velocity, (1,)), jp.reshape(desired_pressure, (1,)),
            jp.reshape(forbidden_dist, (1,)),
            info["action_history"].reshape(-1),
            info["force_history"].reshape(-1),
        ])
        return obs

    # ------------------------------------------------------------------
    # Per-step kinematics shared across reward/termination terms, computed
    # once so each reward-category function below stays cheap and self-
    # contained.
    # ------------------------------------------------------------------
    def _step_kinematics(self, data: mjx.Data, info: Dict[str, Any], model: mjx.Model) -> Dict[str, jax.Array]:
        # Sensor force
        force = data.sensordata[self._force_adr:self._force_adr + self._force_dim]
        force_mag = jp.linalg.norm(force)
        # Both are magnitudes
        max_contact_force = self._max_contact_force(data, model)
        bow_a_force = self._bow_hair_a_string_force(data, model)

        frog, tip, mid, bow_axis = self._bow_geometry(data)
        bow_vel_world = (mid - info["prev_bow_mid"]) / self.dt
        # Tracking error is computed in the erhu's local frame: rotate the
        # (world-frame, finite-difference) bow velocity into erhu_root and
        # keep only the lateral (left/right) component.
        erhu_root = data.xmat[self._erhu_root_id].reshape(3, 3)
        bow_vel_local = erhu_root.T @ bow_vel_world
        # Bow should move in the normal direction of the cynlider.
        lateral_vel = jp.dot(bow_vel_local, self._lateral_axis_local)

        # touching, min_dist = self._bow_string_contact(data)

        bow_a_force_ema = (
            self.contact_force_ema_alpha * bow_a_force
            + (1.0 - self.contact_force_ema_alpha) * info["bow_a_force_ema"]
        )
        touching = jp.where(bow_a_force_ema > 1e-3, jp.asarray(1.0), jp.asarray(0.0))
        # Losing contact for a single step now costs `contact_steps_decay`
        # steps of progress instead of the whole accumulated streak.
        contact_steps = jp.where(
            touching,
            info["contact_steps"] + 1.0,
            jp.maximum(info["contact_steps"] - self.contact_steps_decay, 0.0),
        )

        desired_velocity, desired_pressure = desired_velocity_and_pressure(model, data)
        forbidden_dist = forbidden_area_distance(model, data)

        return dict(
            force=force,
            force_mag=force_mag,
            max_contact_force=max_contact_force,
            bow_a_force=bow_a_force,
            bow_a_force_ema=bow_a_force_ema,
            mid=mid,
            bow_axis=bow_axis,
            lateral_vel=lateral_vel,
            touching=touching,
            contact_steps=contact_steps,
            desired_velocity=desired_velocity,
            desired_pressure=desired_pressure,
            forbidden_dist=forbidden_dist,
        )

    # ------------------------------------------------------------------
    # Reward categories -- each returns its (signed) contribution to the
    # total reward *before* the corresponding `reward_weights` entry is
    # applied. Positive = good, negative = penalty.
    # ------------------------------------------------------------------
    def _reward_velocity(self, k: Dict[str, jax.Array]) -> jax.Array:
        """Tracking accuracy: lateral bow velocity (erhu-local frame) vs. desired velocity."""
        err = (k["lateral_vel"] - k["desired_velocity"]) * self.velocity_kernel_scale
        return 1.0 / (1.0 + jp.square(err))

    def _reward_pressure(self, k: Dict[str, jax.Array]) -> jax.Array:
        """Tracking accuracy: bow-hair/A-string contact force (low-pass
        filtered, see `_step_kinematics`) vs. desired pressure."""
        err = (k["bow_a_force_ema"] - k["desired_pressure"]) * self.pressure_kernel_scale
        return 1.0 / (1.0 + jp.square(err))

    def _reward_contact_duration(self, k: Dict[str, jax.Array]) -> jax.Array:
        """Bonus for sustained (not just momentary) hair-string contact."""
        return jp.tanh(k["contact_steps"] / self.contact_duration_scale)

    def _reward_action_smoothness(self, action: jax.Array, info: Dict[str, Any]) -> jax.Array:
        """Penalize large jumps between consecutive actions."""
        prev_action = info["action_history"][-1]
        return -jp.sum(jp.square(action - prev_action))

    def _reward_bow_flat(self, data: mjx.Data, k: Dict[str, jax.Array]) -> jax.Array:
        """Bow should lie flat: its axis perpendicular to the sound box normal."""
        normal = self._sound_box_normal(data)
        return -jp.square(jp.dot(k["bow_axis"], normal)) # both are unit vectors

    def _reward_bow_touch_box(self, data: mjx.Data, k: Dict[str, jax.Array]) -> jax.Array:
        """Bow should ride at the top surface of (the curve top surface of) the sound box."""
        mid_local = self._relative(k["mid"], self._sound_box_id, data)
        err = (mid_local[2] - self._sound_box_radius) / self._sound_box_radius
        return 1.0 / (1.0 + jp.square(err))

    def _reward_string_center(self, data: mjx.Data, k: Dict[str, jax.Array]) -> jax.Array:
        """Bow center should stay within `string_tolerance` of the between-strings target."""
        target = self._between_strings_target(data)
        string_dist = jp.linalg.norm(k["mid"] - target)
        err = jp.maximum(0.0, string_dist - self.string_tolerance) / self.string_tolerance
        return 1.0 / (1.0 + jp.square(err))

    def _reward_contact_penalty(self, k: Dict[str, jax.Array]) -> jax.Array:
        """Penalize the largest contact force in the scene above the soft
        safety threshold `f_safe` (not just force at the bow-mount sensor)."""
        return jp.maximum(-1, -jp.square(jp.maximum(0.0, k["max_contact_force"] - self.f_safe)))

    def _reward_forbidden(self, k: Dict[str, jax.Array]) -> jax.Array:
        """Penalize approaching the forbidden volume. Inert if disabled."""
        if not self.enable_forbidden_zone:
            return jp.asarray(0.0)
        return -jp.square(jp.maximum(0.0, self.forbidden_margin - k["forbidden_dist"]))

    def _reward_termination(self, terminated: jax.Array) -> jax.Array:
        """Flat penalty for any event that triggers `_termination`."""
        return jp.where(terminated, -100.0, 0.0)

    def _reward_survival(self, terminated: jax.Array) -> jax.Array:
        """Flat bonus of 1 for every step the agent survives (not the
        terminating step itself)."""
        return jp.where(terminated, 0.0, 1.0)

    def _reward_terms(
        self,
        data: mjx.Data,
        k: Dict[str, jax.Array],
        action: jax.Array,
        info: Dict[str, Any],
        terminated: jax.Array,
    ) -> Dict[str, jax.Array]:
        return {
            "velocity": self._reward_velocity(k),
            "pressure": self._reward_pressure(k),
            "contact_duration": self._reward_contact_duration(k),
            "action_smoothness": self._reward_action_smoothness(action, info),
            "bow_flat": self._reward_bow_flat(data, k),
            "bow_touch": self._reward_bow_touch_box(data, k),
            "string_center": self._reward_string_center(data, k),
            "contact_penalty": self._reward_contact_penalty(k),
            "forbidden": self._reward_forbidden(k),
            "termination": self._reward_termination(terminated),
            "survival": self._reward_survival(terminated),
        }

    # ------------------------------------------------------------------
    # Termination categories.
    # ------------------------------------------------------------------
    def _termination(self, data: mjx.Data, k: Dict[str, jax.Array]) -> jax.Array:
        force_exceeded = k["max_contact_force"] > self.f_max
        # bow_crossed_strings = k["touching"] & (k["min_dist"] < -self.clip_penetration_limit)
        bow_crossed_strings = jp.asarray(False)  # disabled for now to prevent early termination
        if self.enable_forbidden_zone:
            entered_forbidden = k["forbidden_dist"] < 0.0
        else:
            entered_forbidden = jp.asarray(False)
        time_up = data.time > self.episode_time_limit
        return force_exceeded | bow_crossed_strings | entered_forbidden | time_up

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        prev_data = state.pipeline_state
        # Per-episode randomized model fields sampled in reset(); carried
        # unchanged in info so the same randomized dynamics apply all
        # episode long. Merged onto the closure-captured base model here
        # rather than carrying a full mjx.Model through info -- see
        # `_effective_model`.
        model = self._effective_model(state.info["dr_params"])
        ctrl = jp.clip(
            prev_data.ctrl + action * self.max_ctrl_delta, self._ctrl_lo, self._ctrl_hi
        )
        data = self.pipeline_step(prev_data, ctrl, model=model)

        info: Dict[str, Any] = state.info.copy()
        k = self._step_kinematics(data, info, model)

        terminated = self._termination(data, k)
        terms = self._reward_terms(data, k, action, info, terminated)
        reward = sum(self.reward_weights[name] * term for name, term in terms.items())
        reward = reward.astype(jp.float32)

        done = terminated.astype(jp.float32)

        info["action_history"] = jp.concatenate(
            [info["action_history"][1:], action[None, :]], axis=0
        )
        info["force_history"] = jp.concatenate(
            [info["force_history"][1:], k["force"][None, :]], axis=0
        )
        info["prev_bow_mid"] = k["mid"]
        info["contact_steps"] = k["contact_steps"]
        info["bow_a_force_ema"] = k["bow_a_force_ema"]

        obs = self._get_obs(data, info, model)
        metrics = {
            "joint_positions": data.qpos,
            "force_mag": k["force_mag"],
            "max_contact_force": k["max_contact_force"],
            "bow_a_force": k["bow_a_force"],
            "bow_a_force_ema": k["bow_a_force_ema"],
            "bow_velocity": k["lateral_vel"],
            "velocity_error": k["lateral_vel"] - k["desired_velocity"],
            "pressure_error": k["bow_a_force_ema"] - k["desired_pressure"],
            "contact": k["touching"].astype(jp.float32),
            "reward_terms": terms,
        }

        return State(data, obs, reward, done, metrics, info)
