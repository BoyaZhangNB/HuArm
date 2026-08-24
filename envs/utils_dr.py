"""utils_dr.py

Per-episode domain randomization of the erhu task's *physics*: what the
policy acts on. `utils_noise.py` covers the other half -- what it sees.

Randomized every `reset()` (see `domain_randomize`):

    bow weight          one factor for the whole bow assembly, on top of the
                        per-body mass jitter applied to the rest of the model.
    actuator params     position-actuator gains (kp/kv) and the first-order
                        filter time constant that stands in for actuation
                        delay -- see `arm.xml`'s <actuator> block. (MuJoCo's
                        position actuator is a PD, not a PID: there is no
                        integral term to randomize.)
    erhu pose           where the instrument sits, drawn from the pre-solved
                        pool in `utils_envs.build_erhu_pose_pool`, plus a slow
                        drift over the episode -- see `sample_erhu_drift`.
    contact params      solref/solimp on the bow-hair/string contact pairs.
    string friction     the bow-hair/string pairs' sliding friction (and the
                        string geoms', for consistency), on an extra factor
                        of its own -- the parameter the bowing task hangs on.

The bow's own position/trajectory is randomized elsewhere: its start pose
comes from the (jittered, verified-threaded) pool entry drawn here, and the
reference stroke it is scored against is resampled continuously by
`utils_traj`.

Everything here is jax-native so it runs inside `reset()` under jit/vmap.
The result is a dict of `mjx.Model` field overrides, carried in
`State.info["dr_params"]` and merged onto the base model with
`mjx.Model.replace` -- rather than a whole randomized model per env.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .utils_envs import (
    quat_mul, quat_nlerp, quat_rotate, sample_erhu_pose_pool, set_joint_ctrl_jax,
)


BOW_BODY_NAMES = ("bow_frog", "bow_link_0", "bow_link_1", "bow_link_7", "bow_tip", "bow_hair")
STRING_GEOM_NAMES = ("string_D_geom", "string_A_geom")
HAIR_PAIR_NAMES = ("bow_hair_string_D_pair", "bow_hair_string_A_pair")


@dataclass(frozen=True)
class DomainRandConfig:
    """Ranges for `domain_randomize`. Every `*_range` field is a
    multiplicative factor on the model's nominal value, sampled uniformly
    and independently per element unless noted otherwise."""

    enabled: bool = True

    friction_range: Tuple[float, float] = (0.7, 1.3)        # all geoms/pairs
    string_friction_range: Tuple[float, float] = (0.7, 1.3)  # extra factor, hair<->string
    mass_range: Tuple[float, float] = (0.8, 1.2)            # all bodies
    bow_mass_range: Tuple[float, float] = (0.7, 1.3)        # extra factor, whole bow
    damping_range: Tuple[float, float] = (0.8, 1.2)         # joint damping
    actuator_gain_range: Tuple[float, float] = (0.85, 1.15)  # kp, per actuator
    actuator_damping_range: Tuple[float, float] = (0.8, 1.25)  # kv, per actuator
    actuator_delay_range: Tuple[float, float] = (0.6, 1.6)  # filter time constant
    # Contact compliance -- bow-hair/string pairs only, see `randomize_model`.
    solref_time_range: Tuple[float, float] = (0.7, 1.4)
    solref_damp_range: Tuple[float, float] = (0.8, 1.25)
    solimp_d_range: Tuple[float, float] = (0.8, 1.2)
    solimp_width_range: Tuple[float, float] = (0.6, 1.6)

    # Slow erhu drift -- see `sample_erhu_drift`. The cylinder spans the
    # instrument along its own +z axis, from the sound box (which rests on
    # the player's lap) up to the top of the neck; see huarm/erhu.xml.
    erhu_axis: Tuple[float, float] = (0.0, 0.6)             # local z, bottom/top
    erhu_drift_bottom_radius: float = 0.01                 # m
    erhu_drift_top_radius: float = 0.05                    # m
    erhu_drift_time_range: Tuple[float, float] = (8.0, 40.0)  # s to reach the target

    @classmethod
    def from_dict(cls, overrides: Mapping[str, Any] = None) -> "DomainRandConfig":
        """Builds a config from (partial) overrides, e.g. a training yaml's
        `env.dr_config` block. Ranges arrive from yaml as lists; they are
        converted to tuples so the config stays hashable/immutable."""
        overrides = dict(overrides or {})
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown domain-randomization option(s): {sorted(unknown)}")
        return cls(**{
            k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in overrides.items()
        })


def dr_indices(mj_model) -> Dict[str, Any]:
    """Static index arrays naming the parts `domain_randomize` treats
    specially, resolved once at env init instead of on every reset."""

    def ids(objtype, names):
        return jnp.asarray([mujoco.mj_name2id(mj_model, objtype, n) for n in names], dtype=jnp.int32)

    return {
        "bow_bodies": ids(mujoco.mjtObj.mjOBJ_BODY, BOW_BODY_NAMES),
        "string_geoms": ids(mujoco.mjtObj.mjOBJ_GEOM, STRING_GEOM_NAMES),
        "hair_pairs": ids(mujoco.mjtObj.mjOBJ_PAIR, HAIR_PAIR_NAMES),
    }


def _factor(rng: jax.Array, shape: Tuple[int, ...], value_range: Tuple[float, float]) -> jax.Array:
    return jax.random.uniform(rng, shape, minval=value_range[0], maxval=value_range[1])


def _randomize_solref(rng: jax.Array, solref: jax.Array, rows: jax.Array,
                      cfg: DomainRandConfig) -> jax.Array:
    """Randomizes the (time constant, damping ratio) columns of `rows`,
    independently, leaving every other row at its nominal value."""
    time_rng, damp_rng = jax.random.split(rng)
    n = rows.shape[0]
    solref = solref.at[rows, 0].multiply(_factor(time_rng, (n,), cfg.solref_time_range))
    return solref.at[rows, 1].multiply(_factor(damp_rng, (n,), cfg.solref_damp_range))


def _randomize_solimp(rng: jax.Array, solimp: jax.Array, rows: jax.Array,
                      cfg: DomainRandConfig) -> jax.Array:
    """Randomizes the (d0, d1, width) columns of `rows`, leaving
    midpoint/power -- and every other row -- alone. d0/d1 are impedances:
    they must stay inside (0, 1) and ordered, which a raw multiplicative
    factor does not guarantee, hence the clips."""
    d0_rng, d1_rng, width_rng = jax.random.split(rng, 3)
    n = rows.shape[0]
    d0 = jnp.clip(solimp[rows, 0] * _factor(d0_rng, (n,), cfg.solimp_d_range), 1e-4, 0.9999)
    d1 = jnp.clip(solimp[rows, 1] * _factor(d1_rng, (n,), cfg.solimp_d_range), d0, 0.9999)
    width = solimp[rows, 2] * _factor(width_rng, (n,), cfg.solimp_width_range)
    return solimp.at[rows, 0].set(d0).at[rows, 1].set(d1).at[rows, 2].set(width)


def randomize_model(mjx_model, rng: jax.Array, cfg: DomainRandConfig,
                    idx: Mapping[str, Any]) -> Dict[str, jax.Array]:
    """Samples this episode's `mjx.Model` field overrides (everything except
    the erhu placement, which comes from the pose pool)."""
    keys = jax.random.split(rng, 10)
    m = mjx_model

    # Sliding friction, per element, with an extra shared factor on the
    # bow-hair/string contacts -- the parameter the whole bowing task hangs
    # on, so it gets a range of its own on top. Those contacts are declared
    # as explicit <pair>s, whose friction overrides the geoms', so the
    # string friction that actually bites lives in `pair_friction`; the
    # string geoms take the same factor only to keep the two consistent.
    string_factor = _factor(keys[0], (), cfg.string_friction_range)
    geom_friction = m.geom_friction * _factor(keys[1], m.geom_friction.shape, cfg.friction_range)
    geom_friction = geom_friction.at[idx["string_geoms"]].multiply(string_factor)
    pair_friction = m.pair_friction * _factor(keys[2], m.pair_friction.shape, cfg.friction_range)
    pair_friction = pair_friction.at[idx["hair_pairs"]].multiply(string_factor)

    # Mass, with one extra factor shared across the bow assembly (a bow is a
    # single physical object -- its parts do not get heavier independently).
    # Inertia is scaled with mass so the bodies stay physically consistent.
    mass_factor = _factor(keys[3], m.body_mass.shape, cfg.mass_range)
    mass_factor = mass_factor.at[idx["bow_bodies"]].multiply(
        _factor(keys[4], (), cfg.bow_mass_range))
    body_mass = m.body_mass * mass_factor
    body_inertia = m.body_inertia * mass_factor[:, None]

    dof_damping = m.dof_damping * _factor(keys[5], m.dof_damping.shape, cfg.damping_range)

    # Position actuators: force = gainprm[0]*act + biasprm[1]*qpos +
    # biasprm[2]*qvel, i.e. kp*(act - qpos) - kv*qvel, so gainprm[0] and
    # biasprm[1] are the *same* gain and have to be scaled together.
    # dynprm[0] is the ctrl low-pass time constant -- the actuation delay.
    nu = m.actuator_gainprm.shape[0]
    kp_factor = _factor(keys[6], (nu,), cfg.actuator_gain_range)
    actuator_gainprm = m.actuator_gainprm.at[:, 0].multiply(kp_factor)
    actuator_biasprm = m.actuator_biasprm.at[:, 1].multiply(kp_factor)
    actuator_biasprm = actuator_biasprm.at[:, 2].multiply(
        _factor(keys[7], (nu,), cfg.actuator_damping_range))
    actuator_dynprm = m.actuator_dynprm.at[:, 0].multiply(
        _factor(keys[8], (nu,), cfg.actuator_delay_range))

    # Contact compliance, on the bow-hair/string pairs only. That contact is
    # the task, and its solimp was hand-tuned in arm.xml to make the hair
    # behave as a spring rather than a switch; everything else in the scene
    # is either an anti-tunneling backstop or a surface the bow should never
    # reach, so randomizing it would perturb the sim without teaching the
    # policy anything.
    contact_rng = jax.random.split(keys[9], 2)
    hair_pairs = idx["hair_pairs"]
    return dict(
        geom_friction=geom_friction,
        pair_friction=pair_friction,
        body_mass=body_mass,
        body_inertia=body_inertia,
        dof_damping=dof_damping,
        actuator_gainprm=actuator_gainprm,
        actuator_biasprm=actuator_biasprm,
        actuator_dynprm=actuator_dynprm,
        pair_solref=_randomize_solref(contact_rng[0], m.pair_solref, hair_pairs, cfg),
        pair_solimp=_randomize_solimp(contact_rng[1], m.pair_solimp, hair_pairs, cfg),
    )


def _disk_point(rng: jax.Array, radius: float, height: float) -> jax.Array:
    """Uniform sample on the disk of radius `radius` at local z = `height`."""
    radius_rng, angle_rng = jax.random.split(rng)
    r = radius * jnp.sqrt(jax.random.uniform(radius_rng, ()))
    theta = jax.random.uniform(angle_rng, (), minval=0.0, maxval=2.0 * jnp.pi)
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta), height])


def _shortest_arc_quat(axis: jax.Array) -> jax.Array:
    """Quaternion of the smallest rotation taking +z onto the unit vector
    `axis`. `axis` stays within a few degrees of +z here (the drift radii are
    centimetres over a 0.6 m instrument), so the antipodal degenerate case
    is not reachable."""
    w = 1.0 + axis[2]
    xyz = jnp.array([-axis[1], axis[0], 0.0])  # cross([0,0,1], axis)
    quat = jnp.concatenate([jnp.reshape(w, (1,)), xyz])
    return quat / (jnp.linalg.norm(quat) + 1e-9)


def sample_erhu_drift(rng: jax.Array, cfg: DomainRandConfig) -> Dict[str, jax.Array]:
    """Samples the episode's slow erhu drift, as a target offset/rotation in
    the erhu's own frame plus the time it takes to get there.

    A real player's instrument is not bolted down: it rocks and slides over
    the course of a phrase. Model that as a cylinder around the erhu -- one
    point sampled on its top cap, one on its bottom cap -- and a target pose
    that brings the instrument's top and bottom centres as close as possible
    to those two points. With only two correspondences that best-fit rigid
    transform is exact and closed-form: align the axis directions, then the
    midpoints. `erhu_pose_at` walks the erhu there over the episode.
    """
    bottom_rng, top_rng, time_rng = jax.random.split(rng, 3)
    z_bottom, z_top = cfg.erhu_axis
    p_bottom = _disk_point(bottom_rng, cfg.erhu_drift_bottom_radius, z_bottom)
    p_top = _disk_point(top_rng, cfg.erhu_drift_top_radius, z_top)

    axis = p_top - p_bottom
    rot = _shortest_arc_quat(axis / (jnp.linalg.norm(axis) + 1e-9))
    centre = jnp.array([0.0, 0.0, 0.5 * (z_bottom + z_top)])
    offset = 0.5 * (p_bottom + p_top) - quat_rotate(rot, centre)

    return {
        "offset": offset,
        "rot": rot,
        "duration": _factor(time_rng, (), cfg.erhu_drift_time_range),
    }


def no_erhu_drift() -> Dict[str, jax.Array]:
    """The identity drift, for `enabled=False` runs -- same pytree structure
    so `State.info` stays a consistent shape either way."""
    return {
        "offset": jnp.zeros(3),
        "rot": jnp.array([1.0, 0.0, 0.0, 0.0]),
        "duration": jnp.asarray(1.0),
    }


def erhu_pose_at(base_pos: jax.Array, base_quat: jax.Array,
                 drift: Mapping[str, jax.Array], time: jax.Array
                 ) -> Tuple[jax.Array, jax.Array]:
    """The erhu's body_pos/body_quat at `time`, drifting from the episode's
    start pose (`base_pos`, `base_quat`) toward the sampled target and
    holding there once the drift's `duration` has elapsed.

    The drift is expressed relative to the start pose rather than as an
    absolute target so that this is the identity at time 0 -- a recorded
    episode replayed from its `dr_params` starts exactly where it did.

    The erhu is welded to the world, so moving it means moving the model
    rather than integrating a body: it has no velocity of its own in the
    sim. At these rates (centimetres over tens of seconds, i.e. microns per
    physics timestep) that is well inside what the contact solver treats as
    quasi-static.
    """
    s = jnp.clip(time / drift["duration"], 0.0, 1.0)
    quat = quat_mul(base_quat, quat_nlerp(jnp.array([1.0, 0.0, 0.0, 0.0]), drift["rot"], s))
    pos = base_pos + quat_rotate(base_quat, s * drift["offset"])
    return pos, quat


def domain_randomize(mjx_model, mjx_data, rng: jax.Array, erhu_pose_pool,
                     arm_idxs, dr_idxs: Mapping[str, Any], cfg: DomainRandConfig):
    """Draws one episode's randomization and returns `(dr_params, drift,
    data)`: the `mjx.Model` field overrides, the erhu's drift for the
    episode (see `sample_erhu_drift`), and the initial `mjx.Data` placed at
    the drawn arm pose.

    `arm_idxs` is `utils_envs.arm_joint_indices(mj_model)` and `dr_idxs` is
    `dr_indices(mj_model)`, both precomputed once at env init to keep the
    underlying `mj_name2id` lookups off the reset path.

    The arm's start pose is *not* jittered here: the poses in the pool are
    threaded through a corridor barely 12 mm wide, a few mrad on each joint
    lifts the hair clean out of the strings, and nothing inside jit can
    check for that. The equivalent jitter is applied -- and verified -- while
    the pool is built instead (see `utils_envs.build_erhu_pose_pool`).
    """
    model_rng, pool_rng, drift_rng = jax.random.split(rng, 3)
    qpos_idxs, ctrl_aids, ctrl_qpos_idxs, act_idxs = arm_idxs

    pose = sample_erhu_pose_pool(erhu_pose_pool, pool_rng)
    dr_params = dict(body_pos=pose["body_pos"], body_quat=pose["body_quat"])
    if cfg.enabled:
        dr_params.update(randomize_model(mjx_model, model_rng, cfg, dr_idxs))
        drift = sample_erhu_drift(drift_rng, cfg)
    else:
        drift = no_erhu_drift()

    mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[qpos_idxs].set(pose["arm_qpos"]))
    mjx_data = mjx.forward(mjx_model.replace(**dr_params), mjx_data)
    mjx_data = set_joint_ctrl_jax(mjx_data, ctrl_aids, ctrl_qpos_idxs, act_idxs)

    return dr_params, drift, mjx_data
