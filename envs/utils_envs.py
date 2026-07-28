import functools
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco


# -----------------------------------------------------------------------------
# 1. Helper: Pre-compute ID lookups on CPU
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HairArmIds:
    """
    Same information the old code kept in a plain `id_dict` dict, just moved
    into a frozen dataclass of ints/tuples-of-ints.

    This is unrelated to the flexcomp -> rigid-body-chain refactor, but is
    needed for it to actually run: `init_huarm` is jax.jit'd with
    `static_argnames=["id_dict"]`, and a plain dict (especially one holding
    jnp.array values) is not hashable, so every call raised
    `TypeError: unhashable type: 'dict'` before it ever got to the
    hair-specific code below. A frozen dataclass with only int/tuple fields
    is hashable, so it can safely be a static (compile-time-constant) jit
    argument; anywhere the old code needed a jnp.array for fancy indexing
    (`.at[idxs].set(...)`), we now build that array from the static tuple at
    the point of use -- it gets baked in as a compile-time constant, same as
    before, just without tripping the jit cache-key hash.
    """
    hair_eq_ids: Tuple[int, ...]
    string_D: int
    string_A: int
    sound_box: int
    sound_box_geom: int
    bow_link_0: int
    bow_tip: int
    arm_qpos_idxs: Tuple[int, ...]
    arm_dof_idxs: Tuple[int, ...]
    hair_body_ids: Tuple[int, ...]
    # Rigid bow-hair chain addressing (see build_id_dict for why these
    # replace the old flexcomp particle addressing).
    hair_root_qpos_adr: int
    hair_root_dof_adr: int
    hair_hinge_qpos_idxs: Tuple[int, ...]
    hair_hinge_dof_idxs: Tuple[int, ...]


def build_id_dict(mj_model: mujoco.MjModel, arm_joint_names=("joint1", "joint2", "joint3", "joint4"), n_hair_vertices=36) -> HairArmIds:
    """Computes all string/body/joint lookups ONCE on CPU ahead of tracing."""
    hair_eq_ids = []
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            hair_eq_ids.append(eq_id)

    # NOTE: the old flexcomp hair had an auto-generated (unnamed) edge-length
    # equality constraint enforcing every segment's length, which used to get
    # collected here as `flex_edge_ids` and disabled alongside the
    # hair_to_frog/hair_to_tip welds before snapping the hair taut. The
    # rigid-body hair chain has no such constraint -- its segments are rigid
    # capsules, not elastic edges -- so there's nothing left to collect, and
    # `hair_eq_ids` alone is now the full set of constraints that needs to be
    # toggled around the taut-snap.

    arm_qpos_idxs = []
    arm_dof_idxs = []
    for jn in arm_joint_names:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        arm_qpos_idxs.append(int(mj_model.jnt_qposadr[jid]))
        arm_dof_idxs.append(int(mj_model.jnt_dofadr[jid]))

    hair_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"bow_hair_{i}")
        for i in range(n_hair_vertices)
    ]

    # The old flexcomp hair was 36 particles, each with 3 independent slide
    # joints -- snap_hair_taut() could set every particle's world position
    # directly. The rigid-body chain is a real kinematic chain instead:
    #   - bow_hair_0 has one free joint (3 pos + 4 quat qpos, 6 dof) -- its
    #     "attachment" to the rest of the bow is really the hair_to_frog
    #     equality constraint, not a literal parent body.
    #   - bow_hair_1 .. bow_hair_{n-1} each have 3 hinge joints
    #     ("_bend_y", "_bend_z", "_twist_x") for that link's rotation
    #     relative to the previous one.
    # So instead of "36 independent qpos triples", positioning the whole
    # chain now means setting the root's free-joint pose and zeroing the
    # interior hinge angles. Precompute those addresses here (static, CPU
    # side) so snap_hair_taut can stay a plain vectorized .at[].set() under
    # jit, with no Python-level loop over hair vertices.
    root_bid = hair_body_ids[0]
    root_jadr = mj_model.body_jntadr[root_bid]
    hair_root_qpos_adr = int(mj_model.jnt_qposadr[root_jadr])
    hair_root_dof_adr = int(mj_model.jnt_dofadr[root_jadr])

    hinge_qpos_idxs = []
    hinge_dof_idxs = []
    for i in range(1, n_hair_vertices):
        for suffix in ("bend_y", "bend_z", "twist_x"):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"bow_hair_{i}_{suffix}")
            hinge_qpos_idxs.append(int(mj_model.jnt_qposadr[jid]))
            hinge_dof_idxs.append(int(mj_model.jnt_dofadr[jid]))

    return HairArmIds(
        hair_eq_ids=tuple(hair_eq_ids),
        string_D=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "string_D"),
        string_A=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "string_A"),
        sound_box=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "sound_box"),
        sound_box_geom=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom"),
        bow_link_0=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0"),
        bow_tip=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip"),
        arm_qpos_idxs=tuple(arm_qpos_idxs),
        arm_dof_idxs=tuple(arm_dof_idxs),
        hair_body_ids=tuple(hair_body_ids),
        hair_root_qpos_adr=hair_root_qpos_adr,
        hair_root_dof_adr=hair_root_dof_adr,
        hair_hinge_qpos_idxs=tuple(hinge_qpos_idxs),
        hair_hinge_dof_idxs=tuple(hinge_dof_idxs),
    )


# -----------------------------------------------------------------------------
# 2. Refactored Pure-JAX Functions
# -----------------------------------------------------------------------------
def fix_hair_anchor_offsets(model: mjx.Model, hair_eq_ids: jax.Array) -> mjx.Model:
    eq_data = model.eq_data
    # Use .at[].set() for functional array updates
    eq_data = eq_data.at[hair_eq_ids, 3:6].set(0.0)
    return model.replace(eq_data=eq_data)


def get_string_contact_point(data: mjx.Data, string_id: int) -> jax.Array:
    xmat = data.xmat[string_id].reshape(3, 3)
    local_tip = jnp.array([0.0, 0.0, -0.6])
    return data.xpos[string_id] + xmat @ local_tip


def between_strings_target(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> jax.Array:
    pt_d = get_string_contact_point(data, id_dict.string_D)
    pt_a = get_string_contact_point(data, id_dict.string_A)
    midpoint = (pt_d + pt_a) / 2.0

    sound_box_top_z = data.xpos[id_dict.sound_box][2] + model.geom_size[id_dict.sound_box_geom][0]

    target_z = jnp.maximum(midpoint[2], sound_box_top_z) + 0.01
    return midpoint.at[2].set(target_z)


def _ik_point_fn(q_arm: jax.Array, model: mjx.Model, data: mjx.Data, body_ids: Tuple[int, int], weights: Tuple[float, float], qpos_idxs: jax.Array) -> jax.Array:
    """Helper that evaluates FK for given arm positions to compute weighted end-effector location."""
    updated_qpos = data.qpos.at[qpos_idxs].set(q_arm)
    data_temp = data.replace(qpos=updated_qpos)
    data_temp = mjx.kinematics(model, data_temp)
    return weights[0] * data_temp.xpos[body_ids[0]] + weights[1] * data_temp.xpos[body_ids[1]]


def jacobian_ik(model: mjx.Model, data: mjx.Data, body_ids: Tuple[int, int], weights: Tuple[float, float], target_pos: jax.Array, qpos_idxs: jax.Array, max_iters=4, damping=1e-2, step_clip=0.1) -> jax.Array:
    """JAX-native IK solver using forward-mode Autodiff for the Jacobian and fori_loop for convergence."""
    q_arm_init = data.qpos[qpos_idxs]

    def ik_step(i, q_arm):
        pt = _ik_point_fn(q_arm, model, data, body_ids, weights, qpos_idxs)
        # Compute exact Jacobian (3 x N_dof) via JAX autodiff
        J = jax.jacfwd(_ik_point_fn, argnums=0)(q_arm, model, data, body_ids, weights, qpos_idxs)
        err = target_pos - pt

        # Damped least squares
        reg = (damping**2) * jnp.eye(3)
        dtheta = J.T @ jnp.linalg.solve(J @ J.T + reg, err)

        # Step clipping
        step_norm = jnp.linalg.norm(dtheta)
        scale = jnp.where(step_norm > step_clip, step_clip / (step_norm + 1e-8), 1.0)
        return q_arm + dtheta * scale

    q_arm_final = jax.lax.fori_loop(0, max_iters, ik_step, q_arm_init)

    # Apply final IK solution to qpos and ctrl
    qpos_new = data.qpos.at[qpos_idxs].set(q_arm_final)
    ctrl_new = data.ctrl.at[qpos_idxs].set(q_arm_final)
    data = data.replace(qpos=qpos_new, ctrl=ctrl_new)
    return mjx.kinematics(model, data)


def _quat_aligning(v_from: jax.Array, v_to: jax.Array) -> jax.Array:
    """
    Branchless (jit-safe) shortest-arc quaternion (w, x, y, z) rotating unit
    vector v_from onto v_to.
    """
    v_from = v_from / jnp.linalg.norm(v_from)
    v_to = v_to / jnp.linalg.norm(v_to)
    dot = jnp.clip(jnp.dot(v_from, v_to), -1.0, 1.0)

    axis_cross = jnp.cross(v_from, v_to)
    axis_cross_norm = jnp.linalg.norm(axis_cross)

    # Fallback axis for the (near-)antiparallel case: any unit vector
    # orthogonal to v_from.
    fallback_axis = jnp.cross(v_from, jnp.array([1.0, 0.0, 0.0]))
    fallback_axis = jnp.where(
        jnp.linalg.norm(fallback_axis) < 1e-6,
        jnp.cross(v_from, jnp.array([0.0, 1.0, 0.0])),
        fallback_axis,
    )
    fallback_axis = fallback_axis / (jnp.linalg.norm(fallback_axis) + 1e-12)

    safe_axis = jnp.where(axis_cross_norm < 1e-9, fallback_axis, axis_cross)
    safe_axis = safe_axis / (jnp.linalg.norm(safe_axis) + 1e-12)

    angle = jnp.arccos(dot)
    quat_general = jnp.concatenate([jnp.cos(angle / 2.0)[None], safe_axis * jnp.sin(angle / 2.0)])
    quat_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    quat_antiparallel = jnp.concatenate([jnp.array([0.0]), fallback_axis])

    is_parallel = dot > 1.0 - 1e-9
    is_antiparallel = dot < -1.0 + 1e-9

    quat = jnp.where(is_parallel, quat_identity, quat_general)
    quat = jnp.where(is_antiparallel, quat_antiparallel, quat)
    return quat


def snap_hair_taut(model: mjx.Model, data: mjx.Data, rear_target: jax.Array, tip_target: jax.Array, id_dict: HairArmIds) -> mjx.Data:
    """
    Rigid-chain replacement for the old per-particle straight-line snap.
    """
    root_qpos_adr = id_dict.hair_root_qpos_adr
    root_dof_adr = id_dict.hair_root_dof_adr
    hinge_qpos_idxs = jnp.array(id_dict.hair_hinge_qpos_idxs, dtype=jnp.int32)
    hinge_dof_idxs = jnp.array(id_dict.hair_hinge_dof_idxs, dtype=jnp.int32)

    direction = tip_target - rear_target
    length = jnp.linalg.norm(direction)
    direction = direction / jnp.where(length > 1e-9, length, 1.0)
    direction = jnp.where(length > 1e-9, direction, jnp.array([1.0, 0.0, 0.0]))

    quat = _quat_aligning(jnp.array([1.0, 0.0, 0.0]), direction)

    qpos = data.qpos
    qpos = qpos.at[root_qpos_adr:root_qpos_adr + 3].set(rear_target)
    qpos = qpos.at[root_qpos_adr + 3:root_qpos_adr + 7].set(quat)
    qpos = qpos.at[hinge_qpos_idxs].set(0.0)

    qvel = data.qvel
    qvel = qvel.at[root_dof_adr:root_dof_adr + 6].set(0.0)
    qvel = qvel.at[hinge_dof_idxs].set(0.0)

    return data.replace(qpos=qpos, qvel=qvel)


def pretension_bow_hair(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> mjx.Data:
    """
    Disable the hair-anchoring welds, lay the rigid hair chain out straight
    and taut between bow_link_0's and bow_tip's current positions, then
    re-enable the welds so the two end connect-constraints take up any
    residual gap (see snap_hair_taut for why that's where the pretension
    comes from in this rigid-body version).
    """
    eq_indices = jnp.array(id_dict.hair_eq_ids, dtype=jnp.int32)
    rear_target = data.xpos[id_dict.bow_link_0]
    tip_target = data.xpos[id_dict.bow_tip]

    # 1. Disable constraints
    data = data.replace(eq_active=data.eq_active.at[eq_indices].set(0))

    # 2. Snap taut & run forward kinematics
    data = snap_hair_taut(model, data, rear_target, tip_target, id_dict)
    data = mjx.forward(model, data)

    # 3. Re-enable constraints & re-run forward kinematics
    data = data.replace(eq_active=data.eq_active.at[eq_indices].set(1))
    return mjx.forward(model, data)


# -----------------------------------------------------------------------------
# 3. Main Init Entry Point
# -----------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=["id_dict"])
def init_huarm(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> Tuple[mjx.Model, mjx.Data]:
    # Update model (anchor offsets)
    model = fix_hair_anchor_offsets(model, jnp.array(id_dict.hair_eq_ids, dtype=jnp.int32))

    # Target calculation & IK
    target = between_strings_target(model, data, id_dict)
    body_ids = (id_dict.bow_link_0, id_dict.bow_tip)
    data = jacobian_ik(model, data, body_ids, (0.5, 0.5), target, jnp.array(id_dict.arm_qpos_idxs, dtype=jnp.int32))

    # Pretension hair
    data = pretension_bow_hair(model, data, id_dict)

    return model, data