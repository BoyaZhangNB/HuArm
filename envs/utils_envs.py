import functools
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
import numpy as np


def fix_hair_anchor_offsets(model):
    """
    hair_to_frog/hair_to_tip are <connect> constraints between the bow-hair
    chain's two end bodies (bow_hair_0, bow_hair_35) and points on the bow
    stick (bow_link_0, bow_tip). Because the hair chain is authored at its
    own XML position, independent of the bow's geometry, the two bodies are
    NOT coincident at compile time (bow_hair_0 sits some distance from
    bow_link_0, bow_hair_35 a similar distance from bow_tip). MuJoCo's
    compiler auto-derives each connect's "anchor in body2's frame" from that
    compile-time offset -- meaning the constraint's real target is "stay
    offset from body2 by the original gap", not "coincide with body2". That's
    what was silently undoing any runtime teleport/pretension. Zeroing the
    body2-frame anchor (eq_data[3:6]) makes the constraint actually pin the
    two points together.

    This bug/fix is unchanged by the flexcomp -> rigid-body-chain refactor:
    the hair is still connected to the bow via the same two named "connect"
    equality constraints, just with a rigid capsule chain hanging off
    bow_hair_0 instead of a flex.
    """
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            model.eq_data[eq_id][3:6] = 0.0

def get_string_contact_point(model, data, string_body_name):
    """
    Returns the world position of the FREE (bottom) end of a string capsule --
    the point near the sound box where bowing contact happens -- NOT the
    body's own xpos, which is the pivot at the TOP of the string (near the
    tuning peg, ~0.6m away from where the string actually crosses the box).
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, string_body_name)
    xmat = data.xmat[bid].reshape(3, 3)
    # The string geom is centered at local (0,0,-0.3) with half-length 0.3,
    # so its far (bottom) end is at local (0,0,-0.6).
    local_tip = np.array([0.0, 0.0, -0.6])
    return data.xpos[bid] + xmat @ local_tip


def between_strings_target(model, data):
    """Midpoint between the two strings' contact points, lifted just clear of
    the sound box surface."""
    midpoint = (get_string_contact_point(model, data, "string_D")
                + get_string_contact_point(model, data, "string_A")) / 2.0

    # The sound box is a cylinder lying on its side (axis along world X), so
    # its surface in z sits at sound_box_z + radius, not at its center z.
    sound_box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sound_box")
    sound_box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom")
    sound_box_top_z = data.xpos[sound_box_id][2] + model.geom_size[sound_box_geom_id][0]
    midpoint[2] = max(midpoint[2], sound_box_top_z) + 0.01  # 1cm clearance
    return midpoint


def weighted_point_and_jacobian(model, data, body_weights, dof_idxs):
    """
    body_weights: list of (body_name, weight), weights summing to 1.
    Returns the weighted-average world position of those bodies' origins and
    the corresponding position Jacobian restricted to dof_idxs (both position
    and Jacobian are linear in body position, so a weighted sum of bodies'
    positions/Jacobians is valid).
    """
    point = np.zeros(3)
    J = np.zeros((3, len(dof_idxs)))
    for name, w in body_weights:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        point += w * data.xpos[bid]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid], bid)
        J += w * jacp[:, dof_idxs]
    return point, J


def jacobian_ik(model, data, body_weights, target_pos, joint_names,
                max_iters=200, damping=1e-2, step_clip=0.1, tol=1e-4):
    """
    Damped-least-squares IK: iteratively adjusts qpos for `joint_names`
    (each assumed to be a 1-dof hinge/slide joint) so that the weighted
    reference point defined by `body_weights` reaches target_pos. This
    directly teleports qpos (as opposed to driving it through actuators over
    time), matching bodies that are part of the arm's actual kinematic tree
    -- unlike the bow_hair chain's bodies, bow_link_0/bow_tip ARE rigid
    bodies in that tree, so their Jacobian w.r.t. the arm joints is
    well-defined and nonzero.
    """
    dof_idxs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in joint_names]
    qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in joint_names]

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        point, J = weighted_point_and_jacobian(model, data, body_weights, dof_idxs)
        err = target_pos - point
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return it, err_norm
        # damped least squares: dtheta = J^T (J J^T + lambda^2 I)^-1 err
        dtheta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        step_norm = np.linalg.norm(dtheta)
        if step_norm > step_clip:
            dtheta *= step_clip / step_norm
        for qidx, d in zip(qpos_idxs, dtheta):
            data.qpos[qidx] += d

    mujoco.mj_forward(model, data)
    point, _ = weighted_point_and_jacobian(model, data, body_weights, dof_idxs)
    return max_iters, np.linalg.norm(target_pos - point)


def set_joint_ctrl(model, data, joint_names):
    """
    Set the control inputs for the specified joints to their current qpos
    values, effectively "locking" them in place.
    """
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = joint_to_actuator_id(model, jn)
        if aid >= 0:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]


def joint_to_actuator_id(model, joint_name):
    """
    Looks up the position actuator driving `joint_name`. arm.xml's actuators
    are declared without an explicit "name" attribute (<position
    joint="joint1"/> etc.), so they can't be found by name -- we instead scan
    actuator_trnid for the one whose joint transmission target matches.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return -1
    for aid in range(model.nu):
        if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[aid, 0] == jid):
            return aid
    return -1


def insert_hair_between_strings(model, data, arm_joint_names=("joint1", "joint2", "joint3", "joint4")):
    """
    Move the bow -- via the arm's joints, using Jacobian-based IK -- so the
    taut hair passes between the two erhu strings, resting just above the
    sound box.

    bow_link_0 and bow_tip ARE rigid bodies in the arm's kinematic chain
    (bow_frog is welded to the arm's end effector, and the bow stick hangs
    off bow_frog), so a real position Jacobian w.r.t. the arm joints exists
    for them. The bow_hair_i bodies, by contrast, are their own separate
    kinematic chain (bow_hair_0 has its own free joint; bow_hair_1..35 are
    its children), only coupled to the bow stick through the
    hair_to_frog/hair_to_tip <connect> equalities -- there's no meaningful
    Jacobian relating a hair link's position to the arm's joints, so we
    still can't target a hair vertex directly here (same reasoning as the
    old flexcomp version, just for a different underlying representation).

    We solve IK to place the *midpoint* of bow_link_0/bow_tip at the target
    (approximating "some point along the taut hair", since the hair runs
    approximately straight between them), then re-run pretension_bow_hair()
    so the hair chain is laid out fresh between the endpoints' new
    positions -- which by then are already close to the target, so no
    large/unstable reconfiguration is needed.
    """
    target = between_strings_target(model, data)
    print(f"Target insertion point (between strings, above sound box): {target}")

    body_weights = [("bow_link_0", 0.5), ("bow_tip", 0.5)]
    iters, err = jacobian_ik(model, data, body_weights, target, list(arm_joint_names))
    print(f"Arm IK converged in {iters} iterations, final position error {err:.6f} m")
    set_joint_ctrl(model, data, arm_joint_names)

    bow_link_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0")
    bow_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip")
    midpoint = 0.5 * (data.xpos[bow_link_0_id] + data.xpos[bow_tip_id])
    print(f"Bow midpoint after IK + pretension: {midpoint} "
          f"(target was {target}, residual {np.linalg.norm(midpoint - target):.4f} m)")


def hair_chain_joint_addrs(model, n_vertices=36):
    """
    Replaces the old particle_qpos_addrs(): the bow_hair_N bodies used to be
    flexcomp cable "particles", each with 3 independent slide joints (x/y/z
    translation) that let every node be positioned directly. The rigid-body
    chain instead looks like this:
      - bow_hair_0 has a single free joint (3 position + 4 quaternion qpos,
        6 dof) -- it's the one body in the chain not fully determined by its
        parent, since its "parent" connection is really the hair_to_frog
        equality constraint, not a literal kinematic-tree parent.
      - bow_hair_1 .. bow_hair_{n-1} each have 3 hinge joints
        ("_bend_y", "_bend_z", "_twist_x") expressing that link's rotation
        relative to the previous one.
    So instead of "one world-space qpos triple per node", we now get "one
    free pose for the root plus one relative rotation triple per link" --
    positioning the whole chain means setting the root pose and the interior
    joint angles, not each body's position independently.
    """
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_hair_0")
    root_jadr = model.body_jntadr[root_bid]
    root_qadr = model.jnt_qposadr[root_jadr]
    root_dadr = model.jnt_dofadr[root_jadr]

    hinge_qadrs = []
    hinge_dadrs = []
    for i in range(1, n_vertices):
        for suffix in ("bend_y", "bend_z", "twist_x"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"bow_hair_{i}_{suffix}")
            hinge_qadrs.append(model.jnt_qposadr[jid])
            hinge_dadrs.append(model.jnt_dofadr[jid])

    return root_qadr, root_dadr, hinge_qadrs, hinge_dadrs


def _quat_aligning(v_from, v_to):
    """Shortest-arc quaternion (w, x, y, z) rotating unit vector v_from onto v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        # 180 degree rotation: any axis orthogonal to v_from will do
        axis = np.cross(v_from, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(v_from, v_to)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])


def snap_hair_taut(model, data, rear_target, tip_target, n_vertices=36):
    """
    Rigid-chain replacement for the old per-particle straight-line snap.

    The hair is now a single kinematic chain (bow_hair_0's free joint plus
    35 downstream bend/twist hinges), so we can't independently place each
    link's world position the way the flexcomp's 3 translational DOFs per
    particle allowed. Instead:
      1. zero every interior bend/twist joint -> the whole chain becomes
         perfectly straight (each link just extends its parent along local
         +x by the link length), and
      2. point bow_hair_0's free-joint orientation so that its local +x axis
         (the direction the capsule chain extends in) lines up with
         (tip_target - rear_target), and place its free-joint position at
         rear_target.

    That reproduces "hair laid out taut along a straight line between the
    frog and tip" for a rigid chain. Because the chain's total length
    (36 * 0.0155 m = 0.558 m) is fixed, this generally won't land the last
    link exactly on tip_target if rear_target/tip_target are closer or
    farther apart than that -- the hair_to_frog/hair_to_tip <connect>
    equalities (re-enabled by the caller) take up any residual gap by
    bending the chain's springy hinges, which is also where the hair's
    "pretension" now comes from, since a rigid chain can't be pre-stretched
    axially the way the old flex could.
    """
    root_qadr, root_dadr, hinge_qadrs, hinge_dadrs = hair_chain_joint_addrs(model, n_vertices)

    direction = tip_target - rear_target
    length = np.linalg.norm(direction)
    direction = direction / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])

    data.qpos[root_qadr:root_qadr + 3] = rear_target
    data.qpos[root_qadr + 3:root_qadr + 7] = _quat_aligning(np.array([1.0, 0.0, 0.0]), direction)
    data.qvel[root_dadr:root_dadr + 6] = 0.0

    for qadr in hinge_qadrs:
        data.qpos[qadr] = 0.0
    for dadr in hinge_dadrs:
        data.qvel[dadr] = 0.0


def pretension_bow_hair(model, data):
    """
    Disable the hair-anchoring welds, lay the rigid hair chain out straight
    and taut between bow_link_0's and bow_tip's current positions, then
    re-enable the welds so the two end connect-constraints take up any
    residual gap (see snap_hair_taut for why that's where the pretension
    comes from in this rigid-body version).

    The old version also had to disable the flexcomp's auto-generated
    (unnamed) edge-length equality constraint before snapping vertices, since
    it was always active and would otherwise fight the teleport. There's no
    equivalent constraint here -- the hair segments are rigid capsules, not
    elastic edges -- so that step is simply gone.
    """
    frog_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_frog")
    tip_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_tip")

    bow_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip")
    bow_link_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0")
    rear_target = data.xpos[bow_link_0_id].copy()
    tip_target = data.xpos[bow_tip_id].copy()

    data.eq_active[frog_weld_id] = 0
    data.eq_active[tip_weld_id] = 0

    snap_hair_taut(model, data, rear_target, tip_target)
    mujoco.mj_forward(model, data)

    data.eq_active[frog_weld_id] = 1
    data.eq_active[tip_weld_id] = 1
    mujoco.mj_forward(model, data)


def init_huarm(model, data, id_dict):
    """
    Prepares the model/data for a new episode. This is called once at env
    reset, and again after every episode ends (so the next episode starts
    with a fresh hair pretension).
    """
    fix_hair_anchor_offsets(model)
    insert_hair_between_strings(model, data)
    pretension_bow_hair(model, data)
    mujoco.mj_forward(model, data)
    return model, data

# ==============================================================================

# # -----------------------------------------------------------------------------
# # 1. Helper: Pre-compute ID lookups on CPU
# # -----------------------------------------------------------------------------
# @dataclass(frozen=True)
# class HairArmIds:
#     """
#     Same information the old code kept in a plain `id_dict` dict, just moved
#     into a frozen dataclass of ints/tuples-of-ints.

#     This is unrelated to the flexcomp -> rigid-body-chain refactor, but is
#     needed for it to actually run: `init_huarm` is jax.jit'd with
#     `static_argnames=["id_dict"]`, and a plain dict (especially one holding
#     jnp.array values) is not hashable, so every call raised
#     `TypeError: unhashable type: 'dict'` before it ever got to the
#     hair-specific code below. A frozen dataclass with only int/tuple fields
#     is hashable, so it can safely be a static (compile-time-constant) jit
#     argument; anywhere the old code needed a jnp.array for fancy indexing
#     (`.at[idxs].set(...)`), we now build that array from the static tuple at
#     the point of use -- it gets baked in as a compile-time constant, same as
#     before, just without tripping the jit cache-key hash.
#     """
#     hair_eq_ids: Tuple[int, ...]
#     string_D: int
#     string_A: int
#     sound_box: int
#     sound_box_geom: int
#     bow_link_0: int
#     bow_tip: int
#     arm_qpos_idxs: Tuple[int, ...]
#     arm_dof_idxs: Tuple[int, ...]
#     hair_body_ids: Tuple[int, ...]
#     # Rigid bow-hair chain addressing (see build_id_dict for why these
#     # replace the old flexcomp particle addressing).
#     hair_root_qpos_adr: int
#     hair_root_dof_adr: int
#     hair_hinge_qpos_idxs: Tuple[int, ...]
#     hair_hinge_dof_idxs: Tuple[int, ...]


# def build_id_dict(mj_model: mujoco.MjModel, arm_joint_names=("joint1", "joint2", "joint3", "joint4"), n_hair_vertices=36) -> HairArmIds:
#     """Computes all string/body/joint lookups ONCE on CPU ahead of tracing."""
#     hair_eq_ids = []
#     for name in ("hair_to_frog", "hair_to_tip"):
#         eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
#         if eq_id >= 0:
#             hair_eq_ids.append(eq_id)

#     arm_qpos_idxs = []
#     arm_dof_idxs = []
#     for jn in arm_joint_names:
#         jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
#         arm_qpos_idxs.append(int(mj_model.jnt_qposadr[jid]))
#         arm_dof_idxs.append(int(mj_model.jnt_dofadr[jid]))

#     hair_body_ids = [
#         mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"bow_hair_{i}")
#         for i in range(n_hair_vertices)
#     ]

#     # The old flexcomp hair was 36 particles, each with 3 independent slide
#     # joints -- snap_hair_taut() could set every particle's world position
#     # directly. The rigid-body chain is a real kinematic chain instead:
#     #   - bow_hair_0 has one free joint (3 pos + 4 quat qpos, 6 dof) -- its
#     #     "attachment" to the rest of the bow is really the hair_to_frog
#     #     equality constraint, not a literal parent body.
#     #   - bow_hair_1 .. bow_hair_{n-1} each have 3 hinge joints
#     #     ("_bend_y", "_bend_z", "_twist_x") for that link's rotation
#     #     relative to the previous one.
#     # So instead of "36 independent qpos triples", positioning the whole
#     # chain now means setting the root's free-joint pose and zeroing the
#     # interior hinge angles. Precompute those addresses here (static, CPU
#     # side) so snap_hair_taut can stay a plain vectorized .at[].set() under
#     # jit, with no Python-level loop over hair vertices.
#     root_bid = hair_body_ids[0]
#     root_jadr = mj_model.body_jntadr[root_bid]
#     hair_root_qpos_adr = int(mj_model.jnt_qposadr[root_jadr])
#     hair_root_dof_adr = int(mj_model.jnt_dofadr[root_jadr])

#     hinge_qpos_idxs = []
#     hinge_dof_idxs = []
#     for i in range(1, n_hair_vertices):
#         for suffix in ("bend_y", "bend_z", "twist_x"):
#             jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"bow_hair_{i}_{suffix}")
#             hinge_qpos_idxs.append(int(mj_model.jnt_qposadr[jid]))
#             hinge_dof_idxs.append(int(mj_model.jnt_dofadr[jid]))

#     return HairArmIds(
#         hair_eq_ids=tuple(hair_eq_ids),
#         string_D=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "string_D"),
#         string_A=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "string_A"),
#         sound_box=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "sound_box"),
#         sound_box_geom=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom"),
#         bow_link_0=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0"),
#         bow_tip=mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip"),
#         arm_qpos_idxs=tuple(arm_qpos_idxs),
#         arm_dof_idxs=tuple(arm_dof_idxs),
#         hair_body_ids=tuple(hair_body_ids),
#         hair_root_qpos_adr=hair_root_qpos_adr,
#         hair_root_dof_adr=hair_root_dof_adr,
#         hair_hinge_qpos_idxs=tuple(hinge_qpos_idxs),
#         hair_hinge_dof_idxs=tuple(hinge_dof_idxs),
#     )


# # -----------------------------------------------------------------------------
# # 2. Refactored Pure-JAX Functions
# # -----------------------------------------------------------------------------
# def fix_hair_anchor_offsets(model: mjx.Model, hair_eq_ids: jax.Array) -> mjx.Model:
#     eq_data = model.eq_data
#     # Use .at[].set() for functional array updates
#     eq_data = eq_data.at[hair_eq_ids, 3:6].set(0.0)
#     return model.replace(eq_data=eq_data)


# def get_string_contact_point(data: mjx.Data, string_id: int) -> jax.Array:
#     xmat = data.xmat[string_id].reshape(3, 3)
#     local_tip = jnp.array([0.0, 0.0, -0.6])
#     return data.xpos[string_id] + xmat @ local_tip


# def between_strings_target(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> jax.Array:
#     pt_d = get_string_contact_point(data, id_dict.string_D)
#     pt_a = get_string_contact_point(data, id_dict.string_A)
#     midpoint = (pt_d + pt_a) / 2.0

#     sound_box_top_z = data.xpos[id_dict.sound_box][2] + model.geom_size[id_dict.sound_box_geom][0]

#     target_z = jnp.maximum(midpoint[2], sound_box_top_z) + 0.01
#     return midpoint.at[2].set(target_z)


# def _ik_point_fn(q_arm: jax.Array, model: mjx.Model, data: mjx.Data, body_ids: Tuple[int, int], weights: Tuple[float, float], qpos_idxs: jax.Array) -> jax.Array:
#     """Helper that evaluates FK for given arm positions to compute weighted end-effector location."""
#     updated_qpos = data.qpos.at[qpos_idxs].set(q_arm)
#     data_temp = data.replace(qpos=updated_qpos)
#     data_temp = mjx.kinematics(model, data_temp)
#     return weights[0] * data_temp.xpos[body_ids[0]] + weights[1] * data_temp.xpos[body_ids[1]]


# def jacobian_ik(model: mjx.Model, data: mjx.Data, body_ids: Tuple[int, int], weights: Tuple[float, float], target_pos: jax.Array, qpos_idxs: jax.Array, max_iters=4, damping=1e-2, step_clip=0.1) -> jax.Array:
#     """JAX-native IK solver using forward-mode Autodiff for the Jacobian and fori_loop for convergence."""
#     q_arm_init = data.qpos[qpos_idxs]

#     def ik_step(i, q_arm):
#         pt = _ik_point_fn(q_arm, model, data, body_ids, weights, qpos_idxs)
#         # Compute exact Jacobian (3 x N_dof) via JAX autodiff
#         J = jax.jacfwd(_ik_point_fn, argnums=0)(q_arm, model, data, body_ids, weights, qpos_idxs)
#         err = target_pos - pt

#         # Damped least squares
#         reg = (damping**2) * jnp.eye(3)
#         dtheta = J.T @ jnp.linalg.solve(J @ J.T + reg, err)

#         # Step clipping
#         step_norm = jnp.linalg.norm(dtheta)
#         scale = jnp.where(step_norm > step_clip, step_clip / (step_norm + 1e-8), 1.0)
#         return q_arm + dtheta * scale

#     q_arm_final = jax.lax.fori_loop(0, max_iters, ik_step, q_arm_init)

#     # Apply final IK solution to qpos and ctrl
#     qpos_new = data.qpos.at[qpos_idxs].set(q_arm_final)
#     ctrl_new = data.ctrl.at[qpos_idxs].set(q_arm_final)
#     data = data.replace(qpos=qpos_new, ctrl=ctrl_new)
#     return mjx.kinematics(model, data)


# def _quat_aligning(v_from: jax.Array, v_to: jax.Array) -> jax.Array:
#     """
#     Branchless (jit-safe) shortest-arc quaternion (w, x, y, z) rotating unit
#     vector v_from onto v_to.
#     """
#     v_from = v_from / jnp.linalg.norm(v_from)
#     v_to = v_to / jnp.linalg.norm(v_to)
#     dot = jnp.clip(jnp.dot(v_from, v_to), -1.0, 1.0)

#     axis_cross = jnp.cross(v_from, v_to)
#     axis_cross_norm = jnp.linalg.norm(axis_cross)

#     # Fallback axis for the (near-)antiparallel case: any unit vector
#     # orthogonal to v_from.
#     fallback_axis = jnp.cross(v_from, jnp.array([1.0, 0.0, 0.0]))
#     fallback_axis = jnp.where(
#         jnp.linalg.norm(fallback_axis) < 1e-6,
#         jnp.cross(v_from, jnp.array([0.0, 1.0, 0.0])),
#         fallback_axis,
#     )
#     fallback_axis = fallback_axis / (jnp.linalg.norm(fallback_axis) + 1e-12)

#     safe_axis = jnp.where(axis_cross_norm < 1e-9, fallback_axis, axis_cross)
#     safe_axis = safe_axis / (jnp.linalg.norm(safe_axis) + 1e-12)

#     angle = jnp.arccos(dot)
#     quat_general = jnp.concatenate([jnp.cos(angle / 2.0)[None], safe_axis * jnp.sin(angle / 2.0)])
#     quat_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
#     quat_antiparallel = jnp.concatenate([jnp.array([0.0]), fallback_axis])

#     is_parallel = dot > 1.0 - 1e-9
#     is_antiparallel = dot < -1.0 + 1e-9

#     quat = jnp.where(is_parallel, quat_identity, quat_general)
#     quat = jnp.where(is_antiparallel, quat_antiparallel, quat)
#     return quat


# def snap_hair_taut(model: mjx.Model, data: mjx.Data, rear_target: jax.Array, tip_target: jax.Array, id_dict: HairArmIds) -> mjx.Data:
#     """
#     Rigid-chain replacement for the old per-particle straight-line snap.
#     """
#     root_qpos_adr = id_dict.hair_root_qpos_adr
#     root_dof_adr = id_dict.hair_root_dof_adr
#     hinge_qpos_idxs = jnp.array(id_dict.hair_hinge_qpos_idxs, dtype=jnp.int32)
#     hinge_dof_idxs = jnp.array(id_dict.hair_hinge_dof_idxs, dtype=jnp.int32)

#     direction = tip_target - rear_target
#     length = jnp.linalg.norm(direction)
#     direction = direction / jnp.where(length > 1e-9, length, 1.0)
#     direction = jnp.where(length > 1e-9, direction, jnp.array([1.0, 0.0, 0.0]))

#     quat = _quat_aligning(jnp.array([1.0, 0.0, 0.0]), direction)

#     qpos = data.qpos
#     qpos = qpos.at[root_qpos_adr:root_qpos_adr + 3].set(rear_target)
#     qpos = qpos.at[root_qpos_adr + 3:root_qpos_adr + 7].set(quat)
#     qpos = qpos.at[hinge_qpos_idxs].set(0.0)

#     qvel = data.qvel
#     qvel = qvel.at[root_dof_adr:root_dof_adr + 6].set(0.0)
#     qvel = qvel.at[hinge_dof_idxs].set(0.0)

#     return data.replace(qpos=qpos, qvel=qvel)


# def pretension_bow_hair(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> mjx.Data:
#     """
#     Disable the hair-anchoring welds, lay the rigid hair chain out straight
#     and taut between bow_link_0's and bow_tip's current positions, then
#     re-enable the welds so the two end connect-constraints take up any
#     residual gap (see snap_hair_taut for why that's where the pretension
#     comes from in this rigid-body version).
#     """
#     eq_indices = jnp.array(id_dict.hair_eq_ids, dtype=jnp.int32)
#     rear_target = data.xpos[id_dict.bow_link_0]
#     tip_target = data.xpos[id_dict.bow_tip]

#     # 1. Disable constraints
#     data = data.replace(eq_active=data.eq_active.at[eq_indices].set(0))

#     # 2. Snap taut & run forward kinematics
#     data = snap_hair_taut(model, data, rear_target, tip_target, id_dict)
#     data = mjx.forward(model, data)

#     # 3. Re-enable constraints & re-run forward kinematics
#     data = data.replace(eq_active=data.eq_active.at[eq_indices].set(1))
#     return mjx.forward(model, data)


# # -----------------------------------------------------------------------------
# # 3. Main Init Entry Point
# # -----------------------------------------------------------------------------
# @functools.partial(jax.jit, static_argnames=["id_dict"])
# def init_huarm(model: mjx.Model, data: mjx.Data, id_dict: HairArmIds) -> Tuple[mjx.Model, mjx.Data]:
#     # Update model (anchor offsets)
#     model = fix_hair_anchor_offsets(model, jnp.array(id_dict.hair_eq_ids, dtype=jnp.int32))

#     # Target calculation & IK
#     target = between_strings_target(model, data, id_dict)
#     body_ids = (id_dict.bow_link_0, id_dict.bow_tip)
#     data = jacobian_ik(model, data, body_ids, (0.5, 0.5), target, jnp.array(id_dict.arm_qpos_idxs, dtype=jnp.int32))

#     # Pretension hair
#     data = pretension_bow_hair(model, data, id_dict)

#     return model, data