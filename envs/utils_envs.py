import functools
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
import mujoco
import numpy as np


def get_string_contact_point(model, data, string_body_name):
    """
    Returns the world position of the free (bottom) end of a string capsule.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, string_body_name)
    xmat = data.xmat[bid].reshape(3, 3)
    local_tip = np.array([0.0, 0.0, -0.6])
    return data.xpos[bid] + xmat @ local_tip


def between_strings_target(model, data):
    """
    Midpoint between the two string contact points, lifted clear of the sound box.
    """
    midpoint = (get_string_contact_point(model, data, "string_D")
                + get_string_contact_point(model, data, "string_A")) / 2.0

    sound_box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sound_box")
    sound_box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sound_box_geom")
    sound_box_top_z = data.xpos[sound_box_id][2] + model.geom_size[sound_box_geom_id][0]
    midpoint[2] = max(midpoint[2], sound_box_top_z) + 0.01  # 1 cm clearance
    return midpoint

def weighted_point_and_jacobian(model, data, body_points, dof_idxs):
    """
    Computes weighted-average world position and position Jacobian for given
    (body_name, local_offset, weight) triples. local_offset is a 3-vector in
    the body's own local frame, letting the target point be somewhere other
    than the body origin (e.g. the midpoint of a capsule that is welded to,
    but geometrically offset/rotated from, the body it's attached to).
    """
    point = np.zeros(3)
    J = np.zeros((3, len(dof_idxs)))
    for name, local_offset, w in body_points:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        xmat = data.xmat[bid].reshape(3, 3)
        world_pt = data.xpos[bid] + xmat @ local_offset
        point += w * world_pt
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, world_pt, bid)
        J += w * jacp[:, dof_idxs]
    return point, J


def jacobian_ik(model, data, body_points, target_pos, joint_names,
                max_iters=200, damping=1e-2, step_clip=0.1, tol=1e-4):
    """
    Damped-least-squares IK to position specified target point combination.

    body_points is a list of (body_name, local_offset, weight) triples, see
    weighted_point_and_jacobian.
    """
    dof_idxs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in joint_names]
    qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in joint_names]

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        point, J = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
        err = target_pos - point
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return it, err_norm
        
        dtheta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        step_norm = np.linalg.norm(dtheta)
        if step_norm > step_clip:
            dtheta *= step_clip / step_norm
        for qidx, d in zip(qpos_idxs, dtheta):
            data.qpos[qidx] += d

    mujoco.mj_forward(model, data)
    point, _ = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
    return max_iters, np.linalg.norm(target_pos - point)


def set_joint_ctrl(model, data, joint_names):
    """
    Sets actuator controls to match the current qpos for selected joints.
    """
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = joint_to_actuator_id(model, jn)
        if aid >= 0:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]


def joint_to_actuator_id(model, joint_name):
    """
    Looks up the actuator driving joint_name.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return -1
    for aid in range(model.nu):
        if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[aid, 0] == jid):
            return aid
    return -1


def insert_hair_between_strings(model, data, arm_joint_names=("joint1", "joint2", "joint3", "joint4"),
                                 joint_noise_std=0.0):
    """
    Solves arm IK to place the bow stick midpoint between the erhu strings.
    Since bow_hair is now a rigid body welded to bow_tip (no free joint, no
    equality constraints), positioning bow_link_0 / bow_tip via arm IK is
    sufficient to place the hair as well -- it simply follows the bow's
    kinematic chain, no separate hair layout/pretension step is needed.

    `joint_noise_std` (radians), if > 0, perturbs each arm joint's qpos by
    independent Gaussian noise after the IK solve converges, then re-runs
    forward kinematics so the noisy pose is reflected everywhere (and bakes
    it into ctrl, so the arm starts a little off-target instead of snapping
    back).
    """
    target = between_strings_target(model, data)
    print(f"Target insertion point (between strings, above sound box): {target}")

    hair_midpoint_local = np.array([0.0, 0.0, -0.25])
    body_points = [("bow_hair", hair_midpoint_local, 1.0)]
    iters, err = jacobian_ik(model, data, body_points, target, list(arm_joint_names))
    print(f"Arm IK converged in {iters} iterations, final position error {err:.6f} m")

    if joint_noise_std > 0:
        qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                     for jn in arm_joint_names]
        for qidx in qpos_idxs:
            data.qpos[qidx] += np.random.normal(0.0, joint_noise_std)
        mujoco.mj_forward(model, data)

    set_joint_ctrl(model, data, arm_joint_names)

    bow_hair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_hair")
    xmat = data.xmat[bow_hair_id].reshape(3, 3)
    midpoint = data.xpos[bow_hair_id] + xmat @ hair_midpoint_local
    print(f"Bow hair midpoint after IK: {midpoint} (target residual: {np.linalg.norm(midpoint - target):.4f} m)")



def init_huarm(model, data, id_dict=None, joint_noise_std=0.0):
    """
    Prepares the model/data for a new episode.
    """
    mujoco.mj_forward(model, data)
    insert_hair_between_strings(model, data, joint_noise_std=joint_noise_std)
    return model, data


# ==================================================
def arm_joint_indices(mj_model, arm_joint_names=("joint1", "joint2", "joint3", "joint4")):
    """
    Precomputes, once per model (e.g. at env init), the static index arrays
    needed by `domain_randomize_jax` / `_set_joint_ctrl_jax` each reset --
    qpos addresses for all arm joints, plus the (actuator id, qpos address)
    pairs for the subset of those joints that are actuated. Passing these in
    as args avoids repeating `mj_name2id`/`joint_to_actuator_id` python
    lookups on every call.
    """
    jids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_joint_names]
    qpos_idxs = jnp.asarray([mj_model.jnt_qposadr[jid] for jid in jids])

    aids = [joint_to_actuator_id(mj_model, jn) for jn in arm_joint_names]
    ctrl_aids = jnp.asarray([a for a in aids if a >= 0], dtype=jnp.int32)
    ctrl_qpos_idxs = jnp.asarray(
        [mj_model.jnt_qposadr[jid] for jid, a in zip(jids, aids) if a >= 0]
    )
    return qpos_idxs, ctrl_aids, ctrl_qpos_idxs


def _set_joint_ctrl_jax(mjx_data, ctrl_aids, ctrl_qpos_idxs):
    """
    JAX/MJX counterpart of set_joint_ctrl: sets actuator controls to match
    the current qpos for selected joints, given their precomputed
    (actuator id, qpos address) index arrays (see `arm_joint_indices`).
    """
    ctrl = mjx_data.ctrl.at[ctrl_aids].set(mjx_data.qpos[ctrl_qpos_idxs])
    return mjx_data.replace(ctrl=ctrl)

def domain_randomize_jax(mjx_model, mjx_data, rng, erhu_pose_pool,
                          arm_qpos_idxs, arm_ctrl_aids, arm_ctrl_qpos_idxs,
                          friction_range=(0.7, 1.3),
                          mass_range=(0.8, 1.2),
                          damping_range=(0.8, 1.2),
                          joint_noise_std=0.005):
    """Samples this episode's randomized dynamics params -- contact friction,
    body mass, joint damping.

    `arm_qpos_idxs`, `arm_ctrl_aids`, `arm_ctrl_qpos_idxs` are the static
    index arrays from `arm_joint_indices(mj_model, arm_joint_names)`,
    precomputed once at env init and passed in here to avoid recomputing
    them (and the underlying mj_name2id lookups) on every reset."""
    friction_rng, mass_rng, damping_rng, pool_rng, joint_rng = jax.random.split(rng, 5)

    friction = mjx_model.geom_friction * jax.random.uniform(
        friction_rng, mjx_model.geom_friction.shape,
        minval=friction_range[0], maxval=friction_range[1],
    )
    body_mass = mjx_model.body_mass * jax.random.uniform(
        mass_rng, mjx_model.body_mass.shape,
        minval=mass_range[0], maxval=mass_range[1],
    )
    dof_damping = mjx_model.dof_damping * jax.random.uniform(
        damping_rng, mjx_model.dof_damping.shape,
        minval=damping_range[0], maxval=damping_range[1],
    )

    pose = sample_erhu_pose_pool(erhu_pose_pool, pool_rng)
    body_pos, body_quat, arm_qpos = pose["body_pos"], pose["body_quat"], pose["arm_qpos"]

    dr_params = dict(
        geom_friction=friction,
        body_mass=body_mass,
        dof_damping=dof_damping,
        body_pos=body_pos,
        body_quat=body_quat,
    )
    randomized_model = mjx_model.replace(**dr_params)

    if joint_noise_std > 0:
        arm_qpos = arm_qpos + joint_noise_std * jax.random.normal(joint_rng, arm_qpos.shape)

    mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[arm_qpos_idxs].set(arm_qpos))
    mjx_data = mjx.forward(randomized_model, mjx_data)
    mjx_data = _set_joint_ctrl_jax(mjx_data, arm_ctrl_aids, arm_ctrl_qpos_idxs)

    return dr_params, mjx_data

# ==================================================

def build_erhu_pose_pool(mj_model, mjx_model, rng, pool_size,
                          arm_joint_names=("joint1", "joint2", "joint3", "joint4"),
                          erhu_pos_std=0.05, erhu_tilt_std=0.03):
    """
    Precomputes `pool_size` random erhu placements (erhu_root's body_pos/
    body_quat, jittered by a small translation and tilt) and the arm's
    Jacobian-IK solution for each
    
    Returns a dict pytree {"body_pos", "body_quat", "arm_qpos"}, each
    array stacked with a leading `pool_size` axis, for
    `sample_erhu_pose_pool` to draw from inside reset().
    """
    erhu_root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "erhu_root")
    qpos_idxs = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in arm_joint_names]

    # mj_model.body_pos/body_quat are mutated in place per sample (mujoco
    # has no cheap immutable "with field set" for MjModel) and restored
    # afterwards, so the shared mj_model object comes out exactly as it
    # went in regardless of how this loop exits.
    base_body_pos = np.array(mjx_model.body_pos)   # (nbody, 3), full-model template
    base_body_quat = np.array(mjx_model.body_quat)  # (nbody, 4)
    nominal_pos = base_body_pos[erhu_root_id].copy()
    nominal_quat = base_body_quat[erhu_root_id].copy()
    hair_midpoint_local = np.array([0.0, 0.0, -0.25])
    body_points = [("bow_hair", hair_midpoint_local, 1.0)]

    body_pos_pool, body_quat_pool, arm_qpos_pool = [], [], []
    try:
        for i in range(pool_size):
            print(f"Building erhu pose pool: sample {i+1}/{pool_size}", end="\r")
            rng, pos_rng, tilt_rng = jax.random.split(rng, 3)

            pos_noise = erhu_pos_std * np.asarray([*jax.random.normal(pos_rng, (1,)), 0.0, 0.0])
            body_pos = nominal_pos + pos_noise
            # Small-angle random tilt composed onto the nominal orientation,
            # as a unit quaternion (w, x, y, z); avoids re-normalizing a
            # hand-built axis so the result stays a valid rotation even at
            # sampled extremes.
            tilt_axis_angle = erhu_tilt_std * np.asarray(jax.random.normal(tilt_rng, (3,)))
            tilt_angle = float(np.linalg.norm(tilt_axis_angle)) + 1e-8
            tilt_axis = tilt_axis_angle / tilt_angle
            tilt_quat = np.concatenate([[np.cos(tilt_angle / 2)], tilt_axis * np.sin(tilt_angle / 2)])
            w0, x0, y0, z0 = nominal_quat
            w1, x1, y1, z1 = tilt_quat
            body_quat = np.array([
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ])

            mj_model.body_pos[erhu_root_id] = body_pos
            mj_model.body_quat[erhu_root_id] = body_quat

            data = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, data)
            target = between_strings_target(mj_model, data)
            jacobian_ik(mj_model, data, body_points, target, list(arm_joint_names))

            body_pos_full = base_body_pos.copy()
            body_pos_full[erhu_root_id] = body_pos
            body_quat_full = base_body_quat.copy()
            body_quat_full[erhu_root_id] = body_quat

            body_pos_pool.append(body_pos_full)
            body_quat_pool.append(body_quat_full)
            arm_qpos_pool.append(np.array([data.qpos[qi] for qi in qpos_idxs]))
    finally:
        mj_model.body_pos[erhu_root_id] = nominal_pos
        mj_model.body_quat[erhu_root_id] = nominal_quat

    return dict(
        body_pos=jnp.asarray(np.stack(body_pos_pool)),
        body_quat=jnp.stack(body_quat_pool),
        arm_qpos=jnp.stack(arm_qpos_pool),
    )


def sample_erhu_pose_pool(pool, rng):
    """Draws one (body_pos, body_quat, arm_qpos) triple from a pool built by
    `build_erhu_pose_pool`: a random index plus a gather on each leaf --
    cheap enough to call every reset(), unlike the IK solve it replaces."""
    pool_size = pool["body_pos"].shape[0]
    idx = jax.random.randint(rng, (), 0, pool_size)
    return jax.tree_util.tree_map(lambda x: x[idx], pool)