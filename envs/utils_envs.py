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


# ==================================================
# JAX / MJX reimplementation of init_huarm.
#
# The numpy version above mutates a single CPU mujoco.MjData once, at env
# construction time. This version is expressed purely with mjx primitives
# (mjx.forward, mjx_support.jac) so it operates on jax.Array-backed mjx.Data
# and can be called every episode from ErhuEnv.reset -- including under
# jit/vmap across a batch of parallel envs. Body/geom/joint ids are still
# resolved via mujoco.mj_name2id on the plain (CPU) mj_model, which is a
# static, untraced lookup and therefore safe to do at every call.
# ==================================================

def _weighted_point_and_jacobian_jax(mjx_model, mjx_data, body_points, dof_idxs):
    """
    JAX/MJX counterpart of weighted_point_and_jacobian: weighted-average
    world position and position Jacobian (3, len(dof_idxs)) for a list of
    (body_id, local_offset, weight) triples.
    """
    dof_idxs = jnp.asarray(dof_idxs)
    point = jnp.zeros(3)
    J = jnp.zeros((3, dof_idxs.shape[0]))
    for bid, local_offset, w in body_points:
        xmat = mjx_data.xmat[bid].reshape(3, 3)
        world_pt = mjx_data.xpos[bid] + xmat @ local_offset
        point = point + w * world_pt
        jacp, _ = mjx_support.jac(mjx_model, mjx_data, world_pt, bid)
        J = J + w * jacp[dof_idxs, :].T
    return point, J


def _jacobian_ik_jax(mjx_model, data, body_points, target_pos, dof_idxs, qpos_idxs,
                      max_iters=10, damping=1e-2, step_clip=0.1):
    """
    JAX/MJX counterpart of jacobian_ik. Runs a fixed number of damped
    least-squares steps via jax.lax.fori_loop (no data-dependent early
    exit on `tol`, since the iteration count must be static under jit) and
    returns the resulting mjx.Data with forward kinematics applied.
    """
    qpos_idxs = jnp.asarray(qpos_idxs)

    def body_fn(_, data):
        data = mjx.forward(mjx_model, data)
        point, J = _weighted_point_and_jacobian_jax(mjx_model, data, body_points, dof_idxs)
        err = target_pos - point
        dtheta = J.T @ jnp.linalg.solve(J @ J.T + damping**2 * jnp.eye(3), err)
        step_norm = jnp.linalg.norm(dtheta)
        dtheta = dtheta * jnp.minimum(1.0, step_clip / (step_norm + 1e-12))
        qpos = data.qpos.at[qpos_idxs].add(dtheta)
        return data.replace(qpos=qpos)

    data = jax.lax.fori_loop(0, max_iters, body_fn, data)
    return mjx.forward(mjx_model, data)


def _set_joint_ctrl_jax(mj_model, mjx_data, joint_names):
    """
    JAX/MJX counterpart of set_joint_ctrl: sets actuator controls to match
    the current qpos for selected joints.
    """
    ctrl = mjx_data.ctrl
    for jn in joint_names:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = joint_to_actuator_id(mj_model, jn)
        if aid >= 0:
            qadr = mj_model.jnt_qposadr[jid]
            ctrl = ctrl.at[aid].set(mjx_data.qpos[qadr])
    return mjx_data.replace(ctrl=ctrl)


def init_huarm_jax(mj_model, mjx_model, mjx_data,
                    arm_joint_names=("joint1", "joint2", "joint3", "joint4"),
                    max_iters=10, damping=1e-2, step_clip=0.1,
                    joint_noise_std=0.0, rng=None):
    """
    JAX/MJX counterpart of init_huarm / insert_hair_between_strings: solves
    arm IK to place the bow hair midpoint between the erhu strings, then
    bakes the result into ctrl so it holds under the position actuators.

    Meant to be called from ErhuEnv.reset (per-episode), not once at env
    construction time.
    """
    data = mjx.forward(mjx_model, mjx_data)

    def bid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def gid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)

    string_d_id, string_a_id = bid("string_D"), bid("string_A")
    sound_box_id, sound_box_geom_id = bid("sound_box"), gid("sound_box_geom")

    local_tip = jnp.array([0.0, 0.0, -0.6])
    d_tip = data.xpos[string_d_id] + data.xmat[string_d_id].reshape(3, 3) @ local_tip
    a_tip = data.xpos[string_a_id] + data.xmat[string_a_id].reshape(3, 3) @ local_tip
    target = 0.5 * (d_tip + a_tip)
    top_z = data.xpos[sound_box_id][2] + mj_model.geom_size[sound_box_geom_id][0]
    target = target.at[2].set(jnp.maximum(target[2], top_z) + 0.01)

    dof_idxs = [mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in arm_joint_names]
    qpos_idxs = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in arm_joint_names]

    bow_hair_id = bid("bow_hair")
    hair_midpoint_local = jnp.array([0.0, 0.0, -0.25])
    body_points = [(bow_hair_id, hair_midpoint_local, 1.0)]

    data = _jacobian_ik_jax(mjx_model, data, body_points, target, dof_idxs, qpos_idxs,
                             max_iters=max_iters, damping=damping, step_clip=step_clip)

    if rng is not None and joint_noise_std > 0:
        noise = joint_noise_std * jax.random.normal(rng, (len(arm_joint_names),))
        qpos = data.qpos.at[jnp.asarray(qpos_idxs)].add(noise)
        data = mjx.forward(mjx_model, data.replace(qpos=qpos))

    data = _set_joint_ctrl_jax(mj_model, data, arm_joint_names)
    return data


def domain_randomize_jax(mj_model, mjx_model, mjx_data, rng,
                          arm_joint_names=("joint1", "joint2", "joint3", "joint4"),
                          friction_range=(0.7, 1.3),
                          mass_range=(0.8, 1.2),
                          damping_range=(0.8, 1.2),
                          erhu_pos_std=0.05,
                          erhu_tilt_std=0.03,
                          joint_noise_std=0.005):
    """
    Samples per-episode randomized model params -- physical params (contact
    friction, body mass, joint damping) plus the erhu's placement in the
    world (erhu_root's body_pos/body_quat, jittered by a small translation
    and tilt) -- and re-solves init_huarm_jax against them, so the bow's
    resting pose is consistent with wherever the erhu ended up and however
    the arm's dynamics were perturbed this episode. The IK solve itself
    also gets `joint_noise_std` radians of Gaussian noise added to each arm
    joint afterwards, so the arm doesn't start from the exact same IK
    solution every episode.

    Meant to be called once per episode from ErhuEnv.reset. Returns
    (dr_params, mjx_data): `dr_params` is a plain dict of just the ~5
    randomized array fields (not a full mjx.Model). Stash *that* dict in
    State.info and merge it onto a closure-captured base mjx_model (e.g.
    `self.mjx_model.replace(**dr_params)`) right before each mjx call,
    rather than carrying a whole mjx.Model through info: mjx.Model's ~160
    static/topology fields get re-hashed (SHA-256, to make them hashable
    pytree "meta" data) on every `jax.jit` dispatch that traces over it, so
    threading a full randomized Model through a jitted step() makes every
    step call pay that cost -- ~30x slower in practice, measured on
    ErhuEnv.step. Carrying only the handful of actually-randomized arrays
    keeps the (expensive-to-hash) static fields as a closure constant,
    flattened once at trace/compile time like before domain randomization
    existed.
    """
    friction_rng, mass_rng, damping_rng, pos_rng, tilt_rng, joint_rng = jax.random.split(rng, 6)

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

    erhu_root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "erhu_root")
    body_pos = mjx_model.body_pos.at[erhu_root_id].add(
        erhu_pos_std * jax.random.normal(pos_rng, (3,))
    )
    # Small-angle random tilt composed onto the nominal orientation, as a
    # unit quaternion (w, x, y, z); avoids re-normalizing a hand-built axis
    # so the result stays a valid rotation even for the sampled extremes.
    tilt_axis_angle = erhu_tilt_std * jax.random.normal(tilt_rng, (3,))
    tilt_angle = jnp.linalg.norm(tilt_axis_angle) + 1e-8
    tilt_axis = tilt_axis_angle / tilt_angle
    tilt_quat = jnp.concatenate([
        jnp.cos(tilt_angle / 2)[None], tilt_axis * jnp.sin(tilt_angle / 2)
    ])
    nominal_quat = mjx_model.body_quat[erhu_root_id]
    w0, x0, y0, z0 = nominal_quat
    w1, x1, y1, z1 = tilt_quat
    body_quat = mjx_model.body_quat.at[erhu_root_id].set(jnp.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ]))

    dr_params = dict(
        geom_friction=friction,
        body_mass=body_mass,
        dof_damping=dof_damping,
        body_pos=body_pos,
        body_quat=body_quat,
    )
    randomized_model = mjx_model.replace(**dr_params)

    mjx_data = init_huarm_jax(mj_model, randomized_model, mjx_data, arm_joint_names,
                               joint_noise_std=joint_noise_std, rng=joint_rng)

    return dr_params, mjx_data