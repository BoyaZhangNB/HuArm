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
    Zeroes the body2-frame anchor offsets (eq_data[3:6]) for the <connect> constraints
    linking 'bow_hair' to 'bow_link_0' and 'bow_tip'. This ensures the constraints 
    pin the endpoints together without static offsets derived at compile time.
    """
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            model.eq_data[eq_id][3:6] = 0.0


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


def weighted_point_and_jacobian(model, data, body_weights, dof_idxs):
    """
    Computes weighted-average world position and position Jacobian for given bodies.
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
    Damped-least-squares IK to position specified target body combination.
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


def insert_hair_between_strings(model, data, arm_joint_names=("joint1", "joint2", "joint3", "joint4")):
    """
    Solves arm IK to place the bow stick midpoint between the erhu strings,
    then updates the single bow_hair body layout accordingly.
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
    print(f"Bow midpoint after IK: {midpoint} (target residual: {np.linalg.norm(midpoint - target):.4f} m)")


def _quat_aligning(v_from, v_to):
    """Computes the shortest-arc quaternion rotating unit vector v_from to v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        axis = np.cross(v_from, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(v_from, v_to)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])


def snap_hair_taut(model, data, rear_target, tip_target):
    """
    Positions and aligns the single rigid capsule body 'bow_hair' between
    rear_target (frog end) and tip_target (tip end) using its free joint.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_hair")
    jadr = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jadr]
    dadr = model.jnt_dofadr[jadr]

    direction = tip_target - rear_target
    length = np.linalg.norm(direction)
    direction = direction / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])

    # Position free joint root at rear target
    data.qpos[qadr:qadr + 3] = rear_target
    # Align local X-axis capsule with target direction
    data.qpos[qadr + 3:qadr + 7] = _quat_aligning(np.array([1.0, 0.0, 0.0]), direction)
    # Zero linear and angular velocity
    data.qvel[dadr:dadr + 6] = 0.0


def pretension_bow_hair(model, data):
    """
    Lays out the single bow_hair capsule straight between bow_link_0 and bow_tip,
    re-engaging equality constraints to hold tension via compliant solref/solimp settings.
    """
    frog_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_frog")
    tip_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_tip")

    bow_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip")
    bow_link_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0")
    rear_target = data.xpos[bow_link_0_id].copy()
    tip_target = data.xpos[bow_tip_id].copy()

    if frog_weld_id >= 0:
        data.eq_active[frog_weld_id] = 0
    if tip_weld_id >= 0:
        data.eq_active[tip_weld_id] = 0

    snap_hair_taut(model, data, rear_target, tip_target)
    mujoco.mj_forward(model, data)

    if frog_weld_id >= 0:
        data.eq_active[frog_weld_id] = 1
    if tip_weld_id >= 0:
        data.eq_active[tip_weld_id] = 1
    mujoco.mj_forward(model, data)


def init_huarm(model, data, id_dict=None):
    """
    Prepares the model/data for a new episode.
    """
    mujoco.mj_forward(model, data)
    fix_hair_anchor_offsets(model)
    insert_hair_between_strings(model, data)
    pretension_bow_hair(model, data)
    return model, data