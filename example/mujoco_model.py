import sys
import numpy as np
import mujoco
import mujoco.viewer
import time
import socket


def get_desired_position(t):
    """
    Computes a target trajectory for the bow over time.
    Simulates standard back-and-forth bowing motion along the Y-axis,
    with a slight downward press along the Z-axis.
    """
    base_x = 0.38
    base_y = 0.30
    base_z = 0.58

    y_offset = 0.15 * np.sin(2 * np.pi * 0.5 * t)
    z_offset = -0.015 * np.abs(np.sin(2 * np.pi * 0.5 * t))

    return np.array([base_x, base_y + y_offset, base_z + z_offset])


def get_teleop_position():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5005))
    while True:
        data, addr = sock.recvfrom(1024)
        position = np.frombuffer(data, dtype=np.float32)
        yield position


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


def get_hair_end_tensions(model, data):
    """
    Returns the constraint reaction force vector for the single rigid bow_hair body 
    at the hair_to_frog and hair_to_tip equality constraints.
    """
    tensions = {}
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0:
            continue
        mask = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq_id)
        if np.any(mask):
            tensions[name] = data.efc_force[mask].copy()
    return tensions


def print_hair_tension(model, data):
    tensions = get_hair_end_tensions(model, data)
    if not tensions:
        print(f"t={data.time:6.3f}  bow hair tension: (no active end constraints)")
        return
    parts = [f"{name} |F|={np.linalg.norm(force):.4f} N" for name, force in tensions.items()]
    print(f"t={data.time:6.3f}  bow hair end tensions  " + "  ".join(parts))


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


def compute_arm_ctrl_for_target(model, ik_data, target_pos, joint_names,
                                 body_name="bow_frog", max_iters=20):
    """
    IK solver for positioning the bow_frog body with arm joint actuation.
    """
    body_weights = [(body_name, 1.0)]
    jacobian_ik(model, ik_data, body_weights, target_pos, list(joint_names), max_iters=max_iters)
    qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in joint_names]
    return {jn: ik_data.qpos[qi] for jn, qi in zip(joint_names, qpos_idxs)}


def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    fix_hair_anchor_offsets(model)
    insert_hair_between_strings(model, data)
    pretension_bow_hair(model, data)

    arm_joint_names = ("joint1", "joint2", "joint3", "joint4")
    arm_actuator_ids = {jn: joint_to_actuator_id(model, jn) for jn in arm_joint_names}

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos
    mujoco.mj_forward(model, ik_data)

    tension_print_interval = 0.5
    next_tension_print = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in viewer to exit.")
        start = time.time()
        while viewer.is_running():
            elapsed_real = time.time() - start
            print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}", end="\r")
            
            if data.time >= elapsed_real:
                time.sleep(0.01)
                continue

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_tension_print:
                next_tension_print = data.time + tension_print_interval
                sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "bow_arm_contact")
                if sensor_id >= 0:
                    force = data.sensor("bow_arm_contact").data.copy()
                    print(f"t={data.time:6.3f} bow-arm contact force: {force} [N]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mujoco_model.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])