import sys
import numpy as np
import mujoco
import mujoco.viewer
import time
import socket

def get_desired_position(t):
    """
    Computes a target trajectory for the bow over time.
    This simulates a standard bowing motion back and forth along the Y-axis,
    with a slight downward press on the Z-axis.
    """
    base_x = 0.38
    base_y = 0.30
    base_z = 0.58

    y_offset = 0.15 * np.sin(2 * np.pi * 0.5 * t)
    z_offset = -0.015 * np.abs(np.sin(2 * np.pi * 0.5 * t))

    return np.array([base_x, base_y + y_offset, base_z + z_offset])

def get_teleop_position():
    socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.bind(("", 5005))
    while True:
        data, addr = socket.recvfrom(1024)
        position = np.frombuffer(data, dtype=np.float32)
        yield position

def fix_hair_anchor_offsets(model):
    """
    hair_to_frog/hair_to_tip are <connect> constraints between the flexcomp
    hair endpoints and points on the bow stick. Because the flexcomp is
    authored at its own XML position, independent of the bow's geometry, the
    two bodies are NOT coincident at compile time (bow_hair_0 sits ~4.7cm from
    bow_link_0, bow_hair_35 sits a similar distance from bow_tip). MuJoCo's
    compiler auto-derives each connect's "anchor in body2's frame" from that
    compile-time offset -- meaning the constraint's real target is "stay
    offset from body2 by the original gap", not "coincide with body2". That's
    what was silently undoing any runtime teleport/pretension. Zeroing the
    body2-frame anchor (eq_data[3:6]) makes the constraint actually pin the
    two points together.
    """
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            model.eq_data[eq_id][3:6] = 0.0

def get_flex_edge_tension(model, data, flex_edge_id=None):
    if flex_edge_id is None:
        for i in range(model.neq):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) is None:
                flex_edge_id = i
                break

    mask = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == flex_edge_id)
    if not np.any(mask):
        return None
    return data.efc_force[mask]

def print_flex_tension(model, data, flex_edge_id=None):
    tension = get_flex_edge_tension(model, data, flex_edge_id)
    if tension is None or tension.size == 0:
        print(f"t={data.time:6.3f}  bow hair tension: (no active edge constraints)")
        return
    print(f"t={data.time:6.3f}  bow hair tension [N]  "
          f"min={tension.min():+.4f}  max={tension.max():+.4f}  "
          f"mean={tension.mean():+.4f}  |mean|={np.abs(tension).mean():.4f}")

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
    -- unlike the bow_hair flex particles, bow_link_0/bow_tip ARE rigid
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
        dof_idx = model.jnt_dofadr[jid]
        data.ctrl[dof_idx] = data.qpos[model.jnt_qposadr[jid]]

def insert_hair_between_strings(model, data, arm_joint_names=("joint1", "joint2", "joint3", "joint4")):
    """
    Move the bow -- via the arm's joints, using Jacobian-based IK -- so the
    taut hair passes between the two erhu strings, resting just above the
    sound box.
 
    This supersedes the earlier single-particle-teleport approach. That
    approach tried to move one bow_hair_i vertex directly; but bow_hair
    particles aren't part of the arm's kinematic tree (they're independent
    flex DOFs coupled only via equality constraints), so there's no
    meaningful Jacobian relating them to the arm joints, and moving one
    particle across the ~0.5m gap between the bow and the strings violated
    the hair's edge-length constraint by ~30x its rest length -- verified to
    produce a massive corrective transient that undid the move within
    ~0.2s of sim time.
 
    bow_link_0 and bow_tip, by contrast, ARE rigid bodies in the arm's
    kinematic chain, so a real position Jacobian w.r.t. the arm joints
    exists. We solve IK to place the *midpoint* of bow_link_0/bow_tip at the
    target (approximating "some point along the taut hair", since the hair
    runs approximately straight between them), then re-run
    pretension_bow_hair() so the flex hair is laid out fresh between the
    endpoints' new positions -- which by then are already close to the
    target, so no large/unstable stretch is needed.
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


def particle_qpos_addrs(model, body_name):
    """
    bow_hair_N bodies are flexcomp cable "particles": 3 independent slide
    joints, not a single free/ball joint. Returns their qpos addresses, dof
    addresses, joint axes, and the body's nominal (compiled) anchor position.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]  # expect 3 (slide x, y, z)
    qadrs = [model.jnt_qposadr[jadr + k] for k in range(jnum)]
    dadrs = [model.jnt_dofadr[jadr + k] for k in range(jnum)]
    axes = [model.jnt_axis[jadr + k] for k in range(jnum)]
    anchor = model.body_pos[bid]
    return qadrs, dadrs, axes, anchor


def snap_hair_taut(model, data, rear_target, tip_target, n_vertices=36):
    """
    Lay all n_vertices hair particles out on a straight line between
    rear_target (frog side) and tip_target (tip side), instead of only
    moving the two endpoints. The cable's edges are enforced by a hard
    (near-inextensible) equality constraint, so moving only the boundary
    vertices leaves the interior in an inconsistent, over-stretched
    configuration that gets rejected almost immediately.
    """
    for i in range(n_vertices):
        t = i / (n_vertices - 1)
        target = rear_target * (1 - t) + tip_target * t
        qadrs, dadrs, axes, anchor = particle_qpos_addrs(model, f"bow_hair_{i}")
        delta = target - anchor
        for qadr, dadr, axis in zip(qadrs, dadrs, axes):
            data.qpos[qadr] = np.dot(delta, axis)
            data.qvel[dadr] = 0.0


def pretension_bow_hair(model, data):
    """
    Disable the hair-anchoring welds AND the flexcomp's auto-generated edge
    (inextensibility) equality constraint, lay the hair out taut along a
    straight line between bow_link_1 and bow_tip, then re-enable everything.
    """
    frog_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_frog")
    tip_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_tip")

    # The <edge equality="true"/> on the bow_hair flexcomp auto-generates one
    # more (unnamed) equality constraint enforcing every edge length. It's
    # always active and was never being disabled, which is what silently
    # undid any manual teleport of just the two endpoint bodies.
    flex_edge_id = None
    for i in range(model.neq):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) is None:
            flex_edge_id = i
            break
    if flex_edge_id is None:
        print("Warning: could not find the flexcomp edge equality constraint; "
              "hair pretension may not hold.")

    bow_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip")
    bow_link_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_0")
    rear_target = data.xpos[bow_link_0_id].copy()
    tip_target = data.xpos[bow_tip_id].copy()

    data.eq_active[frog_weld_id] = 0
    data.eq_active[tip_weld_id] = 0
    if flex_edge_id is not None:
        data.eq_active[flex_edge_id] = 0

    snap_hair_taut(model, data, rear_target, tip_target)
    mujoco.mj_forward(model, data)

    data.eq_active[frog_weld_id] = 1
    data.eq_active[tip_weld_id] = 1
    if flex_edge_id is not None:
        data.eq_active[flex_edge_id] = 1
    mujoco.mj_forward(model, data)



def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    fix_hair_anchor_offsets(model)
    data = mujoco.MjData(model)
    # data.xpos/xquat/etc. are all zero until forward kinematics has run once;
    # pretension_bow_hair() needs real bow_tip/bow_link_0 positions, so populate them now.
    mujoco.mj_forward(model, data)
    insert_hair_between_strings(model, data)
    pretension_bow_hair(model, data)

    try:
        mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_target")
        mocap_idx = model.body_mocapid[mocap_body_id]
    except ValueError:
        print("Error: 'bow_target' body with mocap='true' not found in XML!")
        sys.exit(1)

    tension_print_interval = 0.5
    next_tension_print = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in the viewer to exit.")
        while viewer.is_running():
            step_start = time.time()
            current_sim_time = data.time

            if mocap_idx >= 0:
                desired_pos = get_desired_position(current_sim_time)
                data.mocap_pos[mocap_idx] = desired_pos

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_tension_print:
                # print_flex_tension(model, data)
                next_tension_print = data.time + tension_print_interval
                force = data.sensor("bow_arm_contact").data.copy()
                print(f"t={data.time:6.3f} bow-arm contact force: {force} [N]")

            elapsed_real = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed_real
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python teleop_bow.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])