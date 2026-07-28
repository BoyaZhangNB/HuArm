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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5005))
    while True:
        data, addr = sock.recvfrom(1024)
        position = np.frombuffer(data, dtype=np.float32)
        yield position


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


def get_hair_end_tensions(model, data):
    """
    The old flexcomp had an auto-generated (unnamed) edge equality constraint
    enforcing every hair segment's length, which is what
    get_flex_edge_tension()/print_flex_tension() used to read as "hair
    tension". The rigid-body hair chain has no such constraint -- each
    segment is a rigid capsule, so there's nothing to stretch. Instead,
    "tension" now shows up as the reaction force MuJoCo has to apply at the
    two hair_to_frog/hair_to_tip <connect> equalities to keep the
    (inextensible) chain's ends pinned to bow_link_0 and bow_tip -- i.e. the
    force needed to hold the chain taut against its own hinge springs. That's
    a reasonable like-for-like replacement: it's still "how hard is the hair
    being held taut", just measured at the two endpoints instead of at every
    internal edge.

    Returns a dict {constraint_name: force_vector} for whichever of the two
    constraints currently have an active row in efc_force.
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


def compute_arm_ctrl_for_target(model, ik_data, target_pos, joint_names,
                                 body_name="bow_frog", max_iters=20):
    """
    Solves IK (warm-started from ik_data's current qpos) so `body_name`
    reaches target_pos, and returns {joint_name: solved_angle}.

    This replaces the old mocap-driven teleop. arm.xml no longer has a
    "bow_target" mocap body or a "teleop_coupling" weld -- per its own
    comments, those were removed because the arm now holds and moves the
    bow directly through its own actuated joints. So instead of writing a
    desired position into a mocap body and letting an equality constraint
    do the work, we solve for the joint angles that put the bow frog at the
    desired position and feed those angles to the arm's position actuators.

    ik_data is a scratch MjData reused across calls so the IK solve doesn't
    disturb the live simulation's qpos/qvel, and so each call can warm-start
    from the previous solution (the target moves smoothly, so a handful of
    damped-least-squares iterations per call is enough once warm-started;
    max_iters is kept low here for that reason -- raise it if you throttle
    how often this gets called).
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
    # data.xpos/xquat/etc. are all zero until forward kinematics has run once;
    # pretension_bow_hair() needs real bow_tip/bow_link_0 positions, so populate them now.
    mujoco.mj_forward(model, data)
    fix_hair_anchor_offsets(model)
    insert_hair_between_strings(model, data)
    pretension_bow_hair(model, data)

    arm_joint_names = ("joint1", "joint2", "joint3", "joint4")
    arm_actuator_ids = {jn: joint_to_actuator_id(model, jn) for jn in arm_joint_names}
    # Scratch data for warm-started per-frame IK; seed it with the pose
    # insert_hair_between_strings()/pretension_bow_hair() just settled on.
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos
    mujoco.mj_forward(model, ik_data)

    tension_print_interval = 0.5
    next_tension_print = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in the viewer to exit.")
        start = time.time()
        while viewer.is_running():
            
            # current_sim_time = data.time
            # desired_pos = get_desired_position(current_sim_time)
            # ctrl_targets = compute_arm_ctrl_for_target(model, ik_data, desired_pos, arm_joint_names)
            # for jn, angle in ctrl_targets.items():
            #     aid = arm_actuator_ids[jn]
            #     if aid >= 0:
            #         data.ctrl[aid] = angle
            elapsed_real = time.time() - start
            print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}")
            if data.time >= elapsed_real:
                time.sleep(0.01)
                print(f"Sim time {data.time:.3f} is ahead of real time {elapsed_real:.3f}, sleeping to catch up...")
                continue

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_tension_print:
                # print_hair_tension(model, data)
                next_tension_print = data.time + tension_print_interval
                force = data.sensor("bow_arm_contact").data.copy()
                print(f"t={data.time:6.3f} bow-arm contact force: {force} [N]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python teleop_bow.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])