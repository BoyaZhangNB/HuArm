import sys
import numpy as np
import mujoco
import mujoco.viewer
import time

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


def fix_hair_anchor_offsets(model):
    """
    hair_to_frog/hair_to_tip are <connect> constraints between the flexcomp
    hair endpoints and points on the bow stick. Because the flexcomp is
    authored at its own XML position, independent of the bow's geometry, the
    two bodies are NOT coincident at compile time (bow_hair_0 sits ~4.7cm from
    bow_link_1, bow_hair_35 sits a similar distance from bow_tip). MuJoCo's
    compiler auto-derives each connect's "anchor in body2's frame" from that
    compile-time offset -- meaning the constraint's real target is "stay
    offset from body2 by the original gap", not "coincide with body2". That's
    what was silently undoing any runtime teleport/pretension. Zeroing the
    body2-frame anchor (eq_data[3:6]) makes the constraint actually pin the
    two points together.
    """
    for name in ("hair_to_frog", "hair_to_tip"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        model.eq_data[eq_id][3:6] = 0.0


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
    # pretension_bow_hair() needs real bow_tip/bow_link_1 positions, so populate them now.
    mujoco.mj_forward(model, data)

    try:
        mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_target")
        mocap_idx = model.body_mocapid[mocap_body_id]
    except ValueError:
        print("Error: 'bow_target' body with mocap='true' not found in XML!")
        sys.exit(1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        pretension_bow_hair(model, data)
        viewer.sync()

        print("Teleoperation loop running. Press ESC in the viewer to exit.")

        while viewer.is_running():
            step_start = time.time()
            current_sim_time = data.time

            desired_pos = get_desired_position(current_sim_time)
            data.mocap_pos[mocap_idx] = desired_pos

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed_real = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed_real
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python teleop_bow.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])