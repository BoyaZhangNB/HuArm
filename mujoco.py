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
        if eq_id >= 0:
            model.eq_data[eq_id][3:6] = 0.0

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
        viewer.sync()

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
                print_flex_tension(model, data)
                next_tension_print = data.time + tension_print_interval

            elapsed_real = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed_real
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python teleop_bow.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])