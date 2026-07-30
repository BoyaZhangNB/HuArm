import sys
import numpy as np
import mujoco
import mujoco.viewer
import time
import socket
from envs.utils_envs import jacobian_ik, init_huarm


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

def compute_arm_ctrl_for_target(model, ik_data, target_pos, joint_names,
                                 body_name="bow_frog", local_offset=None, max_iters=20):
    """
    IK solver for positioning a point on `body_name` (its origin, unless
    local_offset is given) with arm joint actuation.
    """
    if local_offset is None:
        local_offset = np.zeros(3)
    body_points = [(body_name, local_offset, 1.0)]
    jacobian_ik(model, ik_data, body_points, target_pos, list(joint_names), max_iters=max_iters)
    qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in joint_names]
    return {jn: ik_data.qpos[qi] for jn, qi in zip(joint_names, qpos_idxs)}

def print_all_contact(model, data):
    print("\n--- Theoretically Allowed Contact Pairs ---")
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            # Exclude geoms on the same body if parent/child contact is disabled
            if model.geom_bodyid[i] == model.geom_bodyid[j]:
                continue
                
            type_i, aff_i = model.geom_contype[i], model.geom_conaffinity[i]
            type_j, aff_j = model.geom_contype[j], model.geom_conaffinity[j]
            
            # Check bitmask condition
            if (type_i & aff_j) or (type_j & aff_i):
                name_i = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                name_j = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j) or f"geom_{j}"
                print(f"{name_i} <---> {name_j}")

def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    model, data = init_huarm(model, data)

    tension_print_interval = 0.5
    next_tension_print = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in viewer to exit.")
        print_all_contact(model, data)
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