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
    # Base starting position of the bow_target
    base_x = 0.38
    base_y = 0.30
    base_z = 0.58
    
    # Bow back and forth along Y-axis (amplitude of 15cm, frequency of 0.5 Hz)
    y_offset = 0.15 * np.sin(2 * np.pi * 0.5 * t)
    
    # Press down slightly dynamically to contact the strings
    z_offset = -0.015 * np.abs(np.sin(2 * np.pi * 0.5 * t))
    
    return np.array([base_x, base_y + y_offset, base_z + z_offset])

def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")
    
    # Load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Get the ID of our mocap target body
    try:
        mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_target")
        # MuJoCo indexes mocap arrays using a specific mocap ID mapping
        mocap_idx = model.body_mocapid[mocap_body_id]
    except ValueError:
        print("Error: 'bow_target' body with mocap='true' not found in XML!")
        sys.exit(1)

    # Launch the passive interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        sim_start_time = data.time

        frog_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_frog")
        tip_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hair_to_tip")
        
        data.eq_active[frog_weld_id] = 0
        data.eq_active[tip_weld_id] = 0

        bow_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_tip")
        bow_link_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_link_1")

        tip_target_pos = data.xpos[bow_tip_id]
        rear_target_pos = data.xpos[bow_link_1_id]

        bow_hair_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_hair_35")
        bow_hair_rear_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bow_hair_0")

        data.qpos[bow_hair_tip_id:bow_hair_tip_id+3] = tip_target_pos
        data.qpos[bow_hair_rear_id:bow_hair_rear_id+3] = rear_target_pos

        mujoco.mj_forward(model, data)

        data.eq_active[frog_weld_id] = 1
        data.eq_active[tip_weld_id] = 1

        print("Teleoperation loop running. Press ESC in the viewer to exit.")
        
        while viewer.is_running():
            step_start = time.time()
            current_sim_time = data.time

            # 1. Get the desired position we want to move the bow to
            desired_pos = get_desired_position(current_sim_time)

            # 2. Update the mocap body's position in the state
            # This instantly translates the 'handle'. The weld constraint will then
            # pull the bow_frog, bow stick, and bow hair smoothly behind it.
            data.mocap_pos[mocap_idx] = desired_pos

            # 3. Step the physical simulation forward
            mujoco.mj_step(model, data)
            
            # 4. Sync the visualizer state
            viewer.sync()
            
            # Real-time clock synchronization
            # Match the execution time of the Python loop to the simulation timestep
            elapsed_real = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed_real
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python teleop_bow.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])