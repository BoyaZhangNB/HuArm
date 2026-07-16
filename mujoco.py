import sys
import mujoco as mujoco
import mujoco.viewer
import time

def main(xml_path):
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    bow_addr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bow_free")]
    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            data.qvel[bow_addr:bow_addr+3] = [float(1), 0, 0]



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_model.py path/to/model.xml")
        sys.exit(1)

    main(sys.argv[1])