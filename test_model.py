import sys
import mujoco
import mujoco.viewer
import time
from envs.utils_envs import init_huarm


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