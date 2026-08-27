import sys
import threading
import numpy as np
import mujoco
import mujoco.viewer
import time
import socket
from mujoco import mjx
import jax
import jax.numpy as jp
from envs.erhu_env import ErhuEnv
from envs.utils_envs import joint_to_actuator_id

from utils import print_jp_dict, MetricsLogger
from agents.obs_normalizer import init_running_norm, update_running_norm, normalize_obs

# Arm joints whose IK is solved so that the bow_frog end-effector reaches the
# single 3D position teleop sends -- teleop only ever specifies where the
# end effector should be, not individual joint angles.
ARM_JOINT_NAMES = ("joint1", "joint2", "joint5", "joint3", "joint4")

# --- TEMPORARY: lets you trigger env.reset() by pressing Enter in the
# terminal while the viewer loop is running. Remove once no longer needed.
def _listen_for_reset_key(reset_event):
    while True:
        try:
            input()
        except EOFError:
            break
        reset_event.set()


def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    env = ErhuEnv(episode_time_limit=1000, max_ctrl_delta=0.05, f_safe=3, f_max=30, dr_pool_size=128, dr_pool_seed=420)
    state = env.reset(jax.random.PRNGKey(0))
    print(f"Environment reset.")
    model = env.mj_model
    data = mjx.get_data(model, state.pipeline_state)

    # Map each arm joint to the actuator that drives it, so slider edits made
    # in the viewer's own Control panel (which write straight into
    # data.ctrl) can be translated into per-actuator delta actions.
    arm_actuator_ids = [joint_to_actuator_id(model, jn) for jn in ARM_JOINT_NAMES]

    log_print_interval = 0.5
    next_tension_print = 0
    metrics_logger = MetricsLogger(live=True)

    norm = init_running_norm(obs_size=state.obs.shape[0], dtype=jp.float32)

    _step = jax.jit(env.step)
    state = _step(state, jp.zeros(env.action_size))
    # Fill the same MjData object the viewer was launched with, rather than
    # rebinding `data` to a new object mujoco.viewer never sees.
    mjx.get_data_into(data, model, state.pipeline_state)
    sim_ctrl = data.ctrl.copy()  # viewer writes directly into data.ctrl on its own thread

    # TEMPORARY: background thread that sets reset_event whenever Enter is
    # pressed in the terminal, so the main loop below can reset the env.
    reset_event = threading.Event()
    reset_key_counter = [0]
    threading.Thread(target=_listen_for_reset_key, args=(reset_event,), daemon=True).start()
    print("Press Enter in this terminal at any time to reset the environment.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in viewer to exit.")
        print("Drag the actuator sliders in the viewer's Control panel to command the arm.")
        start = time.time()
        try:
            while viewer.is_running():
                if reset_event.is_set():
                    reset_event.clear()
                    reset_key_counter[0] += 1
                    state = env.reset(jax.random.PRNGKey(20 + reset_key_counter[0]))
                    state = _step(state, jp.zeros(env.action_size))
                    mjx.get_data_into(data, model, state.pipeline_state)
                    sim_ctrl = data.ctrl.copy()
                    viewer.sync()
                    start = time.time()
                    print("\nEnvironment reset (manual).")

                elapsed_real = time.time() - start
                print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}", end="\r")

                if data.time >= elapsed_real:
                    time.sleep(0.01)
                    continue

                # The viewer writes any Control-panel slider drags directly into
                # data.ctrl on its own thread, so grab a consistent snapshot
                # under the viewer's lock before comparing it against what the
                # sim is actually holding (sim_ctrl) to get the user's command.
                with viewer.lock():
                    target_ctrl = data.ctrl.copy()

                # action[-1] (bow_frog_hinge friction-clamp stiffness delta --
                # see ErhuEnv's action[5] docstring) has no actuator, so
                # nothing in the viewer's Control panel can drive it. Sending
                # -1.0 every step pins info["frog_stiffness"] at its 0.0
                # (loosest) floor -- it starts there already (reset()'s
                # default), so this just keeps this manual-test loop on the
                # hinge's old passive-only behavior instead of silently
                # ramping up friction over time.
                action = np.zeros(env.action_size, dtype=np.float32)
                action[-1] = -1.0
                for aid in arm_actuator_ids:
                    if aid < 0:
                        continue
                    delta = target_ctrl[aid] - sim_ctrl[aid]
                    action[aid] = np.clip(delta / env.max_ctrl_delta, -1.0, 1.0)

                state = _step(state, jp.asarray(action))

                # obs = jp.expand_dims(state.obs, 0)
                # norm = update_running_norm(norm, obs)
                # obs_norm = normalize_obs(norm, obs)
                # print(f"Normalized obs: {obs_norm}")


                if state.done:
                    print(f"\nEpisode terminated")
                    metrics_logger.plot("metrics.png")
                    metrics_logger.close()
                    exit(0)
                mjx.get_data_into(data, model, state.pipeline_state)
                sim_ctrl = data.ctrl.copy()

                viewer.sync()

                if data.time >= next_tension_print:
                    next_tension_print = data.time + log_print_interval
                    # print_jp_dict(state.metrics)
                    metrics_logger.log(data.time, state.info)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")

        metrics_logger.plot("metrics.png")
        metrics_logger.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mujoco_model.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])