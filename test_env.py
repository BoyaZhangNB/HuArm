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

from utils import print_jp_dict, MetricsLogger
from agents.obs_normalizer import init_running_norm, update_running_norm, normalize_obs

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

    env = ErhuEnv(episode_time_limit=1000, max_joint_vel=1.25, f_safe=3, f_max=30, dr_pool_size=128, dr_pool_seed=420)
    state = env.reset(jax.random.PRNGKey(0))
    print(f"Environment reset.")
    model = env.mj_model
    data = mjx.get_data(model, state.pipeline_state)

    # Per-actuator scale of a normalized action (action[i] drives ctrl[i]):
    # arm servo target delta (rad), then the bow_frog_hinge torque delta (N*m).
    # Slider edits made in the viewer's own Control panel (which write
    # straight into data.ctrl) are translated into these delta actions.
    delta_scale = np.full(env.action_size, env.max_ctrl_delta)
    delta_scale[env._frog_aid] = env.max_frog_torque_delta

    log_print_interval = 0.5
    next_tension_print = 0
    metrics_logger = MetricsLogger(live=True)

    norm = init_running_norm(obs_size=state.obs.shape[0], dtype=jp.float32)

    _step = jax.jit(env.step)
    state = _step(state, jp.zeros(env.action_size))
    # Fill the same MjData object the viewer was launched with, rather than
    # rebinding `data` to a new object mujoco.viewer never sees.
    mjx.get_data_into(data, model, state.pipeline_state)
    # The Control-panel targets (servo angles in rad, frog torque in N*m),
    # and the ctrl the env last wrote back into data.ctrl -- see the main loop.
    ctrl_target = data.ctrl.copy()
    last_ctrl = data.ctrl.copy()

    # TEMPORARY: background thread that sets reset_event whenever Enter is
    # pressed in the terminal, so the main loop below can reset the env.
    reset_event = threading.Event()
    reset_key_counter = [0]
    threading.Thread(target=_listen_for_reset_key, args=(reset_event,), daemon=True).start()
    print("Press Enter in this terminal at any time to reset the environment.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in viewer to exit.")
        print("Drag the actuator sliders in the viewer's Control panel to command joint angles")
        print("(the last slider, bow_frog_motor, is a frog hinge torque target in N*m).")
        start = time.time()
        try:
            while viewer.is_running():
                if reset_event.is_set():
                    reset_event.clear()
                    reset_key_counter[0] += 1
                    state = env.reset(jax.random.PRNGKey(20 + reset_key_counter[0]))
                    state = _step(state, jp.zeros(env.action_size))
                    mjx.get_data_into(data, model, state.pipeline_state)
                    ctrl_target = data.ctrl.copy()
                    last_ctrl = data.ctrl.copy()
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
                # under the viewer's lock.
                with viewer.lock():
                    slider_ctrl = data.ctrl.copy()

                # Every action dim is a rate-limited *delta* on the ctrl the
                # env holds (see ErhuEnv's docstring), so each slider is
                # treated as a target to ramp toward. The rate limit means
                # the env writes back an intermediate ctrl each step
                # (get_data_into below), which would otherwise overwrite the
                # slider -- so only take a new target from a slider that
                # moved off what the env last wrote.
                moved = slider_ctrl != last_ctrl
                ctrl_target[moved] = slider_ctrl[moved]
                action = np.clip((ctrl_target - last_ctrl) / delta_scale, -1.0, 1.0).astype(np.float32)

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
                last_ctrl = data.ctrl.copy()

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