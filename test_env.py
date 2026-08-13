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
from envs.utils_envs import init_huarm, jacobian_ik, joint_to_actuator_id

from utils import print_jp_dict, MetricsLogger
from agents.obs_normalizer import init_running_norm, update_running_norm, normalize_obs

# Arm joints whose IK is solved so that the bow_frog end-effector reaches the
# single 3D position teleop sends -- teleop only ever specifies where the
# end effector should be, not individual joint angles.
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")


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

    env = ErhuEnv(episode_time_limit=1000, max_ctrl_delta=0.05, f_safe=3, f_max=30, dr_pool_size=1, dr_pool_seed=1230)
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Teleoperation loop running. Press ESC in viewer to exit.")
        print("Drag the actuator sliders in the viewer's Control panel to command the arm.")
        start = time.time()
        try:
            while viewer.is_running():
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

                action = np.zeros(env.action_size, dtype=np.float32)
                for aid in arm_actuator_ids:
                    if aid < 0:
                        continue
                    delta = target_ctrl[aid] - sim_ctrl[aid]
                    action[aid] = np.clip(delta / env.max_ctrl_delta, -1.0, 1.0)

                state = _step(state, jp.asarray(action))

                obs = jp.expand_dims(state.obs, 0)
                norm = update_running_norm(norm, obs)
                obs_norm = normalize_obs(norm, obs)

                print(obs_norm)

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
                    metrics_logger.log(data.time, state.metrics["reward_terms"])

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")

        metrics_logger.plot("metrics.png")
        metrics_logger.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mujoco_model.py path/to/erhu_model.xml")
        sys.exit(1)

    main(sys.argv[1])