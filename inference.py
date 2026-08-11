import sys
import mujoco
import mujoco.viewer
import time
from mujoco import mjx
import jax
import jax.numpy as jp
import numpy as np
from envs.erhu_env import ErhuEnv
from agents.ppo_agent import PPOAgent
from agents.obs_normalizer import RunningNorm

from utils import print_jp_dict, MetricsLogger
import orbax.checkpoint as ocp
from pathlib import Path


path = Path('checkpoints/model_latest').resolve()
checkpointer = ocp.StandardCheckpointer()

    
def main(xml_path):
    print(f"Using MuJoCo Version: {mujoco.__version__}")


    env = ErhuEnv(episode_time_limit=1000)

    agent = PPOAgent(obs_size=env.observation_size, action_size=env.action_size)

    state = env.reset(jax.random.PRNGKey(0))
    model = env.mj_model
    data = mjx.get_data(model, state.pipeline_state)

    # Initialize network
    agent_state = agent.init(jax.random.PRNGKey(0))
    restored_params = checkpointer.restore(path, agent_state["params"])
    obs_norm = agent_state["obs_norm"]
    obs_norm_path = path.with_name(path.name + "_obs_norm.npz")
    if obs_norm_path.exists():
        # Saved by train.py at the end of training -- see train.py /
        # agents/obs_normalizer.py. Without this, act() would normalize
        # eval-time observations with the untrained (mean=0, var=1) stats
        # instead of what the policy was actually trained on.
        npz = np.load(obs_norm_path)
        obs_norm = RunningNorm(
            mean=jp.asarray(npz["mean"]), var=jp.asarray(npz["var"]), count=jp.asarray(npz["count"])
        )
    else:
        print(f"[WARNING] {obs_norm_path} not found; using untrained obs normalization stats.")
    agent_state = {
        "params": restored_params,
        "opt_state": agent_state["opt_state"],
        "obs_norm": obs_norm,
    }

    log_print_interval = 0.5
    next_tension_print = 0
    metrics_logger = MetricsLogger(live=True)

    _step = jax.jit(env.step)
    state = _step(state, jp.zeros(env.action_size))
    # Fill the same MjData object the viewer was launched with, rather than
    # rebinding `data` to a new object mujoco.viewer never sees.
    mjx.get_data_into(data, model, state.pipeline_state)

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
                
                action, _ = agent.act(
                    agent_state, state.obs, jax.random.PRNGKey(0), deterministic=True
                )
                print(f"Action: {action}")

                state = _step(state, action)
                if state.done:
                    print(f"\nEpisode terminated")
                    metrics_logger.plot("metrics.png")
                    metrics_logger.close()
                    exit(0)
                mjx.get_data_into(data, model, state.pipeline_state)

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