import jax
import orbax.checkpoint as ocp
from pathlib import Path
import mujoco
import time

from agent import PPOAgent
from envs.erhu_env import ErhuEnv

path = Path('checkpoints/model_latest').resolve()
checkpointer = ocp.StandardCheckpointer()

env = ErhuEnv()

agent = PPOAgent(obs_size=env.observation_size, action_size=env.action_size)

FRAME_SKIP = 40

def main():
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    model = env.mj_model
    data = env.mj_data

    # Initialize network
    agent_state = agent.init(jax.random.PRNGKey(0))
    restored_params = checkpointer.restore(path, agent_state["params"])
    agent_state = {"params": restored_params, "opt_state": agent_state["opt_state"]}


    mujoco.mj_forward(model, data)

    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        start = time.time()
        while viewer.is_running():
            elapsed_real = time.time() - start
            print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}", end="\r")
            
            if data.time >= elapsed_real:
                time.sleep(0.01)
                continue

            if step_count % FRAME_SKIP == 0:
                obs = env._get_obs(data, info=None)
                action, _ = agent.act(agent_state, obs, jax.random.PRNGKey(step_count))
                data.ctrl[:] = action
            
            mujoco.mj_step(model, data)

            step_count += 1
            viewer.sync()

if __name__ == "__main__":
    main()