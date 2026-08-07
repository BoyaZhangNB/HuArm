import jax
import orbax.checkpoint as ocp
from pathlib import Path

from training_interface import train
from agent import PPOAgent
from envs.wrappers import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from envs.erhu_env import ErhuEnv
from utils import MetricsLogger, print_jp_dict

NUM_ENVS = 64
EPISODE_LENGTH = 2000
STEPS_PER_ITER = 2000
NUM_ITERS = 20
FRAME_SKIP = 20 # control freq = 0.002 * 20 = 0.04s per step, 25Hz

def main():
    # 1. Build the env: task-specific model wrapped with reusable infra.
    env = ErhuEnv(n_frames=FRAME_SKIP)
    env = EpisodeWrapper(env, episode_length=EPISODE_LENGTH)
    env = AutoResetWrapper(env)
    env = VmapWrapper(env, num_envs=NUM_ENVS)

    # 2. Build any agent satisfying the Agent protocol -- fully decoupled
    #    from the env/model above.
    agent = PPOAgent(obs_size=env.observation_size, action_size=env.action_size)

    # 3. Train. Swapping `agent` swaps the whole algorithm; swapping the
    #    env class swaps the whole task/robot. Neither affects the other.
    logger = MetricsLogger(live=False)

    def log_fn(it, metrics):
        logger.log(it, metrics)
        eval_reward = metrics.get("eval_reward", float("nan"))
        param_norm = float(metrics["param_norm"])
        params_isnan = bool(metrics["params_isnan"])
        print(
            f"iter {it:3d}  loss={float(metrics['loss']):.4f} eval_reward={float(eval_reward):.4f} "
            f"param_norm={param_norm:.4f} params_isnan={params_isnan}"
        )
        if "eval_reward" in metrics:
            print_jp_dict(metrics)

    train_state = train(
        env=env,
        agent=agent,
        rng=jax.random.PRNGKey(0),
        num_iterations=NUM_ITERS,
        steps_per_iteration=STEPS_PER_ITER,
        eval_interval=5,
        log_fn=log_fn,
    )
    logger.plot("metrics.png")
    logger.close()
    params = train_state["params"]

    # Save parameters
    print("Saving model parameters...")
    checkpointer = ocp.StandardCheckpointer()
    path = Path('checkpoints/model_latest').resolve()
    checkpointer.save(path, params)
    checkpointer.wait_until_finished() 


if __name__ == "__main__":
    main()
