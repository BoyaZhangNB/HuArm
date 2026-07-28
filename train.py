import jax

from training_interface import train
from example.example_agent import ReinforceAgent
from envs.wrappers import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from envs.erhu_env import ErhuEnv

NUM_ENVS = 4
EPISODE_LENGTH = 200
STEPS_PER_ITER = 200
NUM_ITERS = 2


def main():
    # 1. Build the env: task-specific model wrapped with reusable infra.
    env = ErhuEnv(n_frames=40)
    env = EpisodeWrapper(env, episode_length=EPISODE_LENGTH)
    env = AutoResetWrapper(env)
    env = VmapWrapper(env, batch_size=NUM_ENVS)

    # 2. Build any agent satisfying the Agent protocol -- fully decoupled
    #    from the env/model above.
    agent = ReinforceAgent(obs_size=env.observation_size, action_size=env.action_size)

    # 3. Train. Swapping `agent` swaps the whole algorithm; swapping the
    #    env class swaps the whole task/robot. Neither affects the other.
    def log_fn(it, metrics):
        print(f"iter {it:3d}  loss={float(metrics['loss']):.4f}")

    train(
        env=env,
        agent=agent,
        rng=jax.random.PRNGKey(0),
        num_iterations=NUM_ITERS,
        steps_per_iteration=STEPS_PER_ITER,
        log_fn=log_fn,
    )


if __name__ == "__main__":
    main()
