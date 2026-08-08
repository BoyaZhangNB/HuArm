import argparse
from pathlib import Path

import jax
import orbax.checkpoint as ocp
import yaml

from training_interface import train
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from envs.erhu_env import ErhuEnv
from utils import MetricsLogger, print_jp_dict, make_env

DEFAULT_CONFIG_PATH = "configs/train.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to training YAML config."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    env_cfg = dict(cfg["env"])
    num_envs = env_cfg.pop("num_envs")
    episode_length = env_cfg.pop("episode_length")
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    # 1. Build the env: task-specific model wrapped with reusable infra.
    #    Remaining env_cfg keys are forwarded as ErhuEnv kwargs.
    env = make_env(
        ErhuEnv, ep_len=episode_length, num_envs=num_envs, **env_cfg
    )

    # 2. Build any agent satisfying the Agent protocol -- fully decoupled
    #    from the env/model above. `algo` selects which one; PPO stays the
    #    default so existing configs are unaffected.
    algo = cfg.get("algo", "ppo")
    agent_cls = {"ppo": PPOAgent, "sac": SACAgent}[algo]
    agent = agent_cls(obs_size=env.observation_size, action_size=env.action_size, **agent_cfg)

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
        rng=jax.random.PRNGKey(cfg["seed"]),
        num_iterations=train_cfg["num_iterations"],
        steps_per_iteration=train_cfg["steps_per_iteration"],
        eval_interval=train_cfg["eval_interval"],
        eval_episodes=train_cfg["eval_episodes"],
        log_fn=log_fn,
    )
    logger.plot(train_cfg["metrics_path"])
    logger.close()
    params = train_state["params"]

    # Save parameters
    print("Saving model parameters...")
    checkpointer = ocp.StandardCheckpointer()
    path = Path(train_cfg["checkpoint_path"]).resolve()
    checkpointer.save(path, params)
    checkpointer.wait_until_finished()


if __name__ == "__main__":
    main()
