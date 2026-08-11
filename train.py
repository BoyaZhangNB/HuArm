import argparse
from pathlib import Path

import jax
import numpy as np
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
    # `agent.update()` carries the running observation-normalization
    # constants through its metrics dict every iteration (see
    # agents/obs_normalizer.py); stash the latest one here so it's still
    # available below however training ends (full run or Ctrl-C -- `train()`
    # catches KeyboardInterrupt internally and returns normally either way).
    last_metrics: dict = {}

    def log_fn(it, metrics):
        nonlocal last_metrics
        last_metrics = metrics
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
    param_key = {"ppo": 'params', "sac": 'actor_params'}[algo]
    params = train_state[param_key]

    # Save parameters
    print("Saving model parameters...")
    checkpointer = ocp.StandardCheckpointer()
    path = Path(train_cfg["checkpoint_path"]).resolve()
    checkpointer.save(path, params)
    checkpointer.wait_until_finished()

    # Save the running observation-normalization constants training ended
    # with, so inference/eval can reproduce the exact same obs scaling the
    # policy was trained on. Prefer train_state (always current) over
    # last_metrics's copy, which is only as fresh as the last logged
    # iteration.
    obs_norm = train_state.get("obs_norm")
    if obs_norm is None and "obs_norm" in last_metrics:
        obs_norm = last_metrics["obs_norm"]
    if obs_norm is not None:
        mean = obs_norm.mean if hasattr(obs_norm, "mean") else obs_norm["mean"]
        var = obs_norm.var if hasattr(obs_norm, "var") else obs_norm["var"]
        count = obs_norm.count if hasattr(obs_norm, "count") else obs_norm["count"]
        obs_norm_path = path.with_name(path.name + "_obs_norm.npz")
        print(f"Saving observation-normalization constants to {obs_norm_path}...")
        np.savez(
            obs_norm_path,
            mean=np.asarray(mean),
            var=np.asarray(var),
            count=np.asarray(count),
        )


if __name__ == "__main__":
    main()
