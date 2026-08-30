import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
import yaml

from training_interface import train
from agents.obs_normalizer import RunningNorm
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from demonstrations.demo_buffer import load_demo_transitions
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
    parser.add_argument(
        "--bc-checkpoint", default=None,
        help="Path to a bc/train_bc.py checkpoint (e.g. bc/checkpoints/bc_sac_latest) to "
             "warm-start the policy from, instead of agent.init()'s random weights. Must "
             "have been trained with --algo matching this config's `algo` (param shapes are "
             "restored structurally against a freshly-init'd agent of that type).",
    )
    parser.add_argument(
        "--demo-dir", nargs="?", const="demonstrations", default=None,
        help="Seed SACAgent's replay buffer with teleop.py demonstrations (demo_*.npz files "
             "under this directory, default 'demonstrations' if flag given with no value) "
             "before training starts, instead of/alongside --bc-checkpoint's supervised "
             "warm start: this is RL *with* prior data rather than imitation -- the demo "
             "(obs, action, reward, next_obs, done) transitions just become the oldest "
             "entries already in the buffer, and training then proceeds completely normally "
             "(see agents/sac_agent.py's `seed_buffer_overrides`). Only valid with `algo: sac`.",
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
    param_key = {"ppo": "params", "sac": "actor_params"}[algo]

    # 2b. Optionally warm-start the policy from a bc/train_bc.py checkpoint
    # (imitation-learned from teleop.py demonstrations) instead of
    # agent.init()'s random weights -- see training_interface.train()'s
    # `init_train_state_overrides`.
    init_overrides = None
    if args.bc_checkpoint:
        bc_path = Path(args.bc_checkpoint).resolve()
        dummy_state = agent.init(jax.random.PRNGKey(cfg["seed"]))
        bc_checkpointer = ocp.StandardCheckpointer()
        restored_params = bc_checkpointer.restore(bc_path, dummy_state[param_key])
        init_overrides = {param_key: restored_params}

        bc_obs_norm_path = bc_path.with_name(bc_path.name + "_obs_norm.npz")
        if bc_obs_norm_path.exists():
            npz = np.load(bc_obs_norm_path)
            init_overrides["obs_norm"] = RunningNorm(
                mean=jp.asarray(npz["mean"]), var=jp.asarray(npz["var"]), count=jp.asarray(npz["count"])
            )
        else:
            print(f"[WARNING] {bc_obs_norm_path} not found; keeping untrained obs-normalization stats.")
        print(f"Warm-starting {algo} policy from BC checkpoint: {bc_path}")

    # 2c. Optionally seed SACAgent's replay buffer with teleop.py
    # demonstrations, so off-policy training starts with real expert
    # transitions already in the buffer alongside whatever the (still
    # randomly-initialized, unless --bc-checkpoint above was also given)
    # policy collects itself -- see agents/sac_agent.py's
    # `seed_buffer_overrides` module-level docstring for why this is RL
    # with prior data rather than a second flavor of BC.
    if args.demo_dir:
        if algo != "sac":
            raise ValueError(
                f"--demo-dir seeds SACAgent's replay buffer and doesn't apply to algo={algo!r} "
                "(no replay buffer to seed); drop --demo-dir or switch to `algo: sac`."
            )
        demo = load_demo_transitions(args.demo_dir)
        assert demo.obs.shape[-1] == env.observation_size, (
            f"demo obs dim {demo.obs.shape[-1]} != ErhuEnv.observation_size "
            f"{env.observation_size} -- demos under {args.demo_dir} were recorded against a "
            "different obs layout than the current ErhuEnv."
        )
        assert demo.action.shape[-1] == env.action_size, (
            f"demo action dim {demo.action.shape[-1]} != ErhuEnv.action_size {env.action_size}."
        )
        demo_overrides = agent.seed_buffer_overrides(
            demo.obs, demo.action, demo.reward, demo.next_obs, demo.done,
            obs_norm=(init_overrides or {}).get("obs_norm"),
        )
        init_overrides = {**(init_overrides or {}), **demo_overrides}
        print(
            f"Seeded SAC replay buffer with {demo.obs.shape[0]} demo transitions "
            f"from {args.demo_dir}"
        )

    # 3. Train. Swapping `agent` swaps the whole algorithm; swapping the
    #    env class swaps the whole task/robot. Neither affects the other.
    logger = MetricsLogger(live=False)
    # `agent.update()` carries the running observation-normalization
    # constants through its metrics dict every iteration (see
    # agents/obs_normalizer.py); stash the latest one here so it's still
    # available below however training ends (full run or Ctrl-C -- `train()`
    # catches KeyboardInterrupt internally and returns normally either way).
    last_metrics: dict = {}

    # Track the best eval reward seen so far and, whenever an eval beats it,
    # save the corresponding params to a dedicated "_best" checkpoint --
    # separate from the final-iteration checkpoint saved below, so a run
    # that overfits/collapses late still leaves behind its best policy.
    best_eval_reward = float("-inf")
    best_checkpointer = ocp.StandardCheckpointer()
    best_checkpoint_path = Path(train_cfg["checkpoint_path"]).resolve()
    best_checkpoint_path = best_checkpoint_path.with_name(best_checkpoint_path.name + "_best")

    def checkpoint_fn(it, train_state, eval_metrics):
        nonlocal best_eval_reward
        eval_reward = float(eval_metrics["reward"])
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_checkpointer.save(best_checkpoint_path, train_state[param_key], force=True)
            print(f"iter {it:3d}  new best eval_reward={eval_reward:.4f} -- saved {best_checkpoint_path}")

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
        init_train_state_overrides=init_overrides,
        # Hold the eval rng fixed across the whole run, so consecutive eval
        # points differ only by the policy -- and give eval enough steps for
        # the requested number of episodes to actually finish, instead of
        # `eval`'s 2000-step default silently averaging whatever fraction of
        # them fits (~3 at num_envs=1 and episode_length=512).
        eval_rng=jax.random.PRNGKey(cfg["seed"] + 1),
        eval_max_steps=train_cfg["eval_episodes"] * episode_length + episode_length,
        checkpoint_fn=checkpoint_fn,
    )
    best_checkpointer.wait_until_finished()
    logger.plot(train_cfg["metrics_path"])
    logger.close()
    params = train_state[param_key]

    # Save parameters
    print("Saving model parameters...")
    checkpointer = ocp.StandardCheckpointer()
    path = Path(train_cfg["checkpoint_path"]).resolve()
    checkpointer.save(path, params)
    checkpointer.wait_until_finished()

    # Copy the exact config used to train this checkpoint, and a copy of the
    # training plot, into the checkpoint folder itself -- so each checkpoint
    # is self-describing even after configs/metrics_path get edited/reused
    # for later runs.
    shutil.copy(Path(args.config).resolve(), path / Path(args.config).name)
    metrics_src = Path(train_cfg["metrics_path"]).resolve()
    if metrics_src.exists():
        shutil.copy(metrics_src, path / metrics_src.name)

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
