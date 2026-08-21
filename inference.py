"""inference.py -- run a trained policy on ErhuEnv in the interactive MuJoCo
viewer.

Works with checkpoints from either training path, selected via --algo:

    train.py         RL checkpoints, e.g. checkpoints/model_latest (ppo),
                      checkpoints/sac_model_latest (sac)
    bc/train_bc.py    BC checkpoints, e.g. bc/checkpoints/bc_sac_latest

Both are saved with orbax's StandardCheckpointer holding the *same* policy
param pytree for a given algo (PPOAgent's `_PolicyNet` params for ppo,
SACAgent's `_ActorNet` params for sac -- see bc/train_bc.py's module
docstring and train.py's save step), plus a "<path>_obs_norm.npz" sidecar of
running observation-normalization stats. So loading either kind only takes
knowing which algo it was trained as; --checkpoint accepts either path.

Usage:
    python inference.py                                              # ppo, checkpoints/model_latest
    python inference.py --algo sac                                   # sac, checkpoints/sac_model_latest
    python inference.py --algo sac --checkpoint bc/checkpoints/bc_sac_latest
    python inference.py --algo sac --checkpoint checkpoints/sac_model_latest --stochastic
    python inference.py --xml huarm/arm_rigid.xml                        # override ErhuEnv's model XML
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
from mujoco import mjx

from agents.obs_normalizer import RunningNorm
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from envs.erhu_env import ErhuEnv
from training_interface import Agent
from utils import MetricsLogger, print_jp_dict

# Same algo -> (agent class, train_state param key) mapping train.py uses.
AGENT_CLASSES = {"ppo": PPOAgent, "sac": SACAgent}
PARAM_KEYS = {"ppo": "params", "sac": "actor_params"}
DEFAULT_CHECKPOINTS = {
    "ppo": "checkpoints/model_latest",
    "sac": "checkpoints/sac_model_latest",
}


def build_agent(algo: str, obs_size: int, action_size: int) -> Agent:
    if algo == "sac":
        # Inference never calls SACAgent.update(), so the (obs_size *
        # buffer_capacity)-sized replay buffer agent.init() would otherwise
        # allocate (500_000 by default) is pure waste here -- shrink it to
        # the minimum instead of paying for training-only state.
        return SACAgent(obs_size=obs_size, action_size=action_size, buffer_capacity=1)
    return AGENT_CLASSES[algo](obs_size=obs_size, action_size=action_size)


def load_policy(agent: Agent, algo: str, checkpoint_path: Path, seed: int = 0) -> Any:
    """Restore `checkpoint_path` (a BC or RL StandardCheckpointer dir, see
    module docstring) into a freshly-initialized `agent`'s train_state,
    along with its "<path>_obs_norm.npz" sidecar if one was saved alongside
    it. Falls back to untrained (mean=0, var=1) obs-normalization stats,
    with a warning, if no sidecar is found.
    """
    param_key = PARAM_KEYS[algo]
    train_state = agent.init(jax.random.PRNGKey(seed))

    checkpointer = ocp.StandardCheckpointer()
    train_state[param_key] = checkpointer.restore(checkpoint_path, train_state[param_key])

    obs_norm_path = checkpoint_path.with_name(checkpoint_path.name + "_obs_norm.npz")
    if obs_norm_path.exists():
        npz = np.load(obs_norm_path)
        train_state["obs_norm"] = RunningNorm(
            mean=jp.asarray(npz["mean"]), var=jp.asarray(npz["var"]), count=jp.asarray(npz["count"])
        )
    else:
        print(f"[WARNING] {obs_norm_path} not found; using untrained obs-normalization stats.")

    return train_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--algo", choices=sorted(AGENT_CLASSES), default="ppo",
        help="Policy architecture the checkpoint was trained with (default: ppo).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to an orbax StandardCheckpointer directory -- a train.py RL "
             "checkpoint or a bc/train_bc.py BC checkpoint (see module docstring). "
             "Default: checkpoints/model_latest (ppo) or checkpoints/sac_model_latest "
             "(sac), matching train.py's default --config checkpoint_path.",
    )
    parser.add_argument(
        "--xml", default="huarm/arm.xml",
        help="Path to the MuJoCo model XML for ErhuEnv (default: huarm/arm.xml).",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions from the policy's distribution instead of its "
             "deterministic (tanh-mean) action.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for env reset and action sampling.")
    parser.add_argument(
        "--episode-time-limit", type=float, default=1000.0,
        help="ErhuEnv episode_time_limit in seconds of sim time (default: 1000).",
    )
    parser.add_argument(
        "--log-interval", type=float, default=0.5,
        help="Seconds of sim time between printed/logged metric snapshots (default: 0.5).",
    )
    parser.add_argument(
        "--metrics-path", default="metrics.png",
        help="Where to save the reward-terms plot on exit (default: metrics.png).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the raw action vector every step.",
    )
    args = parser.parse_args()
    args.checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS[args.algo]).resolve()
    return args


def main() -> None:
    args = parse_args()
    print(f"Using MuJoCo Version: {mujoco.__version__}")

    env = ErhuEnv(xml_path=args.xml, episode_time_limit=args.episode_time_limit, dr_pool_size=128)
    agent = build_agent(args.algo, env.observation_size, env.action_size)

    print(f"Loading {args.algo} checkpoint from {args.checkpoint}...")
    train_state = load_policy(agent, args.algo, args.checkpoint, seed=args.seed)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    model = env.mj_model
    data = mjx.get_data(model, state.pipeline_state)

    metrics_logger = MetricsLogger(live=True)
    deterministic = not args.stochastic

    _step = jax.jit(env.step)
    # Warm up the jit before the viewer opens so the first real step doesn't
    # stall the real-time tracking below.
    state = _step(state, jp.zeros(env.action_size))
    mjx.get_data_into(data, model, state.pipeline_state)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print(f"Running {args.algo} policy ({'deterministic' if deterministic else 'stochastic'}). "
              "Press ESC in viewer to exit.")
        start = time.time()
        next_log = 0.0

        try:
            while viewer.is_running():
                elapsed_real = time.time() - start
                print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}", end="\r")

                if data.time >= elapsed_real:
                    time.sleep(0.01)
                    continue

                rng, act_rng = jax.random.split(rng)
                action, _ = agent.act(train_state, state.obs, act_rng, deterministic=deterministic)
                if args.verbose:
                    print(f"Action: {action}")

                state = _step(state, action)
                if state.done:
                    print("\nEpisode terminated")
                    break
                mjx.get_data_into(data, model, state.pipeline_state)
                viewer.sync()

                if data.time >= next_log:
                    next_log = data.time + args.log_interval
                    print_jp_dict(state.metrics)
                    metrics_logger.log(data.time, state.metrics)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")

        metrics_logger.plot(args.metrics_path)
        metrics_logger.close()


if __name__ == "__main__":
    main()
