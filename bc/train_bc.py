"""
bc/train_bc.py -- behavior cloning on teleop.py demonstrations.

Loads the `demo_*.npz` episodes `teleop.py`'s `DemoRecorder` saves under
`demonstrations/` and pretrains ONE of `agents/`'s two policy networks
(PPOAgent's `_PolicyNet` or SACAgent's `_ActorNet`) via supervised imitation,
so RL fine-tuning (`train.py`) can start from a policy that already roughly
reproduces expert bowing behavior instead of random noise.

Both agents' policies are tanh-squashed Gaussians over the same
`[-1, 1]^action_size` normalized delta-ctrl action space `teleop.py` already
logs `action` in (see envs/erhu_env.py's docstring and teleop.py's module
docstring: "a BC/DAgger policy trained on these demos ... must reproduce
actions in the exact same space"). So the natural BC loss is the negative
log-likelihood of each expert action under the network's squashed-Gaussian
output -- see `squashed_gaussian_nll` below.

This script imports the *actual* `_PolicyNet`/`_ActorNet` classes from
agents/ppo_agent.py / agents/sac_agent.py (rather than redefining a
lookalike network) so the params pytree this script produces is structurally
identical to what `PPOAgent.init`/`SACAgent.init` produce. The checkpoint is
saved with the same layout train.py's end-of-training save uses (an orbax
`StandardCheckpointer` directory plus a "<path>_obs_norm.npz" sidecar of
running observation-normalization constants) -- train.py's `--bc-checkpoint`
flag restores it straight into a fresh agent's train_state as a warm start.

Usage:
    python -m bc.train_bc --algo sac
    python -m bc.train_bc --algo ppo --epochs 300
    python train.py --config configs/train_sac.yaml \\
        --bc-checkpoint bc/checkpoints/bc_sac_latest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np
import optax
import orbax.checkpoint as ocp

from agents.obs_normalizer import init_running_norm, normalize_obs, update_running_norm
from agents.ppo_agent import _PolicyNet
from agents.sac_agent import _ActorNet
from utils import MetricsLogger

Episode = Tuple[Path, np.ndarray, np.ndarray]  # (path, obs [T, obs_size], action [T, action_size])


# ---------------------------------------------------------------------------
# Squashed-Gaussian BC loss -- same log-prob math as agents/ppo_agent.py's
# `_squashed_gaussian_log_prob` / the correction term inside
# agents/sac_agent.py's `_sample_squashed`, evaluated at a fixed (expert)
# action rather than a sampled one. Kept local instead of importing those
# (also-private) helpers so this script only depends on each agent module's
# network class, not its training-loop internals.
# ---------------------------------------------------------------------------

def _atanh(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = jp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jp.log((1.0 + x) / (1.0 - x))


def squashed_gaussian_nll(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Per-sample negative log-likelihood of an already-squashed `action`
    (e.g. a demo's [-1, 1]^5 action) under a tanh-squashed Gaussian with the
    given (pre-tanh) mean/log_std."""
    pre_tanh = _atanh(action)
    std = jp.exp(log_std)
    log_prob = -0.5 * jp.sum(
        ((pre_tanh - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1
    )
    log_prob -= jp.sum(jp.log(1.0 - jp.square(action) + 1e-6), axis=-1)
    return -log_prob


# ---------------------------------------------------------------------------
# Dataset loading -- mirrors demonstrations/clean_demo.py's file glob.
# ---------------------------------------------------------------------------

def load_episodes(demo_dir: Path, min_len: int) -> List[Episode]:
    paths = sorted(demo_dir.glob("demo_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No demo_*.npz files found under {demo_dir}")
    episodes = []
    for p in paths:
        with np.load(p) as d:
            obs, action = d["obs"], d["action"]
        if len(obs) < min_len:
            print(f"[skip] {p.name}: {len(obs)} steps < --min-demo-len {min_len}")
            continue
        episodes.append((p, obs.astype(np.float32), action.astype(np.float32)))
    if not episodes:
        raise ValueError(f"All demos under {demo_dir} were shorter than --min-demo-len {min_len}")
    return episodes


def split_episodes(
    episodes: List[Episode], val_frac: float, seed: int
) -> Tuple[List[Episode], List[Episode]]:
    """Holds out whole episodes for validation rather than individual steps:
    adjacent steps within one episode are highly correlated (same stroke,
    25Hz), so a per-step random split would leak near-duplicate transitions
    into "validation" and understate overfitting."""
    if len(episodes) < 2:
        return episodes, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * val_frac))) if val_frac > 0 else 0
    val_idx = set(order[:n_val].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]
    return train_eps, val_eps


def stack_eps(episodes: List[Episode]) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.concatenate([o for _, o, _ in episodes], axis=0)
    action = np.concatenate([a for _, _, a in episodes], axis=0)
    return obs, action


# ---------------------------------------------------------------------------
# Network -- reuses agents/ppo_agent.py's `_PolicyNet` or
# agents/sac_agent.py's `_ActorNet` verbatim, see module docstring.
# ---------------------------------------------------------------------------

def build_net(algo: str, action_size: int, hidden: int):
    if algo == "ppo":
        return _PolicyNet(action_size=action_size, hidden=hidden)
    if algo == "sac":
        return _ActorNet(action_size=action_size, hidden=hidden)
    raise ValueError(f"Unknown --algo {algo!r} (expected 'ppo' or 'sac')")


def net_forward(algo: str, net, params, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """(mean, log_std) regardless of algo. PPOAgent's `_PolicyNet` also
    returns a value head; it's discarded here -- BC has no reward signal to
    fit it against, so its params just stay at their random `net.init()`
    value and get trained normally once RL fine-tuning (train.py) starts,
    same as any other freshly-initialized critic."""
    if algo == "ppo":
        mean, log_std, _value = net.apply(params, obs)
        return mean, log_std
    return net.apply(params, obs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo-dir", default="demonstrations")
    parser.add_argument(
        "--algo", choices=["ppo", "sac"], default="sac",
        help="Which agents/ policy architecture to pretrain (default: sac -- PPOAgent's "
             "tanh-squashed-mean parameterization is prone to the std-collapse failure mode "
             "documented in agents/ppo_agent.py, which BC pretraining alone doesn't fix).",
    )
    parser.add_argument(
        "--hidden", type=int, default=None,
        help="Policy hidden width. Default: match the target agent's own default "
             "(64 for ppo, 256 for sac) so the checkpoint's param shapes line up with "
             "that agent's agent.init().",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Fraction of *episodes* (not steps) held out for validation.",
    )
    parser.add_argument(
        "--min-demo-len", type=int, default=0,
        help="Skip demo files shorter than this many steps. demonstrations/clean_demo.py "
             "already filters at collection time; this is a second safety net.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path", default=None,
        help="Default: bc/checkpoints/bc_<algo>_latest",
    )
    parser.add_argument(
        "--metrics-path", default=None,
        help="Default: bc/bc_<algo>_metrics.png",
    )
    parser.add_argument(
        "--no-env-check", action="store_true",
        help="Skip building ErhuEnv to cross-check obs/action sizes against the loaded demos.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint_path or f"bc/checkpoints/bc_{args.algo}_latest"
    ).resolve()
    metrics_path = args.metrics_path or f"bc/bc_{args.algo}_metrics.png"
    hidden = args.hidden or {"ppo": 64, "sac": 256}[args.algo]

    # 1. Load + split demos.
    episodes = load_episodes(Path(args.demo_dir), args.min_demo_len)
    train_eps, val_eps = split_episodes(episodes, args.val_frac, args.seed)
    total_steps = sum(len(o) for _, o, _ in episodes)
    print(
        f"Loaded {len(episodes)} episodes ({total_steps} steps total): "
        f"{len(train_eps)} train / {len(val_eps)} val episodes."
    )

    train_obs, train_action = stack_eps(train_eps)
    val_obs: Optional[np.ndarray]
    val_action: Optional[np.ndarray]
    val_obs, val_action = stack_eps(val_eps) if val_eps else (None, None)

    obs_size, action_size = train_obs.shape[-1], train_action.shape[-1]

    if not args.no_env_check:
        from envs.erhu_env import ErhuEnv

        env = ErhuEnv()
        assert env.observation_size == obs_size, (
            f"demo obs dim {obs_size} != ErhuEnv.observation_size {env.observation_size} -- "
            "demos were recorded against a different obs layout than the current ErhuEnv "
            "(pass --no-env-check to skip this check)."
        )
        assert env.action_size == action_size, (
            f"demo action dim {action_size} != ErhuEnv.action_size {env.action_size}."
        )

    # 2. Running obs-normalization stats, fit on the training split only (so
    # validation loss below isn't computed under stats that peeked at it) --
    # the same RunningNorm both RL agents carry through training, see
    # agents/obs_normalizer.py.
    obs_norm = init_running_norm(obs_size)
    obs_norm = update_running_norm(obs_norm, jp.asarray(train_obs))

    # 3. Build + init the target agent's own network.
    net = build_net(args.algo, action_size, hidden)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jp.zeros((obs_size,), dtype=jp.float32))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optimizer.init(params)

    def loss_fn(p, obs_batch, action_batch):
        mean, log_std = net_forward(args.algo, net, p, obs_batch)
        return jp.mean(squashed_gaussian_nll(action_batch, mean, log_std))

    @jax.jit
    def train_step(p, opt_state, obs_batch, action_batch):
        loss, grads = jax.value_and_grad(loss_fn)(p, obs_batch, action_batch)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    @jax.jit
    def eval_metrics(p, obs_batch, action_batch):
        mean, log_std = net_forward(args.algo, net, p, obs_batch)
        nll = jp.mean(squashed_gaussian_nll(action_batch, mean, log_std))
        mae = jp.mean(jp.abs(jp.tanh(mean) - action_batch))
        return nll, mae

    train_obs_n = normalize_obs(obs_norm, jp.asarray(train_obs))
    train_action_j = jp.asarray(train_action)
    val_obs_n = val_action_j = None
    if val_obs is not None:
        val_obs_n = normalize_obs(obs_norm, jp.asarray(val_obs))
        val_action_j = jp.asarray(val_action)

    n_train = train_obs_n.shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    perm_rng = np.random.default_rng(args.seed)
    logger = MetricsLogger(live=False)

    print(
        f"Training BC policy for --algo {args.algo} ({hidden}-wide, {n_train} train steps, "
        f"{val_obs_n.shape[0] if val_obs_n is not None else 0} val steps)..."
    )
    val_nll = val_mae = None
    for epoch in range(args.epochs):
        perm = perm_rng.permutation(n_train)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, train_obs_n[idx], train_action_j[idx]
            )
            epoch_loss += float(loss)
        epoch_loss /= steps_per_epoch

        metrics = {"train_nll": jp.asarray(epoch_loss)}
        if val_obs_n is not None:
            val_nll, val_mae = eval_metrics(params, val_obs_n, val_action_j)
            metrics["val_nll"] = val_nll
            metrics["val_action_mae"] = val_mae
        logger.log(epoch, metrics)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            msg = f"epoch {epoch:4d}  train_nll={epoch_loss:.4f}"
            if val_nll is not None:
                msg += f"  val_nll={float(val_nll):.4f}  val_action_mae={float(val_mae):.4f}"
            print(msg)

    logger.plot(metrics_path)
    logger.close()

    # 4. Save params, structured exactly like agent.init()'s
    # train_state["params"] / ["actor_params"] -- see train.py's
    # --bc-checkpoint warm start.
    print(f"Saving BC-pretrained {args.algo} policy to {checkpoint_path}...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    # force=True: re-running this script (e.g. after collecting more
    # demos) should overwrite the previous BC checkpoint rather than error
    # because the directory already exists.
    checkpointer.save(checkpoint_path, params, force=True)
    checkpointer.wait_until_finished()

    obs_norm_path = checkpoint_path.with_name(checkpoint_path.name + "_obs_norm.npz")
    print(f"Saving observation-normalization constants to {obs_norm_path}...")
    np.savez(
        obs_norm_path,
        mean=np.asarray(obs_norm.mean),
        var=np.asarray(obs_norm.var),
        count=np.asarray(obs_norm.count),
    )

    sac_or_ppo_config = "configs/train_sac.yaml" if args.algo == "sac" else "configs/train.yaml"
    print(
        "\nDone. Warm-start RL fine-tuning from this checkpoint with:\n"
        f"  python train.py --config {sac_or_ppo_config} --bc-checkpoint {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
