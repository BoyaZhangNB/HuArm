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

The bow target the policy chases is set by hand, live, from the operator's
phone -- the same streaming setup teleop.py uses for position, one JSON
object per UDP datagram:

    CommandPacket (UDP, unordered/lossy, one-shot datagrams):
        {"velocity": double, "pressure": double}

`velocity` is the desired signed lateral bow speed in m/s (sign = stroke
direction along the erhu's own left/right axis, see
`ErhuEnv._lateral_axis_local`); `pressure` is the desired bow-hair/A-string
contact force in N. During training both come from `envs.utils_traj`'s
scripted quartic reference stroke, baked into two fixed slots of the
observation vector -- the policy only ever sees those two numbers, so
overwriting the slots with what the operator most recently sent (see
teleop.py's `patch_desired_velocity_pressure`, reused here) is enough to
hand the target over to a human, no retraining or env change involved.
Values are held between arrivals and clipped into the range the policy was
trained on (`--no-clip-command` to send raw values); until the first packet
arrives the policy is fed `--init-velocity`/`--init-pressure`.

Note the env's own `state.metrics` velocity_error/pressure_error stay
scored against its internal scripted trajectory, which the operator is not
following -- errors against the *commanded* target are logged separately
under "command/".

Audio: same live erhu synthesis as teleop.py (see synthesis.py) -- the
policy's bow-hair/string contact force, lateral bow speed and hair position
drive `ErhuSynth` straight off `state`, so the policy's stroke is heard as
well as watched. A read-only consumer of `state`; `--no-audio` (or missing
`dawdreamer`/`sounddevice`/an output device) just runs silently.

Usage:
    python inference.py                                              # ppo, checkpoints/model_latest
    python inference.py --algo sac                                   # sac, checkpoints/sac_model_latest
    python inference.py --algo sac --checkpoint bc/checkpoints/bc_sac_latest
    python inference.py --algo sac --checkpoint checkpoints/sac_model_latest --stochastic
    python inference.py --xml huarm/arm_rigid.xml                        # override ErhuEnv's model XML
    python inference.py --command-port 5007                           # where the phone streams commands
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
# The UDP receiver and the obs desired_velocity/desired_pressure patching
# helpers live in teleop.py, where they were written for the operator's
# phone stream and for BC demo logging; both apply verbatim here, so this
# reuses them rather than restating the packet plumbing and the (fragile,
# layout-derived) obs offset in a second place.
from teleop import (
    UDPReceiver,
    desired_velocity_obs_idx,
    make_synth,
    patch_desired_velocity_pressure,
)
from utils import MetricsLogger, print_jp_dict

# Same algo -> (agent class, train_state param key) mapping train.py uses.
AGENT_CLASSES = {"ppo": PPOAgent, "sac": SACAgent}
PARAM_KEYS = {"ppo": "params", "sac": "actor_params"}
DEFAULT_CHECKPOINTS = {
    "ppo": "checkpoints/model_latest",
    "sac": "checkpoints/sac_model_latest",
}


class CommandReceiver(UDPReceiver):
    """teleop.py's UDP receiver, reading CommandPackets instead of
    PositionPackets -- {"velocity": double, "pressure": double}, the desired
    bow velocity (m/s, signed) and pressure (N) streamed from the operator's
    phone. Both fields are required; a datagram missing either is dropped as
    malformed rather than silently half-applied."""

    @staticmethod
    def parse(pkt: dict) -> dict:
        velocity = float(pkt["velocity"])
        pressure = float(pkt["pressure"])
        if not (np.isfinite(velocity) and np.isfinite(pressure)):
            raise ValueError(f"non-finite command: velocity={velocity}, pressure={pressure}")
        return {"velocity": velocity, "pressure": pressure}


def clip_command(env: ErhuEnv, velocity: float, pressure: float) -> tuple[float, float]:
    """Clamp an operator command into the range the scripted reference
    stroke spanned during training (see ErhuEnv's traj_* args): |velocity|
    <= traj_v_limit, pressure in [0, traj_p_max]. Values outside it are
    off-distribution for the policy -- it never saw such a target, so its
    response is unconstrained rather than merely aggressive."""
    return (
        float(np.clip(velocity, -env.traj_v_limit, env.traj_v_limit)),
        float(np.clip(pressure, 0.0, env.traj_p_max)),
    )


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
    parser.add_argument("--command-host", default="0.0.0.0", help="Bind address for the command socket.")
    parser.add_argument(
        "--command-port", type=int, default=5007,
        help="UDP port the operator streams CommandPackets to (default: 5007; teleop.py's "
             "5005/5006 are left free so a teleop session can run alongside).",
    )
    parser.add_argument(
        "--init-velocity", type=float, default=0.0,
        help="Desired bow velocity (m/s) fed to the policy until the first CommandPacket arrives.",
    )
    parser.add_argument(
        "--init-pressure", type=float, default=0.0,
        help="Desired bow pressure (N) fed to the policy until the first CommandPacket arrives.",
    )
    parser.add_argument(
        "--no-clip-command", dest="clip_command", action="store_false",
        help="Feed operator commands to the policy raw, instead of clipping them into the "
             "range the scripted reference stroke covered during training (see clip_command).",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="do not synthesize the bow stroke (see synthesis.py)",
    )
    parser.add_argument(
        "--audio-device", default=None,
        help="sounddevice output device name or index; default is the system default",
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
    # Where the hair sits along its own length (-1 frog, +1 tip) -- the synth's
    # bow_x, not in state.metrics -- see teleop.py.
    _bow_x = jax.jit(env._bow_stroke_position)

    metrics_logger = MetricsLogger(live=True)
    deterministic = not args.stochastic
    synth = make_synth(not args.no_audio, args.audio_device)

    # Operator-commanded bow target, held between packet arrivals -- see the
    # module docstring. Seeded from the --init-* args so the policy has a
    # well-defined target before the phone has sent anything.
    desired_vp_idx = desired_velocity_obs_idx(model)
    command_velocity, command_pressure = args.init_velocity, args.init_pressure
    if args.clip_command:
        command_velocity, command_pressure = clip_command(env, command_velocity, command_pressure)
    command_receiver = CommandReceiver(args.command_host, args.command_port, label="command")
    command_receiver.start()
    print(f"Listening for UDP command packets on {args.command_host}:{args.command_port} "
          f"(velocity, pressure); holding "
          f"v={command_velocity:.4f} m/s, p={command_pressure:.3f} N until the first arrives.")

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
                print(f"Sim time {data.time:.3f}, elapsed real time {elapsed_real:.3f}, "
                      f"cmd v={command_velocity:+.4f} m/s p={command_pressure:.3f} N", end="\r")

                if data.time >= elapsed_real:
                    time.sleep(0.01)
                    continue

                # Zero-order hold on the operator's stream: only the newest
                # datagram is ever acted on, and the last one stays in force
                # until another lands (or forever, if the phone goes quiet).
                cmd = command_receiver.get_latest()
                if cmd is not None:
                    command_velocity, command_pressure = cmd["velocity"], cmd["pressure"]
                    if args.clip_command:
                        command_velocity, command_pressure = clip_command(
                            env, command_velocity, command_pressure
                        )

                # Swap the env's scripted desired velocity/pressure for the
                # operator's, so the policy chases the hand-set target.
                obs = patch_desired_velocity_pressure(
                    state.obs, desired_vp_idx, command_velocity, command_pressure
                )
                rng, act_rng = jax.random.split(rng)
                action, _ = agent.act(train_state, obs, act_rng, deterministic=deterministic)
                if args.verbose:
                    print(f"Action: {action}")

                state = _step(state, action)
                # Play this step's bow stroke -- read-only, same filtered
                # force/velocity the metrics below log (see synthesis.py).
                synth.update_from_state(state, bow_x=float(_bow_x(state.pipeline_state)))
                if state.done:
                    synth.update(0.0, 0.0, 0.0)
                    print("\nEpisode terminated")
                    break
                mjx.get_data_into(data, model, state.pipeline_state)
                viewer.sync()

                if data.time >= next_log:
                    next_log = data.time + args.log_interval
                    # state.metrics' own velocity_error/pressure_error are
                    # scored against the env's scripted reference stroke,
                    # which the operator is not following -- log the errors
                    # against what was actually commanded alongside them.
                    metrics = dict(state.metrics)
                    metrics["command"] = {
                        "desired_velocity": jp.asarray(command_velocity),
                        "desired_pressure": jp.asarray(command_pressure),
                        "velocity_error": state.metrics["bow_vel_ema"] - command_velocity,
                        "pressure_error": state.metrics["bow_a_force_ema"] - command_pressure,
                    }
                    print_jp_dict(metrics)
                    metrics_logger.log(data.time, metrics)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")
        finally:
            command_receiver.stop()
            synth.stop()

        metrics_logger.plot(args.metrics_path)
        metrics_logger.close()


if __name__ == "__main__":
    main()
