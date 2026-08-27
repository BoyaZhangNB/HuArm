"""test_demonstration.py -- play back a recorded teleop demonstration.

Loads a `.npz` saved by `teleop.py`'s `DemoRecorder` and replays it through
`ErhuEnv.step` one recorded action at a time, in the same domain-randomized
model (friction/mass/damping/erhu placement -- `dr_*` keys) and starting
physics state (`init_*` keys) the demo was originally recorded under. This
is a physics replay, not a scripted animation: the same `env.step` the
policy/BC pipeline sees is exercised, just fed the expert's logged actions
instead of a network's.

Rebuilding the starting `State` never reaches into `ErhuEnv` internals --
only public fields/methods are used (`env.reset`, `env.mjx_model.replace`,
`State.replace`, `mjx.Data.replace`, `mjx.forward`).
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from envs.erhu_env import ErhuEnv
from envs import utils_dr


def latest_demo(demo_dir: Path) -> Path:
    files = sorted(demo_dir.glob("demo_*.npz"))
    if not files:
        raise FileNotFoundError(f"No demo_*.npz files found under {demo_dir}")
    return files[-1]


def load_demo(path: Path):
    """Splits a saved demo's npz into (dr_params, erhu_drift, init_state,
    actions), undoing the `dr_`/`drift_`/`init_` key prefixes
    `DemoRecorder.stop_and_save` adds -- see teleop.py."""
    npz = np.load(path)

    def by_prefix(prefix):
        return {k[len(prefix):]: jp.asarray(v) for k, v in npz.items() if k.startswith(prefix)}

    dr_params, init_state = by_prefix("dr_"), by_prefix("init_")
    if not dr_params or not init_state:
        raise ValueError(
            f"{path} has no dr_*/init_* metadata -- it predates demo playback support "
            "and can't be replayed as the exact recorded model/state."
        )
    # drift_* is younger than dr_*/init_*; a demo recorded before the erhu
    # drift existed was recorded against a stationary erhu, so replay it
    # that way rather than against a freshly drawn drift.
    drift = by_prefix("drift_") or utils_dr.no_erhu_drift()
    return dr_params, drift, init_state, npz["action"]


def build_playback_state(env: ErhuEnv, dr_params, erhu_drift, init_state):
    """Rebuilds the exact State a recorded demo started from: this
    episode's domain-randomized model (`dr_params` plus the erhu's drift
    over the episode) and the live physics state recording began at
    (`init_state`, including the solver warm start). `env.reset()` is only
    called to get a template State
    (for its info/obs pytree shapes) -- its own randomly-drawn
    dr_params/drift/pose are entirely overwritten below."""
    template = env.reset(jax.random.PRNGKey(0))
    info = dict(template.info)
    info["dr_params"] = dr_params
    info["erhu_drift"] = erhu_drift
    data = template.pipeline_state.replace(**init_state)
    data = mjx.forward(env.effective_model(dr_params, erhu_drift, data.time), data)
    return template.replace(pipeline_state=data, info=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", nargs="?", default=None,
                         help="path to a demo_*.npz (default: newest file in --demo-dir)")
    parser.add_argument("--demo-dir", default="demonstrations")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="replay on a loop until the viewer is closed")
    args = parser.parse_args()

    demo_path = Path(args.demo) if args.demo else latest_demo(Path(args.demo_dir))
    print(f"Loading demonstration: {demo_path}")
    dr_params, erhu_drift, init_state, actions = load_demo(demo_path)
    print(f"  {len(actions)} steps, dr_params={sorted(dr_params)}, init_state={sorted(init_state)}")

    print(f"Using MuJoCo Version: {mujoco.__version__}")
    # dr_pool_size=1 -- the pose pool is only used by env.reset()'s throwaway
    # template draw in build_playback_state(), never for the actual replayed
    # episode, so there's no reason to pay for the default 1024-entry build.
    env = ErhuEnv(episode_time_limit=1e9, dr_pool_size=1)
    assert actions.shape[-1] == env.action_size, (
        f"demo action dim {actions.shape[-1]} != ErhuEnv.action_size {env.action_size} "
        f"-- this demo was recorded before the bow_frog_hinge stiffness action dim "
        f"(action[5]) was added and can no longer be replayed against the current env."
    )
    _step = jax.jit(env.step)

    state = build_playback_state(env, dr_params, erhu_drift, init_state)
    model = env.mj_model
    data = mjx.get_data(model, state.pipeline_state)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        print("Replaying recorded demonstration. Press ESC in viewer to exit.")
        try:
            while viewer.is_running():
                wall_start = time.time()
                for i, action in enumerate(actions):
                    if not viewer.is_running():
                        break
                    state = _step(state, jp.asarray(action, dtype=jp.float32))
                    mjx.get_data_into(data, model, state.pipeline_state)
                    viewer.sync()
                    env_time = env.dt / args.speed * (i + 1)
                    elapsed = time.time() - wall_start
                    print(f"Env time: {env_time:.3f}s, elapsed wall time: {elapsed:.3f}s", end="\r")
                    while env_time > elapsed:
                        elapsed = time.time() - wall_start
                        time.sleep(0.001)
                        
                print(f"\n[playback] finished {len(actions)} steps "
                      f"in {time.time() - wall_start:.2f}s (wall)")
                if not args.loop:
                    break
                state = build_playback_state(env, dr_params, erhu_drift, init_state)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")


if __name__ == "__main__":
    main()
