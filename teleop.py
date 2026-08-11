"""teleop.py -- UDP-driven teleoperation + imitation-learning demo capture.

Receives UDP datagrams from a device on the local network, each a JSON
object:

    {"x": float, "y": float, "z": float, "reset": bool, "collect": bool}

(x, y, z) is a desired end-effector displacement *relative to the EE
position at the moment `reset` last went true* -- not an absolute world
coordinate, since the operator has no reason to know the arm's base frame.
Each control step, that target is solved to arm joint angles via damped
least-squares IK (reusing `envs.utils_envs.jacobian_ik`, the same routine
the env's own pose-pool/domain-randomization code uses), converted into the
env's normalized delta-ctrl action space, and applied through
`ErhuEnv.step` -- never by poking `data.ctrl` directly -- so the physics,
reward/termination bookkeeping, and observation vector stay bit-for-bit
identical to what the trained policy sees. That matters for imitation
learning: a BC/DAgger policy trained on these demos will be fed the exact
same `obs` layout (including the action/force history stacks) and must
reproduce actions in the exact same space.

Protocol semantics, chosen for demo collection:
  - `reset`: edge-triggered (False->True). Re-running ErhuEnv.reset() every
    packet would thrash domain randomization and truncate episodes; a
    level signal held down by the operator's UI should only reset once.
    Re-arms the (x, y, z) origin to the post-reset EE position.
  - `collect`: level-triggered recording toggle. Rising edge starts a new
    episode buffer; falling edge saves it to disk. If a `reset` edge
    arrives while still collecting, the in-progress episode is saved and a
    fresh one immediately started (reset marks an episode boundary, not
    necessarily the end of the demonstration session).
  - Env-internal termination (force limit, time limit, ...) also closes out
    the current episode and pauses stepping until the next `reset`, so a
    stale/garbage physics state is never fed back through IK or logged.

UDP is unordered/lossy by nature; only the most recently received packet is
ever acted on (zero-order hold between arrivals), and malformed datagrams
are dropped with a warning rather than crashing the session.
"""

import argparse
import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

from envs.erhu_env import ErhuEnv
from envs.utils_envs import jacobian_ik


# ---------------------------------------------------------------------------
# UDP receiver -- background thread, always exposes the newest valid packet.
# ---------------------------------------------------------------------------

class UDPReceiver(threading.Thread):
    def __init__(self, host: str, port: int, buf_size: int = 2048):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)  # let run() notice stop() without blocking forever
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest = None
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                raw, _addr = self.sock.recvfrom(self._buf_size)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed from stop()
            try:
                pkt = json.loads(raw.decode("utf-8"))
                parsed = {
                    "x": float(pkt["x"]),
                    "y": float(pkt["y"]),
                    "z": float(pkt["z"]),
                    "reset": bool(pkt["reset"]),
                    "collect": bool(pkt["collect"]),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"[udp] dropping malformed packet: {e}")
                continue
            with self._lock:
                self._latest = parsed

    def get_latest(self):
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def stop(self):
        self._stop_evt.set()
        self.sock.close()


# ---------------------------------------------------------------------------
# Demo recorder -- one .npz per collected episode.
# ---------------------------------------------------------------------------

class DemoRecorder:
    """Buffers (obs, action, ...) for imitation learning and flushes each
    episode to its own compressed .npz on `stop_and_save`.

    `obs`/`action` are logged exactly as passed to/returned by `ErhuEnv`, so
    a saved demo can be replayed through the same policy interface used at
    train/inference time (see inference.py's `agent.act(obs, ...)` call).
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.active = False
        self._buf = None

    def start(self):
        self._buf = {
            "obs": [], "action": [], "reward": [], "done": [],
            "ee_target": [], "sim_time": [], "wall_time": [],
        }
        self.active = True
        print("[collect] recording started")

    def log(self, obs, action, reward, done, ee_target, sim_time):
        if not self.active:
            return
        b = self._buf
        b["obs"].append(np.asarray(obs, dtype=np.float32))
        b["action"].append(np.asarray(action, dtype=np.float32))
        b["reward"].append(float(reward))
        b["done"].append(bool(done))
        b["ee_target"].append(np.asarray(ee_target, dtype=np.float32))
        b["sim_time"].append(float(sim_time))
        b["wall_time"].append(time.time())

    def stop_and_save(self):
        if not self.active:
            return
        self.active = False
        buf, self._buf = self._buf, None
        n = len(buf["obs"])
        if n == 0:
            print("[collect] stopped -- 0 steps, nothing saved")
            return
        fname = self.out_dir / f"demo_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
        np.savez_compressed(
            fname,
            obs=np.stack(buf["obs"]),
            action=np.stack(buf["action"]),
            reward=np.asarray(buf["reward"], dtype=np.float32),
            done=np.asarray(buf["done"], dtype=bool),
            ee_target=np.stack(buf["ee_target"]),
            sim_time=np.asarray(buf["sim_time"], dtype=np.float32),
            wall_time=np.asarray(buf["wall_time"], dtype=np.float64),
        )
        print(f"[collect] stopped -- saved {n} steps to {fname}")


# ---------------------------------------------------------------------------

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")


def solve_ik(model, ik_data, current_qpos, target_world_pos):
    """Damped-least-squares IK for the arm's 4 joints, warm-started from
    `current_qpos` (the arm's current live pose) so each call only has to
    correct a small per-step delta -- fast enough for a real-time control
    loop. Returns clipped target joint angles (one per arm actuator, same
    order as ARM_JOINT_NAMES)."""
    ik_data.qpos[:] = current_qpos
    body_points = [("end_effector", np.zeros(3), 1.0)]
    jacobian_ik(
        model, ik_data, body_points, target_world_pos, list(ARM_JOINT_NAMES),
        max_iters=20, damping=1e-2, step_clip=0.05, tol=1e-4,
    )
    qpos_idxs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                 for jn in ARM_JOINT_NAMES]
    return np.array([ik_data.qpos[i] for i in qpos_idxs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=5005, help="UDP bind port")
    parser.add_argument("--demo-dir", default="demonstrations", help="output dir for collected episodes")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed, reseeded (split) on every reset")
    parser.add_argument(
        "--max-offset", type=float, default=0.4,
        help="clip |x|,|y|,|z| (meters) from the operator before it becomes an IK target -- "
             "guards against a bad/garbled UDP reading commanding an unreachable pose",
    )
    parser.add_argument("--episode-time-limit", type=float, default=1000.0)
    args = parser.parse_args()

    print(f"Using MuJoCo Version: {mujoco.__version__}")

    env = ErhuEnv(episode_time_limit=args.episode_time_limit)
    _reset = jax.jit(env.reset)
    _step = jax.jit(env.step)

    model = env.mj_model
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    ctrl_lo = np.array(model.actuator_ctrlrange[:, 0])
    ctrl_hi = np.array(model.actuator_ctrlrange[:, 1])
    ik_data = mujoco.MjData(model)  # scratch data for jacobian_ik; never touched by the viewer

    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    state = _reset(sub)
    data = mjx.get_data(model, state.pipeline_state)

    ee_origin = np.array(data.xpos[ee_body_id])
    awaiting_reset = False  # true once an episode has ended, until the next reset edge

    receiver = UDPReceiver(args.host, args.port)
    receiver.start()
    recorder = DemoRecorder(Path(args.demo_dir))

    prev_reset, prev_collect = False, False

    print(f"Listening for UDP teleop packets on {args.host}:{args.port}")
    print(f"Demo episodes will be saved under {Path(args.demo_dir).resolve()}")
    print("Teleoperation loop running. Press ESC in viewer to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        try:
            while viewer.is_running():
                loop_start = time.time()
                pkt = receiver.get_latest()

                if pkt is not None:
                    rising_reset = pkt["reset"] and not prev_reset
                    prev_reset = pkt["reset"]
                    rising_collect = pkt["collect"] and not prev_collect
                    falling_collect = (not pkt["collect"]) and prev_collect
                    prev_collect = pkt["collect"]

                    if rising_reset:
                        rng, sub = jax.random.split(rng)
                        state = _reset(sub)
                        mjx.get_data_into(data, model, state.pipeline_state)
                        ee_origin = np.array(data.xpos[ee_body_id])
                        awaiting_reset = False
                        print(f"\n[reset] new episode, EE origin = {ee_origin}")
                        if recorder.active:
                            # `collect` is still held -- treat reset as an
                            # episode boundary, not the end of the session.
                            recorder.stop_and_save()
                            recorder.start()

                    if rising_collect:
                        recorder.start()
                    if falling_collect:
                        recorder.stop_and_save()

                    if not awaiting_reset:
                        offset = np.clip(
                            [-pkt["x"], pkt["z"], pkt["y"]], -args.max_offset, args.max_offset
                        )
                        target = ee_origin + offset

                        prev_ctrl = np.array(state.pipeline_state.ctrl)
                        current_qpos = np.array(state.pipeline_state.qpos)
                        target_qpos = solve_ik(model, ik_data, current_qpos, target)
                        target_ctrl = np.clip(target_qpos, ctrl_lo, ctrl_hi)

                        action = np.clip(
                            (target_ctrl - prev_ctrl) / env.max_ctrl_delta, -1.0, 1.0
                        )
                        state = _step(state, jp.asarray(action, dtype=jp.float32))
                        mjx.get_data_into(data, model, state.pipeline_state)
                        viewer.sync()

                        recorder.log(
                            obs=state.obs, action=action, reward=state.reward,
                            done=state.done, ee_target=target, sim_time=data.time,
                        )

                        if bool(state.done):
                            print(f"\n[episode] terminated at sim time {data.time:.3f}")
                            if recorder.active:
                                recorder.stop_and_save()
                            awaiting_reset = True
                else:
                    # No packet has ever arrived yet -- hold at the initial
                    # reset pose and keep the viewer alive/responsive.
                    viewer.sync()

                elapsed = time.time() - loop_start
                sleep_time = env.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")
        finally:
            recorder.stop_and_save()
            receiver.stop()


if __name__ == "__main__":
    main()
