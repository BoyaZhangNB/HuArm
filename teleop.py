"""teleop.py -- UDP/TCP-driven teleoperation + imitation-learning demo capture.

Position and control now arrive over two separate sockets/transports, mirroring
the two structs on the operator (Swift) side:

    PositionPacket (UDP, unordered/lossy, one-shot datagrams):
        {"x": float, "y": float, "z": float, "pitch": float}

    ControlPacket (TCP, ordered/reliable, newline-delimited JSON stream):
        {"reset": bool, "collect": bool}

(x, y, z) is a desired end-effector displacement *relative to the EE
position at the moment `reset` last went true* -- not an absolute world
coordinate, since the operator has no reason to know the arm's base frame.
`pitch` is the only orientation DOF the iOS app exposes (yaw/roll were
dropped there): a delta, in radians, from the angle between the ground and
joint3_body's own link (see `PITCH_BODY`/`PITCH_BODY_LOCAL_DIR`) *at the
moment `reset` last fired* -- same relative-to-origin convention as (x, y,
z), just for that one angle instead of position. Each control step,
position and that single angle constraint are solved together to arm joint
angles via damped least-squares IK (reusing `envs.utils_envs.jacobian_ik`,
the same routine the env's own pose-pool/domain-randomization code uses),
converted into the env's normalized delta-ctrl action space, and applied through
`ErhuEnv.step` -- never by poking `data.ctrl` directly -- so the physics,
reward/termination bookkeeping, and observation vector stay bit-for-bit
identical to what the trained policy sees. That matters for imitation
learning: a BC/DAgger policy trained on these demos will be fed the exact
same `obs` layout (including the action/force history stacks) and must
reproduce actions in the exact same space.

Protocol semantics, chosen for demo collection:
  - `reset`: fires once per *arrival* of a control packet carrying
    reset=true. The operator's UI sends each control packet exactly once
    per button press/toggle -- it is not a level held down and re-sent --
    so a control packet is no longer streamed continuously and the old
    False->True value-transition edge detection (comparing against a
    remembered previous value) can't tell two consecutive reset taps
    apart: both packets carry reset=true, so a naive level comparison
    only fires on the first. Instead each new packet (tracked via a
    monotonic sequence number from TCPReceiver) is itself the discrete
    event; reset fires whenever a just-arrived packet has reset=true,
    regardless of what the previous packet said. Re-arms the (x, y, z)
    origin to the post-reset EE position.
  - `collect`: level-triggered recording toggle carried on the same
    one-shot packets. Because the level only changes value when the
    operator actually toggles it, comparing against the previous *value*
    (not packet arrival) still works: rising edge starts a new episode
    buffer, falling edge saves it to disk. If a `reset` arrives while
    still collecting, the in-progress episode is saved and a fresh one
    immediately started (reset marks an episode boundary, not
    necessarily the end of the demonstration session).
  - Env-internal termination (force limit, time limit, ...) also closes out
    the current episode and pauses stepping until the next `reset`, so a
    stale/garbage physics state is never fed back through IK or logged.

UDP is unordered/lossy by nature; only the most recently received position
packet is ever acted on (zero-order hold between arrivals), and malformed
datagrams are dropped with a warning rather than crashing the session. TCP
is ordered/reliable, but a stream can still deliver partial/merged JSON
objects across `recv()` calls; the control receiver buffers by newline
delimiter and drops only the malformed line, not the whole connection.
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
from envs.utils_envs import jacobian_ik, joint_to_actuator_id


# ---------------------------------------------------------------------------
# UDP receiver -- PositionPacket, background thread, always exposes the
# newest valid packet.
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
                    "pitch": float(pkt["pitch"]),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"[udp] dropping malformed position packet: {e}")
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
# TCP receiver -- ControlPacket, background thread, always exposes the
# newest valid packet. Accepts a single client at a time and reconnects
# transparently if the operator's app drops/re-establishes the stream.
# ---------------------------------------------------------------------------

class TCPReceiver(threading.Thread):
    def __init__(self, host: str, port: int, buf_size: int = 4096):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.settimeout(0.5)  # let run() notice stop() without blocking forever
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0  # bumped on every parsed packet; lets consumers tell
                        # "a new one-shot packet arrived" apart from "the
                        # same last packet is still the latest" -- reset is
                        # a discrete per-arrival event, not a value level.
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                conn, _addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed from stop()
            conn.settimeout(0.5)
            self._serve(conn)

    def _serve(self, conn):
        """Read one client connection until it closes/errors, parsing
        newline-delimited JSON objects out of the (possibly chunked/merged)
        TCP byte stream."""
        buf = b""
        with conn:
            while not self._stop_evt.is_set():
                try:
                    chunk = conn.recv(self._buf_size)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break  # client disconnected
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._parse_and_store(line)

    def _parse_and_store(self, line: bytes):
        try:
            pkt = json.loads(line.decode("utf-8"))
            parsed = {
                "reset": bool(pkt["reset"]),
                "collect": bool(pkt["collect"]),
            }
            print("Received control packet:", parsed)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"[tcp] dropping malformed control packet: {e}")
            return
        with self._lock:
            self._seq += 1
            parsed["seq"] = self._seq
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

ARM_JOINT_NAMES = ("joint1", "joint2", "joint5", "joint3", "joint4")

# joint3_body's own geom runs from its origin to local (-0.30, 0, 0) (see
# arm.xml) -- i.e. local -X is that link's long axis, the one PITCH_BODY's
# angle-to-ground constraint below tracks.
PITCH_BODY = "joint3_body"
PITCH_BODY_LOCAL_DIR = np.array([-1.0, 0.0, 0.0])


def bow_pitch_angle(model, data) -> float:
    """Current angle (radians) between PITCH_BODY_LOCAL_DIR (rotated to
    world) and the ground plane -- the same quantity `jacobian_ik`'s angle
    constraint tracks, computed directly off live sim data so the teleop
    loop can re-zero its delta origin on every reset."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PITCH_BODY)
    xmat = np.array(data.xmat[bid]).reshape(3, 3)
    u = xmat @ PITCH_BODY_LOCAL_DIR
    return float(np.arcsin(np.clip(u[2], -1.0, 1.0)))


def solve_ik(model, ik_data, current_qpos, target_world_pos, prev_ctrl, target_pitch=None):
    """Damped-least-squares IK for the arm's 5 joints, warm-started from
    `current_qpos` (the arm's current live pose) so each call only has to
    correct a small per-step delta -- fast enough for a real-time control
    loop.

    `target_pitch` (radians), when given, is solved for simultaneously with
    position as one extra scalar constraint: the angle between
    `PITCH_BODY`'s link (joint3_body's own geom) and the ground plane (see
    `jacobian_ik`'s angle_body/target_angle args) -- position accuracy is not
    traded away for it since both error terms are driven to zero by the same
    least-squares solve, just weighted against each other.

    Returns a full ctrl-shaped vector of target joint angles, scattered by
    actuator id (so it does not depend on ARM_JOINT_NAMES happening to be in
    the same order as the model's actuators); any actuator not driven by one
    of ARM_JOINT_NAMES keeps its previous target from `prev_ctrl`."""
    ik_data.qpos[:] = current_qpos
    body_points = [("end_effector", np.zeros(3), 1.0)]
    jacobian_ik(
        model, ik_data, body_points, target_world_pos, list(ARM_JOINT_NAMES),
        max_iters=20, damping=1e-2, step_clip=0.05, tol=1e-4,
        angle_body=PITCH_BODY if target_pitch is not None else None,
        angle_local_dir=PITCH_BODY_LOCAL_DIR,
        target_angle=target_pitch,
    )
    target_ctrl = np.array(prev_ctrl, dtype=np.float64)
    for jn in ARM_JOINT_NAMES:
        aid = joint_to_actuator_id(model, jn)
        if aid < 0:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        target_ctrl[aid] = ik_data.qpos[model.jnt_qposadr[jid]]
    return target_ctrl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="bind address for both sockets")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP bind port (PositionPacket)")
    parser.add_argument("--tcp-port", type=int, default=5006, help="TCP bind port (ControlPacket)")
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

    env = ErhuEnv(episode_time_limit=args.episode_time_limit, dr_pool_size=10)
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
    pitch_origin = bow_pitch_angle(model, data)
    awaiting_reset = False  # true once an episode has ended, until the next reset edge

    position_receiver = UDPReceiver(args.host, args.udp_port)
    position_receiver.start()
    control_receiver = TCPReceiver(args.host, args.tcp_port)
    control_receiver.start()
    recorder = DemoRecorder(Path(args.demo_dir))

    prev_collect = False
    last_ctrl_seq = None  # seq of the last control packet already acted on

    print(f"Listening for UDP position packets on {args.host}:{args.udp_port}")
    print(f"Listening for TCP control packets on {args.host}:{args.tcp_port}")
    print(f"Demo episodes will be saved under {Path(args.demo_dir).resolve()}")
    print("Teleoperation loop running. Press ESC in viewer to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        try:
            while viewer.is_running():
                loop_start = time.time()
                pkt = position_receiver.get_latest()
                ctrl_pkt = control_receiver.get_latest()

                if pkt is not None:
                    # A control packet is now sent once per button
                    # press/toggle, not re-streamed while held -- so a
                    # just-arrived packet (new seq) is itself the discrete
                    # event, distinct from "the last packet is still the
                    # latest". `reset` rides on that: two consecutive
                    # reset taps both carry reset=true, so comparing
                    # values alone (the old prev_reset edge check) would
                    # only catch the first.
                    new_ctrl_pkt = ctrl_pkt is not None and ctrl_pkt["seq"] != last_ctrl_seq
                    if new_ctrl_pkt:
                        last_ctrl_seq = ctrl_pkt["seq"]
                    reset = ctrl_pkt["reset"] if ctrl_pkt is not None else False
                    collect = ctrl_pkt["collect"] if ctrl_pkt is not None else False
                    do_reset = new_ctrl_pkt and reset
                    # `collect` is a level, not a one-shot action, and only
                    # changes value when the operator actually toggles it
                    # -- so a plain value-transition edge still works here.
                    rising_collect = collect and not prev_collect
                    falling_collect = (not collect) and prev_collect
                    prev_collect = collect

                    if do_reset:
                        rng, sub = jax.random.split(rng)
                        state = _reset(sub)
                        mjx.get_data_into(data, model, state.pipeline_state)
                        ee_origin = np.array(data.xpos[ee_body_id])
                        pitch_origin = bow_pitch_angle(model, data)
                        awaiting_reset = False
                        print(f"\n[reset] new episode, EE origin = {ee_origin}, "
                              f"pitch origin = {pitch_origin:.4f} rad")
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
                            [pkt["x"], -pkt["z"], pkt["y"]], -args.max_offset, args.max_offset
                        )
                        target = ee_origin + offset

                        # Operator pitch is a delta from PITCH_BODY's angle
                        # at the last reset (pitch_origin), same
                        # relative-to-origin convention as the position
                        # offset above -- so the absolute target handed to
                        # the IK's angle constraint is the two summed.
                        target_pitch = pitch_origin + pkt["pitch"]

                        prev_ctrl = np.array(state.pipeline_state.ctrl)
                        current_qpos = np.array(state.pipeline_state.qpos)
                        target_ctrl = np.clip(
                            solve_ik(model, ik_data, current_qpos, target, prev_ctrl, target_pitch),
                            ctrl_lo, ctrl_hi,
                        )

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
                    # No position packet has ever arrived yet -- hold at the
                    # initial reset pose and keep the viewer alive/responsive.
                    viewer.sync()

                elapsed = time.time() - loop_start
                sleep_time = env.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting.")
        finally:
            recorder.stop_and_save()
            position_receiver.stop()
            control_receiver.stop()


if __name__ == "__main__":
    main()
