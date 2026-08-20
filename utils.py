import math
import multiprocessing as mp
import time
from queue import Empty

from envs.wrappers import AutoResetWrapper, EpisodeWrapper, VmapWrapper

import jax.numpy as jp
import mujoco
import numpy as np
import matplotlib
# Force a non-interactive backend, but only in the main process: mjpython
# runs this module's code off the real OS main thread (it reserves that
# thread for the MuJoCo viewer), and matplotlib's default macOS backend
# creates a Cocoa NSWindow even for savefig(), which crashes with "NSWindow
# should only be instantiated on the main thread!" outside the main thread.
# The spawned LiveMetricsPlotter subprocess re-imports this module (to
# unpickle _live_plot_worker) before running its own interactive plotting
# loop on ITS genuine main thread, so it must be left free to pick the
# normal interactive backend -- skip the override there.
if mp.current_process().name == "MainProcess":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def is_scalar_float(v):
    """True iff v is a 0-dimensional jp.ndarray holding a float value."""
    return (
        isinstance(v, jp.ndarray)
        and v.ndim == 0
        and v.dtype in (jp.float32, jp.float64)
    )


def _drop_nan(xs, ys):
    """Drop (x, y) pairs where y is NaN. Used before plotting metrics that
    are only logged intermittently (e.g. eval_reward): history/steps store
    a NaN placeholder for skipped iterations, and matplotlib breaks the line
    at each NaN, so an isolated real value between two NaNs never gets
    drawn. Filtering them out first lets the real values connect."""
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pairs:
        return [], []
    xs_clean, ys_clean = zip(*pairs)
    return list(xs_clean), list(ys_clean)


def flatten_scalar_dict(d, prefix=""):
    """Recursively walk a (possibly nested) dict of metrics and return a
    flat {key: value} dict containing only the zero-dimensional float
    entries, with nested keys joined by "/" (e.g. eval/reward_terms/velocity).
    Lets callers log dicts like {"reward_terms": {"velocity": ..., ...}}
    without every producer having to flatten them itself.
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_scalar_dict(v, prefix=f"{key}/"))
        elif is_scalar_float(v):
            out[key] = v
    return out


def print_jp_dict(d):
    """Print only the zero-dimensional float entries of d (nested dicts of
    scalars, e.g. reward_terms, are flattened first)."""
    for k, v in flatten_scalar_dict(d).items():
        print(f"{k}: {float(v):.6f}")


def _live_plot_worker(queue, poll_interval=0.05):
    """Runs in its own subprocess so its matplotlib window can use a real
    interactive (Cocoa-backed) backend on that process's genuine main
    thread -- this must stay out of the mjpython process, which reserves
    its true main thread for the MuJoCo viewer.
    """
    import matplotlib.pyplot as plt  # fresh import in the child process

    steps = []
    history = {}  # key -> list of float values (nan where missing)
    fig = None
    axes = {}
    lines = {}

    plt.ion()

    while True:
        got_update = False
        try:
            while True:
                item = queue.get_nowait()
                if item is None:
                    plt.close("all")
                    return
                step, metrics = item
                steps.append(step)
                for k in metrics:
                    if k not in history:
                        history[k] = [float("nan")] * (len(steps) - 1)
                for k, hist in history.items():
                    hist.append(metrics.get(k, float("nan")))
                got_update = True
        except Empty:
            pass

        if got_update:
            keys = sorted(history.keys())
            if fig is None or set(keys) != set(axes.keys()):
                if fig is not None:
                    plt.close(fig)
                ncols = 3
                nrows = math.ceil(len(keys) / ncols)
                fig, axs = plt.subplots(
                    nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
                )
                fig.canvas.manager.set_window_title("Live Metrics")
                axes, lines = {}, {}
                for ax, k in zip(axs.flat, keys):
                    (line,) = ax.plot([], [])
                    ax.set_ylabel(k)
                    ax.set_xlabel("time")
                    ax.grid(True, alpha=0.3)
                    axes[k] = ax
                    lines[k] = line
                for ax in axs.flat[len(keys):]:
                    ax.axis("off")
                fig.tight_layout()

            for k, line in lines.items():
                xs, ys = _drop_nan(steps, history[k])
                line.set_data(xs, ys)
                axes[k].relim()
                axes[k].autoscale_view()
            fig.tight_layout()

        if fig is not None:
            # plt.pause() both redraws and pumps GUI events; it's also what
            # keeps this loop from busy-spinning when there's no update.
            plt.pause(poll_interval)
        else:
            time.sleep(poll_interval)


class LiveMetricsPlotter:
    """Streams scalar metrics to a live-updating matplotlib window running
    in a separate subprocess (required under mjpython -- see
    _live_plot_worker). Use alongside or instead of MetricsLogger.plot().
    """

    def __init__(self):
        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue()
        self.process = ctx.Process(
            target=_live_plot_worker, args=(self.queue,), daemon=True
        )
        self.process.start()
        self._closed = False

    def log(self, step, d):
        if self._closed:
            return
        scalar_items = {k: float(v) for k, v in flatten_scalar_dict(d).items()}
        if scalar_items:
            self.queue.put((float(step), scalar_items))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.queue.put(None)
        except Exception:
            pass


class MetricsLogger:
    """Accumulates scalar float metrics over training iterations and plots them.

    Pass live=True to also stream updates to a live-updating window (see
    LiveMetricsPlotter) as they're logged, in addition to the final
    plot()/savefig().
    """

    def __init__(self, live=False):
        self.steps = []
        self.history = {}  # key -> list of float values (nan where missing)
        self.live_plotter = LiveMetricsPlotter() if live else None

    def log(self, step, d):
        self.steps.append(step)
        scalar_items = {k: float(v) for k, v in flatten_scalar_dict(d).items()}

        for k in scalar_items:
            if k not in self.history:
                self.history[k] = [float("nan")] * (len(self.steps) - 1)

        for k, hist in self.history.items():
            hist.append(scalar_items.get(k, float("nan")))

        if self.live_plotter is not None:
            self.live_plotter.log(step, d)

    def plot(self, save_path="metrics.png", show=False):
        if not self.history:
            print("MetricsLogger: nothing to plot (no scalar float metrics logged).")
            return

        keys = sorted(self.history.keys())
        ncols = 3
        nrows = math.ceil(len(keys) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for ax, k in zip(axes.flat, keys):
            xs, ys = _drop_nan(self.steps, self.history[k])
            ax.plot(xs, ys)
            ax.set_ylabel(k)
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(keys):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved metrics plot to {save_path}")

    def close(self):
        if self.live_plotter is not None:
            self.live_plotter.close()


def weighted_point_and_jacobian(model, data, body_points, dof_idxs):
    """
    Computes weighted-average world position and position Jacobian for given
    (body_name, local_offset, weight) triples. local_offset is a 3-vector in
    the body's own local frame, letting the target point be somewhere other
    than the body origin (e.g. the midpoint of a capsule that is welded to,
    but geometrically offset/rotated from, the body it's attached to).
    """
    point = np.zeros(3)
    J = np.zeros((3, len(dof_idxs)))
    for name, local_offset, w in body_points:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        xmat = data.xmat[bid].reshape(3, 3)
        world_pt = data.xpos[bid] + xmat @ local_offset
        point += w * world_pt
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, world_pt, bid)
        J += w * jacp[:, dof_idxs]
    return point, J


def jacobian_ik(model, data, body_points, target_pos, joint_names,
                max_iters=200, damping=1e-2, step_clip=0.1, tol=1e-4,
                angle_body=None, angle_local_dir=None, target_angle=None,
                angle_weight=1.0, angle_tol=1e-3):
    """
    Damped-least-squares IK to position specified target point combination,
    optionally augmented with a single angle-to-ground constraint.

    body_points is a list of (body_name, local_offset, weight) triples, see
    weighted_point_and_jacobian.

    joint_names may include unactuated joints (e.g. the passive bow_frog_hinge)
    as extra free DOFs for the solver to use -- their qpos gets solved for and
    written just like any actuated joint, it just never gets copied into ctrl
    (see envs.utils_envs.set_joint_ctrl). Joints with hard limits (jnt_limited)
    are clamped to model.jnt_range after every step so the solver can't swing
    them past their physical stops; unlimited joints (e.g. joint1..5) are
    unaffected.

    Passing `target_angle` (radians) together with `angle_body` and
    `angle_local_dir` (a 3-vector fixed in `angle_body`'s local frame, not
    necessarily normalized -- e.g. a link's long axis) adds one extra scalar
    row to the least-squares system: the angle between that direction,
    rotated to world frame, and the horizontal ground plane (arcsin of its
    normalized world-frame z-component) is driven to `target_angle`, via the
    body's rotational Jacobian (d(world_dir)/dq_i = jacr_col_i x world_dir,
    chained through d(arcsin)/du_z) stacked below the position Jacobian, and
    the angle error stacked below the position error -- the same recipe as
    stacking a 3-row orientation constraint, just for one row instead of
    three, which is both cheaper and much better conditioned on an arm with
    few DOF. `angle_weight` scales that row before combining, same role as
    the position/rotation weighting elsewhere. Convergence requires the
    position error norm < tol and, when the angle constraint is enabled, the
    angle error < angle_tol.

    If `target_angle` is None (the default), behavior is unchanged from
    plain position-only IK.
    """
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joint_names]
    dof_idxs = [model.jnt_dofadr[jid] for jid in jids]
    qpos_idxs = [model.jnt_qposadr[jid] for jid in jids]
    limits = [model.jnt_range[jid] if model.jnt_limited[jid] else None for jid in jids]

    angle_bid = None
    if target_angle is not None:
        if angle_body is None or angle_local_dir is None:
            raise ValueError("angle_body and angle_local_dir are required when target_angle is given")
        angle_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, angle_body)
        local_dir = np.asarray(angle_local_dir, dtype=np.float64)
        local_dir = local_dir / np.linalg.norm(local_dir)

    def _angle_err_and_jac():
        xmat = data.xmat[angle_bid].reshape(3, 3)
        u = xmat @ local_dir
        uz = np.clip(u[2], -1.0, 1.0)
        current_angle = np.arcsin(uz)
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, None, jacr, angle_bid)
        duz_dq = np.cross(jacr[:, dof_idxs].T, u)[:, 2]
        denom = max(np.sqrt(max(1.0 - uz**2, 1e-8)), 1e-4)
        dangle_dq = duz_dq / denom
        return target_angle - current_angle, dangle_dq

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        point, J = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
        pos_err = target_pos - point
        pos_err_norm = np.linalg.norm(pos_err)

        if angle_bid is None:
            if pos_err_norm < tol:
                return it, pos_err_norm
            err, Jfull = pos_err, J
        else:
            angle_err, dangle_dq = _angle_err_and_jac()
            if pos_err_norm < tol and abs(angle_err) < angle_tol:
                return it, pos_err_norm, angle_err
            err = np.concatenate([pos_err, [angle_weight * angle_err]])
            Jfull = np.concatenate([J, (angle_weight * dangle_dq).reshape(1, -1)], axis=0)

        dtheta = Jfull.T @ np.linalg.solve(
            Jfull @ Jfull.T + damping**2 * np.eye(Jfull.shape[0]), err
        )
        step_norm = np.linalg.norm(dtheta)
        if step_norm > step_clip:
            dtheta *= step_clip / step_norm
        for qidx, d, lim in zip(qpos_idxs, dtheta, limits):
            data.qpos[qidx] += d
            if lim is not None:
                data.qpos[qidx] = np.clip(data.qpos[qidx], lim[0], lim[1])

    mujoco.mj_forward(model, data)
    point, _ = weighted_point_and_jacobian(model, data, body_points, dof_idxs)
    pos_err_norm = np.linalg.norm(target_pos - point)
    if angle_bid is None:
        return max_iters, pos_err_norm
    angle_err, _ = _angle_err_and_jac()
    return max_iters, pos_err_norm, angle_err


def make_env(env_class, ep_len, num_envs, **kwargs):
    """Builds a single environment instance, wrapped with the standard
    EpisodeWrapper and AutoResetWrapper. kwargs are passed to env_class.
    """
    env = env_class(**kwargs)
    env = EpisodeWrapper(env, episode_length=ep_len)
    env = AutoResetWrapper(env)
    env = VmapWrapper(env, num_envs=num_envs)
    return env

if __name__ == "__main__":
    pass
