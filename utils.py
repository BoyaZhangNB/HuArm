import math
import multiprocessing as mp
import time
from queue import Empty

from envs.wrappers import AutoResetWrapper, EpisodeWrapper, VmapWrapper

import jax.numpy as jp
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
                line.set_data(steps, history[k])
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
            ax.plot(self.steps, self.history[k])
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
