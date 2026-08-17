"""plot_demo.py -- plot arrays from a recorded demonstration .npz.

Usage:
    python demonstrations/plot_demo.py demo.npz
        No --data given: lists every array in the file (name, shape,
        dtype) and plots the per-step ones with <= 5 dims (one figure per
        array, one subplot per dimension -- see plot_timeseries). Wider
        arrays (e.g. the 58-dim obs) and DemoRecorder's dr_*/init_*
        metadata (per-episode model/state, not a per-step series -- see
        teleop.py) are listed but skipped; slice them explicitly via
        --data if you want them plotted.

    python demonstrations/plot_demo.py demo.npz --data "demo['obs'][:, :3], demo['action']"
        Plots exactly what you ask for. --data is Python evaluated with
        `demo` bound to the loaded npz (a name -> ndarray mapping) and `np`
        available -- split on top-level commas, so each comma-separated
        clause becomes its own figure, titled with its own source text.
        Wrapping the whole thing in [...] is also fine.

    python demonstrations/plot_demo.py demo.npz --data "demo['reward']" --x "demo['sim_time']"
        --x picks what goes on the x-axis (default: step index).
"""

import argparse
import re
from pathlib import Path

import numpy as np


def plot_timeseries(t, data, labels=None, title=None, ylim=None, colors=None):
    """
    Plot a (T,) or (T, D) array against time t, one subplot per dimension.

    Parameters
    ----------
    t : array, shape (T,)
        Time values for the x-axis (e.g. sim_time).
    data : array, shape (T,) or (T, D) with D <= 5
        Data to plot. 1-D arrays are treated as a single dimension.
    labels : list of str, optional
        Per-dimension labels for y-axis / legend. Defaults to dim indices.
    title : str, optional
        Figure title.
    ylim : tuple (ymin, ymax), optional
        Shared y-limits applied to every subplot. If None, each subplot
        auto-scales to its own data.
    colors : list, optional
        Per-dimension colors. Defaults to matplotlib's tab10 colormap.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"data must be 1-D or 2-D, got shape {data.shape}")

    n_dims = data.shape[1]
    if n_dims > 7:
        raise ValueError(f"data has {n_dims} dims; this plot supports dim <= 7")
    if data.shape[0] != len(t):
        raise ValueError(
            f"data length ({data.shape[0]}) does not match t length ({len(t)})"
        )

    if labels is None:
        labels = [f"dim {i}" for i in range(n_dims)]
    if len(labels) != n_dims:
        raise ValueError(f"labels has {len(labels)} entries but data has {n_dims} dims")

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_dims, 2)))

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.1 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, data[:, i], color=colors[i], lw=0.9, label=labels[i])
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_ylabel(labels[i])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig


def _split_top_level(expr: str):
    """Splits `expr` on commas not nested inside (), [], {} -- lets each
    comma-separated clause of --data be eval'd (and labeled) independently,
    the way a plain tuple literal reads, without requiring the whole string
    to itself be valid list/tuple syntax."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(expr):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(expr[start:i])
            start = i + 1
    parts.append(expr[start:])
    return [p.strip() for p in parts if p.strip()]


def _strip_wrapping_brackets(expr: str) -> str:
    """`--data "[a, b]"` and `--data "a, b"` should behave the same --
    strip one layer of enclosing [...] or (...) before splitting."""
    expr = expr.strip()
    if len(expr) >= 2 and expr[0] in "([" and expr[-1] in ")]":
        return expr[1:-1]
    return expr


def eval_clauses(expr: str, demo: dict):
    """Returns [(label, array), ...], one per top-level comma-separated
    clause of `expr`, each eval'd against `demo` (and `np`). The clause's
    own source text is used as its label."""
    namespace = {"demo": demo, "np": np}
    out = []
    for clause in _split_top_level(_strip_wrapping_brackets(expr)):
        value = eval(clause, {"__builtins__": {}}, namespace)
        out.append((clause, np.asarray(value)))
    return out


def _safe_filename(label: str) -> str:
    return re.sub(r"\W+", "_", label).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz_path", help="path to a demo_*.npz recorded by teleop.py")
    parser.add_argument(
        "--data", default=None,
        help="Python expression(s) selecting what to plot, e.g. "
             "\"demo['obs'][:, :3], demo['action']\" -- comma-separated "
             "clauses each become their own figure (<= 5 dims each). "
             "Default: every per-step array with <= 5 dims (see --list).",
    )
    parser.add_argument(
        "--x", default=None,
        help="Python expression for the x-axis, e.g. \"demo['sim_time']\" "
             "(default: step index).",
    )
    parser.add_argument("--save", default=None, help="output image path/prefix (default: <npz stem>_plot.png)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    parser.add_argument("--list", action="store_true", help="print array names/shapes/dtypes and exit")
    args = parser.parse_args()

    # Decide the matplotlib backend before importing pyplot: always
    # non-interactive unless --show is passed, matching utils.py's
    # MetricsLogger convention (see utils.py's macOS/mjpython note).
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(args.npz_path)
    npz = np.load(path)
    demo = {k: npz[k] for k in npz.files}

    print(f"{path}: {len(demo)} arrays")
    for k, v in demo.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    if args.list:
        return

    if args.data is not None:
        items = eval_clauses(args.data, demo)
    else:
        # dr_*/init_* are per-episode model/state metadata, not per-step
        # series (see teleop.py's DemoRecorder) -- skip them here, they
        # don't share the other arrays' leading "step" axis. Anything
        # wider than plot_timeseries' 5-dim limit (e.g. the 58-dim obs)
        # needs an explicit --data slice instead.
        items = []
        for k, v in demo.items():
            if k.startswith("dr_") or k.startswith("init_"):
                continue
            n_dims = 1 if v.ndim == 1 else v.shape[1] if v.ndim == 2 else None
            if n_dims is None or n_dims > 7:
                print(f"[plot_demo] skipping '{k}': shape={v.shape} -- pass --data to slice it down to <= 7 dims")
                continue
            items.append((k, v))

    if not items:
        print("[plot_demo] nothing to plot")
        return

    if args.x is not None:
        [(_, t)] = eval_clauses(args.x, demo)
    else:
        t = None  # filled in per-item below (step index of that item's own length)

    base = Path(args.save) if args.save else Path(f"{path.stem}_plot.png")
    multi = len(items) > 1

    for idx, (label, arr) in enumerate(items):
        item_t = t if t is not None else np.arange(arr.shape[0] if arr.ndim else 1)
        try:
            fig = plot_timeseries(item_t, arr, title=label)
        except ValueError as e:
            print(f"[plot_demo] skipping '{label}': {e}")
            continue
        out_path = (
            base.with_name(f"{base.stem}__{idx:02d}_{_safe_filename(label)}{base.suffix}")
            if multi else base
        )
        fig.savefig(out_path)
        print(f"Saved plot to {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
