"""Delete demonstration files whose episode length is below a threshold.

Usage:
    python clean_demo.py [--min-len 200] [--dir demonstrations] [--dry-run]
"""

import argparse
import glob
import os

import numpy as np


def demo_length(path):
    """Return the number of timesteps stored in a demo .npz file."""
    with np.load(path, allow_pickle=True) as data:
        return len(data["obs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-len", type=int, default=200,
                         help="Delete demos with fewer than this many steps.")
    parser.add_argument("--dir", type=str, default="demonstrations",
                         help="Directory containing demo_*.npz files.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Only print what would be deleted.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "demo_*.npz")))
    if not paths:
        print(f"No demo_*.npz files found in {args.dir}")
        return

    deleted, kept = 0, 0
    demo_total_len = 0
    for path in paths:
        try:
            n = demo_length(path)
        except Exception as e:
            print(f"[skip] {os.path.basename(path)}: could not read ({e})")
            continue
        
        if n < args.min_len:
            action = "[dry-run] would delete" if args.dry_run else "[delete]"
            print(f"{action} {os.path.basename(path)} (len={n})")
            if not args.dry_run:
                os.remove(path)
            deleted += 1
        else:
            kept += 1
            demo_total_len += n

    print(f"\n{deleted} demo(s) {'would be ' if args.dry_run else ''}deleted, {kept} kept "
          f"(threshold: len >= {args.min_len}).")
    print(f"Total timesteps in accepted demos: {demo_total_len}")


if __name__ == "__main__":
    main()
