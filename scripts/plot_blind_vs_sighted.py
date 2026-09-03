"""Overhead paths: what does the SAME policy do with and without vision?

`eval_policy` stores a full trajectory per episode, and both halves of a pair
run under the same `--eval_seed`, so episode N is the SAME spawn and the SAME
goal in both. That makes the honest picture a paired overhead plot: two paths
from one start to one goal, one blind, one sighted, drawn over the scene's own
ground labels.

It exists because the summary table cannot settle the question. On
gnd_AUw360 (2026-09-03) grass contact came back 0.000 for BOTH halves of every
pair, so the trespass metric was degenerate -- while the ancestor's outcomes
differed enormously (sighted 19/20 TIMEOUT, blind 20/20 GOAL). A number that
says "no difference" next to outcomes that say "completely different" means the
number is measuring the wrong thing. The paths show what actually happened.

    python scripts/plot_blind_vs_sighted.py \
        --sighted /scratch/.../eval_..._gnd_AUw360_fwd_live \
        --blind   /scratch/.../eval_..._blind_..._gnd_AUw360_fwd_live \
        --scene gnd_AUw360 --out /scratch/.../paths.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NONTRAV_DEFAULT = (0, 1, 3, 5, 10, 11, 12, 13)   # void sky grass water obstacle veg person vehicle


def load(run: Path):
    d = json.loads((run / "metrics.json").read_text())
    return d["summary"], d["episodes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sighted", required=True)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--out", default="blind_vs_sighted_paths.png")
    ap.add_argument("--episodes", type=int, default=12,
                    help="how many paired episodes to draw")
    ap.add_argument("--z_max", type=float, default=0.15)
    args = ap.parse_args()

    s_sum, s_eps = load(Path(args.sighted))
    b_sum, b_eps = load(Path(args.blind))

    gxy = glab = None
    cp = Path(args.clouds_dir) / f"{args.scene}_cloud.npz"
    if cp.exists():
        c = np.load(cp)
        pts, labs = c["points"], c["labels"].astype(int)
        m = pts[:, 2] < args.z_max
        gxy, glab = pts[m][:, :2], labs[m]
        # subsample: 150k points is unreadable and slow to render
        if len(gxy) > 60000:
            k = np.random.default_rng(0).choice(len(gxy), 60000, replace=False)
            gxy, glab = gxy[k], glab[k]
    else:
        print(f"[warn] no cloud at {cp}; drawing paths without terrain")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.episodes, len(s_eps), len(b_eps))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        ax, se, be = axes[i], s_eps[i], b_eps[i]
        st = np.array(se["traj"], dtype=float)
        bt = np.array(be["traj"], dtype=float)
        goal = np.array(se["goal_xy"], dtype=float)

        if gxy is not None:
            # frame the panel on the action, then draw only nearby terrain
            allxy = np.vstack([st[:, :2], bt[:, :2], goal[None, :]])
            lo, hi = allxy.min(0) - 4.0, allxy.max(0) + 4.0
            m = ((gxy[:, 0] > lo[0]) & (gxy[:, 0] < hi[0])
                 & (gxy[:, 1] > lo[1]) & (gxy[:, 1] < hi[1]))
            bad = np.isin(glab[m], NONTRAV_DEFAULT)
            ax.scatter(gxy[m][~bad, 0], gxy[m][~bad, 1], s=1, c="#d9d9d9",
                       linewidths=0)
            ax.scatter(gxy[m][bad, 0], gxy[m][bad, 1], s=1, c="#a8d5a2",
                       linewidths=0)

        ax.plot(st[:, 0], st[:, 1], "-", lw=2.0, c="tab:blue", label="sighted")
        ax.plot(bt[:, 0], bt[:, 1], "-", lw=2.0, c="tab:orange", label="blind")
        ax.plot(*st[0, :2], "ko", ms=6)
        ax.plot(*st[-1, :2], "s", ms=7, c="tab:blue", mec="k", mew=0.5)
        ax.plot(*bt[-1, :2], "s", ms=7, c="tab:orange", mec="k", mew=0.5)
        ax.plot(*goal, "*", ms=16, c="tab:red", mec="k", mew=0.5)
        ax.set_title(f"ep{i}  sighted {se['outcome']} ({se['steps']}st)\n"
                     f"blind {be['outcome']} ({be['steps']}st)", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{args.scene}: same weights, same spawn, same goal — with and without vision\n"
        f"sighted {s_sum['outcomes']}   blind {b_sum['outcomes']}   "
        f"grass share {s_sum['ground_share'].get('grass', 0):.3f} vs "
        f"{b_sum['ground_share'].get('grass', 0):.3f}   "
        f"(black dot = spawn, red star = goal, square = where it stopped; "
        f"green = non-traversable)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")

    # the paired numbers, printed so the figure does not have to be squinted at
    print(f"{'ep':>3} {'sighted':>22} {'blind':>22}   d_start")
    for i in range(n):
        se, be = s_eps[i], b_eps[i]
        print(f"{i:>3} {se['outcome']:>10} {se['steps']:>3}st "
              f"c={se['closed_frac']!s:>6} "
              f"{be['outcome']:>10} {be['steps']:>3}st "
              f"c={be['closed_frac']!s:>6}   {se['d_start']}")


if __name__ == "__main__":
    main()
