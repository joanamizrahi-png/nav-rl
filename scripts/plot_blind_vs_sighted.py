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
    ap.add_argument("--overview_out", default="",
                    help="also draw ALL episodes on ONE whole-scene map. This "
                         "is what shows whether the spawns are CLUSTERED -- "
                         "Joana's 2026-09-03 hypothesis was that gnd_AUw360 "
                         "spawns sit at the start of the walk while the grass "
                         "only appears near the end, which would make a "
                         "grass-avoidance measurement impossible by "
                         "construction rather than by policy behaviour.")
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
        _recorded_path = (np.asarray(c["traj_positions"], dtype=float)
                          * np.array([1.0, -1.0, 1.0]))[:, :2]
    else:
        _recorded_path = None
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
        # Take x,y only and tolerate ragged rows: the first entry of `traj` was
        # written with 4 fields and the rest with 5, so np.array() on the whole
        # thing raises. Old metrics.json files still have that shape.
        st = np.array([r[:2] for r in se["traj"]], dtype=float)
        bt = np.array([r[:2] for r in be["traj"]], dtype=float)
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

    if args.overview_out:
        overview(args, s_eps, b_eps, gxy, glab, _recorded_path)

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


def overview(args, s_eps, b_eps, gxy, glab, path=None):
    """Every episode on one map of the whole scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    if gxy is not None:
        bad = np.isin(glab, NONTRAV_DEFAULT)
        grass = glab == 3
        ax.scatter(gxy[~bad, 0], gxy[~bad, 1], s=1, c="#dedede", linewidths=0,
                   label="traversable")
        ax.scatter(gxy[bad & ~grass, 0], gxy[bad & ~grass, 1], s=1,
                   c="#9ecae1", linewidths=0, label="other non-traversable")
        ax.scatter(gxy[grass, 0], gxy[grass, 1], s=1, c="#74c476", linewidths=0,
                   label="GRASS")
    if path is not None and len(path):
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.2, c="0.35",
                label="recorded walk")
        ax.plot(*path[0], "P", ms=11, c="k")
        ax.annotate("frame 0", path[0], fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
        ax.annotate(f"frame {len(path) - 1}", path[-1], fontsize=9,
                    xytext=(6, 6), textcoords="offset points")

    for eps, c, lbl in ((s_eps, "tab:blue", "sighted"),
                        (b_eps, "tab:orange", "blind")):
        for i, e in enumerate(eps):
            t = np.array([r[:2] for r in e["traj"]], dtype=float)
            ax.plot(t[:, 0], t[:, 1], "-", lw=1.4, c=c, alpha=0.75,
                    label=lbl if i == 0 else None)
    for i, e in enumerate(s_eps):
        t = np.array([r[:2] for r in e["traj"]], dtype=float)
        ax.plot(*t[0], "o", ms=7, c="k", mec="w", mew=0.8,
                label="spawn" if i == 0 else None)
        g = np.array(e["goal_xy"], dtype=float)
        ax.plot(*g, "*", ms=13, c="tab:red", mec="k", mew=0.4,
                label="goal" if i == 0 else None)

    ax.set_aspect("equal")
    ax.legend(fontsize=9, markerscale=3, loc="best")
    ax.set_title(f"{args.scene}: every episode, whole scene\n"
                 f"are the spawns spread along the walk, and is there any "
                 f"grass where the robot actually goes?", fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.overview_out, dpi=150, bbox_inches="tight")
    print(f"==> {args.overview_out}")
    if gxy is not None:
        sp = np.array([np.array(e["traj"][0][:2], dtype=float) for e in s_eps])
        gl = np.array([e["goal_xy"] for e in s_eps], dtype=float)
        print(f"    spawn spread: x {sp[:, 0].min():.1f}..{sp[:, 0].max():.1f} m, "
              f"y {sp[:, 1].min():.1f}..{sp[:, 1].max():.1f} m")
        print(f"    goal  spread: x {gl[:, 0].min():.1f}..{gl[:, 0].max():.1f} m, "
              f"y {gl[:, 1].min():.1f}..{gl[:, 1].max():.1f} m")
        gr = gxy[glab == 3]
        print(f"    grass points in scene: {len(gr)} "
              f"({100.0 * len(gr) / max(len(gxy), 1):.1f}% of ground)")
        if len(gr):
            # distance from each spawn to the nearest grass point -- if this is
            # large for every spawn, the robot was never given the chance to
            # make a terrain decision, and a grass metric of 0.000 says nothing
            # about the policy.
            step = max(1, len(gr) // 20000)          # cap the pairwise cost
            d = np.min(np.linalg.norm(gr[::step][None, :, :] - sp[:, None, :],
                                      axis=2), axis=1)
            print(f"    nearest grass to each spawn: min {d.min():.1f} m, "
                  f"median {np.median(d):.1f} m, max {d.max():.1f} m")


if __name__ == "__main__":
    main()
