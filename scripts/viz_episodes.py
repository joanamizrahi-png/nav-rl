"""Where did the policy actually GO? Episode trajectories on the terrain map.

eval_policy stores per-step [x, y, yaw, collision_frac, ground_class] for every
episode. Success rate collapses all of that into one bit — and when the goal
sits on grass, that bit is the WRONG question anyway (stopping at the boundary
is correct, arriving is a trespass). Drawing the paths over the scene cloud's
ground labels shows the behaviour directly: did it approach and stop, did it
walk onto the grass, or did it wander into unreconstructed space.

Overlaying two policies answers the lineage question in one picture — e.g.
459126 (warm-started, 200k steps under a reward where grass scored 0.75)
against 459127 (cold, J-spec only).

Usage (login node, no GPU):
    python scripts/viz_episodes.py --scene gnd_AUw360 --out eps.png \
        warm=/scratch/.../eval_..._warm_.../metrics.json \
        cold=/scratch/.../eval_..._cur_.../metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk", "road",
       "pavement", "stairs", "obstacle", "vegetation", "person", "vehicle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=/path/to/metrics.json")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--out", default="episodes.png")
    args = ap.parse_args()

    cloud = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = cloud["points"], cloud["labels"].astype(int)
    # same nav-frame y-mirror trap as pick_goal / spawn_audit
    path = np.asarray(cloud["traj_positions"], dtype=float) * np.array([1., -1., 1.])
    path = path[:, :2]
    gnd = pts[:, 2] < args.z_max
    gxy, glab = pts[gnd][:, :2], labs[gnd]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    for cid, col, name in ((6, "0.82", "sidewalk"), (8, "0.82", None),
                           (7, "0.68", "road"), (3, "tab:green", "grass")):
        m = glab == cid
        if m.any():
            ax.scatter(gxy[m][::12, 0], gxy[m][::12, 1], s=1, c=col,
                       linewidths=0, label=name, zorder=1)
    ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2.5,
            label="recorded walk", zorder=3)

    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for k, spec in enumerate(args.runs):
        label, _, p = spec.partition("=")
        d = json.loads(Path(p).read_text())
        eps = d["episodes"]
        c = colors[k % len(colors)]
        goal = None
        n_cr = 0
        for i, e in enumerate(eps):
            t = np.asarray(e["traj"], dtype=float)
            if t.ndim != 2 or len(t) < 2:
                continue
            ax.plot(t[:, 0], t[:, 1], "-", c=c, lw=1.2, alpha=0.65, zorder=4,
                    label=label if i == 0 else None)
            ax.plot(t[0, 0], t[0, 1], "o", c=c, ms=4, mec="k", mew=0.4, zorder=5)
            ax.plot(t[-1, 0], t[-1, 1], "s", c=c, ms=6, mec="k", mew=0.6, zorder=5)
            if t.shape[1] > 3:
                bad = t[t[:, 3] >= args.crash_frac]
                n_cr += len(bad)
                if len(bad):
                    ax.plot(bad[:, 0], bad[:, 1], "x", c="red", ms=7, mew=1.6,
                            zorder=6)
        s = d["summary"]
        gs = s.get("ground_share", {})
        top = ", ".join(f"{k2} {v:.0%}" for k2, v in
                        sorted(gs.items(), key=lambda kv: -kv[1])[:3])
        print(f"{label:<8} eps {len(eps):2d}  success {s.get('success_rate')}  "
              f"return {s.get('mean_return')}  crash-steps drawn {n_cr}  [{top}]")

    ax.plot([], [], "x", c="red", ms=7, mew=1.6,
            label=f"footprint >= {args.crash_frac} non-traversable")
    ax.plot([], [], "o", c="0.3", ms=4, label="spawn")
    ax.plot([], [], "s", c="0.3", ms=6, label="final pose")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(f"{args.scene}: where the policies went\n"
                 f"(goal on grass — stopping at the boundary is the CORRECT "
                 f"outcome, arriving is a trespass)", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
