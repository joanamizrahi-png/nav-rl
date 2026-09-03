"""Does the footprint read TERRAIN, or does it read RECONSTRUCTION COVERAGE?

The YAW LADDER holds the robot's position fixed and sweeps its heading, so the
footprint is the SAME 3528 pixels at every rung -- only the direction it points
changes. That makes it the one measurement that can separate "there is an
obstacle there" from "the world model invented an obstacle where it has no
geometry to render".

Read off the 2026-09-03 sweep on gnd_AU_180 (job 463850): rungs with coverage
below 0.2 averaged collision fraction ~0.62; rungs above 0.5 averaged ~0.026.
A 24x difference with the footprint held constant. If that holds up, the crash
terminator is firing largely on invented terrain -- and every training arm runs
NOGATE=1, so nothing filters it.

Joana's rule, and she is right: a number this consequential does not get
believed without a picture.

    python scripts/plot_yaw_ladder.py /scratch/.../slurm-check-rew-463850.out \
        --out /scratch/.../reward_check/yaw_ladder.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# yaw  in_front  HAS_PIXELS%  px  coll  mean_trav  cov
ROW = re.compile(
    r"^\s*([-+]?[\d.]+)\s+(yes|no)\s+(\d+)%\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse(path: Path):
    rows, inside = [], False
    for ln in path.read_text(errors="ignore").splitlines():
        if "YAW LADDER" in ln:
            inside = True
            continue
        if inside and ln.startswith("====="):
            break
        m = ROW.match(ln)
        if inside and m:
            rows.append((float(m.group(1)), float(m.group(5)),
                         float(m.group(6)), float(m.group(7))))
    if not rows:
        raise SystemExit("no YAW LADDER rows found — is this a --walk yaw run?")
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]      # yaw, coll, trav, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="yaw_ladder.png")
    ap.add_argument("--crash_frac", type=float, default=0.35,
                    help="collision_terminate_frac — the line above which an "
                         "episode ENDS at -1000")
    ap.add_argument("--tau", type=float, default=0.4, help="coherence_tau")
    args = ap.parse_args()

    yaw, coll, trav, cov = parse(Path(args.log))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # LEFT: the claim itself. One dot per rung, identical footprint each time.
    sc = ax1.scatter(cov, coll, c=np.abs(yaw), cmap="viridis", s=70,
                     edgecolor="k", linewidth=0.4)
    ax1.axhline(args.crash_frac, ls="--", lw=1.2, c="tab:red")
    ax1.text(0.02, args.crash_frac + 0.02, f"crash at {args.crash_frac}",
             c="tab:red", fontsize=9)
    ax1.axvline(args.tau, ls=":", lw=1.2, c="0.4")
    ax1.text(args.tau + 0.01, 0.02, f"tau_coh {args.tau}", c="0.35", fontsize=9)
    lo, hi = cov < 0.2, cov > 0.5
    for m, lbl, c in ((lo, "cov<0.2", "tab:red"), (hi, "cov>0.5", "tab:green")):
        if m.any():
            ax1.plot([cov[m].mean()], [coll[m].mean()], "*", ms=20, c=c,
                     mec="k", mew=0.6,
                     label=f"{lbl}: mean coll {coll[m].mean():.3f} (n={m.sum()})")
    ax1.set_xlabel("reconstruction coverage (mean alpha)")
    ax1.set_ylabel("collision fraction of the footprint")
    ax1.set_title("Same 3528 pixels every rung — only the heading changes")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax1, label="|yaw| from spawn heading (deg)")

    # RIGHT: both against yaw, so the "coverage is worst dead ahead" anomaly
    # is visible rather than asserted.
    o = np.argsort(yaw)
    ax2.plot(yaw[o], cov[o], "-o", ms=4, label="coverage")
    ax2.plot(yaw[o], coll[o], "-s", ms=4, label="collision fraction")
    ax2.axhline(args.crash_frac, ls="--", lw=1, c="tab:red")
    ax2.axvline(0, ls=":", lw=1, c="0.4")
    ax2.text(1, 0.96, "spawn heading", fontsize=8, c="0.35")
    ax2.set_xlabel("yaw offset from the spawn heading (deg)")
    ax2.set_ylabel("fraction")
    ax2.set_title("Where is the coverage, and where does it crash?")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")
    if lo.any() and hi.any():
        r = coll[lo].mean() / max(coll[hi].mean(), 1e-6)
        print(f"    low-coverage rungs collide {r:.0f}x more than high-coverage ones")
    i = int(np.argmin(np.abs(yaw)))
    print(f"    at the spawn heading ({yaw[i]:+.1f} deg): coverage {cov[i]:.3f}, "
          f"collision {coll[i]:.3f}")
    print(f"    coverage peaks {cov.max():.3f} at {yaw[int(np.argmax(cov))]:+.1f} deg")


if __name__ == "__main__":
    main()
