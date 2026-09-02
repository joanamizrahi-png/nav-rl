"""Reconstruction coverage vs viewing angle, for one or more spin previews.

drive_preview prints one line per swept frame:
    [12/73] pose 40 1.34s  off  -87.4deg  cov  41.2%
`cov` is mean alpha — the same statistic the coherence cost reads and the same
one the ladder was sorted by. Plotting it against the sweep offset turns a spin
log into the figure that says how far the robot can TURN before the world model
is inventing, which is the rotational twin of "how far can it step off path".

Measured on gnd_AUpano01 forward-only (job 460611, 2026-09-01): 83.5% dead
ahead, ~50% at +-25 deg (exactly Joana's chosen goal cone), ~40% at +-33 deg
(exactly tau_coh), and 0% beyond +-105 deg.

Usage:
    python scripts/plot_cov_curves.py --out cov.png \
        fwd-only=/scratch/.../slurm-drive-prev-460611.out \
        pano=/scratch/.../slurm-drive-prev-460646.out
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"off\s+([-+][\d.]+)deg.*?cov\s+([\d.]+)%")


def curve(path: Path):
    """offset -> mean cov (the sweep visits each offset twice; average them)."""
    acc = {}
    for ln in path.read_text(errors="ignore").splitlines():
        m = LINE.search(ln)
        if m:
            acc.setdefault(float(m.group(1)), []).append(float(m.group(2)))
    if not acc:
        return None, None
    off = np.array(sorted(acc))
    return off, np.array([float(np.mean(acc[o])) for o in off])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=/path/to/slurm-drive-prev-N.out")
    ap.add_argument("--out", default="cov_curves.png")
    ap.add_argument("--tau_coh", type=float, default=0.4)
    ap.add_argument("--cone_deg", type=float, default=50.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for spec in args.runs:
        label, _, p = spec.partition("=")
        off, cov = curve(Path(p))
        if off is None:
            print(f"[skip] {label}: no 'off ... cov ...' lines in {p}")
            continue
        ax.plot(off, cov, "-o", ms=3, label=label)
        plotted += 1
        half = args.cone_deg / 2.0
        at_cone = float(np.interp(half, off, cov))
        # the CONTIGUOUS band around 0 that stays above tau_coh — "how far can
        # it turn and still be looking at the world". Reporting min/max of all
        # sub-threshold offsets instead would just echo the sweep limits.
        ok = cov >= args.tau_coh * 100
        i0 = int(np.argmin(np.abs(off)))
        lo = hi = i0
        if ok[i0]:
            while lo > 0 and ok[lo - 1]:
                lo -= 1
            while hi < len(ok) - 1 and ok[hi + 1]:
                hi += 1
            band = f"{off[lo]:+.0f}..{off[hi]:+.0f}deg"
        else:
            band = "never (cov below tau even straight ahead)"
        print(f"{label:<12} cov@0deg {np.interp(0, off, cov):5.1f}%   "
              f"cov@+-{half:.0f}deg {at_cone:5.1f}%   "
              f"coherent band (cov >= {args.tau_coh}): {band}")

    if not plotted:
        raise SystemExit("nothing to plot — no run produced 'off ... cov ...' "
                         "lines yet (a spin job still loading prints nothing)")

    ax.axhline(args.tau_coh * 100, ls="--", lw=1, c="0.4")
    # anchor to the AXIS, not to the last curve's array — a skipped run used to
    # leave `off` as None here and crash after the useful output had printed.
    ax.text(ax.get_xlim()[0], args.tau_coh * 100 + 1.5,
            f"tau_coh = {args.tau_coh}", fontsize=8, c="0.3")
    for s in (-1, 1):
        ax.axvline(s * args.cone_deg / 2.0, ls=":", lw=1, c="tab:red")
    ax.text(args.cone_deg / 2.0 + 2, 5, f"goal cone +-{args.cone_deg/2:.0f}deg",
            fontsize=8, c="tab:red")
    ax.set_xlabel("viewing offset from the walk direction (deg)")
    ax.set_ylabel("reconstruction coverage, mean alpha (%)")
    ax.set_title("How far can the robot turn before the world model is inventing?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
