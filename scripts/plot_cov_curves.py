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
# 2026-09-02: drive_preview now also prints `dlt` — mean |dRGB| between
# consecutive generated frames. Same ladder, second curve: does the
# frame-difference measure of incoherence agree with the geometric one?
DLT = re.compile(r"off\s+([-+][\d.]+)deg.*?dlt\s+([\d.]+)")


def curve(path: Path, pat=LINE):
    """offset -> mean value (the sweep visits each offset twice; average them)."""
    acc = {}
    for ln in path.read_text(errors="ignore").splitlines():
        m = pat.search(ln)
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
    ax2 = [None]          # list so the twin axis is created at most once
    plotted = 0
    for spec in args.runs:
        label, _, p = spec.partition("=")
        off, cov = curve(Path(p))
        if off is None:
            print(f"[skip] {label}: no 'off ... cov ...' lines in {p}")
            continue
        ax.plot(off, cov, "-o", ms=3, label=f"{label} cov")
        plotted += 1
        # second curve on a twin axis: dlt lives in 0-1, cov in 0-100
        doff, dlt = curve(Path(p), DLT)
        if doff is not None:
            if ax2[0] is None:
                ax2[0] = ax.twinx()
                ax2[0].set_ylabel("mean |dRGB| between consecutive frames")
            ax2[0].plot(doff, dlt, "--s", ms=3, alpha=0.75,
                        label=f"{label} dlt")
            print(f"{label:<12} dlt@0deg {np.interp(0, doff, dlt):.4f}   "
                  f"dlt@+-{args.cone_deg/2:.0f}deg "
                  f"{np.interp(args.cone_deg/2.0, doff, dlt):.4f}   "
                  f"dlt max {dlt.max():.4f} at {doff[int(dlt.argmax())]:+.0f}deg")
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
    ax.legend(loc="upper left", fontsize=8)
    if ax2[0] is not None:
        ax2[0].legend(loc="upper right", fontsize=8)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
