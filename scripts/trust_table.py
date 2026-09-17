"""One table from the withheld-frame grades: footprint accuracy, the coverage
and deviation thresholds each scene implies, and the pooled curve.

    python scripts/trust_table.py /scratch/.../outputs/trust
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--a_min", type=float, default=0.8)
    ap.add_argument("--cov_bins", default="0,0.2,0.4,0.6,0.8,1.01")
    args = ap.parse_args()
    rows, frames = [], []
    for sub in sorted(args.root.glob("*/grade/summary.json")):
        d = json.loads(sub.read_text())
        rows.append((sub.parent.parent.name, d))
        f = sub.parent / "frames.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                try:
                    frames.append((float(r["coverage"]), float(r["deviation"]), float(r["fp_trav_acc"]),
                                   float(r["pix_acc"]), float(r["a_map"])))
                except ValueError:
                    pass
    if not rows:
        raise SystemExit(f"no grades under {args.root}")
    print(f"{'scene_pattern':<26}{'frames':>7}{'pix':>7}{'fp_trav':>9}{'A_map':>7}{'tau':>7}{'tau_term':>10}{'delta_q':>9}")
    for name, d in rows:
        print(f"{name:<26}{d['frames']:>7}{d['pix_acc_mean']:>7.3f}{d['fp_trav_acc_mean']:>9.3f}"
              f"{(d['a_map_mean'] if d.get('a_map_mean') is not None else float('nan')):>7.3f}"
              f"{str(d.get('tau_from_curve')):>7}{str(d.get('tau_term_from_curve')):>10}{str(d.get('delta_q_from_curve')):>9}")
    if frames:
        A = np.array(frames, dtype=float)
        edges = [float(v) for v in args.cov_bins.split(",")]
        print(f"\npooled over {len(A)} withheld frames — footprint traversable-vs-not accuracy by coverage:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (A[:, 0] >= lo) & (A[:, 0] < hi) & np.isfinite(A[:, 2])
            print(f"  coverage [{lo:.1f},{hi:.1f})  n {int(m.sum()):>5}  acc {np.nanmean(A[m, 2]) if m.any() else float('nan'):.3f}")
        ok = [lo for lo, hi in zip(edges[:-1], edges[1:])
              if ((A[:, 0] >= lo) & (A[:, 0] < hi) & np.isfinite(A[:, 2])).sum() > 0
              and np.nanmean(A[((A[:, 0] >= lo) & (A[:, 0] < hi)), 2]) >= args.a_min]
        print(f"  -> pooled tau (lowest coverage bin still >= {args.a_min}): {min(ok) if ok else None}")


if __name__ == "__main__":
    main()
