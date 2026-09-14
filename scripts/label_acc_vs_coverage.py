"""The grader: accuracy of generated labels vs a reference, as a function of
coverage and of pose deviation -> the trust-range curves A(c), A(d).

Inputs are per-frame arrays of one render run (T frames):
  --pred   npz with `labels` [T,H,W] int   generated labels (semantic_labels.npz)
  --alpha  npz with `alpha`  [T,H,W(,1)]   rasterizer opacity of the same views
  --ref    npz with `labels` [T,H,W] int   reference labels for the same views:
           SAM3 on the withheld real frame, the walkthrough's own labels at
           recorded poses, or human labels (0 = void/unlabeled, ignored)
  --hint   (optional) npz `labels` [T,H,W] rasterized labels -> A_map on covered px
  --deviation (optional) csv/npz with one value per frame: Mahalanobis deviation
           of the query pose from the walk (from pose_vs_odom / the render log)

Per frame it computes pixel accuracy (non-void), per-class IoU, and the number
the reward reads: traversable-vs-not accuracy inside the footprint region.
Then it bins frames by coverage (mean alpha) and by deviation and writes:
  <out>/frames.csv       frame, coverage, deviation, pix_acc, fp_trav_acc, a_map
  <out>/bins.csv         axis (c|d), bin_lo, bin_hi, n, pix_acc, fp_trav_acc
  <out>/curves.png       A(c) and A(d)
  <out>/summary.json     thresholds: coverage where fp_trav_acc crosses --a_min,
                         and where it collapses (< --a_collapse); same for d

    python scripts/label_acc_vs_coverage.py --pred run/semantic_labels.npz \
        --alpha run/alpha.npz --ref refs/withheld_sam3.npz --out out/grade \
        [--hint run/holey_labels.npz] [--deviation run/deviation.csv] \
        [--trav 2,4,6,7,8,9] [--footprint 0.62,0.98,0.30,0.70]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_TRAV = "2,4,6,7,8,9"     # v14 palette: dirt, mulch, sidewalk, pavement, trail, road


def load_labels(path: Path, key: str = "labels") -> np.ndarray:
    d = np.load(path)
    arr = d[key] if key in d else d[list(d.keys())[0]]
    return np.asarray(arr)


def load_alpha(path: Path) -> np.ndarray:
    d = np.load(path)
    a = np.asarray(d["alpha"] if "alpha" in d else d[list(d.keys())[0]], dtype=np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    return a


def load_deviation(path: Path | None, T: int):
    if path is None:
        return np.full(T, np.nan)
    if str(path).endswith(".npz"):
        d = np.load(path); v = np.asarray(d[list(d.keys())[0]], dtype=float)
    else:
        rows = [ln.strip().split(",") for ln in open(path) if ln.strip()]
        hdr = rows[0]
        col = hdr.index("mahalanobis") if "mahalanobis" in hdr else len(hdr) - 1
        v = np.array([float(r[col]) for r in rows[1:]], dtype=float)
    if len(v) != T:
        raise SystemExit(f"deviation has {len(v)} values, labels have {T} frames")
    return v


def resize_nn(x: np.ndarray, hw: tuple) -> np.ndarray:
    if x.shape[-2:] == hw:
        return x
    import cv2
    out = np.empty((x.shape[0],) + hw, dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = cv2.resize(x[i].astype(np.float32), (hw[1], hw[0]),
                            interpolation=cv2.INTER_NEAREST).astype(x.dtype)
    return out


def footprint_mask(H: int, W: int, spec: str) -> np.ndarray:
    r0, r1, c0, c1 = [float(v) for v in spec.split(",")]
    m = np.zeros((H, W), bool)
    m[int(r0 * H):int(r1 * H), int(c0 * W):int(c1 * W)] = True
    return m


def per_frame(pred, ref, alpha, hint, trav, fp, C):
    T = pred.shape[0]
    pix_acc = np.full(T, np.nan); fp_acc = np.full(T, np.nan); a_map = np.full(T, np.nan)
    inter = np.zeros(C); union = np.zeros(C)
    trav_set = np.zeros(C, bool); trav_set[list(trav)] = True
    for t in range(T):
        p, r = pred[t], ref[t]
        valid = r != 0
        if valid.any():
            pix_acc[t] = float((p[valid] == r[valid]).mean())
            for c in range(1, C):
                pm, rm = (p == c) & valid, (r == c)
                inter[c] += (pm & rm).sum(); union[c] += (pm | rm).sum()
        v_fp = valid & fp
        if v_fp.any():
            pt = trav_set[np.clip(p[v_fp], 0, C - 1)]
            rt = trav_set[np.clip(r[v_fp], 0, C - 1)]
            fp_acc[t] = float((pt == rt).mean())
        if hint is not None:
            cov_px = (alpha[t] > 0.5) & fp & (hint[t] != 0)
            if cov_px.sum() > 20:
                a_map[t] = float((p[cov_px] == hint[t][cov_px]).mean())
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return pix_acc, fp_acc, a_map, iou


def bin_curve(x, y, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        rows.append((lo, hi, int(m.sum()), float(np.nanmean(y[m])) if m.any() else np.nan))
    return rows


def threshold_from_curve(rows, a_min, a_collapse):
    """Largest bin-lower-edge where mean acc is still >= a_min (coverage axis reads
    from high to low), and the edge below which it falls under a_collapse."""
    keep = [lo for lo, hi, n, a in rows if n > 0 and np.isfinite(a) and a >= a_min]
    coll = [hi for lo, hi, n, a in rows if n > 0 and np.isfinite(a) and a < a_collapse]
    return (min(keep) if keep else None), (max(coll) if coll else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--hint", type=Path, default=None)
    ap.add_argument("--deviation", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trav", default=DEFAULT_TRAV, help="traversable class ids")
    ap.add_argument("--num_classes", type=int, default=14)
    ap.add_argument("--footprint", default="0.62,0.98,0.30,0.70",
                    help="row0,row1,col0,col1 as image fractions (the box ahead)")
    ap.add_argument("--cov_bins", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01")
    ap.add_argument("--dev_bins", default="0,0.5,1,1.5,2,3,4,6,1e9")
    ap.add_argument("--a_min", type=float, default=0.8)
    ap.add_argument("--a_collapse", type=float, default=0.5)
    args = ap.parse_args()

    pred = load_labels(args.pred).astype(np.int64)
    ref = load_labels(args.ref).astype(np.int64)
    alpha = load_alpha(args.alpha)
    T = min(len(pred), len(ref), len(alpha))
    pred, ref, alpha = pred[:T], ref[:T], alpha[:T]
    hw = pred.shape[-2:]
    ref = resize_nn(ref, hw); alpha = resize_nn(alpha, hw)
    hint = resize_nn(load_labels(args.hint).astype(np.int64)[:T], hw) if args.hint else None
    dev = load_deviation(args.deviation, T)
    trav = [int(v) for v in args.trav.split(",") if v.strip()]
    fp = footprint_mask(hw[0], hw[1], args.footprint)
    cov = alpha.reshape(T, -1).mean(1)

    pix_acc, fp_acc, a_map, iou = per_frame(pred, ref, alpha, hint, trav, fp, args.num_classes)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "frames.csv", "w") as fh:
        fh.write("frame,coverage,deviation,pix_acc,fp_trav_acc,a_map\n")
        for t in range(T):
            fh.write(f"{t},{cov[t]:.4f},{dev[t]:.3f},{pix_acc[t]:.4f},{fp_acc[t]:.4f},{a_map[t]:.4f}\n")

    cov_edges = [float(v) for v in args.cov_bins.split(",")]
    dev_edges = [float(v) for v in args.dev_bins.split(",")]
    rows_c = bin_curve(cov, fp_acc, cov_edges); rows_cp = bin_curve(cov, pix_acc, cov_edges)
    rows_d = bin_curve(dev, fp_acc, dev_edges) if np.isfinite(dev).any() else []
    with open(args.out / "bins.csv", "w") as fh:
        fh.write("axis,bin_lo,bin_hi,n,fp_trav_acc,pix_acc\n")
        for (lo, hi, n, a), (_, _, _, ap_) in zip(rows_c, rows_cp):
            fh.write(f"c,{lo},{hi},{n},{a:.4f},{ap_:.4f}\n")
        for lo, hi, n, a in rows_d:
            fh.write(f"d,{lo},{hi},{n},{a:.4f},nan\n")

    tau, tau_term = threshold_from_curve(rows_c, args.a_min, args.a_collapse)
    d_ok = None
    if rows_d:
        ok = [hi for lo, hi, n, a in rows_d if n > 0 and np.isfinite(a) and a >= args.a_min]
        d_ok = max(ok) if ok else None
    summary = {
        "frames": int(T),
        "pix_acc_mean": float(np.nanmean(pix_acc)), "fp_trav_acc_mean": float(np.nanmean(fp_acc)),
        "a_map_mean": float(np.nanmean(a_map)) if np.isfinite(a_map).any() else None,
        "iou_per_class": {int(c): (None if not np.isfinite(iou[c]) else float(iou[c])) for c in range(args.num_classes)},
        "tau_from_curve": tau, "tau_term_from_curve": tau_term, "delta_q_from_curve": d_ok,
        "a_min": args.a_min, "a_collapse": args.a_collapse,
    }
    with open(args.out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2 if rows_d else 1, figsize=(11 if rows_d else 6, 4))
        ax = np.atleast_1d(ax)
        xc = [(lo + min(hi, 1.0)) / 2 for lo, hi, n, a in rows_c]
        ax[0].plot(xc, [a for *_, a in rows_c], "o-", label="footprint trav-vs-not")
        ax[0].plot(xc, [a for *_, a in rows_cp], "s--", label="pixel accuracy")
        ax[0].axhline(args.a_min, color="r", lw=0.8); ax[0].set_xlabel("coverage c"); ax[0].set_ylabel("accuracy")
        ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3); ax[0].legend(); ax[0].set_title("A(c)")
        if rows_d:
            xd = [lo for lo, hi, n, a in rows_d]
            ax[1].plot(xd, [a for *_, a in rows_d], "o-")
            ax[1].axhline(args.a_min, color="r", lw=0.8); ax[1].set_xlabel("pose deviation d (Mahalanobis)")
            ax[1].set_ylim(0, 1); ax[1].grid(alpha=0.3); ax[1].set_title("A(d)")
        fig.savefig(args.out / "curves.png", dpi=140, bbox_inches="tight")
    except Exception as e:
        print(f"[grader] no plot ({e})")

    print(f"[grader] {T} frames | pix acc {np.nanmean(pix_acc):.3f} | footprint trav acc "
          f"{np.nanmean(fp_acc):.3f} | tau {tau} tau_term {tau_term} delta_q {d_ok}")
    print(f"[grader] wrote {args.out}/frames.csv, bins.csv, curves.png, summary.json")


if __name__ == "__main__":
    main()
