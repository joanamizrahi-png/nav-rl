"""Grade an offset sweep (CPU): generated labels vs SAM3 run on the GENERATED
image, vs the rasterized hint, as a function of lateral offset, turn angle and
coverage. Writes curves, a table and comparison panels.

Inputs:
  --sweep   the offset_sweep.py output dir (queries.csv, semantic_labels.npz,
            hint_labels.npz, alpha.npz, frames/)
  --sam3    npz written by NeoVerse/sam3_precompute_labels.py on sweep/frames
            (labels [N,H,W] with embedded class_names; remapped to v14 by name)

Per query:
  agree_pix   generated label == SAM3(generated) over SAM3's non-void pixels
  agree_fp    inside the footprint box: walkable-vs-not agreement (the number
              the reward would read) between the two
  hint_pix    generated label == hint where the hint has geometry (alpha > 0.5)
  fill        share of pixels where the hint is void (nothing rasterized) --
              what the model had to make up
Outputs in --sweep/grade:
  queries_graded.csv, bins.csv (by lat / yaw / coverage), curves.png,
  summary.json, panel_lat.png / panel_yaw.png (one base pose, a few offsets:
  generated | SAM3 on generated | generated labels | hint | coverage)

    python scripts/grade_offsets.py --sweep out/offsets/quad2_01 \
        --sam3 NeoVerse/outputs/sam3_labels/frames.npz [--sem_palette 6]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from label_acc_vs_coverage import footprint_mask, resize_nn  # noqa: E402

DEFAULT_TRAV = [2, 4, 6, 7, 8, 9]


def _taxonomy():
    p = REPO_ROOT.parent / "NeoVerse" / "diffsynth" / "utils" / "class_taxonomy.py"
    spec = importlib.util.spec_from_file_location("class_taxonomy", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--sam3", required=True, type=Path)
    ap.add_argument("--sem_palette", type=int, default=6)
    ap.add_argument("--trav", default=",".join(map(str, DEFAULT_TRAV)))
    ap.add_argument("--footprint", default="0.62,0.98,0.30,0.70")
    ap.add_argument("--cov_bins", default="0,0.2,0.4,0.6,0.8,1.01")
    args = ap.parse_args()

    q = [ln.strip().split(",") for ln in open(args.sweep / "queries.csv") if ln.strip()][1:]
    base = np.array([int(r[1]) for r in q]); kind = np.array([r[2] for r in q])
    off = np.array([float(r[3]) for r in q]); sgn = np.array([int(r[4]) for r in q])
    cov = np.array([float(r[5]) for r in q])
    pred = np.load(args.sweep / "semantic_labels.npz")["labels"].astype(np.int64)
    hint = np.load(args.sweep / "hint_labels.npz")["labels"].astype(np.int64)
    alpha = np.load(args.sweep / "alpha.npz")["alpha"].astype(np.float32)
    if alpha.ndim == 4:
        alpha = alpha[..., 0]
    d = np.load(args.sam3, allow_pickle=True)
    tax = _taxonomy()
    lut = tax.remap_array_from_names(list(d["class_names"]))
    sam = lut[np.asarray(d["labels"]).astype(np.int64)]
    N = min(len(pred), len(sam), len(q))
    pred, hint, alpha, sam = pred[:N], hint[:N], alpha[:N], sam[:N]
    hw = pred.shape[-2:]
    sam = resize_nn(sam, hw); hint = resize_nn(hint, hw); alpha = resize_nn(alpha, hw)
    trav = np.zeros(14, bool); trav[[int(v) for v in args.trav.split(",")]] = True
    fp = footprint_mask(hw[0], hw[1], args.footprint)

    agree_pix = np.full(N, np.nan); agree_fp = np.full(N, np.nan); hint_pix = np.full(N, np.nan); fill = np.full(N, np.nan)
    align = np.full(N, np.nan)          # mean |generated - rasterized| colour on observed pixels, in [0,1]
    import cv2
    for t in range(N):
        _g = cv2.imread(str(args.sweep / "frames" / f"q_{t:04d}.png")); _r = cv2.imread(str(args.sweep / "raster" / f"q_{t:04d}.png"))
        if _g is not None and _r is not None:
            _obs = cv2.resize(alpha[t], (_g.shape[1], _g.shape[0])) > 0.5
            if _obs.sum() > 20:
                align[t] = float(np.abs(_g.astype(np.float32) - cv2.resize(_r, (_g.shape[1], _g.shape[0])).astype(np.float32)).mean(-1)[_obs].mean() / 255.0)
        v = sam[t] != 0
        if v.any():
            agree_pix[t] = float((pred[t][v] == sam[t][v]).mean())
        vf = v & fp
        if vf.any():
            agree_fp[t] = float((trav[np.clip(pred[t][vf], 0, 13)] == trav[np.clip(sam[t][vf], 0, 13)]).mean())
        cvd = (alpha[t] > 0.5) & (hint[t] != 0)
        if cvd.sum() > 20:
            hint_pix[t] = float((pred[t][cvd] == hint[t][cvd]).mean())
        fill[t] = float((hint[t] == 0).mean())

    out = args.sweep / "grade"; out.mkdir(exist_ok=True)
    with open(out / "queries_graded.csv", "w") as fh:
        fh.write("idx,base_frame,kind,offset,sign,coverage,fill,agree_pix,agree_fp,hint_pix,align\n")
        for t in range(N):
            fh.write(f"{t},{base[t]},{kind[t]},{off[t]},{sgn[t]},{cov[t]:.4f},{fill[t]:.4f},"
                     f"{agree_pix[t]:.4f},{agree_fp[t]:.4f},{hint_pix[t]:.4f},{align[t]:.4f}\n")

    def _bin(mask_rows, label):
        rows = []
        for lab, m in mask_rows:
            m = m & np.isfinite(agree_fp)
            rows.append((label, lab, int(m.sum()),
                         float(np.nanmean(agree_fp[m])) if m.any() else np.nan,
                         float(np.nanmean(agree_pix[m])) if m.any() else np.nan,
                         float(np.nanmean(hint_pix[m])) if m.any() else np.nan,
                         float(np.nanmean(cov[m])) if m.any() else np.nan,
                         float(np.nanmean(align[m])) if m.any() and np.isfinite(align[m]).any() else np.nan))
        return rows
    lat_vals = sorted(set(off[kind == "lat"])); yaw_vals = [0.0] + sorted(set(off[kind == "yaw"]))
    rows = _bin([(f"{o:.1f}", (kind == "lat") & (off == o)) for o in lat_vals], "lat")
    rows += _bin([(f"{o:.0f}", ((kind == "yaw") & (off == o)) | ((kind == "lat") & (off == 0.0) & (o == 0.0)))
                  for o in yaw_vals], "yaw")
    edges = [float(v) for v in args.cov_bins.split(",")]
    rows += _bin([(f"{lo:.1f}-{min(hi, 1.0):.1f}", (cov >= lo) & (cov < hi)) for lo, hi in zip(edges[:-1], edges[1:])], "cov")
    with open(out / "bins.csv", "w") as fh:
        fh.write("axis,bin,n,agree_fp,agree_pix,hint_pix,coverage,align\n")
        for r in rows:
            fh.write(",".join(str(v) if not isinstance(v, float) else f"{v:.4f}" for v in r) + "\n")

    summary = {"queries": int(N), "agree_fp_mean": float(np.nanmean(agree_fp)),
               "align_mean": float(np.nanmean(align)) if np.isfinite(align).any() else None,
               "agree_pix_mean": float(np.nanmean(agree_pix)), "hint_pix_mean": float(np.nanmean(hint_pix)),
               "coverage_range": [float(cov.min()), float(cov.max())]}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for k, (axis, xs) in enumerate((("lat", lat_vals), ("yaw", yaw_vals), ("cov", None))):
            rr = [r for r in rows if r[0] == axis]
            x = xs if xs is not None else [float(r[6]) for r in rr]
            ax[k].plot(x, [r[3] for r in rr], "o-", label="footprint walkable-vs-not (gen vs SAM3 on gen)")
            ax[k].plot(x, [r[4] for r in rr], "s--", label="pixel agreement (gen vs SAM3 on gen)")
            ax[k].plot(x, [r[5] for r in rr], "^:", label="gen vs hint, covered pixels")
            ax[k].plot(x, [1.0 - r[7] if np.isfinite(r[7]) else np.nan for r in rr], "d-.", label="1 - colour alignment error (gen vs raster)")
            ax[k].set_ylim(0, 1); ax[k].grid(alpha=0.3)
            ax[k].set_xlabel({"lat": "lateral offset [m]", "yaw": "turn [deg]", "cov": "coverage"}[axis])
            if axis == "cov":
                ax2 = ax[k].twinx(); ax2.bar(x, [r[2] for r in rr], width=0.15, alpha=0.15, color="gray"); ax2.set_ylabel("n")
        ax[0].set_ylabel("agreement"); ax[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(args.sweep.name)
        fig.savefig(out / "curves.png", dpi=140, bbox_inches="tight")
    except Exception as e:
        print(f"[grade_offsets] no curves ({e})")

    # panels: one base pose (the middle one), a few offsets per axis
    try:
        import cv2
        from src.eval.palette import display_palette
        pal = display_palette(int(args.sem_palette))
        H, W = hw
        def _row(t, tag):
            g = cv2.imread(str(args.sweep / "frames" / f"q_{t:04d}.png"))[:, :, ::-1]
            g = cv2.resize(g, (W, H))
            s_ = pal[np.clip(sam[t], 0, 13)]; p_ = pal[np.clip(pred[t], 0, 13)]; h_ = pal[np.clip(hint[t], 0, 13)]
            c_ = cv2.applyColorMap((np.clip(alpha[t], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            _rr = cv2.imread(str(args.sweep / "raster" / f"q_{t:04d}.png"))
            _rr = cv2.resize(_rr, (W, H))[:, :, ::-1] if _rr is not None else np.zeros_like(g)
            r = np.ascontiguousarray(np.concatenate([g, _rr, s_, p_, h_, c_], axis=1).astype(np.uint8))
            for k, nm in enumerate(["GENERATED image", "rasterized colour (hint)", "SAM3 on the generated image",
                                    "GENERATED labels", "rasterized hint labels", "coverage (dark = invented)"]):
                cv2.putText(r, nm, (k * W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(r, nm, (k * W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(r, f"{tag}  coverage {cov[t]:.2f}  footprint agree {agree_fp[t]:.2f}  pixel agree {agree_pix[t]:.2f}  align err {align[t]:.2f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            return r
        bases = sorted(set(base)); b0 = bases[len(bases) // 2]
        for axis, avail in (("lat", lat_vals), ("yaw", yaw_vals)):
            # up to 4 offsets spread over what was actually rendered (a short
            # probe has 0..0.4 m; the full sweep 0..2 m), always including 0
            avail = sorted(set(float(v) for v in avail))
            picks = [avail[int(round(i * (len(avail) - 1) / 3))] for i in range(4)] if len(avail) > 4 else avail
            rws = []
            for o in picks:
                m = (base == b0) & (((kind == axis) & (off == o)) | ((kind == "lat") & (off == 0.0) & (o == 0.0)))
                idx = np.nonzero(m)[0]
                if len(idx):
                    t = int(idx[0]); rws.append(_row(t, f"base frame {b0}  {axis} {sgn[t] * o:+.1f}"))
            if rws:
                cv2.imwrite(str(out / f"panel_{axis}.png"), np.concatenate(rws, axis=0)[:, :, ::-1])
        print(f"[grade_offsets] panels for base frame {b0}: {out}/panel_lat.png, panel_yaw.png")
    except Exception as e:
        print(f"[grade_offsets] panels skipped ({type(e).__name__}: {e})")

    print(f"[grade_offsets] {args.sweep.name}: {N} queries | footprint agree {summary['agree_fp_mean']:.3f} | "
          f"pixel agree {summary['agree_pix_mean']:.3f} | gen-vs-hint {summary['hint_pix_mean']:.3f} | "
          f"coverage {cov.min():.2f}-{cov.max():.2f}")
    print("axis  bin     n   fp_agree  pix_agree  hint_agree  coverage  align_err")
    for r in rows:
        print(f"{r[0]:<5} {r[1]:<7} {r[2]:3d}   {r[3]:.3f}     {r[4]:.3f}      {r[5]:.3f}      {r[6]:.2f}      {r[7]:.3f}")


if __name__ == "__main__":
    main()
