"""Pano vs forward-only, as pictures instead of a coverage number.

The cov curve says pano LOSES 8-12 points at the flanks on gnd_AUpano01 — the
exact region it was built to improve. A number that surprising deserves to be
looked at: at matched sweep offsets, is the pano render visibly degraded, or
is the coverage drop invisible in the image?

Both spins visit identical offsets in the same order (same START, FRAMES,
SPINDEG), so frame i in one video corresponds to frame i in the other, and the
per-frame offset comes from the slurm log's `off ... cov ...` lines.

Usage:
    python scripts/viz_pano_compare.py --out pano_vs_fwd.png \
        --at -85,-60,-25,0,25,60,85 \
        fwd=/scratch/.../drive_preview_fwdonly/DRIVE_x.mp4:/scratch/.../slurm-drive-prev-460611.out \
        pano=/scratch/.../drive_preview_pano/DRIVE_x.mp4:/scratch/.../slurm-drive-prev-460646.out
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"off\s+([-+][\d.]+)deg.*?cov\s+([\d.]+)%")


def log_series(p: Path):
    """[(offset, cov)] in frame order — one entry per rendered frame."""
    out = []
    for ln in p.read_text(errors="ignore").splitlines():
        m = LINE.search(ln)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=video.mp4:slurm.out")
    ap.add_argument("--at", default="-85,-60,-25,0,25,60,85",
                    help="sweep offsets (deg) to show as columns")
    ap.add_argument("--width", type=int, default=340)
    ap.add_argument("--out", default="pano_vs_fwd.png")
    ap.add_argument("--per_offset_dir", default="",
                    help="also write ONE FULL-SIZE image per offset here — the "
                         "combined grid shrinks each tile too far to judge "
                         "whether a render is actually degraded")
    args = ap.parse_args()

    import cv2
    targets = [float(v) for v in args.at.split(",")]
    rows, labels = [], []
    full = {}          # offset -> [(label, native-resolution frame, cov)]

    for spec in args.runs:
        label, _, rest = spec.partition("=")
        vid, _, log = rest.partition(":")
        series = log_series(Path(log))
        if not series:
            print(f"[skip] {label}: no cov lines in {log}")
            continue
        cap = cv2.VideoCapture(vid)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            print(f"[skip] {label}: no frames in {vid}")
            continue
        n = min(len(frames), len(series))
        offs = np.array([s[0] for s in series[:n]])

        tiles = []
        for t in targets:
            i = int(np.argmin(np.abs(offs - t)))
            fr = frames[i]
            full.setdefault(t, []).append((label, fr.copy(), series[i][1],
                                           offs[i]))
            h = int(fr.shape[0] * args.width / fr.shape[1])
            fr = cv2.resize(fr, (args.width, h))
            bar = np.zeros((22, args.width, 3), dtype=np.uint8)
            cv2.putText(bar, f"{offs[i]:+.0f}deg  cov {series[i][1]:.1f}%",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, fr]))
        rows.append(np.hstack(tiles))
        labels.append(f"{label}  ({n} frames)")
        print(f"{label:<10} matched {len(targets)} offsets from {n} frames")

    if args.per_offset_dir:
        od = Path(args.per_offset_dir)
        od.mkdir(parents=True, exist_ok=True)
        for t, items in sorted(full.items()):
            stack = []
            for lab, fr, cv_, off in items:
                bar = np.zeros((30, fr.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, f"{lab}   {off:+.0f}deg   cov {cv_:.1f}%",
                            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                stack += [bar, fr]
            cv2.imwrite(str(od / f"OFF{t:+04.0f}.png"), np.vstack(stack))
        print(f"==> {len(full)} full-size offset images: {od}/OFF*.png")

    if not rows:
        raise SystemExit("nothing to draw — check the video and log paths")

    w = max(r.shape[1] for r in rows)
    out = []
    for r, lab in zip(rows, labels):
        if r.shape[1] < w:
            r = np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)))
        hdr = np.zeros((26, w, 3), dtype=np.uint8)
        cv2.putText(hdr, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out += [hdr, r]
    cv2.imwrite(args.out, np.vstack(out))
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
