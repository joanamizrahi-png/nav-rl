"""Pairwise RGB metrics between two videos of the same camera path.

Two uses:
1. DRIFT (default): finetune render vs vanilla render, same trajectory —
   how far apart are the two models' images? Reports per-thirds (the ramp
   trajectories reach max offset at the last frame, so the end third is the
   off-path story, the start third the on-path sanity check).
2. FIDELITY (--match): render vs the ORIGINAL camera clip — how faithful is
   a model to reality on-path? --match center-crops+resizes video A to
   video B's size (the pipeline's own preprocessing), so pass the original
   clip as A and the render as B. Absolute numbers are only meaningful in
   comparison (e.g. vanilla-vs-real against v10-vs-real; parity measured
   2026-08-12: trail 13.2/13.7 dB, park 19.2/18.7 dB -> one-pass viable).

Usage:
  python rgb_divergence_metric.py <a.mp4> <b.mp4> [--label name] [--match]
"""
import argparse

import cv2
import numpy as np


def read(path, match=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if match is not None:
            th, tw = match
            h, w = f.shape[:2]
            ar = tw / th
            ch = int(w / ar)
            if ch <= h:                      # crop height, keep width
                y0 = (h - ch) // 2
                f = f[y0:y0 + ch]
            else:                            # crop width, keep height
                cw = int(h * ar)
                x0 = (w - cw) // 2
                f = f[:, x0:x0 + cw]
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        frames.append(f.astype(np.float64))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {path}")
    return frames


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    # single-scale grayscale SSIM, 11x11 gaussian window
    a = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    vab = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--label", default="")
    ap.add_argument("--match", action="store_true",
                    help="center-crop+resize A to B's size (pass the original "
                         "clip as A, the render as B) — the FIDELITY mode")
    args = ap.parse_args()

    fb = read(args.video_b)
    fa = read(args.video_a, match=fb[0].shape[:2] if args.match else None)
    n = min(len(fa), len(fb))
    ps = [psnr(fa[i], fb[i]) for i in range(n)]
    ss = [ssim(fa[i], fb[i]) for i in range(n)]
    third = max(1, n // 3)

    def seg(vals):
        return (np.mean(vals[:third]), np.mean(vals[third:2 * third]), np.mean(vals[2 * third:]))

    p1, p2, p3 = seg(ps)
    s1, s2, s3 = seg(ss)
    tag = args.label or "pair"
    print(f"{tag}: {n} frames")
    print(f"  PSNR  start/mid/end thirds: {p1:5.2f} / {p2:5.2f} / {p3:5.2f} dB")
    print(f"  SSIM  start/mid/end thirds: {s1:.4f} / {s2:.4f} / {s3:.4f}")


if __name__ == "__main__":
    main()
