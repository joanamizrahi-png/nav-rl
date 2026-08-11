"""Pairwise RGB metrics between two renders of the same camera path.

Used for the vanilla-vs-finetune comparison: how far does the finetune's RGB
drift from the vanilla pass, per frame, along an off-trajectory motion?
Reports PSNR and SSIM per frame (start / middle / end thirds) — the ramp
trajectories reach max offset at the last frame, so the end-third number is
the off-path story and the start-third is the on-path sanity check.

Usage:
  python rgb_divergence_metric.py <a.mp4> <b.mp4> [--label name]
"""
import argparse

import cv2
import numpy as np


def read(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
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
    args = ap.parse_args()

    fa, fb = read(args.video_a), read(args.video_b)
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
