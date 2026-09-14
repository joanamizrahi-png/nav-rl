"""Semantic sheet: real frame | SAM3 labels | raster | generated | generated labels,
for a handful of frames of one scene, on one PNG (and optionally one mp4).

This is the "look before deciding" artifact: how SAM3 fares on the campus
frames, how the reconstruction looks, and how the semantic model completes it.
(scene_sheet.py is the older overhead-map contact sheet; this one is the
semantics view.)

Inputs:
  --clip     <scene>.mp4            the walkthrough (81 frames, 560x336)
  --sam3     <scene>.npz            SAM3 labels for those frames (key `labels`)
  --run      inference output dir   with rough_rgb.mp4 (raster), rgb.mp4 (generated),
                                    semantic_labels.npz (generated labels); any missing
                                    column is left blank so the sheet still builds
  --frames   indices to show (default 6 evenly spaced)
  --out      sheet png; --mp4 also writes the full-length strip as a video

    python scripts/semantic_sheet.py --clip clips/campusA_00.mp4 --sam3 labels/campusA_00.npz \
        --run outputs/walk_campusA_00 --out sheets/campusA_00.png [--mp4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

V14 = np.array([
    (0, 0, 0), (200, 225, 245), (150, 100, 55), (75, 190, 80), (95, 65, 35),
    (50, 120, 200), (210, 210, 210), (70, 70, 85), (235, 205, 150),
    (220, 140, 80), (185, 55, 50), (170, 200, 55), (205, 70, 145), (110, 130, 220),
], np.uint8)


def read_video(path):
    if path is None or not Path(path).exists():
        return None
    import cv2
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])
    cap.release()
    return np.stack(frames) if frames else None


def colorize(lab: np.ndarray) -> np.ndarray:
    return V14[np.clip(lab.astype(np.int64), 0, len(V14) - 1)]


def overlay(rgb: np.ndarray, lab: np.ndarray, a: float = 0.55) -> np.ndarray:
    import cv2
    if lab.shape != rgb.shape[:2]:
        lab = cv2.resize(lab.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    col = colorize(lab).astype(np.float32)
    return (a * col + (1 - a) * rgb.astype(np.float32)).astype(np.uint8)


def fit(img, hw: tuple, label: str) -> np.ndarray:
    import cv2
    H, W = hw
    if img is None:
        tile = np.full((H, W, 3), 40, np.uint8)
        cv2.putText(tile, f"{label}: n/a", (10, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return tile
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    tile = np.ascontiguousarray(img)
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--sam3", required=True, type=Path)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--tile_w", type=int, default=420)
    args = ap.parse_args()
    import cv2

    real = read_video(args.clip)
    if real is None:
        raise SystemExit(f"cannot read {args.clip}")
    T, H0, W0 = real.shape[:3]
    sam3 = np.load(args.sam3)["labels"]
    rast = gen = glab = None
    if args.run:
        rast = read_video(args.run / "rough_rgb.mp4")
        gen = read_video(args.run / "rgb.mp4")
        p = args.run / "semantic_labels.npz"
        glab = np.load(p)["labels"] if p.exists() else None
    tw = args.tile_w; th = int(round(tw * H0 / W0)); hw = (th, tw)
    idx = args.frames if args.frames else np.linspace(0, T - 1, 6).astype(int).tolist()

    def pick(arr, t):
        return None if arr is None else arr[min(t, len(arr) - 1)]

    def row(t):
        t = int(np.clip(t, 0, T - 1))
        g = pick(gen, t); gl = pick(glab, t)
        tiles = [fit(real[t], hw, f"real f{t}"),
                 fit(overlay(real[t], pick(sam3, t)), hw, "SAM3 on real"),
                 fit(pick(rast, t), hw, "raster"),
                 fit(g, hw, "generated"),
                 fit(overlay(g, gl) if (g is not None and gl is not None) else None, hw, "generated labels")]
        return np.concatenate(tiles, 1)

    sheet = np.concatenate([row(t) for t in idx], 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet[:, :, ::-1])
    print(f"[sheet] wrote {args.out} ({len(idx)} frames x 5 columns)")
    if args.mp4:
        vp = args.out.with_suffix(".mp4")
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 8, (5 * tw, th))
        for t in range(T):
            vw.write(row(t)[:, :, ::-1])
        vw.release()
        print(f"[sheet] wrote {vp}")


if __name__ == "__main__":
    main()
