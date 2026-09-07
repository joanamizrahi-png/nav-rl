"""Line up the GENERATED SEMANTICS of several replay runs on the same frame.

Each replay2_* folder holds REPLAY_<scene>_ep<k>_s<step>.png = four panels
(diffused RGB | generated semantics | raster | alpha), one per label head.
This takes the semantics panel from each folder and puts them side by side,
with the first folder's diffused RGB at the left for context and a palette
legend at the bottom. Joana, 2026-09-06: "what I'm interested in with the
later replays is not their RGB but their semantics".

    python scripts/replay_semantics_panel.py --base <eval dir> \
        --folders replay2_v26e10,replay2_v26be10,replay2_v26be20,replay2_v28be20,replay2_v29e5 \
        --out <eval dir>/semantics_side
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.palette import display_palette, CLASS_COLORS_V14_255, CLASS_NAMES_V14 as CLASS_NAMES  # noqa: E402


def recolor_exact(tile_bgr: np.ndarray, src_pal_rgb: np.ndarray, dst_pal_rgb: np.ndarray) -> np.ndarray:
    """Recolor pixels that EXACTLY match a source palette color to the target
    palette (PNG is lossless, so label pixels match exactly); every other
    pixel (the projected map dots, text) is left untouched."""
    out = tile_bgr.copy()
    for cid in range(min(len(src_pal_rgb), len(dst_pal_rgb))):
        sb = np.array([src_pal_rgb[cid][2], src_pal_rgb[cid][1], src_pal_rgb[cid][0]], np.uint8)
        m = np.all(tile_bgr == sb[None, None, :], axis=2)
        out[m] = (int(dst_pal_rgb[cid][2]), int(dst_pal_rgb[cid][1]), int(dst_pal_rgb[cid][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--folders", required=True, help="comma list of replay folders under base, first = reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=4, help="panels per replay image")
    ap.add_argument("--sem_panel", type=int, default=1, help="index of the semantics panel")
    ap.add_argument("--raster_panel", type=int, default=2, help="index of the raster (original render) panel")
    ap.add_argument("--sem_palette", type=int, default=4, help="palette version for the legend (must match the replays)")
    ap.add_argument("--recolor", action="store_true", help="replays rendered with the static v14 palette: recolor label pixels to --sem_palette")
    args = ap.parse_args()
    import cv2
    base = Path(args.base); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    ref = base / folders[0]
    names = sorted(p.name for p in ref.glob("REPLAY_*_s[0-9][0-9].png"))
    for n in names:
        tiles = []
        img0 = cv2.imread(str(ref / n))
        if img0 is None:
            continue
        W = img0.shape[1] // args.panels
        rgb = img0[:, :W].copy()
        cv2.putText(rgb, f"RGB diffused ({folders[0]})", (8, rgb.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(rgb)
        ras = img0[:, args.raster_panel * W:(args.raster_panel + 1) * W].copy()
        cv2.putText(ras, "RGB raster (reconstruction)", (8, ras.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(ras)
        for f in folders:
            img = cv2.imread(str(base / f / n))
            if img is None:
                tiles.append(np.zeros_like(rgb)); continue
            sem = img[:, args.sem_panel * W:(args.sem_panel + 1) * W].copy()
            if args.recolor:
                sem = recolor_exact(sem, CLASS_COLORS_V14_255, display_palette(args.sem_palette))
            cv2.rectangle(sem, (0, sem.shape[0] - 26), (sem.shape[1], sem.shape[0]), (0, 0, 0), -1)
            cv2.putText(sem, f.replace("replay2_", ""), (8, sem.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(sem)
        row = np.concatenate(tiles, axis=1)
        # legend strip
        leg = np.full((34, row.shape[1], 3), 30, np.uint8)
        x = 8
        for cid, (nm, col) in enumerate(zip(CLASS_NAMES, display_palette(args.sem_palette))):
            bgr = (int(col[2]), int(col[1]), int(col[0]))
            cv2.rectangle(leg, (x, 8), (x + 18, 26), bgr, -1)
            cv2.putText(leg, f"{cid} {nm}", (x + 22, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            x += 22 + 10 * len(f"{cid} {nm}") + 12
        panel = np.concatenate([row, leg], axis=0)
        cv2.imwrite(str(out / n.replace("REPLAY_", "SEM_")), panel)
    print(f"{len(names)} frames -> {out}")


if __name__ == "__main__":
    main()
