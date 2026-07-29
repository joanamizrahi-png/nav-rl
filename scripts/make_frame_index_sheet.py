"""Frame-index contact sheet: which trajectory frame is which, at a glance.

The env samples every clip into 81 trajectory frames (np.linspace over the
video). Goal frames, spawn frames and probe frames all index THAT sequence,
not raw video frames. This sheet shows every 3rd trajectory frame labeled
with its index, so picking e.g. a goal_frame is just reading a number off
the tile you like.

Usage (Mac): python scripts/make_frame_index_sheet.py rugd_trail_00
Output: outputs/goal_maps/<scene>_frame_index_sheet.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = 81
EVERY = 3
COLS = 7


def main():
    from PIL import Image, ImageDraw
    import imageio.v3 as iio

    scene = sys.argv[1] if len(sys.argv) > 1 else "rugd_trail_00"
    clip = REPO_ROOT / f"data/clips/{scene}.mp4"
    frames = list(iio.imiter(str(clip)))
    idxs = np.linspace(0, len(frames) - 1, N_FRAMES, dtype=int)

    picks = list(range(0, N_FRAMES, EVERY))
    W, H = 240, 144
    rows = (len(picks) + COLS - 1) // COLS
    sheet = Image.new("RGB", (W * COLS, (H + 16) * rows), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for n, t in enumerate(picks):
        im = Image.fromarray(frames[idxs[t]]).resize((W, H))
        x, y = (n % COLS) * W, (n // COLS) * (H + 16)
        sheet.paste(im, (x, y + 16))
        dr.text((x + 4, y + 2), f"frame {t}", fill=(0, 255, 0))
    out = REPO_ROOT / f"outputs/goal_maps/{scene}_frame_index_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out} ({len(picks)} tiles, every {EVERY}rd trajectory frame)")


if __name__ == "__main__":
    main()
