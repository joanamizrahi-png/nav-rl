"""Generate a legend PNG showing each class ID, name, color swatch, and score.

Usage:
    python scripts/make_legend.py --out outputs/legend.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.palette import CLASS_COLORS_255
from src.eval.traversability import load_traversability, load_class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/legend.png"))
    args = ap.parse_args()

    scores = load_traversability()
    names = load_class_names()
    n = len(CLASS_COLORS_255)

    # ---- layout ----
    swatch_size = 30
    row_h = 36
    padding = 12
    text_offset = swatch_size + 12
    row_w = 380       # per column
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols

    W = row_w * n_cols + padding * 2
    H = row_h * n_rows + padding * 2 + 40   # +40 for title

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    img = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((padding, padding), "Class legend  (id | color | name | traversability score)",
              fill=(30, 30, 30), font=title_font)

    for cid in range(n):
        col = cid // n_rows
        row = cid % n_rows
        x0 = padding + col * row_w
        y0 = padding + 40 + row * row_h

        color = tuple(int(v) for v in CLASS_COLORS_255[cid])
        # Swatch
        draw.rectangle([x0, y0, x0 + swatch_size, y0 + swatch_size], fill=color, outline=(50, 50, 50))
        # Label
        text = f"{cid:>2}  {names[cid]:<14s}  score={scores[cid]:.2f}"
        # Text color: gray if score <= 0.1 (collision-worthy) for at-a-glance flagging
        text_color = (170, 30, 30) if scores[cid] <= 0.1 else (30, 30, 30)
        draw.text((x0 + text_offset, y0 + 8), text, fill=text_color, font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"[legend] wrote {args.out}")


if __name__ == "__main__":
    main()
