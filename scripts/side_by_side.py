"""One video, two policies: blind on the left, sighted on the right.

Both halves of an eval pair run under the same `--eval_seed`, so episode N is
the SAME spawn and the SAME goal in both. Watching them in separate windows
loses exactly the thing worth seeing -- WHERE the two diverge, and what is on
screen at that moment. Side by side, the divergence is a single frame.

Episodes end at different lengths (a policy that reaches the goal in 7 steps
against one that times out at 90), so the shorter side freezes on its last
frame, dimmed, with its outcome held on screen rather than the video simply
stopping.

Writes with imageio-ffmpeg: OpenCV in this environment has no H.264 encoder and
its mp4v files play as a green screen in QuickTime (2026-09-02).

    python scripts/side_by_side.py --blind <eval_dir> --sighted <eval_dir> \
        --out_dir /scratch/.../side_by_side --episodes 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_mp4(path: Path):
    import imageio.v3 as iio
    return list(iio.imiter(path))


def label_bar(w: int, text: str, rgb, h: int = 26):
    """A solid strip with the text drawn on it, using PIL so no font file is
    needed beyond the default."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), tuple(int(c * 0.22) for c in rgb))
    ImageDraw.Draw(img).text((6, 6), text, fill=tuple(rgb))
    return np.asarray(img)


def outcomes(run: Path):
    try:
        d = json.loads((run / "metrics.json").read_text())
    except Exception:
        return {}
    return {int(e["episode"]): (e["outcome"], e["steps"]) for e in d["episodes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True)
    ap.add_argument("--sighted", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    bdir, sdir = Path(args.blind), Path(args.sighted)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    b_out, s_out = outcomes(bdir), outcomes(sdir)

    import imageio.v3 as iio
    made = 0
    for ep in range(args.episodes):
        bp, sp = bdir / f"episode_{ep}.mp4", sdir / f"episode_{ep}.mp4"
        if not (bp.exists() and sp.exists()):
            continue
        bf, sf = read_mp4(bp), read_mp4(sp)
        if not bf or not sf:
            continue
        h = min(bf[0].shape[0], sf[0].shape[0])
        bf = [f[:h] for f in bf]
        sf = [f[:h] for f in sf]
        n = max(len(bf), len(sf))

        bo, bs = b_out.get(ep, ("?", len(bf)))
        so, ss = s_out.get(ep, ("?", len(sf)))
        bbar = label_bar(bf[0].shape[1], f"BLIND    {bo}  ({bs} steps)", (255, 176, 90))
        sbar = label_bar(sf[0].shape[1], f"SIGHTED  {so}  ({ss} steps)", (120, 190, 255))

        frames = []
        for i in range(n):
            # the shorter side freezes, dimmed, so "it already finished" is
            # visible rather than the panel just going away
            l = bf[i] if i < len(bf) else (bf[-1] * 0.45).astype(np.uint8)
            r = sf[i] if i < len(sf) else (sf[-1] * 0.45).astype(np.uint8)
            w = min(l.shape[1], r.shape[1])
            pair = np.hstack([l[:, :w], r[:, :w]])
            bar = np.hstack([bbar[:, :w], sbar[:, :w]])
            frames.append(np.vstack([bar, pair]))

        dst = out / f"episode_{ep}_blind_vs_sighted.mp4"
        iio.imwrite(dst, np.stack(frames), fps=args.fps, codec="libx264",
                    pixelformat="yuv420p", macro_block_size=1)
        print(f"==> {dst}   blind {bo} {bs}st | sighted {so} {ss}st")
        made += 1
    if not made:
        raise SystemExit("no matching episode_N.mp4 pairs found in both dirs")


if __name__ == "__main__":
    main()
