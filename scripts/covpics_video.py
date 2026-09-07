"""Side-by-side video of the coverage-sweep pictures: for each walk frame, the
camera pans through the yaw offsets; the DYNAMIC reconstruction's panel is the
top row and the STATIC one the bottom row (each panel = raster | diffused RGB |
generated semantics | alpha). Joana, 2026-09-07: "static is so much better!
can these pngs be made into a side by side video".

    python scripts/covpics_video.py --dynamic <dir static0> --static <dir static1> \
        --out <mp4> [--hold 1.0]

Frames are held --hold seconds each; re-encoded to H.264 with the ffmpeg on the
path or the one bundled in the neoverse env (QuickTime shows mp4v as green).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds per yaw step")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    import cv2
    pat = re.compile(r"COV_(?P<scene>.+)_(?P<mode>static|dynamic)_f(?P<f>\d+)_yaw(?P<yaw>[+-]\d+)\.png")
    def index(d, mode):
        # both modes may sit in ONE folder (check_rewards writes to
        # covsweep_<scene>/ by default): keep only this mode's files
        items = {}
        for p in Path(d).glob("COV_*.png"):
            m = pat.match(p.name)
            if m and m["mode"] == mode:
                items[(int(m["f"]), int(m["yaw"]))] = p
        return items
    dyn, sta = index(args.dynamic, "dynamic"), index(args.static, "static")
    print(f"dynamic pictures: {len(dyn)}   static pictures: {len(sta)}   shared (frame, yaw): {len(set(dyn) & set(sta))}")
    keys = sorted(set(dyn) & set(sta), key=lambda k: (k[0], k[1]))
    if not keys:
        raise SystemExit("no matching (frame, yaw) pictures in both folders")
    frames = []
    for f in sorted(set(k[0] for k in keys)):
        yaws = sorted(k[1] for k in keys if k[0] == f)
        # pan left -> right and back, so the eye follows the camera
        seq = yaws + yaws[-2:0:-1]
        for y in seq:
            a = cv2.imread(str(dyn[(f, y)])); b = cv2.imread(str(sta[(f, y)]))
            if a is None or b is None:
                continue
            if a.shape[1] != b.shape[1]:
                b = cv2.resize(b, (a.shape[1], int(b.shape[0] * a.shape[1] / b.shape[1])))
            bar = np.full((28, a.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, f"walk frame {f}   yaw {y:+d} deg   top: DYNAMIC (per-frame Gaussians)   bottom: STATIC (all frames)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(np.concatenate([bar, a, b], axis=0))
    if not frames:
        raise SystemExit("no frames")
    H, W = frames[0].shape[:2]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    reps = max(1, int(round(args.hold * args.fps)))
    for fr in frames:
        for _ in range(reps):
            w.write(fr)
    w.release()
    import shutil, subprocess, os
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        tmp = str(out) + ".tmp.mp4"; os.replace(str(out), tmp)
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)])
        if r.returncode == 0 and out.exists():
            os.remove(tmp)
        else:
            os.replace(tmp, str(out)); print("(ffmpeg re-encode failed; kept mp4v)")
    print(f"{len(frames)} panels ({len(keys)} frame/yaw pairs) -> {out}")


if __name__ == "__main__":
    main()
