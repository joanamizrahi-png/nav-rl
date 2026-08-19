"""Sweep self-coherence reel: play every cached sweep back-to-back.

Why: each sweep is ONE diffusion call, so within-sweep coherence is the
foundation the whole cache stands on. This reel plays all sweeps (subsampled)
in lane/heading order with the sweep name burned in — one sitting to verify
every sweep is coherent with itself, and to spot the rotten ones by name.

CPU-only, streams one mp4 at a time.
    python scripts/cache_sweep_reel.py \
        --cache_root /scratch/.../outputs/ribbon_cache --scene rugd_trail_00 \
        --out /scratch/.../outputs/cache_tour/SWEEPREEL_ribbon_cache.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4,
                    help="every Nth frame of each 81-frame sweep (4 -> ~1s "
                         "per sweep at 20fps)")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import cv2
    scene_dir = Path(args.cache_root) / args.scene
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    names = sorted(sw["file"][:-5] for sw in manifest["sweeps"])

    vw, missing = None, 0
    for si, name in enumerate(names):
        rgb_path = scene_dir / name / "rgb.mp4"
        if not rgb_path.exists():
            missing += 1
            continue
        cap = cv2.VideoCapture(str(rgb_path))
        fi = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if fi % args.stride == 0:
                f = np.ascontiguousarray(f)
                cv2.rectangle(f, (0, 0), (f.shape[1], 20), (0, 0, 0), -1)
                cv2.putText(f, f"[{si + 1}/{len(names)}] {name}  f{fi:02d}",
                            (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if vw is None:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    vw = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps, (f.shape[1], f.shape[0]))
                vw.write(f)
            fi += 1
        cap.release()
        if (si + 1) % 25 == 0:
            print(f"{si + 1}/{len(names)} sweeps", flush=True)
    if vw is not None:
        vw.release()
    print(f"wrote {args.out} ({len(names) - missing} sweeps, "
          f"{missing} missing)")


if __name__ == "__main__":
    main()
