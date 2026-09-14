"""Withheld-frame protocol, step 1 (CPU): build subset clips of a scene with
some frames held out, so the held-out frames become off-walk poses with a REAL
reference image.

From <clips>/<scene>.mp4 (+ _odom.npz, + <labels>/<scene>.npz SAM3 labels) it writes,
per pattern, a new scene the standard pipeline can reconstruct:
  <out>/<scene>_wh_<pattern>.mp4          kept frames only
  <out>/<scene>_wh_<pattern>_odom.npz     t, xyz, quat unchanged; frame_stamps = kept
  <out>/<scene>_wh_<pattern>.npz          SAM3 labels of the kept frames (no SAM3 rerun)
  <out>/<scene>_wh_<pattern>_withheld.json  withheld frame indices, their odometry pose
                                            (x, y, yaw) and their SAM3 label index
Patterns (--patterns): "every4" drops every 4th frame (single-frame gaps: high
coverage), "win5", "win10", "win20" drop windows of that many frames around
--centers (coverage falls with the window). Then on Marlowe: extract_poses on
each subset mp4, and withheld_render.py.

    python scripts/make_withheld_clips.py --scene campusA_00 --clips_dir clips \
        --labels_dir labels --out clips_withheld [--patterns every4 win5 win10 win20]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_frames(p: Path):
    import cv2
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def write_frames(p: Path, frames, fps=16):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def withheld_set(T: int, pattern: str, centers):
    if pattern.startswith("every"):
        k = int(pattern[5:])
        return sorted(set(range(k // 2, T, k)))
    if pattern.startswith("win"):
        n = int(pattern[3:])
        s = set()
        for c in centers:
            s.update(range(max(0, c - n // 2), min(T, c - n // 2 + n)))
        return sorted(s)
    raise SystemExit(f"unknown pattern {pattern}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True, type=Path)
    ap.add_argument("--labels_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--patterns", nargs="+", default=["every4", "win5", "win10", "win20"])
    ap.add_argument("--centers", type=int, nargs="+", default=[20, 40, 60])
    args = ap.parse_args()

    frames = read_frames(args.clips_dir / f"{args.scene}.mp4")
    T = len(frames)
    od = np.load(args.clips_dir / f"{args.scene}_odom.npz")
    fs = np.asarray(od["frame_stamps"]) if "frame_stamps" in od else np.linspace(od["t"][0], od["t"][-1], T)
    labels = np.load(args.labels_dir / f"{args.scene}.npz")["labels"]
    assert len(fs) == T and len(labels) >= T, (len(fs), len(labels), T)
    t, xyz, quat = np.asarray(od["t"]), np.asarray(od["xyz"]), np.asarray(od["quat"])
    yaw = np.unwrap(np.array([yaw_of(q) for q in quat]))
    args.out.mkdir(parents=True, exist_ok=True)

    for pat in args.patterns:
        wh = withheld_set(T, pat, args.centers)
        keep = [i for i in range(T) if i not in set(wh)]
        # NeoVerse reconstructs 4k+1 frames; trim the kept list to the largest such
        # count so the subset clip is used whole (no duplicated frames from resampling)
        k41 = 4 * ((len(keep) - 1) // 4) + 1
        keep = keep[:k41]
        stem = args.out / f"{args.scene}_wh_{pat}"
        write_frames(Path(f"{stem}.mp4"), [frames[i] for i in keep])
        np.savez_compressed(f"{stem}_odom.npz", t=t, xyz=xyz, quat=quat, frame_stamps=fs[keep])
        np.savez_compressed(f"{stem}.npz", labels=labels[keep])
        poses = []
        for i in wh:
            poses.append({"frame": int(i), "t": float(fs[i]),
                          "x": float(np.interp(fs[i], t, xyz[:, 0])),
                          "y": float(np.interp(fs[i], t, xyz[:, 1])),
                          "yaw": float(np.interp(fs[i], t, yaw))})
        with open(f"{stem}_withheld.json", "w") as fh:
            json.dump({"scene": args.scene, "pattern": pat, "kept": keep, "num_frames": len(keep), "withheld": poses,
                       "full_labels": str(args.labels_dir / f"{args.scene}.npz"),
                       "full_clip": str(args.clips_dir / f"{args.scene}.mp4")}, fh, indent=1)
        print(f"[withheld] {pat}: kept {len(keep)} frames (use --num_frames {len(keep)} downstream), withheld {len(wh)} -> {stem}.mp4")
    print("next: extract_poses.py --videos <out>/*_wh_*.mp4 ..., then withheld_render.py")


if __name__ == "__main__":
    main()
