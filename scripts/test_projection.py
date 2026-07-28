"""Test B — the next-step projection test: project the robot's NEXT position into the
current image and check it lands well. Runs on the Mac (no GPU, no cluster).

For each frame t of the real trajectory, find the future frame j ~2 m ahead
(the robot's actual next position — we know it, the robot drove it) and draw
its footprint into frame t's image TWICE:

  RED    = assuming flat ground (z=0)              <- what the reward uses today
  YELLOW = using the TRUE local ground height       <- from the trajectory itself
           (camera z at j minus mount height)
  CYAN   = using the GAUSSIAN-estimated ground      <- p40 of splat heights within
           (needs outputs/scene_clouds/clouds/)        0.75 m; the locked estimator

The vertical pixel gap between the two IS the projection error caused by the
flat-ground assumption — the projection inaccuracy we need to handle. Output:
an overlay MP4 per scene + per-scene offset statistics.

Usage: python scripts/test_projection.py [scene ...]   (default: the 3 local scenes)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.load_clip import load_clip
from src.eval.reward_2d import (
    _footprint_corners_world, _project_points, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)

LOOK_AHEAD_M = 2.0
CAMERA_HEIGHT_M = 0.25   # RUGD paper Sec III-A: viewpoint <25 cm off ground
GROUND_PCT = 40.0        # locked 2026-07-23 (sweep: mean 2.7 cm, creek cliff beyond 40)
GROUND_RADIUS = 0.75


def future_index(positions: np.ndarray, t: int, dist_m: float) -> int:
    d = np.linalg.norm(positions[t:, :2] - positions[t, :2], axis=1)
    ahead = np.nonzero(d >= dist_m)[0]
    return t + int(ahead[0]) if len(ahead) else -1


def run_scene(scene: str):
    from PIL import Image, ImageDraw
    import imageio.v3 as iio

    clip = load_clip(
        video_path=REPO_ROOT / f"data/{scene}.mp4",
        labels_path=(REPO_ROOT / f"data/{scene}.npz"
                     if (REPO_ROOT / f"data/{scene}.npz").exists() else None),
        pose_source="npz",
        poses_npz_path=REPO_ROOT / f"data/poses/{scene}_poses.npz",
    )
    d = np.load(REPO_ROOT / f"data/poses/{scene}_poses.npz")
    cam_z = d["cam_positions"][:, 2]                # camera height above plane per frame
    T = len(clip.frames_rgb)

    # Gaussian-ground estimator (same as the locked Test A setting).
    ground_est = None
    cloud_path = REPO_ROOT / f"outputs/scene_clouds/clouds/{scene}_cloud.npz"
    if cloud_path.exists():
        from scipy.spatial import cKDTree
        cpts = np.load(cloud_path)["points"]
        rng = np.random.default_rng(0)
        if len(cpts) > 150_000:
            cpts = cpts[rng.choice(len(cpts), 150_000, replace=False)]
        ctree = cKDTree(cpts[:, :2])

        def ground_est(xy):
            idx = ctree.query_ball_point(xy, GROUND_RADIUS)
            return float(np.percentile(cpts[idx, 2], GROUND_PCT)) if len(idx) >= 20 else None

    frames, offsets, gauss_offsets = [], [], []
    for t in range(T):
        j = future_index(clip.positions, t, LOOK_AHEAD_M)
        img = Image.fromarray(clip.frames_rgb[t].copy())
        draw = ImageDraw.Draw(img, "RGBA")
        if j > 0:
            pos_flat = clip.positions[j].copy(); pos_flat[2] = 0.0
            true_ground_z = float(cam_z[j] - CAMERA_HEIGHT_M)
            pos_true = clip.positions[j].copy(); pos_true[2] = true_ground_z
            heading_j = clip.headings[j]

            variants = [(pos_flat, (255, 60, 60, 255)),
                        (pos_true, (255, 220, 40, 255))]
            if ground_est is not None:
                gz = ground_est(clip.positions[j][:2])
                if gz is not None:
                    pos_gauss = clip.positions[j].copy(); pos_gauss[2] = gz
                    variants.append((pos_gauss, (60, 220, 255, 255)))

            uv_pair = []
            for pos, color in variants:
                corners = _footprint_corners_world(
                    pos, heading_j, look_ahead_dist=0.0,
                    length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
                uv, in_front = _project_points(corners, clip.K, clip.w2c[t])
                if in_front.all():
                    poly = [(float(u), float(v)) for u, v in uv]
                    for i in range(4):
                        draw.line([poly[i], poly[(i + 1) % 4]], fill=color, width=2)
                    uv_pair.append(uv.mean(0))
            if len(uv_pair) >= 2:
                offsets.append(float(np.linalg.norm(uv_pair[0] - uv_pair[1])))
            if len(uv_pair) == 3:
                gauss_offsets.append(float(np.linalg.norm(uv_pair[2] - uv_pair[1])))

        draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0, 180))
        off_txt = f"{offsets[-1]:.0f}px" if offsets else "n/a"
        gauss_txt = f"{gauss_offsets[-1]:.0f}px" if gauss_offsets else "n/a"
        draw.text((4, 2), f"{scene} t={t:2d}  RED=flat  YELLOW=true  CYAN=gaussian-p40  "
                          f"flat-err={off_txt}  gauss-err={gauss_txt}",
                  fill=(255, 255, 255, 255))
        frames.append(np.array(img.convert("RGB")))

    out = REPO_ROOT / f"outputs/projection_test/{scene}_projection.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out), np.stack(frames), fps=8, codec="libx264",
                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    arr = np.array(offsets)
    print(f"{scene}: wrote {out}")
    print(f"  flat-vs-true offset:     median {np.median(arr):.0f}px, "
          f"p90 {np.percentile(arr, 90):.0f}px, max {arr.max():.0f}px")
    if gauss_offsets:
        g = np.array(gauss_offsets)
        print(f"  gaussian-vs-true offset: median {np.median(g):.0f}px, "
              f"p90 {np.percentile(g, 90):.0f}px, max {g.max():.0f}px")
    return arr


if __name__ == "__main__":
    scenes = sys.argv[1:] or ["rugd_trail_00", "rugd_creek_00", "rugd_park-1_00"]
    for s in scenes:
        run_scene(s)
