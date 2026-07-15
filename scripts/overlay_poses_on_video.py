"""Overlay the reconstructor's estimated poses on top of source RGB frames.

Produces one MP4 per clip that shows:
  * The RGB source frame
  * A text banner with (x, y, z) position + heading vector for the current frame
  * A top-down mini-map in the corner showing the trajectory so far, with a
    green dot at the current position and an arrow for current heading

Lets you visually verify: does the reconstructor's per-frame pose track how
the camera actually moves in the source video?

Usage:
    python scripts/overlay_poses_on_video.py \
        --clips rugd_trail_00 rugd_creek_00 rugd_park-1_00 cityscapes_stuttgart_00_00

Outputs:
    outputs/pose_overlay/<clip>_poses_overlay.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio


def _load_video_frames(path: Path, num_frames: int = 81) -> np.ndarray:
    """Load `num_frames` sampled uniformly from `path`. Returns (T, H, W, 3) uint8."""
    all_frames = list(iio.imiter(str(path)))
    idxs = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    sampled = [all_frames[i] for i in idxs]
    # Center-crop to 560x336 to match label / pose resolution used elsewhere.
    from src.eval.load_clip import _center_crop_frames
    return _center_crop_frames(np.stack(sampled, axis=0), 560, 336)


def _draw_trajectory_minimap(
    positions: np.ndarray, cur_frame_idx: int, size: int = 140
) -> Image.Image:
    """Return a `size x size` top-down mini-map with the trajectory drawn.

    Uses x vs z (since x-z projection is where most motion happens per our diagnostic).
    """
    img = Image.new("RGBA", (size, size), (30, 30, 30, 220))
    d = ImageDraw.Draw(img)

    # Choose the two axes with the largest ranges — usually x and z.
    ranges = positions.max(axis=0) - positions.min(axis=0)
    top2 = np.argsort(ranges)[-2:]        # indices of two largest-range axes
    ax_a, ax_b = int(top2[0]), int(top2[1])

    px = positions[:, ax_a]
    py = positions[:, ax_b]
    # Normalize to [8, size-8] with a bit of padding.
    def _norm(vs, target_lo=8, target_hi=size - 8):
        lo, hi = vs.min(), vs.max()
        if hi - lo < 1e-6:
            return np.full_like(vs, size // 2)
        return target_lo + (vs - lo) * (target_hi - target_lo) / (hi - lo)

    xs = _norm(px)
    ys = _norm(py)

    # Trail (past): light gray
    for i in range(1, cur_frame_idx + 1):
        d.line([(xs[i - 1], ys[i - 1]), (xs[i], ys[i])], fill=(180, 180, 180, 255), width=1)
    # Future: darker gray
    for i in range(cur_frame_idx + 1, len(xs)):
        d.line([(xs[i - 1], ys[i - 1]), (xs[i], ys[i])], fill=(90, 90, 90, 220), width=1)

    # Current position: green dot
    x, y = float(xs[cur_frame_idx]), float(ys[cur_frame_idx])
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 220, 0, 255), outline=(0, 0, 0, 255))
    # Start dot: yellow
    d.ellipse((xs[0] - 3, ys[0] - 3, xs[0] + 3, ys[0] + 3), fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))

    # Axis labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except Exception:
        font = ImageFont.load_default()
    axis_names = ["x", "y", "z"]
    d.text((2, size - 12), f"{axis_names[ax_a]}→", fill=(220, 220, 220, 255), font=font)
    d.text((2, 2), f"{axis_names[ax_b]}↑", fill=(220, 220, 220, 255), font=font)
    return img


def _overlay_one_frame(
    rgb: np.ndarray, positions: np.ndarray, headings: np.ndarray, frame_idx: int
) -> Image.Image:
    """Overlay text banner + mini-map on one frame. Returns PIL Image."""
    H, W = rgb.shape[:2]
    banner_h = 46
    canvas = Image.new("RGB", (W, H + banner_h), (0, 0, 0))
    canvas.paste(Image.fromarray(rgb), (0, banner_h))

    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    # Distance travelled from origin
    dist_from_origin = float(np.linalg.norm(positions[frame_idx] - positions[0]))
    # Distance travelled step-to-step (cumulative)
    if frame_idx > 0:
        cum_dist = float(np.linalg.norm(np.diff(positions[:frame_idx + 1], axis=0), axis=1).sum())
    else:
        cum_dist = 0.0

    p = positions[frame_idx]
    h = headings[frame_idx]
    line1 = (f"frame {frame_idx:3d}   pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})   "
             f"heading=({h[0]:+.2f},{h[1]:+.2f},{h[2]:+.2f})")
    line2 = (f"dist from start={dist_from_origin:.3f} recon-units   "
             f"cumulative path length={cum_dist:.3f}")
    d.text((6, 4), line1, fill=(255, 255, 255), font=font)
    d.text((6, 24), line2, fill=(200, 200, 200), font=font)

    # Mini-map in the top-right corner of the RGB region.
    mini = _draw_trajectory_minimap(positions, frame_idx)
    canvas.paste(mini, (W - mini.width - 8, banner_h + 8), mini)
    return canvas


def process_clip(clip: str, data_dir: Path, out_dir: Path) -> None:
    video_path = data_dir / f"{clip}.mp4"
    poses_path = data_dir / f"{clip}_poses.npz"
    if not video_path.exists():
        print(f"[overlay] SKIP {clip}: no video at {video_path}")
        return
    if not poses_path.exists():
        print(f"[overlay] SKIP {clip}: no poses at {poses_path}")
        return

    frames = _load_video_frames(video_path)
    poses = np.load(poses_path)
    positions = poses["positions"]
    headings = poses["headings"]

    T = min(len(frames), len(positions))
    out_frames = []
    for t in range(T):
        img = _overlay_one_frame(frames[t], positions, headings, t)
        out_frames.append(np.array(img))

    out_path = out_dir / f"{clip}_poses_overlay.mp4"
    iio.imwrite(str(out_path), np.stack(out_frames, axis=0), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[overlay] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+",
                    default=["rugd_trail_00", "rugd_creek_00", "rugd_park-1_00", "cityscapes_stuttgart_00_00"])
    ap.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "pose_overlay")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        process_clip(clip, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
