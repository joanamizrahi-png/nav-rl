"""Pose -> image direction probe (advisor meeting item, 2026-08-10).

Question: when the RL env constructs a robot pose and renders it, does the
image actually face the direction the pose claims? The policy could silently
learn around a consistent rotation offset, and replay validation only covers
RECORDED camera poses, not the env's constructed robot poses.

Probe: spawn on the recorded trajectory, then render
  1. the spawn pose,
  2. the pose yawed +30 deg (image content must shift RIGHT: camera turning
     left brings scene content from the left, so existing content moves right),
  3. the pose yawed -30 deg (content shifts left),
  4. the pose stepped 1.0 m along its heading (content must expand/zoom,
     centered on the horizon point we are walking toward).

Output: one labeled PNG strip per scene + printed cross-correlation shifts
between the yawed renders and the base render. PASS = measured horizontal
shift matches the sign and rough magnitude implied by the intrinsics
(shift_px ~ fx * tan(30 deg)).

Usage (cluster):
  python scripts/pose_direction_check.py --scene rugd_trail_00 \
      --clips_dir ... --poses_dir ... --labels_dir ... \
      --model_path ... --reconstructor_path ... --out_dir outputs/pose_check
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

import numpy as np

from src.env.real_calibrated import CalibratedRealWorldBackend, CalibratedBackendConfig


def yawed(pose: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 4x4 robot pose about world +z by deg (matches SceneEnv._advance_pose)."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    out = pose.copy()
    out[:3, :3] = R @ pose[:3, :3]
    return out


def stepped(pose: np.ndarray, dist: float) -> np.ndarray:
    """Advance a 4x4 robot pose along its local +x heading (SceneEnv convention)."""
    out = pose.copy()
    fwd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[:3, 3] = pose[:3, 3] + dist * fwd
    return out


def h_shift(a: np.ndarray, b: np.ndarray, max_px: int = 350) -> tuple[int, float]:
    """Horizontal shift of b vs a: masked column-profile correlation.

    Holes (black, no geometry) dominate off-path renders and fool a naive
    profile correlation (v1 of this probe returned +0 px on a panel that had
    visibly shifted). So: use the central row band only, treat near-black as
    missing, and correlate mean-brightness profiles computed over CONTENT
    pixels; the correlation for a candidate shift only counts columns where
    both profiles have content. Returns (shift_px, content_fraction_of_b).
    """
    H = a.shape[0]
    band = slice(H // 4, 3 * H // 4)
    def profile(img):
        band_img = img[band].astype(np.float64)
        content = band_img.sum(axis=2) > 30.0          # non-hole pixels
        counts = content.sum(axis=0).astype(np.float64)  # per column
        sums = (band_img.mean(axis=2) * content).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return p
    pa, pb = profile(a), profile(b)
    frac_b = float(np.mean(~np.isnan(pb)))
    best, best_v = 0, -np.inf
    for s in range(-max_px, max_px + 1):
        if s >= 0:
            xa, xb = pa[s:], pb[:len(pb) - s]
        else:
            xa, xb = pa[:s], pb[-s:]
        ok = ~(np.isnan(xa) | np.isnan(xb))
        if ok.sum() < 40:                              # too little overlap
            continue
        va, vb = xa[ok] - xa[ok].mean(), xb[ok] - xb[ok].mean()
        denom = np.sqrt((va * va).sum() * (vb * vb).sum())
        if denom <= 0:
            continue
        v = float((va * vb).sum() / denom)             # normalized correlation
        if v > best_v:
            best_v, best = v, s
    # Sign convention: the loop's positive s means b matches a shifted LEFT;
    # negate so the returned value reads naturally (positive = content moved
    # RIGHT, which is what a LEFT turn produces) and matches the printed
    # expected signs.
    return -best, frac_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--spawn_frame", type=int, default=10)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--reconstructor_path", required=True)
    ap.add_argument("--out_dir", default="outputs/pose_check")
    args = ap.parse_args()

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    world.load_scene(args.scene)

    # Use the env's OWN pose construction — that is exactly what's under test.
    cal = world._calib[args.scene]
    base = np.asarray(cal.robot_pose_nav(args.spawn_frame), dtype=np.float32)

    poses = {
        "base": base,
        "yaw+30 (turn left)": yawed(base, +30),
        "yaw-30 (turn right)": yawed(base, -30),
        "forward 1.0m": stepped(base, 1.0),
    }
    renders = {}
    for name, p in poses.items():
        rgb, K, _ = world.render(p)
        renders[name] = np.asarray(rgb)
    fx = float(K[0, 0]) if K is not None else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio
    try:
        import cv2
        def put(img, txt):
            img = np.ascontiguousarray(img.copy())
            cv2.rectangle(img, (0, 0), (10 + 9 * len(txt), 22), (0, 0, 0), -1)
            cv2.putText(img, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return img
    except ImportError:
        def put(img, txt):
            return img
    strip = np.hstack([put(renders[k], k) for k in poses])
    iio.imwrite(out / f"pose_direction_{args.scene}.png", strip)

    # Sweep VIDEOS — the eyeball-proof version. Static frames are ambiguous
    # (repeating forest texture fools shift measurement; holes fool eyes);
    # continuous motion is not: as the label counts the yaw LEFT (+), scene
    # content must flow RIGHT, smoothly. Any flip is instantly visible.
    yaw_frames = []
    for deg in np.linspace(-30, 30, 61):
        rgb, _, _ = world.render(yawed(base, float(deg)))
        arrow = "turning LEFT" if deg > 1 else ("turning RIGHT" if deg < -1 else "center")
        yaw_frames.append(put(np.asarray(rgb), f"yaw {deg:+.0f} deg  ({arrow})  -> content should flow RIGHT as yaw increases"))
    iio.imwrite(out / f"yaw_sweep_{args.scene}.mp4", np.stack(yaw_frames), fps=10,
                codec="libx264", macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])

    fwd_frames = []
    for dist in np.linspace(0.0, 1.5, 46):
        rgb, _, _ = world.render(stepped(base, float(dist)))
        fwd_frames.append(put(np.asarray(rgb), f"forward {dist:+.2f} m  -> scene should approach, centered"))
    iio.imwrite(out / f"forward_sweep_{args.scene}.mp4", np.stack(fwd_frames), fps=10,
                codec="libx264", macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"sweep videos: {out}/yaw_sweep_{args.scene}.mp4, {out}/forward_sweep_{args.scene}.mp4")

    print(f"=== pose->image direction probe, {args.scene}, frame {args.spawn_frame} ===")
    expected = fx * np.tan(np.deg2rad(30)) if fx else None
    for name in ["yaw+30 (turn left)", "yaw-30 (turn right)"]:
        s, frac = h_shift(renders["base"], renders[name])
        want = "+" if "+30" in name else "-"
        note = " [LOW CONTENT — treat with care]" if frac < 0.35 else ""
        exp = f", |expected| ~{expected:.0f} px" if expected else ""
        print(f"{name}: shift {s:+d} px (content {frac:.0%}{exp}, expected sign {want}){note}")
    print(f"strip: {out / f'pose_direction_{args.scene}.png'}")
    print("PASS if: yaw+30 shift is positive, yaw-30 negative, magnitudes near "
          "expected, and the forward panel looks like walking 1 m up the trail.")


if __name__ == "__main__":
    main()
