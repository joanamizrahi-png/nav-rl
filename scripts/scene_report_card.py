"""Per-scene reconstruction report card (VGGT-accuracy check, and the
RL scene-vetting gate for dataset expansion).

Three numbers per scene, all measured against things we actually have:
  1. REPLAY FIDELITY — rasterize at every recorded robot pose and score
     PSNR/SSIM against the real camera frames. "The reconstruction reproduces
     reality at X quality" — the quantitative form of the replay video.
  2. COVERAGE vs OFFSET — fraction of non-hole pixels at the recorded poses
     and at lateral/heading offsets. Quantifies the ribbon per scene (and
     feeds the cache-width choice).
  3. (ground error is reported from the existing scene_clouds/ground eval —
     not recomputed here.)

Verdict line: PASS if replay SSIM >= --min_ssim (default 0.45) and on-path
coverage >= --min_cov (default 0.90). Scenes that fail get a look before
joining RL training.

Usage (cluster, GPU):
  python scripts/scene_report_card.py --scene rugd_trail_00 \
      --clips_dir ... --poses_dir ... --labels_dir ... \
      --model_path ... --reconstructor_path ... --out_dir outputs/report_card
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

import numpy as np

from src.env.real_calibrated import CalibratedRealWorldBackend, CalibratedBackendConfig


def _center_crop_resize(frame: np.ndarray, th: int, tw: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    ar = tw / th
    ch = int(w / ar)
    if ch <= h:
        y0 = (h - ch) // 2
        frame = frame[y0:y0 + ch]
    else:
        cw = int(h * ar)
        x0 = (w - cw) // 2
        frame = frame[:, x0:x0 + cw]
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def _psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def _ssim(a, b):
    import cv2
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, vab = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    return float((((2 * ma * mb + C1) * (2 * vab + C2))
                  / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean())


def _yawed(pose, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    out = pose.copy()
    out[:3, :3] = R @ pose[:3, :3]
    return out


def _lateral(pose, m):
    out = pose.copy()
    out[:3, 3] = pose[:3, 3] + m * pose[:3, 1]      # +y = left (right-handed frame)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--reconstructor_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=4, help="score every Nth frame")
    ap.add_argument("--min_ssim", type=float, default=0.45)
    ap.add_argument("--min_cov", type=float, default=0.90)
    args = ap.parse_args()

    import cv2
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
    cal = world._calib[args.scene]

    cap = cv2.VideoCapture(f"{args.clips_dir}/{args.scene}.mp4")
    real = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        real.append(f[:, :, ::-1])                   # BGR -> RGB
    cap.release()

    # ---- 1. replay fidelity ----
    ps, ss = [], []
    H = W = None
    frames = range(0, min(len(real), len(cal.positions)), args.stride)
    for i in frames:
        rgb, _, _ = world.render(np.asarray(cal.robot_pose_nav(i), dtype=np.float32))
        if H is None:
            H, W = rgb.shape[:2]
        ref = _center_crop_resize(real[i], H, W)
        ps.append(_psnr(ref, rgb))
        ss.append(_ssim(ref, rgb))
    replay_psnr, replay_ssim = float(np.mean(ps)), float(np.mean(ss))

    # ---- 2. coverage vs offset ----
    def coverage(pose):
        rgb, _, _ = world.render(pose.astype(np.float32))
        return float((rgb.sum(axis=2) > 15).mean())

    offsets = {"onpath": lambda p: p,
               "lat+0.5": lambda p: _lateral(p, 0.5), "lat+1.0": lambda p: _lateral(p, 1.0),
               "lat+1.5": lambda p: _lateral(p, 1.5), "lat-1.0": lambda p: _lateral(p, -1.0),
               "yaw+45": lambda p: _yawed(p, 45), "yaw+90": lambda p: _yawed(p, 90),
               "yaw180": lambda p: _yawed(p, 180)}
    cov = {}
    probe_frames = list(range(5, min(len(cal.positions), 76), 14))
    for name, fn in offsets.items():
        cov[name] = float(np.mean([
            coverage(fn(np.asarray(cal.robot_pose_nav(i), dtype=np.float64)))
            for i in probe_frames]))

    verdict = "PASS" if (replay_ssim >= args.min_ssim and cov["onpath"] >= args.min_cov) else "REVIEW"
    card = {"scene": args.scene, "replay_psnr_db": round(replay_psnr, 2),
            "replay_ssim": round(replay_ssim, 4),
            "coverage": {k: round(v, 3) for k, v in cov.items()},
            "frames_scored": len(ps), "verdict": verdict}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.scene}.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"==> {args.scene}: {verdict} (replay {replay_ssim:.3f} SSIM / "
          f"{replay_psnr:.1f} dB, on-path coverage {cov['onpath']:.1%})")


if __name__ == "__main__":
    main()
