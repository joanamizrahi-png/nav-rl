"""Dump a scene's Gaussian cloud to npz for Mac-side 3D viz + Test A (ground shape).

What it does, in plain terms:
  1. Rebuilds the scene exactly like the RL env does (CalibratedRealWorldBackend:
     reconstruct the clip -> Gaussians with SAM3 labels attached).
  2. Pulls out each Gaussian's 3D center (its "mean"), plus its semantic label
     and color where available.
  3. Converts the centers from the reconstructor's normalized frame into our
     metric nav frame (meters, z-up, ground~0) with the SAME calibration the
     env uses — so the cloud, the trajectory, and the reward all live in one
     frame and can be overlaid directly.
  4. Subsamples (the full cloud is ~15M points; 400k is plenty for viz/stats)
     and saves ONE npz per scene next to the poses files.

Runs on Marlowe (GPU). ~30 s per scene on a warm node.

Usage:
    python scripts/dump_scene_cloud.py \
        --scenes rugd_trail_00 rugd_creek_00 rugd_park-1_00 rugd_park-2_00 \
        --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
        --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
        --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels \
        --out_dir /scratch/m000204-pm06b/joana/outputs/scene_clouds
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import CalibratedRealWorldBackend, CalibratedBackendConfig


def collect_gaussian_fields(splats):
    """Walk the nested splats structure (list[batch] -> list[Gaussians]) and pull
    per-gaussian arrays: means (N,3) + labels (N,) + colors (N,3) where present.
    Defensive: prints what it finds; missing fields come back as None."""
    import torch

    means, labels, colors = [], [], []

    def _np(x):
        return x.detach().float().cpu().numpy()

    def _collect(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
            return
        m = getattr(node, "means", None)
        if not torch.is_tensor(m):
            return
        n = m.shape[0]
        means.append(_np(m).reshape(-1, 3))
        lab = getattr(node, "labels", None)
        if torch.is_tensor(lab) and lab.shape[0] == n:
            l = _np(lab)
            labels.append(l.argmax(-1) if l.ndim == 2 else l)   # one-hot or ids
        else:
            labels.append(np.full(n, -1, dtype=np.int16))
        sh = getattr(node, "sh", None) if getattr(node, "sh", None) is not None \
            else getattr(node, "harmonics", None)
        if torch.is_tensor(sh):
            s = _np(sh).reshape(n, -1, 3)
            colors.append(np.clip(0.2820948 * s[:, 0, :] + 0.5, 0, 1))  # SH DC term
        else:
            colors.append(np.full((n, 3), 0.5, dtype=np.float32))

    _collect(splats)
    if not means:
        raise RuntimeError(f"no means found in splats structure {type(splats)}")
    return (np.concatenate(means), np.concatenate(labels).astype(np.int16),
            np.concatenate(colors).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_points", type=int, default=400_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in args.scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in args.scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in args.scenes},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    rng = np.random.default_rng(0)

    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        world.load_scene(scene)
        cache = world._cache[scene]
        cal = world._calib[scene]

        means_recon, labels, colors = collect_gaussian_fields(cache["gaussians"])
        print(f"  {len(means_recon)} gaussians "
              f"(labels present: {(labels >= 0).mean():.0%})", flush=True)

        # recon -> metric nav frame: same map the env/reward uses.
        means_nav = (cal.A @ means_recon.T).T
        means_nav[:, 2] -= cal.ground_z
        means_nav *= cal.scale

        if len(means_nav) > args.max_points:
            idx = rng.choice(len(means_nav), args.max_points, replace=False)
            means_nav, labels, colors = means_nav[idx], labels[idx], colors[idx]

        d = np.load(f"{args.poses_dir}/{scene}_poses.npz")
        out = args.out_dir / f"{scene}_cloud.npz"
        np.savez_compressed(
            out, points=means_nav.astype(np.float32), labels=labels,
            colors=(colors * 255).astype(np.uint8),
            traj_positions=d["positions"], traj_cam_z=d["cam_positions"][:, 2],
            camera_height_m=np.float32(0.6),
        )
        print(f"  wrote {out}", flush=True)
        # Free the scene before the next one (each holds ~GBs on GPU).
        world._cache.pop(scene, None)
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
