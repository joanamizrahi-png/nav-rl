"""Extract REAL per-frame robot trajectories from NeoVerse's reconstructor.

Milestone A / TODO item: "Extract REAL trajectories from NeoVerse reconstructor
per clip". Replaces validate_reward.py's synthetic straight-line trajectory with
the path the camera actually took — which for RUGD IS the robot's ground-truth
trajectory (the clip was recorded from the robot).

Runs on Marlowe (needs GPU + reconstructor checkpoint). One reconstruction per
clip (~1-2 min each on an H100 after the one-time model load); no diffusion.

Per clip it saves `<stem>_poses.npz` with exactly the keys load_clip's
`pose_source="npz"` branch expects:

    positions  (T, 3)  robot feet-on-ground positions, meters, z-up world
    headings   (T, 3)  robot forward unit vectors, horizontal
    w2c        (T, 4, 4) world -> camera, consistent with `positions`' frame
    K          (3, 3)  intrinsics from the reconstructor (not the pinhole guess)

plus diagnostics (c2w, raw camera positions, fitted plane, scale factor).

The geometry, step by step (conventions verified in src/env/real_backend.py):

  1. Reconstructor outputs camera-to-world poses in ITS frame: x=right,
     y=DOWN, z=forward — and in NORMALIZED per-clip units, not meters.
  2. Rotate into the eval/scene frame (x=fwd, y=right, z=up) with the fixed
     R_SCENE_TO_RECON^T from real_backend (single source of truth).
  3. That only makes z *approximately* up (exact if the frame-0 camera was
     level). So we RANSAC-fit a ground plane to the Gaussian means
     (src/eval/ground_plane.py) and rotate the world so the plane normal is
     exactly +z, with the ground at z=0.
  4. Metric scale: the camera's median height above the fitted plane, in recon
     units, must equal the physical mount height (--camera_height_m; RUGD was
     recorded on a Clearpath Husky, camera ~0.6 m up). scale = mount_m / median.
     Scaling world points AND c2w translations by the same factor leaves pixel
     projections bit-identical, so the reward's meter-based footprint
     (GO2 body 0.6x0.3 m, look_ahead 0.5 m) becomes meaningful without
     changing what projects where.
  5. Robot position = camera position dropped to the ground (z=0); robot
     heading = camera forward projected to horizontal.

Usage (on Marlowe):
    python scripts/extract_poses.py \
        --videos /scratch/m000204-pm06b/joana/data/rugd_clips/*.mp4 \
        --output_dir /scratch/m000204-pm06b/joana/outputs/poses

Then (on Mac, after scp):
    python scripts/validate_reward.py --video ... --labels ... \
        --pose_source npz --poses_npz outputs/poses/<stem>_poses.npz ...
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

from src.env.real_backend import R_SCENE_TO_RECON
from src.eval.ground_plane import fit_ground_plane_ransac, GroundPlane


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy — unit-testable on Mac without torch)
# ---------------------------------------------------------------------------

def rotation_aligning_to_z(normal: np.ndarray) -> np.ndarray:
    """Rodrigues rotation taking `normal` (unit, up-ish) exactly onto +z."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(n @ z, -1.0, 1.0))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:                      # anti-parallel: flip around x
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(n, z)
    axis /= np.linalg.norm(axis)
    s = np.sqrt(1.0 - c * c)
    Kx = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + s * Kx + (1.0 - c) * (Kx @ Kx)


def poses_from_c2w_recon(
    c2w_recon: np.ndarray,          # (T, 4, 4) camera-to-world, recon frame, recon units
    gaussian_means_recon: np.ndarray,   # (N, 3) recon frame, recon units
    camera_height_m: float,
) -> dict:
    """Recon-frame camera poses -> z-up, ground-at-z0, metric-scale robot poses.

    Returns dict with positions/headings/w2c/c2w plus diagnostics. Pure numpy.
    """
    T = len(c2w_recon)
    R_rs = R_SCENE_TO_RECON.T.astype(np.float64)     # recon -> scene(z-up-ish)

    # --- step 2: fixed rotation into the approx z-up scene frame ---
    c2w = c2w_recon.astype(np.float64).copy()
    c2w[:, :3, :3] = R_rs @ c2w[:, :3, :3]
    c2w[:, :3, 3] = c2w[:, :3, 3] @ R_rs.T
    means = gaussian_means_recon.astype(np.float64) @ R_rs.T

    # --- step 3: exact gravity alignment from a RANSAC ground plane ---
    # inlier_thresh is documented as meters but we're still in recon units here,
    # so make it scale-relative: 1% of the scene's vertical extent.
    extent = float(np.percentile(means[:, 2], 98) - np.percentile(means[:, 2], 2))
    plane: GroundPlane = fit_ground_plane_ransac(
        means, inlier_thresh=max(1e-6, 0.01 * extent))
    R_up = rotation_aligning_to_z(plane.normal)
    c2w[:, :3, :3] = R_up @ c2w[:, :3, :3]
    c2w[:, :3, 3] = c2w[:, :3, 3] @ R_up.T
    means = means @ R_up.T
    # After R_up the plane is horizontal at z = -offset (normal@p+offset=0 with
    # normal now +z). Shift so the ground sits at z=0.
    ground_z = -plane.offset
    c2w[:, 2, 3] -= ground_z
    means[:, 2] -= ground_z

    # --- step 4: metric scale from camera mount height ---
    cam_heights = c2w[:, 2, 3]
    h_median = float(np.median(cam_heights))
    if h_median <= 0:
        raise RuntimeError(
            f"median camera height above fitted ground is {h_median:.4f} <= 0; "
            "the plane fit or the recon->scene rotation is wrong for this clip. "
            "Inspect the saved diagnostics before trusting anything.")
    scale = camera_height_m / h_median
    c2w[:, :3, 3] *= scale
    means *= scale

    # --- step 5: robot poses on the ground ---
    cam_pos = c2w[:, :3, 3].copy()
    positions = cam_pos.copy()
    positions[:, 2] = 0.0
    fwd = c2w[:, :3, 2].copy()                        # camera z-axis in world
    fwd[:, 2] = 0.0                                    # horizontal projection
    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    if (norms < 1e-3).any():
        raise RuntimeError("near-vertical camera in some frame; heading undefined")
    headings = fwd / norms

    w2c = np.zeros_like(c2w)
    for t in range(T):
        R = c2w[t, :3, :3]
        p = c2w[t, :3, 3]
        w2c[t, :3, :3] = R.T
        w2c[t, :3, 3] = -R.T @ p
        w2c[t, 3, 3] = 1.0

    step_sizes = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    return {
        "positions": positions.astype(np.float32),
        "headings": headings.astype(np.float32),
        "w2c": w2c.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "cam_positions": cam_pos.astype(np.float32),
        "scale_m_per_unit": np.float32(scale),
        "camera_height_units_median": np.float32(h_median),
        "plane_normal_scene": plane.normal.astype(np.float32),
        "plane_offset_scene": np.float32(plane.offset),
        "step_sizes_m": step_sizes.astype(np.float32),
    }


def _extract_gaussian_means(splats) -> np.ndarray:
    """Pull (N, 3) position means out of the reconstructor's splats structure,
    tolerating dict / attribute / per-frame-list layouts."""
    import torch

    def _to_np(x):
        return x.detach().float().cpu().numpy().reshape(-1, 3)

    for key in ("means", "means3d", "xyz", "positions"):
        val = None
        if isinstance(splats, dict) and key in splats:
            val = splats[key]
        elif hasattr(splats, key):
            val = getattr(splats, key)
        if val is None:
            continue
        if torch.is_tensor(val):
            return _to_np(val)
        if isinstance(val, (list, tuple)) and len(val) and torch.is_tensor(val[0]):
            return np.concatenate([_to_np(v) for v in val], axis=0)
    raise RuntimeError(
        f"couldn't find gaussian means; splats type={type(splats)}, "
        f"keys/attrs={list(splats.keys()) if isinstance(splats, dict) else dir(splats)}")


# ---------------------------------------------------------------------------
# GPU side: run the reconstructor (mirrors real_backend._reconstruct_scene)
# ---------------------------------------------------------------------------

def load_reconstructor(reconstructor_path: str):
    import torch
    from diffsynth.utils import ModelConfig
    from diffsynth.models import ModelManager

    print(f"[extract_poses] loading reconstructor from {reconstructor_path} ...", flush=True)
    mm = ModelManager()
    cfg = ModelConfig(path=reconstructor_path, offload_device="cuda")
    cfg.download_if_necessary()
    mm.load_model(cfg.path, device="cuda", torch_dtype=torch.bfloat16)
    return mm.fetch_model("reconstructor")


def reconstruct_clip(reconstructor, video_path: Path, num_frames: int, width: int, height: int):
    import torch
    from torchvision.transforms import functional as F
    from diffsynth.utils.auxiliary import load_video

    device = next(reconstructor.parameters()).device
    dtype = next(reconstructor.parameters()).dtype

    # Same loader + center_crop as SAM3 labeling and load_clip, so frame i here
    # is frame i everywhere.
    images = load_video(str(video_path), num_frames,
                        resolution=(width, height), resize_mode="center_crop",
                        static_scene=False)
    views = {
        "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
        "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
    }
    # no_grad is CRITICAL — see real_backend._reconstruct_scene (activation
    # memory otherwise blows past VRAM before anything can be reclaimed).
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = reconstructor(views, is_inference=True, use_motion=False)

    c2w = pred["rendered_extrinsics"][0].detach().float().cpu().numpy()   # (T,4,4)
    K_all = pred["rendered_intrinsics"][0].detach().float().cpu().numpy()  # (T,3,3)
    means = _extract_gaussian_means(pred["splats"])

    del pred, views
    torch.cuda.empty_cache()
    return c2w, K_all, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, type=Path,
                    help="one or more clip MP4s (shell glob ok)")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--camera_height_m", type=float, default=0.6,
                    help="physical camera mount height used ONLY for metric scale "
                         "(RUGD: Clearpath Husky, ~0.6 m). Getting this off by 20%% "
                         "scales the footprint/look-ahead by 20%% — mild, not fatal.")
    ap.add_argument("--max_gaussians_for_plane", type=int, default=200_000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reconstructor = load_reconstructor(args.reconstructor_path)

    rng = np.random.default_rng(0)
    for video in args.videos:
        print(f"\n[extract_poses] === {video.name} ===", flush=True)
        c2w_recon, K_all, means = reconstruct_clip(
            reconstructor, video, args.num_frames, args.width, args.height)
        print(f"[extract_poses] {len(c2w_recon)} poses, {len(means)} gaussians", flush=True)

        if len(means) > args.max_gaussians_for_plane:
            means = means[rng.choice(len(means), args.max_gaussians_for_plane, replace=False)]

        out = poses_from_c2w_recon(c2w_recon, means, camera_height_m=args.camera_height_m)
        out["K"] = K_all[0].astype(np.float32)
        out["K_all"] = K_all.astype(np.float32)
        out["video"] = str(video)
        out["camera_height_m"] = np.float32(args.camera_height_m)

        # Sanity numbers for eyeballing before trusting the npz downstream.
        steps = out["step_sizes_m"]
        length_m = float(steps.sum())
        print(f"[extract_poses] scale = {float(out['scale_m_per_unit']):.4f} m/unit "
              f"(median cam height {float(out['camera_height_units_median']):.4f} units)", flush=True)
        print(f"[extract_poses] trajectory: {length_m:.1f} m total, "
              f"step {steps.mean():.3f} m/frame (min {steps.min():.3f}, max {steps.max():.3f})", flush=True)
        cam_z = out["cam_positions"][:, 2]
        print(f"[extract_poses] camera height over ground: "
              f"mean {cam_z.mean():.2f} m, std {cam_z.std():.3f} m "
              f"(std >> 0.1 m means bad plane fit or non-flat terrain)", flush=True)

        out_path = args.output_dir / f"{video.stem}_poses.npz"
        np.savez_compressed(out_path, **out)
        print(f"[extract_poses] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
