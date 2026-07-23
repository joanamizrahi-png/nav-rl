"""Load a recorded clip + its SAM3 labels + a robot trajectory.

For Milestone A, we don't have ground-truth robot trajectories for RUGD
(RUGD ships as raw image sequences, no ego-motion data). Two options:

1. Synthetic trajectory — assume the camera is a forward-moving robot at
   constant speed. Gives us SOMETHING to plot; not a real evaluation.
2. Reconstructor-derived trajectory — run NeoVerse's reconstructor on the
   clip, save `rendered_extrinsics` per frame, use those as poses. Requires
   a one-time GPU run per clip. This is the proper approach.

`load_clip()` supports either mode via the `pose_source` arg.

Camera intrinsics: we assume a standard pinhole model with the same
resolution as the SAM3 labels (560x336 in our pipeline). If the recorded
clip has metadata specifying intrinsics, we'll wire that in later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Clip:
    """One recorded clip loaded for reward evaluation."""
    name: str
    frames_rgb: np.ndarray             # (T, H, W, 3) uint8
    labels: np.ndarray                 # (T, H, W) int32 - class ids per pixel
    positions: np.ndarray              # (T, 3) float32 - robot position per frame in world frame
    headings: np.ndarray               # (T, 3) float32 - robot forward direction, unit vectors
    K: np.ndarray                      # (3, 3) intrinsics — same for all frames
    w2c: np.ndarray                    # (T, 4, 4) world -> camera per frame
    goal: np.ndarray                   # (3,) inferred goal (end of trajectory) in world


def _center_crop_frames(frames: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop or letterbox each frame to (target_h, target_w). Assumes RGB."""
    T, H, W = frames.shape[:3]
    if H == target_h and W == target_w:
        return frames.astype(np.uint8)
    # Resize (aspect-preserving fit) then crop, matching NeoVerse's center_crop.
    # Cheap path: use PIL per frame.
    from PIL import Image
    out = np.empty((T, target_h, target_w, 3), dtype=np.uint8)
    for i in range(T):
        im = Image.fromarray(frames[i])
        # Scale so smaller of (H/target_h, W/target_w) matches, then center-crop.
        scale = max(target_w / W, target_h / H)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        out[i] = np.array(im.crop((left, top, left + target_w, top + target_h)))
    return out


def _default_intrinsics(W: int, H: int) -> np.ndarray:
    """Assume a mid-FOV pinhole camera with principal point at image center."""
    # ~60 degree horizontal FOV; f = W / (2 * tan(fov/2)) ≈ 0.866 * W
    fx = fy = 0.866 * W
    cx = W / 2.0
    cy = H / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _synthetic_trajectory(
    n_frames: int,
    step_size: float = 0.3,
    camera_height: float = 0.4,       # meters; approximate Go2 / Jackal camera mount
    camera_pitch_down: float = 0.05,  # radians (~3°); RUGD-style near-level camera
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robot walks along +x at `step_size` m/frame; camera mounted above ground.

    Convention: world frame X = forward, Y = right, Z = up. Camera frame
    Z = forward (positive depth), X = right, Y = down (standard pinhole).

    `camera_height` (m) is the mount height above the ground plane. `camera_pitch_down`
    (radians) tilts the camera slightly forward-and-down so the ground ahead
    projects to the LOWER half of the image (not just the horizon line).

    Returns (positions, headings, w2c). positions/headings are the ROBOT's
    (feet on ground); w2c is the CAMERA's transform (mounted at height).
    """
    positions = np.zeros((n_frames, 3), dtype=np.float32)
    positions[:, 0] = np.arange(n_frames) * step_size          # forward along x

    headings = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_frames, 1))

    # Base world -> camera rotation (camera level, looking along +x).
    # cam_x = world_y (right), cam_y = -world_z (down), cam_z = world_x (fwd).
    R_level = np.array([
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0],
    ], dtype=np.float32)

    # Pitch the camera downward by `camera_pitch_down` rad. In the camera frame,
    # pitch-down = rotation around the camera's +x axis (right). Positive angle
    # tilts the camera's z axis downward in world coords.
    c, s = np.cos(camera_pitch_down), np.sin(camera_pitch_down)
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  -s],
        [0.0,  s,   c],
    ], dtype=np.float32)
    R_world_to_cam = R_pitch @ R_level

    w2c = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    w2c[:, :3, :3] = R_world_to_cam
    # Camera position in world = robot position + (0, 0, camera_height).
    # Translation for world->camera: t = -R @ camera_pos_world.
    for i in range(n_frames):
        camera_pos_world = positions[i] + np.array([0.0, 0.0, camera_height], dtype=np.float32)
        w2c[i, :3, 3] = -R_world_to_cam @ camera_pos_world

    return positions, headings, w2c


def load_clip(
    *,
    video_path: Path,
    labels_path: Path,
    pose_source: str = "synthetic",     # "synthetic" or "npz"
    poses_npz_path: Optional[Path] = None,
    num_frames: int = 81,
    intrinsics: Optional[np.ndarray] = None,
) -> Clip:
    """Load a clip's frames, SAM3 labels, and a trajectory.

    Args:
        video_path: path to the MP4.
        labels_path: path to the SAM3 .npz produced by sam3_precompute_labels.py
            (must contain a "labels" array of shape (T, H, W)).
        pose_source: "synthetic" (linear forward motion) or "npz" (poses saved
            from NeoVerse's reconstructor).
        poses_npz_path: only used when pose_source="npz". Expected keys:
            "positions" (T, 3), "headings" (T, 3), optionally "w2c" (T, 4, 4)
            and "K" (3, 3).
        num_frames: how many frames to load (matches SAM3 default).
        intrinsics: optional (3, 3) K matrix. If None, use `_default_intrinsics`.

    Returns:
        A Clip dataclass ready to feed to `reward_2d.compute_reward`.
    """
    # --- frames ---
    # Load via imageio (Mac-friendly; no decord dependency). Then center-crop
    # to 560x336 to match SAM3's label resolution.
    #
    # IMPORTANT: SAM3 sampled frames via np.linspace(0, total-1, num_frames) in
    # sam3_precompute_labels.load_video. We MUST use the same sampling here so
    # RGB frame i corresponds to label frame i. Otherwise labels drift ahead
    # of RGB (SAM3 covers the whole video length; sequential-first-N covers
    # only the initial fraction).
    import imageio.v3 as iio
    all_frames = list(iio.imiter(str(video_path)))
    total_frames = len(all_frames)
    if total_frames < num_frames:
        raise ValueError(f"video has {total_frames} frames, need >= {num_frames}")
    idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled = [all_frames[i] for i in idxs]
    frames_rgb = _center_crop_frames(np.stack(sampled, axis=0), 560, 336)
    T, H, W = frames_rgb.shape[:3]

    # --- labels (optional: pose-only consumers like test_projection pass None) ---
    if labels_path is None:
        labels = None
    else:
        d = np.load(labels_path)
        labels = d["labels"].astype(np.int32)
        if labels.shape[0] > T:
            labels = labels[:T]
        elif labels.shape[0] < T:
            raise ValueError(f"label frames {labels.shape[0]} < video frames {T}; "
                             f"re-run SAM3 with --num_frames {T}")

    # --- poses ---
    if pose_source == "synthetic":
        positions, headings, w2c = _synthetic_trajectory(T)
        K = intrinsics if intrinsics is not None else _default_intrinsics(W, H)
    elif pose_source == "npz":
        if poses_npz_path is None:
            raise ValueError("pose_source='npz' requires poses_npz_path")
        p = np.load(poses_npz_path)
        positions = p["positions"].astype(np.float32)[:T]
        headings = p["headings"].astype(np.float32)[:T]
        w2c = p["w2c"].astype(np.float32)[:T] if "w2c" in p else None
        K = p["K"].astype(np.float32) if "K" in p else (intrinsics or _default_intrinsics(W, H))
        if w2c is None:
            raise ValueError("poses npz must include 'w2c' key for now")
    else:
        raise ValueError(f"unknown pose_source: {pose_source}")

    # --- goal = last position of the trajectory (offline eval convention) ---
    goal = positions[-1].copy()

    return Clip(
        name=video_path.stem,
        frames_rgb=frames_rgb,
        labels=labels,
        positions=positions,
        headings=headings,
        K=K,
        w2c=w2c,
        goal=goal,
    )
