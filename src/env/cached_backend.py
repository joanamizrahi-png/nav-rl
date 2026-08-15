"""Ribbon-cache world backend: observations from precomputed diffused views.

The diffusion model is far too slow for RL steps (~30 s/render), so cache_gen.sh
pre-renders the ribbon (sweeps of 81 poses along the recorded path x lateral
offsets x headings, one diffusion call per sweep) through v10 + the reader.
This backend serves those views at training time: render(pose) returns the
nearest cached view (position + heading nearest-neighbor), in microseconds.

Reward semantics (the "use the diffused labels directly, safely" design,
2026-08-15): each cached view stores the diffused label map AND the alpha mask
(real-geometry evidence per pixel). At load, labels are masked: alpha False ->
class 0 (void). The existing reward machinery then does the right thing
unchanged — real pixels score by the model's own labels, invented pixels pay
the void cost. GaussianLabelBackend works as-is on top of this backend (it
just reads _last_semantic_image).

Coordinate contract matches CalibratedRealWorldBackend.render(): poses are
ROBOT-to-nav-world (4,4) in meters on the post-yaw-fix right-handed frame;
returns (rgb, K, w2c_nav) where w2c_nav is the CACHED view's camera (nearest
neighbor), so reward projection stays consistent with the image shown.

No GPU, no reconstructor, no diffusion at training time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .real_calibrated import NavCalibration, CalibratedRealWorldBackend

# robot(x fwd, y left, z up) -> camera(x right, y down, z fwd); same constant
# as CalibratedRealWorldBackend.render() and make_ribbon_trajectories.py.
R_CAM_LOCAL = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
# Heading-vs-position tradeoff for nearest-neighbor lookup: 45 deg of heading
# error counts like ~0.8 m of position error (one lateral lane).
_M_PER_RAD = 0.8 / np.deg2rad(45)


def _read_video(path: Path) -> np.ndarray:
    """[T, H, W, 3] uint8 RGB."""
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(str(path), plugin="pyav"))
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f[:, :, ::-1])  # BGR -> RGB
        cap.release()
        if not fr:
            raise RuntimeError(f"no frames read from {path}")
        return np.stack(fr)


class CachedDiffusedBackend(CalibratedRealWorldBackend):
    """Serves ribbon-cache views; interface-compatible with the live backend."""

    def __init__(self, cfg, cache_root: str):
        super().__init__(cfg)
        self._cache_root = Path(cache_root)
        self._entries = {}          # scene_id -> dict of arrays

    # -- scene loading: calibration + cache arrays; NO reconstruction --------

    def load_scene(self, scene_id: str) -> None:
        if scene_id in self._entries:
            self._current_scene_id = scene_id
            return
        cal = NavCalibration.from_npz(self.cfg.scene_poses_paths[scene_id])
        self._calib[scene_id] = cal
        K = np.load(self.cfg.scene_poses_paths[scene_id])["K"].astype(np.float32)

        scene_dir = self._cache_root / scene_id
        manifest = json.loads((scene_dir / "manifest.json").read_text())
        rgbs, labels, xyyaw = [], [], []
        skipped = 0
        for sw in manifest["sweeps"]:
            d = scene_dir / sw["file"][:-5]
            if not (d / "semantic_labels.npz").exists() or not (d / "alpha.npz").exists():
                skipped += 1
                continue
            rgb = _read_video(d / "rgb.mp4")
            lab = np.load(d / "semantic_labels.npz")["labels"].astype(np.int8)
            alpha = np.load(d / "alpha.npz")["alpha"]
            n = min(len(rgb), len(lab), len(alpha), len(sw["nav_xyyaw"]))
            lab = lab[:n].copy()
            lab[~alpha[:n]] = 0                      # invented -> void (reward pays void cost)
            rgbs.append(rgb[:n])
            labels.append(lab)
            xyyaw.append(np.asarray(sw["nav_xyyaw"], dtype=np.float64)[:n])
        if not rgbs:
            raise RuntimeError(f"no complete sweeps under {scene_dir}")
        if skipped:
            print(f"[CachedDiffusedBackend] {scene_id}: {skipped} sweeps missing/incomplete (cache still generating?)")

        rgb_all = np.concatenate(rgbs)               # [N, H, W, 3] uint8
        lab_all = np.concatenate(labels)             # [N, H, W]    int8 (void-masked)
        nav = np.concatenate(xyyaw)                  # [N, 3] x, y, yaw_deg

        # Precompute each entry's camera w2c in the nav frame (reward projection).
        w2cs = np.empty((len(nav), 4, 4), dtype=np.float32)
        for i, (x, y, yaw) in enumerate(nav):
            r = np.deg2rad(yaw)
            c, s = np.cos(r), np.sin(r)
            c2w = np.eye(4)
            c2w[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R_CAM_LOCAL
            c2w[:3, 3] = [x, y, cal.camera_height_m]
            w2cs[i] = np.linalg.inv(c2w)

        yaw_rad = np.deg2rad(nav[:, 2])
        self._entries[scene_id] = {
            "rgb": rgb_all, "labels": lab_all, "w2c": w2cs, "K": K,
            "xy": nav[:, :2].astype(np.float64),
            "hdg": np.stack([np.cos(yaw_rad), np.sin(yaw_rad)], axis=1) * _M_PER_RAD,
        }
        self._current_scene_id = scene_id
        mb = (rgb_all.nbytes + lab_all.nbytes) / 1e6
        print(f"[CachedDiffusedBackend] {scene_id}: {len(nav)} cached views "
              f"({len(rgbs)} sweeps, {mb:.0f} MB)")

    # -- lookup instead of rendering ------------------------------------------

    def render(self, pose_nav_robot: np.ndarray):
        e = self._entries[self._current_scene_id]
        x, y = float(pose_nav_robot[0, 3]), float(pose_nav_robot[1, 3])
        fwd = pose_nav_robot[:2, 0]
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        d2 = ((e["xy"][:, 0] - x) ** 2 + (e["xy"][:, 1] - y) ** 2
              + ((e["hdg"][:, 0] - fwd[0] * _M_PER_RAD) ** 2
                 + (e["hdg"][:, 1] - fwd[1] * _M_PER_RAD) ** 2))
        i = int(np.argmin(d2))
        self._last_semantic_image = e["labels"][i]
        return e["rgb"][i], e["K"], e["w2c"][i]
