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

    def __init__(self, cfg, cache_root: str, alpha_gate: bool = True,
                 family_switch_penalty_m: float = 0.15,
                 sweep_switch_penalty_m: float = 0.0):
        super().__init__(cfg)
        # STICKY-SWEEP (0 = off, preserves all trained-policy behavior):
        # every sweep is one diffusion call = one coherent dream, so hopping
        # sweeps between consecutive lookups flickers the dream content.
        # Charge this distance penalty for leaving the current sweep — the
        # served view stays inside one dream until a competitor is genuinely
        # closer. Serving-side only; the cache itself is untouched.
        self._sweep_switch_penalty = float(sweep_switch_penalty_m)
        self._last_sweep = None
        # HONEST-OBS mode (her 8/19 idea): black out invented pixels in the
        # SERVED RGB, so the policy only ever sees real reconstructed content
        # — dreams (and their people/cars/inconsistency) never reach the
        # observations. Uses the stored alpha; serving-side only.
        self.obs_black_invented = False
        # HYBRID mode: comma-separated roots (e.g. "ribbon_cache,ribbon_cache_spin")
        # load together — path-threaded sweeps serve translation, spins serve
        # rotation, nearest-neighbor arbitrates. The switch penalty makes the
        # lookup STICKY: staying in the current family (one diffusion call's
        # coherent dream) beats hopping families for marginal pose gains.
        self._cache_roots = [Path(r.strip()) for r in str(cache_root).split(",") if r.strip()]
        self._cache_root = self._cache_roots[0]
        self._family_switch_penalty = float(family_switch_penalty_m)
        self._last_family = None
        self._alpha_gate = bool(alpha_gate)   # False = UNGATED: reward trusts
        # diffused labels everywhere (justified 2026-08-17 by the measured
        # cross-sweep coherence of invented content: label agreement 81.8%
        # in both-invented regions vs 89.2% in real — ratio 0.92)
        self._entries = {}          # scene_id -> dict of arrays

    # -- scene loading: calibration + cache arrays; NO reconstruction --------

    def load_scene(self, scene_id: str) -> None:
        if scene_id in self._entries:
            self._current_scene_id = scene_id
            return
        cal = NavCalibration.from_npz(self.cfg.scene_poses_paths[scene_id])
        self._calib[scene_id] = cal
        K = np.load(self.cfg.scene_poses_paths[scene_id])["K"].astype(np.float32)

        rgbs, labels, alphas, xyyaw = [], [], [], []
        families, sweep_ids = [], []
        skipped = 0
        all_sweeps = []
        for fam_id, root in enumerate(self._cache_roots):
            scene_dir = root / scene_id
            manifest = json.loads((scene_dir / "manifest.json").read_text())
            for sw in manifest["sweeps"]:
                all_sweeps.append((fam_id, scene_dir, sw))
        for fam_id, scene_dir, sw in all_sweeps:
            d = scene_dir / sw["file"][:-5]
            if not (d / "semantic_labels.npz").exists() or not (d / "alpha.npz").exists():
                skipped += 1
                continue
            rgb = _read_video(d / "rgb.mp4")
            lab = np.load(d / "semantic_labels.npz")["labels"].astype(np.int8)
            alpha = np.load(d / "alpha.npz")["alpha"]
            n = min(len(rgb), len(lab), len(alpha), len(sw["nav_xyyaw"]))
            rgbs.append(rgb[:n])
            labels.append(lab[:n])                   # RAW labels; gating applied per-lookup
            alphas.append(alpha[:n])
            xyyaw.append(np.asarray(sw["nav_xyyaw"], dtype=np.float64)[:n])
            families.append(np.full(n, fam_id, dtype=np.int16))
            sweep_ids.append(np.full(n, len(sweep_ids), dtype=np.int32))
        if not rgbs:
            raise RuntimeError(f"no complete sweeps under {scene_dir}")
        if skipped:
            print(f"[CachedDiffusedBackend] {scene_id}: {skipped} sweeps missing/incomplete (cache still generating?)")

        rgb_all = np.concatenate(rgbs)               # [N, H, W, 3] uint8
        lab_all = np.concatenate(labels)             # [N, H, W]    int8 RAW labels
        alpha_all = np.concatenate(alphas)           # [N, H, W]    bool (True = real)
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
            "rgb": rgb_all, "labels": lab_all, "alpha": alpha_all,
            "w2c": w2cs, "K": K,
            "xy": nav[:, :2].astype(np.float64),
            "hdg": np.stack([np.cos(yaw_rad), np.sin(yaw_rad)], axis=1) * _M_PER_RAD,
            "family": np.concatenate(families) if families else np.zeros(0, np.int16),
            "sweep": np.concatenate(sweep_ids),
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
        if len(self._cache_roots) > 1 and self._last_family is not None:
            # sticky-family: hopping families mid-motion swaps dream worlds;
            # charge a distance penalty for leaving the current one.
            d2 = d2 + (e["family"] != self._last_family) * (self._family_switch_penalty ** 2)
        if self._sweep_switch_penalty > 0 and self._last_sweep is not None:
            d2 = d2 + (e["sweep"] != self._last_sweep) * (self._sweep_switch_penalty ** 2)
        i = int(np.argmin(d2))
        self._last_family = int(e["family"][i]) if len(e.get("family", [])) else None
        self._last_sweep = int(e["sweep"][i])
        # Lookup-error telemetry for the smoke gate: how far was the nearest
        # photo, really? Rollout HUD prints this; sustained values near the
        # grid's half-spacing (12.5 cm / 7.5 deg) are expected, spikes mean a
        # cache hole (missing sweep) at that pose.
        pos_err = float(np.hypot(e["xy"][i, 0] - x, e["xy"][i, 1] - y))
        cosang = float(np.clip((e["hdg"][i] @ np.array([fwd[0], fwd[1]]) * _M_PER_RAD)
                               / (_M_PER_RAD ** 2), -1.0, 1.0))
        self._last_lookup = (pos_err, float(np.degrees(np.arccos(cosang))))
        # Reward reads _last_semantic_image; raw belief kept separately so
        # rollout videos can show model-believed vs reward-used side by side.
        raw = e["labels"][i]
        self._last_semantic_raw = raw
        if self._alpha_gate:
            gated = raw.copy()
            gated[~e["alpha"][i]] = 0        # invented -> void (reward pays void cost)
            self._last_semantic_image = gated
        else:
            self._last_semantic_image = raw
        rgb = e["rgb"][i]
        if self.obs_black_invented:
            rgb = rgb.copy()
            rgb[~e["alpha"][i]] = 0
        return rgb, e["K"], e["w2c"][i]
