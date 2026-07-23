"""One-time metric recalibration: camera height 0.6 m (guess) -> 0.25 m (RUGD paper).

RUGD_IROS2019.pdf, Sec III-A: viewpoint "less than 25 centimeters off the ground";
robot top speed 1.0 m/s. Our scale was DEFINED by asserting median camera height
= camera_height_m, so correcting it is a uniform multiplication of every metric
quantity by ratio = 0.25/0.6. Sanity: implied robot speeds drop from an
implausible 0.6-1.2 m/s to 0.25-0.5 m/s (teleop below top speed).

Rescales IN PLACE (idempotent — skips files already at 0.25):
  data/poses/*_poses.npz          positions, cam_positions, w2c/c2w translations,
                                  scale_m_per_unit, step_sizes_m, camera_height_m
  outputs/scene_clouds/clouds/*.npz  points, sizes, traj_positions, traj_cam_z,
                                  camera_height_m
(.ply files are NOT touched — regenerate on the cluster when convenient; only
their absolute scale is off, which SuperSplat viewing doesn't care about.)

Usage: python scripts/rescale_metric.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
H_OLD, H_NEW = 0.6, 0.25
RATIO = H_NEW / H_OLD


def rescale_npz(path: Path, fields_linear, mat_translation_fields=()):
    d = dict(np.load(path))
    h = float(d.get("camera_height_m", H_OLD))
    if abs(h - H_NEW) < 1e-6:
        print(f"  {path.name}: already at {H_NEW} m, skipped")
        return
    if abs(h - H_OLD) > 1e-6:
        print(f"  {path.name}: unexpected camera_height_m={h}, skipped (check manually)")
        return
    for k in fields_linear:
        if k in d:
            d[k] = (d[k].astype(np.float64) * RATIO).astype(d[k].dtype)
    for k in mat_translation_fields:
        if k in d:
            m = d[k].astype(np.float64)
            m[..., :3, 3] *= RATIO
            d[k] = m.astype(np.float32)
    d["camera_height_m"] = np.float32(H_NEW)
    np.savez_compressed(path, **d)
    print(f"  {path.name}: rescaled x{RATIO:.4f}")


def main():
    print(f"ratio = {H_NEW}/{H_OLD} = {RATIO:.4f}")
    print("poses:")
    for p in sorted((REPO_ROOT / "data/poses").glob("*_poses.npz")):
        rescale_npz(p,
                    fields_linear=("positions", "cam_positions", "step_sizes_m",
                                   "scale_m_per_unit"),
                    mat_translation_fields=("w2c", "c2w"))
    clouds_dir = REPO_ROOT / "outputs/scene_clouds/clouds"
    print("clouds:")
    for p in sorted(clouds_dir.glob("*_cloud.npz")):
        rescale_npz(p,
                    fields_linear=("points", "sizes", "traj_positions", "traj_cam_z"))
    print("\ndone. Implied speed check (should be ~0.25-0.5 m/s, top speed 1.0):")
    for p in sorted((REPO_ROOT / "data/poses").glob("*_poses.npz"))[:6]:
        d = np.load(p)
        v = d["step_sizes_m"].mean() * 3.0          # every-5th frame at 15fps = 1/3 s
        print(f"  {p.stem.replace('_poses',''):24s} mean speed {v:.2f} m/s")


if __name__ == "__main__":
    main()
