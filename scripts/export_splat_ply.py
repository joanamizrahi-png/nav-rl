"""Export a dumped scene cloud as a 3D Gaussian Splatting .ply, so it can be
opened in a real splat viewer (drag and drop into https://superspl.at/editor or
https://antimatter15.com/splat) and rendered as actual splats rather than points.

Three colourings, all from the same Gaussians, so the views are comparable:
  <scene>_rgb.ply    as reconstructed
  <scene>_class.ply  fused v14 semantic labels
  <scene>_trav.ply   traversable (warm) vs not (cool) -- the reward's view

Our clouds carry an isotropic scale and no rotation, so each Gaussian is
exported as a sphere with identity rotation. That is what the reconstructor
gives us; it is not an approximation added here.

    python3 scripts/export_splat_ply.py --cloud gnd_AUw360_cloud.npz --out_dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

V14 = [
    (0, 0, 0), (200, 225, 245), (150, 100, 55), (75, 190, 80), (95, 65, 35),
    (50, 120, 200), (210, 210, 210), (70, 70, 85), (235, 205, 150),
    (220, 140, 80), (185, 55, 50), (170, 200, 55), (205, 70, 145), (110, 130, 220),
]
NONTRAV = {0, 1, 3, 5, 10, 11, 12, 13}
SH_C0 = 0.28209479177387814          # DC term of the SH basis

HEADER = """ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""


def write_ply(path: Path, xyz, rgb, opacity, scale):
    n = len(xyz)
    f_dc = (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0
    op = np.clip(opacity.astype(np.float32), 1e-4, 1 - 1e-4)
    op_logit = np.log(op / (1.0 - op))                     # viewers apply sigmoid
    log_scale = np.log(np.clip(scale.astype(np.float32), 1e-6, None))
    rec = np.zeros((n, 17), np.float32)
    rec[:, 0:3] = xyz
    rec[:, 3:6] = 0.0                                       # normals, unused
    rec[:, 6:9] = f_dc
    rec[:, 9] = op_logit
    rec[:, 10:13] = log_scale[:, None]                      # isotropic
    rec[:, 13] = 1.0                                        # identity quaternion
    with open(path, "wb") as fh:
        fh.write(HEADER.format(n=n).encode())
        fh.write(np.ascontiguousarray(rec).tobytes())
    print(f"wrote {path}  ({n:,} gaussians, {path.stat().st_size / 2**20:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--hide_sky", action="store_true", default=True,
                    help="drop void and sky (default on; they hollow the view out)")
    ap.add_argument("--keep_sky", dest="hide_sky", action="store_false")
    ap.add_argument("--min_opacity", type=float, default=0.05)
    ap.add_argument("--size_scale", type=float, default=1.0,
                    help="multiply every Gaussian's radius (raise it if the view looks sparse)")
    args = ap.parse_args()

    c = np.load(args.cloud)
    P = np.asarray(c["points"], np.float32).copy()
    P[:, 1] *= -1.0                                         # match the map builder
    L = np.asarray(c["labels"]).astype(int)
    rgb = np.asarray(c["colors"])
    rgb = (rgb * 255 if rgb.max() <= 1.01 else rgb).astype(np.uint8)[:, :3]
    op = np.asarray(c["opacities"], np.float32) if "opacities" in c else np.ones(len(P), np.float32)
    sz = np.asarray(c["sizes"], np.float32)
    if sz.ndim > 1:
        sz = sz.mean(1)
    sz = np.abs(sz) * args.size_scale

    keep = op >= args.min_opacity
    if args.hide_sky:
        keep &= ~np.isin(L, (0, 1))
    P, L, rgb, op, sz = P[keep], L[keep], rgb[keep], op[keep], sz[keep]
    print(f"kept {keep.mean():.0%} of points -> {len(P):,} gaussians")

    # Ours is z-UP (sky at +19 m, ground at 0). The 3DGS .ply convention comes
    # from COLMAP, where +y points DOWN, and viewers assume it -- so a y-up file
    # shows upside down. Rotate +90 deg about x: (x, y, z) -> (x, -z, y).
    xyz = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)

    cls = np.clip(L, 0, len(V14) - 1)
    col_class = np.array(V14, np.uint8)[cls]
    warm, cool = np.array([232, 150, 45], np.uint8), np.array([70, 110, 165], np.uint8)
    col_trav = np.where(np.isin(cls, list(NONTRAV))[:, None], cool, warm)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.cloud).name.replace("_cloud.npz", "")
    for tag, col in [("rgb", rgb), ("class", col_class), ("trav", col_trav)]:
        write_ply(out / f"{stem}_{tag}.ply", xyz, col, op, sz)
    print("\nOpen at https://superspl.at/editor or https://antimatter15.com/splat "
          "(drag the .ply in). Both render real splats, and both are local to your "
          "browser -- the file is not uploaded anywhere.")


if __name__ == "__main__":
    main()
