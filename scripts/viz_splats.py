"""Render a dumped Gaussian cloud as SPLATS (disks sized by each Gaussian's
scale), not as points, for figures and slides.

Three colourings of the same view, so they overlay exactly:
  <scene>_splats_rgb.png    the scene as reconstructed
  <scene>_splats_class.png  the fused v14 semantic labels
  <scene>_splats_trav.png   traversable (warm) vs not (cool), the reward's view

Pure numpy plus ffmpeg for the PNG write, so it runs on the laptop with no
graphics stack. A z-buffer keeps the nearest splat per pixel, which gives the
crisp look of a real splat render rather than a fog of points.

    python3 scripts/viz_splats.py --cloud <scene>_cloud.npz --out_dir .
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

V14 = [
    (0, 0, 0), (200, 225, 245), (150, 100, 55), (75, 190, 80), (95, 65, 35),
    (50, 120, 200), (210, 210, 210), (70, 70, 85), (235, 205, 150),
    (220, 140, 80), (185, 55, 50), (170, 200, 55), (205, 70, 145), (110, 130, 220),
]
# traversability_v14_walkway: these classes score 0 -> the reward will not step there
NONTRAV = {0, 1, 3, 5, 10, 11, 12, 13}


def write_png(path: Path, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-", str(path)],
        input=np.ascontiguousarray(img.astype(np.uint8)).tobytes(), check=True)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])            # world -> camera rows


def splat(P, C, sizes, W, H, eye, target, fov_deg, bg):
    """z-buffered disk rasterisation. P [N,3], C [N,3] uint8, sizes [N]."""
    R = look_at(eye, target)
    cam = (P - np.asarray(eye, float)) @ R.T          # x right, y up, z forward
    z = cam[:, 2]
    keep = z > 0.15
    cam, C, sizes, z = cam[keep], C[keep], sizes[keep], z[keep]
    f = 0.5 * W / np.tan(np.radians(fov_deg) * 0.5)
    u = (W * 0.5 + f * cam[:, 0] / z)
    v = (H * 0.5 - f * cam[:, 1] / z)
    rad = np.clip(f * sizes / z, 0.6, 9.0)
    inside = (u > -12) & (u < W + 12) & (v > -12) & (v < H + 12)
    u, v, z, C, rad = u[inside], v[inside], z[inside], C[inside], rad[inside]

    zbuf = np.full(H * W, np.inf, np.float32)
    cbuf = np.zeros((H * W, 3), np.uint8)
    cbuf[:] = bg
    ui, vi = np.round(u).astype(np.int32), np.round(v).astype(np.int32)

    for lo, hi in [(0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.5), (5.5, 9.1)]:
        m = (rad >= lo) & (rad < hi)
        if not m.any():
            continue
        rr = int(np.ceil(hi))
        dy, dx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disk = (dx * dx + dy * dy) <= rr * rr
        offs = np.stack([dx[disk], dy[disk]], 1)      # [K,2]
        uu = ui[m][:, None] + offs[None, :, 0]
        vv = vi[m][:, None] + offs[None, :, 1]
        zz = np.repeat(z[m][:, None], offs.shape[0], 1)
        cc = np.repeat(C[m][:, None, :], offs.shape[0], 1)
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        idx = (vv[ok] * W + uu[ok]).astype(np.int64)
        zf, cf = zz[ok].astype(np.float32), cc[ok]
        np.minimum.at(zbuf, idx, zf)
        win = zf <= zbuf[idx] + 1e-6
        cbuf[idx[win]] = cf[win]
    return cbuf.reshape(H, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=850)
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--frame", type=int, default=None,
                    help="walk frame to look from (default: the middle)")
    ap.add_argument("--eye_up", type=float, default=7.0, help="metres above the walk")
    ap.add_argument("--eye_back", type=float, default=11.0, help="metres behind it")
    ap.add_argument("--size_scale", type=float, default=1.0)
    args = ap.parse_args()

    c = np.load(args.cloud)
    P = np.asarray(c["points"], np.float32).copy()
    P[:, 1] *= -1.0                                   # same flip the map builder uses
    L = np.asarray(c["labels"]).astype(int)
    rgb = np.asarray(c["colors"])
    rgb = (rgb * 255 if rgb.max() <= 1.01 else rgb).astype(np.uint8)[:, :3]
    sizes = np.asarray(c["sizes"], np.float32)
    if sizes.ndim > 1:
        sizes = sizes.mean(1)
    sizes = np.abs(sizes) * args.size_scale
    walk = (np.asarray(c["traj_positions"], np.float32) * np.array([1, -1, 1], np.float32))

    i = args.frame if args.frame is not None else len(walk) // 2
    i = int(np.clip(i, 0, len(walk) - 1))
    fwd = walk[min(i + 5, len(walk) - 1)] - walk[max(i - 5, 0)]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    target = walk[i] + fwd * 6.0
    eye = walk[i] - fwd * args.eye_back + np.array([0, 0, args.eye_up], np.float32)

    cls = np.clip(L, 0, len(V14) - 1)
    col_class = np.array(V14, np.uint8)[cls]
    warm, cool = np.array([232, 150, 45], np.uint8), np.array([70, 110, 165], np.uint8)
    col_trav = np.where(np.isin(cls, list(NONTRAV))[:, None], cool, warm)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.cloud).name.replace("_cloud.npz", "")
    for tag, C, bg in [("rgb", rgb, (250, 250, 250)),
                       ("class", col_class, (255, 255, 255)),
                       ("trav", col_trav, (255, 255, 255))]:
        img = splat(P, C, sizes, args.width, args.height, eye, target, args.fov, bg)
        write_png(out / f"{stem}_splats_{tag}.png", img)
        print("wrote", out / f"{stem}_splats_{tag}.png")
    print(f"view: walk frame {i}, eye {args.eye_up:.0f} m up / {args.eye_back:.0f} m back, "
          f"fov {args.fov:.0f} deg, {len(P):,} Gaussians")


if __name__ == "__main__":
    main()
