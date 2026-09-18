"""Lawn-to-path height step, measured from a scene cloud (numpy only).

The argument for a semantic score: at a lawn/pavement boundary the ground is
level, so no height map can tell the two apart. This measures it. For a cloud
(points + v14 labels, from dump_scene_cloud.py):
  1. bin the points into --cell m ground cells; the cell's ground height is the
     20th percentile of z (robust to walls/trees standing on the cell);
  2. the cell's class is the majority label among its points within
     --ground_band m of that height (ground points only);
  3. every 4-neighbour pair of cells with one path class ({trail, sidewalk,
     pavement, road}) and one grass cell is a boundary pair; the step is the
     absolute height difference.
Prints, per scene: boundary pairs, median step, share below 5 cm, share above
12 cm (a kerb). First measured 2026-09-08 on the gnd_* scenes (1.5 cm median);
rewritten 2026-09-18 for the campus scenes.

    python scripts/boundary_step.py <cloud.npz> [<cloud.npz> ...] [--cell 0.25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PATH = (2, 6, 7, 8)     # trail, sidewalk, road, pavement
GRASS = 3


def measure(npz: Path, cell: float, band: float, max_z_above_cam: float, near_m: float, purity: float):
    d = np.load(npz)
    P = np.asarray(d["points"], np.float32); L = np.asarray(d["labels"]).astype(np.int64)
    traj = np.asarray(d["traj_positions"], np.float32)
    # keep points near the walk and below the camera (drops sky/tree canopy)
    cam_h = float(d["camera_height_m"]) if "camera_height_m" in d.files else 0.6
    keep = P[:, 2] < cam_h + max_z_above_cam
    if near_m > 0:
        keep &= np.min(np.linalg.norm(P[:, None, :2] - traj[None, :, :2], axis=-1), axis=1) < near_m
    P, L = P[keep], L[keep]
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.argsort(key); key, P, L = key[order], P[order], L[order]
    uniq, start = np.unique(key, return_index=True)
    end = np.append(start[1:], len(key))
    cells = {}
    for u, s, e in zip(uniq, start, end):
        if e - s < 5:
            continue
        z = P[s:e, 2]; h = float(np.percentile(z, 20))
        g = np.abs(z - h) <= band
        if g.sum() < 5:
            continue
        lab = np.bincount(L[s:e][g], minlength=14)
        c = int(np.argmax(lab))
        if lab[c] < purity * lab.sum():     # mixed cell: not a clean surface
            continue
        cells[(int(u // 100000), int(u % 100000))] = (h, c)
    steps = []
    for (i, j), (h, c) in cells.items():
        for (di, dj) in ((1, 0), (0, 1)):
            nb = cells.get((i + di, j + dj))
            if nb is None:
                continue
            h2, c2 = nb
            if (c in PATH and c2 == GRASS) or (c2 in PATH and c == GRASS):
                steps.append(abs(h - h2))
    steps = np.asarray(steps)
    return len(cells), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clouds", nargs="+", type=Path)
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--ground_band", type=float, default=0.15)
    ap.add_argument("--max_z_above_cam", type=float, default=0.3)
    ap.add_argument("--near_m", type=float, default=0.0, help="keep points within this distance of the walk (0 = all)")
    ap.add_argument("--purity", type=float, default=0.5, help="majority share needed to call a cell one class")
    args = ap.parse_args()
    print(f"{'scene':<16} {'cells':>6} {'pairs':>6} {'median':>8} {'<5cm':>6} {'>12cm':>6}")
    allsteps = []
    for c in args.clouds:
        n, st = measure(c, args.cell, args.ground_band, args.max_z_above_cam, args.near_m, args.purity)
        name = c.stem.replace("_cloud", "")
        if len(st) == 0:
            print(f"{name:<16} {n:6d} {0:6d}      n/a"); continue
        allsteps.append(st)
        print(f"{name:<16} {n:6d} {len(st):6d} {100 * np.median(st):6.1f} cm {100 * (st < 0.05).mean():5.0f}% {100 * (st > 0.12).mean():5.0f}%")
    if allsteps:
        st = np.concatenate(allsteps)
        print(f"{'ALL':<16} {'':>6} {len(st):6d} {100 * np.median(st):6.1f} cm {100 * (st < 0.05).mean():5.0f}% {100 * (st > 0.12).mean():5.0f}%")


if __name__ == "__main__":
    main()
