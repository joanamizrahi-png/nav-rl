"""Dump each scene's label grid + walk to a small npz so overheads and eval-path
overlays can be drawn on a laptop after the cluster is gone (2026-09-27).
    python scripts/export_scene_grids.py quad2_00 sequoia1_21 ... --out outputs/scene_grids [--clips DIR]
Writes <out>/<scene>_grid.npz with: labels [H,W] int (-1 = void), res, x0, y0,
n_points [H,W] (0 = label invented by fill), body_ok [H,W] bool (walkable for the robot body), walk [N,2] recorded poses (m).
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_corner_scan import load_scene, CLOUDS, TRAV, REPO

ap = argparse.ArgumentParser()
ap.add_argument("scenes", nargs="+")
ap.add_argument("--out", default="outputs/scene_grids")
ap.add_argument("--clouds_dir", default=CLOUDS)
ap.add_argument("--trav", default=str(REPO / TRAV))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for s in a.scenes:
    try:
        grid, maps, walk = load_scene(s, a.clouds_dir, a.trav)
    except SystemExit as e:
        print(f"  SKIP {s}: {e}"); continue
    np.savez_compressed(out / f"{s}_grid.npz", labels=np.asarray(grid.labels, dtype=np.int16), res=float(grid.res),
                        x0=float(grid.x0), y0=float(grid.y0), n_points=np.asarray(grid.n_points, dtype=np.int32), body_ok=np.asarray(maps["body_ok"], dtype=bool),
                        walk=np.asarray(walk, dtype=np.float32))
    print(f"  wrote {out / f'{s}_grid.npz'}  {grid.labels.shape} res {grid.res}")
