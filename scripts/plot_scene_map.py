"""Draw the reward map of a scene: the top-down label grid the map reward
reads (src/eval/reward_map.py), as classes and as the binary traversable /
non-traversable / void map. Look at this BEFORE trusting it as a reward: the
labels are SAM3's, and what SAM3 got wrong is now a wall the policy must avoid.

    python scripts/plot_scene_map.py --scenes gnd_AUw360 gnd_AU_180 \
        --trav config/traversability_v14_walkway.yaml --out_dir outputs/scene_maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reward_map import build_label_grid, VOID  # noqa: E402
from src.eval.traversability import load_traversability  # noqa: E402

V14 = {0: "void", 1: "sky", 2: "trail", 3: "grass", 4: "rough", 5: "water", 6: "sidewalk",
       7: "road", 8: "pavement", 9: "stairs", 10: "obstacle", 11: "vegetation", 12: "person", 13: "vehicle"}
COL = {2: "#bde0b0", 3: "#74c476", 6: "#bdbdbd", 7: "#9e9ac8", 8: "#80cdc1", 9: "#9ecae1",
       10: "#252525", 11: "#fb6a4a", 12: "#e7298a", 13: "#8c6d31", 0: "#ffffff", 1: "#deebf7",
       4: "#8c8c00", 5: "#3182bd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--clouds_dir", default="/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--trav", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--inflate", type=float, default=0.2)
    ap.add_argument("--out_dir", default="/scratch/m000204-pm06b/joana/outputs/scene_maps")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    scores = load_traversability(args.trav)
    nontrav = scores <= args.collision_threshold

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    for sc in args.scenes:
        c = np.load(Path(args.clouds_dir) / f"{sc}_cloud.npz")
        pts, labs = c["points"], c["labels"].astype(int)
        g = build_label_grid(pts, labs, nontrav, res=args.res, inflate_m=args.inflate)
        L = g.labels
        known = L >= 0
        nt = known & nontrav[np.clip(L, 0, len(nontrav) - 1)]
        path = (np.asarray(c["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
        ext = (g.x0, g.x0 + L.shape[1] * g.res, g.y0, g.y0 + L.shape[0] * g.res)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        # classes
        cmap = ListedColormap([COL.get(i, "#000000") for i in range(14)])
        img = np.ma.masked_where(~known, L)
        axes[0].imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=13.5, interpolation="nearest")
        axes[0].plot(path[:, 0], path[:, 1], "-", c="k", lw=1.2)
        present = sorted(set(int(v) for v in np.unique(L[known])))
        axes[0].legend(handles=[Patch(color=COL.get(i, "#000"), label=V14.get(i, str(i))) for i in present]
                       + [Patch(color="w", label="void (no reconstruction)")], fontsize=8, loc="best")
        axes[0].set_title(f"{sc}: reward map, classes ({g.res} m cells)")
        # binary
        b = np.full(L.shape, 0.5)
        b[known & ~nt] = 1.0
        b[nt] = 0.0
        axes[1].imshow(b, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].plot(path[:, 0], path[:, 1], "-", c="tab:red", lw=1.2, label="recorded walk")
        axes[1].set_title(f"traversable (white) / non-traversable (black) / void (grey)\n"
                          f"known {known.mean():.0%} of cells, non-traversable {nt.sum() / max(known.sum(), 1):.0%} of known")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
        fig.tight_layout()
        out = Path(args.out_dir) / f"{sc}_reward_map.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        np.savez_compressed(Path(args.out_dir) / f"{sc}_reward_map.npz", labels=L, x0=g.x0, y0=g.y0, res=g.res)
        print(f"==> {out}   known {known.mean():.0%}  non-trav {nt.sum() / max(known.sum(), 1):.0%}")


if __name__ == "__main__":
    main()
