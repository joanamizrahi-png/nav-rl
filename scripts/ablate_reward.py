"""Reward-component ablation on a single clip.

Runs validate_reward.py with 4 weight configurations:
  * FULL:    default weights (sem + goal + collision)
  * SEM:     semantic only (goal=0, col=0)
  * SEM+G:   semantic + goal (col=0)
  * SEM+C:   semantic + collision (goal=0)

Then overlays the 4 TOTAL reward curves on one plot so we can see which
components carry the signal. Also produces per-component subplot for the
FULL run so we can talk about magnitude and correlation.

Usage:
    python scripts/ablate_reward.py \\
        --video data/rugd_trail_00.mp4 \\
        --labels data/rugd_trail_00.npz \\
        --output_dir outputs/ablation/rugd_trail_00/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- Ablation configs ---------------------------------------------------
CONFIGS = [
    # (label, w_sem, w_goal, w_col)
    ("FULL",   1.0, 0.5, 5.0),
    ("SEM",    1.0, 0.0, 0.0),
    ("SEM+G",  1.0, 0.5, 0.0),
    ("SEM+C",  1.0, 0.0, 5.0),
]


def run_one(config_label: str, w_sem: float, w_goal: float, w_col: float,
            video: Path, labels: Path, base_out: Path, look_ahead_dist: float):
    """Invoke validate_reward.py as a subprocess for one config."""
    out_dir = base_out / config_label
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "validate_reward.py"),
        "--video", str(video),
        "--labels", str(labels),
        "--output_dir", str(out_dir),
        "--look_ahead_dist", str(look_ahead_dist),
        "--w_sem", str(w_sem),
        "--w_goal", str(w_goal),
        "--w_col", str(w_col),
    ]
    print(f"[ablate] {config_label:8s} sem={w_sem} goal={w_goal} col={w_col}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return out_dir


def load_totals(csv_path: Path):
    """Return list of (frame, total, semantic, goal, collision) tuples."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["frame"]),
                "total": float(r["total"]),
                "semantic": float(r["semantic"]),
                "goal": float(r["goal"]),
                "collision": float(r["collision"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--look_ahead_dist", type=float, default=1.5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run each config ----
    per_config_rows = {}
    for label, ws, wg, wc in CONFIGS:
        out_dir = run_one(label, ws, wg, wc, args.video, args.labels, args.output_dir, args.look_ahead_dist)
        per_config_rows[label] = load_totals(out_dir / "reward.csv")

    # ---- Plot comparison ----
    _plot_ablation(per_config_rows, args.output_dir / "ablation_totals.png")
    _plot_full_components(per_config_rows["FULL"], args.output_dir / "full_components.png")
    _write_summary(per_config_rows, args.output_dir / "ablation_summary.csv")

    print(f"\n[ablate] wrote {args.output_dir}/ablation_totals.png")
    print(f"[ablate] wrote {args.output_dir}/full_components.png")
    print(f"[ablate] wrote {args.output_dir}/ablation_summary.csv")


def _plot_ablation(per_config_rows: dict, out_path: Path):
    """4 total-reward curves on one axis."""
    import matplotlib.pyplot as plt

    colors = {"FULL": "black", "SEM": "tab:green", "SEM+G": "tab:blue", "SEM+C": "tab:red"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, rows in per_config_rows.items():
        frames = [r["frame"] for r in rows]
        totals = [r["total"] for r in rows]
        ax.plot(frames, totals, label=label, color=colors.get(label, "gray"), linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("frame")
    ax.set_ylabel("total reward")
    ax.set_title("Reward ablation: which components carry the signal?")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_full_components(rows_full: list, out_path: Path):
    """Per-component subplot for the FULL config (each component on its own axis)."""
    import matplotlib.pyplot as plt

    frames = [r["frame"] for r in rows_full]
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, key, color in [
        (axes[0], "total",     "black"),
        (axes[1], "semantic",  "tab:green"),
        (axes[2], "goal",      "tab:blue"),
        (axes[3], "collision", "tab:red"),
    ]:
        ax.plot(frames, [r[key] for r in rows_full], color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    fig.suptitle("FULL config — per-component breakdown")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_summary(per_config_rows: dict, out_path: Path):
    """Per-config mean/min/max summary."""
    import numpy as np
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "total_mean", "total_min", "total_max", "sem_mean", "goal_mean", "col_mean"])
        for label, rows in per_config_rows.items():
            totals = np.array([r["total"] for r in rows])
            sem = np.array([r["semantic"] for r in rows])
            goal = np.array([r["goal"] for r in rows])
            col = np.array([r["collision"] for r in rows])
            writer.writerow([
                label,
                f"{totals.mean():+.3f}", f"{totals.min():+.3f}", f"{totals.max():+.3f}",
                f"{sem.mean():+.3f}", f"{goal.mean():+.3f}", f"{col.mean():+.3f}",
            ])


if __name__ == "__main__":
    main()
