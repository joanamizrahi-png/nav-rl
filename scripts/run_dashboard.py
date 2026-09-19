"""Print the training curves of policy runs from their tensorboard event files,
no wandb, no screenshots (2026-09-19, Joana: "is there a way to show you these
results without screenshots").

For each run directory (the ppo_live_* output folder) it reads
<run>/tensorboard/*/events.out.tfevents.* and prints, for the chosen keys,
the latest value and a short history at evenly spaced steps.

    python scripts/run_dashboard.py /scratch/.../outputs/ppo_live_* \
        [--keys curriculum/goal_dist reward/crash reward/goal_bonus ...] [--points 6]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

DEFAULT_KEYS = ["curriculum/goal_dist", "reward/goal_bonus", "reward/crash", "reward/coherence_crash",
                "reward/end_goal_dist_frac", "rollout/ep_len_mean", "rollout/ep_rew_mean", "diag/throttle"]


def load_scalars(run_dir: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # pragma: no cover
        sys.exit(f"tensorboard not importable ({e}); run with the neoverse env on PATH")
    series = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents.*"), recursive=True)):
        acc = EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            pts = [(ev.step, ev.value) for ev in acc.Scalars(tag)]
            series.setdefault(tag, []).extend(pts)
    for tag in series:
        series[tag].sort()
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--points", type=int, default=6, help="history samples per key")
    args = ap.parse_args()
    for run in args.runs:
        series = load_scalars(run)
        name = os.path.basename(run.rstrip("/"))
        tag = ("memory" if "bm5" in name else "chunk" if "chunk" in name else
               "A-map" if ("hyb" in name or "map_then" in name) else "A (4 GPU)" if "_g4_" in name else "A-like (2 GPU)")
        steps = max((s[-1][0] for s in series.values() if s), default=0)
        print(f"\n== {tag}: ...{name[-48:]}  ({steps} env steps logged)")
        if not series:
            print("   no tensorboard events found"); continue
        for k in args.keys:
            s = series.get(k)
            if not s:
                print(f"   {k:<28} (not logged)"); continue
            n = len(s); idx = sorted(set(int(round(i * (n - 1) / max(1, args.points - 1))) for i in range(args.points)))
            hist = "  ".join(f"{s[i][0] // 1000}k:{s[i][1]:.3g}" for i in idx)
            print(f"   {k:<28} latest {s[-1][1]:9.3f}   | {hist}")


if __name__ == "__main__":
    main()
