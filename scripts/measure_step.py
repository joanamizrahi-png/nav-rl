"""How far does the robot ACTUALLY move per step?

`step_size_m` is the MAXIMUM: the linear action lives in [0,1] (forward_only
clamps the negative half away) and scales it, so a policy running at half
throttle covers 0.15 m, not 0.3. That difference decides whether a goal is
reachable inside max_steps, and nothing has ever measured it -- every
"is 60 steps enough" argument today assumed the maximum.

Reads the per-step [x, y, ...] rows eval_policy stores in metrics.json.

Usage:
    python scripts/measure_step.py /scratch/.../eval_.../metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    steps, ep_len, closed = [], [], []
    for e in d.get("episodes", []):
        raw = e.get("traj") or []
        if len(raw) < 2:
            continue
        xy = np.array([[r[0], r[1]] for r in raw], dtype=float)
        st = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        steps.append(st)
        ep_len.append(len(st))
        if e.get("d_start") and e.get("final_dist") is not None:
            closed.append(e["d_start"] - e["final_dist"])
    if not steps:
        print(f"{Path(path).parent.name}: no trajectories")
        continue
    s = np.concatenate(steps)
    cfg_max = d.get("summary", {}).get("max_steps")
    print(f"\n=== {Path(path).parent.name}")
    print(f"  steps per episode   mean {np.mean(ep_len):.1f}  max {max(ep_len)}")
    print(f"  displacement/step   mean {s.mean():.3f} m   median "
          f"{np.median(s):.3f}   p90 {np.percentile(s, 90):.3f}   "
          f"max {s.max():.3f}")
    print(f"  throttle vs 0.3 m   {100 * s.mean() / 0.3:.0f}% of maximum")
    print(f"  reach in 60 steps   {60 * s.mean():.1f} m at the observed rate "
          f"(18.0 m at full speed)")
    if closed:
        print(f"  distance closed     mean {np.mean(closed):+.2f} m")
