#!/usr/bin/env python3
"""The distance-curriculum gate must not advance a policy that is still at chance.

2026-09-25. Every arm's reward curve peaked at 45-80k and plateaued near 0 at the
10 m band. The band had climbed 2 -> 10 m in 40-60k steps: the gate passed on
>= 50 episodes (one rollout) at 50% success and grew 0.5 m per notch, so a policy
at coin-flip success was pushed to the ceiling and left there. This runs the REAL
gate from GoalDistCurriculum against a synthetic win stream at a true 50% success:
the old settings must climb, the slow settings (CURTHRESH=0.75 CURMINEP=100
CURNOTCH=0.25) must not.
"""
import ast, pathlib, sys, textwrap
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def gate_source():
    src = (ROOT / "scripts" / "train_ppo_real.py").read_text()
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "GoalDistCurriculum")
    roll = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_on_rollout_start")
    body = ast.get_source_segment(src, roll).split("rngs = self.training_env")[0]
    return textwrap.dedent("\n".join(body.splitlines()[1:]))


class Gate:
    def __init__(self, start, end, window, threshold, notch, min_episodes=None):
        self.d, self.end = start, end
        self.window, self.threshold, self.notch = window, threshold, notch
        self._wins = []
        self.min_episodes = int(min_episodes) if min_episodes else window // 2
        self._src = gate_source()

    def rollout(self):
        exec(self._src, {"np": np}, {"self": self})


def climb(true_success, rollouts=400, seed=0, **kw):
    rng = np.random.default_rng(seed)
    g = Gate(2.0, 10.0, **kw)
    n = 0
    while g.d < 10.0 and n < rollouts:
        g._wins += list((rng.random(50) < true_success).astype(float))   # ~50 episodes per rollout
        g.rollout(); n += 1
    return g.d, n


def main():
    fails = []
    def ck(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok: fails.append(name)

    print("the OLD gate (threshold 0.5, >=50 episodes, +0.5 m) at a TRUE 50% success:")
    d, n = climb(0.5, window=100, threshold=0.5, notch=0.5)
    ck("climbs to 10 m", d == 10.0, f"in {n} rollouts = {n*2048//1000}k steps (this is the bug)")

    print("\nthe SLOW gate (0.75, full 100-episode window, +0.25 m):")
    d, n = climb(0.5, window=100, threshold=0.75, notch=0.25, min_episodes=100)
    ck("does NOT advance a 50%-success policy", d == 2.0, f"band {d} m after {n} rollouts")
    d, n = climb(0.85, window=100, threshold=0.75, notch=0.25, min_episodes=100)
    ck("does advance an 85%-success policy", d == 10.0, f"in {n} rollouts = {n*2048//1000}k steps")
    ck("and takes at least 2 rollouts per notch", n >= 2 * 32, f"{n} rollouts for 32 notches")

    print("\nknobs reach the constructor and the launcher:")
    tr = (ROOT / "scripts" / "train_ppo_real.py").read_text()
    ck("--cur_threshold / --cur_min_episodes / --cur_notch exist",
       all(k in tr for k in ('"--cur_threshold"', '"--cur_min_episodes"', '"--cur_notch"')))
    ck("constructor receives them", 'min_episodes=getattr(args, "cur_min_episodes"' in tr)
    ck("env_config records them", '"cur_min_episodes":' in tr)
    sh = (ROOT / "scripts" / "slurm" / "train_ppo_real.sh").read_text()
    ck("launcher knows CURTHRESH/CURMINEP/CURNOTCH", "CURTHRESH|CURWIN|CURMINEP|CURNOTCH" in sh and "--cur_min_episodes $CURMINEP" in sh)
    ck("base arm config defines them (submit_arm refuses undefined knobs)",
       all(k in (ROOT / "configs" / "arms" / "_base_campus.env").read_text() for k in ("CURTHRESH=", "CURMINEP=", "CURNOTCH=")))

    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print("\nOK: the slow gate holds a coin-flip policy at 2 m and lets a competent one climb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
