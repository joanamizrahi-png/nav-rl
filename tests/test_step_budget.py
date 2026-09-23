#!/usr/bin/env python3
"""The episode budget must grow with the goal band, and eval must test where the
curriculum actually got to rather than at its ceiling.

2026-09-23. max_steps sat at 90 from when goals were 2-4 m. The curriculum now runs
to 10 m, and at the cruise these policies command (0.070 m/step, measured from
memCNN's own trajectories) 90 steps buys 6.3 m. Every goal past ~7 m was therefore
unreachable whatever the policy did: the goal bonus stopped arriving, only the step
and terrain costs remained, and A_dino settled on 1.92 m of travel per episode at
0.09 throttle REGARDLESS of goal distance. Meanwhile eval adopted goal_dist from
env_config -- the CEILING, 10 m -- while curriculum_state.json recorded the band
actually reached, 7.5 m. Policies were tested a full band beyond their training.
"""
import ast, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRUISE = 0.070          # m/step, measured from the working memCNN's trajectories
STEP_M = 0.25

def budget(d, base=40, per_m=16.0):
    return round(base + per_m * d)

def main():
    fails = []
    def ck(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok: fails.append(name)

    print("the budget must let the measured cruise reach the top of the band:")
    for d in (2, 4, 6, 7.5, 10):
        n = budget(d)
        ck(f"band {d:>4} m -> {n:>3} steps", n * CRUISE >= d * 1.35,   # 1.35x covers a corner detour
           f"covers {n*CRUISE:.1f} m, needs {d} m")

    print("\nthe old fixed budget must FAIL the same check past ~6 m (this is the bug):")
    ck("90 steps cannot reach 7.5 m at cruise", 90 * CRUISE < 7.5,
       f"90 steps covers {90*CRUISE:.1f} m")
    ck("90 steps cannot reach 10 m at cruise", 90 * CRUISE < 10.0)

    print("\nthe budget must stay within the kinematic ceiling (no free lunch):")
    for d in (2, 10):
        n = budget(d)
        ck(f"band {d} m needs less than full-throttle travel", d < n * STEP_M,
           f"{d} m vs {n*STEP_M:.1f} m at throttle 1.0")

    print("\nscene_env applies it where the curriculum moves the band:")
    src = (ROOT / "src" / "env" / "scene_env.py").read_text()
    ast.parse(src)
    ck("max_steps_per_m is a SceneEnvConfig field", "max_steps_per_m: float" in src)
    ck("set_goal_dist updates max_steps", "self.cfg.max_steps = int(round(" in src)
    ck("it keys off the TOP of the band, not the midpoint",
       'goal_dist_range", (0.0, d))[1]' in src)

    print("\neval prefers the curriculum's reach over the configured ceiling:")
    ev = (ROOT / "scripts" / "eval_policy.py").read_text()
    ast.parse(ev)
    ck("eval reads curriculum_state.json", 'curriculum_state.json' in ev)
    ck("and overrides goal_dist with it", '_tr["goal_dist"] = _reached' in ev)
    ck("and adopts the budget knobs", '"max_steps_base", "max_steps_per_m"' in ev)

    print("\ntraining records them so eval can reproduce the budget:")
    tr = (ROOT / "scripts" / "train_ppo_real.py").read_text()
    ck("max_steps_per_m in env_config", '"max_steps_per_m":' in tr)

    print("\neval applies the same rule (it never calls set_goal_dist):")
    ck("eval has its own budget function", "_eval_step_budget" in ev)
    ck("and uses it for max_steps", "max_steps=_eval_step_budget(args)" in ev)
    ck("old checkpoints keep their fixed budget", "if per_m <= 0.0:" in ev)

    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print("\nOK: the budget tracks the band, and eval tests where the curriculum got to")
    return 0

if __name__ == "__main__":
    sys.exit(main())
