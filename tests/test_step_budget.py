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
    ck("and uses it for max_steps", "max_steps=_resolved_budget(args)" in ev and "_eval_step_budget(args)" in ev)
    ck("old checkpoints keep their fixed budget", "if per_m <= 0.0:" in ev)

    print("\na FIXED-GOAL test must give every arm the same budget (2026-09-24):")
    # The real function, not a restatement of it. chunk5 reached 4 m of curriculum
    # and memDINO_warm 7.5 m; both were run on the SAME 8.1 m far corner and got
    # 104 and 160 actions. chunk5 cannot cover 8.1 m in 104 actions at any throttle
    # it has shown, so its TIMEOUT measured the budget, not the policy.
    # exec ONLY this function out of eval_policy.py: importing the module pulls in
    # torch and sb3, which are not installed where these tests run, but the point
    # is to exercise the real function rather than a restatement of it.
    _budget = None
    try:
        _tree = ast.parse(ev)
        _fn = next(n for n in _tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "_eval_step_budget")
        _ns = {}
        exec(compile(ast.Module(body=[_fn], type_ignores=[]), "eval_policy.py", "exec"), _ns)
        _budget = _ns["_eval_step_budget"]
    except Exception as e:
        ck("the real _eval_step_budget can be exercised", False, f"({e})")
    if _budget is not None:
        class A:                       # a stand-in for argparse's namespace
            def __init__(self, **kw): self.__dict__.update(kw)
        base = dict(max_steps=90, max_steps_base=40, max_steps_per_m=16.0, test_goal_dist=None)
        chunk5  = A(**{**base, "goal_dist": 4.0})
        memdino = A(**{**base, "goal_dist": 7.5})
        ck("without the fix the two arms differ on one test",
           _budget(chunk5) != _budget(memdino),
           f"chunk5 {_budget(chunk5)} vs memDINO {_budget(memdino)}")
        chunk5.test_goal_dist = 8.11; memdino.test_goal_dist = 8.11
        b1, b2 = _budget(chunk5), _budget(memdino)
        ck("with the test distance they match", b1 == b2, f"both {b1}")
        ck("and the budget can actually cover the test", b1 * CRUISE >= 8.11,
           f"{b1} steps covers {b1*CRUISE:.1f} m, test is 8.11 m")
        far = A(**{**base, "goal_dist": 10.0, "test_goal_dist": 3.9})
        ck("a SHORT test never shrinks a longer curriculum's budget",
           _budget(far) == round(40 + 16 * 10.0), f"{_budget(far)}")
        old_ck = A(max_steps=90, max_steps_base=40, max_steps_per_m=0.0,
                   goal_dist=4.0, test_goal_dist=8.11)
        ck("pre-budget checkpoints still keep their fixed 90", _budget(old_ck) == 90)

    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print("\nOK: the budget tracks the band, and eval tests where the curriculum got to")
    return 0

if __name__ == "__main__":
    sys.exit(main())
