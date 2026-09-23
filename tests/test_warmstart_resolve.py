#!/usr/bin/env python3
"""--warmstart as a directory: newest checkpoint + curriculum resume.

Added because three continuations (500937-9) were queued on a --dependency and
would run this path for the first time unattended. Imports the real function out
of scripts/train_ppo_real.py without importing torch, so it runs on a laptop.
"""
import ast, json, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_fn():
    """Pull just resolve_warmstart out of the training script (it imports torch at module level)."""
    tree = ast.parse((ROOT / "scripts" / "train_ppo_real.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_warmstart":
            mod = types.ModuleType("m")
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<fn>", "exec"), mod.__dict__)
            return mod.resolve_warmstart
    sys.exit("FAIL: resolve_warmstart not found in scripts/train_ppo_real.py")

def main():
    f = load_fn()
    quiet = lambda *_a, **_k: None
    fails = []
    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}")
        if not ok: fails.append(f"{name}: got {got!r}, want {want!r}")

    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "ppo_run_j499454"; ck = run / "checkpoints"; ck.mkdir(parents=True)
        for n in (9000, 10000, 60000, 70000):      # 9000 vs 10000 catches a text sort
            (ck / f"ppo_{n}_steps.zip").touch()

        print("a directory with no curriculum state:")
        c, g = f(str(ck), 2.0, log=quiet)
        check("newest checkpoint", Path(c).name, "ppo_70000_steps.zip")
        check("curriculum untouched", g, 2.0)

        print("with the parent's curriculum state:")
        (run / "curriculum_state.json").write_text(json.dumps(
            {"goal_dist": 5.36, "goal_dist_range": [3.36, 5.36], "num_timesteps": 70000}))
        c, g = f(str(ck), 2.0, log=quiet)
        check("resumes the parent's distance", g, 5.36)

        print("a parent BEHIND the configured start must not drag it backwards:")
        (run / "curriculum_state.json").write_text(json.dumps({"goal_dist": 1.0}))
        _, g = f(str(ck), 2.0, log=quiet)
        check("keeps the larger", g, 2.0)

        print("a corrupt state file must not kill the run:")
        (run / "curriculum_state.json").write_text("{not json")
        _, g = f(str(ck), 2.0, log=quiet)
        check("falls back to the configured start", g, 2.0)

        print("an explicit checkpoint file is passed through unchanged:")
        c, g = f(str(ck / "ppo_60000_steps.zip"), 2.0, log=quiet)
        check("file untouched", Path(c).name, "ppo_60000_steps.zip")
        check("curriculum untouched", g, 2.0)

        print("no warmstart at all:")
        c, g = f(None, 2.0, log=quiet)
        check("None passes through", c, None)

        print("an empty directory must REFUSE, not start from scratch silently:")
        empty = Path(td) / "empty"; empty.mkdir()
        try:
            f(str(empty), 2.0, log=quiet); check("refused", "no exception", "SystemExit")
        except SystemExit as e:
            check("refused", "REFUSED" in str(e), True)

    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print("\nOK: warmstart directory resolution and curriculum resume behave")
    return 0

if __name__ == "__main__":
    sys.exit(main())
