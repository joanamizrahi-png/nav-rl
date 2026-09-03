"""Do the kwargs a script passes to each config dataclass actually exist?

A dataclass rejects an unknown keyword with a TypeError at construction --
which on the cluster means a job that waited an hour in the queue dies four
seconds after starting, having loaded nothing. On 2026-09-03 six evals died
this way because two BACKEND fields (spawn_yaw_jitter_deg, spawn_lat_jitter_m)
were passed to SceneEnvConfig instead of CalibratedBackendConfig. Nothing local
caught it because nothing local ever CONSTRUCTED the configs.

This finds every `Name(` call for the known config classes in the given
scripts and checks each `kw=` against the dataclass's annotated fields,
following nested calls (RewardWeights inside SceneEnvConfig) correctly.
Pure AST, no imports of the heavy modules, runs anywhere.

    python scripts/check_config_kwargs.py scripts/eval_policy.py scripts/train_ppo_real.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "SceneEnvConfig": "src/env/scene_env.py",
    "RewardWeights": "src/eval/reward_2d.py",
    "CalibratedBackendConfig": "src/env/real_calibrated.py",
    "RealWorldBackendConfig": "src/env/real_backend.py",
}


def fields_of(cls: str) -> set[str]:
    """Annotated fields of the dataclass, including inherited ones we know."""
    out: set[str] = set()
    path = ROOT / CLASSES[cls]
    if not path.exists():
        return out
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    out.add(b.target.id)
            for base in node.bases:
                bname = getattr(base, "id", None)
                if bname in CLASSES:
                    out |= fields_of(bname)
    return out


def main(paths):
    bad = 0
    cache = {c: fields_of(c) for c in CLASSES}
    for p in paths:
        tree = ast.parse(Path(p).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in CLASSES:
                continue
            allowed = cache[name]
            if not allowed:
                print(f"[warn] {name}: could not read its fields"); continue
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in allowed:
                    print(f"{p}:{kw.lineno}  {name}(... {kw.arg}= ...)  "
                          f"<-- NOT A FIELD of {name}")
                    bad += 1
    print("unknown kwargs:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["scripts/eval_policy.py", "scripts/train_ppo_real.py"]))
