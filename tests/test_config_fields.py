#!/usr/bin/env python3
"""Every keyword passed to a config dataclass must be a field of THAT dataclass.

2026-09-22: seven training arms died eight seconds after starting with
    TypeError: CalibratedBackendConfig.__init__() got an unexpected keyword
    argument 'obs_frame_stack'
obs_frame_stack is a SceneEnvConfig field (the stacking happens in
SceneEnv._obs). Training passed it to the BACKEND config; eval passed it to the
scene config and was correct all along. A syntax check cannot see this and the
frame-stack unit test could not either, because it built the env directly and
never went through the launcher's config construction.

Static on purpose: it reads the dataclass definitions and the call sites out of
the source, so it runs on a laptop with no torch and no GPU, which means there
is no excuse for not running it before a launch.
"""
import ast, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALLERS = ["scripts/train_ppo_real.py", "scripts/eval_policy.py"]
# base classes FIRST: a subclass inherits its bases' fields
DEFS = ["src/env/real_backend.py", "src/env/real_calibrated.py", "src/env/scene_env.py"]


def dataclass_fields(paths):
    """{ClassName: {field, ...}} from annotated assignments in each class body."""
    out = {}
    for rel in paths:
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Config"):
                continue
            fields = {b.target.id for b in node.body
                      if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)}
            # a subclass inherits its bases' fields
            for base in node.bases:
                if isinstance(base, ast.Name):
                    fields |= out.get(base.id, set())
            out[node.name] = fields
    return out


def call_sites(rel, known):
    """[(ClassName, keyword, lineno)] for every config constructed in the file."""
    tree = ast.parse((ROOT / rel).read_text())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in known:
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                hits.append((name, kw.arg, kw.lineno))
    return hits


def main():
    fields = dataclass_fields(DEFS)
    if not fields:
        print("FAIL: no *Config dataclasses found; the parser needs updating")
        return 1
    print(f"== config fields: " + ", ".join(f"{k} ({len(v)})" for k, v in sorted(fields.items())))
    bad = []
    checked = 0
    for rel in CALLERS:
        for cls, kw, line in call_sites(rel, fields):
            checked += 1
            if kw not in fields[cls]:
                bad.append((rel, line, cls, kw))
    print(f"   {checked} keyword arguments checked across {len(CALLERS)} callers")
    if bad:
        print("\n   KEYWORD PASSED TO A CONFIG THAT HAS NO SUCH FIELD:")
        for rel, line, cls, kw in bad:
            owner = [c for c, f in fields.items() if kw in f]
            where = f" (it is a field of {', '.join(owner)})" if owner else ""
            print(f"     {rel}:{line}  {cls}(..., {kw}=...){where}")
        print("\nFAIL")
        return 1
    print("\nOK: every config keyword is a field of the config it is passed to")
    return 0


if __name__ == "__main__":
    sys.exit(main())
