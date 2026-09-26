#!/usr/bin/env python3
"""rgb_alignment(): the similarity the RGB-alignment cost is charged on (2026-09-25).
Pure numpy, lifted out of scene_env.py, exercised on synthetic frames."""
import ast, pathlib, sys, textwrap
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "env" / "scene_env.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rgb_alignment")
ns = {"np": np}; exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns); f = ns["rgb_alignment"]

def main():
    fails = []
    def ck(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok: fails.append(name)
    rng = np.random.default_rng(0)
    H, W = 60, 90
    raster = rng.integers(0, 256, (H, W, 3)).astype(np.uint8)
    alpha = np.ones((H, W), np.float32)
    ck("identical frames -> 1.0", abs(f(raster, raster, alpha) - 1.0) < 1e-6)
    inv = (255 - raster.astype(int)).astype(np.uint8)
    s = f(inv, raster, alpha)
    ck("inverted frame -> well below 1", s is not None and s < 0.6, f"{s:.3f}")
    alpha2 = np.zeros((H, W), np.float32); alpha2[:, :W // 2] = 1.0
    mixed = raster.copy(); mixed[:, W // 2:] = 0                  # differs ONLY where alpha is 0
    ck("differences outside the covered pixels are ignored", abs(f(mixed, raster, alpha2) - 1.0) < 1e-6)
    ck("too little coverage -> None (nothing to be faithful to)", f(raster, raster, alpha * 0.0) is None)
    ck("shape mismatch -> None", f(raster[:, :-1], raster, alpha) is None)
    ck("alpha as HxWx1 accepted", abs(f(raster, raster, alpha[..., None]) - 1.0) < 1e-6)
    # the cost as charged: w * max(0, tau - sim), on the 0.01 reward scale
    for sim, cost in ((0.9, 0.0), (0.6, 100 * 0.2 * 0.01)):
        c = 100 * max(0.0, 0.8 - sim) * 0.01
        ck(f"cost at similarity {sim} = {cost:.2f}/step", abs(c - cost) < 1e-9, f"{c:.2f}")
    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print("\nOK"); return 0

if __name__ == "__main__":
    sys.exit(main())
