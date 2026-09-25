#!/usr/bin/env python3
"""Does the wandb top-down panel draw, before a training run finds out?

2026-09-25. WandbImagePanel.topdown() is pure numpy + cv2 over plain inputs, so it
can be exercised on the login node with no env and no GPU. This lifts the REAL
method out of train_ppo_real.py, calls it on a synthetic scene (a walkable strip
with a wall, a recorded walk, one finished episode, one in progress, a goal) and
on the no-grid fallback, checks the result is a sane RGB image, and writes the
PNGs next to this file so they can be looked at.

    /users/jmizrahi/.conda/envs/neoverse/bin/python tests/test_topdown_panel.py
"""
import ast, pathlib, sys, textwrap, types
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "_topdown_test"


def lift_topdown():
    src = (ROOT / "scripts" / "train_ppo_real.py").read_text()
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "WandbImagePanel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "topdown")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    body = body.replace("@staticmethod\n", "")
    ns = {"np": np}
    exec(body, ns)
    return ns["topdown"]


def main():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("cv2 not importable here; run with the neoverse python on the cluster"); return 2
    topdown = lift_topdown()
    fails = []
    def ck(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok: fails.append(name)
    OUT.mkdir(exist_ok=True)

    # a 30 m x 20 m scene at 0.1 m: void everywhere, a 3 m wide walkable strip along y=10,
    # a wall block across part of it
    res = 0.1
    L = np.full((200, 300), -1, np.int16)
    L[85:115, :] = 6                       # sidewalk strip
    L[85:115, 150:160] = 10                # a wall across it
    grid = types.SimpleNamespace(labels=L, res=res, x0=0.0, y0=0.0)
    walk = np.stack([np.linspace(1, 29, 60), 10 + 0.3 * np.sin(np.linspace(0, 6, 60))], 1)
    last = [(2 + 0.25 * k, 9.5 + 0.02 * k) for k in range(40)]          # finished episode
    cur = [(2 + 0.25 * k, 10.5 - 0.03 * k) for k in range(15)]          # in progress
    goal = (13.0, 10.0)

    non_trav = np.zeros(14, bool); non_trav[[0, 3, 10, 11, 12]] = True    # void, grass, obstacle, veg, person
    img = topdown(grid, walk, [last, cur], goal, "synthetic", non_trav=non_trav)
    ck("returns an image", img is not None)
    if img is not None:
        ck("RGB uint8", img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, str(img.shape))
        ck("not blank", float(img.std()) > 10, f"std {img.std():.1f}")
        ck("has the walkable light tone", (img == (228, 232, 232)).all(-1).sum() > 1000 or (img == (232, 232, 228)).all(-1).sum() > 1000)
        ck("cropped to the paths, not the whole 30 m scene", img.shape[1] < 3000, f"width {img.shape[1]}")
        ck("the wall is drawn (dark red cells present)", (img == (190, 60, 60)).all(-1).sum() > 200,
           f"{(img == (190, 60, 60)).all(-1).sum()} wall pixels")
        import cv2
        cv2.imwrite(str(OUT / "topdown_with_grid.png"), img[:, :, ::-1])

    img2 = topdown(None, walk, [last, cur], goal, "synthetic-no-grid")
    ck("no-grid fallback returns an image", img2 is not None and img2.ndim == 3)
    if img2 is not None:
        import cv2
        cv2.imwrite(str(OUT / "topdown_no_grid.png"), img2[:, :, ::-1])

    img3 = topdown(grid, walk, [None, []], None, "empty")
    ck("empty paths and no goal do not crash", True, f"-> {'image' if img3 is not None else 'None'}")

    img4 = topdown(grid, walk, [[(-50, -50), (400, 400)]], (1e3, 1e3), "off-grid")
    ck("a path far outside the grid does not crash", True, f"-> {'image' if img4 is not None else 'None'}")

    if fails:
        print("\nFAIL\n  " + "\n  ".join(fails)); return 1
    print(f"\nOK: PNGs in {OUT}/  (scp them over and look)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
