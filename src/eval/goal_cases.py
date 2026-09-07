"""Goal-case classification on the label grid (2026-09-07): is the straight
line from spawn to goal clear (open), blocked by non-walkable ground with a
walkable way around (corner), through a narrow corridor (narrow), or with no
walkable way at all (blocked)? Shared by scripts/scan_goal_cases.py (offline
sheets) and SceneEnv (per-episode labelling and the goal-case mix), so the
training tables and the sheets use one rule. numpy only; scipy when present.
"""
from __future__ import annotations

from collections import deque

import numpy as np

BODY_W = 0.3


def distance_transform(free: np.ndarray, res: float) -> np.ndarray:
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(free) * res
    except Exception:
        d = np.zeros(free.shape, np.float32)
        cur = free.copy(); k = 0
        while cur.any() and k < 60:
            k += 1
            nxt = cur.copy()
            nxt[1:] &= cur[:-1]; nxt[:-1] &= cur[1:]; nxt[:, 1:] &= cur[:, :-1]; nxt[:, :-1] &= cur[:, 1:]
            d[cur & ~nxt] = k * res
            cur = nxt
        d[cur] = k * res
        return d


def local_clearance(dist: np.ndarray, radius_m: float, res: float) -> np.ndarray:
    r = max(1, int(round(radius_m / res)))
    try:
        from scipy import ndimage
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return ndimage.maximum_filter(dist, footprint=(yy ** 2 + xx ** 2) <= r * r)
    except Exception:
        out = dist.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r:
                    continue
                out = np.maximum(out, np.roll(np.roll(dist, dy, axis=0), dx, axis=1))
        return out


def snap(mask: np.ndarray, ij: tuple, max_cells: int = 6):
    if 0 <= ij[0] < mask.shape[0] and 0 <= ij[1] < mask.shape[1] and mask[ij]:
        return ij
    best, bd = None, 1e9
    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            n = (ij[0] + dy, ij[1] + dx)
            if 0 <= n[0] < mask.shape[0] and 0 <= n[1] < mask.shape[1] and mask[n]:
                d = dy * dy + dx * dx
                if d < bd:
                    best, bd = n, d
    return best


def bfs_path(free: np.ndarray, a: tuple, b: tuple, box: tuple):
    y0, y1, x0, x1 = box
    if not (free[a] and free[b]):
        return None
    prev = {a: None}; q = deque([a])
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        c = q.popleft()
        if c == b:
            break
        for dy, dx in nbrs:
            n = (c[0] + dy, c[1] + dx)
            if y0 <= n[0] < y1 and x0 <= n[1] < x1 and free[n] and n not in prev:
                prev[n] = c; q.append(n)
    if b not in prev:
        return None
    path = []; c = b
    while c is not None:
        path.append(c); c = prev[c]
    return path[::-1]


def scene_case_maps(grid, non_trav: np.ndarray, clear_radius_m: float = 0.8) -> dict:
    """Per-scene arrays for classify_pair: walkable mask, distance to the
    nearest non-walkable/unknown cell, medial clearance, body-clear mask."""
    L = grid.labels; known = L >= 0
    free = known & ~non_trav[np.clip(L, 0, len(non_trav) - 1)]
    dist = distance_transform(free, grid.res)
    return {"free": free, "dist": dist, "clear": local_clearance(dist, clear_radius_m, grid.res),
            "body_ok": dist >= BODY_W / 2.0}


def classify_pair(grid, maps: dict, spawn_xy, goal_xy, narrow_m: float = 1.0, detour_min: float = 1.15,
                  box_m: float = 6.0, want_path: bool = False) -> dict:
    """Classify one spawn->goal pair. Returns cls in {open, corner, narrow,
    corner+narrow, blocked, offmap}, detour (path/straight), min_width_m."""
    res = grid.res; L = grid.labels
    def cell(xy):
        return (int((xy[1] - grid.y0) / res), int((xy[0] - grid.x0) / res))
    def inb(ij):
        return 0 <= ij[0] < L.shape[0] and 0 <= ij[1] < L.shape[1]
    s = np.asarray(spawn_xy, float)[:2]; g = np.asarray(goal_xy, float)[:2]
    sc_ij, g_ij = cell(s), cell(g)
    if not (inb(sc_ij) and inb(g_ij)):
        return {"cls": "offmap", "detour": float("inf"), "min_width_m": 0.0}
    straight = float(np.linalg.norm(g - s))
    n = max(2, int(np.ceil(straight / (res / 2))))
    line = s[None, :] + (g - s)[None, :] * np.linspace(0, 1, n)[:, None]
    ok = True
    for q in line:
        ij = cell(q)
        if not inb(ij) or not maps["body_ok"][ij]:
            ok = False; break
    rec = {"straight_m": straight}
    if ok:
        rec.update(cls="open", detour=1.0, min_width_m=float(min(maps["clear"][cell(q)] for q in line)) * 2.0)
    else:
        m = int(box_m / res)
        y0 = max(0, min(sc_ij[0], g_ij[0]) - m); y1 = min(L.shape[0], max(sc_ij[0], g_ij[0]) + m)
        x0 = max(0, min(sc_ij[1], g_ij[1]) - m); x1 = min(L.shape[1], max(sc_ij[1], g_ij[1]) + m)
        a_ij, b_ij = snap(maps["body_ok"], sc_ij), snap(maps["body_ok"], g_ij)
        path = bfs_path(maps["body_ok"], a_ij, b_ij, (y0, y1, x0, x1)) if (a_ij is not None and b_ij is not None) else None
        if path is None:
            rec.update(cls="blocked", detour=float("inf"), min_width_m=0.0)
        else:
            p = np.asarray(path, float)
            plen = float((np.linalg.norm(np.diff(p, axis=0), axis=1) * res).sum()) if len(p) > 1 else 0.0
            ratio = plen / max(straight, 1e-6)
            rec.update(cls=("corner" if ratio >= detour_min else "open"), detour=round(ratio, 3),
                       min_width_m=float(min(maps["clear"][ij] for ij in path)) * 2.0)
            if want_path:
                rec["path"] = [[float(grid.x0 + (j + 0.5) * res), float(grid.y0 + (i + 0.5) * res)] for i, j in path[::5]]
    if rec["cls"] != "blocked" and rec["min_width_m"] < narrow_m:
        rec["cls"] = "narrow" if rec["cls"] == "open" else "corner+narrow"
    return rec
