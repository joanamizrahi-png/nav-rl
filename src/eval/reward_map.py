"""Reward from the reconstruction's own semantic map instead of the generated
semantic image (2026-09-03, after the crash audit: most crashes were the
generator painting "obstacle" on ground the reconstruction has as pavement).

The map is a top-down label grid built once per scene from the scene cloud
(<scene>_cloud.npz: points [N,3] in the nav frame, labels [N] v14 ids). A cell
is non-traversable if enough non-traversable points sit in it at ANY height up
to z_max (a wall, a trunk, a person -- but not a canopy above 1.2 m); otherwise
it takes the majority ground label; with no points it is VOID (-1), which the
reward prices exactly like the generated path prices class 0. Two cleanups:
a 3x3 majority vote and removal of isolated non-traversable cells, because the
labels are SAM3's and SAM3 reads sidewalk as obstacle ~17% of the time.

compute_reward_map() pays the SAME terms with the SAME formulas as
reward_2d.compute_reward (semantic, goal, collision, void, step), so every
log key, eval and dashboard reads unchanged; only the label source differs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reward_2d import RewardBreakdown, RewardWeights, GO2_BODY_LENGTH, GO2_BODY_WIDTH

VOID = -1


@dataclass
class LabelGrid:
    x0: float
    y0: float
    res: float
    labels: np.ndarray        # [H, W] int16, VOID where nothing was reconstructed
    n_points: np.ndarray      # [H, W] int32, how many cloud points fed each cell

    def lookup(self, xy: np.ndarray) -> np.ndarray:
        """xy [K,2] -> labels [K] (VOID outside the grid)."""
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        out = np.full(len(xy), VOID, dtype=np.int16)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.labels.shape[1]) & (iy < self.labels.shape[0])
        out[ok] = self.labels[iy[ok], ix[ok]]
        return out


def build_label_grid(pts: np.ndarray, labs: np.ndarray, non_trav_mask: np.ndarray,
                     res: float = 0.1, z_ground: float = 0.15, z_max: float = 1.2,
                     min_points: int = 2, min_nontrav_points: int = 3,
                     clean: bool = True, inflate_m: float = 0.1, fill_m: float = 0.3,
                     fill_max_area_m2: float = 10.0, ignore_classes=(),
                     walk_xy=None, walk_halfwidth_m: float = 0.4) -> LabelGrid:
    pts = np.asarray(pts, dtype=np.float32)
    labs = np.asarray(labs).astype(int)
    keep = (pts[:, 2] < z_max) & (labs >= 0) & (labs < len(non_trav_mask))
    # dynamic agents (person, vehicle) captured mid-walk are frozen into the
    # cloud at wherever they stood; they are not terrain and do not vote
    if ignore_classes:
        keep &= ~np.isin(labs, list(ignore_classes))
    pts, labs = pts[keep], labs[keep]
    x0, y0 = float(pts[:, 0].min()) - res, float(pts[:, 1].min()) - res
    W = int(np.ceil((pts[:, 0].max() - x0) / res)) + 2
    H = int(np.ceil((pts[:, 1].max() - y0) / res)) + 2
    ix = np.floor((pts[:, 0] - x0) / res).astype(int)
    iy = np.floor((pts[:, 1] - y0) / res).astype(int)
    flat = iy * W + ix
    n_cls = len(non_trav_mask)
    # per-cell class counts: ground points vote for their class, non-traversable
    # points at any height vote for theirs
    ground = pts[:, 2] < z_ground
    vote = ground | non_trav_mask[labs]
    counts = np.zeros((H * W, n_cls), dtype=np.int32)
    np.add.at(counts, (flat[vote], labs[vote]), 1)
    n_all = np.bincount(flat, minlength=H * W).astype(np.int32)
    nt = counts[:, non_trav_mask].sum(1)
    labels = np.full(H * W, VOID, dtype=np.int16)
    has = n_all >= min_points
    # majority over all voting classes...
    maj = counts.argmax(1)
    labels[has] = maj[has]
    # ...but a cell with enough non-traversable points IS non-traversable, even
    # if dense ground outvotes the wall (a wall is a thin line in x,y)
    ntc = np.where(non_trav_mask[None, :], counts, 0).argmax(1)
    force = has & (nt >= min_nontrav_points)
    labels[force] = ntc[force]
    labels = labels.reshape(H, W)
    n_points = n_all.reshape(H, W)
    if clean:
        labels = _clean(labels, non_trav_mask)
    if fill_m > 0:
        # reconstruction dropouts on the walkway (textureless pavement) are
        # void cells surrounded by walkable ones. Fill a void cell from the
        # majority of its known neighbours within fill_m, only when most of
        # those neighbours are known -- so a hole gets its surroundings'
        # label and the real edge of the reconstruction stays void.
        for _ in range(3):                # thin gaps: close from the outside in
            labels = _fill_holes(labels, int(round(fill_m / res)))
        # enclosed holes of any width up to fill_max_area_m2: a void region
        # completely surrounded by known cells is a reconstruction dropout,
        # however wide; the rim's majority label fills it. A region touching
        # the grid border is the outside world and stays void.
        labels = _fill_enclosed(labels, max_cells=int(round(fill_max_area_m2 / (res * res))))
    if inflate_m > 0:
        # costmap inflation: a wall is one cell thick in x,y, so without this a
        # 0.6 m footprint crossing it is only ~1/6 non-traversable and never
        # trips the 0.35 crash terminal. Grow every non-traversable cell by
        # the robot's half-width so the footprint sees the wall the way the
        # projected image does (a wall fills the view).
        labels = _inflate(labels, non_trav_mask, int(round(inflate_m / res)))
    if walk_xy is not None and walk_halfwidth_m > 0:
        # The recorded walk is proof of traversability: a person carried the
        # camera through these cells. Whatever SAM3 called them, and whatever
        # inflation did to them, a corridor of walk_halfwidth_m each side is
        # walkable. Applied LAST so nothing above can undo it.
        labels = _walk_corridor(labels, non_trav_mask, np.asarray(walk_xy, dtype=float),
                                x0, y0, res, int(round(walk_halfwidth_m / res)))
    return LabelGrid(x0=x0, y0=y0, res=res, labels=labels, n_points=n_points)


def _clean(labels: np.ndarray, non_trav_mask: np.ndarray) -> np.ndarray:
    """3x3 majority vote over known cells, then drop isolated non-traversable
    cells (fewer than 3 non-traversable neighbours in the 3x3)."""
    H, W = labels.shape
    n_cls = len(non_trav_mask)
    pad = np.pad(labels, 1, constant_values=VOID)
    stack = np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])  # [9,H,W]
    known = stack >= 0
    onehot = np.zeros((n_cls, H, W), dtype=np.int16)
    for c in range(n_cls):
        onehot[c] = ((stack == c) & known).sum(0)
    voted = onehot.argmax(0).astype(np.int16)
    has_any = known.any(0)
    out = np.where(has_any, voted, VOID).astype(np.int16)
    out[labels == VOID] = VOID            # never invent ground where there was none
    # isolated non-traversable cells -> majority of their walkable neighbours
    nt_here = (out >= 0) & non_trav_mask[np.clip(out, 0, n_cls - 1)]
    pad2 = np.pad(nt_here, 1, constant_values=False)
    nt_nb = sum(pad2[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)).astype(int) - nt_here
    iso = nt_here & (nt_nb < 3)
    if iso.any():
        walk = np.zeros((n_cls, H, W), dtype=np.int16)
        for c in range(n_cls):
            if not non_trav_mask[c]:
                walk[c] = ((stack == c)).sum(0)
        repl = walk.argmax(0).astype(np.int16)
        okw = walk.max(0) > 0
        out[iso & okw] = repl[iso & okw]
    return out


def _fill_holes(labels: np.ndarray, r: int, min_known_frac: float = 0.6) -> np.ndarray:
    if r <= 0:
        return labels
    H, W = labels.shape
    n_cls = int(labels.max()) + 1 if labels.size else 1
    pad = np.pad(labels, r, constant_values=VOID)
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    stack = np.stack([pad[r + dy:r + dy + H, r + dx:r + dx + W] for dy, dx in offs])   # [K,H,W]
    known = stack >= 0
    frac_known = known.mean(0)
    votes = np.zeros((n_cls, H, W), dtype=np.int32)
    for c in range(n_cls):
        votes[c] = ((stack == c) & known).sum(0)
    fill = (labels == VOID) & (frac_known >= min_known_frac)
    out = labels.copy()
    out[fill] = votes.argmax(0)[fill].astype(np.int16)
    return out


def _fill_enclosed(labels: np.ndarray, max_cells: int) -> np.ndarray:
    """Fill void regions that touch no grid border and have at most max_cells
    cells, with the majority label of their 1-cell rim."""
    if max_cells <= 0:
        return labels
    try:
        from scipy import ndimage
    except Exception:
        return labels                    # no scipy: keep the pass-based fill only
    void = labels == VOID
    comp, n = ndimage.label(void)
    if n == 0:
        return labels
    H, W = labels.shape
    border = set(np.unique(np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]])))
    sizes = np.bincount(comp.ravel())
    out = labels.copy()
    for k in range(1, n + 1):
        if k in border or sizes[k] > max_cells:
            continue
        m = comp == k
        rim = ndimage.binary_dilation(m, iterations=1) & ~m & (labels >= 0)
        if not rim.any():
            continue
        out[m] = np.bincount(labels[rim]).argmax()
    return out


def _walk_corridor(labels: np.ndarray, non_trav_mask: np.ndarray, walk_xy: np.ndarray,
                   x0: float, y0: float, res: float, r: int) -> np.ndarray:
    H, W = labels.shape
    n_cls = len(non_trav_mask)
    mask = np.zeros((H, W), dtype=bool)
    # densify the walk so consecutive frames a metre apart still leave a solid strip
    seg = np.concatenate([np.linspace(walk_xy[i], walk_xy[i + 1], 8, endpoint=False)
                          for i in range(len(walk_xy) - 1)] + [walk_xy[-1:]])
    ix = np.round((seg[:, 0] - x0) / res).astype(int)
    iy = np.round((seg[:, 1] - y0) / res).astype(int)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            xx, yy = ix + dx, iy + dy
            ok = (xx >= 0) & (yy >= 0) & (xx < W) & (yy < H)
            mask[yy[ok], xx[ok]] = True
    known = labels >= 0
    walkable = known & ~non_trav_mask[np.clip(labels, 0, n_cls - 1)]
    # the class to paint: the walkable class most common under the corridor,
    # sidewalk if the corridor has none
    under = labels[mask & walkable]
    paint = int(np.bincount(under).argmax()) if len(under) else 6
    out = labels.copy()
    out[mask & ~walkable] = paint
    return out


def _inflate(labels: np.ndarray, non_trav_mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return labels
    H, W = labels.shape
    n_cls = len(non_trav_mask)
    known = labels >= 0
    nt = known & non_trav_mask[np.clip(labels, 0, n_cls - 1)]
    out = labels.copy()
    pad = np.pad(labels, r, constant_values=VOID)
    padnt = np.pad(nt, r, constant_values=False)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            src = padnt[r + dy:r + dy + H, r + dx:r + dx + W]
            lab = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            take = src & known & ~nt          # only overwrite walkable known cells
            out[take] = lab[take]
    return out


def footprint_samples(center_xy: np.ndarray, heading_xy: np.ndarray, length: float,
                      width: float, step: float) -> np.ndarray:
    """Lattice of points inside an oriented rectangle, [K,2]."""
    h = heading_xy / (np.linalg.norm(heading_xy) + 1e-9)
    n = np.array([-h[1], h[0]])
    a = np.arange(-length / 2, length / 2 + 1e-6, step)
    b = np.arange(-width / 2, width / 2 + 1e-6, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    return center_xy[None, :] + A.reshape(-1, 1) * h[None, :] + B.reshape(-1, 1) * n[None, :]


def compute_reward_map(grid: LabelGrid, robot_position: np.ndarray, robot_heading: np.ndarray,
                       goal: np.ndarray, traversability_scores: np.ndarray,
                       non_traversable_mask: np.ndarray, previous_position,
                       look_ahead_dist: float = 1.5, collision_look_ahead_dist=None,
                       body_length: float = GO2_BODY_LENGTH, body_width: float = GO2_BODY_WIDTH,
                       weights: RewardWeights = RewardWeights()) -> RewardBreakdown:
    pos = np.asarray(robot_position, dtype=float)[:2]
    hd = np.asarray(robot_heading, dtype=float)[:2]
    step = grid.res / 2.0
    fp = footprint_samples(pos + look_ahead_dist * hd / (np.linalg.norm(hd) + 1e-9), hd,
                           body_length, body_width, step)
    classes = grid.lookup(fp).astype(int)
    classes = np.where(classes < 0, 0, classes)          # VOID -> class 0, as the image path
    scores = traversability_scores[classes]
    mean_class_score = float(scores.mean())
    sem_score = mean_class_score
    n_footprint = int(len(classes))
    n_trav = int((~non_traversable_mask[classes]).sum())
    if weights.void_cost > 0:
        void_sel = classes == 0
        collision_frac = float((non_traversable_mask[classes] & ~void_sel).mean())
        void_frac = float(void_sel.mean())
        if weights.void_exclude_from_semantic:
            known = ~void_sel
            sem_score = float(scores[known].mean()) if known.any() else 1.0
            mean_class_score = sem_score
    else:
        collision_frac = float(non_traversable_mask[classes].mean())
        void_frac = 0.0
    counts = np.bincount(classes, minlength=len(traversability_scores))
    dominant = int(np.argmax(counts))
    # near collision box, when the env keeps a separate (closer) one
    if collision_look_ahead_dist is not None:
        fpc = footprint_samples(pos + float(collision_look_ahead_dist) * hd / (np.linalg.norm(hd) + 1e-9),
                                hd, body_length, body_width, step)
        cc = grid.lookup(fpc).astype(int)
        cc = np.where(cc < 0, 0, cc)
        if weights.void_cost > 0:
            collision_frac = float((non_traversable_mask[cc] & ~(cc == 0)).mean())
        else:
            collision_frac = float(non_traversable_mask[cc].mean())
    if previous_position is None:
        goal_score = 0.0
    else:
        g = np.asarray(goal, dtype=float)
        p0 = np.asarray(previous_position, dtype=float)
        p1 = np.asarray(robot_position, dtype=float)
        goal_score = float(np.linalg.norm(p0 - g)) - float(np.linalg.norm(p1 - g))
    semantic_term = weights.semantic * ((sem_score - 1.0) if weights.terrain_as_cost else sem_score)
    goal_term = weights.goal * goal_score
    collision_term = -weights.collision * collision_frac
    void_term = -weights.void_cost * void_frac
    step_term = -weights.step_cost
    total = semantic_term + goal_term + collision_term + void_term + step_term
    return RewardBreakdown(total=float(total), semantic=float(semantic_term), goal=float(goal_term),
                           collision=float(collision_term), step=float(step_term), void=float(void_term),
                           n_footprint_pixels=n_footprint, n_traversable_pixels=n_trav,
                           mean_class_score=mean_class_score, collision_off_frame=0.0,
                           dominant_class_id=dominant, off_frame_frac=0.0, void_frac=float(void_frac))
