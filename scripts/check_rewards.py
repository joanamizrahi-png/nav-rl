"""Reward mechanism check: GATE / VOID / IMGVOID, measured not argued.

Renders the SAME live-diffusion views training uses, at poses drawn from the
J-spec distribution, and prints for each step what every reward mechanism
actually computes — over a SWEEP of alpha-gate thresholds on identical renders.

The renders happen ONCE (ungated). Every threshold is then applied in numpy to
the same labels + alpha, so the tau column is the only thing that varies: no
diffusion-noise difference between rows, and tau=0.0 IS the ungated baseline.

Answers, with numbers:
  * ALPHA        the support distribution as a histogram, whole-image AND
                 inside the footprint (the region the reward actually reads),
                 so the 0.5 gate threshold stops being an inherited guess
  * SWEEP        for each tau: img void, footprint void, collision, crash-level
                 steps, and every reward term — the whole trade curve at once
  * IMGVOID      whole-image void share (her measure) next to mean alpha (the
                 `cov` number the spin certificates print) — different
                 statistics that nobody had compared
  * VOID         footprint void share, and proof it is EXCLUDED from
                 collision_frac when void_cost > 0
  * REWARD       every term at the real x0.01 scale, so we can see what
                 dominates and whether phantom terrain is driving crashes

Usage (GPU node):
    python scripts/check_rewards.py --scene gnd_AUw360 \
        --trav_path config/traversability_v14_walkway.yaml \
        --sweep 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8

Obstacle probe (drive at a KNOWN real obstacle instead of a sampled goal, to
test whether the gate also deletes obstacles that ARE observed):
    ... --goal_xy 1.57,-6.06 --spawn_frame 40
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import CalibratedBackendConfig
from src.env.vec_live_env import BatchedLiveDiffusedBackend
from src.eval.traversability import load_traversability
from src.eval.reward_2d import (
    RewardWeights, compute_reward, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
    _footprint_corners_world, _project_points, _fill_polygon,
)

V14_NAMES = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
             "road", "pavement", "stairs", "obstacle", "vegetation", "person",
             "vehicle"]


def pose_at(xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = (c, s, 0.0)
    m[:3, 1] = (-s, c, 0.0)
    m[:3, 3] = (xy[0], xy[1], 0.0)
    return m


def episode_poses(cal, rng, n_steps, cone_deg, dist_range, yaw_jit, lat_jit,
                  spawn_min=10, goal_xy=None, spawn_frame=None):
    """One J-spec episode: jittered spawn on the recorded path, goal in the
    tangent cone, then a straight walk toward it in 0.3 m steps.

    goal_xy overrides the sampled goal (obstacle probe): the walk then heads at
    a fixed world point, so the same real obstacle is approached every episode.
    """
    path = np.asarray(cal.positions, dtype=float)[:, :2]
    if spawn_frame is not None:
        f = int(np.clip(spawn_frame, 1, len(path) - 2))
    else:
        f = int(rng.integers(spawn_min, max(len(path) - 6, spawn_min + 1)))
    fw = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
    base = float(np.arctan2(fw[1], fw[0]))
    p = path[f] + rng.uniform(-1, 1) * lat_jit * np.array(
        [-np.sin(base), np.cos(base)])
    yaw = base + np.deg2rad(rng.uniform(-1, 1) * yaw_jit)
    if goal_xy is not None:
        goal = np.asarray(goal_xy, dtype=float)
    else:
        th = base + np.deg2rad(rng.uniform(-1, 1) * cone_deg / 2.0)
        goal = p + rng.uniform(*dist_range) * np.array([np.cos(th), np.sin(th)])

    u = (goal - p) / max(np.linalg.norm(goal - p), 1e-6)
    poses = [pose_at(p, yaw)]
    for k in range(1, n_steps):
        poses.append(pose_at(p + u * 0.3 * k, float(np.arctan2(u[1], u[0]))))
    return poses, goal


def render_episodes(args):
    """Render every J-spec pose ONCE, ungated. Keep labels + alpha so the gate
    can be applied afterwards at any threshold on identical pixels."""
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        sem_palette_version=args.sem_palette,
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height, W=args.width,
    )
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt,
                                       alpha_gate=False)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]

    goal_xy = None
    if args.goal_xy:
        goal_xy = [float(v) for v in args.goal_xy.split(",")]

    rng = np.random.default_rng(args.seed)
    recs = []
    for ep in range(args.episodes):
        poses, goal = episode_poses(
            cal, rng, args.steps, args.cone_deg,
            tuple(float(v) for v in args.dist_range.split(",")),
            args.spawn_yaw_jitter, args.spawn_lat_jitter,
            goal_xy=goal_xy, spawn_frame=args.spawn_frame)
        prev = None
        for si, pose in enumerate(poses):
            (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
            alpha = getattr(world, "last_alpha", None)
            a = None if not alpha else np.asarray(alpha[0], dtype=np.float32).copy()
            _ras = getattr(world, "last_raster", None)
            _sr = getattr(world, "last_sem_raster", None)
            pos = np.array([pose[0, 3], pose[1, 3], 0.0], dtype=float)
            recs.append(dict(
                ep=ep, step=si, lab=np.asarray(lab).copy(), alpha=a,
                K=np.asarray(K).copy(), w2c=np.asarray(w2c).copy(),
                pos=pos, head=np.asarray(pose[:3, 0], dtype=float).copy(),
                goal=np.array([goal[0], goal[1], 0.0], dtype=float),
                prev=None if prev is None else prev.copy(),
                rgb=rgb.copy() if ep == 0 else None,
                ras=(_ras[0].copy() if (ep == 0 and _ras) else None),
                sras=(_sr[0].copy() if (ep == 0 and _sr) else None)))
            prev = pos.copy()
        print(f"  rendered episode {ep + 1}/{args.episodes}", flush=True)
    return recs


def footprint_mask(rec, look_ahead):
    """The exact pixels compute_reward scores — so alpha can be measured THERE
    and not just over the whole image."""
    corners = _footprint_corners_world(
        rec["pos"], rec["head"], look_ahead_dist=look_ahead,
        length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
    uv, in_front = _project_points(corners, rec["K"], rec["w2c"])
    if not in_front.all():
        return None
    h, w = rec["lab"].shape[:2]
    return _fill_polygon(h, w, uv)


def score(recs, tau, trav, non_trav, weights, args):
    """Apply the gate at threshold tau and re-score every rendered step."""
    rows = []
    for r in recs:
        lab = r["lab"]
        if tau > 0.0 and r["alpha"] is not None:
            lab = np.where(r["alpha"] > tau, lab, 0)
        b = compute_reward(
            semantic_image=lab, K=r["K"], w2c=r["w2c"],
            robot_position=r["pos"], robot_heading=r["head"], goal=r["goal"],
            traversability_scores=trav, non_traversable_mask=non_trav,
            previous_position=r["prev"], look_ahead_dist=args.look_ahead,
            body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH,
            weights=weights)
        cls, cnt = np.unique(lab, return_counts=True)
        rows.append(dict(
            tau=tau, ep=r["ep"], step=r["step"],
            img_void=float((lab == 0).mean()),
            mean_alpha=float(r["alpha"].mean()) if r["alpha"] is not None else float("nan"),
            a10=float(np.percentile(r["alpha"], 10)) if r["alpha"] is not None else float("nan"),
            a50=float(np.percentile(r["alpha"], 50)) if r["alpha"] is not None else float("nan"),
            a90=float(np.percentile(r["alpha"], 90)) if r["alpha"] is not None else float("nan"),
            sup=float((r["alpha"] > tau).mean()) if r["alpha"] is not None else float("nan"),
            fp_void=float(b.void_frac),
            fp_coll=float(-b.collision / max(weights.collision, 1e-6)),
            fp_ok=float(max(0.0, 1.0 - b.void_frac
                            + b.collision / max(weights.collision, 1e-6))),
            sem=float(b.semantic), goal_t=float(b.goal),
            coll_t=float(b.collision), void_t=float(b.void),
            total=float(b.total),
            dom=V14_NAMES[int(cls[np.argmax(cnt)])]))
    return rows


def alpha_histogram(vals, nbins=10):
    h, edges = np.histogram(np.clip(vals, 0.0, 1.0), bins=nbins, range=(0.0, 1.0))
    frac = h / max(h.sum(), 1)
    lines = []
    for i in range(nbins):
        bar = "#" * int(round(frac[i] * 60))
        lines.append(f"    [{edges[i]:.1f},{edges[i+1]:.1f})  {frac[i]*100:5.1f}%  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips_dir", default="/scratch/m000204-pm06b/joana/data/rugd_clips")
    ap.add_argument("--poses_dir", default="/scratch/m000204-pm06b/joana/outputs/poses")
    ap.add_argument("--labels_dir",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm06b/joana/runs/train_semantic_v21/checkpoint-epoch-12.safetensors")
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--trav_path", default="config/traversability_v14_walkway.yaml")
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--num_steps", type=int, default=4)
    ap.add_argument("--sem_palette", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cone_deg", type=float, default=50.0)
    ap.add_argument("--dist_range", default="5,10")
    ap.add_argument("--spawn_yaw_jitter", type=float, default=20.0)
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.4)
    ap.add_argument("--semantic_weight", type=float, default=5.0)
    ap.add_argument("--goal_weight", type=float, default=10.0)
    ap.add_argument("--collision_threshold", type=float, default=0.1)
    ap.add_argument("--look_ahead", type=float, default=1.5)
    ap.add_argument("--reward_scale", type=float, default=0.01)
    ap.add_argument("--crash_frac", type=float, default=0.35)
    ap.add_argument("--sweep", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                    help="alpha-gate thresholds to score; 0 = ungated")
    ap.add_argument("--gate_tau", type=float, default=0.5,
                    help="the threshold shown in the panels and detail blocks")
    ap.add_argument("--goal_xy", default="",
                    help="obstacle probe: fixed world goal 'x,y' for all episodes")
    ap.add_argument("--spawn_frame", type=int, default=None,
                    help="obstacle probe: fixed spawn frame on the recorded path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out",
                    default="/scratch/m000204-pm06b/joana/outputs/reward_check")
    args = ap.parse_args()

    taus = [float(v) for v in args.sweep.split(",")]
    if args.gate_tau not in taus:
        taus.append(args.gate_tau)
    taus = sorted(set(taus))

    print(f"=== {args.scene} | trav {args.trav_path} | cone {args.cone_deg} | "
          f"jitter {args.spawn_yaw_jitter}deg/{args.spawn_lat_jitter}m | "
          f"reward_scale {args.reward_scale} | look_ahead {args.look_ahead}",
          flush=True)
    if args.goal_xy:
        print(f"=== OBSTACLE PROBE: fixed goal {args.goal_xy} "
              f"spawn_frame {args.spawn_frame}", flush=True)
    print(f"=== sweep taus {taus}", flush=True)

    recs = render_episodes(args)
    n = len(recs)
    print(f"==> {n} rendered steps, gating applied post-hoc on identical pixels",
          flush=True)

    trav = load_traversability(Path(args.trav_path) if args.trav_path else None)
    non_trav = trav <= args.collision_threshold
    weights = RewardWeights(semantic=args.semantic_weight, goal=args.goal_weight,
                            collision=1.0, step_cost=0.05, void_cost=0.3,
                            terrain_as_cost=True)

    # ---- the support distribution: whole image AND inside the footprint ----
    have_alpha = recs[0]["alpha"] is not None
    if have_alpha:
        all_a = np.concatenate([r["alpha"].ravel() for r in recs])
        fp_a = []
        for r in recs:
            m = footprint_mask(r, args.look_ahead)
            if m is not None and m.any():
                fp_a.append(r["alpha"][m])
        fp_a = np.concatenate(fp_a) if fp_a else np.zeros(1, dtype=np.float32)
        print("\n===== ALPHA DISTRIBUTION (whole image, all steps pooled) =====")
        print(f"  mean {all_a.mean():.3f}  p10 {np.percentile(all_a, 10):.3f}  "
              f"p50 {np.percentile(all_a, 50):.3f}  "
              f"p90 {np.percentile(all_a, 90):.3f}")
        print(alpha_histogram(all_a))
        print("\n===== ALPHA DISTRIBUTION (FOOTPRINT ONLY — what the reward reads) =====")
        print(f"  mean {fp_a.mean():.3f}  p10 {np.percentile(fp_a, 10):.3f}  "
              f"p50 {np.percentile(fp_a, 50):.3f}  "
              f"p90 {np.percentile(fp_a, 90):.3f}")
        print(alpha_histogram(fp_a))
        print("  A GAP between two humps = a principled threshold. A smooth "
              "ramp = any threshold is a policy choice, not a fact.", flush=True)

    # ---- the sweep ----
    per_tau = {t: score(recs, t, trav, non_trav, weights, args) for t in taus}

    s = args.reward_scale
    print(f"\n===== GATE THRESHOLD SWEEP ({n} steps, identical renders) =====")
    print("   tau   sup   imgvoid  fp_void  fp_coll  CRASH   sem       void      "
          "coll      total")
    for t in taus:
        A = {k: np.array([r[k] for r in per_tau[t]], dtype=float)
             for k in per_tau[t][0] if k != "dom"}
        crash = int((A["fp_coll"] >= args.crash_frac).sum())
        mark = "  <- ungated" if t == 0.0 else (
            "  <- current gate" if t == args.gate_tau else "")
        print(f"  {t:4.2f}  {A['sup'].mean():.3f}   {A['img_void'].mean():.3f}   "
              f"{A['fp_void'].mean():.3f}    {A['fp_coll'].mean():.3f}   "
              f"{crash:2d}/{n}  {A['sem'].mean()*s:+.4f}  {A['void_t'].mean()*s:+.4f}  "
              f"{A['coll_t'].mean()*s:+.4f}  {A['total'].mean()*s:+.4f}{mark}")

    # ---- detail blocks for ungated and the chosen gate ----
    for t in [t for t in (0.0, args.gate_tau) if t in per_tau]:
        rows = per_tau[t]
        A = {k: np.array([r[k] for r in rows], dtype=float)
             for k in rows[0] if k != "dom"}
        tag = "UNGATED" if t == 0.0 else f"GATED tau={t}"
        print(f"\n===== {tag} ({n} rendered steps) =====")
        print(f"  SUPPORT      frac(alpha>{t}) {A['sup'].mean():.3f}")
        print(f"  IMGVOID      whole-image void {A['img_void'].mean():.3f}   "
              f"(max over steps {A['img_void'].max():.3f})")
        print(f"  VOID (fp)    footprint void  {A['fp_void'].mean():.3f}   "
              f"(max {A['fp_void'].max():.3f})")
        print(f"  COLLISION    footprint coll  {A['fp_coll'].mean():.3f}   "
              f"steps >={args.crash_frac} (would CRASH): "
              f"{int((A['fp_coll'] >= args.crash_frac).sum())}/{n}")
        print(f"  REWARD x{s}  semantic {A['sem'].mean()*s:+.4f}   "
              f"goal {A['goal_t'].mean()*s:+.4f}   "
              f"collision {A['coll_t'].mean()*s:+.4f}   "
              f"void {A['void_t'].mean()*s:+.4f}   "
              f"total {A['total'].mean()*s:+.4f}")
        doms = {}
        for r in rows:
            doms[r["dom"]] = doms.get(r["dom"], 0) + 1
        print("  DOMINANT     (share of STEPS whose top class is) " + ", ".join(
            f"{k} {100.0 * v / n:.0f}%" for k, v in
            sorted(doms.items(), key=lambda kv: -kv[1])[:5]))

    # ---- every number, auditable ----
    import csv as _csv
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    stem = f"CHECK_{args.scene}{args.tag}"
    csv_p = od / f"{stem}.csv"
    with open(csv_p, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(list(per_tau[taus[0]][0].keys()))
        for t in taus:
            for r in per_tau[t]:
                w.writerow(list(r.values()))
    print(f"\n==> every per-step number: {csv_p}", flush=True)

    # ---- the visual: RGB | raster | labels ungated | labels gated | ... ----
    try:
        import cv2
        from diffsynth.utils.class_taxonomy import v14_palette
        pal = (v14_palette(args.sem_palette).numpy() * 255).astype(np.uint8)
        ep0 = [r for r in recs if r["ep"] == 0]
        u_rows = {(r["ep"], r["step"]): r for r in per_tau[0.0]}
        g_rows = {(r["ep"], r["step"]): r for r in per_tau[args.gate_tau]}
        for i, r in enumerate(ep0):
            rgb, lu, au = r["rgb"], r["lab"], r["alpha"]
            lg = lu if au is None else np.where(au > args.gate_tau, lu, 0)
            pu, pg = pal[np.clip(lu, 0, 13)], pal[np.clip(lg, 0, 13)]
            ras = np.zeros_like(rgb) if r["ras"] is None else r["ras"]
            psr = (np.zeros_like(rgb) if r["sras"] is None
                   else pal[np.clip(r["sras"], 0, 13).astype(int)])
            if au is not None:
                amax = float(np.percentile(au, 99)) or 1.0
                heat = cv2.applyColorMap(
                    (np.clip(au / max(amax, 1e-6), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            else:
                heat = np.zeros_like(rgb)
            panel = np.hstack([rgb, ras, pu, pg, psr, heat])[:, :, ::-1]
            panel = np.ascontiguousarray(panel)
            ru = u_rows[(r["ep"], r["step"])]; rg = g_rows[(r["ep"], r["step"])]
            hud = (f"step {i}  UNGATED coll {ru['fp_coll']:.2f} void "
                   f"{ru['fp_void']:.2f} | tau={args.gate_tau} coll "
                   f"{rg['fp_coll']:.2f} void {rg['fp_void']:.2f} | imgvoid "
                   f"{rg['img_void']:.2f} alpha_mean {rg['mean_alpha']:.2f}")
            cv2.putText(panel, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            for k, name in enumerate(["RGB diffused", "RGB raster (splats)",
                                      "SEM diffused ungated",
                                      f"SEM diffused gated tau={args.gate_tau}",
                                      "SEM raster (splat labels)",
                                      "SUPPORT (alpha)"]):
                cv2.putText(panel, name, (k * args.width + 8,
                                          panel.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imwrite(str(od / f"{stem}_step{i:02d}.png"), panel)
        print(f"==> visual frames: {od}/{stem}_step*.png", flush=True)
    except Exception as e:
        print(f"[viz] skipped: {e}", flush=True)

    uc = np.array([r["fp_coll"] for r in per_tau[0.0]])
    gc = np.array([r["fp_coll"] for r in per_tau[args.gate_tau]])
    print(f"\n===== GATE EFFECT at tau={args.gate_tau} (identical renders) =====")
    print(f"  crash-level steps  ungated "
          f"{int((uc >= args.crash_frac).sum())}/{n}   ->  gated "
          f"{int((gc >= args.crash_frac).sum())}/{n}")
    print(f"  mean collision     ungated {uc.mean():.3f}  ->  gated {gc.mean():.3f}")
    print(f"  mean img void      ungated "
          f"{np.mean([r['img_void'] for r in per_tau[0.0]]):.3f}  ->  gated "
          f"{np.mean([r['img_void'] for r in per_tau[args.gate_tau]]):.3f}")
    print("\n  Crash-level steps collapsing to 0 under the gate means those "
          "crashes were\n  phantom terrain in unobserved regions. It does NOT "
          "prove the gate keeps REAL\n  obstacles — for that, run the obstacle "
          "probe (--goal_xy at a known obstacle)\n  and check that crash steps "
          "SURVIVE the gate.", flush=True)


if __name__ == "__main__":
    main()
