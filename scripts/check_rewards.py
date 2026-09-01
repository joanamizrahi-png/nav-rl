"""Reward mechanism check: GATE / VOID / IMGVOID, measured not argued.

Renders the SAME live-diffusion views training uses, at poses drawn from the
J-spec distribution, and prints for each step what every reward mechanism
actually computes — with the alpha gate ON and OFF over identical poses.

Answers, with numbers:
  * ALPHA        what the support distribution looks like (p10..p90), so the
                 0.5 gate threshold stops being an inherited guess
  * GATE         how many pixels the gate converts to void, and whether the
                 gated/ungated label maps differ ONLY there
  * IMGVOID      whole-image void share (her measure) next to mean alpha (the
                 `cov` number the spin certificates print) — they are different
                 statistics and nobody had compared them
  * VOID         footprint void share, and proof it is EXCLUDED from
                 collision_frac when void_cost > 0
  * REWARD       every term at the real x0.01 scale, so we can see what
                 dominates and whether phantom terrain is driving crashes

Usage (GPU node):
    python scripts/check_rewards.py --scene gnd_AUw360 \
        --clips_dir ... --poses_dir ... --labels_dir ... \
        --live_ckpt ... --trav_path config/traversability_v14_walkway.yaml
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
)

V14_NAMES = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
             "road", "pavement", "stairs", "obstacle", "vegetation", "person",
             "vehicle"]


def episode_poses(cal, rng, n_steps, cone_deg, dist_range, yaw_jit, lat_jit,
                  spawn_min=10):
    """One J-spec episode: jittered spawn on the recorded path, goal in the
    tangent cone, then a straight walk toward it in 0.3 m steps."""
    path = np.asarray(cal.positions, dtype=float)[:, :2]
    f = int(rng.integers(spawn_min, max(len(path) - 6, spawn_min + 1)))
    fw = path[min(f + 1, len(path) - 1)] - path[max(f - 1, 0)]
    base = float(np.arctan2(fw[1], fw[0]))
    p = path[f] + rng.uniform(-1, 1) * lat_jit * np.array(
        [-np.sin(base), np.cos(base)])
    yaw = base + np.deg2rad(rng.uniform(-1, 1) * yaw_jit)
    th = base + np.deg2rad(rng.uniform(-1, 1) * cone_deg / 2.0)
    goal = p + rng.uniform(*dist_range) * np.array([np.cos(th), np.sin(th)])

    def pose_at(xy, heading):
        c, s = np.cos(heading), np.sin(heading)
        m = np.eye(4, dtype=np.float32)
        m[:3, 0] = (c, s, 0.0)
        m[:3, 1] = (-s, c, 0.0)
        m[:3, 3] = (xy[0], xy[1], 0.0)
        return m

    u = (goal - p) / max(np.linalg.norm(goal - p), 1e-6)
    poses = [pose_at(p, yaw)]
    for k in range(1, n_steps):
        poses.append(pose_at(p + u * 0.3 * k, float(np.arctan2(u[1], u[0]))))
    return poses, goal


def run(args, gate: bool):
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
                                       alpha_gate=gate)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]

    trav = load_traversability(Path(args.trav_path) if args.trav_path else None)
    non_trav = trav <= args.collision_threshold
    weights = RewardWeights(semantic=args.semantic_weight, goal=args.goal_weight,
                            collision=1.0, step_cost=0.05, void_cost=0.3,
                            terrain_as_cost=True)

    rng = np.random.default_rng(args.seed)
    rows, keep = [], []
    for ep in range(args.episodes):
        poses, goal = episode_poses(
            cal, rng, args.steps, args.cone_deg,
            tuple(float(v) for v in args.dist_range.split(",")),
            args.spawn_yaw_jitter, args.spawn_lat_jitter)
        prev = None
        for si, pose in enumerate(poses):
            (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
            alpha = getattr(world, "last_alpha", None)
            a = alpha[0] if alpha else None
            pos = pose[:2, 3]
            b = compute_reward(
                semantic_image=lab, K=K, w2c=w2c,
                robot_position=np.array([pos[0], pos[1], 0.0]),
                robot_heading=pose[:3, 0],
                goal=np.array([goal[0], goal[1], 0.0]),
                traversability_scores=trav, non_traversable_mask=non_trav,
                previous_position=prev, look_ahead_dist=1.5,
                body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH,
                weights=weights)
            prev = np.array([pos[0], pos[1], 0.0])
            if ep == 0:                      # frames for the visual check
                _ras = getattr(world, "last_raster", None)
                _sr = getattr(world, "last_sem_raster", None)
                keep.append((rgb.copy(), lab.copy(),
                             None if a is None else a.copy(),
                             None if not _ras else _ras[0].copy(),
                             None if not _sr else _sr[0].copy()))
            cls, cnt = np.unique(lab, return_counts=True)
            top = V14_NAMES[int(cls[np.argmax(cnt)])]
            rows.append(dict(
                ep=ep, step=si,
                img_void=float((lab == 0).mean()),
                mean_alpha=float(a.mean()) if a is not None else float("nan"),
                a50=float(np.percentile(a, 50)) if a is not None else float("nan"),
                a10=float(np.percentile(a, 10)) if a is not None else float("nan"),
                a90=float(np.percentile(a, 90)) if a is not None else float("nan"),
                sup=float((a > 0.5).mean()) if a is not None else float("nan"),
                fp_void=float(b.void_frac),
                fp_coll=float(-b.collision / max(weights.collision, 1e-6)),
                sem=float(b.semantic), goal_t=float(b.goal),
                coll_t=float(b.collision), void_t=float(b.void),
                total=float(b.total), dom=top))
            if si == 0 and ep == 0:
                print(f"  [{'GATED' if gate else 'ungated'}] alpha percentiles "
                      f"p10 {rows[-1]['a10']:.2f}  p50 {rows[-1]['a50']:.2f}  "
                      f"p90 {rows[-1]['a90']:.2f}", flush=True)
    return rows, keep


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
    ap.add_argument("--reward_scale", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default="/scratch/m000204-pm06b/joana/outputs/reward_check")
    args = ap.parse_args()

    print(f"=== {args.scene} | trav {args.trav_path} | "
          f"cone {args.cone_deg} | jitter {args.spawn_yaw_jitter}deg/"
          f"{args.spawn_lat_jitter}m | reward_scale {args.reward_scale}",
          flush=True)

    out, frames = {}, {}
    for gate in (False, True):
        tag = "gated" if gate else "ungated"
        out[tag], frames[tag] = run(args, gate)

    # ---- every number, auditable ----
    import csv as _csv
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    csv_p = od / f"CHECK_{args.scene}.csv"
    with open(csv_p, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["mode"] + list(out["ungated"][0].keys()))
        for tag, rows in out.items():
            for r in rows:
                w.writerow([tag] + list(r.values()))
    print(f"\n==> every per-step number: {csv_p}", flush=True)

    # ---- the visual: RGB | labels ungated | labels gated | support ----
    try:
        import cv2
        from diffsynth.utils.class_taxonomy import v14_palette
        pal = (v14_palette(args.sem_palette).numpy() * 255).astype(np.uint8)
        for i, (fu, fg) in enumerate(zip(frames["ungated"], frames["gated"])):
            rgb, lu, au, ras, sras = fu
            lg = fg[1]
            pu, pg = pal[np.clip(lu, 0, 13)], pal[np.clip(lg, 0, 13)]
            ras = np.zeros_like(rgb) if ras is None else ras
            psr = (np.zeros_like(rgb) if sras is None
                   else pal[np.clip(sras, 0, 13).astype(int)])
            if au is not None:
                amax = float(np.percentile(au, 99)) or 1.0
                heat = cv2.applyColorMap(
                    (np.clip(au / max(amax, 1e-6), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            else:
                heat = np.zeros_like(rgb)
            panel = np.hstack([rgb, ras, pu, pg, psr, heat])[:, :, ::-1]
            panel = np.ascontiguousarray(panel)
            ru = out["ungated"][i]; rg = out["gated"][i]
            hud = (f"step {i}  UNGATED coll {ru['fp_coll']:.2f} void "
                   f"{ru['fp_void']:.2f} | GATED coll {rg['fp_coll']:.2f} void "
                   f"{rg['fp_void']:.2f} | imgvoid {rg['img_void']:.2f} "
                   f"alpha_mean {rg['mean_alpha']:.2f}")
            cv2.putText(panel, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            for k, name in enumerate(["RGB diffused", "RGB raster (splats)",
                                      "SEM diffused ungated",
                                      "SEM diffused gated",
                                      "SEM raster (splat labels)",
                                      "SUPPORT (alpha)"]):
                cv2.putText(panel, name, (k * args.width + 8,
                                          panel.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imwrite(str(od / f"CHECK_{args.scene}_step{i:02d}.png"), panel)
        print(f"==> visual frames: {od}/CHECK_{args.scene}_step*.png", flush=True)
    except Exception as e:
        print(f"[viz] skipped: {e}", flush=True)

    for tag, rows in out.items():
        A = {k: np.array([r[k] for r in rows], dtype=float)
             for k in rows[0] if k != "dom"}
        s = args.reward_scale
        print(f"\n===== {tag.upper()} ({len(rows)} rendered steps) =====")
        print(f"  ALPHA        mean {A['mean_alpha'].mean():.3f}   "
              f"p10 {A['a10'].mean():.3f}  p50 {A['a50'].mean():.3f}  "
              f"p90 {A['a90'].mean():.3f}")
        print(f"  SUPPORT      frac(alpha>0.5) {A['sup'].mean():.3f}    "
              f"<- the gate's threshold, as a share of pixels")
        print(f"  IMGVOID      whole-image void {A['img_void'].mean():.3f}   "
              f"(max over steps {A['img_void'].max():.3f})")
        print(f"  VOID (fp)    footprint void  {A['fp_void'].mean():.3f}   "
              f"(max {A['fp_void'].max():.3f})")
        print(f"  COLLISION    footprint coll  {A['fp_coll'].mean():.3f}   "
              f"steps >=0.35 (would CRASH): "
              f"{int((A['fp_coll'] >= 0.35).sum())}/{len(rows)}")
        print(f"  REWARD x{s}  semantic {A['sem'].mean() * s:+.4f}   "
              f"goal {A['goal_t'].mean() * s:+.4f}   "
              f"collision {A['coll_t'].mean() * s:+.4f}   "
              f"void {A['void_t'].mean() * s:+.4f}   "
              f"total {A['total'].mean() * s:+.4f}")
        doms = {}
        for r in rows:
            doms[r["dom"]] = doms.get(r["dom"], 0) + 1
        print("  DOMINANT     " + ", ".join(
            f"{k} {100.0 * v / len(rows):.0f}%" for k, v in
            sorted(doms.items(), key=lambda kv: -kv[1])[:5]))

    g, u = out["gated"], out["ungated"]
    gc = np.array([r["fp_coll"] for r in g])
    uc = np.array([r["fp_coll"] for r in u])
    print("\n===== GATE EFFECT (identical poses, identical seed) =====")
    print(f"  crash-level steps  ungated {int((uc >= 0.35).sum())}/{len(uc)}"
          f"   ->  gated {int((gc >= 0.35).sum())}/{len(gc)}")
    print(f"  mean collision     ungated {uc.mean():.3f}  ->  gated {gc.mean():.3f}")
    print(f"  mean img void      ungated "
          f"{np.mean([r['img_void'] for r in u]):.3f}  ->  gated "
          f"{np.mean([r['img_void'] for r in g]):.3f}")
    print("\n  If gated crash-level steps collapse toward 0 while the ungated "
          "run shows many,\n  the crashes were phantom terrain in unobserved "
          "regions — not real obstacles.", flush=True)


if __name__ == "__main__":
    main()
