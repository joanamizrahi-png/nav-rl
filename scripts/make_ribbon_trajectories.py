"""Write ribbon-cache sweep trajectories for a scene.

A sweep = 81 camera poses following the recorded path at ONE (lateral offset,
heading offset) combination. Each sweep renders as a single inference_semantic
call (--trajectory_file), so its 81 views share one generation — hallucinated
content stays coherent within the sweep. The full cache = all combinations:
default 5 lateral offsets x 8 headings (full ring, robot can spin) = 40 sweeps
~= 3240 cached views per scene.

Poses follow the recorded path frame-by-frame, so frame i renders at source
timestamp i — the dynamic-Gaussian time association (debugged 2026-07-21)
holds automatically.

Camera math replicates CalibratedRealWorldBackend.render() EXACTLY (mount
height + robot->camera axes + nav->recon), on the post-yaw-fix right-handed
frame. If render() changes, change this too.

Outputs (per scene, under --out_dir):
  sweep_lat{L}_yaw{Y}.json   -- CameraTrajectory matrix format, mode=global (recon frame)
  manifest.json              -- per-sweep: file, lateral_m, heading_deg, and the
                                nav-frame (x, y, yaw) of each of its 81 poses
                                (what the cache backend indexes for lookups)
  sweep_map.png              -- VERIFICATION GATE: top-down plot of the path +
                                every sweep pose with heading ticks. Eyeball
                                before rendering anything.

Usage (cluster):
  python scripts/make_ribbon_trajectories.py --scene rugd_trail_00 \
      --poses_dir .../outputs/poses --out_dir .../outputs/ribbon_traj/rugd_trail_00
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.env.real_calibrated import NavCalibration

# robot(x fwd, y left, z up) -> camera(x right, y down, z fwd); copy of the
# constant inside CalibratedRealWorldBackend.render().
R_CAM_LOCAL = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def sweep_poses(cal: NavCalibration, lateral_m: float, heading_deg: float):
    """(recon c2w [T,4,4], nav (x, y, yaw_deg) [T,3]) for one sweep."""
    r = np.deg2rad(heading_deg)
    c, s = np.cos(r), np.sin(r)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    mats, nav = [], []
    for i in range(len(cal.positions)):
        robot = np.asarray(cal.robot_pose_nav(i), dtype=np.float64)
        # lateral offset along the robot's +y (LEFT, right-handed frame)
        robot[:3, 3] += lateral_m * robot[:3, 1]
        # heading offset about world +z (matches SceneEnv._advance_pose)
        robot[:3, :3] = R_yaw @ robot[:3, :3]
        yaw = float(np.degrees(np.arctan2(robot[1, 0], robot[0, 0])))
        nav.append([float(robot[0, 3]), float(robot[1, 3]), yaw])
        # robot -> camera (render() math): mount height, then axes swap
        c2w_nav = robot.copy()
        c2w_nav[:3, 3] += np.array([0.0, 0.0, cal.camera_height_m])
        c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_CAM_LOCAL
        mats.append(cal.nav_cam_to_recon_cam(c2w_nav))
    return np.stack(mats), np.array(nav)


def spin_poses(cal: NavCalibration, anchor_i: int, lateral_m: float):
    """(recon c2w [T,4,4], nav (x, y, yaw_deg) [T,3]) for one SPIN sweep:
    camera parked at path frame `anchor_i` (+ lateral), heading rotating a
    full 360 deg across the sweep's frames (~4.4 deg/frame — the pan rate the
    TRUE360 keyframe spin validated). Scene-time is FROZEN at the anchor
    frame (all frame_indices = anchor_i): fine for static scenes; revisit
    for dynamic ones (SCAND pedestrians).

    Why spins (2026-08-17): path-threaded sweeps made TURNING cross
    diffusion calls exactly where alpha=0 — 24 unrelated dreams per ring
    (the camera-dream slideshow). Spin sweeps put rotation WITHIN one call
    (one coherent dream per spot); walking crosses calls only at forward
    headings where ~95% real geometry pins consecutive calls together."""
    n = len(cal.positions)
    base = np.asarray(cal.robot_pose_nav(anchor_i), dtype=np.float64)
    base[:3, 3] += lateral_m * base[:3, 1]
    mats, nav = [], []
    for f in range(n):
        r = np.deg2rad(f * 360.0 / n)
        c, s = np.cos(r), np.sin(r)
        R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        robot = base.copy()
        robot[:3, :3] = R_yaw @ robot[:3, :3]
        yaw = float(np.degrees(np.arctan2(robot[1, 0], robot[0, 0])))
        nav.append([float(robot[0, 3]), float(robot[1, 3]), yaw])
        c2w_nav = robot.copy()
        c2w_nav[:3, 3] += np.array([0.0, 0.0, cal.camera_height_m])
        c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_CAM_LOCAL
        mats.append(cal.nav_cam_to_recon_cam(c2w_nav))
    return np.stack(mats), np.array(nav)


class DensifiedCal:
    """NavCalibration view with anchors RESAMPLED every step_m along the path.

    Fixes the coarse-clip artifact (GND: ~1 m between recorded frames -> the
    robot walks 3-4 env steps before the nearest cached view changes). Anchors
    interpolate position along the recorded path; heading = local tangent;
    scene time (frame_of) = nearest recorded frame, so the dynamic-Gaussian
    time association still holds.
    """

    def __init__(self, cal: NavCalibration, step_m: float):
        self.cal = cal
        self.camera_height_m = cal.camera_height_m
        p = cal.positions[:, :2]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(s[-1] / step_m) + 1)
        ts = np.linspace(0.0, s[-1] - 1e-6, n)
        self._poses, self._frames, pos = [], [], []
        for t in ts:
            i = int(np.searchsorted(s, t)) - 1
            i = max(0, min(i, len(p) - 2))
            f = (t - s[i]) / max(s[i + 1] - s[i], 1e-9)
            xy = p[i] + f * (p[i + 1] - p[i])
            d = p[i + 1] - p[i]
            yaw = np.arctan2(d[1], d[0])
            P = np.asarray(cal.robot_pose_nav(i), dtype=np.float64).copy()
            c, si = np.cos(yaw), np.sin(yaw)
            P[:3, :3] = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
            P[0, 3], P[1, 3] = float(xy[0]), float(xy[1])   # z stays frame i's
            self._poses.append(P)
            self._frames.append(int(i if f < 0.5 else i + 1))
            pos.append([float(xy[0]), float(xy[1]), float(P[2, 3])])
        self.positions = np.array(pos)

    def robot_pose_nav(self, i):
        # copy: callers offset the returned pose in place (fan/sweep lanes);
        # returning the stored array let lane offsets accumulate onto the path
        return self._poses[i].copy()

    def frame_of(self, i):
        return self._frames[i]

    def nav_cam_to_recon_cam(self, m):
        return self.cal.nav_cam_to_recon_cam(m)


def fan_poses(cal: NavCalibration, seg_start: int, seg_len: int,
              lateral_m: float, fan_deg: float, fan_steps: int):
    """(recon c2w [T,4,4], nav [T,3], frame_indices [T]) for one FAN cell:
    seg_len consecutive path positions x fan_steps headings (+-fan_deg around
    the path tangent), serpentining through heading at each position so the
    in-call camera motion stays smooth.

    Why fans (2026-08-19): the policy's action = forward + turn(+-17 deg),
    but lane sweeps put every heading in its own diffusion call (15-deg
    families) -> EVERY turn action swaps dreams (the tour flicker). A fan
    cell keeps a whole turn-and-advance neighborhood inside ONE call; seams
    move to segment/lane boundaries, which forward motion crosses rarely and
    sticky lookup smooths."""
    offs = np.linspace(-fan_deg, fan_deg, fan_steps)
    mats, nav, fidx = [], [], []
    for k, fi in enumerate(range(seg_start, seg_start + seg_len)):
        base = np.array(cal.robot_pose_nav(fi), dtype=np.float64)
        base[:3, 3] += lateral_m * base[:3, 1]
        for hd in (offs if k % 2 == 0 else offs[::-1]):
            r = np.deg2rad(hd)
            c, s = np.cos(r), np.sin(r)
            R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            robot = base.copy()
            robot[:3, :3] = R_yaw @ robot[:3, :3]
            yaw = float(np.degrees(np.arctan2(robot[1, 0], robot[0, 0])))
            nav.append([float(robot[0, 3]), float(robot[1, 3]), yaw])
            c2w_nav = robot.copy()
            c2w_nav[:3, 3] += np.array([0.0, 0.0, cal.camera_height_m])
            c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_CAM_LOCAL
            mats.append(cal.nav_cam_to_recon_cam(c2w_nav))
            # scene-time follows the position (densified cal maps anchor ->
            # nearest recorded frame)
            fidx.append(cal.frame_of(fi) if hasattr(cal, "frame_of") else int(fi))
    return np.stack(mats), np.array(nav), fidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--laterals", default="-1.5,-0.75,0,0.75,1.5",
                    help="meters, +left (comma list)")
    ap.add_argument("--headings", default="0,45,90,135,180,225,270,315",
                    help="degrees offset from path heading (comma list; full "
                         "ring because the robot can spin in place)")
    ap.add_argument("--spin", action="store_true",
                    help="SPIN sweeps (one 360 per anchor x lane) instead of "
                         "path-threaded sweeps; --headings is ignored")
    ap.add_argument("--anchor_stride", type=int, default=2,
                    help="spin mode: place an anchor every Nth path frame")
    ap.add_argument("--fan", action="store_true",
                    help="FAN cells (segment x heading-fan per call) instead "
                         "of path-threaded sweeps; --headings is ignored")
    ap.add_argument("--fan_deg", type=float, default=40.0)
    ap.add_argument("--fan_steps", type=int, default=9)
    ap.add_argument("--seg_len", type=int, default=9)
    ap.add_argument("--seg_starts", default=None,
                    help="fan mode: comma list of segment start frames "
                         "(pilot: one cell). Default: tile the whole path")
    ap.add_argument("--densify_m", type=float, default=None,
                    help="fan mode: resample anchors every N meters along the "
                         "path (0.25 recommended for coarse clips like GND's "
                         "~1 m/frame — kills the frozen-observation artifact). "
                         "Default: anchors at recorded frames")
    args = ap.parse_args()

    cal = NavCalibration.from_npz(Path(args.poses_dir) / f"{args.scene}_poses.npz")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    laterals = [float(x) for x in args.laterals.split(",")]
    headings = [float(x) for x in args.headings.split(",")]
    manifest = {"scene": args.scene, "num_frames": len(cal.positions), "sweeps": []}

    if args.fan:
        if args.densify_m:
            cal = DensifiedCal(cal, args.densify_m)
            print(f"densified anchors: {len(cal.positions)} at "
                  f"{args.densify_m} m spacing (from "
                  f"{len(cal.cal.positions)} recorded frames)")
        n_call = args.seg_len * args.fan_steps
        # The video VAE compresses time 4x -> pose counts must be 4k+1
        # (81, 121, 153, ...). 81 = the training window; anything larger is
        # EXPERIMENTAL (model never trained there — judge outputs by eye).
        if n_call % 4 != 1:
            sys.exit(f"fan cell = seg_len*fan_steps = {n_call} poses; video "
                     f"VAE needs 4k+1 frames (81, 121, 153, ...)")
        if n_call != 81:
            print(f"WARNING: {n_call}-pose cells exceed the 81-frame training "
                  f"window — experimental, eyeball before trusting")
        if args.seg_starts:
            starts = [int(x) for x in args.seg_starts.split(",")]
        else:
            starts = list(range(0, len(cal.positions) - args.seg_len + 1,
                                args.seg_len))
        for lat in laterals:
            for s0 in starts:
                mats, nav, fidx = fan_poses(cal, s0, args.seg_len, lat,
                                            args.fan_deg, args.fan_steps)
                name = f"fan_f{s0:02d}-{s0 + args.seg_len - 1:02d}_lat{lat:+.2f}"
                with open(out / f"{name}.json", "w") as f:
                    json.dump({
                        "mode": "global",
                        "num_frames": int(mats.shape[0]),
                        "name": name,
                        "trajectory": {
                            "frame_indices": fidx,
                            "frame_matrices": mats.tolist(),
                        },
                    }, f)
                manifest["sweeps"].append({
                    "file": f"{name}.json",
                    "seg_start": s0,
                    "lateral_m": lat,
                    "fan": True,
                    "nav_xyyaw": nav.tolist(),
                })
        print(f"FAN grid: {len(starts)} segments x {len(laterals)} lanes "
              f"= {len(manifest['sweeps'])} cells "
              f"(+-{args.fan_deg} deg in {args.fan_steps} steps)")
    elif args.spin:
        anchors = list(range(0, len(cal.positions), args.anchor_stride))
        for lat in laterals:
            for ai in anchors:
                mats, nav = spin_poses(cal, ai, lat)
                name = f"spin_f{ai:02d}_lat{lat:+.2f}"
                with open(out / f"{name}.json", "w") as f:
                    json.dump({
                        "mode": "global",
                        "num_frames": int(mats.shape[0]),
                        "name": name,
                        "trajectory": {
                            # time frozen at the anchor: every rendered frame
                            # uses the anchor's source timestamp
                            "frame_indices": [int(ai)] * int(mats.shape[0]),
                            "frame_matrices": mats.tolist(),
                        },
                    }, f)
                manifest["sweeps"].append({
                    "file": f"{name}.json",
                    "anchor_frame": ai,
                    "lateral_m": lat,
                    "spin": True,
                    "nav_xyyaw": nav.tolist(),
                })
        print(f"SPIN grid: {len(anchors)} anchors x {len(laterals)} lanes "
              f"= {len(manifest['sweeps'])} sweeps")
    else:
        for lat in laterals:
            for yaw in headings:
                mats, nav = sweep_poses(cal, lat, yaw)
                name = f"sweep_lat{lat:+.2f}_yaw{yaw:03.0f}"
                with open(out / f"{name}.json", "w") as f:
                    json.dump({
                        "mode": "global",
                        "num_frames": int(mats.shape[0]),
                        "name": name,
                        "trajectory": {
                            "frame_indices": list(range(mats.shape[0])),
                            "frame_matrices": mats.tolist(),
                        },
                    }, f)
                manifest["sweeps"].append({
                    "file": f"{name}.json",
                    "lateral_m": lat,
                    "heading_deg": yaw,
                    "nav_xyyaw": nav.tolist(),
                })
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"{len(manifest['sweeps'])} sweeps x {manifest['num_frames']} poses -> {out}")

    # ---- verification gate: top-down map ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(cal.positions[:, 0], cal.positions[:, 1], "k-", lw=2, label="recorded path")
        # subsample for legibility on full grids; small pilots plot EVERY pose
        # (a fan cell stacks 9 headings on each position — subsampling hides them)
        step = 8 if len(manifest["sweeps"]) > 10 else 1
        for sw in manifest["sweeps"]:
            nav = np.array(sw["nav_xyyaw"])
            pts = nav[::step]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5)
            for x, y, yaw in pts[::2 if step > 1 else 1]:
                r = np.deg2rad(yaw)
                ax.plot([x, x + 0.3 * np.cos(r)], [y, y + 0.3 * np.sin(r)],
                        "-", lw=0.5, alpha=0.4, color="gray")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{args.scene}: ribbon sweeps (dots = poses, ticks = headings)\n"
                     f"GATE: comb of lines along the path, ticks fanning all directions")
        fig.savefig(out / "sweep_map.png", dpi=130, bbox_inches="tight")
        print(f"gate plot: {out / 'sweep_map.png'}")
    except ImportError:
        print("matplotlib unavailable — gate plot skipped")


if __name__ == "__main__":
    main()
