"""Reconstructed camera path vs recorded odometry, frame by frame.

The reconstructor guesses each frame's camera from the images. The odometry
says where the robot really was. This script lines the two paths up (one
similarity transform: rotation, translation, scale -- Umeyama) and reports the
residual per frame: position error in metres and heading error in degrees.

Where the residual is small the reconstruction placed that frame's Gaussians
correctly; where it is large the geometry from that frame is misplaced (this is
what fuses into ghosting) and spawns/goals should avoid it. The spread of the
residuals gives the Mahalanobis scales used to bound exploration.

Inputs (both written by the standard pipeline):
  <scene>_poses.npz   from extract_poses.py: positions (T,3) z-up metres,
                      headings (T,3) unit forward vectors
  <scene>_odom.npz    from prepare_rosbag_clips.py: t, xyz (N,3), quat (N,4),
                      frame_stamps (T,) seconds -- the odometry time of each
                      of the T sampled frames

Outputs (next to --out_dir/<scene>_pose_vs_odom.*):
  .csv   frame, t_sec, ex_m (along-track), ey_m (lateral), e_pos_m, e_yaw_deg,
         mahalanobis
  .png   residuals along the walk + the two paths overlaid
  .json  summary: mean/median/max errors, scales (along, lateral, yaw),
         frames above --maha_thresh

    python scripts/pose_vs_odom.py --poses clips/campusA_00_poses.npz \
        --odom clips/campusA_00_odom.npz --out_dir out/pose_vs_odom
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def odom_at_frames(od: dict, T: int):
    """Odometry pose (xy, yaw) at each sampled frame time, by interpolation."""
    t = np.asarray(od["t"], dtype=np.float64)
    xyz = np.asarray(od["xyz"], dtype=np.float64)
    yaw = np.unwrap(quat_to_yaw(np.asarray(od["quat"], dtype=np.float64)))
    fs = od.get("frame_stamps")
    if fs is None or len(fs) != T:
        # older odom files: assume frames are evenly spread over the window
        fs = np.linspace(t[0], t[-1], T)
    fs = np.asarray(fs, dtype=np.float64)
    xy = np.stack([np.interp(fs, t, xyz[:, 0]), np.interp(fs, t, xyz[:, 1])], 1)
    yw = np.interp(fs, t, yaw)
    return fs, xy, yw


def umeyama_2d(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform s*R @ src + t that best maps src onto dst (2D)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1.0, d])
    R = U @ Dm @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(sig) @ Dm) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, type=Path)
    ap.add_argument("--odom", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--no_scale", action="store_true",
                    help="rigid fit only (poses already metric); default fits scale too")
    ap.add_argument("--maha_thresh", type=float, default=3.0,
                    help="frames with Mahalanobis distance above this are flagged")
    args = ap.parse_args()

    P = np.load(args.poses)
    # poses.npz is in the reconstruction's scene frame (y mirrored w.r.t. ROS);
    # NavCalibration applies diag(1,-1,1) on load -- do the same here
    F = np.array([1.0, -1.0, 1.0])
    pos = (np.asarray(P["positions"], dtype=np.float64) * F)[:, :2]
    hd = np.asarray(P["headings"], dtype=np.float64) * F
    yaw_rec = np.unwrap(np.arctan2(hd[:, 1], hd[:, 0]))
    T = len(pos)
    od = np.load(args.odom)
    fs, xy_od, yaw_od = odom_at_frames(od, T)

    s, R, t = umeyama_2d(pos, xy_od, with_scale=not args.no_scale)
    pos_al = (s * (R @ pos.T)).T + t
    rot = math.atan2(R[1, 0], R[0, 0])
    yaw_al = yaw_rec + rot

    # residuals in the odometry frame, then split along/lateral to the walk
    e = xy_od - pos_al                                        # (T,2)
    fwd = np.stack([np.cos(yaw_od), np.sin(yaw_od)], 1)
    lat = np.stack([-np.sin(yaw_od), np.cos(yaw_od)], 1)
    ex = (e * fwd).sum(1)                                     # along-track
    ey = (e * lat).sum(1)                                     # lateral
    e_pos = np.linalg.norm(e, axis=1)
    e_yaw = np.degrees((yaw_al - yaw_od + np.pi) % (2 * np.pi) - np.pi)

    # Mahalanobis distance of each frame's error under the walk's error spread
    E = np.stack([ex, ey, np.radians(e_yaw)], 1)
    mu = E.mean(0)
    cov = np.cov(E.T) + 1e-9 * np.eye(3)
    icov = np.linalg.inv(cov)
    maha = np.sqrt(np.einsum("ij,jk,ik->i", E - mu, icov, E - mu))
    flagged = np.nonzero(maha > args.maha_thresh)[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / (args.poses.stem.replace("_poses", "") + "_pose_vs_odom")
    with open(f"{stem}.csv", "w") as fh:
        fh.write("frame,t_sec,ex_m,ey_m,e_pos_m,e_yaw_deg,mahalanobis\n")
        for i in range(T):
            fh.write(f"{i},{fs[i]:.3f},{ex[i]:.3f},{ey[i]:.3f},{e_pos[i]:.3f},"
                     f"{e_yaw[i]:.2f},{maha[i]:.2f}\n")
    summary = {
        "frames": int(T),
        "fit_scale": float(s), "fit_rot_deg": float(math.degrees(rot)),
        "pos_err_m": {"mean": float(e_pos.mean()), "median": float(np.median(e_pos)),
                      "max": float(e_pos.max())},
        "yaw_err_deg": {"mean": float(np.abs(e_yaw).mean()), "median": float(np.median(np.abs(e_yaw))),
                        "max": float(np.abs(e_yaw).max())},
        "scales": {"along_m": float(ex.std()), "lateral_m": float(ey.std()),
                   "yaw_deg": float(e_yaw.std())},
        "maha_thresh": float(args.maha_thresh),
        "flagged_frames": flagged.tolist(),
    }
    with open(f"{stem}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(pos_al[:, 0], pos_al[:, 1], "o-", ms=3, label="reconstructed (aligned)")
        ax[0].plot(xy_od[:, 0], xy_od[:, 1], "s-", ms=3, label="odometry")
        if len(flagged):
            ax[0].plot(xy_od[flagged, 0], xy_od[flagged, 1], "rx", ms=9, label="flagged frames")
        ax[0].set_aspect("equal"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[0].set_title("paths")
        ax[1].plot(e_pos, label="position error [m]")
        ax[1].plot(np.abs(e_yaw) / 10.0, label="|heading error| / 10 [deg]")
        ax[1].plot(maha, ":", label="Mahalanobis")
        ax[1].axhline(args.maha_thresh, color="r", lw=0.8)
        ax[1].set_xlabel("frame"); ax[1].legend(); ax[1].grid(alpha=0.3)
        ax[1].set_title("residual along the walk")
        fig.suptitle(stem.name)
        fig.savefig(f"{stem}.png", dpi=140, bbox_inches="tight")
    except Exception as ex_:
        print(f"[pose_vs_odom] no plot ({ex_})")

    print(f"[pose_vs_odom] {args.poses.stem.replace('_poses', ''):<16} {T} frames | scale {s:.3f} rot {math.degrees(rot):.1f} deg | "
          f"pos err mean {e_pos.mean():.2f} m max {e_pos.max():.2f} m | "
          f"yaw err mean {np.abs(e_yaw).mean():.1f} deg max {np.abs(e_yaw).max():.1f} deg | "
          f"scales along {ex.std():.2f} m lateral {ey.std():.2f} m yaw {e_yaw.std():.1f} deg | "
          f"flagged {len(flagged)} frames > {args.maha_thresh}")
    print(f"[pose_vs_odom] wrote {stem}.csv/.json/.png")


if __name__ == "__main__":
    main()
