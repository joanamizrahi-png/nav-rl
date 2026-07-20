"""Round-trip tests for NavCalibration (nav metric frame <-> recon frame).

Run on the Mac (numpy only): python scripts/test_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))   # nav-rl's own extract_poses.py

from src.env.real_calibrated import NavCalibration
from src.env.real_backend import R_SCENE_TO_RECON


def test_point_roundtrip_real_npz():
    npz = REPO_ROOT / "data/poses/rugd_trail_00_poses.npz"
    cal = NavCalibration.from_npz(npz)
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 5, (100, 3))
    for p in pts:
        assert np.allclose(cal.recon_to_nav_point(cal.nav_to_recon_point(p)), p, atol=1e-9)
    # Trajectory consistency: converting the npz's own camera positions
    # (nav frame, = positions + mount height) into recon and back is identity.
    d = np.load(npz)
    cam_nav = d["cam_positions"].astype(np.float64)
    for p in cam_nav[::10]:
        assert np.allclose(cal.recon_to_nav_point(cal.nav_to_recon_point(p)), p, atol=1e-9)
    print("point round-trip on real trail_00 calibration: OK")


def test_full_loop_against_extract_poses():
    """Synthetic recon world -> extract_poses forward math -> NavCalibration
    inverse must recover the original recon camera pose."""
    from extract_poses import poses_from_c2w_recon  # NeoVerse/scripts

    rng = np.random.default_rng(1)
    T, h_units, step = 12, 0.05, 0.03
    c2w_scene = np.tile(np.eye(4), (T, 1, 1))
    c2w_scene[:, :3, 0] = [0, 1, 0]
    c2w_scene[:, :3, 1] = [0, 0, -1]
    c2w_scene[:, :3, 2] = [1, 0, 0]
    for t in range(T):
        c2w_scene[t, :3, 3] = [t * step, 0.01 * t, h_units]
    ground = np.c_[rng.uniform(-1, 2, 4000), rng.uniform(-1, 1, 4000), rng.normal(0, 1e-4, 4000)]
    R_sr = R_SCENE_TO_RECON.astype(np.float64)
    c2w_recon = c2w_scene.copy()
    c2w_recon[:, :3, :3] = R_sr @ c2w_scene[:, :3, :3]
    c2w_recon[:, :3, 3] = c2w_scene[:, :3, 3] @ R_sr.T
    means_recon = ground @ R_sr.T

    out = poses_from_c2w_recon(c2w_recon, means_recon, camera_height_m=0.6)

    # Build a NavCalibration from the same fields extract_poses saves.
    class D(dict):
        pass
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, **{k: out[k] for k in
                            ("positions", "headings", "plane_normal_scene",
                             "plane_offset_scene", "scale_m_per_unit")},
                 camera_height_m=np.float32(0.6))
        path = f.name
    cal = NavCalibration.from_npz(path)
    os.unlink(path)

    # nav camera pose (from extract output c2w) -> recon must equal the input.
    c2w_nav = out["c2w"].astype(np.float64)
    for t in range(T):
        rec = cal.nav_cam_to_recon_cam(c2w_nav[t])
        assert np.allclose(rec[:3, :3], c2w_recon[t, :3, :3], atol=1e-5), t
        assert np.allclose(rec[:3, 3], c2w_recon[t, :3, 3], atol=2e-3), \
            (t, rec[:3, 3], c2w_recon[t, :3, 3])

    # Projection invariance through the homogeneous map: a nav-frame point
    # projected with (w2c_recon @ M) equals projecting the recon point directly.
    K = np.array([[500, 0, 280], [0, 500, 168], [0, 0, 1.0]])
    M = cal.nav_world_to_recon_homogeneous()
    p_nav = np.array([2.0, 0.3, 0.0, 1.0])
    p_recon = np.r_[cal.nav_to_recon_point(p_nav[:3]), 1.0]
    w2c_recon0 = np.linalg.inv(c2w_recon[0])
    a = (w2c_recon0 @ M @ p_nav)[:3]
    b = (w2c_recon0 @ p_recon)[:3]
    uv_a = (K @ (a / a[2]))[:2]
    uv_b = (K @ (b / b[2]))[:2]
    assert np.allclose(uv_a, uv_b, atol=1e-6), (uv_a, uv_b)
    print("full loop vs extract_poses + homogeneous projection: OK")


def test_robot_pose_frame():
    npz = REPO_ROOT / "data/poses/rugd_trail_00_poses.npz"
    cal = NavCalibration.from_npz(npz)
    pose = cal.robot_pose_nav(0)
    R = pose[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "rotation not orthonormal"
    assert np.linalg.det(R) > 0.99, "not right-handed"
    assert pose[2, 3] == 0.0, "robot must start on the ground"
    assert abs(pose[2, 0]) < 1e-6, "heading must be horizontal"
    print("robot start pose frame: OK")


if __name__ == "__main__":
    test_point_roundtrip_real_npz()
    test_full_loop_against_extract_poses()
    test_robot_pose_frame()
    print("ALL CALIBRATION TESTS PASSED")
