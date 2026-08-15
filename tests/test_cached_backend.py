"""Synthetic-cache regression test for CachedDiffusedBackend.

Builds a miniature fake cache (2 sweeps, 8x8 frames, values encoding
(sweep, frame) with codec-proof spacing) and verifies:
  1. exact-pose lookup returns exactly that view (telemetry ~0),
  2. heading outweighs position in nearest-neighbor choice,
  3. alpha=False pixels are served as void (class 0) — the reward-safety property,
  4. telemetry reports honest position/heading errors.

Run: /path/to/python-with-torch-cv2 tests/test_cached_backend.py
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = "mock_scene"
    T = 81
    positions = np.stack([np.linspace(0, 8, T), np.zeros(T), np.zeros(T)], axis=1)
    np.savez(tmp / f"{scene}_poses.npz",
             plane_normal_scene=np.array([0.0, 0.0, 1.0]),
             plane_offset_scene=0.0, scale_m_per_unit=1.0,
             positions=positions, headings=np.tile([1.0, 0.0, 0.0], (T, 1)),
             camera_height_m=0.25,
             K=np.array([[300.0, 0, 280], [0, 300.0, 168], [0, 0, 1]]))

    import cv2
    cache = tmp / "cache" / scene
    sweeps = []
    VAL = lambda si, i: 30 + 120 * si + 15 * i
    for si, (lat, yaw) in enumerate([(0.0, 0.0), (0.5, 90.0)]):
        name = f"sweep_lat{lat:+.2f}_yaw{yaw:03.0f}"
        d = cache / name
        d.mkdir(parents=True)
        vw = cv2.VideoWriter(str(d / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (8, 8))
        for i in range(5):
            vw.write(np.full((8, 8, 3), VAL(si, i), np.uint8))
        vw.release()
        alpha = np.ones((5, 8, 8), bool)
        alpha[:, :, :4] = False
        np.savez(d / "semantic_labels.npz", labels=np.full((5, 8, 8), 5, np.int8))
        np.savez(d / "alpha.npz", alpha=alpha)
        sweeps.append({"file": name + ".json", "lateral_m": lat, "heading_deg": yaw,
                       "nav_xyyaw": [[float(i), lat, yaw] for i in range(5)]})
    (cache / "manifest.json").write_text(
        json.dumps({"scene": scene, "num_frames": 5, "sweeps": sweeps}))

    from src.env.real_calibrated import CalibratedBackendConfig
    from src.env.cached_backend import CachedDiffusedBackend
    cfg = CalibratedBackendConfig(scene_video_paths={scene: "u.mp4"},
                                  scene_poses_paths={scene: str(tmp / f"{scene}_poses.npz")},
                                  scene_labels_paths={scene: "u.npz"})
    world = CachedDiffusedBackend(cfg, cache_root=str(tmp / "cache"))
    world.load_scene(scene)

    def robot_pose(x, y, yaw_deg):
        r = np.deg2rad(yaw_deg)
        c, s = np.cos(r), np.sin(r)
        p = np.eye(4, dtype=np.float32)
        p[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p[:3, 3] = [x, y, 0]
        return p

    rgb, _, _ = world.render(robot_pose(2.0, 0.0, 0.0))
    assert abs(int(rgb[0, 0, 0]) - VAL(0, 2)) <= 6
    assert world._last_lookup[0] < 1e-6 and world._last_lookup[1] < 1e-3

    rgb, _, _ = world.render(robot_pose(3.0, 0.25, 90.0))
    assert abs(int(rgb[0, 0, 0]) - VAL(1, 3)) <= 6
    assert 0.2 < world._last_lookup[0] < 0.3
    lab = world._last_semantic_image
    assert (lab[:, 4:] == 5).all() and (lab[:, :4] == 0).all()

    world.render(robot_pose(2.0, 0.0, 40.0))
    assert 39 < world._last_lookup[1] < 51
    print("PASS: exact hit / heading-weighted choice / alpha->void masking / telemetry")


if __name__ == "__main__":
    main()
