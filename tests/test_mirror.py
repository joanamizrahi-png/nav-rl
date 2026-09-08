"""Mirror-augmentation self-test on the mock world (no GPU, no cluster).
Checks: the image is flipped, the goal's lateral offset and bearing are
negated, and a mirrored yaw command turns the robot the opposite way in the
real world, i.e. the mirrored MDP is the reflection of the real one."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.env.mock_backend import MockWorldBackend, MockSemanticBackend
from src.env.scene_env import SceneEnv, SceneEnvConfig

def build(mirror):
    cfg = SceneEnvConfig(mirror_prob=mirror, max_steps=20, step_size_m=0.25, yaw_step_rad=0.3)
    w = MockWorldBackend(H=84, W=84, seed=0); sem = MockSemanticBackend(w)
    return SceneEnv(world_backend=w, semantic_backend=sem, scene_ids=["mock"], cfg=cfg)

a, b = build(0.0), build(1.0)
oa, _ = a.reset(seed=3); ob, _ = b.reset(seed=3)
print("mirrored flag:", getattr(a, "_mirrored"), getattr(b, "_mirrored"))
print("goal plain   :", np.round(oa["goal"], 3))
print("goal mirrored:", np.round(ob["goal"], 3), " (lateral and bearing must be negated)")
ok_goal = np.allclose(oa["goal"][0], ob["goal"][0]) and np.allclose(oa["goal"][1], -ob["goal"][1]) and np.allclose(oa["goal"][2], -ob["goal"][2])
ok_img = np.array_equal(oa["rgb"][:, ::-1], ob["rgb"])
print("goal reflected:", ok_goal, " image flipped:", ok_img)
# same yaw command, opposite real-world turn
ya0 = float(np.arctan2(a._robot_pose_world[1, 0], a._robot_pose_world[0, 0]))
yb0 = float(np.arctan2(b._robot_pose_world[1, 0], b._robot_pose_world[0, 0]))
act = np.array([1.0, 0.6], dtype=np.float32)
a.step(act); b.step(act)
da = float(np.arctan2(a._robot_pose_world[1, 0], a._robot_pose_world[0, 0])) - ya0
db = float(np.arctan2(b._robot_pose_world[1, 0], b._robot_pose_world[0, 0])) - yb0
print(f"yaw change with the SAME command: plain {np.degrees(da):+.1f} deg, mirrored {np.degrees(db):+.1f} deg")
ok = ok_goal and ok_img and abs(da + db) < 1e-6 and abs(da) > 1e-3
print("RESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
