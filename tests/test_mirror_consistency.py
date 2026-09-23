"""Is the mirror self-consistent? (2026-09-22, Joana: "can the mirrored scenes have an effect on
this if they're wrong?")

Half of every episode since 09-21 is presented mirrored: the image is flipped left-right, the goal's
lateral offset and bearing are negated, and the policy's yaw command is negated back before it
touches the world. On 09-21 we verified the image half by pixels. Nobody checked that the three
negations agree with each other, and if they do not, half of training teaches the opposite turn for
the same view, the yaw signal cancels, and a policy that cannot tell left from right has no reason
to drive. The arms trained WITHOUT the mirror have a mean throttle of +0.99; the ones WITH it have
-0.26 and +0.20.

Exercises the real SceneEnv methods against a stub, no GPU:
  1. obs: a goal to the LEFT in the world must read as a goal to the RIGHT when mirrored, and the
     forward component must NOT flip.
  2. obs: the image the policy receives is the left-right flip of the rendered one.
  3. step: a yaw command issued in the mirrored frame reaches the world negated, and the throttle
     does not, so "turn toward the goal" means the same thing in both frames.
  4. the round trip: mirrored(obs) + action a  drives the world the same way as
     unmirrored(obs) + mirror(a). This is the property the policy actually needs.

    python tests/test_mirror_consistency.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.scene_env import SceneEnv                      # noqa: E402

H, W = 24, 40


def make_env(mirrored: bool, goal_robot):
    env = object.__new__(SceneEnv)
    env.cfg = SimpleNamespace(obs_frame_stack=1, goal_noise_std=0.0, obs_out_hw=None,
                              mirror_prob=0.5, action_chunk=1, stop_action=False, forward_only=True)
    env._mirrored = mirrored
    env._scene_id, env._steps = "s", 0
    env._rgb_hist, env._rgb_hist_key = None, None
    env._goal_in_robot_frame = lambda: np.asarray(goal_robot, np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = 200                      # a bright block on the LEFT of the rendered view
    env._last_rgb = rgb
    env._robot_pose_world = np.eye(4)
    return env


def test_goal_negation():
    g = [3.0, 1.5, 0.4]                          # 1.5 m to the LEFT, bearing +0.4 rad
    plain = make_env(False, g)._obs()["goal"]
    mirror = make_env(True, g)._obs()["goal"]
    assert np.isclose(plain[0], mirror[0]), f"forward component flipped: {plain[0]} vs {mirror[0]}"
    assert np.isclose(mirror[1], -plain[1]), f"lateral not negated: {plain[1]} -> {mirror[1]}"
    assert np.isclose(mirror[2], -plain[2]), f"bearing not negated: {plain[2]} -> {mirror[2]}"
    print(f"   goal  (fwd {plain[0]:+.1f}, lat {plain[1]:+.1f}, yaw {plain[2]:+.2f})"
          f"  ->  mirrored (fwd {mirror[0]:+.1f}, lat {mirror[1]:+.1f}, yaw {mirror[2]:+.2f})   OK")


def test_image_flip():
    plain = make_env(False, [3.0, 0.0, 0.0])._obs()["rgb"]
    mirror = make_env(True, [3.0, 0.0, 0.0])._obs()["rgb"]
    assert np.array_equal(mirror, plain[:, ::-1]), "mirrored image is not the flip of the rendered one"
    left_plain = plain[:, : W // 2].mean(); left_mirror = mirror[:, : W // 2].mean()
    assert left_plain > 100 and left_mirror < 100, "the bright block did not move to the other side"
    print(f"   image bright side: plain LEFT ({left_plain:.0f}) -> mirrored LEFT ({left_mirror:.0f})   OK")


def _captured_action(mirrored, action):
    env = make_env(mirrored, [3.0, 1.5, 0.4])
    seen = {}
    env._step_single = lambda a: seen.setdefault("a", np.asarray(a, np.float32).copy())
    SceneEnv.step(env, np.asarray(action, np.float32))
    return seen["a"]


def test_action_negation():
    a = [0.8, 0.5]                                # full throttle, turn one way
    plain = _captured_action(False, a)
    mirror = _captured_action(True, a)
    assert np.isclose(plain[0], mirror[0]), f"throttle was negated: {plain[0]} vs {mirror[0]}"
    assert np.isclose(mirror[1], -plain[1]), f"yaw not negated in the mirrored frame: {plain[1]} -> {mirror[1]}"
    print(f"   action (v {plain[0]:+.1f}, w {plain[1]:+.1f}) issued mirrored reaches the world as "
          f"(v {mirror[0]:+.1f}, w {mirror[1]:+.1f})   OK")


def test_round_trip():
    """The property the policy needs: acting on a mirrored observation must drive the world the
    same way as acting on the plain observation with the mirrored action."""
    g = [3.0, 1.5, 0.4]
    obs_m = make_env(True, g)._obs()
    obs_p = make_env(False, g)._obs()
    assert np.array_equal(obs_m["rgb"], obs_p["rgb"][:, ::-1])
    assert np.allclose(obs_m["goal"], obs_p["goal"] * np.array([1, -1, -1], np.float32))
    a_m = np.array([0.8, 0.5], np.float32)                     # what a policy outputs on obs_m
    a_equiv = np.array([0.8, -0.5], np.float32)                # the same intent on obs_p
    world_from_mirror = _captured_action(True, a_m)
    world_from_plain = _captured_action(False, a_equiv)
    assert np.allclose(world_from_mirror, world_from_plain), (
        f"a mirrored episode does NOT drive the world like its unmirrored twin: "
        f"{world_from_mirror} vs {world_from_plain}")
    print(f"   round trip: mirrored obs + {a_m.tolist()} and plain obs + {a_equiv.tolist()} "
          f"both reach the world as {world_from_mirror.tolist()}   OK")


if __name__ == "__main__":
    print("== mirror consistency")
    test_goal_negation()
    test_image_flip()
    test_action_negation()
    test_round_trip()
    print("OK: the image flip, the goal negation and the action negation agree")
