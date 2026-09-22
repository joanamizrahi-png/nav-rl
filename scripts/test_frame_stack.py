"""Test of the MULTIPLE IMAGES arm before it burns a GPU (2026-09-22).
Runs on the login node, CPU only, needs the neoverse env + PYTHONNOUSERSITE=1.

1. env side: the deque in SceneEnv._obs, exercised through a stub whose _obs is the real method.
   - at episode start the stack is the first view repeated K times
   - each step shifts by one, oldest first, newest last
   - calling _obs twice for the same step does NOT shift the history
   - a new episode clears it
2. policy side: FrozenBackboneExtractor on a (3K, H, W) space
   - output width is head_dim + goal_dim
   - changing only the OLDEST frame changes the features (the history is really read)
   - with K=1 nothing changes versus the single-frame path

    python scripts/test_frame_stack.py [K]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
H, W = 56, 84          # small: this test is about plumbing, not pixels


def test_env_side():
    from types import SimpleNamespace
    from src.env.scene_env import SceneEnv
    env = object.__new__(SceneEnv)
    env.cfg = SimpleNamespace(obs_frame_stack=K, goal_noise_std=0.0, obs_out_hw=None, mirror_prob=0.0)
    env._scene_id, env._steps = "s", 0
    env._rgb_hist, env._rgb_hist_key = None, None
    env._mirrored = False
    env._goal_in_robot_frame = lambda: np.array([5.0, 0.0, 0.0], np.float32)
    def frame(v):
        return np.full((H, W, 3), v, np.uint8)
    env._last_rgb = frame(10)
    o = SceneEnv._obs(env)
    assert o["rgb"].shape == (H, W, 3 * K), o["rgb"].shape
    got = [int(o["rgb"][0, 0, 3 * i]) for i in range(K)]
    assert got == [10] * K, f"episode start should repeat the first view, got {got}"
    o2 = SceneEnv._obs(env)                                   # same step, must not shift
    assert [int(o2["rgb"][0, 0, 3 * i]) for i in range(K)] == [10] * K, "history shifted on a repeated _obs call"
    for step, v in enumerate([20, 30, 40, 50], start=1):
        env._steps = step; env._last_rgb = frame(v)
        o = SceneEnv._obs(env)
    seen = [int(o["rgb"][0, 0, 3 * i]) for i in range(K)]
    expect = ([10] * K + [20, 30, 40, 50])[-K:]
    assert seen == expect, f"stack after 4 steps: got {seen}, expected {expect} (oldest first)"
    env._rgb_hist, env._rgb_hist_key, env._steps = None, None, 0      # what reset() does
    env._last_rgb = frame(99)
    o = SceneEnv._obs(env)
    assert [int(o["rgb"][0, 0, 3 * i]) for i in range(K)] == [99] * K, "history survived a reset"
    print(f"   env side OK: shape {o['rgb'].shape}, oldest-first order, no shift on repeated calls, cleared on reset")


def test_policy_side():
    import torch, gymnasium as gym
    from src.policy.encoders import FrozenBackboneExtractor
    space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (3 * K, H, W), np.uint8),
                             "goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)})
    ex = FrozenBackboneExtractor(space, backbone="dinov2", head_dim=256)
    assert ex.n_stack == K, (ex.n_stack, K)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (1, 3 * K, H, W)).astype(np.float32)
    g = torch.zeros(1, 3)
    def feats(arr):
        with torch.no_grad():
            return ex({"rgb": torch.as_tensor(arr), "goal": g})[0]
    f0 = feats(base)
    assert f0.shape[0] == 256 + 3, f0.shape
    if K > 1:
        old = base.copy(); old[:, 0:3] = rng.integers(0, 255, (1, 3, H, W))      # OLDEST frame only
        new = base.copy(); new[:, -3:] = rng.integers(0, 255, (1, 3, H, W))      # NEWEST frame only
        d_old = float((feats(old) - f0).abs().max()); d_new = float((feats(new) - f0).abs().max())
        assert d_old > 1e-4, "changing the oldest frame did nothing: the history is not read"
        assert d_new > 1e-4, "changing the newest frame did nothing"
        print(f"   policy side OK: features {tuple(f0.shape)}, oldest frame moves them {d_old:.4f}, newest {d_new:.4f}")
    else:
        print(f"   policy side OK: features {tuple(f0.shape)} with K=1 (single-frame path unchanged)")


def test_cnn_side():
    """The nature CNN arm must accept 3K channels too: SB3 picks NatureCNN for any uint8 image
    subspace regardless of channel count, but that is worth proving rather than assuming."""
    import torch, gymnasium as gym
    from stable_baselines3.common.torch_layers import CombinedExtractor
    space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (3 * K, H, W), np.uint8),
                             "goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)})
    ex = CombinedExtractor(space)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (1, 3 * K, H, W)).astype(np.float32)
    g = torch.zeros(1, 3)
    def feats(arr):
        with torch.no_grad():
            return ex({"rgb": torch.as_tensor(arr), "goal": g})[0]
    f0 = feats(base)
    if K > 1:
        old = base.copy(); old[:, 0:3] = rng.integers(0, 255, (1, 3, H, W))
        d_old = float((feats(old) - f0).abs().max())
        assert d_old > 1e-4, "nature CNN ignores the oldest frame"
        print(f"   nature CNN OK: features {tuple(f0.shape)}, oldest frame moves them {d_old:.4f}")
    else:
        print(f"   nature CNN OK: features {tuple(f0.shape)} with K=1")


if __name__ == "__main__":
    print(f"== MULTIPLE IMAGES test, K = {K}")
    test_env_side()
    test_policy_side()
    test_cnn_side()
    print("OK")
