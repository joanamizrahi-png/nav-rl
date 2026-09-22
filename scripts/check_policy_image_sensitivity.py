"""Does the trained policy USE the image? (2026-09-22, Joana: "are we sure the blind arms were
actually blind?" -- blind and sighted evals gave trajectories within 4 mm of each other.)
Loads a PPO checkpoint, feeds the SAME goal vector with different images (two real eval frames,
zeros, noise, a flipped frame, white) and prints the actions, the feature-vector differences the
extractor produces, and the first policy layer's weight mass on image features vs goal features.
    python scripts/check_policy_image_sensitivity.py <ckpt.zip> <eval_episode.mp4>
Runs on the login node (CPU) in the neoverse env."""
import sys, numpy as np, cv2, torch
# checkpoints were pickled under numpy 2 ("numpy._core"); on a numpy-1 interpreter map those
# module names onto numpy.core so the unpickler finds them (login node, 2026-09-22)
if not hasattr(np, "_core"):
    import importlib, types
    for sub in ("", ".numeric", ".multiarray", ".umath", "._multiarray_umath", ".fromnumeric", "._methods"):
        try:
            sys.modules["numpy._core" + sub] = importlib.import_module("numpy.core" + sub)
        except Exception:
            pass
print("numpy", np.__version__, "torch", torch.__version__, "python", sys.executable)
sys.path.insert(0, "/scratch/m000204-pm06b/joana/nav-rl")
from stable_baselines3 import PPO

ckpt, video = sys.argv[1], sys.argv[2]
model = PPO.load(ckpt, device="cpu")
_shp = model.observation_space["rgb"].shape
CHW = _shp[0] == 3                                   # SB3 stores the transposed (3,H,W) space
H, W = (_shp[1], _shp[2]) if CHW else (_shp[0], _shp[1])
def to_obs(img_hwc):
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1)) if CHW else img_hwc
cap = cv2.VideoCapture(video); frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % 6 == 0: frames.append(f)
    i += 1
def obs_from(f):
    left = f[:, :f.shape[1] // 2]                     # the policy-view panel of the eval video
    return cv2.cvtColor(cv2.resize(left, (W, H)), cv2.COLOR_BGR2RGB).astype(np.uint8)
rng = np.random.default_rng(0)
images = {"frame_a": obs_from(frames[0]), "frame_b": obs_from(frames[min(2, len(frames) - 1)]),
          "zeros": np.zeros((H, W, 3), np.uint8), "noise": rng.integers(0, 256, (H, W, 3)).astype(np.uint8),
          "flipped_a": obs_from(frames[0])[:, ::-1].copy(), "white": np.full((H, W, 3), 255, np.uint8)}
print(f"checkpoint {ckpt}\nobs rgb space {_shp} -> H={H} W={W}, {len(frames)} frames sampled from {video}")
def feats(img, g):
    obs_t, _ = model.policy.obs_to_tensor({"rgb": to_obs(img)[None], "goal": g[None]})
    with torch.no_grad():
        return model.policy.extract_features(obs_t, model.policy.pi_features_extractor)[0].cpu().numpy()
for g in (np.array([5.0, 0.0, 0.0], np.float32), np.array([5.0, 2.0, 0.4], np.float32), np.array([3.0, -2.0, -0.6], np.float32)):
    print(f"\n== goal vector {g.tolist()}")
    acts, fts = {}, {}
    for name, img in images.items():
        a, _ = model.predict({"rgb": to_obs(img), "goal": g}, deterministic=True); acts[name] = np.asarray(a, np.float32); fts[name] = feats(img, g)
        print(f"   {name:10s} action v={acts[name][0]:+.4f} w={acts[name][1]:+.4f}")
    base = acts["frame_a"]; fb = fts["frame_a"]
    print("   max |action - action(frame_a)|:", {k: round(float(np.abs(v - base).max()), 5) for k, v in acts.items()})
    print("   |features - features(frame_a)| / |features(frame_a)|:", {k: round(float(np.linalg.norm(v - fb) / (np.linalg.norm(fb) + 1e-9)), 4) for k, v in fts.items()})
# ---- where does the image signal die? DINO tokens -> head Linear+ReLU -> 256 features ----
ext = model.policy.pi_features_extractor
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def dino_feat(img_hwc):
    x = torch.as_tensor(to_obs(img_hwc)[None]).float() / 255.0
    x = (x - IM_MEAN) / IM_STD
    with torch.no_grad():
        out = ext.dino.forward_features(x); tok = out["x_norm_patchtokens"]
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        grid = tok.transpose(1, 2).reshape(1, ext.dino_dim, ph, pw)
        grid = torch.nn.functional.adaptive_avg_pool2d(grid, ext.grid)
        feat = torch.cat([out["x_norm_clstoken"], grid.flatten(1)], dim=-1)
        pre = ext.head[0](feat); post = ext.head(feat)
    return feat[0], pre[0], post[0]
fa, pa, ha = dino_feat(images["frame_a"]); fz, pz, hz = dino_feat(images["zeros"]); fn, pn, hn = dino_feat(images["noise"])
print("\n== where the image signal dies")
print(f"   DINO features: |frame_a - zeros| / |frame_a| = {float((fa - fz).norm() / fa.norm()):.3f}, |frame_a - noise| / |frame_a| = {float((fa - fn).norm() / fa.norm()):.3f}   (should be large)")
print(f"   head pre-activation (Linear out, 256): frame_a mean {float(pa.mean()):+.4f} max {float(pa.max()):+.4f}; units > 0: {int((pa > 0).sum())}/256 (frame_a), {int((pz > 0).sum())}/256 (zeros), {int((pn > 0).sum())}/256 (noise)")
print(f"   head output after ReLU: |frame_a| = {float(ha.norm()):.4f}, |frame_a - zeros| = {float((ha - hz).norm()):.4f}, goal part norm for [5,0,0] = 5.0")
W = ext.head[0].weight.detach(); b = ext.head[0].bias.detach()
print(f"   head Linear: mean |W| {float(W.abs().mean()):.6f}, bias mean {float(b.mean()):+.4f} min {float(b.min()):+.4f} max {float(b.max()):+.4f}")
# optional: more checkpoints -> alive units over training
for extra in sys.argv[3:]:
    m2 = PPO.load(extra, device="cpu"); e2 = m2.policy.pi_features_extractor
    x = torch.as_tensor(to_obs(images["frame_a"])[None]).float() / 255.0; x = (x - IM_MEAN) / IM_STD
    with torch.no_grad():
        out = e2.dino.forward_features(x); tok = out["x_norm_patchtokens"]; ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        grid = torch.nn.functional.adaptive_avg_pool2d(tok.transpose(1, 2).reshape(1, e2.dino_dim, ph, pw), e2.grid)
        pre = e2.head[0](torch.cat([out["x_norm_clstoken"], grid.flatten(1)], dim=-1))[0]
    print(f"   {extra.split('/')[-3][-12:]}/{extra.split('/')[-1]}: alive units {int((pre > 0).sum())}/256, pre-activation max {float(pre.max()):+.4f}, bias mean {float(e2.head[0].bias.mean()):+.4f}")

# weight mass of the first policy layer on the image part vs the goal part of the feature vector
n_goal = int(model.observation_space["goal"].shape[0]); n_feat = fts["frame_a"].shape[0]
first = None
for m in model.policy.mlp_extractor.policy_net:
    if isinstance(m, torch.nn.Linear): first = m; break
if first is not None and first.weight.shape[1] == n_feat:
    w = first.weight.detach().abs()
    print(f"\nfirst policy layer: mean |w| on image features {w[:, :n_feat - n_goal].mean():.5f} vs on goal features {w[:, n_feat - n_goal:].mean():.5f} (feature dim {n_feat}, goal dim {n_goal})")
else:
    print("\nfirst policy layer not found or feature layout differs; skip weight check", None if first is None else tuple(first.weight.shape), n_feat)
