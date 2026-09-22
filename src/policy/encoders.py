"""Policy visual encoders (advisor ask 2026-08-30: "is the CNN from any
well-developed model? DINOv2 is a good method if CNN doesn't work well").

The SB3 default is NatureCNN — 3 conv layers from the 2015 Atari DQN,
trained FROM SCRATCH by the reward signal alone. Hypothesis: that starving
encoder is a root of odometry-reliance (reward is a terrible perception
teacher when a clean goal vector sits in the same observation).

This module provides a frozen-pretrained alternative for the Dict obs
{"rgb": HxWx3 uint8, "goal": 3}:

  backbone="dinov2":   ViT-S/14 self-supervised features (layout/geometry;
                       our 336x224 divides by the 14-px patches exactly).
  backbone="dinov2b":  ViT-B/14, 768-d tokens, ~3x the cost (2026-09-07:
                       the capacity row of the ablation).
  backbone="resnet18": ImageNet-supervised baseline row for the ablation.
  grid=(gh, gw):       region grid the patch tokens are pooled into; 3x4 by
                       default, 6x8 keeps narrow gaps from vanishing in a cell.

Both run FROZEN (requires_grad=False, eval mode) with the same small linear
head, so the comparison isolates feature quality; PPO only trains the head +
policy MLP. Select via train_ppo_real --encoder / launcher ENCODER knob.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FrozenBackboneExtractor(BaseFeaturesExtractor):
    """Frozen pretrained backbone on obs["rgb"] + passthrough obs["goal"]."""

    GRID = (3, 4)   # D1 (Joana 2026-08-30): pool DINOv2 patches into a 3x4
                    # region grid — location survives ("tree on my LEFT"),
                    # which mean-pooling destroys and avoidance needs.
    DINO = {"dinov2": ("dinov2_vits14", 384), "dinov2b": ("dinov2_vitb14", 768)}

    def __init__(self, observation_space: gym.spaces.Dict,
                 backbone: str = "dinov2", head_dim: int = 256,
                 grid: "tuple[int, int] | None" = None):
        goal_dim = int(observation_space["goal"].shape[0])
        super().__init__(observation_space, features_dim=head_dim + goal_dim)
        self.backbone_name = backbone
        self.grid = tuple(int(v) for v in grid) if grid else self.GRID
        self.dino = self.resnet = None
        self.dino_dim = 0
        feat_dim = 0
        gh, gw = self.grid
        if backbone in self.DINO or backbone == "both":
            name, self.dino_dim = self.DINO.get(backbone, self.DINO["dinov2"])
            self.dino = torch.hub.load("facebookresearch/dinov2", name)
            feat_dim += self.dino_dim * (1 + gh * gw)   # CLS + region grid
            print(f"[FrozenBackboneExtractor] {name} frozen, {self.dino_dim}-d tokens, "
                  f"{gh}x{gw} region grid -> {feat_dim} features -> {head_dim}", flush=True)
        if backbone in ("resnet18", "both"):
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.resnet = nn.Sequential(*list(net.children())[:-1])
            feat_dim += 512
        if feat_dim == 0:
            raise ValueError(f"unknown backbone {backbone}")
        for m in (self.dino, self.resnet):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
                m.eval()
        self.head = nn.Sequential(nn.Linear(feat_dim, head_dim), nn.ReLU())

    def train(self, mode: bool = True):
        # keep the frozen backbones in eval mode regardless of policy mode
        super().train(mode)
        for m in (self.dino, self.resnet):
            if m is not None:
                m.eval()
        return self

    def forward(self, obs: dict) -> torch.Tensor:
        # 2026-09-22 (Joana: "are we sure the blind arms were actually blind?"): SB3's
        # preprocess_obs ALREADY divides uint8 image spaces by 255 when the policy is built
        # with normalize_images=True (its default), so every DINO policy trained before this
        # date received the image divided by 255 TWICE -- a near-black frame whose tokens
        # barely depend on the scene. Actions changed by <0.003 between a real frame and
        # zeros. Build the policy with normalize_images=False (train: --image_norm_fix) so this
        # single division is the only one.
        rgb = obs["rgb"].float() / 255.0      # [B,3,H,W] via VecTransposeImage
        rgb = (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)
        parts = []
        gh, gw = self.grid
        with torch.no_grad():
            if self.dino is not None:
                out = self.dino.forward_features(rgb)
                tok = out["x_norm_patchtokens"]           # [B, ph*pw, D]
                b = tok.shape[0]
                ph, pw = rgb.shape[-2] // 14, rgb.shape[-1] // 14
                grid = tok.transpose(1, 2).reshape(b, self.dino_dim, ph, pw)
                grid = torch.nn.functional.adaptive_avg_pool2d(grid, (gh, gw))
                parts += [out["x_norm_clstoken"], grid.flatten(1)]
            if self.resnet is not None:
                parts.append(self.resnet(rgb).flatten(1))
        feat = torch.cat(parts, dim=-1)
        return torch.cat([self.head(feat), obs["goal"].float()], dim=-1)


def image_sensitivity(model, goal=(5.0, 0.0, 0.0), seed=0) -> dict:
    """How much the deterministic action changes between a random image and a black one for
    one goal vector. A policy that uses its camera moves by 0.01-1; a blind one by <0.003
    (measured 2026-09-22 on every checkpoint trained with the double /255)."""
    import numpy as np
    shp = tuple(model.observation_space["rgb"].shape)
    rng = np.random.default_rng(seed)
    g = np.asarray(goal, dtype=np.float32)[: int(model.observation_space["goal"].shape[0])]
    img = rng.integers(0, 256, shp).astype(np.uint8)
    a_img, _ = model.predict({"rgb": img, "goal": g}, deterministic=True)
    a_zero, _ = model.predict({"rgb": np.zeros(shp, np.uint8), "goal": g}, deterministic=True)
    return {"delta": float(np.abs(np.asarray(a_img) - np.asarray(a_zero)).max()),
            "normalize_images": bool(getattr(model.policy, "normalize_images", True))}
