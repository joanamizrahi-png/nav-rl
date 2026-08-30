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
  backbone="resnet18": ImageNet-supervised baseline row for the ablation.

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

    def __init__(self, observation_space: gym.spaces.Dict,
                 backbone: str = "dinov2", head_dim: int = 256):
        goal_dim = int(observation_space["goal"].shape[0])
        super().__init__(observation_space, features_dim=head_dim + goal_dim)
        self.backbone_name = backbone
        self.dino = self.resnet = None
        feat_dim = 0
        gh, gw = self.GRID
        if backbone in ("dinov2", "both"):
            self.dino = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14")
            feat_dim += 384 + 384 * gh * gw   # CLS + 3x4 region grid
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
        rgb = obs["rgb"].float() / 255.0      # [B,3,H,W] via VecTransposeImage
        rgb = (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)
        parts = []
        gh, gw = self.GRID
        with torch.no_grad():
            if self.dino is not None:
                out = self.dino.forward_features(rgb)
                tok = out["x_norm_patchtokens"]           # [B, ph*pw, 384]
                b = tok.shape[0]
                ph, pw = rgb.shape[-2] // 14, rgb.shape[-1] // 14
                grid = tok.transpose(1, 2).reshape(b, 384, ph, pw)
                grid = torch.nn.functional.adaptive_avg_pool2d(grid, (gh, gw))
                parts += [out["x_norm_clstoken"], grid.flatten(1)]
            if self.resnet is not None:
                parts.append(self.resnet(rgb).flatten(1))
        feat = torch.cat(parts, dim=-1)
        return torch.cat([self.head(feat), obs["goal"].float()], dim=-1)
