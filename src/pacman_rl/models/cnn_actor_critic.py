from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActorCriticOutput:
    logits: torch.Tensor
    value: torch.Tensor


class CNNActorCritic(nn.Module):
    def __init__(self, in_channels: int, actions: int) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(nn.Linear(64 * 4 * 4, 256), nn.ReLU())

        self.pi = nn.Linear(256, actions)
        self.v = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> ActorCriticOutput:
        x = self.backbone(obs)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return ActorCriticOutput(logits=self.pi(x), value=self.v(x).squeeze(-1))
