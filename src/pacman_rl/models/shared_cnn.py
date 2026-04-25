from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    value: torch.Tensor


class SharedCNNActorCritic(nn.Module):
    def __init__(self, *, in_channels: int, actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, actions)
        self.v = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.trunk(x)
        h = self.head(z)
        return ModelOutput(logits=self.pi(h), value=self.v(h).squeeze(-1))

