from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)

    gae = torch.zeros_like(last_value)
    for i in reversed(range(t)):
        next_nonterminal = 1.0 - dones[i].to(torch.float32)
        next_value = last_value if i == t - 1 else values[i + 1]
        delta = rewards[i] + gamma * next_value * next_nonterminal - values[i]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        adv[i] = gae

    ret = adv + values
    return adv, ret
