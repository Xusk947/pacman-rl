from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from pacman_rl.models.shared_cnn import SharedCNNActorCritic


@dataclass(frozen=True)
class A2CStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float


def a2c_update(
    model: SharedCNNActorCritic,
    opt: torch.optim.Optimizer,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> A2CStats:
    out = model(obs)
    dist = Categorical(logits=out.logits)
    logp = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    policy_loss = -(logp * adv.detach()).mean()
    value_loss = 0.5 * (returns - out.value).pow(2).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    opt.step()

    return A2CStats(
        loss=float(loss.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        entropy=float(entropy.detach().item()),
    )

