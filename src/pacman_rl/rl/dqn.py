from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pacman_rl.models.q_cnn import QNetwork
from pacman_rl.rl.replay_buffer import ReplayBatch


@dataclass(frozen=True)
class DQNStats:
    loss: float


def dqn_update(
    q: QNetwork,
    target_q: QNetwork,
    opt: torch.optim.Optimizer,
    *,
    batch: ReplayBatch,
    gamma: float,
    max_grad_norm: float = 10.0,
) -> DQNStats:
    q_values = q(batch.obs)
    qa = q_values.gather(1, batch.actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_q = target_q(batch.next_obs).max(dim=1).values
        target = batch.rewards + (1.0 - batch.dones.to(torch.float32)) * gamma * next_q

    loss = F.smooth_l1_loss(qa, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
    opt.step()
    return DQNStats(loss=float(loss.detach().item()))

