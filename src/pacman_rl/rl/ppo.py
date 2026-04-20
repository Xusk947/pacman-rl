from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from pacman_rl.config import PPOConfig
from pacman_rl.models import CNNActorCritic
from pacman_rl.rl.rollout_buffer import RolloutBatch


@dataclass(frozen=True)
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


def _iter_minibatches(total: int, minibatch_size: int, device: torch.device) -> list[torch.Tensor]:
    idx = torch.randperm(total, device=device)
    return [idx[i : i + minibatch_size] for i in range(0, total, minibatch_size)]


def ppo_update(
    model: CNNActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: PPOConfig,
) -> PPOStats:
    obs = batch.obs
    actions = batch.actions
    old_logprobs = batch.logprobs
    advantages = batch.advantages
    returns = batch.returns
    old_values = batch.values

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total = obs.shape[0]
    device = obs.device

    last_loss = 0.0
    last_pi = 0.0
    last_v = 0.0
    last_ent = 0.0
    last_kl = 0.0

    for _ in range(cfg.epochs):
        for mb_idx in _iter_minibatches(total, cfg.minibatch_size, device):
            out = model(obs[mb_idx])
            dist = Categorical(logits=out.logits)
            logprob = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()

            ratio = (logprob - old_logprobs[mb_idx]).exp()
            pg1 = -advantages[mb_idx] * ratio
            pg2 = -advantages[mb_idx] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(pg1, pg2).mean()

            v_pred = out.value
            v_old = old_values[mb_idx]
            v_unclipped = (v_pred - returns[mb_idx]).pow(2)
            v_clipped = (v_old + torch.clamp(v_pred - v_old, -cfg.clip_coef, cfg.clip_coef) - returns[mb_idx]).pow(2)
            value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            approx_kl = 0.5 * ((logprob - old_logprobs[mb_idx]).pow(2)).mean()

            last_loss = float(loss.detach().cpu())
            last_pi = float(policy_loss.detach().cpu())
            last_v = float(value_loss.detach().cpu())
            last_ent = float(entropy.detach().cpu())
            last_kl = float(approx_kl.detach().cpu())

    return PPOStats(loss=last_loss, policy_loss=last_pi, value_loss=last_v, entropy=last_ent, approx_kl=last_kl)
