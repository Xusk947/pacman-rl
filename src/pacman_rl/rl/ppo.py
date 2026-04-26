from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from pacman_rl.config import PPOConfig
from pacman_rl.models.shared_cnn import SharedCNNActorCritic


@dataclass(frozen=True)
class GAEOutput:
    advantages: torch.Tensor
    returns: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0])

    for i in reversed(range(t)):
        done = dones[i].to(torch.float32)
        not_done = 1.0 - done
        next_val = last_values if i == t - 1 else values[i + 1]
        delta = rewards[i] + gamma * next_val * not_done - values[i]
        last_adv = delta + gamma * gae_lambda * not_done * last_adv
        adv[i] = last_adv

    ret = adv + values
    return GAEOutput(advantages=adv, returns=ret)


@dataclass(frozen=True)
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


def ppo_update(
    model: SharedCNNActorCritic,
    opt: torch.optim.Optimizer,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> PPOStats:
    adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = obs.shape[0]
    batch_size = int(cfg.minibatch_size)
    if batch_size <= 0:
        batch_size = n

    total_loss = 0.0
    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    total_kl = 0.0
    steps = 0

    for _ in range(int(cfg.epochs)):
        perm = torch.randperm(n, device=obs.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = model(obs[idx])
            dist = Categorical(logits=out.logits)
            logp = dist.log_prob(actions[idx])
            ratio = (logp - old_logp[idx]).exp()

            pg1 = -adv[idx] * ratio
            pg2 = -adv[idx] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(pg1, pg2).mean()

            value_loss = 0.5 * (returns[idx] - out.value).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = (old_logp[idx] - logp).mean()

            total_loss += float(loss.detach().item())
            total_pi += float(policy_loss.detach().item())
            total_v += float(value_loss.detach().item())
            total_ent += float(entropy.detach().item())
            total_kl += float(approx_kl.detach().item())
            steps += 1

    if steps <= 0:
        return PPOStats(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)

    return PPOStats(
        loss=total_loss / steps,
        policy_loss=total_pi / steps,
        value_loss=total_v / steps,
        entropy=total_ent / steps,
        approx_kl=total_kl / steps,
    )

