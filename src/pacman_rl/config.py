from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    scared_steps: int = 40
    max_steps: int = 512

    reward_pellet: float = 1.0
    reward_power: float = 5.0
    reward_eat_ghost: float = 10.0
    reward_death: float = -20.0
    reward_step: float = -0.01
    reward_wall_bump: float = -0.02

    ghost_reward_catch: float = 20.0
    ghost_reward_eaten: float = -10.0
    ghost_reward_step: float = -0.01
    ghost_reward_wall_bump: float = -0.02


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 4096

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2

    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    learning_rate: float = 3e-4


@dataclass(frozen=True)
class SelfPlayConfig:
    snapshot_every_updates: int = 25
    opponent_pool_size: int = 10
    opponent_sample_prob: float = 0.5
