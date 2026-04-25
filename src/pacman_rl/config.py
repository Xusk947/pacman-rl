from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 512
    frightened_steps: int = 40
    reward_pellet: float = 1.0
    reward_power: float = 5.0
    reward_ghost_eat: float = 10.0
    reward_win: float = 50.0
    reward_death: float = -50.0
    reward_step_pacman: float = -0.01
    reward_step_ghost: float = -0.01
    reward_ghost_catch: float = 50.0


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 4096


@dataclass(frozen=True)
class LogConfig:
    sqlite_path: str = "runs/metrics.sqlite"
    sqlite_flush_every_steps: int = 1000
    telegram_progress_every_steps: int = 100
    telegram_db_every_steps: int = 2500

