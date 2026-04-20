from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Checkpoint:
    update: int
    pacman: dict[str, Any]
    ghosts: dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    update: int,
    pacman_model: torch.nn.Module,
    pacman_opt: torch.optim.Optimizer,
    ghosts_model: torch.nn.Module,
    ghosts_opt: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "pacman": {"model": pacman_model.state_dict(), "opt": pacman_opt.state_dict()},
        "ghosts": {"model": ghosts_model.state_dict(), "opt": ghosts_opt.state_dict()},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    pacman_model: torch.nn.Module,
    pacman_opt: torch.optim.Optimizer,
    ghosts_model: torch.nn.Module,
    ghosts_opt: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location)
    pacman_model.load_state_dict(payload["pacman"]["model"])
    pacman_opt.load_state_dict(payload["pacman"]["opt"])
    ghosts_model.load_state_dict(payload["ghosts"]["model"])
    ghosts_opt.load_state_dict(payload["ghosts"]["opt"])
    return int(payload["update"])
