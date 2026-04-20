from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from pacman_rl.config import EnvConfig
from pacman_rl.env import TorchPacmanEnv
from pacman_rl.layouts import ParsedLayout
from pacman_rl.models import CNNActorCritic


@dataclass(frozen=True)
class GameRecordConfig:
    max_steps: int = 512
    idle_steps: int = 120


def _sample_action(model: CNNActorCritic, obs: torch.Tensor) -> torch.Tensor:
    out = model(obs)
    dist = Categorical(logits=out.logits)
    return dist.sample()


def record_game(
    path: Path,
    *,
    layout: ParsedLayout,
    pacman: CNNActorCritic,
    ghosts: CNNActorCritic,
    device: torch.device,
    env_cfg: EnvConfig,
    cfg: GameRecordConfig = GameRecordConfig(),
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)

    env = TorchPacmanEnv([layout], batch_size=1, device=device, cfg=env_cfg)
    env.reset_with_layout_indices(torch.zeros((1,), device=device, dtype=torch.int64))

    pacman.eval()
    ghosts.eval()

    frames: list[dict[str, Any]] = []
    pac_obs, ghost_obs = env.get_obs()

    initial_pellets = int(env.pellets.sum().item())
    initial_power = int(env.power.sum().item())
    total_pac_reward = 0.0
    ghosts_eaten = 0
    ended_by = "max_steps"

    idle = 0
    last_state = None

    for t in range(cfg.max_steps):
        with torch.no_grad():
            pac_action = _sample_action(pacman, pac_obs)
            g_flat = ghost_obs.view(env.GMAX, ghost_obs.shape[2], env.height, env.width)
            g_action = _sample_action(ghosts, g_flat).view(1, env.GMAX)
            g_action = torch.where(env.ghost_present, g_action, torch.zeros_like(g_action))

        out = env.step(pac_action, g_action)
        total_pac_reward += float(out.pac_reward[0].item())
        ghosts_eaten += int(out.ghosts_eaten[0].item())

        pellets_left = int(env.pellets.sum().item())
        power_left = int(env.power.sum().item())

        state = (
            tuple(env.pac_xy[0].tolist()),
            tuple(tuple(x) for x in env.ghost_xy[0].tolist()),
            tuple(env.ghost_present[0].tolist()),
            tuple(env.scared[0].tolist()),
            pellets_left,
            power_left,
        )
        if last_state is not None and state == last_state:
            idle += 1
        else:
            idle = 0
        last_state = state

        frames.append(
            {
                "t": t,
                "pac_xy": env.pac_xy[0].tolist(),
                "ghost_xy": env.ghost_xy[0].tolist(),
                "ghost_present": env.ghost_present[0].tolist(),
                "scared": env.scared[0].tolist(),
                "pac_action": int(pac_action.item()),
                "ghost_action": g_action[0].tolist(),
                "pac_reward": float(out.pac_reward[0].item()),
                "ghost_reward": out.ghost_reward[0].tolist(),
                "done": bool(out.done[0].item()),
                "pellets_left": pellets_left,
                "power_left": power_left,
            }
        )

        pac_obs, ghost_obs = out.pac_obs, out.ghost_obs
        if out.done[0].item():
            if bool(out.all_pellets_done[0].item()):
                ended_by = "win"
            elif bool(out.pac_dead[0].item()):
                ended_by = "death"
            elif bool(out.timeout[0].item()):
                ended_by = "timeout"
            else:
                ended_by = "done"
            break
        if cfg.idle_steps > 0 and idle >= cfg.idle_steps:
            ended_by = "idle"
            break

    pellets_left = int(env.pellets.sum().item())
    power_left = int(env.power.sum().item())

    summary = {
        "layout": layout.name,
        "steps": int(len(frames)),
        "total_pac_reward": float(total_pac_reward),
        "pellets_eaten": int(initial_pellets - pellets_left),
        "power_eaten": int(initial_power - power_left),
        "ghosts_eaten": int(ghosts_eaten),
        "ended_by": ended_by,
    }

    payload = {
        "layout": {"name": layout.name, "rows": layout.rows, "height": layout.height, "width": layout.width},
        "summary": summary,
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary
