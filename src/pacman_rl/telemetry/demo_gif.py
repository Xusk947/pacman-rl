from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageDraw

from pacman_rl.env import TorchPacmanEnv
from pacman_rl.ghosts import ClassicGhostPolicy
from pacman_rl.layouts import ParsedLayout


def _frame_from_env(env: TorchPacmanEnv, *, scale: int = 8) -> Image.Image:
    h = int(env.height)
    w = int(env.width)
    img = Image.new("RGB", (w * scale, h * scale), (0, 0, 0))
    d = ImageDraw.Draw(img)

    walls = env.walls[0].to(torch.bool)
    pellets = env.pellets[0].to(torch.bool)
    power = env.power[0].to(torch.bool)
    pac = env.pacman[0].to(torch.int64)
    ghosts = env.ghosts[0].to(torch.int64)
    frightened = bool((env.frightened[0] > 0).item())

    for r in range(h):
        for c in range(w):
            x0 = c * scale
            y0 = r * scale
            x1 = x0 + scale - 1
            y1 = y0 + scale - 1
            if bool(walls[r, c].item()):
                d.rectangle([x0, y0, x1, y1], fill=(30, 30, 180))
            elif bool(power[r, c].item()):
                d.rectangle([x0, y0, x1, y1], fill=(0, 180, 255))
            elif bool(pellets[r, c].item()):
                d.rectangle([x0, y0, x1, y1], fill=(230, 230, 230))

    pr = int(pac[0].item())
    pc = int(pac[1].item())
    d.ellipse(
        [pc * scale + 1, pr * scale + 1, pc * scale + scale - 2, pr * scale + scale - 2],
        fill=(255, 220, 0),
    )

    colors = [(255, 0, 0), (255, 105, 180), (0, 255, 255), (255, 165, 0)]
    if frightened:
        colors = [(0, 120, 255)] * 4

    for gi in range(4):
        gr = int(ghosts[gi, 0].item())
        gc = int(ghosts[gi, 1].item())
        col = colors[gi]
        d.ellipse(
            [gc * scale + 1, gr * scale + 1, gc * scale + scale - 2, gr * scale + scale - 2],
            fill=col,
        )

    return img


def make_demo_gif(
    *,
    layout: ParsedLayout,
    device: torch.device,
    pacman_action: Callable[[torch.Tensor], torch.Tensor],
    max_steps: int = 256,
    scale: int = 8,
    out_path: Path,
) -> Path:
    env = TorchPacmanEnv([layout], batch_size=1, device=device, cfg=env_cfg_default())
    env.reset(seed=0)
    ghosts = ClassicGhostPolicy(device=device)
    ghosts.reset(batch_size=1, pacman_pos=env.pacman)

    frames: list[Image.Image] = []
    obs = env.get_obs()
    for _ in range(int(max_steps)):
        frames.append(_frame_from_env(env, scale=scale))
        pac_obs = obs[:, 0]
        with torch.no_grad():
            pac_act = pacman_action(pac_obs).to(device=device, dtype=torch.int64).view(1)
        ghost_act = ghosts.act(
            walls=env.walls,
            pacman_pos=env.pacman,
            ghost_pos=env.ghosts,
            frightened=env.frightened,
            step_in_ep=env.steps,
        ).view(1, 4)

        acts = torch.zeros((1, env.AGENTS), device=device, dtype=torch.int64)
        acts[:, 0] = pac_act
        acts[:, 1:] = ghost_act

        out = env.step(acts)
        obs = out.obs
        if bool(out.done.item()):
            frames.append(_frame_from_env(env, scale=scale))
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        frames = [Image.new("RGB", (env.width * scale, env.height * scale), (0, 0, 0))]
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    return out_path


def env_cfg_default():
    from pacman_rl.config import EnvConfig

    return EnvConfig()

