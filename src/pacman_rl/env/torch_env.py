from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from pacman_rl.config import EnvConfig
from pacman_rl.layouts import ParsedLayout


@dataclass(frozen=True)
class StepOutput:
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: dict[str, torch.Tensor]


class TorchPacmanEnv:
    ACTIONS: Final[int] = 5
    AGENTS: Final[int] = 5

    def __init__(self, layouts: list[ParsedLayout], *, batch_size: int, device: torch.device, cfg: EnvConfig) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not layouts:
            raise ValueError("layouts must be non-empty")
        self.layouts = layouts
        self.batch_size = int(batch_size)
        self.device = device
        self.cfg = cfg

        self.layout_idx = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64)
        self.height = max(l.height for l in layouts)
        self.width = max(l.width for l in layouts)

        self._walls = torch.zeros((len(layouts), self.height, self.width), dtype=torch.bool)
        self._pellets0 = torch.zeros((len(layouts), self.height, self.width), dtype=torch.bool)
        self._power0 = torch.zeros((len(layouts), self.height, self.width), dtype=torch.bool)
        self._pac0 = torch.zeros((len(layouts), 2), dtype=torch.int64)
        self._ghost0 = torch.zeros((len(layouts), 4, 2), dtype=torch.int64)

        for i, lay in enumerate(layouts):
            for r in range(self.height):
                if r >= lay.height:
                    continue
                row = lay.rows[r]
                for c in range(self.width):
                    if c >= lay.width:
                        continue
                    ch = row[c]
                    if ch == "#":
                        self._walls[i, r, c] = True
                    elif ch == ".":
                        self._pellets0[i, r, c] = True
                    elif ch == "o":
                        self._power0[i, r, c] = True
            self._pac0[i] = torch.tensor(list(lay.pacman_spawn), dtype=torch.int64)
            for j, g in enumerate(("B", "P", "I", "C")):
                self._ghost0[i, j] = torch.tensor(list(lay.ghost_spawns[g]), dtype=torch.int64)

        self._walls = self._walls.to(self.device)
        self._pellets0 = self._pellets0.to(self.device)
        self._power0 = self._power0.to(self.device)
        self._pac0 = self._pac0.to(self.device)
        self._ghost0 = self._ghost0.to(self.device)

        self.reset()

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        g = None
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(seed))

        idx = torch.randint(low=0, high=len(self.layouts), size=(self.batch_size,), generator=g, device="cpu")
        self.layout_idx = idx.to(self.device, dtype=torch.int64)

        self.steps = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64)
        self.frightened = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64)

        self.walls = self._walls[self.layout_idx]
        self.pellets = self._pellets0[self.layout_idx].clone()
        self.power = self._power0[self.layout_idx].clone()

        self.pacman = self._pac0[self.layout_idx].clone()
        self.ghosts = self._ghost0[self.layout_idx].clone()
        self.ghosts_home = self._ghost0[self.layout_idx].clone()

        self._done = torch.zeros((self.batch_size,), device=self.device, dtype=torch.bool)
        return self.get_obs()

    def reset_done(self, done: torch.Tensor) -> torch.Tensor:
        done = done.to(self.device, dtype=torch.bool)
        if not bool(done.any().item()):
            return self.get_obs()

        idx = done.nonzero(as_tuple=False).squeeze(-1)
        replace = torch.randint(low=0, high=len(self.layouts), size=(idx.numel(),), device="cpu").to(self.device)
        self.layout_idx[idx] = replace
        self.steps[idx] = 0
        self.frightened[idx] = 0

        self.walls[idx] = self._walls[self.layout_idx[idx]]
        self.pellets[idx] = self._pellets0[self.layout_idx[idx]].clone()
        self.power[idx] = self._power0[self.layout_idx[idx]].clone()
        self.pacman[idx] = self._pac0[self.layout_idx[idx]].clone()
        self.ghosts[idx] = self._ghost0[self.layout_idx[idx]].clone()
        self.ghosts_home[idx] = self._ghost0[self.layout_idx[idx]].clone()
        self._done[idx] = False
        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
        b = self.batch_size
        h = self.height
        w = self.width

        base = torch.zeros((b, 8, h, w), device=self.device, dtype=torch.float32)
        base[:, 0] = self.walls.to(torch.float32)
        base[:, 1] = self.pellets.to(torch.float32)
        base[:, 2] = self.power.to(torch.float32)

        pr = self.pacman[:, 0].clamp(0, h - 1)
        pc = self.pacman[:, 1].clamp(0, w - 1)
        base[torch.arange(b, device=self.device), 3, pr, pc] = 1.0

        for gi in range(4):
            gr = self.ghosts[:, gi, 0].clamp(0, h - 1)
            gc = self.ghosts[:, gi, 1].clamp(0, w - 1)
            base[torch.arange(b, device=self.device), 4 + gi, gr, gc] = 1.0

        obs = torch.zeros((b, self.AGENTS, 13, h, w), device=self.device, dtype=torch.float32)
        obs[:, :, :8] = base[:, None]
        for a in range(self.AGENTS):
            obs[:, a, 8 + a] = 1.0
        return obs

    def step(self, actions: torch.Tensor) -> StepOutput:
        if actions.shape != (self.batch_size, self.AGENTS):
            raise ValueError(f"actions must have shape (batch, {self.AGENTS})")

        actions = actions.to(self.device, dtype=torch.int64)
        pac_act = actions[:, 0]
        ghost_act = actions[:, 1:]

        self.steps += 1
        self.frightened = torch.clamp(self.frightened - 1, min=0)

        pac_next = self._apply_move(self.pacman, pac_act)
        ghosts_next = self._apply_move_ghosts(self.ghosts, ghost_act)

        self.pacman = pac_next
        self.ghosts = ghosts_next

        b = self.batch_size
        h = self.height
        w = self.width
        idx = torch.arange(b, device=self.device)
        pr = self.pacman[:, 0]
        pc = self.pacman[:, 1]

        pellet_here = self.pellets[idx, pr, pc]
        power_here = self.power[idx, pr, pc]
        self.pellets[idx, pr, pc] = torch.where(pellet_here, torch.zeros_like(pellet_here), self.pellets[idx, pr, pc])
        self.power[idx, pr, pc] = torch.where(power_here, torch.zeros_like(power_here), self.power[idx, pr, pc])

        frightened_now = power_here.to(torch.int64) * int(self.cfg.frightened_steps)
        self.frightened = torch.maximum(self.frightened, frightened_now)

        pac_reward = torch.full((b,), float(self.cfg.reward_step_pacman), device=self.device, dtype=torch.float32)
        pac_reward += pellet_here.to(torch.float32) * float(self.cfg.reward_pellet)
        pac_reward += power_here.to(torch.float32) * float(self.cfg.reward_power)

        ghost_reward = torch.full((b, 4), float(self.cfg.reward_step_ghost), device=self.device, dtype=torch.float32)

        coll = (self.ghosts == self.pacman[:, None]).all(dim=-1)
        any_coll = coll.any(dim=-1)
        frightened = self.frightened > 0

        eat_mask = any_coll & frightened
        die_mask = any_coll & (~frightened)

        if bool(eat_mask.any().item()):
            pac_reward = pac_reward + eat_mask.to(torch.float32) * float(self.cfg.reward_ghost_eat)

            for gi in range(4):
                eaten = eat_mask & coll[:, gi]
                if bool(eaten.any().item()):
                    self.ghosts[eaten, gi] = self.ghosts_home[eaten, gi]
                    ghost_reward[eaten, gi] = ghost_reward[eaten, gi] - float(self.cfg.reward_ghost_eat)

        if bool(die_mask.any().item()):
            pac_reward = pac_reward + die_mask.to(torch.float32) * float(self.cfg.reward_death)
            ghost_reward = ghost_reward + die_mask[:, None].to(torch.float32) * (float(self.cfg.reward_ghost_catch) / 4.0)

        pellets_left = self.pellets.any(dim=-1).any(dim=-1) | self.power.any(dim=-1).any(dim=-1)
        win_mask = ~pellets_left

        timeout_mask = self.steps >= int(self.cfg.max_steps)

        done = die_mask | win_mask | timeout_mask
        self._done = done

        if bool(win_mask.any().item()):
            pac_reward = pac_reward + win_mask.to(torch.float32) * float(self.cfg.reward_win)
            ghost_reward = ghost_reward - win_mask[:, None].to(torch.float32) * (float(self.cfg.reward_win) / 4.0)

        reward = torch.zeros((b, self.AGENTS), device=self.device, dtype=torch.float32)
        reward[:, 0] = pac_reward
        reward[:, 1:] = ghost_reward

        obs = self.get_obs()

        info = {
            "pellet_eaten": pellet_here.to(torch.int64),
            "power_eaten": power_here.to(torch.int64),
            "frightened": self.frightened.to(torch.int64),
            "pac_dead": die_mask.to(torch.int64),
            "win": win_mask.to(torch.int64),
            "timeout": timeout_mask.to(torch.int64),
            "layout_idx": self.layout_idx.to(torch.int64),
        }

        return StepOutput(obs=obs, reward=reward, done=done, info=info)

    def _apply_move(self, pos: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        delta = self._delta(act)
        nxt = pos + delta
        nxt = self._clip(nxt)
        ok = ~self.walls[torch.arange(self.batch_size, device=self.device), nxt[:, 0], nxt[:, 1]]
        return torch.where(ok[:, None], nxt, pos)

    def _apply_move_ghosts(self, pos: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        b = self.batch_size
        delta = self._delta(act.reshape(b * 4)).reshape(b, 4, 2)
        nxt = pos + delta
        nxt = self._clip(nxt)

        br = torch.arange(b, device=self.device)[:, None].repeat(1, 4)
        ok = ~self.walls[br, nxt[:, :, 0], nxt[:, :, 1]]
        return torch.where(ok[:, :, None], nxt, pos)

    def _clip(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.clone()
        pos[..., 0] = pos[..., 0].clamp(0, self.height - 1)
        pos[..., 1] = pos[..., 1].clamp(0, self.width - 1)
        return pos

    def _delta(self, act: torch.Tensor) -> torch.Tensor:
        act = act.to(self.device, dtype=torch.int64)
        dr = torch.zeros_like(act)
        dc = torch.zeros_like(act)

        dr = torch.where(act == 0, -torch.ones_like(act), dr)
        dr = torch.where(act == 1, torch.ones_like(act), dr)
        dc = torch.where(act == 2, -torch.ones_like(act), dc)
        dc = torch.where(act == 3, torch.ones_like(act), dc)
        return torch.stack([dr, dc], dim=-1)
