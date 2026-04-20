from __future__ import annotations

from dataclasses import dataclass

import torch

from pacman_rl.config import EnvConfig
from pacman_rl.layouts import ParsedLayout


@dataclass(frozen=True)
class StepOutput:
    pac_obs: torch.Tensor
    ghost_obs: torch.Tensor
    pac_reward: torch.Tensor
    ghost_reward: torch.Tensor
    done: torch.Tensor
    pellet_eaten: torch.Tensor
    power_eaten: torch.Tensor
    ghosts_eaten: torch.Tensor
    pac_dead: torch.Tensor
    all_pellets_done: torch.Tensor
    timeout: torch.Tensor


def _pos_to_index(pos_rc: torch.Tensor, width: int) -> torch.Tensor:
    return pos_rc[..., 0] * width + pos_rc[..., 1]


def _in_bounds(pos_rc: torch.Tensor, height: int, width: int) -> torch.Tensor:
    r_ok = (0 <= pos_rc[..., 0]) & (pos_rc[..., 0] < height)
    c_ok = (0 <= pos_rc[..., 1]) & (pos_rc[..., 1] < width)
    return r_ok & c_ok


def _gather_bool_grid(grid_bhw: torch.Tensor, pos_b2: torch.Tensor) -> torch.Tensor:
    b, h, w = grid_bhw.shape
    flat = grid_bhw.view(b, h * w)
    idx = _pos_to_index(pos_b2, w)
    return flat[torch.arange(b, device=grid_bhw.device), idx]


def _scatter_bool_grid(grid_bhw: torch.Tensor, pos_b2: torch.Tensor, value: bool) -> torch.Tensor:
    b, h, w = grid_bhw.shape
    flat = grid_bhw.view(b, h * w)
    idx = _pos_to_index(pos_b2, w)
    flat[torch.arange(b, device=grid_bhw.device), idx] = value
    return grid_bhw


def _one_hot_positions(pos: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b = pos.shape[0]
    out = torch.zeros((b, height, width), device=pos.device, dtype=torch.float32)
    idx = _pos_to_index(pos, width)
    out.view(b, height * width)[torch.arange(b, device=pos.device), idx] = 1.0
    return out


def _one_hot_positions_multi(pos: torch.Tensor, present: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b, g, _ = pos.shape
    out = torch.zeros((b, g, height, width), device=pos.device, dtype=torch.float32)
    idx = _pos_to_index(pos, width)
    flat = out.view(b, g, height * width)
    flat[torch.arange(b, device=pos.device)[:, None], torch.arange(g, device=pos.device)[None, :], idx] = present.to(
        torch.float32
    )
    return out


def _scared_channels(pos: torch.Tensor, scared: torch.Tensor, present: torch.Tensor, scared_steps: int, height: int, width: int) -> torch.Tensor:
    b, g, _ = pos.shape
    out = torch.zeros((b, g, height, width), device=pos.device, dtype=torch.float32)
    idx = _pos_to_index(pos, width)
    flat = out.view(b, g, height * width)
    val = (scared.to(torch.float32) / float(scared_steps)) * present.to(torch.float32)
    flat[torch.arange(b, device=pos.device)[:, None], torch.arange(g, device=pos.device)[None, :], idx] = val
    return out


class TorchPacmanEnv:
    GMAX = 4
    ACTIONS = 4

    def __init__(self, layouts: list[ParsedLayout], *, batch_size: int, device: torch.device, cfg: EnvConfig) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not layouts:
            raise ValueError("layouts must be non-empty")

        h0 = max(l.height for l in layouts)
        w0 = max(l.width for l in layouts)

        self.device = device
        self.cfg = cfg
        self.batch_size = batch_size
        self.height = h0
        self.width = w0

        self._build_layout_bank(layouts)
        self.reset()

    def _build_layout_bank(self, layouts: list[ParsedLayout]) -> None:
        h, w = self.height, self.width
        l = len(layouts)

        walls = torch.ones((l, h, w), dtype=torch.bool)
        pellets = torch.zeros((l, h, w), dtype=torch.bool)
        power = torch.zeros((l, h, w), dtype=torch.bool)
        pac_spawn = torch.zeros((l, 2), dtype=torch.int64)
        ghost_spawn = torch.zeros((l, self.GMAX, 2), dtype=torch.int64)
        ghost_present = torch.zeros((l, self.GMAX), dtype=torch.bool)

        for i, lay in enumerate(layouts):
            pac_spawn[i] = torch.tensor(lay.pacman_spawn, dtype=torch.int64)

            g = min(len(lay.ghost_spawns), self.GMAX)
            if g <= 0:
                raise ValueError("layout must contain at least one ghost spawn")
            ghost_spawn[i, :g] = torch.tensor(lay.ghost_spawns[:g], dtype=torch.int64)
            ghost_present[i, :g] = True

            for r, line in enumerate(lay.rows):
                for c, ch in enumerate(line):
                    if ch == "%":
                        walls[i, r, c] = True
                    elif ch == ".":
                        pellets[i, r, c] = True
                        walls[i, r, c] = False
                    elif ch == "o":
                        power[i, r, c] = True
                        walls[i, r, c] = False
                    else:
                        walls[i, r, c] = False

        self._bank_walls = walls.to(self.device)
        self._bank_pellets = pellets.to(self.device)
        self._bank_power = power.to(self.device)
        self._bank_pac_spawn = pac_spawn.to(self.device)
        self._bank_ghost_spawn = ghost_spawn.to(self.device)
        self._bank_ghost_present = ghost_present.to(self.device)
        self._bank_size = l

    def reset(self) -> StepOutput:
        b, h, w = self.batch_size, self.height, self.width
        dev = self.device

        layout_idx = torch.randint(0, self._bank_size, size=(b,), device=dev)
        self._layout_idx = layout_idx

        self.walls = self._bank_walls.index_select(0, layout_idx)
        self._pellets0 = self._bank_pellets.index_select(0, layout_idx)
        self._power0 = self._bank_power.index_select(0, layout_idx)

        self.pellets = self._pellets0.clone()
        self.power = self._power0.clone()

        self.pac_xy = self._bank_pac_spawn.index_select(0, layout_idx).clone()
        self.ghost_spawn = self._bank_ghost_spawn.index_select(0, layout_idx).clone()
        self.ghost_present = self._bank_ghost_present.index_select(0, layout_idx).clone()

        self.ghost_xy = self.ghost_spawn.clone()
        self.scared = torch.zeros((b, self.GMAX), device=dev, dtype=torch.int64)
        self.step_count = torch.zeros((b,), device=dev, dtype=torch.int64)

        pac_obs, ghost_obs = self._build_obs()
        zeros_b = torch.zeros((b,), device=dev, dtype=torch.float32)
        zeros_bg = torch.zeros((b, self.GMAX), device=dev, dtype=torch.float32)
        done = torch.zeros((b,), device=dev, dtype=torch.bool)
        zeros_b_bool = torch.zeros((b,), device=dev, dtype=torch.bool)
        zeros_b_i64 = torch.zeros((b,), device=dev, dtype=torch.int64)
        return StepOutput(
            pac_obs=pac_obs,
            ghost_obs=ghost_obs,
            pac_reward=zeros_b,
            ghost_reward=zeros_bg,
            done=done,
            pellet_eaten=zeros_b_bool,
            power_eaten=zeros_b_bool,
            ghosts_eaten=zeros_b_i64,
            pac_dead=zeros_b_bool,
            all_pellets_done=zeros_b_bool,
            timeout=zeros_b_bool,
        )

    def reset_with_layout_indices(self, layout_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layout_idx = layout_idx.to(device=self.device, dtype=torch.int64)
        if layout_idx.shape != (self.batch_size,):
            raise ValueError("layout_idx must be shape [B]")
        if layout_idx.min().item() < 0 or layout_idx.max().item() >= self._bank_size:
            raise ValueError("layout_idx out of bounds")

        self._layout_idx = layout_idx

        self.walls = self._bank_walls.index_select(0, layout_idx)
        self._pellets0 = self._bank_pellets.index_select(0, layout_idx)
        self._power0 = self._bank_power.index_select(0, layout_idx)

        self.pellets = self._pellets0.clone()
        self.power = self._power0.clone()

        self.pac_xy = self._bank_pac_spawn.index_select(0, layout_idx).clone()
        self.ghost_spawn = self._bank_ghost_spawn.index_select(0, layout_idx).clone()
        self.ghost_present = self._bank_ghost_present.index_select(0, layout_idx).clone()

        self.ghost_xy = self.ghost_spawn.clone()
        self.scared = torch.zeros((self.batch_size, self.GMAX), device=self.device, dtype=torch.int64)
        self.step_count = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64)

        return self._build_obs()

    def reset_done(self, done_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        done_mask = done_mask.to(device=self.device, dtype=torch.bool)
        if done_mask.ndim != 1 or done_mask.shape[0] != self.batch_size:
            raise ValueError("done_mask must be shape [B]")

        if not done_mask.any():
            return self._build_obs()

        b_idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
        new_layout_idx = torch.randint(0, self._bank_size, size=(b_idx.shape[0],), device=self.device)
        self._layout_idx[b_idx] = new_layout_idx

        self.walls[b_idx] = self._bank_walls.index_select(0, new_layout_idx)
        self._pellets0[b_idx] = self._bank_pellets.index_select(0, new_layout_idx)
        self._power0[b_idx] = self._bank_power.index_select(0, new_layout_idx)

        self.pellets[b_idx] = self._pellets0[b_idx]
        self.power[b_idx] = self._power0[b_idx]

        self.pac_xy[b_idx] = self._bank_pac_spawn.index_select(0, new_layout_idx)
        self.ghost_spawn[b_idx] = self._bank_ghost_spawn.index_select(0, new_layout_idx)
        self.ghost_present[b_idx] = self._bank_ghost_present.index_select(0, new_layout_idx)

        self.ghost_xy[b_idx] = self.ghost_spawn[b_idx]
        self.scared[b_idx] = 0
        self.step_count[b_idx] = 0

        return self._build_obs()

    def _apply_move(self, pos: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = pos.shape[0]
        h, w = self.height, self.width
        dev = pos.device

        delta = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            device=dev,
            dtype=torch.int64,
        )

        cand = pos + delta[action]
        inb = _in_bounds(cand, h, w)

        cand_r = torch.clamp(cand[:, 0], 0, h - 1)
        cand_c = torch.clamp(cand[:, 1], 0, w - 1)
        cand_clamped = torch.stack([cand_r, cand_c], dim=-1)

        wall = self.walls[torch.arange(b, device=dev), cand_clamped[:, 0], cand_clamped[:, 1]]
        ok = inb & ~wall
        next_pos = torch.where(ok[:, None], cand, pos)
        wall_bump = ~ok
        return next_pos, wall_bump

    def _apply_move_ghosts(self, pos: torch.Tensor, action: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, g, _ = pos.shape
        next_pos = pos.clone()
        wall_bump = torch.zeros((b, g), device=pos.device, dtype=torch.bool)
        for i in range(g):
            npos, bump = self._apply_move(pos[:, i], action[:, i])
            next_pos[:, i] = torch.where(present[:, i, None], npos, pos[:, i])
            wall_bump[:, i] = torch.where(present[:, i], bump, torch.zeros_like(bump))
        return next_pos, wall_bump

    def step(self, pac_action: torch.Tensor, ghost_action: torch.Tensor) -> StepOutput:
        b = self.batch_size
        dev = self.device

        pac_action = pac_action.to(device=dev, dtype=torch.int64)
        ghost_action = ghost_action.to(device=dev, dtype=torch.int64)

        pac_prev = self.pac_xy
        ghost_prev = self.ghost_xy

        pac_next, bump_pac = self._apply_move(self.pac_xy, pac_action)
        ghost_next, bump_ghost = self._apply_move_ghosts(self.ghost_xy, ghost_action, self.ghost_present)

        self.pac_xy = pac_next
        self.ghost_xy = ghost_next

        same = (pac_next[:, None, :] == ghost_next).all(dim=-1)
        swap = ((pac_next[:, None, :] == ghost_prev).all(dim=-1)) & ((ghost_next == pac_prev[:, None, :]).all(dim=-1))
        collide = (same | swap) & self.ghost_present

        pellet_eaten = _gather_bool_grid(self.pellets, pac_next)
        power_eaten = _gather_bool_grid(self.power, pac_next)

        self.pellets = _scatter_bool_grid(self.pellets, pac_next, False)
        self.power = _scatter_bool_grid(self.power, pac_next, False)

        if self.cfg.scared_steps <= 0:
            raise ValueError("scared_steps must be positive")

        set_scared = power_eaten[:, None] & self.ghost_present
        self.scared = torch.where(set_scared, torch.full_like(self.scared, self.cfg.scared_steps), self.scared)

        scared_now = self.scared > 0
        eaten = collide & scared_now
        killers = collide & ~scared_now

        pac_dead = killers.any(dim=-1)

        if eaten.any():
            for i in range(self.GMAX):
                to_reset = eaten[:, i]
                self.ghost_xy[:, i] = torch.where(to_reset[:, None], self.ghost_spawn[:, i], self.ghost_xy[:, i])
                self.scared[:, i] = torch.where(to_reset, torch.zeros_like(self.scared[:, i]), self.scared[:, i])

        self.scared = torch.clamp(self.scared - 1, min=0)

        all_pellets_done = ~self.pellets.view(b, -1).any(dim=-1)
        next_step = self.step_count + 1
        timeout = next_step >= self.cfg.max_steps
        done = pac_dead | all_pellets_done | timeout
        self.step_count = torch.where(done, torch.zeros_like(self.step_count), next_step)

        eat_ghost_count = eaten.to(torch.int64).sum(dim=-1)

        pac_reward = torch.zeros((b,), device=dev, dtype=torch.float32)
        pac_reward = pac_reward + pellet_eaten.to(torch.float32) * self.cfg.reward_pellet
        pac_reward = pac_reward + power_eaten.to(torch.float32) * self.cfg.reward_power
        pac_reward = pac_reward + eat_ghost_count.to(torch.float32) * self.cfg.reward_eat_ghost
        pac_reward = pac_reward + pac_dead.to(torch.float32) * self.cfg.reward_death
        pac_reward = pac_reward + torch.full((b,), self.cfg.reward_step, device=dev, dtype=torch.float32)
        pac_reward = pac_reward + bump_pac.to(torch.float32) * self.cfg.reward_wall_bump

        catch = pac_dead.to(torch.float32) * self.cfg.ghost_reward_catch
        eaten_penalty = eat_ghost_count.to(torch.float32) * self.cfg.ghost_reward_eaten
        step_penalty = torch.full((b,), self.cfg.ghost_reward_step, device=dev, dtype=torch.float32)
        bump_penalty = bump_ghost.to(torch.float32).mean(dim=-1) * self.cfg.ghost_reward_wall_bump
        team_reward = catch + eaten_penalty + step_penalty + bump_penalty

        ghost_reward = team_reward[:, None].repeat(1, self.GMAX) * self.ghost_present.to(torch.float32)

        pac_obs, ghost_obs = self._build_obs()
        return StepOutput(
            pac_obs=pac_obs,
            ghost_obs=ghost_obs,
            pac_reward=pac_reward,
            ghost_reward=ghost_reward,
            done=done,
            pellet_eaten=pellet_eaten,
            power_eaten=power_eaten,
            ghosts_eaten=eat_ghost_count,
            pac_dead=pac_dead,
            all_pellets_done=all_pellets_done,
            timeout=timeout,
        )

    def get_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._build_obs()

    def _build_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, w = self.batch_size, self.height, self.width
        dev = self.device

        walls = self.walls.to(torch.float32)
        pellets = self.pellets.to(torch.float32)
        power = self.power.to(torch.float32)
        pac_pos = _one_hot_positions(self.pac_xy, h, w)

        ghosts_pos = _one_hot_positions_multi(self.ghost_xy, self.ghost_present, h, w)
        ghosts_scared = _scared_channels(self.ghost_xy, self.scared, self.ghost_present, self.cfg.scared_steps, h, w)

        pac_obs = torch.cat(
            [
                walls[:, None],
                pellets[:, None],
                power[:, None],
                pac_pos[:, None],
                ghosts_pos,
                ghosts_scared,
            ],
            dim=1,
        )

        ghosts_other = torch.zeros((b, self.GMAX, h, w), device=dev, dtype=torch.float32)
        ghosts_other_scared = torch.zeros((b, self.GMAX, h, w), device=dev, dtype=torch.float32)

        for i in range(self.GMAX):
            other_mask = torch.ones((self.GMAX,), device=dev, dtype=torch.bool)
            other_mask[i] = False

            other_pos = ghosts_pos[:, other_mask].sum(dim=1).clamp(max=1.0)
            other_scared = ghosts_scared[:, other_mask].sum(dim=1).clamp(max=1.0)

            ghosts_other[:, i] = other_pos
            ghosts_other_scared[:, i] = other_scared

        self_pos = _one_hot_positions_multi(self.ghost_xy, self.ghost_present, h, w)
        self_scared = _scared_channels(self.ghost_xy, self.scared, self.ghost_present, self.cfg.scared_steps, h, w)

        ghost_obs = torch.cat(
            [
                walls[:, None, None].repeat(1, self.GMAX, 1, 1, 1),
                pellets[:, None, None].repeat(1, self.GMAX, 1, 1, 1),
                power[:, None, None].repeat(1, self.GMAX, 1, 1, 1),
                pac_pos[:, None, None].repeat(1, self.GMAX, 1, 1, 1),
                self_pos[:, :, None],
                self_scared[:, :, None],
                ghosts_other[:, :, None],
                ghosts_other_scared[:, :, None],
            ],
            dim=2,
        )

        return pac_obs, ghost_obs
