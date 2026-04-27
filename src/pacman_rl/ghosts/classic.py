from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GhostPhaseConfig:
    scatter_steps: int = 70
    chase_steps: int = 200


class ClassicGhostPolicy:
    def __init__(self, *, device: torch.device, cfg: GhostPhaseConfig | None = None) -> None:
        self.device = device
        self.cfg = cfg or GhostPhaseConfig()
        self._prev_pac: torch.Tensor | None = None

    def reset(self, *, batch_size: int, pacman_pos: torch.Tensor) -> None:
        self._prev_pac = pacman_pos.detach().clone().to(self.device).reshape(batch_size, 2)

    def act(
        self,
        *,
        walls: torch.Tensor,
        pacman_pos: torch.Tensor,
        ghost_pos: torch.Tensor,
        frightened: torch.Tensor,
        step_in_ep: torch.Tensor,
    ) -> torch.Tensor:
        b, h, w = walls.shape
        pac = pacman_pos.to(self.device, dtype=torch.int64).reshape(b, 2)
        ghosts = ghost_pos.to(self.device, dtype=torch.int64).reshape(b, 4, 2)
        frightened = frightened.to(self.device, dtype=torch.int64).reshape(b)
        step_in_ep = step_in_ep.to(self.device, dtype=torch.int64).reshape(b)

        if self._prev_pac is None or self._prev_pac.shape[0] != b:
            self.reset(batch_size=b, pacman_pos=pac)

        pac_delta = (pac - self._prev_pac).clamp(-1, 1)
        self._prev_pac = pac.detach().clone()
        pac_ahead = pac + 4 * pac_delta
        pac_ahead_r = pac_ahead[:, 0].clamp(0, h - 1)
        pac_ahead_c = pac_ahead[:, 1].clamp(0, w - 1)
        pac_ahead = torch.stack([pac_ahead_r, pac_ahead_c], dim=-1)

        corners = torch.tensor(
            [
                [1, w - 2],
                [1, 1],
                [h - 2, w - 2],
                [h - 2, 1],
            ],
            device=self.device,
            dtype=torch.int64,
        ).unsqueeze(0).repeat(b, 1, 1)

        cycle = self.cfg.scatter_steps + self.cfg.chase_steps
        in_scatter = (step_in_ep % cycle) < int(self.cfg.scatter_steps)

        targets = torch.zeros((b, 4, 2), device=self.device, dtype=torch.int64)

        targets[:, 0] = torch.where(in_scatter[:, None], corners[:, 0], pac)
        targets[:, 1] = torch.where(in_scatter[:, None], corners[:, 1], pac_ahead)

        two_ahead = pac + 2 * pac_delta
        two_ahead_r = two_ahead[:, 0].clamp(0, h - 1)
        two_ahead_c = two_ahead[:, 1].clamp(0, w - 1)
        two_ahead = torch.stack([two_ahead_r, two_ahead_c], dim=-1)
        inky_vec = two_ahead - ghosts[:, 0]
        inky_t = ghosts[:, 0] + 2 * inky_vec
        inky_t_r = inky_t[:, 0].clamp(0, h - 1)
        inky_t_c = inky_t[:, 1].clamp(0, w - 1)
        inky_t = torch.stack([inky_t_r, inky_t_c], dim=-1)
        targets[:, 2] = torch.where(in_scatter[:, None], corners[:, 2], inky_t)

        d_clyde = (ghosts[:, 3] - pac).abs().sum(dim=-1)
        clyde_chase = d_clyde >= 8
        targets[:, 3] = torch.where(
            in_scatter[:, None],
            corners[:, 3],
            torch.where(clyde_chase[:, None], pac, corners[:, 3]),
        )

        actions = torch.zeros((b, 4), device=self.device, dtype=torch.int64)
        for gi in range(4):
            target = targets[:, gi]
            actions[:, gi] = self._best_action(walls=walls, pos=ghosts[:, gi], target=target, frightened=frightened)

        return actions

    def _best_action(
        self,
        *,
        walls: torch.Tensor,
        pos: torch.Tensor,
        target: torch.Tensor,
        frightened: torch.Tensor,
    ) -> torch.Tensor:
        b, h, w = walls.shape
        pos = pos.reshape(b, 2)
        target = target.reshape(b, 2)

        acts = torch.tensor([0, 2, 1, 3, 4], device=self.device, dtype=torch.int64)
        deltas = torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]], device=self.device, dtype=torch.int64)

        cand = pos[:, None, :] + deltas[None, :, :]
        cand_r = cand[:, :, 0].clamp(0, h - 1)
        cand_c = cand[:, :, 1].clamp(0, w - 1)

        br = torch.arange(b, device=self.device)[:, None].repeat(1, acts.shape[0])
        valid = ~walls[br, cand_r, cand_c]

        dist = (cand_r - target[:, None, 0]).abs() + (cand_c - target[:, None, 1]).abs()
        dist = dist.to(torch.float32) + torch.arange(acts.shape[0], device=self.device, dtype=torch.float32)[None, :] * 1e-3

        frightened_mask = frightened[:, None] > 0
        score = torch.where(frightened_mask, -dist, dist)
        score = torch.where(valid, score, torch.full_like(score, float("inf")))

        best = score.argmin(dim=1)
        return acts[best]
