from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Snapshot:
    state_dict: dict[str, Any]


class SnapshotPool:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[Snapshot] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, model: nn.Module) -> None:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._items.append(Snapshot(state_dict=state))
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def sample_into(self, model: nn.Module, *, rng: torch.Generator | None = None) -> bool:
        if not self._items:
            return False
        idx = int(torch.randint(0, len(self._items), size=(1,), generator=rng).item())
        model.load_state_dict(self._items[idx].state_dict)
        return True
