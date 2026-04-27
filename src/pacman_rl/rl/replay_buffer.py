from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, *, capacity: int, obs_shape: tuple[int, int, int], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        c, h, w = obs_shape
        self.obs = torch.zeros((self.capacity, c, h, w), dtype=torch.uint8, device=torch.device("cpu"))
        self.next_obs = torch.zeros((self.capacity, c, h, w), dtype=torch.uint8, device=torch.device("cpu"))
        self.actions = torch.zeros((self.capacity,), dtype=torch.int64, device=torch.device("cpu"))
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=torch.device("cpu"))
        self.dones = torch.zeros((self.capacity,), dtype=torch.bool, device=torch.device("cpu"))
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(
        self, *, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor
    ) -> None:
        b = obs.shape[0]
        obs_u8 = obs.to(torch.uint8).to(torch.device("cpu"))
        next_u8 = next_obs.to(torch.uint8).to(torch.device("cpu"))
        actions = actions.to(torch.int64).to(torch.device("cpu"))
        rewards = rewards.to(torch.float32).to(torch.device("cpu"))
        dones = dones.to(torch.bool).to(torch.device("cpu"))

        for i in range(b):
            self.obs[self._pos] = obs_u8[i]
            self.next_obs[self._pos] = next_u8[i]
            self.actions[self._pos] = actions[i]
            self.rewards[self._pos] = rewards[i]
            self.dones[self._pos] = dones[i]
            self._pos = (self._pos + 1) % self.capacity
            self._size = min(self.capacity, self._size + 1)

    def sample(self, *, batch_size: int) -> ReplayBatch:
        if self._size <= 0:
            raise ValueError("replay buffer is empty")
        n = int(batch_size)
        idx = torch.randint(low=0, high=self._size, size=(n,), device=torch.device("cpu"))
        obs = self.obs[idx].to(self.device, dtype=torch.float32)
        next_obs = self.next_obs[idx].to(self.device, dtype=torch.float32)
        actions = self.actions[idx].to(self.device)
        rewards = self.rewards[idx].to(self.device)
        dones = self.dones[idx].to(self.device)
        return ReplayBatch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)

