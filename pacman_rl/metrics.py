from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardEvents:
    pellets: int
    power_pellets: int
    ghosts: int


def parse_pacman_reward_events(reward: float) -> RewardEvents:
    r = int(round(float(reward)))

    if r in (1, 10):
        return RewardEvents(pellets=1, power_pellets=0, ghosts=0)
    if r in (5, 50):
        return RewardEvents(pellets=0, power_pellets=1, ghosts=0)
    if r in (20, 40, 80, 160, 200, 400, 800, 1600):
        return RewardEvents(pellets=0, power_pellets=0, ghosts=1)

    return RewardEvents(pellets=0, power_pellets=0, ghosts=0)


class PelletTotalEstimator:
    def __init__(self, *, min_blob_size: int = 2, max_blob_size: int = 30) -> None:
        self._min_blob_size = int(min_blob_size)
        self._max_blob_size = int(max_blob_size)

    def estimate_total_from_rgb(self, frame_rgb: np.ndarray) -> int:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            return 0
        gray = (
            0.2126 * frame_rgb[..., 0].astype(np.float32)
            + 0.7152 * frame_rgb[..., 1].astype(np.float32)
            + 0.0722 * frame_rgb[..., 2].astype(np.float32)
        )
        mask = gray > 200.0
        return self._count_small_blobs(mask)

    def _count_small_blobs(self, mask: np.ndarray) -> int:
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        count = 0

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue

                stack: list[tuple[int, int]] = [(y, x)]
                visited[y, x] = 1
                size = 0

                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    if size > self._max_blob_size:
                        stack.clear()
                        break

                    ny = cy - 1
                    if ny >= 0 and mask[ny, cx] and not visited[ny, cx]:
                        visited[ny, cx] = 1
                        stack.append((ny, cx))
                    ny = cy + 1
                    if ny < h and mask[ny, cx] and not visited[ny, cx]:
                        visited[ny, cx] = 1
                        stack.append((ny, cx))
                    nx = cx - 1
                    if nx >= 0 and mask[cy, nx] and not visited[cy, nx]:
                        visited[cy, nx] = 1
                        stack.append((cy, nx))
                    nx = cx + 1
                    if nx < w and mask[cy, nx] and not visited[cy, nx]:
                        visited[cy, nx] = 1
                        stack.append((cy, nx))

                if self._min_blob_size <= size <= self._max_blob_size:
                    count += 1

        return count
