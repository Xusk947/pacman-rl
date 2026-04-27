from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

NOOP_ACTION = 0
DEFAULT_MOVE_ACTIONS: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_STICKY_STEPS = 12
DEFAULT_CHANGE_PROB = 0.08


class ActionSpace(Protocol):
    n: int


class Policy(Protocol):
    def reset(self, *, action_space: ActionSpace, seed: int) -> None: ...

    def act(self, obs: Any) -> int: ...

    def observe(self, *, reward: float, done: bool, info: Any) -> None: ...


@dataclass
class RandomPolicy:
    _rng: np.random.Generator | None = None
    _action_n: int = 0

    def reset(self, *, action_space: ActionSpace, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._action_n = int(getattr(action_space, "n", 0))

    def act(self, obs: Any) -> int:
        if self._rng is None or self._action_n <= 0:
            return 0
        return int(self._rng.integers(0, self._action_n))

    def observe(self, *, reward: float, done: bool, info: Any) -> None:
        return None


@dataclass
class StickyHeuristicPolicy:
    move_actions: tuple[int, ...] = DEFAULT_MOVE_ACTIONS
    sticky_steps: int = DEFAULT_STICKY_STEPS
    change_prob: float = DEFAULT_CHANGE_PROB

    _rng: np.random.Generator | None = None
    _action_n: int = 0
    _current_action: int = NOOP_ACTION
    _steps_left: int = 0
    _last_reward: float = 0.0

    def reset(self, *, action_space: ActionSpace, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._action_n = int(getattr(action_space, "n", 0))
        self._current_action = NOOP_ACTION
        self._steps_left = 0
        self._last_reward = 0.0

    def act(self, obs: Any) -> int:
        if self._rng is None or self._action_n <= 0:
            return 0

        if self._steps_left <= 0:
            self._current_action = self._sample_move_action()
            self._steps_left = max(1, int(self.sticky_steps))
        else:
            force_change = self._last_reward < 0.0
            random_change = float(self._rng.random()) < float(self.change_prob)
            if force_change or random_change:
                self._current_action = self._sample_move_action(exclude=self._current_action)
                self._steps_left = max(1, int(self.sticky_steps))

        self._steps_left -= 1
        return int(self._current_action)

    def observe(self, *, reward: float, done: bool, info: Any) -> None:
        self._last_reward = float(reward)
        if done:
            self._steps_left = 0

    def _sample_move_action(self, exclude: int | None = None) -> int:
        if self._rng is None:
            return 0

        candidates = [int(a) for a in self.move_actions if int(a) >= 0 and int(a) < self._action_n and int(a) != int(exclude or -1)]
        if candidates:
            return int(candidates[int(self._rng.integers(0, len(candidates)))])

        fallback = [a for a in range(self._action_n) if a != int(exclude or -1)]
        if not fallback:
            return 0
        return int(fallback[int(self._rng.integers(0, len(fallback)))])


def make_baseline(name: str) -> Policy:
    key = str(name).strip().lower()
    if key in ("random", "rand"):
        return RandomPolicy()
    if key in ("heuristic", "sticky", "sticky_heuristic"):
        return StickyHeuristicPolicy()
    raise ValueError(f"Unknown baseline: {name}")

