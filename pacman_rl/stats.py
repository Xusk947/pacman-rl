from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> MeanCI:
    xs = np.array([float(v) for v in values if v == v], dtype=np.float64)
    n = int(xs.size)
    if n <= 0:
        return MeanCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    if n == 1:
        m = float(xs[0])
        return MeanCI(mean=m, lo=m, hi=m, n=1)

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(max(1, int(n_boot)), n))
    means = xs[idx].mean(axis=1)

    alpha = (1.0 - float(confidence)) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return MeanCI(mean=float(xs.mean()), lo=lo, hi=hi, n=n)

