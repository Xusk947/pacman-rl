from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pacman_rl.baselines import make_baseline
    from pacman_rl.evaluation import EvalConfig, EpisodeResult, evaluate_policy, evaluate_sb3_model
    from pacman_rl.stats import bootstrap_mean_ci
    from pacman_rl.utils import parse_int_tuple
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from pacman_rl.baselines import make_baseline
    from pacman_rl.evaluation import EvalConfig, EpisodeResult, evaluate_policy, evaluate_sb3_model
    from pacman_rl.stats import bootstrap_mean_ci
    from pacman_rl.utils import parse_int_tuple

DEFAULT_ENV_ID = "ALE/Pacman-v5"
DEFAULT_ALGOS: tuple[str, ...] = ("ppo", "a2c")
DEFAULT_BASELINES: tuple[str, ...] = ("random", "heuristic")
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_WIN_SCORE_THRESHOLD = 500.0
DEFAULT_GRID_POINTS = 60
DEFAULT_BOOTSTRAP_SAMPLES = 4000


@dataclass(frozen=True)
class RunRef:
    run_id: str
    algo: str
    seed: int
    rowid: int


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fetch_latest_runs_by_algo_seed(db_path: str, *, algos: tuple[str, ...]) -> list[RunRef]:
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
        ended_col = "ended_at" if "ended_at" in cols else ("ended_at_unix" if "ended_at_unix" in cols else None)

        where = ""
        if ended_col is not None:
            where = f"WHERE {ended_col} IS NOT NULL"
        rows = con.execute(f"SELECT run_id, algo, seed, rowid FROM runs {where} ORDER BY rowid DESC").fetchall()

        wanted = {str(a).strip().lower() for a in algos if str(a).strip()}
        best: dict[tuple[str, int], RunRef] = {}
        for run_id, algo, seed, rowid in rows:
            a = str(algo).strip().lower()
            if a not in wanted:
                continue
            s = int(seed)
            key = (a, s)
            if key not in best:
                best[key] = RunRef(run_id=str(run_id), algo=a, seed=s, rowid=int(rowid))
        out = [best[k] for k in sorted(best.keys(), key=lambda x: (x[0], x[1]))]
        return out
    finally:
        con.close()


def _read_training_series(db_path: str, run_id: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT timestep, metrics_json FROM training_metrics WHERE run_id = ? ORDER BY timestep ASC",
            (str(run_id),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for t, js in rows:
            d = {}
            try:
                d = json.loads(str(js))
            except Exception:
                d = {}
            d["timestep"] = int(t)
            out.append(d)
        return out
    finally:
        con.close()


def _extract_xy(series: list[dict[str, Any]], *, key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[float] = []
    for d in series:
        try:
            x = int(d.get("timestep", 0))
        except Exception:
            continue
        try:
            y = float(d.get(key, float("nan")))
        except Exception:
            y = float("nan")
        if y != y:
            continue
        xs.append(x)
        ys.append(float(y))
    if not xs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    x_arr = np.array(xs, dtype=np.int64)
    y_arr = np.array(ys, dtype=np.float64)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    uniq_x: list[int] = []
    uniq_y: list[float] = []
    last_x = None
    for x, y in zip(x_arr.tolist(), y_arr.tolist(), strict=False):
        if last_x is not None and int(x) == int(last_x):
            uniq_y[-1] = float(y)
            continue
        uniq_x.append(int(x))
        uniq_y.append(float(y))
        last_x = int(x)

    return np.array(uniq_x, dtype=np.int64), np.array(uniq_y, dtype=np.float64)


def _common_grid(curves: list[tuple[np.ndarray, np.ndarray]], *, points: int) -> np.ndarray:
    if not curves:
        return np.array([], dtype=np.int64)
    mins = [int(x[0]) for x, _ in curves if x.size > 0]
    maxs = [int(x[-1]) for x, _ in curves if x.size > 0]
    if not mins or not maxs:
        return np.array([], dtype=np.int64)
    lo = int(max(mins))
    hi = int(min(maxs))
    if hi <= lo:
        lo = int(min(mins))
        hi = int(max(maxs))
    pts = max(3, int(points))
    grid = np.linspace(lo, hi, num=pts)
    return np.array([int(round(v)) for v in grid], dtype=np.int64)


def _bootstrap_ci_per_timestep(values_by_seed: np.ndarray, *, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for i in range(values_by_seed.shape[1]):
        xs = [float(v) for v in values_by_seed[:, i].tolist() if float(v) == float(v)]
        ci = bootstrap_mean_ci(xs, n_boot=int(n_boot), seed=int(seed))
        means.append(float(ci.mean))
        los.append(float(ci.lo))
        his.append(float(ci.hi))
    return np.array(means, dtype=np.float64), np.array(los, dtype=np.float64), np.array(his, dtype=np.float64)


def _plot_line_ci(out_path: Path, *, xs: np.ndarray, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, mean, linewidth=2.0)
    ax.fill_between(xs, lo, hi, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_overlay_with_ci(
    out_path: Path,
    *,
    xs: np.ndarray,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    for label, (mean, lo, hi) in series.items():
        ax.plot(xs, mean, linewidth=2.0, label=label)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _evaluate_episode_results(results: list[EpisodeResult], *, win_score_threshold: float) -> tuple[float, float]:
    if not results:
        return float("nan"), float("nan")
    returns = [float(r.episode_return) for r in results]
    wins = [1.0 if float(r.episode_return) >= float(win_score_threshold) else 0.0 for r in results]
    return float(np.mean(returns)), float(np.mean(wins))


def _evaluate_model(model_path: str, *, cfg: EvalConfig, eval_seeds: tuple[int, ...], episodes: int, max_steps: int, win_score_threshold: float) -> tuple[float, float]:
    all_eps: list[EpisodeResult] = []
    for s in eval_seeds:
        res = evaluate_sb3_model(
            str(model_path),
            cfg=cfg,
            seed=int(s),
            episodes=int(episodes),
            max_steps=int(max_steps),
        )
        all_eps.extend(res)
    return _evaluate_episode_results(all_eps, win_score_threshold=float(win_score_threshold))


def _evaluate_baseline(name: str, *, cfg: EvalConfig, eval_seeds: tuple[int, ...], episodes: int, max_steps: int, win_score_threshold: float) -> tuple[float, float]:
    all_eps: list[EpisodeResult] = []
    for s in eval_seeds:
        policy = make_baseline(str(name))
        res = evaluate_policy(
            policy,
            cfg=cfg,
            seed=int(s),
            episodes=int(episodes),
            max_steps=int(max_steps),
        )
        all_eps.extend(res)
    return _evaluate_episode_results(all_eps, win_score_threshold=float(win_score_threshold))


def _plot_bar_ci(
    out_path: Path,
    *,
    labels: list[str],
    means: list[float],
    lo: list[float],
    hi: list[float],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(len(labels), dtype=np.int64)
    y = np.array(means, dtype=np.float64)
    yerr_lo = y - np.array(lo, dtype=np.float64)
    yerr_hi = np.array(hi, dtype=np.float64) - y
    yerr = np.vstack([yerr_lo, yerr_hi])

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(xs, y, yerr=yerr, capsize=4, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="generate_plots")
    parser.add_argument("--db", dest="db_path", required=True)
    parser.add_argument("--out-dir", default="artifacts/analysis_plots")
    parser.add_argument("--algos", nargs="+", default=list(DEFAULT_ALGOS))
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--models-dir", default="")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--win-score-threshold", type=float, default=DEFAULT_WIN_SCORE_THRESHOLD)
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))

    ns = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    algos = tuple(str(a).strip().lower() for a in ns.algos if str(a).strip())
    out_dir = Path(str(ns.out_dir))
    _safe_mkdir(out_dir)

    runs = _fetch_latest_runs_by_algo_seed(str(ns.db_path), algos=algos)
    if not runs:
        raise RuntimeError(f"No runs found for algos={algos} in {ns.db_path}")

    algo_to_curves_return: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    algo_to_curves_win: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for r in runs:
        s = _read_training_series(str(ns.db_path), r.run_id)
        x_ret, y_ret = _extract_xy(s, key="window/return_mean")
        x_win, y_win = _extract_xy(s, key="window/winrate")
        if x_ret.size > 2:
            algo_to_curves_return.setdefault(r.algo, []).append((x_ret, y_ret))
        if x_win.size > 2:
            algo_to_curves_win.setdefault(r.algo, []).append((x_win, y_win))

    all_curves = []
    for a in algos:
        all_curves.extend(algo_to_curves_return.get(a, []))
    grid = _common_grid(all_curves, points=int(ns.grid_points))
    if grid.size <= 0:
        raise RuntimeError("Failed to build a common timestep grid from training_metrics")

    overlay_return: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    overlay_win: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for a in algos:
        curves_ret = algo_to_curves_return.get(a, [])
        curves_win = algo_to_curves_win.get(a, [])

        if curves_ret:
            vals = []
            for x, y in curves_ret:
                vals.append(np.interp(grid.astype(np.float64), x.astype(np.float64), y.astype(np.float64)))
            mat = np.stack(vals, axis=0)
            mean, lo, hi = _bootstrap_ci_per_timestep(mat, n_boot=int(ns.n_boot), seed=int(ns.seed))
            overlay_return[a] = (mean, lo, hi)

        if curves_win:
            vals = []
            for x, y in curves_win:
                vals.append(np.interp(grid.astype(np.float64), x.astype(np.float64), y.astype(np.float64)))
            mat = np.stack(vals, axis=0)
            mean, lo, hi = _bootstrap_ci_per_timestep(mat, n_boot=int(ns.n_boot), seed=int(ns.seed))
            overlay_win[a] = (mean, lo, hi)

    if overlay_return:
        _plot_overlay_with_ci(
            out_dir / "training_return_mean_ci.png",
            xs=grid,
            series=overlay_return,
            title="Training: return_mean (window) with bootstrap CI across training seeds",
            ylabel="return_mean (window)",
        )
    if overlay_win:
        _plot_overlay_with_ci(
            out_dir / "training_winrate_ci.png",
            xs=grid,
            series=overlay_win,
            title="Training: winrate (window) with bootstrap CI across training seeds",
            ylabel="winrate (window)",
        )

    models_dir = str(ns.models_dir).strip()
    if models_dir:
        cfg = EvalConfig(
            env_id=DEFAULT_ENV_ID,
            frame_stack=4,
            device="auto",
            deterministic=True,
            render_mode="rgb_array",
            render_fps=60,
        )

        eval_seeds = parse_int_tuple(str(ns.eval_seeds))
        if not eval_seeds:
            raise ValueError("--eval-seeds must contain at least one integer")

        models_root = Path(models_dir)
        algo_to_seed_eval_return: dict[str, list[float]] = {}
        algo_to_seed_eval_win: dict[str, list[float]] = {}
        for r in runs:
            mp = models_root / f"{r.run_id}_{r.algo}.zip"
            if not mp.exists():
                continue
            m_ret, m_win = _evaluate_model(
                str(mp),
                cfg=cfg,
                eval_seeds=eval_seeds,
                episodes=int(ns.episodes),
                max_steps=int(ns.max_steps),
                win_score_threshold=float(ns.win_score_threshold),
            )
            algo_to_seed_eval_return.setdefault(r.algo, []).append(float(m_ret))
            algo_to_seed_eval_win.setdefault(r.algo, []).append(float(m_win))

        labels: list[str] = []
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []

        for a in algos:
            xs = algo_to_seed_eval_return.get(a, [])
            if not xs:
                continue
            ci = bootstrap_mean_ci(xs, n_boot=int(ns.n_boot), seed=int(ns.seed))
            labels.append(a)
            means.append(float(ci.mean))
            los.append(float(ci.lo))
            his.append(float(ci.hi))

        for b in [str(x).strip().lower() for x in ns.baselines if str(x).strip()]:
            b_ret, _ = _evaluate_baseline(
                b,
                cfg=cfg,
                eval_seeds=eval_seeds,
                episodes=int(ns.episodes),
                max_steps=int(ns.max_steps),
                win_score_threshold=float(ns.win_score_threshold),
            )
            labels.append(b)
            means.append(float(b_ret))
            los.append(float(b_ret))
            his.append(float(b_ret))

        if labels:
            _plot_bar_ci(
                out_dir / "eval_return_mean_ci.png",
                labels=labels,
                means=means,
                lo=los,
                hi=his,
                title="Evaluation: return_mean (baselines vs RL) with CI across training seeds",
                ylabel="return_mean",
            )

        labels = []
        means = []
        los = []
        his = []
        for a in algos:
            xs = algo_to_seed_eval_win.get(a, [])
            if not xs:
                continue
            ci = bootstrap_mean_ci(xs, n_boot=int(ns.n_boot), seed=int(ns.seed))
            labels.append(a)
            means.append(float(ci.mean))
            los.append(float(ci.lo))
            his.append(float(ci.hi))

        for b in [str(x).strip().lower() for x in ns.baselines if str(x).strip()]:
            _, b_win = _evaluate_baseline(
                b,
                cfg=cfg,
                eval_seeds=eval_seeds,
                episodes=int(ns.episodes),
                max_steps=int(ns.max_steps),
                win_score_threshold=float(ns.win_score_threshold),
            )
            labels.append(b)
            means.append(float(b_win))
            los.append(float(b_win))
            his.append(float(b_win))

        if labels:
            _plot_bar_ci(
                out_dir / "eval_winrate_ci.png",
                labels=labels,
                means=means,
                lo=los,
                hi=his,
                title="Evaluation: winrate (baselines vs RL) with CI across training seeds",
                ylabel="winrate",
            )


if __name__ == "__main__":
    main()
