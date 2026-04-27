import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pacman_rl.baselines import make_baseline
from pacman_rl.evaluation import EvalConfig, EpisodeResult, evaluate_policy, evaluate_sb3_model
from pacman_rl.play import PlayArgs, play
from pacman_rl.stats import bootstrap_mean_ci
from pacman_rl.utils import parse_int_tuple

DEFAULT_WIN_SCORE_THRESHOLD = 500.0


@dataclass(frozen=True)
class ReportArgs:
    db_path: str
    models_dir: str
    out_dir: str
    episodes: int
    max_steps: int
    frame_stack: int
    deterministic: bool
    device: str
    video_length: int
    render_fps: int
    algos: tuple[str, ...] = ("ppo", "a2c")
    eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    baselines: tuple[str, ...] = ("random", "heuristic")
    win_score_threshold: float = DEFAULT_WIN_SCORE_THRESHOLD
    skip_videos: bool = False


def parse_args(argv: list[str] | None = None) -> ReportArgs:
    parser = argparse.ArgumentParser(prog="pacman-rl-report")
    parser.add_argument("--db", dest="db_path", default="runs.sqlite")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=os.environ.get("PACMAN_RL_DEVICE", "auto"))
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--algos", nargs="+", default=["ppo", "a2c"])
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--baselines", nargs="+", default=["random", "heuristic"])
    parser.add_argument("--win-score-threshold", type=float, default=DEFAULT_WIN_SCORE_THRESHOLD)
    parser.add_argument("--skip-videos", action="store_true")
    ns = parser.parse_args(argv)
    algos = tuple(str(a).strip().lower() for a in ns.algos if str(a).strip())
    eval_seeds = parse_int_tuple(str(ns.eval_seeds))
    if not eval_seeds:
        raise ValueError("--eval-seeds must contain at least one integer")
    baselines = tuple(str(b).strip().lower() for b in ns.baselines if str(b).strip())
    return ReportArgs(
        db_path=str(ns.db_path),
        models_dir=str(ns.models_dir),
        out_dir=str(ns.out_dir),
        episodes=int(ns.episodes),
        max_steps=int(ns.max_steps),
        frame_stack=int(ns.frame_stack),
        deterministic=bool(ns.deterministic),
        device=str(ns.device),
        video_length=int(ns.video_length),
        render_fps=int(ns.render_fps),
        algos=algos,
        eval_seeds=eval_seeds,
        baselines=baselines,
        win_score_threshold=float(ns.win_score_threshold),
        skip_videos=bool(ns.skip_videos),
    )


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fetch_runs(db_path: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
        ended_col = "ended_at" if "ended_at" in cols else ("ended_at_unix" if "ended_at_unix" in cols else None)
        if ended_col is None:
            rows = con.execute("SELECT run_id, algo, seed, config_json, rowid FROM runs ORDER BY rowid ASC").fetchall()
        else:
            rows = con.execute(f"SELECT run_id, algo, seed, config_json, rowid FROM runs WHERE {ended_col} IS NOT NULL ORDER BY rowid DESC").fetchall()

        out: list[dict[str, Any]] = []
        for run_id, algo, seed, config_json, rowid in rows:
            cfg = {}
            try:
                cfg = json.loads(str(config_json))
            except Exception:
                cfg = {}
            win_thr = cfg.get("win_score_threshold")
            try:
                win_thr_f = float(win_thr)
            except Exception:
                win_thr_f = DEFAULT_WIN_SCORE_THRESHOLD

            out.append(
                {
                    "run_id": str(run_id),
                    "algo": str(algo).lower(),
                    "seed": int(seed),
                    "win_score_threshold": float(win_thr_f),
                    "rowid": int(rowid),
                }
            )
        return out
    finally:
        con.close()


def _read_training_series(db_path: str, run_id: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT timestep, metrics_json FROM training_metrics WHERE run_id = ? ORDER BY timestep ASC",
            (run_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for t, js in rows:
            try:
                d = json.loads(js)
            except Exception:
                d = {}
            d["timestep"] = int(t)
            out.append(d)
        return out
    finally:
        con.close()


def _plot_series(out_path: Path, *, x: list[int], y: list[float], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_overlay(out_path: Path, *, xs: list[int], series: dict[str, list[float]], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    for label, ys in series.items():
        ax.plot(xs, ys, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _select_latest_runs_by_algo_seed(runs: list[dict[str, Any]], *, algos: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = {str(a).lower() for a in algos}
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in runs:
        algo = str(r.get("algo", "")).lower()
        if algo not in wanted:
            continue
        seed = int(r.get("seed", 0))
        key = (algo, seed)
        prev = best.get(key)
        if prev is None or int(r.get("rowid", 0)) > int(prev.get("rowid", 0)):
            best[key] = r
    return [best[k] for k in sorted(best.keys(), key=lambda x: (x[0], x[1]))]


def _flatten_episode_results(results: list[EpisodeResult]) -> list[float]:
    return [float(r.episode_return) for r in results]


def _winrate(results: list[EpisodeResult], *, threshold: float) -> float:
    if not results:
        return float("nan")
    wins = [1.0 if float(r.episode_return) >= float(threshold) else 0.0 for r in results]
    return float(sum(wins) / float(len(wins)))


def _evaluate_model_across_eval_seeds(
    *,
    model_path: str,
    cfg: EvalConfig,
    eval_seeds: tuple[int, ...],
    episodes_per_seed: int,
    max_steps: int,
    win_score_threshold: float,
) -> dict[str, Any]:
    ep_returns: list[float] = []
    ep_results: list[EpisodeResult] = []
    for s in eval_seeds:
        res = evaluate_sb3_model(
            model_path,
            cfg=cfg,
            seed=int(s),
            episodes=int(episodes_per_seed),
            max_steps=int(max_steps),
        )
        ep_results.extend(res)
        ep_returns.extend(_flatten_episode_results(res))

    return {
        "episodes": int(len(ep_returns)),
        "return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "winrate": float(_winrate(ep_results, threshold=float(win_score_threshold))),
    }


def _evaluate_baseline_across_eval_seeds(
    *,
    baseline_name: str,
    cfg: EvalConfig,
    eval_seeds: tuple[int, ...],
    episodes_per_seed: int,
    max_steps: int,
    win_score_threshold: float,
) -> dict[str, Any]:
    ep_results: list[EpisodeResult] = []
    for s in eval_seeds:
        policy = make_baseline(baseline_name)
        res = evaluate_policy(
            policy,
            cfg=cfg,
            seed=int(s),
            episodes=int(episodes_per_seed),
            max_steps=int(max_steps),
        )
        ep_results.extend(res)

    ep_returns = _flatten_episode_results(ep_results)
    return {
        "episodes": int(len(ep_returns)),
        "return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "winrate": float(_winrate(ep_results, threshold=float(win_score_threshold))),
    }



def generate_report(args: ReportArgs) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    summary_dir = out_dir / "summary"
    plots_dir = out_dir / "plots"
    _safe_mkdir(videos_dir)
    _safe_mkdir(summary_dir)
    _safe_mkdir(plots_dir)

    runs = _fetch_runs(args.db_path)
    if not runs:
        raise RuntimeError(f"No runs found in {args.db_path}")

    selected_runs = _select_latest_runs_by_algo_seed(runs, algos=args.algos)
    if not selected_runs:
        raise RuntimeError(f"No completed runs found for algos={args.algos} in {args.db_path}")

    cfg = EvalConfig(
        env_id="ALE/Pacman-v5",
        frame_stack=int(args.frame_stack),
        device=str(args.device),
        deterministic=bool(args.deterministic),
        render_mode="rgb_array",
        render_fps=int(args.render_fps),
    )

    summary: dict[str, Any] = {
        "algos": {},
        "baselines": {},
        "eval": {"eval_seeds": list(args.eval_seeds), "episodes_per_seed": int(args.episodes), "max_steps": int(args.max_steps)},
    }

    algo_to_seed_scores: dict[str, list[float]] = {}
    algo_to_seed_winrates: dict[str, list[float]] = {}

    models_dir = Path(args.models_dir)
    for r in selected_runs:
        run_id = str(r["run_id"])
        seed = int(r.get("seed", 0))
        win_thr = float(r.get("win_score_threshold", args.win_score_threshold))
        algo = str(r["algo"]).lower()
        model_path = models_dir / f"{run_id}_{algo}.zip"
        if not model_path.exists():
            continue
        if not bool(args.skip_videos):
            play_args = PlayArgs(
                model_path=str(model_path),
                env_id="ALE/Pacman-v5",
                seed=int(args.eval_seeds[0]),
                episodes=args.episodes,
                max_steps=args.max_steps,
                frame_stack=args.frame_stack,
                deterministic=args.deterministic,
                device=args.device,
                render_mode="rgb_array",
                record_video_dir=str(videos_dir / f"{algo}_{run_id}"),
                video_length=args.video_length,
                video_trigger_steps=1,
                render_fps=args.render_fps,
                video_name_prefix=f"{algo}_{run_id}",
            )
            play(play_args)

        eval_metrics = _evaluate_model_across_eval_seeds(
            model_path=str(model_path),
            cfg=cfg,
            eval_seeds=args.eval_seeds,
            episodes_per_seed=int(args.episodes),
            max_steps=int(args.max_steps),
            win_score_threshold=float(win_thr),
        )

        summary["algos"].setdefault(algo, {"runs": []})
        summary["algos"][algo]["runs"].append({"run_id": run_id, "train_seed": int(seed), "eval": eval_metrics})
        algo_to_seed_scores.setdefault(algo, []).append(float(eval_metrics["return_mean"]))
        algo_to_seed_winrates.setdefault(algo, []).append(float(eval_metrics["winrate"]))

        series = _read_training_series(args.db_path, run_id)
        if not series:
            continue

        xs = [int(d.get("timestep", 0)) for d in series]
        ret_mean = [float(d.get("window/return_mean", float("nan"))) for d in series]
        winrate = [float(d.get("window/winrate", float("nan"))) for d in series]
        ep_rew_mean = [float(d.get("rollout/ep_rew_mean", float("nan"))) for d in series]

        if any(v == v for v in ret_mean):
            _plot_series(
                plots_dir / f"{algo}_{run_id}_return_mean.png",
                x=xs,
                y=ret_mean,
                title=f"{algo} {run_id} return_mean (window)",
                ylabel="return_mean",
            )
        if any(v == v for v in winrate):
            _plot_series(
                plots_dir / f"{algo}_{run_id}_winrate.png",
                x=xs,
                y=winrate,
                title=f"{algo} {run_id} winrate (window)",
                ylabel="winrate",
            )
        if any(v == v for v in ep_rew_mean):
            _plot_series(
                plots_dir / f"{algo}_{run_id}_ep_rew_mean.png",
                x=xs,
                y=ep_rew_mean,
                title=f"{algo} {run_id} ep_rew_mean (sb3)",
                ylabel="ep_rew_mean",
            )

    for algo, seed_means in algo_to_seed_scores.items():
        ci = bootstrap_mean_ci(seed_means, seed=0)
        wci = bootstrap_mean_ci(algo_to_seed_winrates.get(algo, []), seed=0)
        summary["algos"][algo]["return_mean_ci"] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}
        summary["algos"][algo]["winrate_ci"] = {"mean": wci.mean, "lo": wci.lo, "hi": wci.hi, "n": wci.n}

    for b in args.baselines:
        b_metrics = _evaluate_baseline_across_eval_seeds(
            baseline_name=str(b),
            cfg=cfg,
            eval_seeds=args.eval_seeds,
            episodes_per_seed=int(args.episodes),
            max_steps=int(args.max_steps),
            win_score_threshold=float(args.win_score_threshold),
        )
        summary["baselines"][str(b)] = b_metrics

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    lines = ["name,type,n,return_mean,return_ci_lo,return_ci_hi,winrate,win_ci_lo,win_ci_hi"]
    for algo in sorted(summary["algos"].keys()):
        ci = summary["algos"][algo].get("return_mean_ci", {})
        wci = summary["algos"][algo].get("winrate_ci", {})
        lines.append(
            f"{algo},algo,{ci.get('n','')},{ci.get('mean','')},{ci.get('lo','')},{ci.get('hi','')},{wci.get('mean','')},{wci.get('lo','')},{wci.get('hi','')}"
        )
    for b in sorted(summary["baselines"].keys()):
        m = summary["baselines"][b]
        lines.append(f"{b},baseline,{m.get('episodes','')},{m.get('return_mean','')},,,{m.get('winrate','')},,")
    (summary_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    overlay_xs = sorted({int(r.get("seed", 0)) for r in selected_runs})
    if overlay_xs:
        overlay_series: dict[str, list[float]] = {}
        for algo in sorted(algo_to_seed_scores.keys()):
            ys = []
            for s in overlay_xs:
                found = None
                for r in summary["algos"][algo]["runs"]:
                    if int(r.get("train_seed", -1)) == int(s):
                        found = r
                        break
                ys.append(float(found["eval"]["return_mean"]) if found else float("nan"))
            overlay_series[algo] = ys
        _plot_overlay(
            plots_dir / "eval_return_by_train_seed.png",
            xs=overlay_xs,
            series=overlay_series,
            title="Evaluation return_mean by training seed",
            ylabel="return_mean",
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_report(args)


if __name__ == "__main__":
    main()
