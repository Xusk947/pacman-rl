import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pacman_rl.play import PlayArgs, play


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
    parser.add_argument("--video-length", type=int, default=3000)
    parser.add_argument("--render-fps", type=int, default=60)
    ns = parser.parse_args(argv)
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
    )


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fetch_runs(db_path: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
        ended_col = "ended_at" if "ended_at" in cols else ("ended_at_unix" if "ended_at_unix" in cols else None)
        if ended_col is None:
            rows = con.execute("SELECT run_id, algo FROM runs ORDER BY rowid ASC").fetchall()
        else:
            rows = con.execute(f"SELECT run_id, algo FROM runs WHERE {ended_col} IS NOT NULL ORDER BY rowid ASC").fetchall()
        return [{"run_id": str(r[0]), "algo": str(r[1])} for r in rows]
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


def generate_report(args: ReportArgs) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    plots_dir = out_dir / "plots"
    _safe_mkdir(videos_dir)
    _safe_mkdir(plots_dir)

    runs = _fetch_runs(args.db_path)
    if not runs:
        raise RuntimeError(f"No runs found in {args.db_path}")

    models_dir = Path(args.models_dir)
    for r in runs:
        run_id = str(r["run_id"])
        algo = str(r["algo"]).lower()
        model_path = models_dir / f"{run_id}_{algo}.zip"
        if not model_path.exists():
            continue

        play_args = PlayArgs(
            model_path=str(model_path),
            env_id="ALE/Pacman-v5",
            seed=0,
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_report(args)


if __name__ == "__main__":
    main()
