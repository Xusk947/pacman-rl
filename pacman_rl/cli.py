import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from pacman_rl.utils import parse_int_tuple, pick_device


@dataclass(frozen=True)
class CliArgs:
    env_id: str
    db_path: str
    total_timesteps: int
    algos: tuple[str, ...]
    seeds: tuple[int, ...]
    n_envs: int
    frame_stack: int
    win_score_threshold: float
    log_every_steps: int
    print_every_percent: int
    stats_window_episodes: int
    device: str
    models_dir: str
    record_video_dir: str | None
    video_length: int
    video_trigger_steps: int


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(prog="pacman-rl-train")
    parser.add_argument("--env-id", default="ALE/Pacman-v5")
    parser.add_argument("--db", dest="db_path", default="runs.sqlite")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--algos", nargs="+", default=["ppo", "a2c"])
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--win-score-threshold", type=float, default=500.0)
    parser.add_argument("--log-every-steps", type=int, default=10_000)
    parser.add_argument("--print-every-percent", type=int, default=5)
    parser.add_argument("--stats-window-episodes", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=os.environ.get("PACMAN_RL_DEVICE", "auto"),
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--record-video-dir", default=os.environ.get("PACMAN_RL_VIDEO_DIR"))
    parser.add_argument("--video-length", type=int, default=1800)
    parser.add_argument("--video-trigger-steps", type=int, default=50_000)

    ns = parser.parse_args(argv)
    algos = tuple(a.strip().lower() for a in ns.algos)
    seeds = parse_int_tuple(str(ns.seeds))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return CliArgs(
        env_id=str(ns.env_id),
        db_path=str(ns.db_path),
        total_timesteps=int(ns.total_timesteps),
        algos=algos,
        seeds=seeds,
        n_envs=int(ns.n_envs),
        frame_stack=int(ns.frame_stack),
        win_score_threshold=float(ns.win_score_threshold),
        log_every_steps=int(ns.log_every_steps),
        print_every_percent=int(ns.print_every_percent),
        stats_window_episodes=int(ns.stats_window_episodes),
        device=str(ns.device),
        models_dir=str(ns.models_dir),
        record_video_dir=str(ns.record_video_dir) if ns.record_video_dir else None,
        video_length=int(ns.video_length),
        video_trigger_steps=int(ns.video_trigger_steps),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    requested_device = args.device
    if requested_device in ("cpu", "auto", "cuda"):
        has_nvidia = Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists()
        if requested_device == "cpu" or (requested_device in ("auto", "cuda") and not has_nvidia):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            if requested_device == "cuda":
                requested_device = "cpu"

    device = pick_device(requested_device)
    os.makedirs(args.models_dir, exist_ok=True)

    from pacman_rl.train import TrainJob, run_train_job

    job = TrainJob(
        env_id=args.env_id,
        db_path=args.db_path,
        total_timesteps=args.total_timesteps,
        algos=args.algos,
        seeds=args.seeds,
        n_envs=args.n_envs,
        frame_stack=args.frame_stack,
        win_score_threshold=args.win_score_threshold,
        log_every_steps=args.log_every_steps,
        print_every_percent=args.print_every_percent,
        stats_window_episodes=args.stats_window_episodes,
        device=device,
        models_dir=args.models_dir,
        record_video_dir=args.record_video_dir,
        video_length=args.video_length,
        video_trigger_steps=args.video_trigger_steps,
    )
    run_train_job(job)


if __name__ == "__main__":
    main()
