from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

from pacman_rl.play import PlayArgs, play
from pacman_rl.telegram_reporter import TelegramReporter, detect_telegram_config


@dataclass(frozen=True)
class TrainedArgs:
    models_dir: str
    out_dir: str
    algos: tuple[str, ...]
    device: str
    frame_stack: int
    episodes: int
    max_steps: int
    video_length: int
    render_fps: int


def parse_args(argv: list[str] | None = None) -> TrainedArgs:
    parser = argparse.ArgumentParser(prog="pacman-rl-trained")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--algos", nargs="+", default=["ppo", "a2c"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=os.environ.get("PACMAN_RL_DEVICE", "auto"))
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument("--render-fps", type=int, default=60)
    ns = parser.parse_args(argv)
    algos = tuple(str(a).strip().lower() for a in ns.algos if str(a).strip())
    return TrainedArgs(
        models_dir=str(ns.models_dir),
        out_dir=str(ns.out_dir),
        algos=algos,
        device=str(ns.device),
        frame_stack=int(ns.frame_stack),
        episodes=max(1, int(ns.episodes)),
        max_steps=max(1, int(ns.max_steps)),
        video_length=max(1, int(ns.video_length)),
        render_fps=max(1, int(ns.render_fps)),
    )


def _algo_from_model_name(model_path: str) -> str:
    stem = Path(model_path).stem.lower()
    for algo in ("ppo", "a2c"):
        if stem.endswith(f"_{algo}"):
            return algo
    return "model"


def run_trained(args: TrainedArgs) -> None:
    selected: list[tuple[str, str]] = []
    for algo in args.algos:
        candidates = sorted(glob.glob(os.path.join(args.models_dir, f"*_{algo}.zip")))
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: os.path.getmtime(p))
        selected.append((algo, latest))

    if not selected:
        raise RuntimeError(f"No model files found in {args.models_dir} for algos={args.algos}")

    videos_root = os.path.join(args.out_dir, "trained_videos")
    os.makedirs(videos_root, exist_ok=True)

    generated_videos: list[tuple[str, str]] = []
    for algo, mp in selected:
        run_id = Path(mp).stem[:32]
        out_dir = os.path.join(videos_root, f"{algo}_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        play(
            PlayArgs(
                model_path=mp,
                env_id="ALE/Pacman-v5",
                seed=0,
                episodes=args.episodes,
                max_steps=args.max_steps,
                frame_stack=args.frame_stack,
                deterministic=True,
                device=args.device,
                render_mode="rgb_array",
                record_video_dir=out_dir,
                video_length=min(args.video_length, args.max_steps),
                video_trigger_steps=1,
                render_fps=args.render_fps,
                video_name_prefix=f"trained_{algo}",
            )
        )

        files = sorted(glob.glob(os.path.join(out_dir, "*.mp4")))
        if files:
            generated_videos.append((algo, files[-1]))

    tg = TelegramReporter(detect_telegram_config())
    if not tg.enabled:
        return

    max_bytes = int(os.environ.get("PACMAN_RL_TG_MAX_BYTES", str(45 * 1024 * 1024)))
    sent = 0
    skipped = 0
    failed = 0
    for algo, vp in generated_videos:
        try:
            if os.path.getsize(vp) > max_bytes:
                skipped += 1
                continue
        except Exception:
            failed += 1
            continue

        caption = f"(TRAINED) {algo.upper()} {os.path.basename(vp)}"
        if tg.send_video(vp, caption=caption):
            sent += 1
        else:
            failed += 1

    tg.send_or_edit(f"(TRAINED) done\nvideos: sent={sent}, skipped={skipped}, failed={failed}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_trained(args)


if __name__ == "__main__":
    main()
