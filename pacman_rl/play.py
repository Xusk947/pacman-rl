import argparse
import os
import time
from dataclasses import dataclass

from pacman_rl.env import make_pacman_env
from pacman_rl.utils import pick_device


@dataclass(frozen=True)
class PlayArgs:
    model_path: str
    env_id: str
    seed: int
    episodes: int
    max_steps: int
    frame_stack: int
    deterministic: bool
    device: str
    render_mode: str
    record_video_dir: str | None
    video_length: int
    video_trigger_steps: int
    render_fps: int
    video_name_prefix: str


def parse_args(argv: list[str] | None = None) -> PlayArgs:
    parser = argparse.ArgumentParser(prog="pacman-rl-play")
    parser.add_argument("--model", dest="model_path", required=True)
    parser.add_argument("--env-id", default="ALE/Pacman-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=os.environ.get("PACMAN_RL_DEVICE", "auto"),
    )
    parser.add_argument(
        "--render",
        dest="render_mode",
        choices=["human", "rgb_array"],
        default="human",
    )
    parser.add_argument("--record-video-dir", default=os.environ.get("PACMAN_RL_VIDEO_DIR"))
    parser.add_argument("--video-length", type=int, default=3000)
    parser.add_argument("--video-trigger-steps", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--video-name-prefix", default="play")

    ns = parser.parse_args(argv)
    return PlayArgs(
        model_path=str(ns.model_path),
        env_id=str(ns.env_id),
        seed=int(ns.seed),
        episodes=int(ns.episodes),
        max_steps=int(ns.max_steps),
        frame_stack=int(ns.frame_stack),
        deterministic=bool(ns.deterministic),
        device=str(ns.device),
        render_mode=str(ns.render_mode),
        record_video_dir=str(ns.record_video_dir) if ns.record_video_dir else None,
        video_length=int(ns.video_length),
        video_trigger_steps=int(ns.video_trigger_steps),
        render_fps=int(ns.render_fps),
        video_name_prefix=str(ns.video_name_prefix),
    )


def _load_model(model_path: str, *, device: str, env):
    from stable_baselines3 import A2C, PPO

    errors: list[Exception] = []
    for cls in (PPO, A2C):
        try:
            return cls.load(model_path, env=env, device=device)
        except Exception as e:
            errors.append(e)
    raise RuntimeError(f"Failed to load model: {model_path}. Errors: {errors!r}")


def play(args: PlayArgs) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    device = pick_device(args.device)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    venv = DummyVecEnv([make_pacman_env(args.env_id, seed=args.seed, render_mode=args.render_mode)])
    try:
        venv.envs[0].metadata["render_fps"] = int(args.render_fps)
    except Exception:
        pass

    if args.record_video_dir is not None and str(args.record_video_dir).strip() != "":
        from stable_baselines3.common.vec_env import VecVideoRecorder

        venv = VecVideoRecorder(
            venv,
            video_folder=str(args.record_video_dir),
            record_video_trigger=lambda step: int(step) % max(1, int(args.video_trigger_steps)) == 0,
            video_length=int(args.video_length),
            name_prefix=str(args.video_name_prefix),
        )

    venv = VecTransposeImage(venv)
    if args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(args.frame_stack))

    model = _load_model(args.model_path, device=device, env=venv)

    obs = venv.reset()
    total_steps = 0
    episodes_done = 0

    while episodes_done < max(1, int(args.episodes)) and total_steps < max(1, int(args.max_steps)):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = venv.step(action)
        total_steps += 1

        if args.render_mode == "human":
            try:
                venv.render()
            except Exception:
                pass
            if args.render_fps > 0:
                time.sleep(1.0 / float(args.render_fps))

        if bool(done[0]):
            episodes_done += 1

    venv.close()
    try:
        del model
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    play(args)


if __name__ == "__main__":
    main()
