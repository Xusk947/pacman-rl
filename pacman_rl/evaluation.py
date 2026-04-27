from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pacman_rl.baselines import Policy
from pacman_rl.env import make_pacman_env
from pacman_rl.utils import pick_device


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    frame_stack: int
    device: str
    deterministic: bool
    render_mode: str
    render_fps: int


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    episode_length: int


def build_vec_env(
    *,
    env_id: str,
    seed: int,
    frame_stack: int,
    render_mode: str,
    render_fps: int,
    record_video_dir: str | None,
    video_length: int,
    video_trigger_steps: int,
    video_name_prefix: str,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    venv = DummyVecEnv([make_pacman_env(env_id, seed=int(seed), render_mode=str(render_mode))])
    try:
        venv.envs[0].metadata["render_fps"] = int(render_fps)
    except Exception:
        pass

    if record_video_dir is not None and str(record_video_dir).strip() != "":
        from stable_baselines3.common.vec_env import VecVideoRecorder

        venv = VecVideoRecorder(
            venv,
            video_folder=str(record_video_dir),
            record_video_trigger=lambda step: int(step) % max(1, int(video_trigger_steps)) == 0,
            video_length=int(video_length),
            name_prefix=str(video_name_prefix),
        )

    venv = VecTransposeImage(venv)
    if int(frame_stack) > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))
    return venv


def load_sb3_model(model_path: str, *, device: str, env):
    from stable_baselines3 import A2C, PPO

    errors: list[Exception] = []
    for cls in (PPO, A2C):
        try:
            return cls.load(model_path, env=env, device=device)
        except Exception as e:
            errors.append(e)
    raise RuntimeError(f"Failed to load model: {model_path}. Errors: {errors!r}")


def evaluate_sb3_model(
    model_path: str,
    *,
    cfg: EvalConfig,
    seed: int,
    episodes: int,
    max_steps: int,
    record_video_dir: str | None = None,
    video_length: int = 600,
    video_trigger_steps: int = 1,
    video_name_prefix: str = "eval",
) -> list[EpisodeResult]:
    device = pick_device(cfg.device)
    venv = build_vec_env(
        env_id=cfg.env_id,
        seed=int(seed),
        frame_stack=int(cfg.frame_stack),
        render_mode=str(cfg.render_mode),
        render_fps=int(cfg.render_fps),
        record_video_dir=record_video_dir,
        video_length=int(video_length),
        video_trigger_steps=int(video_trigger_steps),
        video_name_prefix=str(video_name_prefix),
    )

    try:
        model = load_sb3_model(model_path, device=device, env=venv)
        obs = venv.reset()

        out: list[EpisodeResult] = []
        ep_ret = 0.0
        ep_len = 0
        total_steps = 0

        while len(out) < max(1, int(episodes)) and total_steps < max(1, int(max_steps)):
            action, _ = model.predict(obs, deterministic=bool(cfg.deterministic))
            obs, reward, done, _ = venv.step(action)
            r0 = float(np.array(reward).reshape((-1,))[0])
            d0 = bool(np.array(done).reshape((-1,))[0])

            ep_ret += r0
            ep_len += 1
            total_steps += 1

            if d0:
                out.append(EpisodeResult(episode_return=float(ep_ret), episode_length=int(ep_len)))
                ep_ret = 0.0
                ep_len = 0

        return out
    finally:
        venv.close()
        try:
            del model
        except Exception:
            pass


def evaluate_policy(
    policy: Policy,
    *,
    cfg: EvalConfig,
    seed: int,
    episodes: int,
    max_steps: int,
    record_video_dir: str | None = None,
    video_length: int = 600,
    video_trigger_steps: int = 1,
    video_name_prefix: str = "eval",
) -> list[EpisodeResult]:
    venv = build_vec_env(
        env_id=cfg.env_id,
        seed=int(seed),
        frame_stack=int(cfg.frame_stack),
        render_mode=str(cfg.render_mode),
        render_fps=int(cfg.render_fps),
        record_video_dir=record_video_dir,
        video_length=int(video_length),
        video_trigger_steps=int(video_trigger_steps),
        video_name_prefix=str(video_name_prefix),
    )

    try:
        policy.reset(action_space=venv.action_space, seed=int(seed))
        obs = venv.reset()

        out: list[EpisodeResult] = []
        ep_ret = 0.0
        ep_len = 0
        total_steps = 0

        while len(out) < max(1, int(episodes)) and total_steps < max(1, int(max_steps)):
            action = int(policy.act(obs))
            obs, reward, done, info = venv.step(np.array([action], dtype=np.int64))
            r0 = float(np.array(reward).reshape((-1,))[0])
            d0 = bool(np.array(done).reshape((-1,))[0])
            i0: Any = info[0] if isinstance(info, (list, tuple)) and info else {}

            ep_ret += r0
            ep_len += 1
            total_steps += 1

            policy.observe(reward=r0, done=d0, info=i0)
            if d0:
                out.append(EpisodeResult(episode_return=float(ep_ret), episode_length=int(ep_len)))
                ep_ret = 0.0
                ep_len = 0

        return out
    finally:
        venv.close()
