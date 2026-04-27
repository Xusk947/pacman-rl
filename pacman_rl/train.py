from __future__ import annotations

import os
import glob
import logging
from dataclasses import dataclass, asdict
from typing import Any

from pacman_rl.callbacks import PacmanSQLiteCallback
from pacman_rl.db import SQLiteLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainJob:
    env_id: str
    db_path: str
    total_timesteps: int
    algos: tuple[str, ...]
    seed: int
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


def _build_vec_env(*, env_id: str, seed: int, n_envs: int, frame_stack: int, record_video_dir: str | None, video_length: int, video_trigger_steps: int) -> Any:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    from pacman_rl.env import make_pacman_env

    env_fns = [make_pacman_env(env_id, seed=seed + i) for i in range(n_envs)]
    venv = DummyVecEnv(env_fns)

    if record_video_dir is not None and str(record_video_dir).strip() != "":
        from stable_baselines3.common.vec_env import VecVideoRecorder

        venv = VecVideoRecorder(
            venv,
            video_folder=str(record_video_dir),
            record_video_trigger=lambda step: int(step) % max(1, int(video_trigger_steps)) == 0,
            video_length=int(video_length),
            name_prefix="train",
        )

    venv = VecTransposeImage(venv)
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv


def _make_model(algo: str, *, env: Any, device: str, seed: int) -> Any:
    algo = algo.lower()
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO("CnnPolicy", env, device=device, seed=seed, verbose=1)
    if algo == "a2c":
        from stable_baselines3 import A2C

        return A2C("CnnPolicy", env, device=device, seed=seed, verbose=1)
    if algo == "dqn":
        from stable_baselines3 import DQN

        return DQN(
            "CnnPolicy",
            env,
            device=device,
            seed=seed,
            verbose=1,
            buffer_size=250_000,
            learning_starts=50_000,
            train_freq=4,
            target_update_interval=10_000,
        )

    raise ValueError(f"Unknown algo: {algo}")


def run_train_job(job: TrainJob) -> None:
    telegram = None
    try:
        from pacman_rl.telegram_reporter import TelegramReporter, detect_telegram_config

        telegram = TelegramReporter(detect_telegram_config())
    except Exception:
        telegram = None

    db = SQLiteLogger(job.db_path)
    try:
        if telegram and telegram.enabled:
            telegram.send_or_edit(f"обучаем модель 0/{job.total_timesteps}\nэтап 0/{len(job.algos)}")

        model_total = max(1, len(job.algos))
        for model_index, algo in enumerate(job.algos, start=1):
            n_envs = 1 if algo.lower() == "dqn" else max(1, int(job.n_envs))
            venv = _build_vec_env(
                env_id=job.env_id,
                seed=job.seed,
                n_envs=n_envs,
                frame_stack=job.frame_stack,
                record_video_dir=job.record_video_dir,
                video_length=job.video_length,
                video_trigger_steps=job.video_trigger_steps,
            )
            run_id: str | None = None
            try:
                config: dict[str, Any] = asdict(job) | {"algo": algo, "n_envs_effective": n_envs}
                run_id = db.start_run(
                    algo=algo,
                    env_id=job.env_id,
                    seed=job.seed,
                    device=job.device,
                    total_timesteps=job.total_timesteps,
                    config=config,
                )

                callback = PacmanSQLiteCallback(
                    db=db,
                    run_id=run_id,
                    algo=algo,
                    model_index=model_index,
                    model_total=model_total,
                    total_timesteps=job.total_timesteps,
                    win_score_threshold=job.win_score_threshold,
                    log_every_steps=job.log_every_steps,
                    estimate_total_pellets=True,
                    print_every_percent=job.print_every_percent,
                    stats_window_episodes=job.stats_window_episodes,
                    telegram=telegram if (telegram and telegram.enabled) else None,
                )
                model = _make_model(algo, env=venv, device=job.device, seed=job.seed)

                try:
                    model.learn(total_timesteps=job.total_timesteps, callback=callback)
                except KeyboardInterrupt:
                    pass
                finally:
                    model_path = os.path.join(job.models_dir, f"{run_id}_{algo}.zip")
                    try:
                        model.save(model_path)
                    except Exception:
                        pass
            finally:
                if run_id is not None:
                    try:
                        db.end_run(run_id)
                    except Exception:
                        pass
                venv.close()
    except Exception as e:
        if telegram and telegram.enabled:
            try:
                telegram.send_or_edit(f"ошибка обучения: {e}")
            except Exception:
                pass
        raise
    finally:
        db.close()

    if telegram and telegram.enabled:
        try:
            from pacman_rl.report import ReportArgs, generate_report

            out_dir = "artifacts"
            report_episodes = int(os.environ.get("PACMAN_RL_TG_REPORT_EPISODES", "1"))
            report_max_steps = int(os.environ.get("PACMAN_RL_TG_REPORT_MAX_STEPS", "5000"))
            report_video_length = int(os.environ.get("PACMAN_RL_TG_REPORT_VIDEO_LENGTH", "1500"))
            report_render_fps = int(os.environ.get("PACMAN_RL_TG_REPORT_FPS", "60"))
            generate_report(
                ReportArgs(
                    db_path=job.db_path,
                    models_dir=job.models_dir,
                    out_dir=out_dir,
                    episodes=report_episodes,
                    max_steps=report_max_steps,
                    frame_stack=job.frame_stack,
                    deterministic=True,
                    device=job.device,
                    video_length=report_video_length,
                    render_fps=report_render_fps,
                )
            )

            telegram.send_or_edit("обучение завершено, отправляю артефакты…")

            videos = sorted(glob.glob(os.path.join(out_dir, "videos", "**", "*.mp4"), recursive=True))
            for vp in videos:
                telegram.send_video(vp, caption=os.path.basename(vp))

            telegram.send_document(job.db_path, caption="runs.sqlite")
            telegram.send_or_edit("готово")
        except Exception as e:
            logger.error("Telegram finalization failed: %s", e)
            try:
                telegram.send_or_edit(f"ошибка при отправке артефактов: {e}")
            except Exception:
                pass
