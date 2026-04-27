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


def _build_vec_env(
    *,
    env_id: str,
    seed: int,
    n_envs: int,
    frame_stack: int,
    record_video_dir: str | None,
    video_length: int,
    video_trigger_steps: int,
    video_name_prefix: str,
) -> Any:
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
            name_prefix=str(video_name_prefix),
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


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    raw = str(text or "").replace(";", ",").replace(" ", ",")
    out: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            continue
    return tuple(out)


def run_train_job(job: TrainJob) -> None:
    telegram = None
    try:
        from pacman_rl.telegram_reporter import TelegramReporter, detect_telegram_config

        telegram = TelegramReporter(detect_telegram_config())
    except Exception:
        telegram = None

    max_bytes = int(os.environ.get("PACMAN_RL_TG_MAX_BYTES", str(45 * 1024 * 1024)))
    milestone_pcts = _parse_int_tuple(os.environ.get("PACMAN_RL_TG_MILESTONE_PCTS", ""))
    milestone_video_dir_raw = os.environ.get("PACMAN_RL_TG_MILESTONE_DIR", "").strip()
    milestone_video_dir = milestone_video_dir_raw if milestone_video_dir_raw else None
    milestone_episodes = int(os.environ.get("PACMAN_RL_TG_MILESTONE_EPISODES", "1"))
    milestone_max_steps_default = min(3000, max(1, int(job.total_timesteps)))
    milestone_max_steps = int(os.environ.get("PACMAN_RL_TG_MILESTONE_MAX_STEPS", str(milestone_max_steps_default)))
    milestone_video_length_default = min(1500, max(1, int(milestone_max_steps)))
    milestone_video_length = int(os.environ.get("PACMAN_RL_TG_MILESTONE_VIDEO_LENGTH", str(milestone_video_length_default)))
    milestone_render_fps = int(os.environ.get("PACMAN_RL_TG_MILESTONE_FPS", "60"))

    db = SQLiteLogger(job.db_path)
    saved_models: list[str] = []
    try:
        if telegram and telegram.enabled:
            telegram.send_or_edit(f"training steps 0/{job.total_timesteps}\nstage 0/{len(job.algos)}")

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
                video_name_prefix=f"train_{algo.lower()}",
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
                    milestone_percents=milestone_pcts if (telegram and telegram.enabled) else (),
                    milestone_models_dir=job.models_dir,
                    milestone_video_dir=milestone_video_dir if (telegram and telegram.enabled) else None,
                    milestone_env_id=job.env_id,
                    milestone_seed=job.seed,
                    milestone_frame_stack=job.frame_stack,
                    milestone_device=job.device,
                    milestone_episodes=milestone_episodes,
                    milestone_max_steps=milestone_max_steps,
                    milestone_video_length=milestone_video_length,
                    milestone_render_fps=milestone_render_fps,
                    milestone_max_bytes=max_bytes,
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
                        saved_models.append(str(model_path))
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
                telegram.send_or_edit(f"training error: {e}")
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
            report_max_steps_default = min(5000, max(1, int(job.total_timesteps)))
            report_max_steps = int(os.environ.get("PACMAN_RL_TG_REPORT_MAX_STEPS", str(report_max_steps_default)))
            report_video_length_default = min(1500, max(1, int(report_max_steps)))
            report_video_length = int(os.environ.get("PACMAN_RL_TG_REPORT_VIDEO_LENGTH", str(report_video_length_default)))
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

            telegram.send_or_edit("training finished, sending artifacts...")

            sent_videos = 0
            skipped_videos = 0
            failed_videos = 0

            videos = sorted(glob.glob(os.path.join(out_dir, "videos", "**", "*.mp4"), recursive=True))
            for vp in videos:
                try:
                    size = os.path.getsize(vp)
                except Exception:
                    size = 0
                if size > max_bytes:
                    skipped_videos += 1
                    continue
                try:
                    parent = os.path.basename(os.path.dirname(vp))
                except Exception:
                    parent = ""
                label = parent if parent else "video"
                caption = f"{label}: {os.path.basename(vp)}"
                if telegram.send_video(vp, caption=caption):
                    sent_videos += 1
                else:
                    failed_videos += 1

            sent_models = 0
            skipped_models = 0
            failed_models = 0
            for mp in saved_models:
                try:
                    size = os.path.getsize(mp)
                except Exception:
                    size = 0
                if size <= 0:
                    failed_models += 1
                    continue
                if size > max_bytes:
                    skipped_models += 1
                    continue
                if telegram.send_document(mp, caption=os.path.basename(mp)):
                    sent_models += 1
                else:
                    failed_models += 1

            db_sent = False
            try:
                if os.path.getsize(job.db_path) <= max_bytes:
                    db_sent = telegram.send_document(job.db_path, caption=os.path.basename(job.db_path))
            except Exception:
                db_sent = False

            telegram.send_or_edit(
                f"done\nvideos: sent={sent_videos}, skipped(>limit)={skipped_videos}, failed={failed_videos}\nmodels: sent={sent_models}, skipped(>limit)={skipped_models}, failed={failed_models}\ndb: {'sent' if db_sent else 'not sent'}"
            )
        except Exception as e:
            logger.error("Telegram finalization failed: %s", e)
            try:
                telegram.send_or_edit(f"artifact sending error: {e}")
            except Exception:
                pass
