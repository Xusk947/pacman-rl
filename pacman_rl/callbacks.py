from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from pacman_rl.db import EpisodeRow, SQLiteLogger, TrainingRow, now_timestamptz
from pacman_rl.metrics import PelletTotalEstimator, parse_pacman_reward_events

logger = logging.getLogger(__name__)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as e:  # pragma: no cover
    BaseCallback = object  # type: ignore[assignment]
    _sb3_import_error = e
else:  # pragma: no cover
    _sb3_import_error = None


class PacmanSQLiteCallback(BaseCallback):
    def __init__(
        self,
        *,
        db: SQLiteLogger,
        run_id: str,
        algo: str,
        model_index: int,
        model_total: int,
        total_timesteps: int,
        win_score_threshold: float,
        log_every_steps: int,
        estimate_total_pellets: bool,
        print_every_percent: int,
        stats_window_episodes: int,
        telegram: Any | None,
    ) -> None:
        if _sb3_import_error is not None:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is required") from _sb3_import_error

        super().__init__(verbose=0)
        self._db = db
        self._run_id = run_id
        self._algo = str(algo)
        self._model_index = int(model_index)
        self._model_total = int(model_total)
        self._total_timesteps = int(total_timesteps)
        self._win_score_threshold = float(win_score_threshold)
        self._log_every_steps = int(log_every_steps)
        self._estimate_total_pellets_enabled = bool(estimate_total_pellets)
        self._print_every_percent = max(1, int(print_every_percent))
        self._stats_window_episodes = max(1, int(stats_window_episodes))
        self._telegram = telegram
        self._estimator = PelletTotalEstimator()

        self._n_envs = 0
        self._episode_index: np.ndarray | None = None
        self._pellets: np.ndarray | None = None
        self._power_pellets: np.ndarray | None = None
        self._ghosts: np.ndarray | None = None
        self._total_pellets: np.ndarray | None = None
        self._last_training_log = 0
        self._last_print_percent = -1
        self._recent_returns: list[float] = []
        self._recent_wins: list[int] = []
        self._recent_pellets: list[int] = []
        self._recent_power_pellets: list[int] = []
        self._recent_ghosts: list[int] = []

    def _on_training_start(self) -> None:
        self._n_envs = int(self.training_env.num_envs)
        self._episode_index = np.zeros((self._n_envs,), dtype=np.int64)
        self._pellets = np.zeros((self._n_envs,), dtype=np.int64)
        self._power_pellets = np.zeros((self._n_envs,), dtype=np.int64)
        self._ghosts = np.zeros((self._n_envs,), dtype=np.int64)
        self._total_pellets = np.zeros((self._n_envs,), dtype=np.int64)
        self._last_training_log = 0
        if self._estimate_total_pellets_enabled:
            for i in range(self._n_envs):
                self._total_pellets[i] = self._estimate_total_pellets(i)

        self._maybe_send_telegram(0, "")

    def _estimate_total_pellets(self, env_index: int) -> int:
        try:
            env = self.training_env.envs[env_index]
            ale = env.unwrapped.ale
            frame = ale.getScreenRGB()
            return int(self._estimator.estimate_total_from_rgb(frame))
        except Exception:
            return 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if rewards is None or dones is None or infos is None:
            return True

        if self._episode_index is None or self._pellets is None or self._power_pellets is None or self._ghosts is None or self._total_pellets is None:
            return True

        rewards_arr = np.array(rewards).reshape((-1,))
        dones_arr = np.array(dones).reshape((-1,))

        for i in range(self._n_envs):
            ev = parse_pacman_reward_events(float(rewards_arr[i]))
            self._pellets[i] += ev.pellets
            self._power_pellets[i] += ev.power_pellets
            self._ghosts[i] += ev.ghosts

            if not bool(dones_arr[i]):
                continue

            info = infos[i] if i < len(infos) else {}
            ep = info.get("episode") if isinstance(info, dict) else None
            if not isinstance(ep, dict):
                episode_return = float("nan")
                episode_length = -1
            else:
                episode_return = float(ep.get("r", float("nan")))
                episode_length = int(ep.get("l", -1))

            total = int(self._total_pellets[i])
            eaten = int(self._pellets[i] + self._power_pellets[i])
            percent_cleared = (100.0 * eaten / total) if total > 0 else None
            win = 1 if episode_return >= self._win_score_threshold else 0

            self._db.log_episode(
                EpisodeRow(
                    run_id=self._run_id,
                    env_index=i,
                    episode_index=int(self._episode_index[i]),
                    episode_return=float(episode_return),
                    episode_length=int(episode_length),
                    win=int(win),
                    pellets=int(self._pellets[i]),
                    power_pellets=int(self._power_pellets[i]),
                    ghosts=int(self._ghosts[i]),
                    percent_cleared=percent_cleared,
                    ended_at=now_timestamptz(),
                )
            )
            self._db.commit()

            self._append_recent(
                episode_return=float(episode_return),
                win=int(win),
                pellets=int(self._pellets[i]),
                power_pellets=int(self._power_pellets[i]),
                ghosts=int(self._ghosts[i]),
            )

            self._episode_index[i] += 1
            self._pellets[i] = 0
            self._power_pellets[i] = 0
            self._ghosts[i] = 0
            if self._estimate_total_pellets_enabled and self._total_pellets[i] <= 0:
                self._total_pellets[i] = self._estimate_total_pellets(i)

        self._maybe_print_progress()

        if self._log_every_steps > 0 and (self.num_timesteps - self._last_training_log) >= self._log_every_steps:
            self._last_training_log = int(self.num_timesteps)
            metrics: dict[str, Any] = {}
            logger = getattr(self.model, "logger", None)
            name_to_value = getattr(logger, "name_to_value", None)
            if isinstance(name_to_value, dict):
                for k, v in name_to_value.items():
                    fv = _safe_float(v)
                    if fv is not None:
                        metrics[str(k)] = fv

            self._db.log_training(
                TrainingRow(
                    run_id=self._run_id,
                    timestep=int(self.num_timesteps),
                    metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    logged_at=now_timestamptz(),
                )
            )
            self._db.commit()

        return True

    def _append_recent(self, *, episode_return: float, win: int, pellets: int, power_pellets: int, ghosts: int) -> None:
        self._recent_returns.append(float(episode_return))
        self._recent_wins.append(int(win))
        self._recent_pellets.append(int(pellets))
        self._recent_power_pellets.append(int(power_pellets))
        self._recent_ghosts.append(int(ghosts))

        max_len = self._stats_window_episodes
        if len(self._recent_returns) > max_len:
            del self._recent_returns[: len(self._recent_returns) - max_len]
            del self._recent_wins[: len(self._recent_wins) - max_len]
            del self._recent_pellets[: len(self._recent_pellets) - max_len]
            del self._recent_power_pellets[: len(self._recent_power_pellets) - max_len]
            del self._recent_ghosts[: len(self._recent_ghosts) - max_len]

    def _maybe_print_progress(self) -> None:
        if self._total_timesteps <= 0:
            return

        percent = int(100 * min(self.num_timesteps, self._total_timesteps) / self._total_timesteps)
        if percent <= self._last_print_percent:
            return
        if percent % self._print_every_percent != 0 and percent != 100:
            return

        self._last_print_percent = percent

        metrics_out: dict[str, Any] = {
            "progress/percent": float(percent),
            "progress/steps": float(self.num_timesteps),
            "progress/total_timesteps": float(self._total_timesteps),
        }

        if self._recent_returns:
            rets = np.array(self._recent_returns, dtype=np.float64)
            wins = np.array(self._recent_wins, dtype=np.float64)
            pellets = np.array(self._recent_pellets, dtype=np.float64)
            pp = np.array(self._recent_power_pellets, dtype=np.float64)
            ghosts = np.array(self._recent_ghosts, dtype=np.float64)

            metrics_out |= {
                "window/episodes": float(len(rets)),
                "window/winrate": float(wins.mean()),
                "window/return_mean": float(rets.mean()),
                "window/return_std": float(rets.std(ddof=0)),
                "window/pellets_mean": float(pellets.mean()),
                "window/power_pellets_mean": float(pp.mean()),
                "window/ghosts_mean": float(ghosts.mean()),
            }

            ep_buf = getattr(self.model, "ep_info_buffer", None)
            if ep_buf:
                try:
                    ep_rets = np.array([float(d.get("r")) for d in ep_buf if isinstance(d, dict) and "r" in d], dtype=np.float64)
                    ep_lens = np.array([float(d.get("l")) for d in ep_buf if isinstance(d, dict) and "l" in d], dtype=np.float64)
                    if ep_rets.size > 0:
                        metrics_out["rollout/ep_rew_mean"] = float(ep_rets.mean())
                    if ep_lens.size > 0:
                        metrics_out["rollout/ep_len_mean"] = float(ep_lens.mean())
                except Exception:
                    pass

            msg = (
                f"[{self._algo}] {percent:3d}%  steps={self.num_timesteps}/{self._total_timesteps}  "
                f"ep_window={len(rets)}  winrate={wins.mean():.3f}  "
                f"ret_mean={rets.mean():.1f}±{rets.std(ddof=0):.1f}  "
                f"pel={pellets.mean():.1f}  big={pp.mean():.2f}  ghosts={ghosts.mean():.2f}"
            )
        else:
            msg = f"[{self._algo}] {percent:3d}%  steps={self.num_timesteps}/{self._total_timesteps}"

        print(msg, flush=True)
        self._maybe_send_telegram(percent, msg)

        self._db.log_training(
            TrainingRow(
                run_id=self._run_id,
                timestep=int(self.num_timesteps),
                metrics_json=json.dumps(metrics_out, ensure_ascii=False, sort_keys=True),
                logged_at=now_timestamptz(),
            )
        )
        self._db.commit()

    def _maybe_send_telegram(self, percent: int, console_line: str) -> None:
        if not self._telegram:
            return
        try:
            extra = ""
            if self._recent_returns:
                rets = np.array(self._recent_returns, dtype=np.float64)
                wins = np.array(self._recent_wins, dtype=np.float64)
                extra = f"winrate={wins.mean():.3f}  ret_mean={rets.mean():.1f}±{rets.std(ddof=0):.1f}"

            steps = int(self.num_timesteps)
            total = int(self._total_timesteps)
            text = self._telegram.format_progress(
                self._algo,
                steps,
                total,
                model_index=self._model_index,
                model_total=self._model_total,
                extra=extra,
            )
            self._telegram.send_or_edit(text)
        except Exception as e:
            logger.error("Telegram update failed: %s", e)
