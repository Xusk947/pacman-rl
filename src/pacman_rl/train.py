from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from pacman_rl.config import EnvConfig, PPOConfig, SelfPlayConfig
from pacman_rl.env import TorchPacmanEnv
from pacman_rl.layouts import load_layouts_from_dir
from pacman_rl.models import CNNActorCritic
from pacman_rl.rl import RolloutBatch, compute_gae, ppo_update
from pacman_rl.rl.snapshot_pool import SnapshotPool
from pacman_rl.telemetry import GameRecordConfig, TelemetryBuffer, record_game, write_telemetry_xlsx
from pacman_rl.telemetry.gif import render_game_gif
from pacman_rl.telemetry.telegram import (
    TelegramRateLimitError,
    send_document,
    send_media_group,
    send_message,
    telegram_target_from_env,
)
from pacman_rl.utils import load_checkpoint, load_dotenv, resolve_device, save_checkpoint, save_model_weights
from pacman_rl.utils.device import cuda_compatibility


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    updates: int
    device: str
    layout_dir: Path
    run_dir: Path
    resume: Path | None
    report_every: int
    telegram: bool
    bot_token_env: str
    chat_id_env: str
    record_max_steps: int
    record_idle_steps: int
    telegram_status_every_episodes: int
    telegram_sleep_s: float
    telegram_send_recordings: bool


@dataclass
class EpisodeStats:
    episodes: int = 0
    pellets: int = 0
    power: int = 0
    ghosts: int = 0
    steps: int = 0
    deaths: int = 0
    wins: int = 0
    timeouts: int = 0


class EpisodeAccumulator:
    def __init__(self, *, batch_size: int, device: torch.device) -> None:
        self._pellets = torch.zeros((batch_size,), device=device, dtype=torch.int64)
        self._power = torch.zeros((batch_size,), device=device, dtype=torch.int64)
        self._ghosts = torch.zeros((batch_size,), device=device, dtype=torch.int64)
        self._steps = torch.zeros((batch_size,), device=device, dtype=torch.int64)

        self.totals = EpisodeStats()

    def observe(self, out: Any) -> None:
        self._pellets += out.pellet_eaten.to(torch.int64)
        self._power += out.power_eaten.to(torch.int64)
        self._ghosts += out.ghosts_eaten.to(torch.int64)
        self._steps += 1

        done = out.done
        if not bool(done.any().item()):
            return

        idx = done.nonzero(as_tuple=False).squeeze(-1)
        self.totals.episodes += int(idx.shape[0])
        self.totals.pellets += int(self._pellets[idx].sum().item())
        self.totals.power += int(self._power[idx].sum().item())
        self.totals.ghosts += int(self._ghosts[idx].sum().item())
        self.totals.steps += int(self._steps[idx].sum().item())
        self.totals.deaths += int(out.pac_dead[idx].to(torch.int64).sum().item())
        self.totals.wins += int(out.all_pellets_done[idx].to(torch.int64).sum().item())
        self.totals.timeouts += int(out.timeout[idx].to(torch.int64).sum().item())

        self._pellets[idx] = 0
        self._power[idx] = 0
        self._ghosts[idx] = 0
        self._steps[idx] = 0


def _episode_means(s: EpisodeStats) -> dict[str, float]:
    if s.episodes <= 0:
        return {
            "episodes_finished": 0.0,
            "pellets_eaten_per_episode": 0.0,
            "power_eaten_per_episode": 0.0,
            "ghosts_eaten_per_episode": 0.0,
            "steps_per_episode": 0.0,
            "death_rate": 0.0,
            "win_rate": 0.0,
            "timeout_rate": 0.0,
        }

    e = float(s.episodes)
    return {
        "episodes_finished": float(s.episodes),
        "pellets_eaten_per_episode": float(s.pellets) / e,
        "power_eaten_per_episode": float(s.power) / e,
        "ghosts_eaten_per_episode": float(s.ghosts) / e,
        "steps_per_episode": float(s.steps) / e,
        "death_rate": float(s.deaths) / e,
        "win_rate": float(s.wins) / e,
        "timeout_rate": float(s.timeouts) / e,
    }


def _maybe_send_file(
    *,
    target: Any,
    file_path: Path,
    caption: str,
    max_mb: int = 45,
) -> bool:
    if not file_path.exists():
        return False
    size = file_path.stat().st_size
    if size > max_mb * 1024 * 1024:
        try:
            send_message(
                target=target,
                text=f"Skip file (too large): {file_path.name} size={size / (1024 * 1024):.1f}MB",
            )
        except Exception:
            pass
        return False
    for _ in range(3):
        try:
            send_document(target=target, file_path=file_path, caption=caption)
            return True
        except TelegramRateLimitError as e:
            time.sleep(float(e.retry_after_s) + 0.5)
    send_document(target=target, file_path=file_path, caption=caption)
    return True


def _choose_layout_group(layout_dir: Path) -> list:
    layouts = load_layouts_from_dir(layout_dir)
    h = max(l.height for l in layouts)
    w = max(l.width for l in layouts)
    print(f"Using {len(layouts)} layouts (max size H={h} W={w})")
    return layouts


def _build_env(layouts: list, *, batch_size: int, device: torch.device, env_cfg: EnvConfig) -> TorchPacmanEnv:
    return TorchPacmanEnv(layouts, batch_size=batch_size, device=device, cfg=env_cfg)


def _sample_actions(model: CNNActorCritic, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(obs)
    dist = Categorical(logits=out.logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action, logprob, out.value


def _copy_model(dst: CNNActorCritic, src: CNNActorCritic) -> None:
    dst.load_state_dict(src.state_dict())


def _collect_rollout_for_pacman(
    env: TorchPacmanEnv,
    pacman: CNNActorCritic,
    ghosts_opponent: CNNActorCritic,
    ppo_cfg: PPOConfig,
    acc: EpisodeAccumulator | None = None,
) -> tuple[RolloutBatch, torch.Tensor]:
    b = env.batch_size
    t = ppo_cfg.rollout_steps
    device = env.device

    pac_obs, ghost_obs = env.get_obs()

    obs_buf = torch.zeros((t, b) + tuple(pac_obs.shape[1:]), device=device, dtype=torch.float32)
    act_buf = torch.zeros((t, b), device=device, dtype=torch.int64)
    logp_buf = torch.zeros((t, b), device=device, dtype=torch.float32)
    rew_buf = torch.zeros((t, b), device=device, dtype=torch.float32)
    done_buf = torch.zeros((t, b), device=device, dtype=torch.bool)
    val_buf = torch.zeros((t, b), device=device, dtype=torch.float32)

    for step in range(t):
        with torch.no_grad():
            pac_action, pac_logp, pac_val = _sample_actions(pacman, pac_obs)

            g_flat = ghost_obs.view(b * env.GMAX, ghost_obs.shape[2], env.height, env.width)
            g_action, _, _ = _sample_actions(ghosts_opponent, g_flat)
            ghost_action = g_action.view(b, env.GMAX)

        out = env.step(pac_action, ghost_action)
        if acc is not None:
            acc.observe(out)

        obs_buf[step] = pac_obs
        act_buf[step] = pac_action
        logp_buf[step] = pac_logp
        rew_buf[step] = out.pac_reward
        done_buf[step] = out.done
        val_buf[step] = pac_val

        pac_obs, ghost_obs = out.pac_obs, out.ghost_obs
        if out.done.any():
            pac_obs, ghost_obs = env.reset_done(out.done)

    with torch.no_grad():
        last_value = pacman(pac_obs).value

    adv, ret = compute_gae(
        rew_buf,
        done_buf,
        val_buf,
        last_value,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )

    batch = RolloutBatch(
        obs=obs_buf.view(t * b, *pac_obs.shape[1:]),
        actions=act_buf.view(t * b),
        logprobs=logp_buf.view(t * b),
        advantages=adv.view(t * b),
        returns=ret.view(t * b),
        values=val_buf.view(t * b),
    )
    return batch, rew_buf.mean()


def _collect_rollout_for_ghosts(
    env: TorchPacmanEnv,
    pacman_opponent: CNNActorCritic,
    ghosts: CNNActorCritic,
    ppo_cfg: PPOConfig,
    acc: EpisodeAccumulator | None = None,
) -> tuple[RolloutBatch, torch.Tensor]:
    b = env.batch_size
    t = ppo_cfg.rollout_steps
    device = env.device

    pac_obs, ghost_obs = env.get_obs()
    present = env.ghost_present

    obs_buf = torch.zeros((t, b, env.GMAX) + tuple(ghost_obs.shape[2:]), device=device, dtype=torch.float32)
    act_buf = torch.zeros((t, b, env.GMAX), device=device, dtype=torch.int64)
    logp_buf = torch.zeros((t, b, env.GMAX), device=device, dtype=torch.float32)
    rew_buf = torch.zeros((t, b, env.GMAX), device=device, dtype=torch.float32)
    done_buf = torch.zeros((t, b), device=device, dtype=torch.bool)
    val_buf = torch.zeros((t, b, env.GMAX), device=device, dtype=torch.float32)

    for step in range(t):
        with torch.no_grad():
            pac_action, _, _ = _sample_actions(pacman_opponent, pac_obs)

            g_flat = ghost_obs.view(b * env.GMAX, ghost_obs.shape[2], env.height, env.width)
            g_action, g_logp, g_val = _sample_actions(ghosts, g_flat)
            ghost_action = g_action.view(b, env.GMAX)
            ghost_logp = g_logp.view(b, env.GMAX)
            ghost_val = g_val.view(b, env.GMAX)

            ghost_action = torch.where(present, ghost_action, torch.zeros_like(ghost_action))
            ghost_logp = torch.where(present, ghost_logp, torch.zeros_like(ghost_logp))
            ghost_val = torch.where(present, ghost_val, torch.zeros_like(ghost_val))

        out = env.step(pac_action, ghost_action)
        if acc is not None:
            acc.observe(out)

        obs_buf[step] = ghost_obs
        act_buf[step] = ghost_action
        logp_buf[step] = ghost_logp
        rew_buf[step] = out.ghost_reward
        done_buf[step] = out.done
        val_buf[step] = ghost_val

        pac_obs, ghost_obs = out.pac_obs, out.ghost_obs
        if out.done.any():
            pac_obs, ghost_obs = env.reset_done(out.done)
            present = env.ghost_present

    with torch.no_grad():
        g_flat = ghost_obs.view(b * env.GMAX, ghost_obs.shape[2], env.height, env.width)
        last_value = ghosts(g_flat).value.view(b, env.GMAX)
        last_value = torch.where(present, last_value, torch.zeros_like(last_value))

    dones_exp = done_buf[:, :, None].repeat(1, 1, env.GMAX)
    adv, ret = compute_gae(
        rew_buf,
        dones_exp,
        val_buf,
        last_value,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )

    mask = present[None].repeat(t, 1, 1).view(t * b * env.GMAX)

    obs_flat = obs_buf.view(t * b * env.GMAX, ghost_obs.shape[2], env.height, env.width)
    act_flat = act_buf.view(t * b * env.GMAX)
    logp_flat = logp_buf.view(t * b * env.GMAX)
    adv_flat = adv.view(t * b * env.GMAX)
    ret_flat = ret.view(t * b * env.GMAX)
    val_flat = val_buf.view(t * b * env.GMAX)

    obs_flat = obs_flat[mask]
    act_flat = act_flat[mask]
    logp_flat = logp_flat[mask]
    adv_flat = adv_flat[mask]
    ret_flat = ret_flat[mask]
    val_flat = val_flat[mask]

    batch = RolloutBatch(
        obs=obs_flat,
        actions=act_flat,
        logprobs=logp_flat,
        advantages=adv_flat,
        returns=ret_flat,
        values=val_flat,
    )

    return batch, rew_buf.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-dir", type=Path, default=Path("layouts"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--bot-token-env", type=str, default="BOT_TOKEN")
    parser.add_argument("--chat-id-env", type=str, default="CHAT_ID")
    parser.add_argument("--record-max-steps", type=int, default=512)
    parser.add_argument("--record-idle-steps", type=int, default=120)
    parser.add_argument("--telegram-status-every-episodes", type=int, default=50)
    parser.add_argument("--telegram-sleep-s", type=float, default=0.5)
    parser.add_argument("--telegram-send-recordings", action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        updates=args.updates,
        device=args.device,
        layout_dir=args.layout_dir,
        run_dir=args.run_dir,
        resume=args.resume,
        report_every=args.report_every,
        telegram=bool(args.telegram),
        bot_token_env=args.bot_token_env,
        chat_id_env=args.chat_id_env,
        record_max_steps=args.record_max_steps,
        record_idle_steps=args.record_idle_steps,
        telegram_status_every_episodes=args.telegram_status_every_episodes,
        telegram_sleep_s=float(args.telegram_sleep_s),
        telegram_send_recordings=bool(args.telegram_send_recordings),
    )

    if cfg.device == "auto":
        ok, reason = cuda_compatibility()
        if ok:
            device = torch.device("cuda")
            print("✅ Using GPU 🚀")
        else:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                print("⚠️ CUDA GPU detected but incompatible with this PyTorch build; using CPU. reason=" + reason)
            else:
                print("⚠️ GPU not found, using CPU 🧠")
    else:
        device = resolve_device(cfg.device)
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    sp_cfg = SelfPlayConfig()

    chosen_layouts = _choose_layout_group(cfg.layout_dir)
    env = _build_env(chosen_layouts, batch_size=cfg.batch_size, device=device, env_cfg=env_cfg)

    pac_obs, ghost_obs = env.get_obs()
    pacman = CNNActorCritic(in_channels=pac_obs.shape[1], actions=env.ACTIONS).to(device)
    ghosts = CNNActorCritic(in_channels=ghost_obs.shape[2], actions=env.ACTIONS).to(device)

    pacman_opt = torch.optim.Adam(pacman.parameters(), lr=ppo_cfg.learning_rate)
    ghosts_opt = torch.optim.Adam(ghosts.parameters(), lr=ppo_cfg.learning_rate)

    start_update = 0
    if cfg.resume is not None and cfg.resume.exists():
        start_update = load_checkpoint(
            cfg.resume,
            pacman_model=pacman,
            pacman_opt=pacman_opt,
            ghosts_model=ghosts,
            ghosts_opt=ghosts_opt,
            map_location=device,
        )
        print(f"Resumed from {cfg.resume}, update={start_update}")

    pacman_opponent = CNNActorCritic(in_channels=pac_obs.shape[1], actions=env.ACTIONS).to(device)
    ghosts_opponent = CNNActorCritic(in_channels=ghost_obs.shape[2], actions=env.ACTIONS).to(device)

    pacman_pool = SnapshotPool(capacity=sp_cfg.opponent_pool_size)
    ghosts_pool = SnapshotPool(capacity=sp_cfg.opponent_pool_size)

    rng = torch.Generator(device="cpu")
    telemetry = TelemetryBuffer()
    episode_acc = EpisodeAccumulator(batch_size=cfg.batch_size, device=device)

    telegram_target = None
    if cfg.telegram:
        try:
            telegram_target = telegram_target_from_env(bot_token_env=cfg.bot_token_env, chat_id_env=cfg.chat_id_env)
        except Exception as e:
            print("telegram_init_failed=" + str(e))
            telegram_target = None

    train_start_s = time.time()
    total_episodes = 0
    last_status_sent = 0
    last_report_sent = 0

    for update in range(start_update, cfg.updates):
        update_start_s = time.time()
        if update % sp_cfg.snapshot_every_updates == 0 and update != start_update:
            pacman_pool.add(pacman)
            ghosts_pool.add(ghosts)

        use_old_ghosts = (len(ghosts_pool) > 0) and (torch.rand((), generator=rng).item() < sp_cfg.opponent_sample_prob)
        if use_old_ghosts:
            ghosts_pool.sample_into(ghosts_opponent, rng=rng)
        else:
            _copy_model(ghosts_opponent, ghosts)

        episode_acc.totals = EpisodeStats()
        pac_batch, pac_rew = _collect_rollout_for_pacman(env, pacman, ghosts_opponent, ppo_cfg, episode_acc)
        pac_stats = ppo_update(pacman, pacman_opt, pac_batch, ppo_cfg)

        use_old_pac = (len(pacman_pool) > 0) and (torch.rand((), generator=rng).item() < sp_cfg.opponent_sample_prob)
        if use_old_pac:
            pacman_pool.sample_into(pacman_opponent, rng=rng)
        else:
            _copy_model(pacman_opponent, pacman)

        ghost_batch, ghost_rew = _collect_rollout_for_ghosts(env, pacman_opponent, ghosts, ppo_cfg, episode_acc)
        ghost_stats = ppo_update(ghosts, ghosts_opt, ghost_batch, ppo_cfg)

        update_time_s = time.time() - update_start_s
        elapsed_s = time.time() - train_start_s
        steps_per_update = cfg.batch_size * ppo_cfg.rollout_steps * 2
        step = int((update + 1) * steps_per_update)
        ep = _episode_means(episode_acc.totals)
        total_episodes += int(episode_acc.totals.episodes)
        telemetry.add(
            {
                "update": update + 1,
                "step": step,
                "pacman_reward_mean": float(pac_rew),
                "ghosts_reward_mean": float(ghost_rew),
                "episodes_finished": ep["episodes_finished"],
                "pellets_eaten_per_episode": ep["pellets_eaten_per_episode"],
                "power_eaten_per_episode": ep["power_eaten_per_episode"],
                "ghosts_eaten_per_episode": ep["ghosts_eaten_per_episode"],
                "steps_per_episode": ep["steps_per_episode"],
                "win_rate": ep["win_rate"],
                "death_rate": ep["death_rate"],
                "timeout_rate": ep["timeout_rate"],
                "update_time_s": float(update_time_s),
                "elapsed_s": float(elapsed_s),
                "pacman_loss": pac_stats.loss,
                "pacman_policy_loss": pac_stats.policy_loss,
                "pacman_value_loss": pac_stats.value_loss,
                "pacman_entropy": pac_stats.entropy,
                "pacman_approx_kl": pac_stats.approx_kl,
                "ghosts_loss": ghost_stats.loss,
                "ghosts_policy_loss": ghost_stats.policy_loss,
                "ghosts_value_loss": ghost_stats.value_loss,
                "ghosts_entropy": ghost_stats.entropy,
                "ghosts_approx_kl": ghost_stats.approx_kl,
            }
        )

        if telegram_target is not None and cfg.telegram_status_every_episodes > 0:
            status_to_send = (total_episodes // cfg.telegram_status_every_episodes) * cfg.telegram_status_every_episodes
            if status_to_send > last_status_sent:
                try:
                    send_message(target=telegram_target, text=f"episodes={status_to_send}")
                except Exception as e:
                    print("telegram_send_failed=" + str(e))
                last_status_sent = status_to_send

        if (update + 1) % 10 == 0:
            print(
                "update="
                + str(update + 1)
                + " pacman_reward="
                + f"{float(pac_rew):.3f}"
                + " ghosts_reward="
                + f"{float(ghost_rew):.3f}"
                + " pac_kl="
                + f"{pac_stats.approx_kl:.6f}"
                + " ghost_kl="
                + f"{ghost_stats.approx_kl:.6f}"
            )

        if telegram_target is not None and cfg.report_every > 0:
            report_to_send = (total_episodes // cfg.report_every) * cfg.report_every
            if report_to_send > last_report_sent:
                report_episodes = report_to_send
                last_report_sent = report_to_send

                ckpt_path = cfg.run_dir / "checkpoints" / f"episodes_{report_episodes}.pt"
                save_checkpoint(
                    ckpt_path,
                    update=update + 1,
                    pacman_model=pacman,
                    pacman_opt=pacman_opt,
                    ghosts_model=ghosts,
                    ghosts_opt=ghosts_opt,
                )

                report_dir = cfg.run_dir / "reports" / f"episodes_{report_episodes}"
                xlsx_path = report_dir / f"telemetry_{report_episodes}.xlsx"
                weights_path = report_dir / f"weights_{report_episodes}.pt"

                rows = telemetry.to_rows()
                write_telemetry_xlsx(xlsx_path, rows=rows)
                try:
                    save_model_weights(weights_path, update=update + 1, pacman_model=pacman, ghosts_model=ghosts)
                except Exception as e:
                    print("weights_save_failed=" + str(e))

                if cfg.telegram_send_recordings:
                    demo_cfg = GameRecordConfig(max_steps=cfg.record_max_steps, idle_steps=cfg.record_idle_steps)
                    gif_paths: list[Path] = []
                    for lay in chosen_layouts:
                        game_path = report_dir / f"game_{lay.name}_{report_episodes}.json"
                        gif_path = report_dir / f"game_{lay.name}_{report_episodes}.gif"
                        try:
                            record_game(
                                game_path,
                                layout=lay,
                                pacman=pacman,
                                ghosts=ghosts,
                                device=device,
                                env_cfg=env_cfg,
                                cfg=demo_cfg,
                            )
                            render_game_gif(game_path, gif_path)
                            if gif_path.exists():
                                gif_paths.append(gif_path)
                        except Exception as e:
                            print("demo_send_failed=" + str(e))
                    if gif_paths:
                        for i in range(0, len(gif_paths), 10):
                            batch = gif_paths[i : i + 10]
                            for _ in range(3):
                                try:
                                    send_media_group(target=telegram_target, file_paths=batch)
                                    break
                                except TelegramRateLimitError as e:
                                    time.sleep(float(e.retry_after_s) + 0.5)
                            time.sleep(cfg.telegram_sleep_s)

                _maybe_send_file(target=telegram_target, file_path=xlsx_path, caption="")
                time.sleep(cfg.telegram_sleep_s)
                _maybe_send_file(target=telegram_target, file_path=weights_path, caption="")
                time.sleep(cfg.telegram_sleep_s)

    save_checkpoint(
        cfg.run_dir / "checkpoints" / f"final_update_{cfg.updates}.pt",
        update=cfg.updates,
        pacman_model=pacman,
        pacman_opt=pacman_opt,
        ghosts_model=ghosts,
        ghosts_opt=ghosts_opt,
    )


if __name__ == "__main__":
    main()
