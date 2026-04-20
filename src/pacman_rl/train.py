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
from pacman_rl.layouts import group_layouts_by_size, load_layouts_from_dir
from pacman_rl.models import CNNActorCritic
from pacman_rl.rl import RolloutBatch, compute_gae, ppo_update
from pacman_rl.rl.snapshot_pool import SnapshotPool
from pacman_rl.telemetry import GameRecordConfig, TelemetryBuffer, record_game, write_telemetry_xlsx
from pacman_rl.telemetry.gif import render_game_gif
from pacman_rl.telemetry.plot_png import render_rewards_png
from pacman_rl.telemetry.telegram import send_document, send_message, telegram_target_from_env
from pacman_rl.utils import load_checkpoint, load_dotenv, resolve_device, save_checkpoint


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _report_caption(*, update: int, rows: list[dict[str, Any]], elapsed_s: float) -> str:
    pac = _mean([float(r.get("pacman_reward_mean", 0.0)) for r in rows])
    ghosts = _mean([float(r.get("ghosts_reward_mean", 0.0)) for r in rows])
    return (
        "pacman-rl"
        + " update="
        + str(update)
        + " mean_reward(pac="
        + f"{pac:.3f}"
        + " ghosts="
        + f"{ghosts:.3f}"
        + ") elapsed="
        + _format_duration(elapsed_s)
    )


def _final_summary_text(
    *,
    start_update: int,
    total_updates: int,
    rows: list[dict[str, Any]],
    elapsed_s: float,
    batch_size: int,
    rollout_steps: int,
) -> str:
    pac_values = [float(r.get("pacman_reward_mean", 0.0)) for r in rows]
    ghost_values = [float(r.get("ghosts_reward_mean", 0.0)) for r in rows]
    pac_last = pac_values[-1] if pac_values else 0.0
    ghost_last = ghost_values[-1] if ghost_values else 0.0
    pac_best = max(pac_values) if pac_values else 0.0
    ghost_best = max(ghost_values) if ghost_values else 0.0

    updates_done = max(0, total_updates - start_update)
    total_env_steps = updates_done * rollout_steps * batch_size

    return (
        "Обучение завершено\n"
        + "updates="
        + str(total_updates)
        + " (start="
        + str(start_update)
        + ")\n"
        + "elapsed="
        + _format_duration(elapsed_s)
        + "\n"
        + "env_steps≈"
        + str(total_env_steps)
        + "\n"
        + "last_reward(pac="
        + f"{pac_last:.3f}"
        + " ghosts="
        + f"{ghost_last:.3f}"
        + ")\n"
        + "best_reward(pac="
        + f"{pac_best:.3f}"
        + " ghosts="
        + f"{ghost_best:.3f}"
        + ")"
    )


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
    send_document(target=target, file_path=file_path, caption=caption)
    return True


def _choose_layout_group(layout_dir: Path) -> list:
    layouts = load_layouts_from_dir(layout_dir)
    groups = group_layouts_by_size(layouts)

    sizes = sorted(groups.keys(), key=lambda x: (x[0] * x[1], x[0], x[1]))
    chosen_size = sizes[-1]
    chosen = groups[chosen_size]

    print(f"Using {len(chosen)} layouts with size H={chosen_size[0]} W={chosen_size[1]}")
    return chosen


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
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--bot-token-env", type=str, default="BOT_TOKEN")
    parser.add_argument("--chat-id-env", type=str, default="CHAT_ID")
    parser.add_argument("--record-max-steps", type=int, default=512)
    parser.add_argument("--record-idle-steps", type=int, default=120)

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
    )

    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("✅ Используем GPU 🚀")
        else:
            device = torch.device("cpu")
            print("⚠️ GPU не найден, используем CPU 🧠")
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

    telegram_target = None
    if cfg.telegram:
        try:
            telegram_target = telegram_target_from_env(bot_token_env=cfg.bot_token_env, chat_id_env=cfg.chat_id_env)
        except Exception as e:
            print("telegram_init_failed=" + str(e))
            telegram_target = None

    train_start_s = time.time()
    if telegram_target is not None:
        try:
            send_message(
                target=telegram_target,
                text=(
                    "Обучение началось\n"
                    + "run_dir="
                    + str(cfg.run_dir)
                    + "\n"
                    + "device="
                    + str(device)
                    + "\n"
                    + "batch_size="
                    + str(cfg.batch_size)
                    + " updates="
                    + str(cfg.updates)
                    + " start_update="
                    + str(start_update)
                    + "\n"
                    + "layouts="
                    + str(len(chosen_layouts))
                ),
            )
        except Exception as e:
            print("telegram_send_failed=" + str(e))

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

        pac_batch, pac_rew = _collect_rollout_for_pacman(env, pacman, ghosts_opponent, ppo_cfg)
        pac_stats = ppo_update(pacman, pacman_opt, pac_batch, ppo_cfg)

        use_old_pac = (len(pacman_pool) > 0) and (torch.rand((), generator=rng).item() < sp_cfg.opponent_sample_prob)
        if use_old_pac:
            pacman_pool.sample_into(pacman_opponent, rng=rng)
        else:
            _copy_model(pacman_opponent, pacman)

        ghost_batch, ghost_rew = _collect_rollout_for_ghosts(env, pacman_opponent, ghosts, ppo_cfg)
        ghost_stats = ppo_update(ghosts, ghosts_opt, ghost_batch, ppo_cfg)

        update_time_s = time.time() - update_start_s
        elapsed_s = time.time() - train_start_s
        telemetry.add(
            {
                "update": update + 1,
                "pacman_reward_mean": float(pac_rew),
                "ghosts_reward_mean": float(ghost_rew),
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

        if cfg.report_every > 0 and (update + 1) % cfg.report_every == 0:
            ckpt_path = cfg.run_dir / "checkpoints" / f"update_{update + 1}.pt"
            save_checkpoint(
                ckpt_path,
                update=update + 1,
                pacman_model=pacman,
                pacman_opt=pacman_opt,
                ghosts_model=ghosts,
                ghosts_opt=ghosts_opt,
            )

            report_dir = cfg.run_dir / "reports" / f"update_{update + 1}"
            xlsx_path = report_dir / "telemetry.xlsx"
            game_path = report_dir / "game.json"
            gif_path = report_dir / "game.gif"
            plot_path = report_dir / "rewards.png"

            rows = telemetry.to_rows()
            write_telemetry_xlsx(xlsx_path, rows=rows)
            try:
                render_rewards_png(plot_path, rows=rows)
            except Exception as e:
                print("plot_render_failed=" + str(e))
            record_game(
                game_path,
                layout=chosen_layouts[0],
                pacman=pacman,
                ghosts=ghosts,
                device=device,
                env_cfg=env_cfg,
                cfg=GameRecordConfig(max_steps=cfg.record_max_steps, idle_steps=cfg.record_idle_steps),
            )
            try:
                render_game_gif(game_path, gif_path)
            except Exception as e:
                print("gif_render_failed=" + str(e))

            if telegram_target is not None:
                try:
                    window = rows[-cfg.report_every :] if cfg.report_every > 0 else rows
                    caption = _report_caption(update=update + 1, rows=window, elapsed_s=time.time() - train_start_s)
                    _maybe_send_file(target=telegram_target, file_path=gif_path, caption=caption)
                    _maybe_send_file(target=telegram_target, file_path=plot_path, caption=caption)
                    _maybe_send_file(target=telegram_target, file_path=xlsx_path, caption=caption)
                    _maybe_send_file(target=telegram_target, file_path=game_path, caption=caption)
                    _maybe_send_file(target=telegram_target, file_path=ckpt_path, caption=caption)
                except Exception as e:
                    print("telegram_send_failed=" + str(e))

    save_checkpoint(
        cfg.run_dir / "checkpoints" / f"final_update_{cfg.updates}.pt",
        update=cfg.updates,
        pacman_model=pacman,
        pacman_opt=pacman_opt,
        ghosts_model=ghosts,
        ghosts_opt=ghosts_opt,
    )
    if telegram_target is not None:
        try:
            send_message(
                target=telegram_target,
                text=_final_summary_text(
                    start_update=start_update,
                    total_updates=cfg.updates,
                    rows=telemetry.to_rows(),
                    elapsed_s=time.time() - train_start_s,
                    batch_size=cfg.batch_size,
                    rollout_steps=ppo_cfg.rollout_steps,
                ),
            )
        except Exception as e:
            print("telegram_send_failed=" + str(e))


if __name__ == "__main__":
    main()
