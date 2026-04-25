from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributions import Categorical

from pacman_rl.config import EnvConfig, LogConfig, PPOConfig
from pacman_rl.env import TorchPacmanEnv
from pacman_rl.layouts import load_layouts_from_dir
from pacman_rl.models import SharedCNNActorCritic
from pacman_rl.rl import compute_gae, ppo_update
from pacman_rl.telemetry import SqliteLogger, TelegramReporter, telegram_target_from_env
from pacman_rl.telemetry.telegram_api import TelegramTarget
from pacman_rl.telemetry.sqlite_logger import MetricsRow
from pacman_rl.utils import resolve_device


@dataclass(frozen=True)
class TrainConfig:
    layout_dir: Path
    run_dir: Path
    device: str
    batch_size: int
    total_steps: int
    seed: int
    telegram: bool
    telegram_dry_run: bool


def _sample(model: SharedCNNActorCritic, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(obs_flat)
    dist = Categorical(logits=out.logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    return action, logp, out.value


def _collect_rollout(
    *,
    env: TorchPacmanEnv,
    model: SharedCNNActorCritic,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = int(cfg.rollout_steps)
    b = env.batch_size
    a = env.AGENTS
    c = 13
    h = env.height
    w = env.width

    obs_buf = torch.zeros((t, b, a, c, h, w), device=env.device, dtype=torch.float32)
    act_buf = torch.zeros((t, b, a), device=env.device, dtype=torch.int64)
    logp_buf = torch.zeros((t, b, a), device=env.device, dtype=torch.float32)
    val_buf = torch.zeros((t, b, a), device=env.device, dtype=torch.float32)
    rew_buf = torch.zeros((t, b, a), device=env.device, dtype=torch.float32)
    done_buf = torch.zeros((t, b), device=env.device, dtype=torch.bool)
    info_acc = torch.zeros((t, b, 3), device=env.device, dtype=torch.int64)

    obs = env.get_obs()
    for step in range(t):
        obs_buf[step] = obs
        obs_flat = obs.reshape(b * a, c, h, w)
        with torch.no_grad():
            action_flat, logp_flat, value_flat = _sample(model, obs_flat)
        action = action_flat.reshape(b, a)
        logp = logp_flat.reshape(b, a)
        value = value_flat.reshape(b, a)

        out = env.step(action)

        act_buf[step] = action
        logp_buf[step] = logp
        val_buf[step] = value
        rew_buf[step] = out.reward
        done_buf[step] = out.done
        info_acc[step, :, 0] = out.info["pellet_eaten"]
        info_acc[step, :, 1] = out.info["power_eaten"]
        info_acc[step, :, 2] = out.info["pac_dead"]

        obs = out.obs
        if out.done.any():
            obs = env.reset_done(out.done)

    with torch.no_grad():
        last_obs = env.get_obs()
        last_flat = last_obs.reshape(b * a, c, h, w)
        last_values = model(last_flat).value.reshape(b, a)

    return obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, last_values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout-dir", type=Path, default=Path("layouts"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--telegram-dry-run", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        layout_dir=args.layout_dir,
        run_dir=args.run_dir,
        device=args.device,
        batch_size=int(args.batch_size),
        total_steps=int(args.total_steps),
        seed=int(args.seed),
        telegram=bool(args.telegram),
        telegram_dry_run=bool(args.telegram_dry_run),
    )

    device = resolve_device(cfg.device)
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    log_cfg = LogConfig(sqlite_path=str((cfg.run_dir / "metrics.sqlite").as_posix()))

    layouts = load_layouts_from_dir(cfg.layout_dir)
    env = TorchPacmanEnv(layouts, batch_size=cfg.batch_size, device=device, cfg=env_cfg)
    env.reset(seed=cfg.seed)

    model = SharedCNNActorCritic(in_channels=13, actions=env.ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

    sqlite = SqliteLogger(db_path=Path(log_cfg.sqlite_path))

    reporter = None
    if cfg.telegram:
        if cfg.telegram_dry_run:
            target = TelegramTarget(bot_token="dry_run", chat_id="dry_run")
        else:
            target = telegram_target_from_env()
        reporter = TelegramReporter(target=target, dry_run=cfg.telegram_dry_run)

    steps_per_update = cfg.batch_size * ppo_cfg.rollout_steps * env.AGENTS
    updates = max(1, (cfg.total_steps + steps_per_update - 1) // steps_per_update)

    global_step = 0
    total_episodes = 0
    last_progress_step = -1
    last_sqlite_step = -1
    last_sqlite_sent_step = -1

    train_start = time.time()
    for update in range(int(updates)):
        upd_start = time.time()
        obs, actions, old_logp, values, rewards, dones, last_values = _collect_rollout(env=env, model=model, cfg=ppo_cfg)

        dones_exp = dones[:, :, None].expand(-1, -1, env.AGENTS)
        gae = compute_gae(
            rewards,
            dones_exp,
            values,
            last_values,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        t, b, a, c, h, w = obs.shape
        obs_flat = obs.reshape(t * b * a, c, h, w)
        act_flat = actions.reshape(t * b * a)
        logp_flat = old_logp.reshape(t * b * a)
        adv_flat = gae.advantages.reshape(t * b * a)
        ret_flat = gae.returns.reshape(t * b * a)

        stats = ppo_update(
            model,
            opt,
            obs=obs_flat,
            actions=act_flat,
            old_logp=logp_flat,
            advantages=adv_flat,
            returns=ret_flat,
            cfg=ppo_cfg,
        )

        global_step += steps_per_update

        pac_rew = float(rewards[:, :, 0].mean().item())
        ghost_rew = float(rewards[:, :, 1:].mean().item())

        upd_time = time.time() - upd_start
        fps = float(steps_per_update / max(1e-6, upd_time))

        if global_step // log_cfg.sqlite_flush_every_steps > last_sqlite_step // log_cfg.sqlite_flush_every_steps:
            sqlite.write_metrics(
                MetricsRow(
                    global_step=int(global_step),
                    episode=int(total_episodes),
                    pacman_reward_mean=pac_rew,
                    ghosts_reward_mean=ghost_rew,
                    loss=float(stats.loss),
                    policy_loss=float(stats.policy_loss),
                    value_loss=float(stats.value_loss),
                    entropy=float(stats.entropy),
                    approx_kl=float(stats.approx_kl),
                    fps=fps,
                )
            )
            last_sqlite_step = global_step

        if reporter is not None and log_cfg.telegram_progress_every_steps > 0:
            if global_step // log_cfg.telegram_progress_every_steps > last_progress_step // log_cfg.telegram_progress_every_steps:
                elapsed = time.time() - train_start
                text = (
                    f"step={global_step} update={update + 1}/{updates}\n"
                    f"pac_rew_mean={pac_rew:.3f} ghost_rew_mean={ghost_rew:.3f}\n"
                    f"loss={stats.loss:.4f} kl={stats.approx_kl:.6f} ent={stats.entropy:.4f}\n"
                    f"fps={fps:.0f} elapsed_s={elapsed:.0f}"
                )
                reporter.upsert_progress(text=text)
                last_progress_step = global_step

        if reporter is not None and log_cfg.telegram_db_every_steps > 0:
            if global_step // log_cfg.telegram_db_every_steps > last_sqlite_sent_step // log_cfg.telegram_db_every_steps:
                reporter.send_sqlite(db_path=Path(log_cfg.sqlite_path), caption=f"metrics step={global_step}")
                last_sqlite_sent_step = global_step

        if (update + 1) % 10 == 0:
            print(
                "update="
                + str(update + 1)
                + " step="
                + str(global_step)
                + " pac_rew="
                + f"{pac_rew:.3f}"
                + " ghost_rew="
                + f"{ghost_rew:.3f}"
                + " fps="
                + f"{fps:.0f}"
            )

    sqlite.close()


if __name__ == "__main__":
    main()
