from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributions import Categorical

from pacman_rl.config import EnvConfig, LogConfig, PPOConfig
from pacman_rl.env import TorchPacmanEnv
from pacman_rl.ghosts import ClassicGhostPolicy
from pacman_rl.layouts import load_layouts_from_dir
from pacman_rl.models import QNetwork, SharedCNNActorCritic
from pacman_rl.rl import ReplayBuffer, a2c_update, compute_gae, dqn_update, ppo_update
from pacman_rl.telemetry import SqliteLogger, TelegramReporter, telegram_target_auto
from pacman_rl.telemetry.sqlite_logger import MetricsRow
from pacman_rl.telemetry.telegram_api import TelegramTarget
from pacman_rl.utils import resolve_device


@dataclass
class AlgoState:
    algo: str
    step: int = 0
    pellets: int = 0
    power: int = 0
    avg_reward: float | None = None
    win_rate: float | None = None
    death_rate: float | None = None
    fps: float | None = None


def _fmt_step(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_pct(done: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(min(100.0, max(0.0, 100.0 * float(done) / float(total))))


def _summary_text(*, pct: int, step: int, total: int, elapsed_s: float, states: list[AlgoState]) -> str:
    lines: list[str] = []
    lines.append(f"Progress: {pct}%  total_step={_fmt_step(step)} / {_fmt_step(total)}  elapsed={int(elapsed_s)}s")
    lines.append("")
    for st in states:
        lines.append(f"**{st.algo.upper()}**")
        lines.append(f"step: {_fmt_step(st.step)}")
        lines.append(f"pellets: {st.pellets}")
        lines.append(f"power: {st.power}")
        lines.append(f"avg_reward: {('-' if st.avg_reward is None else f'{st.avg_reward:.3f}')}")
        lines.append(f"win_rate: {('-' if st.win_rate is None else f'{st.win_rate * 100.0:.1f}%')}")
        lines.append(f"death_rate: {('-' if st.death_rate is None else f'{st.death_rate * 100.0:.1f}%')}")
        lines.append(f"fps: {('-' if st.fps is None else f'{st.fps:.0f}')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _sample_ac(model: SharedCNNActorCritic, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(obs)
    dist = Categorical(logits=out.logits)
    act = dist.sample()
    logp = dist.log_prob(act)
    return act, logp, out.value


@dataclass(frozen=True)
class RolloutAC:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_values: torch.Tensor
    pellets_eaten: int
    power_eaten: int
    episodes: int
    wins: int
    deaths: int


def _collect_rollout_ac(*, env: TorchPacmanEnv, model: SharedCNNActorCritic, ghosts: ClassicGhostPolicy, steps: int) -> RolloutAC:
    b = env.batch_size
    c = 13
    h = env.height
    w = env.width

    obs_buf = torch.zeros((steps, b, c, h, w), device=env.device, dtype=torch.float32)
    act_buf = torch.zeros((steps, b), device=env.device, dtype=torch.int64)
    logp_buf = torch.zeros((steps, b), device=env.device, dtype=torch.float32)
    val_buf = torch.zeros((steps, b), device=env.device, dtype=torch.float32)
    rew_buf = torch.zeros((steps, b), device=env.device, dtype=torch.float32)
    done_buf = torch.zeros((steps, b), device=env.device, dtype=torch.bool)

    pellets_eaten = 0
    power_eaten = 0
    episodes = 0
    wins = 0
    deaths = 0

    obs = env.get_obs()
    for t in range(int(steps)):
        pac_obs = obs[:, 0]
        obs_buf[t] = pac_obs

        with torch.no_grad():
            pac_action, pac_logp, pac_val = _sample_ac(model, pac_obs)

        ghost_action = ghosts.act(
            walls=env.walls,
            pacman_pos=env.pacman,
            ghost_pos=env.ghosts,
            frightened=env.frightened,
            step_in_ep=env.steps,
        )

        acts = torch.zeros((b, env.AGENTS), device=env.device, dtype=torch.int64)
        acts[:, 0] = pac_action
        acts[:, 1:] = ghost_action

        out = env.step(acts)

        act_buf[t] = pac_action
        logp_buf[t] = pac_logp
        val_buf[t] = pac_val
        rew_buf[t] = out.reward[:, 0]
        done_buf[t] = out.done

        pellets_eaten += int(out.info["pellet_eaten"].sum().item())
        power_eaten += int(out.info["power_eaten"].sum().item())
        episodes += int(out.done.to(torch.int64).sum().item())
        wins += int(out.info["win"].sum().item())
        deaths += int(out.info["pac_dead"].sum().item())

        obs = out.obs
        if out.done.any():
            obs = env.reset_done(out.done)
            ghosts.reset(batch_size=b, pacman_pos=env.pacman)

    with torch.no_grad():
        last_values = model(env.get_obs()[:, 0]).value

    return RolloutAC(
        obs=obs_buf,
        actions=act_buf,
        logp=logp_buf,
        values=val_buf,
        rewards=rew_buf,
        dones=done_buf,
        last_values=last_values,
        pellets_eaten=pellets_eaten,
        power_eaten=power_eaten,
        episodes=episodes,
        wins=wins,
        deaths=deaths,
    )


def _run_ppo_update(*, model: SharedCNNActorCritic, opt: torch.optim.Optimizer, roll: RolloutAC, ppo: PPOConfig) -> tuple[float, float, float]:
    gae = compute_gae(
        roll.rewards,
        roll.dones,
        roll.values,
        roll.last_values,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    t, b, c, h, w = roll.obs.shape
    obs_flat = roll.obs.reshape(t * b, c, h, w)
    act_flat = roll.actions.reshape(t * b)
    logp_flat = roll.logp.reshape(t * b)
    adv_flat = gae.advantages.reshape(t * b)
    ret_flat = gae.returns.reshape(t * b)
    stats = ppo_update(
        model,
        opt,
        obs=obs_flat,
        actions=act_flat,
        old_logp=logp_flat,
        advantages=adv_flat,
        returns=ret_flat,
        cfg=ppo,
    )
    return float(stats.loss), float(stats.entropy), float(stats.approx_kl)


def _run_a2c_update(*, model: SharedCNNActorCritic, opt: torch.optim.Optimizer, roll: RolloutAC, ppo: PPOConfig) -> tuple[float, float, float]:
    gae = compute_gae(
        roll.rewards,
        roll.dones,
        roll.values,
        roll.last_values,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    t, b, c, h, w = roll.obs.shape
    obs_flat = roll.obs.reshape(t * b, c, h, w)
    act_flat = roll.actions.reshape(t * b)
    adv_flat = gae.advantages.reshape(t * b)
    ret_flat = gae.returns.reshape(t * b)
    stats = a2c_update(
        model,
        opt,
        obs=obs_flat,
        actions=act_flat,
        returns=ret_flat,
        advantages=adv_flat,
        value_coef=ppo.value_coef,
        entropy_coef=ppo.entropy_coef,
        max_grad_norm=ppo.max_grad_norm,
    )
    return float(stats.loss), float(stats.entropy), 0.0


def _epsilon(*, step: int, total: int) -> float:
    if total <= 0:
        return 0.05
    frac = min(1.0, max(0.0, float(step) / float(total)))
    return float(1.0 - 0.95 * frac)


def _maybe_report(
    *,
    reporter: TelegramReporter | None,
    sqlite: SqliteLogger,
    sqlite_path: Path,
    pct_step: int,
    db_step: int,
    total_steps: int,
    combined_step: int,
    start_unix: float,
    states: list[AlgoState],
    next_report_at: list[int],
    next_db_at: list[int],
) -> None:
    if reporter is None:
        return
    if total_steps <= 0:
        return

    if combined_step >= next_report_at[0]:
        pct = _fmt_pct(combined_step, total_steps)
        text = _summary_text(pct=pct, step=combined_step, total=total_steps, elapsed_s=time.time() - start_unix, states=states)
        reporter.upsert_progress(text=text)
        next_report_at[0] += max(1, int(total_steps * pct_step / 100))

    if combined_step >= next_db_at[0]:
        reporter.send_sqlite(db_path=sqlite_path, caption=f"metrics step={combined_step}")
        next_db_at[0] += max(1, int(total_steps * db_step / 100))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout-dir", type=Path, default=Path("layouts"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "dqn", "sweep"])
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--telegram-dry-run", action="store_true")
    p.add_argument("--telegram-progress-every-percent", type=int, default=5)
    p.add_argument("--telegram-db-every-percent", type=int, default=25)
    args = p.parse_args()

    device = resolve_device(str(args.device))
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    log_cfg = LogConfig(sqlite_path=str((Path(args.run_dir) / "metrics.sqlite").as_posix()))

    layouts = load_layouts_from_dir(Path(args.layout_dir))
    sqlite_path = Path(log_cfg.sqlite_path)
    sqlite = SqliteLogger(db_path=sqlite_path)

    reporter: TelegramReporter | None = None
    if bool(args.telegram):
        if bool(args.telegram_dry_run):
            target = TelegramTarget(bot_token="dry_run", chat_id="dry_run")
        else:
            target = telegram_target_auto()
        reporter = TelegramReporter(target=target, dry_run=bool(args.telegram_dry_run))

    algo_list = ["ppo", "a2c", "dqn"] if str(args.algo) == "sweep" else [str(args.algo)]
    total_steps_per_algo = int(args.total_steps)
    total_steps_all = total_steps_per_algo * len(algo_list)

    pct_step = max(1, int(args.telegram_progress_every_percent))
    db_step = max(1, int(args.telegram_db_every_percent))
    next_report_at = [max(1, int(total_steps_all * pct_step / 100))]
    next_db_at = [max(1, int(total_steps_all * db_step / 100))]

    states = [AlgoState(algo=a) for a in ["ppo", "a2c", "dqn"]]
    state_by_algo = {s.algo: s for s in states}

    combined_step = 0
    start_unix = time.time()

    for algo in algo_list:
        env = TorchPacmanEnv(layouts, batch_size=int(args.batch_size), device=device, cfg=env_cfg)
        env.reset(seed=int(args.seed))
        ghosts = ClassicGhostPolicy(device=device)
        ghosts.reset(batch_size=env.batch_size, pacman_pos=env.pacman)

        steps_per_iter = int(env.batch_size) * int(ppo_cfg.rollout_steps)
        iters = max(1, (total_steps_per_algo + steps_per_iter - 1) // steps_per_iter)

        st = state_by_algo[algo]
        st.step = 0
        st.pellets = 0
        st.power = 0
        st.avg_reward = None
        st.win_rate = None
        st.death_rate = None
        st.fps = None

        if algo in ("ppo", "a2c"):
            model = SharedCNNActorCritic(in_channels=13, actions=env.ACTIONS).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

            for i in range(int(iters)):
                t0 = time.time()
                roll = _collect_rollout_ac(env=env, model=model, ghosts=ghosts, steps=int(ppo_cfg.rollout_steps))
                if algo == "ppo":
                    loss, entropy, kl = _run_ppo_update(model=model, opt=opt, roll=roll, ppo=ppo_cfg)
                else:
                    loss, entropy, kl = _run_a2c_update(model=model, opt=opt, roll=roll, ppo=ppo_cfg)

                st.step += steps_per_iter
                combined_step += steps_per_iter
                st.pellets += int(roll.pellets_eaten)
                st.power += int(roll.power_eaten)
                st.avg_reward = float(roll.rewards.mean().item())
                st.win_rate = (float(roll.wins) / float(roll.episodes)) if roll.episodes > 0 else None
                st.death_rate = (float(roll.deaths) / float(roll.episodes)) if roll.episodes > 0 else None
                st.fps = float(steps_per_iter / max(1e-6, time.time() - t0))

                if st.step // log_cfg.sqlite_flush_every_steps != (st.step - steps_per_iter) // log_cfg.sqlite_flush_every_steps:
                    sqlite.write_metrics(
                        MetricsRow(
                            algo=algo,
                            global_step=int(st.step),
                            episode=0,
                            pellets_eaten=int(st.pellets),
                            power_eaten=int(st.power),
                            pacman_reward_mean=float(st.avg_reward or 0.0),
                            ghosts_reward_mean=0.0,
                            win_rate=st.win_rate,
                            death_rate=st.death_rate,
                            loss=float(loss),
                            policy_loss=None,
                            value_loss=None,
                            entropy=float(entropy),
                            approx_kl=float(kl),
                            fps=float(st.fps or 0.0),
                            elapsed_s=float(time.time() - start_unix),
                        )
                    )

                _maybe_report(
                    reporter=reporter,
                    sqlite=sqlite,
                    sqlite_path=sqlite_path,
                    pct_step=pct_step,
                    db_step=db_step,
                    total_steps=total_steps_all,
                    combined_step=combined_step,
                    start_unix=start_unix,
                    states=states,
                    next_report_at=next_report_at,
                    next_db_at=next_db_at,
                )

                if (i + 1) % 10 == 0:
                    print(
                        "algo="
                        + algo
                        + " iter="
                        + str(i + 1)
                        + " step="
                        + str(st.step)
                        + " avg_reward="
                        + f"{float(st.avg_reward or 0.0):.3f}"
                        + " fps="
                        + f"{float(st.fps or 0.0):.0f}"
                    )

        if algo == "dqn":
            q = QNetwork(in_channels=13, actions=env.ACTIONS).to(device)
            target_q = QNetwork(in_channels=13, actions=env.ACTIONS).to(device)
            target_q.load_state_dict(q.state_dict())
            opt = torch.optim.Adam(q.parameters(), lr=ppo_cfg.learning_rate)

            obs_shape = (13, env.height, env.width)
            replay = ReplayBuffer(capacity=200_000, obs_shape=obs_shape, device=device)
            min_replay = 5_000
            batch_size = 512
            target_sync_every = 5_000
            train_steps_per_iter = 256

            obs = env.get_obs()[:, 0]
            for i in range(int(iters)):
                t0 = time.time()
                ep_done = 0
                win = 0
                death = 0
                reward_sum = 0.0

                for _ in range(int(ppo_cfg.rollout_steps)):
                    eps = _epsilon(step=st.step, total=total_steps_per_algo)
                    with torch.no_grad():
                        qv = q(obs).detach()
                        greedy = qv.argmax(dim=1)
                    rnd = torch.randint(low=0, high=env.ACTIONS, size=greedy.shape, device=device)
                    choose_rand = (torch.rand(greedy.shape, device=device) < eps)
                    pac_action = torch.where(choose_rand, rnd, greedy)

                    ghost_action = ghosts.act(
                        walls=env.walls,
                        pacman_pos=env.pacman,
                        ghost_pos=env.ghosts,
                        frightened=env.frightened,
                        step_in_ep=env.steps,
                    )
                    acts = torch.zeros((env.batch_size, env.AGENTS), device=device, dtype=torch.int64)
                    acts[:, 0] = pac_action
                    acts[:, 1:] = ghost_action

                    out = env.step(acts)
                    next_obs = out.obs[:, 0]

                    replay.add(
                        obs=obs,
                        actions=pac_action,
                        rewards=out.reward[:, 0],
                        next_obs=next_obs,
                        dones=out.done,
                    )

                    reward_sum += float(out.reward[:, 0].mean().item())
                    st.pellets += int(out.info["pellet_eaten"].sum().item())
                    st.power += int(out.info["power_eaten"].sum().item())
                    ep_done += int(out.done.to(torch.int64).sum().item())
                    win += int(out.info["win"].sum().item())
                    death += int(out.info["pac_dead"].sum().item())

                    obs = next_obs
                    if out.done.any():
                        env.reset_done(out.done)
                        ghosts.reset(batch_size=env.batch_size, pacman_pos=env.pacman)
                        obs = env.get_obs()[:, 0]

                if len(replay) >= min_replay:
                    for _ in range(int(train_steps_per_iter)):
                        batch = replay.sample(batch_size=batch_size)
                        dqn_update(q, target_q, opt, batch=batch, gamma=ppo_cfg.gamma, max_grad_norm=10.0)
                        if (st.step + 1) % target_sync_every == 0:
                            target_q.load_state_dict(q.state_dict())

                st.step += steps_per_iter
                combined_step += steps_per_iter
                st.avg_reward = float(reward_sum) / float(max(1, int(ppo_cfg.rollout_steps)))
                st.win_rate = (float(win) / float(ep_done)) if ep_done > 0 else None
                st.death_rate = (float(death) / float(ep_done)) if ep_done > 0 else None
                st.fps = float(steps_per_iter / max(1e-6, time.time() - t0))

                if st.step // log_cfg.sqlite_flush_every_steps != (st.step - steps_per_iter) // log_cfg.sqlite_flush_every_steps:
                    sqlite.write_metrics(
                        MetricsRow(
                            algo=algo,
                            global_step=int(st.step),
                            episode=0,
                            pellets_eaten=int(st.pellets),
                            power_eaten=int(st.power),
                            pacman_reward_mean=float(st.avg_reward or 0.0),
                            ghosts_reward_mean=0.0,
                            win_rate=st.win_rate,
                            death_rate=st.death_rate,
                            loss=None,
                            policy_loss=None,
                            value_loss=None,
                            entropy=None,
                            approx_kl=None,
                            fps=float(st.fps or 0.0),
                            elapsed_s=float(time.time() - start_unix),
                        )
                    )

                _maybe_report(
                    reporter=reporter,
                    sqlite=sqlite,
                    sqlite_path=sqlite_path,
                    pct_step=pct_step,
                    db_step=db_step,
                    total_steps=total_steps_all,
                    combined_step=combined_step,
                    start_unix=start_unix,
                    states=states,
                    next_report_at=next_report_at,
                    next_db_at=next_db_at,
                )

                if (i + 1) % 10 == 0:
                    print(
                        "algo="
                        + algo
                        + " iter="
                        + str(i + 1)
                        + " step="
                        + str(st.step)
                        + " avg_reward="
                        + f"{float(st.avg_reward or 0.0):.3f}"
                        + " fps="
                        + f"{float(st.fps or 0.0):.0f}"
                    )

    _maybe_report(
        reporter=reporter,
        sqlite=sqlite,
        sqlite_path=sqlite_path,
        pct_step=pct_step,
        db_step=db_step,
        total_steps=total_steps_all,
        combined_step=total_steps_all,
        start_unix=start_unix,
        states=states,
        next_report_at=[0],
        next_db_at=[total_steps_all + 1],
    )
    sqlite.close()


if __name__ == "__main__":
    main()
