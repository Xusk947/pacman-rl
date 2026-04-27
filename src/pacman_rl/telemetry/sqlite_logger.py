from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricsRow:
    algo: str
    global_step: int
    episode: int
    pellets_eaten: int
    power_eaten: int
    pacman_reward_mean: float
    ghosts_reward_mean: float
    win_rate: float | None
    death_rate: float | None
    loss: float | None
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None
    approx_kl: float | None
    fps: float | None
    elapsed_s: float | None


class SqliteLogger:
    def __init__(self, *, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
              id INTEGER PRIMARY KEY,
              algo TEXT NOT NULL DEFAULT 'ppo',
              global_step INTEGER NOT NULL,
              episode INTEGER NOT NULL,
              ts_unix INTEGER NOT NULL,
              pellets_eaten INTEGER NOT NULL DEFAULT 0,
              power_eaten INTEGER NOT NULL DEFAULT 0,
              pacman_reward_mean REAL NOT NULL,
              ghosts_reward_mean REAL NOT NULL,
              win_rate REAL,
              death_rate REAL,
              loss REAL,
              policy_loss REAL,
              value_loss REAL,
              entropy REAL,
              approx_kl REAL,
              fps REAL,
              elapsed_s REAL
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_algo_step ON metrics(algo, global_step);")
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(metrics);").fetchall()}
        if "algo" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN algo TEXT NOT NULL DEFAULT 'ppo';")
        if "pellets_eaten" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN pellets_eaten INTEGER NOT NULL DEFAULT 0;")
        if "power_eaten" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN power_eaten INTEGER NOT NULL DEFAULT 0;")
        if "win_rate" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN win_rate REAL;")
        if "death_rate" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN death_rate REAL;")
        if "elapsed_s" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN elapsed_s REAL;")

    def write_metrics(self, row: MetricsRow) -> None:
        ts = int(time.time())
        self._conn.execute("BEGIN;")
        try:
            self._conn.execute(
                """
                INSERT INTO metrics(
                  algo, global_step, episode, ts_unix, pellets_eaten, power_eaten,
                  pacman_reward_mean, ghosts_reward_mean,
                  win_rate, death_rate,
                  loss, policy_loss, value_loss, entropy, approx_kl, fps, elapsed_s
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(row.algo),
                    int(row.global_step),
                    int(row.episode),
                    ts,
                    int(row.pellets_eaten),
                    int(row.power_eaten),
                    float(row.pacman_reward_mean),
                    float(row.ghosts_reward_mean),
                    None if row.win_rate is None else float(row.win_rate),
                    None if row.death_rate is None else float(row.death_rate),
                    None if row.loss is None else float(row.loss),
                    None if row.policy_loss is None else float(row.policy_loss),
                    None if row.value_loss is None else float(row.value_loss),
                    None if row.entropy is None else float(row.entropy),
                    None if row.approx_kl is None else float(row.approx_kl),
                    None if row.fps is None else float(row.fps),
                    None if row.elapsed_s is None else float(row.elapsed_s),
                ),
            )
        except Exception:
            self._conn.execute("ROLLBACK;")
            raise
        self._conn.execute("COMMIT;")

    def last_n_metrics(self, n: int) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT algo, global_step, episode, ts_unix, pellets_eaten, power_eaten,
                   pacman_reward_mean, ghosts_reward_mean,
                   win_rate, death_rate,
                   loss, policy_loss, value_loss, entropy, approx_kl, fps, elapsed_s
            FROM metrics
            ORDER BY id DESC
            LIMIT ?;
            """,
            (int(n),),
        )
        rows = cur.fetchall()
        return [
            {
                "algo": r[0],
                "global_step": r[1],
                "episode": r[2],
                "ts_unix": r[3],
                "pellets_eaten": r[4],
                "power_eaten": r[5],
                "pacman_reward_mean": r[6],
                "ghosts_reward_mean": r[7],
                "win_rate": r[8],
                "death_rate": r[9],
                "loss": r[10],
                "policy_loss": r[11],
                "value_loss": r[12],
                "entropy": r[13],
                "approx_kl": r[14],
                "fps": r[15],
                "elapsed_s": r[16],
            }
            for r in rows
        ]
