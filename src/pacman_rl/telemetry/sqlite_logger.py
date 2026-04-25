from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricsRow:
    global_step: int
    episode: int
    pellets_eaten: int
    power_eaten: int
    pacman_reward_mean: float
    ghosts_reward_mean: float
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
              global_step INTEGER NOT NULL,
              episode INTEGER NOT NULL,
              ts_unix INTEGER NOT NULL,
              pellets_eaten INTEGER NOT NULL DEFAULT 0,
              power_eaten INTEGER NOT NULL DEFAULT 0,
              pacman_reward_mean REAL NOT NULL,
              ghosts_reward_mean REAL NOT NULL,
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
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_global_step ON metrics(global_step);")
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(metrics);").fetchall()}
        if "pellets_eaten" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN pellets_eaten INTEGER NOT NULL DEFAULT 0;")
        if "power_eaten" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN power_eaten INTEGER NOT NULL DEFAULT 0;")
        if "elapsed_s" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN elapsed_s REAL;")

    def write_metrics(self, row: MetricsRow) -> None:
        ts = int(time.time())
        self._conn.execute("BEGIN;")
        try:
            self._conn.execute(
                """
                INSERT INTO metrics(
                  global_step, episode, ts_unix, pellets_eaten, power_eaten,
                  pacman_reward_mean, ghosts_reward_mean,
                  loss, policy_loss, value_loss, entropy, approx_kl, fps, elapsed_s
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    int(row.global_step),
                    int(row.episode),
                    ts,
                    int(row.pellets_eaten),
                    int(row.power_eaten),
                    float(row.pacman_reward_mean),
                    float(row.ghosts_reward_mean),
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
            SELECT global_step, episode, ts_unix, pellets_eaten, power_eaten,
                   pacman_reward_mean, ghosts_reward_mean,
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
                "global_step": r[0],
                "episode": r[1],
                "ts_unix": r[2],
                "pellets_eaten": r[3],
                "power_eaten": r[4],
                "pacman_reward_mean": r[5],
                "ghosts_reward_mean": r[6],
                "loss": r[7],
                "policy_loss": r[8],
                "value_loss": r[9],
                "entropy": r[10],
                "approx_kl": r[11],
                "fps": r[12],
                "elapsed_s": r[13],
            }
            for r in rows
        ]
