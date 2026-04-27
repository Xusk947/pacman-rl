from __future__ import annotations

import json
import sqlite3
import time
import uuid
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRow:
    run_id: str
    algo: str
    env_id: str
    seed: int
    device: str
    total_timesteps: int
    started_at: str
    ended_at: str | None
    config_json: str


@dataclass(frozen=True)
class EpisodeRow:
    run_id: str
    env_index: int
    episode_index: int
    episode_return: float
    episode_length: int
    win: int
    pellets: int
    power_pellets: int
    ghosts: int
    percent_cleared: float | None
    ended_at: str


@dataclass(frozen=True)
class TrainingRow:
    run_id: str
    timestep: int
    metrics_json: str
    logged_at: str


def now_timestamptz() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SQLiteLogger:
    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, timeout=30)
        try:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_schema()
            self._try_set_wal_mode()
            self._refresh_table_columns()
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "disk" in msg or "i/o" in msg:
                self._conn.close()
                self._recover_broken_db_file()
                self._conn = sqlite3.connect(self._db_path, timeout=30)
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._ensure_schema()
                self._try_set_wal_mode()
                self._refresh_table_columns()
            else:
                raise

    def _refresh_table_columns(self) -> None:
        self._runs_cols = self._get_table_cols("runs")
        self._episode_cols = self._get_table_cols("episode_metrics")
        self._training_cols = self._get_table_cols("training_metrics")

    def _get_table_cols(self, table: str) -> set[str]:
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _try_set_wal_mode(self) -> None:
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              algo TEXT NOT NULL,
              env_id TEXT NOT NULL,
              seed INTEGER NOT NULL,
              device TEXT NOT NULL,
              total_timesteps INTEGER NOT NULL,
              started_at TEXT NOT NULL,
              ended_at TEXT,
              config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episode_metrics (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              env_index INTEGER NOT NULL,
              episode_index INTEGER NOT NULL,
              episode_return REAL NOT NULL,
              episode_length INTEGER NOT NULL,
              win INTEGER NOT NULL,
              pellets INTEGER NOT NULL,
              power_pellets INTEGER NOT NULL,
              ghosts INTEGER NOT NULL,
              percent_cleared REAL,
              ended_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_episode_metrics_run_id ON episode_metrics(run_id);

            CREATE TABLE IF NOT EXISTS training_metrics (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              timestep INTEGER NOT NULL,
              metrics_json TEXT NOT NULL,
              logged_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_training_metrics_run_id ON training_metrics(run_id);
            """
        )

        self._migrate_from_unix_timestamps_if_needed()

    def _recover_broken_db_file(self) -> None:
        db_path = Path(self._db_path)
        if not db_path.exists():
            return

        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = db_path.with_name(f"{db_path.name}.corrupt.{suffix}")
        sidecars = [
            db_path.with_name(f"{db_path.name}-wal"),
            db_path.with_name(f"{db_path.name}-shm"),
            db_path.with_name(f"{db_path.name}-journal"),
        ]
        try:
            shutil.copy2(db_path, backup_path)
            db_path.unlink(missing_ok=True)
            for p in sidecars:
                p.unlink(missing_ok=True)
        except Exception:
            try:
                db_path.replace(backup_path)
            except Exception:
                pass

    def _migrate_from_unix_timestamps_if_needed(self) -> None:
        self._migrate_table_timestamp(table="runs", old_column="started_at_unix", new_column="started_at")
        self._migrate_table_timestamp(table="runs", old_column="ended_at_unix", new_column="ended_at")
        self._migrate_table_timestamp(table="episode_metrics", old_column="ended_at_unix", new_column="ended_at")
        self._migrate_table_timestamp(table="training_metrics", old_column="logged_at_unix", new_column="logged_at")

    def _migrate_table_timestamp(self, *, table: str, old_column: str, new_column: str) -> None:
        cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if old_column not in cols:
            return
        if new_column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {new_column} TEXT")

        self._conn.execute(
            f"UPDATE {table} SET {new_column} = strftime('%Y-%m-%dT%H:%M:%fZ', {old_column}, 'unixepoch') WHERE {new_column} IS NULL AND {old_column} IS NOT NULL"
        )
        self._conn.commit()

    def start_run(self, *, algo: str, env_id: str, seed: int, device: str, total_timesteps: int, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        started_at = now_timestamptz()
        started_at_unix = time.time()
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)

        cols: list[str] = ["run_id", "algo", "env_id", "seed", "device", "total_timesteps", "config_json"]
        vals: list[Any] = [run_id, algo, env_id, seed, device, total_timesteps, config_json]

        if "started_at" in self._runs_cols:
            cols.append("started_at")
            vals.append(started_at)
        if "started_at_unix" in self._runs_cols:
            cols.append("started_at_unix")
            vals.append(started_at_unix)
        if "ended_at" in self._runs_cols:
            cols.append("ended_at")
            vals.append(None)
        if "ended_at_unix" in self._runs_cols:
            cols.append("ended_at_unix")
            vals.append(None)

        col_sql = ", ".join(cols)
        q_sql = ", ".join(["?"] * len(cols))
        self._conn.execute(f"INSERT INTO runs({col_sql}) VALUES ({q_sql})", tuple(vals))
        self._conn.commit()
        return run_id

    def end_run(self, run_id: str) -> None:
        ended_at = now_timestamptz()
        ended_at_unix = time.time()
        if "ended_at" in self._runs_cols:
            self._conn.execute("UPDATE runs SET ended_at = ? WHERE run_id = ?", (ended_at, run_id))
        if "ended_at_unix" in self._runs_cols:
            self._conn.execute("UPDATE runs SET ended_at_unix = ? WHERE run_id = ?", (ended_at_unix, run_id))
        self._conn.commit()

    def log_episode(self, row: EpisodeRow) -> None:
        cols: list[str] = [
            "run_id",
            "env_index",
            "episode_index",
            "episode_return",
            "episode_length",
            "win",
            "pellets",
            "power_pellets",
            "ghosts",
            "percent_cleared",
        ]
        vals: list[Any] = [
            row.run_id,
            row.env_index,
            row.episode_index,
            row.episode_return,
            row.episode_length,
            row.win,
            row.pellets,
            row.power_pellets,
            row.ghosts,
            row.percent_cleared,
        ]

        if "ended_at" in self._episode_cols:
            cols.append("ended_at")
            vals.append(row.ended_at)
        if "ended_at_unix" in self._episode_cols:
            cols.append("ended_at_unix")
            vals.append(time.time())

        col_sql = ", ".join(cols)
        q_sql = ", ".join(["?"] * len(cols))
        self._conn.execute(f"INSERT INTO episode_metrics({col_sql}) VALUES ({q_sql})", tuple(vals))

    def log_training(self, row: TrainingRow) -> None:
        cols: list[str] = ["run_id", "timestep", "metrics_json"]
        vals: list[Any] = [row.run_id, row.timestep, row.metrics_json]

        if "logged_at" in self._training_cols:
            cols.append("logged_at")
            vals.append(row.logged_at)
        if "logged_at_unix" in self._training_cols:
            cols.append("logged_at_unix")
            vals.append(time.time())

        col_sql = ", ".join(cols)
        q_sql = ", ".join(["?"] * len(cols))
        self._conn.execute(f"INSERT INTO training_metrics({col_sql}) VALUES ({q_sql})", tuple(vals))

    def commit(self) -> None:
        self._conn.commit()
