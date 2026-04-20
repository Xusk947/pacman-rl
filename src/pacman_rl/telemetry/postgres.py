from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any


def _get_secret(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def _connect(database_url: str) -> Any:
    try:
        import psycopg

        return psycopg.connect(database_url)
    except Exception:
        pass
    try:
        import psycopg2

        return psycopg2.connect(database_url)
    except Exception as e:
        raise ModuleNotFoundError("Install 'psycopg' or 'psycopg2-binary' to enable Postgres logging") from e


@dataclass
class PostgresLogger:
    database_url: str
    run_uuid: str
    conn: Any

    @classmethod
    def from_env(cls, *, url_env: str = "DATABASE_URL") -> PostgresLogger | None:
        url = _get_secret(url_env)
        if not url:
            return None
        run_uuid = uuid.uuid4().hex
        conn = _connect(url)
        try:
            conn.autocommit = True
        except Exception:
            pass
        out = cls(database_url=url, run_uuid=run_uuid, conn=conn)
        out.ensure_schema()
        return out

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def ensure_schema(self) -> None:
        sql = """
        create table if not exists runs (
          run_uuid text primary key,
          started_at timestamptz not null default now(),
          meta jsonb
        );

        create table if not exists episodes (
          run_uuid text not null,
          episode_id bigint not null,
          layout text,
          steps int,
          pacman_reward double precision,
          ghosts_reward double precision,
          pellets_eaten int,
          power_eaten int,
          ghosts_eaten int,
          ended_by text,
          created_at timestamptz not null default now(),
          primary key (run_uuid, episode_id)
        );

        create index if not exists episodes_run_uuid_idx on episodes(run_uuid);
        """
        cur = self.conn.cursor()
        cur.execute(sql)
        try:
            cur.close()
        except Exception:
            pass

    def log_run_start(self, *, meta: dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "insert into runs(run_uuid, meta) values (%s, %s) on conflict (run_uuid) do nothing",
            (self.run_uuid, meta),
        )
        try:
            cur.close()
        except Exception:
            pass

    def log_episodes(self, *, rows: list[dict[str, Any]], retries: int = 5) -> None:
        if not rows:
            return
        values = []
        for r in rows:
            values.append(
                (
                    self.run_uuid,
                    int(r["episode_id"]),
                    r.get("layout"),
                    int(r.get("steps", 0)),
                    float(r.get("pacman_reward", 0.0)),
                    float(r.get("ghosts_reward", 0.0)),
                    int(r.get("pellets_eaten", 0)),
                    int(r.get("power_eaten", 0)),
                    int(r.get("ghosts_eaten", 0)),
                    r.get("ended_by"),
                )
            )
        sql = (
            "insert into episodes(run_uuid, episode_id, layout, steps, pacman_reward, ghosts_reward, pellets_eaten, power_eaten, ghosts_eaten, ended_by) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "on conflict (run_uuid, episode_id) do nothing"
        )

        last_err = None
        for i in range(retries):
            try:
                cur = self.conn.cursor()
                cur.executemany(sql, values)
                try:
                    cur.close()
                except Exception:
                    pass
                return
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2**i))
        raise last_err
