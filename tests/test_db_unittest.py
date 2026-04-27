import os
import sqlite3
import tempfile
import unittest

from pacman_rl.db import EpisodeRow, SQLiteLogger, TrainingRow, now_timestamptz


class DbTests(unittest.TestCase):
    def test_sqlite_schema_and_inserts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "runs.sqlite")
            db = SQLiteLogger(db_path)
            try:
                run_id = db.start_run(
                    algo="ppo",
                    env_id="ALE/Pacman-v5",
                    seed=0,
                    device="cpu",
                    total_timesteps=100,
                    config={"a": 1},
                )
                db.log_episode(
                    EpisodeRow(
                        run_id=run_id,
                        env_index=0,
                        episode_index=0,
                        episode_return=123.0,
                        episode_length=10,
                        win=0,
                        pellets=1,
                        power_pellets=0,
                        ghosts=0,
                        percent_cleared=None,
                        ended_at=now_timestamptz(),
                    )
                )
                db.log_training(
                    TrainingRow(
                        run_id=run_id,
                        timestep=10,
                        metrics_json="{}",
                        logged_at=now_timestamptz(),
                    )
                )
                db.commit()
                db.end_run(run_id)
            finally:
                db.close()

            conn = sqlite3.connect(db_path)
            try:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM episode_metrics").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0], 1)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
