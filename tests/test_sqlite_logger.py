from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pacman_rl.telemetry.sqlite_logger import MetricsRow, SqliteLogger


class TestSqliteLogger(unittest.TestCase):
    def test_write_and_read_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "m.sqlite"
            log = SqliteLogger(db_path=db_path)
            log.write_metrics(
                MetricsRow(
                    algo="ppo",
                    global_step=1000,
                    episode=10,
                    pellets_eaten=123,
                    power_eaten=4,
                    pacman_reward_mean=1.25,
                    ghosts_reward_mean=-0.5,
                    win_rate=0.1,
                    death_rate=0.2,
                    loss=0.1,
                    policy_loss=0.2,
                    value_loss=0.3,
                    entropy=0.4,
                    approx_kl=0.01,
                    fps=999.0,
                    elapsed_s=12.34,
                )
            )
            rows = log.last_n_metrics(1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["algo"], "ppo")
            self.assertEqual(rows[0]["global_step"], 1000)
            self.assertEqual(rows[0]["pellets_eaten"], 123)
            self.assertEqual(rows[0]["power_eaten"], 4)
            self.assertAlmostEqual(rows[0]["elapsed_s"], 12.34, places=2)
            log.close()


if __name__ == "__main__":
    unittest.main()
