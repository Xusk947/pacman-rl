from __future__ import annotations

import unittest

from pacman_rl.telemetry.telegram_api import TelegramTarget
from pacman_rl.telemetry.telegram_reporter import TelegramReporter


class TestTelegramReporter(unittest.TestCase):
    def test_active_message_window(self) -> None:
        r = TelegramReporter(target=TelegramTarget(bot_token="x", chat_id="y"), dry_run=True, active_window_s=3600)

        r.upsert_progress(text="a", now_unix=100)
        self.assertIsNotNone(r.active)
        first_id = r.active.message_id

        r.upsert_progress(text="b", now_unix=200)
        self.assertEqual(r.active.message_id, first_id)

        r.upsert_progress(text="c", now_unix=100 + 3600 + 1)
        self.assertNotEqual(r.active.message_id, first_id)


if __name__ == "__main__":
    unittest.main()

