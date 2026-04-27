from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pacman_rl.telemetry.telegram_api import TelegramRateLimitError, TelegramTarget, edit_message_text, send_document, send_message


@dataclass
class ActiveMessage:
    message_id: int
    sent_at_unix: int


class TelegramReporter:
    def __init__(
        self,
        *,
        target: TelegramTarget,
        dry_run: bool = False,
        active_window_s: int = 3600,
    ) -> None:
        self.target = target
        self.dry_run = bool(dry_run)
        self.active_window_s = int(active_window_s)
        self.active: ActiveMessage | None = None
        self.events: list[dict] = []

    def _now(self) -> int:
        return int(time.time())

    def upsert_progress(self, *, text: str, now_unix: int | None = None) -> None:
        now = int(now_unix) if now_unix is not None else self._now()
        if self.dry_run:
            self.events.append({"type": "progress", "ts": now, "text": text})
            if self.active is None or (now - self.active.sent_at_unix) > self.active_window_s:
                self.active = ActiveMessage(message_id=int(len(self.events)), sent_at_unix=now)
            return

        if self.active is None or (now - self.active.sent_at_unix) > self.active_window_s:
            resp = send_message(target=self.target, text=text)
            mid = int(resp.get("message_id") or 0)
            if mid <= 0:
                raise RuntimeError("telegram sendMessage did not return message_id")
            self.active = ActiveMessage(message_id=mid, sent_at_unix=now)
            return

        try:
            edit_message_text(target=self.target, message_id=self.active.message_id, text=text)
        except TelegramRateLimitError as e:
            time.sleep(float(e.retry_after_s) + 0.5)
            edit_message_text(target=self.target, message_id=self.active.message_id, text=text)
        except Exception:
            resp = send_message(target=self.target, text=text)
            mid = int(resp.get("message_id") or 0)
            if mid > 0:
                self.active = ActiveMessage(message_id=mid, sent_at_unix=now)

    def send_sqlite(self, *, db_path: Path, caption: str = "") -> None:
        if self.dry_run:
            self.events.append({"type": "sqlite", "path": str(db_path), "caption": caption})
            return
        if not db_path.exists():
            return
        for _ in range(3):
            try:
                send_document(target=self.target, file_path=db_path, caption=caption)
                return
            except TelegramRateLimitError as e:
                time.sleep(float(e.retry_after_s) + 0.5)
        send_document(target=self.target, file_path=db_path, caption=caption)

    def send_gif(self, *, gif_path: Path, caption: str = "") -> None:
        if self.dry_run:
            self.events.append({"type": "gif", "path": str(gif_path), "caption": caption})
            return
        if not gif_path.exists():
            return
        for _ in range(3):
            try:
                send_document(target=self.target, file_path=gif_path, caption=caption)
                return
            except TelegramRateLimitError as e:
                time.sleep(float(e.retry_after_s) + 0.5)
        send_document(target=self.target, file_path=gif_path, caption=caption)
