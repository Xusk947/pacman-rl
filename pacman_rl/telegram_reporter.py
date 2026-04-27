from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    enabled: bool


def detect_telegram_config() -> TelegramConfig:
    bot_token = os.environ.get("BOT_TOKEN", "").strip()
    chat_id = os.environ.get("USER_ID", "").strip()

    if not bot_token or not chat_id:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            client = UserSecretsClient()
            if not bot_token:
                bot_token = str(client.get_secret("BOT_TOKEN") or "").strip()
            if not chat_id:
                chat_id = str(client.get_secret("USER_ID") or "").strip()
        except Exception:
            pass

    enabled = bool(bot_token and chat_id)
    return TelegramConfig(bot_token=bot_token, chat_id=chat_id, enabled=enabled)


class TelegramReporter:
    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._last_message_id: str | None = None
        self._last_text_hash: int = 0
        self._last_edit_time: float = 0.0
        self._min_edit_interval: float = 1.5

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def _post(self, method: str, data: dict[str, Any] | None = None, files: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self._config.enabled:
            return None

        url = f"https://api.telegram.org/bot{self._config.bot_token}/{method}"

        try:
            import requests  # type: ignore

            for attempt in range(3):
                if files:
                    resp = requests.post(url, data=data or {}, files=files, timeout=60)
                else:
                    resp = requests.post(url, json=data or {}, timeout=30)

                try:
                    payload = resp.json()
                except Exception:
                    payload = None

                retry_after = None
                if isinstance(payload, dict):
                    params = payload.get("parameters")
                    if isinstance(params, dict) and "retry_after" in params:
                        try:
                            retry_after = int(params["retry_after"])
                        except Exception:
                            retry_after = None

                if resp.status_code == 200 and isinstance(payload, dict) and payload.get("ok"):
                    return payload

                if resp.status_code == 429 and retry_after is not None and attempt < 2:
                    time.sleep(max(1, retry_after))
                    continue

                if resp.status_code != 200:
                    logger.error("Telegram HTTP error %s: %s", resp.status_code, resp.text[:2000])
                elif isinstance(payload, dict):
                    logger.error("Telegram API error: %s", json.dumps(payload, ensure_ascii=False)[:2000])
                else:
                    logger.error("Telegram API error: invalid response")
                return None
        except Exception:
            pass

        import urllib.error
        import urllib.request

        body = None
        headers: dict[str, str] = {}

        if files:
            boundary = f"----TelegramBoundary{int(time.time()*1000)}"
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            parts: list[bytes] = []
            for key, val in (data or {}).items():
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode("utf-8"))
            for key, (filename, content, content_type) in files.items():
                parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode(
                        "utf-8"
                    )
                )
                parts.append(content)
                parts.append(b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode("utf-8"))
            body = b"".join(parts)
        else:
            body = json.dumps(data or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.error("Telegram HTTP error %s: %s", e.code, e.read().decode("utf-8", errors="ignore")[:2000])
        except Exception as e:
            logger.error("Telegram request failed: %s", e)
        return None

    def send_message(self, text: str, *, parse_mode: str = "HTML") -> str | None:
        if not self._config.enabled:
            return None

        resp = self._post("sendMessage", {"chat_id": self._config.chat_id, "text": text, "parse_mode": parse_mode})
        if resp and resp.get("ok"):
            result = resp.get("result", {})
            self._last_message_id = str(result.get("message_id", ""))
            return self._last_message_id
        return None

    def edit_message(self, message_id: str, text: str, *, parse_mode: str = "HTML") -> bool:
        if not self._config.enabled:
            return False

        text_hash = hash(text)
        now = time.time()
        if now - self._last_edit_time < self._min_edit_interval:
            time.sleep(self._min_edit_interval - (now - self._last_edit_time))

        now = time.time()
        if message_id == self._last_message_id and text_hash == self._last_text_hash:
            self._last_edit_time = now
            return True

        resp = self._post(
            "editMessageText",
            {"chat_id": self._config.chat_id, "message_id": int(message_id), "text": text, "parse_mode": parse_mode},
        )
        if resp and resp.get("ok"):
            self._last_text_hash = text_hash
            self._last_edit_time = now
            return True
        return False

    def send_or_edit(self, text: str, *, parse_mode: str = "HTML") -> str | None:
        if self._last_message_id:
            if self.edit_message(self._last_message_id, text, parse_mode=parse_mode):
                return self._last_message_id
        return self.send_message(text, parse_mode=parse_mode)

    def send_video(self, video_path: str, caption: str = "", *, parse_mode: str = "HTML") -> bool:
        if not self._config.enabled:
            return False

        p = Path(video_path)
        if not p.exists():
            logger.error("Video file not found: %s", video_path)
            return False

        try:
            with open(p, "rb") as f:
                resp = self._post(
                    "sendVideo",
                    {"chat_id": self._config.chat_id, "caption": caption, "parse_mode": parse_mode},
                    files={"video": (p.name, f, "video/mp4")},
                )
            return bool(resp and resp.get("ok"))
        except Exception as e:
            logger.error("send_video failed: %s", e)
            return False

    def send_document(self, file_path: str, caption: str = "", *, parse_mode: str = "HTML") -> bool:
        if not self._config.enabled:
            return False

        p = Path(file_path)
        if not p.exists():
            logger.error("Document file not found: %s", file_path)
            return False

        try:
            with open(p, "rb") as f:
                resp = self._post(
                    "sendDocument",
                    {"chat_id": self._config.chat_id, "caption": caption, "parse_mode": parse_mode},
                    files={"document": (p.name, f, "application/octet-stream")},
                )
            return bool(resp and resp.get("ok"))
        except Exception as e:
            logger.error("send_document failed: %s", e)
            return False

    def format_progress(self, algo: str, current_step: int, total_steps: int, *, model_index: int, model_total: int, extra: str = "") -> str:
        percent = int(100 * min(current_step, total_steps) / max(1, total_steps))
        bar_len = 20
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        text = f"<b>{algo.upper()}</b> [{bar}] {percent}%\ntraining steps {current_step}/{total_steps}\nstage {model_index}/{model_total}"
        if extra:
            text += f"\n{extra}"
        return text
