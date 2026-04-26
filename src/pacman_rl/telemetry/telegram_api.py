from __future__ import annotations

import json
import mimetypes
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse
from urllib import request
from urllib.error import HTTPError


@dataclass(frozen=True)
class TelegramTarget:
    bot_token: str
    chat_id: str


class TelegramRateLimitError(RuntimeError):
    def __init__(self, *, retry_after_s: int, payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"telegram_rate_limited retry_after_s={retry_after_s}")
        self.retry_after_s = int(retry_after_s)
        self.payload = payload


def telegram_target_from_env(*, bot_token_env: str = "TELEGRAM_BOT_TOKEN", chat_id_env: str = "TELEGRAM_CHAT_ID") -> TelegramTarget:
    bot_token = os.environ.get(bot_token_env, "")
    chat_id = os.environ.get(chat_id_env, "")
    if not bot_token:
        raise ValueError(f"missing env var: {bot_token_env}")
    if not chat_id:
        raise ValueError(f"missing env var: {chat_id_env}")
    return TelegramTarget(bot_token=bot_token, chat_id=chat_id)


def _handle_http_error(e: HTTPError) -> None:
    if e.code != 429:
        raise
    try:
        raw = e.read()
        payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
    except Exception:
        payload = None

    retry_after = None
    if isinstance(payload, dict):
        params = payload.get("parameters")
        if isinstance(params, dict) and "retry_after" in params:
            retry_after = int(params["retry_after"])
        else:
            desc = payload.get("description", "")
            if isinstance(desc, str) and "retry after" in desc:
                try:
                    retry_after = int(desc.split("retry after", 1)[1].strip().split()[0])
                except Exception:
                    retry_after = None

    raise TelegramRateLimitError(retry_after_s=retry_after or 3, payload=payload)


def send_message(*, target: TelegramTarget, text: str, timeout_s: int = 30) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{target.bot_token}/sendMessage"
    body = parse.urlencode({"chat_id": target.chat_id, "text": text}).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Content-Length", str(len(body)))

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except HTTPError as e:
        _handle_http_error(e)

    try:
        payload = json.loads(data.decode("utf-8", errors="replace"))
        msg_id = payload.get("result", {}).get("message_id")
    except Exception:
        msg_id = None
    return {"status": "ok", "message_id": msg_id}


def edit_message_text(
    *, target: TelegramTarget, message_id: int, text: str, timeout_s: int = 30
) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{target.bot_token}/editMessageText"
    body = parse.urlencode({"chat_id": target.chat_id, "message_id": str(int(message_id)), "text": text}).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Content-Length", str(len(body)))
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except HTTPError as e:
        _handle_http_error(e)
    return {"status": "ok", "bytes": len(data)}


def _encode_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----pacmanrl-" + uuid.uuid4().hex
    lines: list[bytes] = []

    for name, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(value.encode())
        lines.append(b"\r\n")

    for field_name, (filename, content, content_type) in files.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
        )
        lines.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        lines.append(content)
        lines.append(b"\r\n")

    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    return body, boundary


def send_document(*, target: TelegramTarget, file_path: Path, caption: str = "", timeout_s: int = 60) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{target.bot_token}/sendDocument"
    content = file_path.read_bytes()
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    body, boundary = _encode_multipart(
        fields={"chat_id": target.chat_id, "caption": caption},
        files={"document": (file_path.name, content, content_type)},
    )
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except HTTPError as e:
        _handle_http_error(e)
    return {"status": "ok", "bytes": len(data)}

