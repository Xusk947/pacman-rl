from __future__ import annotations

import mimetypes
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse
from urllib import request


@dataclass(frozen=True)
class TelegramTarget:
    bot_token: str
    chat_id: str


def telegram_target_from_env(*, bot_token_env: str, chat_id_env: str) -> TelegramTarget:
    bot_token = os.environ.get(bot_token_env, "")
    chat_id = os.environ.get(chat_id_env, "")
    if not bot_token:
        raise ValueError(f"missing env var: {bot_token_env}")
    if not chat_id:
        raise ValueError(f"missing env var: {chat_id_env}")
    return TelegramTarget(bot_token=bot_token, chat_id=chat_id)


def send_message(*, target: TelegramTarget, text: str, timeout_s: int = 30) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{target.bot_token}/sendMessage"
    body = parse.urlencode({"chat_id": target.chat_id, "text": text}).encode("utf-8")

    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Content-Length", str(len(body)))

    with request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
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


def send_document(
    *,
    target: TelegramTarget,
    file_path: Path,
    caption: str = "",
    timeout_s: int = 60,
) -> dict[str, Any]:
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

    with request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return {"status": "ok", "bytes": len(data)}
