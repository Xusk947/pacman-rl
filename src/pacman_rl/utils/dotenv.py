from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = Path(".env"), *, override: bool = False) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
            value = value[1:-1]

        if not override and key in os.environ:
            continue

        os.environ[key] = value

    return True
