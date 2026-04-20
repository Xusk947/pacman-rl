from __future__ import annotations

from pathlib import Path
from typing import Any


def write_telemetry_xlsx(path: Path, *, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("openpyxl is required to export telemetry to .xlsx") from e

    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"

    if not rows:
        ws["A1"] = "no data"
        wb.save(path)
        return

    keys = sorted({k for r in rows for k in r.keys()})
    ws.append(keys)
    for r in rows:
        ws.append([r.get(k) for k in keys])

    wb.save(path)
