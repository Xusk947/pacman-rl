from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryBuffer:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def to_rows(self) -> list[dict[str, Any]]:
        return list(self.rows)
