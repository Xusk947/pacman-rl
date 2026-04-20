from __future__ import annotations

from pathlib import Path
from typing import Any


def write_telemetry_xlsx(path: Path, *, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.chart import LineChart, Reference
        from openpyxl.chart.series import SeriesLabel
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

    def add_chart(
        title: str,
        y_keys: list[str],
        anchor: str,
        *,
        y_title: str,
        series_titles: dict[str, str] | None = None,
    ) -> None:
        existing = [k for k in y_keys if k in keys]
        if not existing:
            return

        if "step" in keys:
            x_key = "step"
        elif "update" in keys:
            x_key = "update"
        else:
            x_key = keys[0]
        x_idx = keys.index(x_key) + 1

        chart = LineChart()
        chart.title = title
        chart.y_axis.title = y_title
        chart.x_axis.title = x_key

        min_row = 1
        max_row = 1 + len(rows)
        cats = Reference(ws, min_col=x_idx, min_row=min_row + 1, max_row=max_row)
        chart.set_categories(cats)

        for k in existing:
            col = keys.index(k) + 1
            data = Reference(ws, min_col=col, min_row=min_row, max_row=max_row)
            chart.add_data(data, titles_from_data=True)
            if series_titles is not None and k in series_titles:
                chart.series[-1].title = SeriesLabel(v=series_titles[k])

        ws.add_chart(chart, anchor)

    add_chart(
        "Reward",
        ["pacman_reward_mean", "ghosts_reward_mean"],
        "A" + str(len(rows) + 3),
        y_title="reward",
        series_titles={"pacman_reward_mean": "Pacman", "ghosts_reward_mean": "Ghosts"},
    )
    add_chart("Loss / KL", ["pacman_loss", "pacman_approx_kl", "ghosts_loss", "ghosts_approx_kl"], "I" + str(len(rows) + 3), y_title="value")

    wb.save(path)
