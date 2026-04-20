from __future__ import annotations

from pathlib import Path
from typing import Any


def render_rewards_png(path: Path, *, rows: list[dict[str, Any]], width: int = 1100, height: int = 600) -> None:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Pillow is required to render telemetry PNG") from e

    path.parent.mkdir(parents=True, exist_ok=True)

    updates: list[int] = []
    pac: list[float] = []
    ghosts: list[float] = []
    for r in rows:
        if "update" not in r:
            continue
        updates.append(int(r["update"]))
        pac.append(float(r.get("pacman_reward_mean", 0.0)))
        ghosts.append(float(r.get("ghosts_reward_mean", 0.0)))

    img = Image.new("RGB", (width, height), (16, 18, 22))
    d = ImageDraw.Draw(img)

    if not updates:
        d.text((20, 20), "no data", fill=(230, 230, 230))
        img.save(path, format="PNG")
        return

    left = 70
    right = 30
    top = 30
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_min = min(min(pac), min(ghosts))
    y_max = max(max(pac), max(ghosts))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    x_min = updates[0]
    x_max = updates[-1]
    if x_min == x_max:
        x_max += 1

    def x_to_px(x: int) -> float:
        return left + (float(x - x_min) / float(x_max - x_min)) * plot_w

    def y_to_px(y: float) -> float:
        return top + (1.0 - (float(y - y_min) / float(y_max - y_min))) * plot_h

    grid_color = (40, 45, 55)
    axis_color = (200, 200, 200)

    d.rectangle((left, top, left + plot_w, top + plot_h), outline=grid_color, width=1)

    for i in range(1, 5):
        y = top + (plot_h * i) / 5.0
        d.line((left, y, left + plot_w, y), fill=grid_color, width=1)

    for i in range(0, 6):
        yv = y_min + (y_max - y_min) * (1.0 - i / 5.0)
        yp = y_to_px(yv)
        label = f"{yv:.2f}"
        d.text((10, yp - 7), label, fill=(180, 180, 180))

    d.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis_color, width=2)
    d.line((left, top, left, top + plot_h), fill=axis_color, width=2)

    def draw_series(values: list[float], *, color: tuple[int, int, int]) -> None:
        pts = [(x_to_px(x), y_to_px(y)) for x, y in zip(updates, values)]
        for i in range(1, len(pts)):
            d.line((pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]), fill=color, width=3)

    pac_color = (255, 215, 0)
    ghost_color = (60, 200, 255)
    draw_series(pac, color=pac_color)
    draw_series(ghosts, color=ghost_color)

    legend_y = height - bottom + 18
    d.rectangle((left, legend_y, left + 18, legend_y + 18), fill=pac_color)
    d.text((left + 26, legend_y + 1), "pacman_reward_mean", fill=(230, 230, 230))

    x2 = left + 300
    d.rectangle((x2, legend_y, x2 + 18, legend_y + 18), fill=ghost_color)
    d.text((x2 + 26, legend_y + 1), "ghosts_reward_mean", fill=(230, 230, 230))

    d.text((left, 8), "Rewards", fill=(240, 240, 240))

    img.save(path, format="PNG")
