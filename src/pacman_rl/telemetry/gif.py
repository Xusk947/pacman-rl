from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GifRenderConfig:
    cell_px: int = 24
    fps: int = 12
    max_frames: int = 300
    stride: int = 1


def render_game_gif(
    game_json_path: Path,
    gif_path: Path,
    *,
    cfg: GifRenderConfig = GifRenderConfig(),
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Pillow is required to render game.gif") from e

    payload = json.loads(game_json_path.read_text(encoding="utf-8"))
    layout = payload["layout"]
    rows: list[str] = layout["rows"]
    h = int(layout["height"])
    w = int(layout["width"])
    frames: list[dict[str, Any]] = payload["frames"]

    walls: set[tuple[int, int]] = set()
    pellets: set[tuple[int, int]] = set()
    power: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch == "%":
                walls.add((r, c))
            elif ch == ".":
                pellets.add((r, c))
            elif ch == "o":
                power.add((r, c))

    def cell_rect(rc: tuple[int, int]) -> tuple[int, int, int, int]:
        r, c = rc
        x0 = c * cfg.cell_px
        y0 = r * cfg.cell_px
        x1 = x0 + cfg.cell_px
        y1 = y0 + cfg.cell_px
        return x0, y0, x1, y1

    def draw_circle(draw: ImageDraw.ImageDraw, rc: tuple[int, int], *, color: tuple[int, int, int], inset: int) -> None:
        x0, y0, x1, y1 = cell_rect(rc)
        draw.ellipse((x0 + inset, y0 + inset, x1 - inset, y1 - inset), fill=color)

    out_frames: list[Image.Image] = []
    used = frames[:: max(1, cfg.stride)]
    if cfg.max_frames > 0:
        used = used[: cfg.max_frames]

    ghost_colors: list[tuple[int, int, int]] = [
        (255, 0, 0),
        (255, 105, 180),
        (0, 255, 255),
        (255, 165, 0),
    ]

    for f in used:
        pac_rc = tuple(f["pac_xy"])
        pellets.discard(pac_rc)
        power.discard(pac_rc)

        img = Image.new("RGB", (w * cfg.cell_px, h * cfg.cell_px), (0, 0, 0))
        d = ImageDraw.Draw(img)

        for rc in walls:
            x0, y0, x1, y1 = cell_rect(rc)
            d.rectangle((x0, y0, x1, y1), fill=(25, 25, 160))

        for rc in pellets:
            draw_circle(d, rc, color=(245, 245, 245), inset=cfg.cell_px // 2 - 2)

        for rc in power:
            draw_circle(d, rc, color=(245, 245, 245), inset=cfg.cell_px // 2 - 6)

        draw_circle(d, pac_rc, color=(255, 215, 0), inset=3)

        ghost_xy: list[list[int]] = f["ghost_xy"]
        ghost_present: list[bool] = f["ghost_present"]
        scared: list[int] = f["scared"]

        for i, pos in enumerate(ghost_xy):
            if i >= len(ghost_colors):
                break
            if not bool(ghost_present[i]):
                continue
            rc = (int(pos[0]), int(pos[1]))
            if int(scared[i]) > 0:
                color = (60, 120, 255)
            else:
                color = ghost_colors[i]
            draw_circle(d, rc, color=color, inset=3)

        out_frames.append(img)

        if bool(f.get("done", False)):
            break

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_frames:
        Image.new("RGB", (w * cfg.cell_px, h * cfg.cell_px), (0, 0, 0)).save(gif_path, format="GIF")
        return

    duration_ms = int(1000 / max(1, cfg.fps))
    out_frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=out_frames[1:],
        duration=duration_ms,
        loop=0,
    )
