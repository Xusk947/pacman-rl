from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParsedLayout:
    name: str
    rows: list[str]
    height: int
    width: int
    pacman_spawn: tuple[int, int]
    ghost_spawns: list[tuple[int, int]]


def _normalize_lines(text: str) -> list[str]:
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip("\n") != ""]
    if not lines:
        raise ValueError("layout is empty")
    width = max(len(line) for line in lines)
    return [line.ljust(width) for line in lines]


def parse_layout_text(text: str, *, name: str) -> ParsedLayout:
    rows = _normalize_lines(text)
    h = len(rows)
    w = len(rows[0])

    pac = None
    ghosts: list[tuple[int, int]] = []
    for r, line in enumerate(rows):
        if len(line) != w:
            raise ValueError("layout must be rectangular after normalization")
        for c, ch in enumerate(line):
            if ch == "P":
                if pac is not None:
                    raise ValueError("layout contains more than one Pacman spawn")
                pac = (r, c)
            elif ch == "G":
                ghosts.append((r, c))

    if pac is None:
        raise ValueError("layout must contain a Pacman spawn 'P'")
    if not ghosts:
        raise ValueError("layout must contain at least one ghost spawn 'G'")

    return ParsedLayout(
        name=name,
        rows=rows,
        height=h,
        width=w,
        pacman_spawn=pac,
        ghost_spawns=ghosts,
    )


def load_layout_file(path: Path) -> ParsedLayout:
    return parse_layout_text(path.read_text(encoding="utf-8"), name=path.stem)


def load_layouts_from_dir(path: Path) -> list[ParsedLayout]:
    files = sorted([p for p in path.glob("*.txt") if p.is_file()])
    if not files:
        raise ValueError(f"no .txt layouts found in {path}")
    return [load_layout_file(p) for p in files]


def group_layouts_by_size(layouts: Iterable[ParsedLayout]) -> dict[tuple[int, int], list[ParsedLayout]]:
    groups: dict[tuple[int, int], list[ParsedLayout]] = {}
    for lay in layouts:
        key = (lay.height, lay.width)
        groups.setdefault(key, []).append(lay)
    return groups
