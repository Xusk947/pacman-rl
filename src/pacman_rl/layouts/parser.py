from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedLayout:
    name: str
    rows: list[str]
    height: int
    width: int
    pacman_spawn: tuple[int, int]
    ghost_spawns: dict[str, tuple[int, int]]


ALLOWED = {"#", " ", ".", "o", "0", "B", "P", "I", "C"}
GHOSTS = ("B", "P", "I", "C")


def _normalize_lines(text: str) -> list[str]:
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip("\n") != ""]
    if not lines:
        raise ValueError("layout is empty")
    w = max(len(line) for line in lines)
    return [line.ljust(w) for line in lines]


def parse_layout_text(text: str, *, name: str) -> ParsedLayout:
    rows = _normalize_lines(text)
    h = len(rows)
    w = len(rows[0])

    pac: tuple[int, int] | None = None
    ghosts: dict[str, tuple[int, int]] = {}
    pellets = 0

    for r, line in enumerate(rows):
        if len(line) != w:
            raise ValueError("layout must be rectangular after normalization")
        for c, ch in enumerate(line):
            if ch not in ALLOWED:
                raise ValueError(f"unknown symbol: {ch} at r={r} c={c}")
            if ch == "0":
                if pac is not None:
                    raise ValueError("layout contains more than one Pacman spawn '0'")
                pac = (r, c)
            elif ch in GHOSTS:
                if ch in ghosts:
                    raise ValueError(f"layout contains more than one ghost spawn '{ch}'")
                ghosts[ch] = (r, c)
            elif ch in (".", "o"):
                pellets += 1

    if pac is None:
        raise ValueError("layout must contain Pacman spawn '0'")
    for g in GHOSTS:
        if g not in ghosts:
            raise ValueError(f"layout must contain ghost spawn '{g}'")
    if pellets <= 0:
        raise ValueError("layout must contain at least one pellet '.' or power pellet 'o'")

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

