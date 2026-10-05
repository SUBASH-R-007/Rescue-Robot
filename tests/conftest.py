"""Shared fixtures for the test suite.

Imports of ``rescue`` modules happen inside fixtures so that collecting one test module
never fails because an unrelated module is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

#: Repository root, computed from this file's location (tests/conftest.py).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
#: Directory holding the ASCII map files.
MAPS_DIR: Path = REPO_ROOT / "maps"


def open_ascii(rows: int, cols: int, start: tuple[int, int], survivor: tuple[int, int]) -> str:
    """ASCII text of an all-open ``rows`` x ``cols`` grid with the given start and survivor."""
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if (r, c) == start:
                row.append("S")
            elif (r, c) == survivor:
                row.append("P")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


@pytest.fixture
def repo_root() -> Path:
    """Repository root directory."""
    return REPO_ROOT


@pytest.fixture
def maps_dir() -> Path:
    """Directory containing the bundled maps."""
    return MAPS_DIR


@pytest.fixture
def deck_demo() -> Any:
    """``maps/deck_demo.txt`` loaded as a TrueGrid."""
    from rescue.grid import TrueGrid

    return TrueGrid.load(MAPS_DIR / "deck_demo.txt")


@pytest.fixture
def grid_from() -> Callable[[str], Any]:
    """Factory: ASCII text -> TrueGrid."""
    from rescue.grid import TrueGrid

    def make(text: str) -> Any:
        return TrueGrid.from_ascii(text)

    return make


@pytest.fixture
def open_grid() -> Callable[..., Any]:
    """Factory: all-open TrueGrid of a given size with start/survivor positions."""
    from rescue.grid import TrueGrid

    def make(
        rows: int,
        cols: int,
        start: tuple[int, int] = (0, 0),
        survivor: tuple[int, int] | None = None,
    ) -> Any:
        if survivor is None:
            survivor = (rows - 1, cols - 1)
        return TrueGrid.from_ascii(open_ascii(rows, cols, start, survivor))

    return make


@pytest.fixture
def small_walled() -> Any:
    """5x5 grid where the bottom-right 2x2 pocket is walled off from the start.

    ::

        S....
        .....
        ...##
        ...#.
        P..#.
    """
    from rescue.grid import TrueGrid

    return TrueGrid.from_ascii("S....\n.....\n...##\n...#.\nP..#.")
