"""The agent's belief map: what it knows about each cell, updated from percepts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from rescue.grid import Cell, Pos, in_bounds


@dataclass(frozen=True)
class BeliefUpdate:
    """Result of one sensing update."""

    #: Cells that were ``UNKNOWN`` before this update, in observation order.
    newly_known: tuple[Pos, ...]
    #: Cells sensed as ``DEBRIS`` where the belief held ``UNKNOWN`` or ``OPEN`` (D8).
    contradictions: tuple[Pos, ...]


class BeliefMap:
    """Per-cell knowledge as an ``int8`` array of :class:`Cell` values.

    Starts all ``UNKNOWN`` except the (known) survivor cell. Planning is optimistic:
    :meth:`passable_mask` treats ``UNKNOWN`` as passable.
    """

    def __init__(self, rows: int, cols: int, survivor: Pos) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got {rows}x{cols}")
        if not in_bounds(survivor, rows, cols):
            raise ValueError(f"survivor {survivor} is outside the {rows}x{cols} grid")
        self.rows: int = rows
        self.cols: int = cols
        self.survivor: Pos = (int(survivor[0]), int(survivor[1]))
        self.grid: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self.grid[self.survivor] = Cell.SURVIVOR

    def _check(self, p: Pos) -> None:
        if not in_bounds(p, self.rows, self.cols):
            raise ValueError(f"position {p} is outside the {self.rows}x{self.cols} grid")

    def get(self, p: Pos) -> Cell:
        """Believed contents of ``p``."""
        self._check(p)
        return Cell(int(self.grid[p]))

    def is_known(self, p: Pos) -> bool:
        """True if ``p`` has been observed (or is the survivor cell)."""
        self._check(p)
        return bool(self.grid[p] != Cell.UNKNOWN)

    def update(
        self, observations: Mapping[Pos, Cell] | Iterable[tuple[Pos, Cell]]
    ) -> BeliefUpdate:
        """Write observed cell values and report what changed.

        ``newly_known`` lists cells that were ``UNKNOWN``; ``contradictions`` lists cells sensed
        as ``DEBRIS`` where the belief held ``UNKNOWN`` or ``OPEN`` (D8). Observing
        ``UNKNOWN``, an invalid cell value or an off-grid cell raises ``ValueError``; the update
        is all-or-nothing, so the belief is left unchanged when it raises.
        """
        items = observations.items() if isinstance(observations, Mapping) else observations
        # Validate every observation before writing any of them.
        checked: list[tuple[Pos, Cell]] = []
        for raw_pos, raw_cell in items:
            p = (int(raw_pos[0]), int(raw_pos[1]))
            self._check(p)
            cell = Cell(raw_cell)
            if cell is Cell.UNKNOWN:
                raise ValueError(f"cannot observe UNKNOWN at {p}")
            checked.append((p, cell))
        newly_known: list[Pos] = []
        contradictions: list[Pos] = []
        for p, cell in checked:
            old = Cell(int(self.grid[p]))
            if old is Cell.UNKNOWN:
                newly_known.append(p)
            if cell is Cell.DEBRIS and old in (Cell.UNKNOWN, Cell.OPEN):
                contradictions.append(p)
            self.grid[p] = cell
        return BeliefUpdate(tuple(newly_known), tuple(contradictions))

    def passable_mask(self) -> np.ndarray:
        """Optimistic passability: True wherever the belief is not ``DEBRIS``."""
        return self.grid != Cell.DEBRIS

    def explored_count(self) -> int:
        """Number of non-``UNKNOWN`` cells (the survivor cell counts from the start)."""
        return int(np.count_nonzero(self.grid != Cell.UNKNOWN))

    def copy(self) -> BeliefMap:
        """Independent deep copy."""
        other = BeliefMap(self.rows, self.cols, self.survivor)
        other.grid = self.grid.copy()
        return other

    def __repr__(self) -> str:
        return (
            f"BeliefMap({self.rows}x{self.cols}, survivor={self.survivor}, "
            f"explored={self.explored_count()})"
        )
