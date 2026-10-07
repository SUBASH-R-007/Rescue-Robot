"""Grid primitives: positions, cell types, neighbourhoods and the true map with ASCII I/O.

Internal coordinates are 0-indexed ``(row, col)``; anything shown to a human goes through
:func:`fmt_pos`, which prints them 1-indexed.

ASCII map format (one character per cell):

* ``.`` open, ``#`` debris, ``S`` start (open), ``P`` survivor, ``~`` open + high slip.
* Lines are stripped; blank lines and lines starting with ``;`` are ignored (comments).
* All rows have equal length; exactly one ``S`` and one ``P`` (``multi=True`` accepts one or
  more ``P``: the first in row-major order is the primary survivor, the rest are extras;
  plan §3.14).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

Pos = tuple[int, int]

_COMMENT_PREFIX = ";"
_CH_OPEN = "."
_CH_DEBRIS = "#"
_CH_START = "S"
_CH_SURVIVOR = "P"
_CH_HIGH_SLIP = "~"


class Cell(IntEnum):
    """Cell contents. ``UNKNOWN`` is 0 so ``np.zeros`` is an all-unknown belief (D1)."""

    UNKNOWN = 0
    OPEN = 1
    DEBRIS = 2
    SURVIVOR = 3


#: Values a true-map cell may hold (never UNKNOWN).
_TRUE_CELL_VALUES: tuple[int, ...] = (int(Cell.OPEN), int(Cell.DEBRIS), int(Cell.SURVIVOR))


def fmt_pos(p: Pos) -> str:
    """Format an internal 0-indexed position for humans, 1-indexed: ``(0, 0) -> "(1,1)"``."""
    return f"({p[0] + 1},{p[1] + 1})"


def manhattan(a: Pos, b: Pos) -> int:
    """Manhattan (L1) distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(p: Pos, rows: int, cols: int) -> bool:
    """True if ``p`` lies on a ``rows`` x ``cols`` grid."""
    return 0 <= p[0] < rows and 0 <= p[1] < cols


def neighbours4(p: Pos, rows: int, cols: int) -> list[Pos]:
    """In-grid 4-neighbours of ``p`` in the fixed order N, E, S, W."""
    r, c = p
    candidates = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))
    return [q for q in candidates if 0 <= q[0] < rows and 0 <= q[1] < cols]


def footprint(p: Pos, radius: int, rows: int, cols: int) -> list[Pos]:
    """In-grid cells within Manhattan distance ``radius`` of ``p`` (incl. ``p``), row-major."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    r0, c0 = p
    cells: list[Pos] = []
    for r in range(max(0, r0 - radius), min(rows, r0 + radius + 1)):
        span = radius - abs(r - r0)
        for c in range(max(0, c0 - span), min(cols, c0 + span + 1)):
            cells.append((r, c))
    return cells


@dataclass(eq=False)
class TrueGrid:
    """The true disaster-zone map used by the simulator.

    ``cells`` is an ``int8`` array holding ``OPEN`` / ``DEBRIS`` / ``SURVIVOR`` (never
    ``UNKNOWN``); ``high_slip`` is a separate bool mask of high-slip terrain (D2).
    Invariants (``ValueError`` otherwise): ``start`` is ``OPEN``, the ``SURVIVOR`` cells are
    exactly :attr:`survivors` (``survivor`` plus the optional ``extra_survivors`` of the
    multi-survivor extension, plan §3.14: distinct, in bounds, never the start), neither the
    start nor any survivor is marked high-slip, and no ``DEBRIS`` cell is marked high-slip (so
    every valid grid round-trips through ASCII exactly). The grid owns private copies of the
    arrays it is given.
    """

    cells: np.ndarray
    start: Pos
    survivor: Pos
    high_slip: np.ndarray
    name: str = ""
    #: Further survivors (multi-survivor extension, plan §3.14); empty for core maps.
    extra_survivors: tuple[Pos, ...] = ()

    def __post_init__(self) -> None:
        raw_cells = np.asarray(self.cells)
        if raw_cells.ndim != 2 or raw_cells.size == 0:
            raise ValueError(f"cells must be a non-empty 2-D array, got shape {raw_cells.shape}")
        # Validate before casting: an int8 cast would silently wrap out-of-range values.
        if not np.isin(raw_cells, _TRUE_CELL_VALUES).all():
            raise ValueError("true grid cells must be OPEN, DEBRIS or SURVIVOR")
        # np.array (not asarray) copies, so the caller's buffers never alias the grid.
        self.cells = np.array(raw_cells, dtype=np.int8)
        self.high_slip = np.array(self.high_slip, dtype=bool)
        self.start = (int(self.start[0]), int(self.start[1]))
        self.survivor = (int(self.survivor[0]), int(self.survivor[1]))
        self.extra_survivors = tuple((int(q[0]), int(q[1])) for q in self.extra_survivors)
        if self.high_slip.shape != self.cells.shape:
            raise ValueError(
                f"high_slip shape {self.high_slip.shape} != cells shape {self.cells.shape}"
            )
        for label, p in (("start", self.start), ("survivor", self.survivor)):
            if not self.in_bounds(p):
                raise ValueError(f"{label} {p} is outside the {self.rows}x{self.cols} grid")
        for p in self.extra_survivors:
            if not self.in_bounds(p):
                raise ValueError(
                    f"extra survivor {p} is outside the {self.rows}x{self.cols} grid"
                )
        if self.start == self.survivor:
            raise ValueError("start and survivor must differ")
        if self.start in self.extra_survivors:
            raise ValueError(f"extra survivor {fmt_pos(self.start)} cannot be the start")
        if len(set(self.survivors)) != len(self.survivors):
            raise ValueError("survivor positions must be distinct")
        if self.cells[self.start] != Cell.OPEN:
            raise ValueError(f"start {fmt_pos(self.start)} must be an OPEN cell")
        if self.cells[self.survivor] != Cell.SURVIVOR:
            raise ValueError(f"survivor {fmt_pos(self.survivor)} must be a SURVIVOR cell")
        for p in self.extra_survivors:
            if self.cells[p] != Cell.SURVIVOR:
                raise ValueError(f"extra survivor {fmt_pos(p)} must be a SURVIVOR cell")
        if int(np.count_nonzero(self.cells == Cell.SURVIVOR)) != len(self.survivors):
            if not self.extra_survivors:
                raise ValueError("the grid must contain exactly one SURVIVOR cell")
            raise ValueError(
                f"the grid must contain exactly {len(self.survivors)} SURVIVOR cells "
                "(survivor + extra_survivors)"
            )
        # The ASCII format writes these cells as 'S' / 'P', which cannot carry the '~' flag;
        # rejecting it keeps from_ascii(to_ascii(g)) exact for every valid grid.
        for label, p in (("start", self.start), *(("survivor", q) for q in self.survivors)):
            if self.high_slip[p]:
                raise ValueError(f"{label} {fmt_pos(p)} cannot be high-slip terrain")
        # Debris is written as '#', which cannot carry the '~' flag either.
        slippery_debris = np.argwhere(self.high_slip & (self.cells == Cell.DEBRIS))
        if slippery_debris.size:
            first = (int(slippery_debris[0][0]), int(slippery_debris[0][1]))
            raise ValueError(f"debris cell {fmt_pos(first)} cannot be high-slip terrain")

    # --- shape -----------------------------------------------------------------------------
    @property
    def rows(self) -> int:
        """Number of rows."""
        return int(self.cells.shape[0])

    @property
    def cols(self) -> int:
        """Number of columns."""
        return int(self.cells.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        return (self.rows, self.cols)

    @property
    def survivors(self) -> tuple[Pos, ...]:
        """Every survivor cell: ``(survivor, *extra_survivors)`` (plan §3.14)."""
        return (self.survivor, *self.extra_survivors)

    # --- queries ---------------------------------------------------------------------------
    def in_bounds(self, p: Pos) -> bool:
        """True if ``p`` lies on this grid."""
        return in_bounds(p, self.rows, self.cols)

    def is_debris(self, p: Pos) -> bool:
        """True if ``p`` is an in-grid debris cell (False off-grid)."""
        return self.in_bounds(p) and bool(self.cells[p] == Cell.DEBRIS)

    def passable_mask(self) -> np.ndarray:
        """Bool array, True where the cell is not debris."""
        return self.cells != Cell.DEBRIS

    def copy(self) -> TrueGrid:
        """Deep copy."""
        return TrueGrid(
            cells=self.cells.copy(),
            start=self.start,
            survivor=self.survivor,
            high_slip=self.high_slip.copy(),
            name=self.name,
            extra_survivors=self.extra_survivors,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrueGrid):
            return NotImplemented
        return (
            self.start == other.start
            and self.survivor == other.survivor
            and self.extra_survivors == other.extra_survivors
            and self.name == other.name
            and np.array_equal(self.cells, other.cells)
            and np.array_equal(self.high_slip, other.high_slip)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    # --- mutation --------------------------------------------------------------------------
    def add_debris(self, p: Pos) -> None:
        """Turn cell ``p`` into debris (used by aftershocks).

        Clears any high-slip flag on ``p`` (debris is not terrain, and keeping the flag would
        break the ASCII round trip). Raises ``ValueError`` if ``p`` is off-grid, the start or
        any survivor cell. Other constraints (robot cell, connectivity) are the world's
        responsibility.
        """
        if not self.in_bounds(p):
            raise ValueError(f"cannot add debris off-grid at {p}")
        if p == self.start or p in self.survivors:
            raise ValueError(f"cannot add debris on the start/survivor cell {fmt_pos(p)}")
        self.cells[p] = Cell.DEBRIS
        self.high_slip[p] = False

    # --- ASCII I/O -------------------------------------------------------------------------
    def to_ascii(self) -> str:
        """Render the map in the ASCII format (one line per row, trailing newline).

        The start and every survivor are written as ``S`` / ``P`` (a valid grid never marks
        them high-slip), so ``from_ascii(to_ascii(g))`` round-trips exactly (pass
        ``multi=True`` when the grid has extra survivors).
        """
        lines: list[str] = []
        for r in range(self.rows):
            chars: list[str] = []
            for c in range(self.cols):
                p = (r, c)
                value = self.cells[p]
                if p == self.start:
                    chars.append(_CH_START)
                elif value == Cell.SURVIVOR:
                    chars.append(_CH_SURVIVOR)
                elif value == Cell.DEBRIS:
                    chars.append(_CH_DEBRIS)
                elif self.high_slip[p]:
                    chars.append(_CH_HIGH_SLIP)
                else:
                    chars.append(_CH_OPEN)
            lines.append("".join(chars))
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the map to ``path`` in the ASCII format (UTF-8, LF line endings on every OS)."""
        Path(path).write_text(self.to_ascii(), encoding="utf-8", newline="\n")

    @classmethod
    def from_ascii(cls, text: str, name: str = "", *, multi: bool = False) -> TrueGrid:
        """Parse an ASCII map; errors raise ``ValueError`` with 1-indexed line/column.

        Line numbers count every source line (including comments and blank lines); columns
        count characters from the start of the source line (including leading whitespace).
        With ``multi=False`` (the default) the map needs exactly one ``P``; with ``multi=True``
        at least one: the first in row-major order becomes ``survivor`` and the rest
        ``extra_survivors`` (plan §3.14).
        """
        rows_chars: list[str] = []
        cells_rows: list[list[int]] = []
        slip_rows: list[list[bool]] = []
        starts: list[tuple[Pos, int]] = []
        survivors: list[tuple[Pos, int]] = []
        width: int | None = None
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith(_COMMENT_PREFIX):
                continue
            # Leading whitespace is stripped but still counts towards reported columns.
            col_offset = len(raw) - len(raw.lstrip())
            if width is None:
                width = len(line)
            elif len(line) != width:
                raise ValueError(
                    f"line {line_no}: row has {len(line)} cells, expected {width}"
                )
            r = len(rows_chars)
            row_cells: list[int] = []
            row_slip: list[bool] = []
            for col_no, ch in enumerate(line, start=1):
                slip = False
                if ch in (_CH_OPEN, _CH_START):
                    value = Cell.OPEN
                elif ch == _CH_DEBRIS:
                    value = Cell.DEBRIS
                elif ch == _CH_SURVIVOR:
                    value = Cell.SURVIVOR
                elif ch == _CH_HIGH_SLIP:
                    value, slip = Cell.OPEN, True
                else:
                    raise ValueError(
                        f"line {line_no}, col {col_offset + col_no}: unknown map character {ch!r}"
                    )
                if ch == _CH_START:
                    starts.append(((r, col_no - 1), line_no))
                elif ch == _CH_SURVIVOR:
                    survivors.append(((r, col_no - 1), line_no))
                row_cells.append(int(value))
                row_slip.append(slip)
            rows_chars.append(line)
            cells_rows.append(row_cells)
            slip_rows.append(row_slip)
        if not rows_chars:
            raise ValueError("map contains no rows")
        checks: list[tuple[str, str, list[tuple[Pos, int]]]] = [("start", _CH_START, starts)]
        if not multi:
            checks.append(("survivor", _CH_SURVIVOR, survivors))
        elif not survivors:
            raise ValueError(f"map needs at least one survivor {_CH_SURVIVOR!r} (none found)")
        for label, ch, found in checks:
            if len(found) != 1:
                where = ", ".join(f"line {ln}" for _, ln in found) or "none found"
                raise ValueError(f"map needs exactly one {label} {ch!r} ({where})")
        return cls(
            cells=np.array(cells_rows, dtype=np.int8),
            start=starts[0][0],
            survivor=survivors[0][0],
            high_slip=np.array(slip_rows, dtype=bool),
            name=name,
            extra_survivors=tuple(p for p, _ in survivors[1:]),
        )

    @classmethod
    def load(cls, path: str | Path, *, multi: bool = False) -> TrueGrid:
        """Load an ASCII map file; the grid is named after the file stem.

        ``multi`` is passed to :meth:`from_ascii` (``True`` accepts several ``P``).
        """
        p = Path(path)
        return cls.from_ascii(p.read_text(encoding="utf-8"), name=p.stem, multi=multi)
