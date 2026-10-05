"""Seeded random true maps for the Phase 3 agent / runner / invariant tests.

Independent of ``rescue.mapgen`` (Phase 5) on purpose: the Phase 3 tests only need a cheap,
seeded generator of solvable rubble maps, optionally with high-slip ('~') terrain.
"""

from __future__ import annotations

import numpy as np

from rescue.grid import Cell, Pos, TrueGrid, manhattan
from rescue.planners.bfs import reachable

#: Regeneration attempts before :func:`random_solvable_grid` gives up.
MAX_ATTEMPTS: int = 500
#: Start and survivor are at least this fraction of the grid's Manhattan diameter apart.
MIN_SEPARATION_FRACTION: float = 0.5


def _far_pair(rng: np.random.Generator, rows: int, cols: int) -> tuple[Pos, Pos]:
    """Distinct cells at least half the grid diameter apart (Manhattan)."""
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    start = cells[int(rng.integers(len(cells)))]
    diameter = rows + cols - 2
    need = max(1, int(np.ceil(MIN_SEPARATION_FRACTION * diameter)))
    far = [p for p in cells if manhattan(p, start) >= need]
    if not far:  # tiny grids: any other cell will do
        far = [p for p in cells if p != start]
    survivor = far[int(rng.integers(len(far)))]
    return start, survivor


def random_solvable_grid(
    rows: int,
    cols: int,
    density: float,
    seed: int,
    terrain_fraction: float = 0.0,
) -> TrueGrid:
    """Seeded random rubble map whose survivor is BFS-reachable from the start.

    Each cell is debris with probability ``density``; start and survivor are distinct free
    cells far apart; with ``terrain_fraction > 0`` that fraction (in expectation) of the other
    open cells is high-slip terrain ('~'). The map is redrawn from the same seeded generator
    until it is solvable; ``RuntimeError`` after :data:`MAX_ATTEMPTS` attempts.
    """
    if rows * cols < 2:
        raise ValueError("grid needs at least two cells")
    rng = np.random.default_rng([int(seed), rows, cols])
    for _ in range(MAX_ATTEMPTS):
        debris = rng.random((rows, cols)) < density
        slippery = rng.random((rows, cols)) < terrain_fraction
        start, survivor = _far_pair(rng, rows, cols)
        cells = np.where(debris, int(Cell.DEBRIS), int(Cell.OPEN)).astype(np.int8)
        cells[start] = Cell.OPEN
        cells[survivor] = Cell.SURVIVOR
        high_slip = slippery & (cells == Cell.OPEN)
        high_slip[start] = False
        if not reachable(cells != Cell.DEBRIS, start, survivor):
            continue
        return TrueGrid(
            cells=cells,
            start=start,
            survivor=survivor,
            high_slip=high_slip,
            name=f"rand{rows}x{cols}_s{seed}",
        )
    raise RuntimeError(
        f"no solvable {rows}x{cols} map at density {density} for seed {seed} "
        f"after {MAX_ATTEMPTS} attempts"
    )
