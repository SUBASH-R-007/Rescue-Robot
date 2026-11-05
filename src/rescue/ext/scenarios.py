"""Multi-survivor scenarios (brief §8.5, docs/plan.md §3.14).

:func:`multi_survivor_grid` takes a core scenario map (:func:`rescue.mapgen.generate`, so the
start, the primary survivor and every obstacle are exactly the core map's) and adds ``k − 1``
extra survivors on free cells reachable from the start on the true map.
"""

from __future__ import annotations

import numpy as np

from rescue.config import EXT_MIN_SURVIVOR_SPACING, EXT_SCENARIO_STREAM_KEY
from rescue.grid import Cell, Pos, TrueGrid, manhattan
from rescue.planners.bfs import UNREACHABLE, bfs_distances

__all__ = ["add_survivors", "multi_survivor_grid"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), EXT_SCENARIO_STREAM_KEY]))


def add_survivors(grid: TrueGrid, seed: int, k: int) -> TrueGrid:
    """A copy of ``grid`` with survivors added until it has ``k`` (``grid`` is not changed).

    Candidates are ``OPEN``, non-high-slip cells reachable from the start on the true map,
    other than the start and the existing survivors, visited in the order of one seeded
    permutation (``SeedSequence([seed, EXT_SCENARIO_STREAM_KEY])``). Each new survivor is the
    first candidate at Manhattan distance ``>= EXT_MIN_SURVIVOR_SPACING`` from every survivor
    placed so far; if there is none, the spacing is relaxed one step at a time (down to 1).
    Because survivor ``i`` depends only on survivors ``1..i−1``, the survivors for ``k`` are a
    prefix of those for ``k + 1`` (same base map and seed), which keeps conditions paired.
    ``ValueError`` if ``k < 1``, ``k`` is below the grid's survivor count, or there are not
    enough candidate cells.
    """
    k = int(k)
    have = list(grid.survivors)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k < len(have):
        raise ValueError(f"the grid already has {len(have)} survivors (> k = {k})")
    if k == len(have):
        return grid.copy()
    dist = bfs_distances(grid.passable_mask(), grid.start)
    free = (grid.cells == Cell.OPEN) & ~grid.high_slip & (dist != UNREACHABLE)
    free[grid.start] = False
    order = _rng(seed).permutation(grid.rows * grid.cols).tolist()
    candidates: list[Pos] = [
        divmod(index, grid.cols) for index in order if free.flat[index]
    ]
    if len(candidates) < k - len(have):
        raise ValueError(
            f"only {len(candidates)} free reachable cells for {k - len(have)} extra survivors"
        )
    taken: set[Pos] = set()
    while len(have) < k:
        chosen: Pos | None = None
        for spacing in range(EXT_MIN_SURVIVOR_SPACING, 0, -1):
            chosen = next(
                (c for c in candidates
                 if c not in taken and all(manhattan(c, s) >= spacing for s in have)),
                None,
            )
            if chosen is not None:
                break
        if chosen is None:  # pragma: no cover - the length check above guarantees a cell
            raise ValueError("no free cell left for another survivor")
        taken.add(chosen)
        have.append(chosen)
    cells = grid.cells.copy()
    extras = tuple(have[1:])
    for p in extras[len(grid.extra_survivors):]:
        cells[p] = Cell.SURVIVOR
    return TrueGrid(
        cells=cells,
        start=grid.start,
        survivor=grid.survivor,
        high_slip=grid.high_slip.copy(),
        name=grid.name,
        extra_survivors=extras,
    )


def multi_survivor_grid(scenario: str, rows: int, cols: int, seed: int, k: int) -> TrueGrid:
    """Core scenario ``scenario`` (``mapgen.generate(scenario, rows, cols, seed)``) with ``k``
    survivors in total: the core survivor plus ``k − 1`` placed by :func:`add_survivors`
    (pairwise Manhattan ``>= EXT_MIN_SURVIVOR_SPACING``, reachable, never on the start).

    Deterministic in the arguments; ``k = 1`` returns the core map unchanged.
    """
    # Imported here, not at module import: mapgen needs scipy, which the browser app
    # (rescue.webapi, docs/plan.md §3.16) loads only on demand; add_survivors does not need it.
    from rescue.mapgen import generate

    base = generate(scenario, rows, cols, seed)
    if int(k) == 1:
        return base
    return add_survivors(base, seed, k)
