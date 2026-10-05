"""A* search on a 4-connected bool passability mask (Manhattan heuristic).

Ties on ``f`` are broken in favour of the larger ``g`` (deeper nodes first, which avoids
expanding whole plateaus of equal ``f``), then by insertion order. Stale heap entries are
skipped lazily. Neighbours are generated in the fixed order N, E, S, W.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from rescue.grid import Pos, manhattan, neighbours4

#: Cost of one move on the grid.
STEP_COST = 1


@dataclass(frozen=True)
class AStarResult:
    """Path from start to goal (both inclusive, ``None`` if unreachable) and search effort."""

    path: list[Pos] | None
    nodes_expanded: int


def astar(passable: np.ndarray, start: Pos, goal: Pos) -> AStarResult:
    """A* from ``start`` to ``goal``.

    ``nodes_expanded`` counts nodes popped from the open list and expanded (neighbours
    generated); popping the goal ends the search and is not counted. The heap key is
    ``(f, -g, counter)``. Returns ``path=None`` if ``start`` or ``goal`` is blocked or
    off-grid, or the goal is unreachable; ``start == goal`` gives ``[start]``.
    """
    mask = np.asarray(passable, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"passable must be 2-D, got shape {mask.shape}")
    rows, cols = int(mask.shape[0]), int(mask.shape[1])
    free: list[list[bool]] = mask.tolist()

    def is_open(p: Pos) -> bool:
        return 0 <= p[0] < rows and 0 <= p[1] < cols and free[p[0]][p[1]]

    if not (is_open(start) and is_open(goal)):
        return AStarResult(path=None, nodes_expanded=0)

    g_score: dict[Pos, int] = {start: 0}
    parent: dict[Pos, Pos | None] = {start: None}
    closed: set[Pos] = set()
    counter = 0
    heap: list[tuple[int, int, int, Pos]] = [(manhattan(start, goal), 0, counter, start)]
    expanded = 0
    while heap:
        _, neg_g, _, p = heapq.heappop(heap)
        g = -neg_g
        if p in closed or g > g_score[p]:
            continue  # stale entry
        if p == goal:
            path: list[Pos] = []
            node: Pos | None = p
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return AStarResult(path=path, nodes_expanded=expanded)
        closed.add(p)
        expanded += 1
        g_next = g + STEP_COST
        for q in neighbours4(p, rows, cols):
            if not free[q[0]][q[1]] or q in closed:
                continue
            if g_next < g_score.get(q, g_next + 1):
                g_score[q] = g_next
                parent[q] = p
                counter += 1
                heapq.heappush(heap, (g_next + manhattan(q, goal), -g_next, counter, q))
    return AStarResult(path=None, nodes_expanded=expanded)
