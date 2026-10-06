"""A* search on a 4-connected bool passability mask (Manhattan heuristic).

Ties on ``f`` are broken in favour of the larger ``g`` (deeper nodes first, which avoids
expanding whole plateaus of equal ``f``), then — with ``tie_break="diagonal"`` — by the D13
``diag`` key of AND-OR's ``ORDERED_ACTIONS`` (smaller first: stay near the straight line to the
goal), then by insertion order. Stale heap entries are skipped lazily. Neighbours are generated
in the fixed order N, E, S, W. The tie-break never changes the heuristic, so A* stays optimal.
The match with AND-OR's ordering is only approximate: ``f`` and ``g`` still come before
``diag``, and the path follows each node's first cheapest generator, whereas AND-OR orders
forward by BFS distance on the belief, then ``diag``; near obstacles the two can pick different
equally short routes (D13; H7 reports how often matched runs still differ).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from rescue.config import ASTAR_TIE_BREAKS, AStarTieBreak
from rescue.grid import Pos, manhattan, neighbours4

#: Cost of one move on the grid.
STEP_COST = 1


@dataclass(frozen=True)
class AStarResult:
    """Path from start to goal (both inclusive, ``None`` if unreachable) and search effort."""

    path: list[Pos] | None
    nodes_expanded: int


def astar(
    passable: np.ndarray, start: Pos, goal: Pos, tie_break: AStarTieBreak = "g"
) -> AStarResult:
    """A* from ``start`` to ``goal``.

    ``nodes_expanded`` counts nodes popped from the open list and expanded (neighbours
    generated); popping the goal ends the search and is not counted. The heap key is
    ``(f, -g, counter)`` with ``tie_break="g"`` (the default) and ``(f, -g, diag(node),
    counter)`` with ``tie_break="diagonal"``, where ``diag(p) = |abs(goal.r - p.r) -
    abs(goal.c - p.c)|`` exactly as in AND-OR's D13 ordering (docs/plan.md D13). Returns
    ``path=None`` if ``start`` or ``goal`` is blocked or off-grid, or the goal is unreachable;
    ``start == goal`` gives ``[start]``.
    """
    if tie_break not in ASTAR_TIE_BREAKS:
        raise ValueError(f"unknown A* tie_break {tie_break!r}")
    diagonal = tie_break == "diagonal"
    goal_r, goal_c = goal
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
    def diag(p: Pos) -> int:
        # Constant 0 for "g": the key then orders exactly like (f, -g, counter).
        return abs(abs(goal_r - p[0]) - abs(goal_c - p[1])) if diagonal else 0

    counter = 0
    heap: list[tuple[int, int, int, int, Pos]] = [
        (manhattan(start, goal), 0, diag(start), counter, start)
    ]
    expanded = 0
    while heap:
        _, neg_g, _, _, p = heapq.heappop(heap)
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
                heapq.heappush(
                    heap, (g_next + manhattan(q, goal), -g_next, diag(q), counter, q)
                )
    return AStarResult(path=None, nodes_expanded=expanded)
