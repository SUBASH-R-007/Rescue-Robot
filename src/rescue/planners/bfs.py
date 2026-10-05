"""Breadth-first search on a 4-connected bool passability mask.

Neighbours are expanded in the fixed order N, E, S, W so results are deterministic.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rescue.grid import Pos, neighbours4

#: Distance value for blocked or unreachable cells.
UNREACHABLE: int = -1


def _as_lists(passable: np.ndarray) -> tuple[list[list[bool]], int, int]:
    mask = np.asarray(passable, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"passable must be 2-D, got shape {mask.shape}")
    return mask.tolist(), int(mask.shape[0]), int(mask.shape[1])


def _open(free: list[list[bool]], p: Pos, rows: int, cols: int) -> bool:
    return 0 <= p[0] < rows and 0 <= p[1] < cols and free[p[0]][p[1]]


def bfs_distances(passable: np.ndarray, goal: Pos) -> np.ndarray:
    """Step distance from every cell to ``goal`` (``int32``; ``UNREACHABLE`` if none)."""
    free, rows, cols = _as_lists(passable)
    dist = [[UNREACHABLE] * cols for _ in range(rows)]
    if _open(free, goal, rows, cols):
        dist[goal[0]][goal[1]] = 0
        queue: deque[Pos] = deque([goal])
        while queue:
            p = queue.popleft()
            d = dist[p[0]][p[1]] + 1
            for q in neighbours4(p, rows, cols):
                if free[q[0]][q[1]] and dist[q[0]][q[1]] == UNREACHABLE:
                    dist[q[0]][q[1]] = d
                    queue.append(q)
    return np.array(dist, dtype=np.int32)


def _search(passable: np.ndarray, start: Pos, goal: Pos) -> dict[Pos, Pos | None] | None:
    """BFS from ``start``; returns the parent map if ``goal`` was reached, else ``None``."""
    free, rows, cols = _as_lists(passable)
    if not (_open(free, start, rows, cols) and _open(free, goal, rows, cols)):
        return None
    parent: dict[Pos, Pos | None] = {start: None}
    if start == goal:
        return parent
    queue: deque[Pos] = deque([start])
    while queue:
        p = queue.popleft()
        for q in neighbours4(p, rows, cols):
            if free[q[0]][q[1]] and q not in parent:
                parent[q] = p
                if q == goal:
                    return parent
                queue.append(q)
    return None


def shortest_path(passable: np.ndarray, start: Pos, goal: Pos) -> list[Pos] | None:
    """A shortest 4-connected path from ``start`` to ``goal`` (both inclusive), or ``None``."""
    parent = _search(passable, start, goal)
    if parent is None:
        return None
    path: list[Pos] = []
    node: Pos | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def shortest_path_length(passable: np.ndarray, start: Pos, goal: Pos) -> int | None:
    """Number of steps on a shortest path, or ``None`` if ``goal`` is unreachable."""
    path = shortest_path(passable, start, goal)
    return None if path is None else len(path) - 1


def reachable(passable: np.ndarray, start: Pos, goal: Pos) -> bool:
    """True if ``goal`` can be reached from ``start`` (stops as soon as it is found)."""
    return _search(passable, start, goal) is not None
