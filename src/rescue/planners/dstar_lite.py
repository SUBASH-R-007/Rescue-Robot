"""D* Lite (Koenig & Likhachev 2002, optimized version) on the 4-connected belief grid.

The search runs *backwards*, from the goal (survivor) to the robot, so that the robot's
moves only change the heuristic, not the search tree. ``g(s)`` estimates the distance from
``s`` to the goal and ``rhs(s) = min_{s'} c(s, s') + g(s')`` is its one-step lookahead; a
vertex is *inconsistent* when they differ and then sits in the priority queue ``U`` with key

    k(s) = [min(g, rhs) + h(s_start, s) + km,  min(g, rhs)]

``h`` is the Manhattan distance (consistent for unit costs) and ``km`` accumulates
``h(s_last, s_start)`` every time the robot moved before new keys are computed, so the keys
already in ``U`` stay lower bounds and need not be recomputed (the "optimized" version).

Edge costs follow the agent's optimistic belief: ``c(u, v) = 1`` when both cells are passable
(unknown counts as open), ``inf`` otherwise. When the passability of a cell changes (newly
sensed debris, or the reverse), every edge touching it changes: the cell and its four
neighbours get their ``rhs`` recomputed and are passed to ``UpdateVertex``; the next
:meth:`DStarLite.compute_shortest_path` repairs only the affected part of the search tree.

After ``ComputeShortestPath`` the robot follows a shortest path by moving greedily to the
neighbour minimising ``c + g`` (Koenig & Likhachev 2002, Theorem 5), ties broken in
``ACTIONS`` order (N, E, S, W).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from rescue.grid import Pos, fmt_pos
from rescue.model import ACTIONS
from rescue.planners.astar import STEP_COST

INF = math.inf
Key = tuple[float, float]
#: Key of an empty queue.
_EMPTY_KEY: Key = (INF, INF)


@dataclass(frozen=True)
class DStarResult:
    """Path from the robot to the goal (both inclusive; ``None`` if unreachable) and the
    vertices expanded by the ``ComputeShortestPath`` call that produced it."""

    path: list[Pos] | None
    nodes_expanded: int


class DStarLite:
    """Incremental shortest paths from a moving start to a fixed goal on a 4-connected grid.

    ``passable`` is copied; change it only through :meth:`set_passable` / :meth:`sync`, and
    move the start with :meth:`move_start`. :meth:`plan` does all three plus the search.
    """

    def __init__(self, passable: np.ndarray, start: Pos, goal: Pos) -> None:
        mask = np.array(passable, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"passable must be 2-D, got shape {mask.shape}")
        self.rows: int = int(mask.shape[0])
        self.cols: int = int(mask.shape[1])
        self._check(start)
        self._check(goal)
        self._free: list[bool] = mask.ravel().tolist()
        n = self.rows * self.cols
        self.goal: Pos = (int(goal[0]), int(goal[1]))
        self.start: Pos = (int(start[0]), int(start[1]))
        self._last: Pos = self.start
        self.km: float = 0.0
        self._g: list[float] = [INF] * n
        self._rhs: list[float] = [INF] * n
        self._open: dict[int, Key] = {}
        self._heap: list[tuple[float, float, int]] = []
        # Neighbour indices in ACTIONS order (off-grid omitted), precomputed once.
        self._nbrs: list[tuple[int, ...]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                cells = []
                for a in ACTIONS:
                    nr, nc = r + a.delta[0], c + a.delta[1]
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        cells.append(nr * self.cols + nc)
                self._nbrs.append(tuple(cells))
        self.total_expanded: int = 0
        gi = self._idx(self.goal)
        if self._free[gi]:
            self._rhs[gi] = 0.0
            self._insert(gi)

    # --- helpers ---------------------------------------------------------------------------
    def _check(self, p: Pos) -> None:
        if not (0 <= p[0] < self.rows and 0 <= p[1] < self.cols):
            raise ValueError(f"{fmt_pos(p)} is outside the {self.rows}x{self.cols} grid")

    def _idx(self, p: Pos) -> int:
        return p[0] * self.cols + p[1]

    def _pos(self, i: int) -> Pos:
        return divmod(i, self.cols)

    def _h(self, i: int) -> float:
        """Manhattan distance from the current start to vertex ``i``."""
        r, c = divmod(i, self.cols)
        return float(abs(r - self.start[0]) + abs(c - self.start[1]))

    def _cost(self, u: int, v: int) -> float:
        return STEP_COST if self._free[u] and self._free[v] else INF

    def _key(self, i: int) -> Key:
        m = min(self._g[i], self._rhs[i])
        return (m + self._h(i) + self.km, m)

    def _insert(self, i: int) -> None:
        key = self._key(i)
        self._open[i] = key
        heapq.heappush(self._heap, (key[0], key[1], i))

    def _top(self) -> tuple[Key, int]:
        """Smallest live key and its vertex (``(_EMPTY_KEY, -1)`` if the queue is empty)."""
        heap, live = self._heap, self._open
        while heap:
            k1, k2, i = heap[0]
            if live.get(i) == (k1, k2):
                return (k1, k2), i
            heapq.heappop(heap)  # stale entry (removed or re-keyed vertex)
        return _EMPTY_KEY, -1

    def _best_rhs(self, i: int) -> float:
        """``min_{s'} c(i, s') + g(s')`` (``inf`` for a blocked vertex)."""
        if not self._free[i]:
            return INF
        g, free = self._g, self._free
        best = INF
        for j in self._nbrs[i]:
            if free[j]:
                cand = STEP_COST + g[j]
                if cand < best:
                    best = cand
        return best

    def _update_vertex(self, i: int) -> None:
        """Queue ``i`` with a fresh key iff it is inconsistent (optimized UpdateVertex)."""
        if self._g[i] != self._rhs[i]:
            self._insert(i)  # insert or re-key (the old heap entry goes stale)
        else:
            self._open.pop(i, None)

    def _sync_start(self) -> None:
        """``km += h(s_last, s_start)`` once the robot has moved, before any new key."""
        if self.start != self._last:
            self.km += abs(self.start[0] - self._last[0]) + abs(self.start[1] - self._last[1])
            self._last = self.start

    # --- public API ------------------------------------------------------------------------
    def g(self, p: Pos) -> float:
        """Current ``g`` value of ``p`` (distance estimate to the goal)."""
        return self._g[self._idx(p)]

    def rhs(self, p: Pos) -> float:
        """Current ``rhs`` value of ``p``."""
        return self._rhs[self._idx(p)]

    def is_passable(self, p: Pos) -> bool:
        """Passability of ``p`` in the search's own map."""
        return self._free[self._idx(p)]

    def move_start(self, start: Pos) -> None:
        """The robot is now at ``start`` (the heuristic is re-anchored lazily via ``km``)."""
        self._check(start)
        self.start = (int(start[0]), int(start[1]))

    def set_passable(self, changes: Iterable[tuple[Pos, bool]]) -> tuple[Pos, ...]:
        """Apply passability changes; returns the cells whose passability actually changed.

        Every edge touching a changed cell changes cost, so the cell and its neighbours get
        their ``rhs`` recomputed (the goal keeps ``rhs = 0`` while passable) and are passed to
        ``UpdateVertex``.
        """
        changed: list[int] = []
        for p, free in changes:
            self._check(p)
            i = self._idx(p)
            if self._free[i] != bool(free):
                self._free[i] = bool(free)
                changed.append(i)
        if not changed:
            return ()
        self._sync_start()
        goal = self._idx(self.goal)
        touched: dict[int, None] = {}
        for i in changed:
            touched[i] = None
            for j in self._nbrs[i]:
                touched[j] = None
        for i in touched:
            if i == goal:
                self._rhs[i] = 0.0 if self._free[i] else INF
            else:
                self._rhs[i] = self._best_rhs(i)
            self._update_vertex(i)
        return tuple(self._pos(i) for i in changed)

    def sync(self, passable: np.ndarray) -> tuple[Pos, ...]:
        """Bring the search's map in line with ``passable``; returns the changed cells."""
        mask = np.asarray(passable, dtype=bool)
        if mask.shape != (self.rows, self.cols):
            raise ValueError(f"mask shape {mask.shape} != {(self.rows, self.cols)}")
        current = np.array(self._free, dtype=bool).reshape(self.rows, self.cols)
        diff = np.argwhere(current != mask)
        return self.set_passable(((int(r), int(c)), bool(mask[r, c])) for r, c in diff)

    def compute_shortest_path(self) -> int:
        """Optimized ComputeShortestPath; returns the number of vertices expanded.

        A vertex counts as expanded when it is popped and made consistent (overconsistent) or
        reset to ``inf`` (underconsistent); re-keying a stale key does not count.
        """
        self._sync_start()
        s = self._idx(self.start)
        g, rhs, nbrs, free = self._g, self._rhs, self._nbrs, self._free
        goal = self._idx(self.goal)
        expanded = 0
        while True:
            k_old, u = self._top()
            if u < 0:
                break
            if not (k_old < self._key(s) or rhs[s] > g[s]):
                break
            k_new = self._key(u)
            if k_old < k_new:
                self._insert(u)  # re-key
                continue
            expanded += 1
            if g[u] > rhs[u]:  # overconsistent: settle
                g[u] = rhs[u]
                del self._open[u]
                if free[u]:
                    cand = STEP_COST + g[u]
                    for p in nbrs[u]:
                        if p != goal and free[p] and cand < rhs[p]:
                            rhs[p] = cand
                            self._update_vertex(p)
            else:  # underconsistent: raise to inf and repair the dependants
                g_old = g[u]
                g[u] = INF
                for p in nbrs[u]:
                    if p != goal and rhs[p] == self._cost(p, u) + g_old:
                        rhs[p] = self._best_rhs(p)
                    self._update_vertex(p)
                if u != goal:
                    rhs[u] = self._best_rhs(u)
                self._update_vertex(u)
        self.total_expanded += expanded
        return expanded

    def extract_path(self) -> list[Pos] | None:
        """Greedy ``argmin c + g`` walk from the start (ties in ``ACTIONS`` order).

        ``None`` when the goal is unreachable. Raises ``RuntimeError`` if the walk fails to
        reach the goal in ``rows * cols`` steps (only possible before ``compute_shortest_path``).
        """
        s = self._idx(self.start)
        goal = self._idx(self.goal)
        # The optimized termination test leaves s_start possibly overconsistent (g > rhs):
        # rhs(s_start) is its exact distance, and the walk only reads successors' g values.
        if not self._free[s] or self._rhs[s] == INF:
            return None
        path = [s]
        g, free = self._g, self._free
        for _ in range(self.rows * self.cols):
            if s == goal:
                return [self._pos(i) for i in path]
            best, nxt = INF, -1
            for j in self._nbrs[s]:
                if free[j]:
                    cand = STEP_COST + g[j]
                    if cand < best:
                        best, nxt = cand, j
            if nxt < 0 or best == INF:
                return None
            s = nxt
            path.append(s)
        if s == goal:
            return [self._pos(i) for i in path]
        raise RuntimeError("D* Lite path extraction did not reach the goal (stale g values)")

    def plan(self, passable: np.ndarray, start: Pos) -> DStarResult:
        """Move the start, apply map changes, repair the search and extract the path."""
        self.move_start(start)
        self.sync(passable)
        expanded = self.compute_shortest_path()
        return DStarResult(path=self.extract_path(), nodes_expanded=expanded)
