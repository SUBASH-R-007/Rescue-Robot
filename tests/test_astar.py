"""Tests for rescue.planners.astar (docs/plan.md section 3.7): optimality against BFS."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pytest

from rescue.grid import Pos, manhattan
from rescue.planners.astar import AStarResult, astar
from rescue.planners.bfs import shortest_path_length
from tests.helpers_phase2 import random_instance

N_RANDOM_MAPS = 240
MAX_SIDE = 15


def _check_path(path: list[Pos], passable: np.ndarray, start: Pos, goal: Pos) -> None:
    assert path[0] == start
    assert path[-1] == goal
    for a, b in pairwise(path):
        assert manhattan(a, b) == 1
    for r, c in path:
        assert 0 <= r < passable.shape[0]
        assert 0 <= c < passable.shape[1]
        assert passable[r, c]
    assert len(set(path)) == len(path)


def test_result_is_frozen() -> None:
    res = astar(np.ones((2, 2), dtype=bool), (0, 0), (1, 1))
    assert isinstance(res, AStarResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.nodes_expanded = 0  # type: ignore[misc]


def test_open_grid_manhattan_length() -> None:
    passable = np.ones((6, 7), dtype=bool)
    res = astar(passable, (0, 0), (5, 6))
    assert res.path is not None
    assert len(res.path) - 1 == 11
    _check_path(res.path, passable, (0, 0), (5, 6))


def test_matches_bfs_length_on_random_maps() -> None:
    rng = np.random.default_rng(20240607)
    solved = 0
    for _ in range(N_RANDOM_MAPS):
        rows = int(rng.integers(1, MAX_SIDE + 1))
        cols = int(rng.integers(2, MAX_SIDE + 1))
        density = float(rng.uniform(0.1, 0.4))
        passable, start, goal = random_instance(rng, rows, cols, density)
        res = astar(passable, start, goal)
        want = shortest_path_length(passable, start, goal)
        free = int(passable.sum())
        assert 0 <= res.nodes_expanded <= free
        if want is None:
            assert res.path is None
            continue
        solved += 1
        assert res.path is not None
        assert len(res.path) - 1 == want
        _check_path(res.path, passable, start, goal)
    assert solved >= N_RANDOM_MAPS // 4


def test_unreachable_returns_none() -> None:
    passable = np.ones((5, 5), dtype=bool)
    passable[2, :] = False
    res = astar(passable, (0, 0), (4, 4))
    assert res.path is None
    assert res.nodes_expanded <= 10


def test_goal_enclosed_returns_none() -> None:
    passable = np.ones((4, 4), dtype=bool)
    passable[2, 3] = passable[3, 2] = False
    assert astar(passable, (0, 0), (3, 3)).path is None


def test_start_equals_goal() -> None:
    passable = np.ones((3, 3), dtype=bool)
    res = astar(passable, (1, 1), (1, 1))
    assert res.path == [(1, 1)]
    assert res.nodes_expanded == 0  # popping the goal ends the search uncounted (§3.7)


ROWS, COLS = 4, 5


@pytest.mark.parametrize(
    ("start", "goal", "blocked"),
    [
        ((0, 0), (3, 4), (0, 0)),  # blocked start
        ((0, 0), (3, 4), (3, 4)),  # blocked goal
        ((-1, 0), (3, 4), None),  # start off-grid (row)
        ((0, COLS), (3, 4), None),  # start off-grid (column)
        ((0, 0), (ROWS, 0), None),  # goal off-grid (row)
        ((0, 0), (0, -1), None),  # goal off-grid (column)
    ],
    ids=["blocked-start", "blocked-goal", "start-row", "start-col", "goal-row", "goal-col"],
)
def test_blocked_or_offgrid_endpoints(start: Pos, goal: Pos, blocked: Pos | None) -> None:
    passable = np.ones((ROWS, COLS), dtype=bool)
    if blocked is not None:
        passable[blocked] = False
    assert astar(passable, start, goal) == AStarResult(path=None, nodes_expanded=0)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4), ()], ids=["1-D", "3-D", "0-D"])
def test_non_2d_mask_raises(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        astar(np.ones(shape, dtype=bool), (0, 0), (0, 1))


def test_deterministic_tie_breaking() -> None:
    rng = np.random.default_rng(5)
    for _ in range(30):
        passable, start, goal = random_instance(rng, 9, 9, 0.2)
        first = astar(passable, start, goal)
        for _ in range(3):
            again = astar(passable.copy(), start, goal)
            assert again.path == first.path
            assert again.nodes_expanded == first.nodes_expanded


def test_open_grid_expands_only_the_corridor() -> None:
    # Larger-g tie-breaking on an open grid heads straight for the goal: every expanded node
    # lies on an optimal path, so at most rows + cols - 1 nodes are expanded (and the path).
    passable = np.ones((10, 10), dtype=bool)
    res = astar(passable, (0, 0), (9, 9))
    assert res.path is not None
    assert len(res.path) == 19
    assert res.nodes_expanded <= 19


def test_debris_avoided() -> None:
    passable = np.ones((5, 5), dtype=bool)
    for r in range(4):
        passable[r, 2] = False
    res = astar(passable, (0, 0), (0, 4))
    assert res.path is not None
    assert len(res.path) - 1 == shortest_path_length(passable, (0, 0), (0, 4)) == 12
    _check_path(res.path, passable, (0, 0), (0, 4))
