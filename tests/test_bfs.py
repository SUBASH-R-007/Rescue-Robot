"""Tests for rescue.planners.bfs (docs/plan.md section 3.7)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from rescue.grid import manhattan
from rescue.planners.bfs import (
    UNREACHABLE,
    bfs_distances,
    reachable,
    shortest_path,
    shortest_path_length,
)


def _open(rows: int, cols: int) -> np.ndarray:
    return np.ones((rows, cols), dtype=bool)


def _check_path(path: list[tuple[int, int]], passable: np.ndarray) -> None:
    for a, b in pairwise(path):
        assert manhattan(a, b) == 1
    for r, c in path:
        assert passable[r, c]


def test_unreachable_constant() -> None:
    assert UNREACHABLE == -1


def test_open_grid_distances_are_manhattan() -> None:
    d = bfs_distances(_open(4, 5), (1, 2))
    assert d.dtype == np.int32
    assert d.shape == (4, 5)
    for r in range(4):
        for c in range(5):
            assert d[r, c] == manhattan((r, c), (1, 2))


def test_walls_force_detour() -> None:
    passable = _open(3, 3)
    passable[0, 1] = passable[1, 1] = False
    d = bfs_distances(passable, (0, 0))
    # ( 0  -  6 )
    # ( 1  -  5 )
    # ( 2  3  4 )
    expected = np.array([[0, UNREACHABLE, 6], [1, UNREACHABLE, 5], [2, 3, 4]])
    assert np.array_equal(d, expected)


def test_walled_off_pocket_is_unreachable(small_walled) -> None:
    passable = small_walled.passable_mask()
    d = bfs_distances(passable, small_walled.survivor)
    assert d[3, 4] == UNREACHABLE
    assert d[4, 4] == UNREACHABLE
    assert d[2, 3] == UNREACHABLE  # debris cells too
    assert d[4, 0] == 0
    assert d[0, 0] == 4
    assert not reachable(passable, (3, 4), small_walled.survivor)
    assert reachable(passable, (0, 0), small_walled.survivor)
    assert shortest_path(passable, (4, 4), small_walled.survivor) is None
    assert shortest_path_length(passable, (4, 4), small_walled.survivor) is None


def test_deck_demo_distances(deck_demo) -> None:
    d = bfs_distances(deck_demo.passable_mask(), deck_demo.survivor)
    assert d[0, 0] == 6
    assert d[3, 3] == 0
    assert d[3, 1] == 8  # (4,2) must detour around both debris cells
    assert d[2, 1] == UNREACHABLE  # debris
    # optimistic t=0 belief (all open): d(2,2) in 1-indexed = 4 (D14)
    d_opt = bfs_distances(_open(4, 4), (3, 3))
    assert d_opt[1, 1] == 4
    assert d_opt[0, 0] == 6


def test_shortest_path_inclusive_endpoints() -> None:
    passable = _open(4, 4)
    passable[1, 0:3] = False
    path = shortest_path(passable, (0, 0), (3, 0))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 0)
    d = bfs_distances(passable, (3, 0))
    assert len(path) == d[0, 0] + 1
    assert shortest_path_length(passable, (0, 0), (3, 0)) == len(path) - 1 == 9
    _check_path(path, passable)


def test_shortest_path_same_cell() -> None:
    passable = _open(2, 2)
    assert shortest_path(passable, (1, 1), (1, 1)) == [(1, 1)]
    assert shortest_path_length(passable, (1, 1), (1, 1)) == 0
    assert reachable(passable, (1, 1), (1, 1))


@pytest.mark.parametrize("seed", range(10))
def test_random_maps_consistent(seed: int) -> None:
    rng = np.random.default_rng(seed)
    passable = rng.random((7, 8)) > 0.3
    passable[0, 0] = passable[6, 7] = True
    goal = (6, 7)
    d = bfs_distances(passable, goal)
    assert (d[~passable] == UNREACHABLE).all()
    for r in range(7):
        for c in range(8):
            start = (r, c)
            if not passable[r, c]:
                continue
            path = shortest_path(passable, start, goal)
            if d[r, c] == UNREACHABLE:
                assert path is None
                assert not reachable(passable, start, goal)
                assert shortest_path_length(passable, start, goal) is None
            else:
                assert path is not None
                assert path[0] == start
                assert path[-1] == goal
                assert len(path) == d[r, c] + 1
                assert shortest_path_length(passable, start, goal) == d[r, c]
                assert reachable(passable, start, goal)
                _check_path(path, passable)


def test_bfs_does_not_mutate_input() -> None:
    passable = _open(3, 3)
    passable[1, 1] = False
    before = passable.copy()
    bfs_distances(passable, (0, 0))
    shortest_path(passable, (0, 0), (2, 2))
    assert np.array_equal(passable, before)


def test_blocked_goal_or_start() -> None:
    passable = _open(3, 3)
    passable[2, 2] = False
    # blocked goal: nothing is reachable, not even the goal cell itself
    d = bfs_distances(passable, (2, 2))
    assert d.shape == (3, 3)
    assert (d == UNREACHABLE).all()
    assert shortest_path(passable, (0, 0), (2, 2)) is None
    assert shortest_path_length(passable, (0, 0), (2, 2)) is None
    assert not reachable(passable, (0, 0), (2, 2))
    # blocked start
    assert shortest_path(passable, (2, 2), (0, 0)) is None
    assert shortest_path_length(passable, (2, 2), (0, 0)) is None
    assert not reachable(passable, (2, 2), (0, 0))
    assert not reachable(passable, (2, 2), (2, 2))
    # off-grid endpoints are never reachable either
    assert (bfs_distances(_open(2, 2), (5, 5)) == UNREACHABLE).all()
    assert shortest_path(_open(2, 2), (0, 0), (2, 0)) is None
    assert not reachable(_open(2, 2), (-1, 0), (1, 1))


@pytest.mark.parametrize("bad", [np.ones(3, dtype=bool), np.ones((2, 2, 2), dtype=bool)])
def test_non_2d_mask_raises(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bfs_distances(bad, (0, 0))
    with pytest.raises(ValueError):
        shortest_path(bad, (0, 0), (0, 1))
    with pytest.raises(ValueError):
        reachable(bad, (0, 0), (0, 1))


def test_hand_computed_maze() -> None:
    """Independent oracle: distances worked out by hand on a small serpentine maze.

    ::

        G . . . .
        # # # # .
        . . . . .
        . # # # #
        . . . . .
    """
    rows = ["G....", "####.", ".....", ".####", "....."]
    passable = np.array([[ch != "#" for ch in row] for row in rows])
    X = UNREACHABLE
    expected = np.array(
        [
            [0, 1, 2, 3, 4],
            [X, X, X, X, 5],
            [10, 9, 8, 7, 6],
            [11, X, X, X, X],
            [12, 13, 14, 15, 16],
        ]
    )
    d = bfs_distances(passable, (0, 0))
    assert np.array_equal(d, expected)
    path = shortest_path(passable, (4, 4), (0, 0))
    assert path is not None
    assert len(path) == 17
    assert path[:6] == [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0), (3, 0)]
    _check_path(path, passable)
    assert shortest_path_length(passable, (2, 2), (0, 0)) == 8
