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


# --- tie-break (PlannerConfig.astar_tie_break, docs/plan.md D13 / section 3.7) -----------------

#: Seeded maps of the "default A* unchanged" regression: SHA-256 of ``repr((path,
#: nodes_expanded))`` for every map, computed with the pre-``tie_break`` implementation (heap
#: key ``(f, -g, counter)``) before the parameter existed.
LEGACY_SEED = 424242
LEGACY_N_MAPS = 400
LEGACY_DIGEST = "18b9da393c7c1e5c23da95a5a63e0999910f55360cc224391483306ddca83392"
N_TIE_MAPS = 300


def _legacy_maps() -> list[tuple[np.ndarray, Pos, Pos]]:
    rng = np.random.default_rng(LEGACY_SEED)
    maps = []
    for _ in range(LEGACY_N_MAPS):
        rows, cols = int(rng.integers(2, 16)), int(rng.integers(2, 16))
        maps.append(random_instance(rng, rows, cols, float(rng.uniform(0.0, 0.35))))
    return maps


def test_default_tie_break_is_bit_identical_to_the_legacy_search() -> None:
    import hashlib

    digest = hashlib.sha256()
    for passable, start, goal in _legacy_maps():
        res = astar(passable, start, goal)
        assert astar(passable, start, goal, "g") == res  # "g" is the default
        digest.update(repr((res.path, res.nodes_expanded)).encode())
    assert digest.hexdigest() == LEGACY_DIGEST


def test_diagonal_tie_break_is_optimal_and_changes_paths() -> None:
    rng = np.random.default_rng(97)
    solved = differs = 0
    for _ in range(N_TIE_MAPS):
        rows, cols = int(rng.integers(2, MAX_SIDE + 1)), int(rng.integers(2, MAX_SIDE + 1))
        passable, start, goal = random_instance(rng, rows, cols, float(rng.uniform(0.0, 0.35)))
        diag = astar(passable, start, goal, "diagonal")
        plain = astar(passable, start, goal, "g")
        want = shortest_path_length(passable, start, goal)
        assert 0 <= diag.nodes_expanded <= int(passable.sum())
        if want is None:
            assert diag.path is None and plain.path is None
            continue
        solved += 1
        assert diag.path is not None
        assert len(diag.path) - 1 == want  # heuristic unchanged: still optimal
        _check_path(diag.path, passable, start, goal)
        differs += diag.path != plain.path
        assert astar(passable.copy(), start, goal, "diagonal") == diag  # deterministic
    assert solved >= 200
    assert differs > 0


def test_diagonal_tie_break_hugs_the_straight_line() -> None:
    # Open grid: every monotone path is optimal. "g" (FIFO among equal g, neighbours N,E,S,W)
    # runs along the top edge first (an L); "diagonal" keeps |dr - dc| <= 1 (a staircase).
    passable = np.ones((6, 6), dtype=bool)
    plain = astar(passable, (0, 0), (5, 5)).path
    diag = astar(passable, (0, 0), (5, 5), "diagonal").path
    assert plain is not None and diag is not None
    assert len(plain) == len(diag) == 11
    assert plain[:6] == [(0, c) for c in range(6)]
    assert all(abs(r - c) <= 1 for r, c in diag)


def test_unknown_tie_break_raises() -> None:
    with pytest.raises(ValueError):
        astar(np.ones((2, 2), dtype=bool), (0, 0), (1, 1), "random")  # type: ignore[arg-type]


def test_planner_config_astar_tie_break_validation() -> None:
    from rescue.config import PlannerConfig, planner_config_from_dict

    assert PlannerConfig().astar_tie_break == "g"
    assert PlannerConfig(astar_tie_break="diagonal").astar_tie_break == "diagonal"
    with pytest.raises(ValueError):
        PlannerConfig(astar_tie_break="fixed")  # type: ignore[arg-type]
    loaded = planner_config_from_dict({"astar_tie_break": "diagonal", "tie_break": "fixed"})
    assert loaded == PlannerConfig(tie_break="fixed", astar_tie_break="diagonal")
    with pytest.raises(ValueError):
        planner_config_from_dict({"astar_tie_break": "coin"})


@pytest.mark.parametrize("agent_name", ["astar_fixed", "astar_replan"])
def test_astar_agents_use_the_configured_tie_break(agent_name: str) -> None:
    from rescue.agents import make_agent
    from rescue.config import PlannerConfig, WorldConfig
    from rescue.grid import TrueGrid
    from rescue.runner import make_transition, run_mission
    from tests.conftest import open_ascii

    grid = TrueGrid.from_ascii(open_ascii(6, 6, (0, 0), (5, 5)))
    world = WorldConfig(slip_p=0.0)
    first_paths = {}
    for tie in ("g", "diagonal"):
        agent = make_agent(agent_name, make_transition(grid, world),
                           PlannerConfig(astar_tie_break=tie))
        mission = run_mission(grid, agent, world, seed=1)
        assert mission.metrics().success
        first_paths[tie] = agent.snapshots[0].path
        # Optimistic belief at t = 0 on an open map: every cell passable.
        assert list(first_paths[tie]) == astar(np.ones((6, 6), dtype=bool), (0, 0), (5, 5),
                                               tie).path
    assert first_paths["g"] != first_paths["diagonal"]


def test_dstar_lite_ignores_the_astar_tie_break() -> None:
    from rescue.agents import make_agent
    from rescue.config import PlannerConfig, WorldConfig
    from rescue.grid import TrueGrid
    from rescue.runner import make_transition, run_mission
    from tests.conftest import open_ascii

    grid = TrueGrid.from_ascii(open_ascii(6, 6, (0, 0), (5, 5)))
    world = WorldConfig(slip_p=0.2)
    traces = []
    for tie in ("g", "diagonal"):
        agent = make_agent("dstar_lite", make_transition(grid, world),
                           PlannerConfig(astar_tie_break=tie))
        mission = run_mission(grid, agent, world, seed=3)
        traces.append([(e.t, e.pos) for e in mission.events])
    assert traces[0] == traces[1]
