"""Tests for rescue.world (docs/plan.md section 3.5, decisions D3, D6, D7, D9)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import chisquare

from rescue.belief import BeliefMap
from rescue.config import AftershockConfig
from rescue.grid import Cell, TrueGrid, footprint, manhattan
from rescue.model import (
    ACTIONS,
    OUTCOME_ORDER,
    Action,
    OutcomeKind,
    TerrainSlip,
    UniformSlip,
    step_pos,
)
from rescue.planners.bfs import reachable
from rescue.world import Percept, RngStreams, StepResult, World
from tests.conftest import open_ascii

A, SL, KL, KR = (
    OutcomeKind.ARRIVED,
    OutcomeKind.SLIPPED,
    OutcomeKind.SKID_L,
    OutcomeKind.SKID_R,
)
P_VALUE_FLOOR = 1e-3
N_WORLD_SAMPLES = 100_000


# --- helpers -----------------------------------------------------------------------------


def _random_grid(rows: int, cols: int, density: float, seed: int) -> TrueGrid:
    """Seeded random rubble map whose survivor is reachable from the start."""
    for attempt in range(100):
        rng = np.random.default_rng([seed, attempt])
        debris = rng.random((rows, cols)) < density
        start, survivor = (0, 0), (rows - 1, cols - 1)
        lines = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if (r, c) == start:
                    row.append("S")
                elif (r, c) == survivor:
                    row.append("P")
                else:
                    row.append("#" if debris[r, c] else ".")
            lines.append("".join(row))
        grid = TrueGrid.from_ascii("\n".join(lines))
        if reachable(grid.passable_mask(), start, survivor):
            return grid
    raise AssertionError("could not build a solvable grid")


def _steer(pos: tuple[int, int], centre: tuple[int, int], t: int) -> Action:
    """Action that moves the robot back towards ``centre`` (cycles when already there)."""
    r, c = pos
    if r > centre[0]:
        return Action.N
    if r < centre[0]:
        return Action.S
    if c > centre[1]:
        return Action.W
    if c < centre[1]:
        return Action.E
    return ACTIONS[t % len(ACTIONS)]


def _expected_new_pos(prev: tuple[int, int], action: Action, kind: OutcomeKind):
    if kind is A:
        return step_pos(prev, action)
    if kind is SL:
        return prev
    if kind is KL:
        return step_pos(prev, action.left)
    return step_pos(prev, action.right)


def _debris_set(grid: TrueGrid) -> set[tuple[int, int]]:
    rs, cs = np.nonzero(grid.cells == Cell.DEBRIS)
    return {(int(r), int(c)) for r, c in zip(rs, cs, strict=True)}


# --- construction and sensing ------------------------------------------------------------


def test_world_initial_state(deck_demo: TrueGrid) -> None:
    w = World(deck_demo, UniformSlip(0.2), seed=1)
    assert w.pos == deck_demo.start
    assert w.t == 0
    assert w.aftershock_events == 0
    assert w.grid is not deck_demo
    assert np.array_equal(w.grid.cells, deck_demo.cells)
    assert w.grid.survivor == deck_demo.survivor


def test_world_rejects_negative_sensor_radius(deck_demo: TrueGrid) -> None:
    with pytest.raises(ValueError):
        World(deck_demo, UniformSlip(0.2), sensor_radius=-1)


def test_percept_as_dict() -> None:
    p = Percept(cells=(((0, 0), Cell.OPEN), ((0, 1), Cell.DEBRIS)))
    assert p.as_dict() == {(0, 0): Cell.OPEN, (0, 1): Cell.DEBRIS}
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.cells = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("start", "radius", "size"),
    [
        ((2, 2), 1, 5),  # interior: self + 4 neighbours
        ((0, 0), 1, 3),  # corner
        ((4, 4), 1, 3),  # corner
        ((0, 2), 1, 4),  # edge
        ((2, 0), 1, 4),  # edge
        ((2, 2), 2, 13),  # r=2 diamond
        ((2, 2), 0, 1),  # r=0: own cell only
        ((0, 0), 2, 6),
    ],
)
def test_sensor_footprint_sizes(open_grid, start, radius: int, size: int) -> None:
    survivor = (4, 4) if start != (4, 4) else (0, 0)
    grid = open_grid(5, 5, start=start, survivor=survivor)
    w = World(grid, UniformSlip(0.2), sensor_radius=radius, seed=0)
    percept = w.sense()
    positions = [p for p, _ in percept.cells]
    assert len(positions) == size
    assert len(set(positions)) == size
    assert set(positions) == set(footprint(start, radius, 5, 5))
    assert all(manhattan(p, start) <= radius for p in positions)
    assert start in positions


def test_sensor_r1_is_exactly_four_neighbours_plus_self(open_grid) -> None:
    grid = open_grid(5, 5, start=(2, 2), survivor=(4, 4))
    w = World(grid, UniformSlip(0.2), sensor_radius=1, seed=0)
    assert set(w.sense().as_dict()) == {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}


def test_sensed_values_are_true_values() -> None:
    grid = TrueGrid.from_ascii("~#.\n#S.\n.~P")
    w = World(grid, UniformSlip(0.2), sensor_radius=2, seed=0)
    d = w.sense().as_dict()
    assert d[(1, 1)] == Cell.OPEN
    assert d[(0, 1)] == Cell.DEBRIS
    assert d[(1, 0)] == Cell.DEBRIS
    assert d[(1, 2)] == Cell.OPEN
    assert d[(2, 1)] == Cell.OPEN  # '~' is open
    assert d[(2, 2)] == Cell.SURVIVOR
    assert d[(0, 2)] == Cell.OPEN
    assert d[(0, 0)] == Cell.OPEN
    for p, v in w.sense().cells:
        assert v == Cell(int(w.grid.cells[p]))


def test_step_percept_is_footprint_of_new_position() -> None:
    grid = _random_grid(6, 6, 0.2, seed=5)
    w = World(grid, UniformSlip(0.3), sensor_radius=1, seed=9)
    rng = np.random.default_rng(1)
    for _ in range(200):
        res = w.step(ACTIONS[int(rng.integers(4))])
        d = res.percept.as_dict()
        assert len(res.percept.cells) == len(d)  # no cell reported twice
        fp = set(footprint(res.new_pos, 1, 6, 6))
        assert fp <= set(d)
        assert set(d) - fp <= {res.target}  # only a bumped cell may be added
        for p, v in d.items():
            assert v == Cell(int(w.grid.cells[p]))


# --- step semantics ----------------------------------------------------------------------


def test_step_p_zero_always_arrives_when_free(open_grid) -> None:
    grid = open_grid(6, 6, start=(2, 2), survivor=(5, 5))
    w = World(grid, UniformSlip(0.0), seed=4)
    for t in range(300):
        a = _steer(w.pos, (2, 2), t)
        prev = w.pos
        res = w.step(a)
        assert isinstance(res, StepResult)
        assert res.kind is A
        assert res.prev == prev
        assert res.action is a
        assert res.target == step_pos(prev, a)
        assert res.new_pos == step_pos(prev, a)
        assert not res.blocked
        assert not res.hit_debris
        assert res.aftershock_cells == ()
        assert w.pos == res.new_pos
        assert w.t == t + 1


def test_step_off_grid_is_blocked(open_grid) -> None:
    w = World(open_grid(3, 3), UniformSlip(0.0), seed=0)
    res = w.step(Action.N)
    assert res.kind is A
    assert res.new_pos == (0, 0)
    assert res.blocked
    assert not res.hit_debris
    assert w.pos == (0, 0)
    res = w.step(Action.W)
    assert res.blocked and res.new_pos == (0, 0)


def test_bump_into_debris_with_r0() -> None:
    grid = TrueGrid.from_ascii("S#.\n...\n..P")
    w = World(grid, UniformSlip(0.0), sensor_radius=0, seed=0)
    assert w.sense().as_dict() == {(0, 0): Cell.OPEN}
    res = w.step(Action.E)
    assert res.kind is A
    assert res.target == (0, 1)
    assert res.hit_debris
    assert res.blocked
    assert res.new_pos == (0, 0)
    assert w.pos == (0, 0)
    d = res.percept.as_dict()
    assert d[(0, 1)] == Cell.DEBRIS  # touch sensing reveals the bumped cell
    assert ((0, 1), Cell.DEBRIS) in res.percept.cells
    assert d[(0, 0)] == Cell.OPEN
    assert set(d) == {(0, 0), (0, 1)}


def test_non_bump_step_with_r0_senses_own_cell_only() -> None:
    grid = TrueGrid.from_ascii("S#.\n...\n..P")
    w = World(grid, UniformSlip(0.0), sensor_radius=0, seed=0)
    res = w.step(Action.S)
    assert res.new_pos == (1, 0)
    assert res.percept.as_dict() == {(1, 0): Cell.OPEN}


def test_slipped_target_is_prev_and_not_blocked() -> None:
    grid = TrueGrid.from_ascii(open_ascii(7, 7, (3, 3), (6, 6)))
    w = World(grid, UniformSlip(0.4), sensor_radius=0, seed=12)
    seen = set()
    for t in range(400):
        res = w.step(_steer(w.pos, (3, 3), t))
        seen.add(res.kind)
        if res.kind is SL:
            assert res.target == res.prev
            assert res.new_pos == res.prev
            assert not res.blocked
            assert not res.hit_debris
    assert seen == set(OUTCOME_ORDER)


def test_skid_into_debris_is_blocked_and_hits() -> None:
    # robot boxed in on its left/right by debris: every skid hits debris
    grid = TrueGrid.from_ascii("#.#\n#S#\n#.#\n..P")
    w = World(grid, UniformSlip(0.4), sensor_radius=1, seed=3)
    skids = 0
    for t in range(400):
        a = Action.S if t % 2 == 0 else Action.N
        if w.pos != (1, 1):
            a = Action.N if w.pos[0] > 1 else Action.S
        prev = w.pos
        res = w.step(a)
        assert prev[1] == 1 and prev[0] <= 2
        if res.kind in (KL, KR):
            skids += 1
            assert res.hit_debris
            assert res.blocked
            assert res.new_pos == prev
            assert w.grid.is_debris(res.target)
    assert skids > 0


def test_new_pos_matches_kind_in_open_interior() -> None:
    grid = TrueGrid.from_ascii(open_ascii(9, 9, (4, 4), (0, 0)))
    w = World(grid, UniformSlip(0.3), sensor_radius=1, seed=21)
    for t in range(2_000):
        a = _steer(w.pos, (4, 4), t)
        res = w.step(a)
        assert not res.blocked
        assert res.new_pos == _expected_new_pos(res.prev, a, res.kind)


def test_world_chi_square_end_to_end() -> None:
    """100k steps through World.step on an open grid; no outcome is ever blocked."""
    grid = TrueGrid.from_ascii(open_ascii(9, 9, (4, 4), (0, 0)))
    w = World(grid, UniformSlip(0.2), sensor_radius=0, seed=2024)
    index = {k: i for i, k in enumerate(OUTCOME_ORDER)}
    counts = np.zeros(4)
    for t in range(N_WORLD_SAMPLES):
        a = _steer(w.pos, (4, 4), t)
        res = w.step(a)
        counts[index[res.kind]] += 1
        assert not res.blocked
    expected = N_WORLD_SAMPLES * np.array([0.80, 0.15, 0.025, 0.025])
    result = chisquare(counts, expected)
    assert result.pvalue > P_VALUE_FLOOR, (counts, result)


def test_terrain_slip_in_world() -> None:
    lines = []
    for r in range(7):
        lines.append(
            "".join("S" if (r, c) == (3, 3) else "P" if (r, c) == (6, 6) else "~" for c in range(7))
        )
    grid = TrueGrid.from_ascii("\n".join(lines))
    t_model = TerrainSlip(0.0, 0.4, grid.high_slip)
    w = World(grid, t_model, sensor_radius=0, seed=8)
    on_plain = []
    on_mud = []
    for t in range(3_000):
        prev = w.pos
        res = w.step(_steer(w.pos, (3, 3), t))
        (on_mud if grid.high_slip[prev] else on_plain).append(res.kind)
    assert on_plain and all(k is A for k in on_plain)
    frac_arrived = sum(k is A for k in on_mud) / len(on_mud)
    assert 0.5 < frac_arrived < 0.7  # 1 - 0.4 = 0.6


# --- determinism and common random numbers -----------------------------------------------


def _run(world: World, actions: list[Action]) -> list[StepResult]:
    return [world.step(a) for a in actions]


def test_same_seed_identical_results() -> None:
    grid = _random_grid(8, 8, 0.2, seed=3)
    rng = np.random.default_rng(77)
    actions = [ACTIONS[int(i)] for i in rng.integers(0, 4, size=300)]
    cfg = AftershockConfig(q=0.3, k=2)
    w1 = World(grid, UniformSlip(0.3), sensor_radius=1, seed=42, aftershock=cfg)
    w2 = World(grid, UniformSlip(0.3), sensor_radius=1, seed=42, aftershock=cfg)
    r1, r2 = _run(w1, actions), _run(w2, actions)
    assert r1 == r2
    assert w1.pos == w2.pos
    assert w1.aftershock_events == w2.aftershock_events
    assert np.array_equal(w1.grid.cells, w2.grid.cells)


def test_different_seeds_differ() -> None:
    grid = TrueGrid.from_ascii(open_ascii(7, 7, (3, 3), (6, 6)))
    actions = [_steer((3, 3), (3, 3), t) for t in range(200)]
    k1 = [r.kind for r in _run(World(grid, UniformSlip(0.3), seed=1), actions)]
    k2 = [r.kind for r in _run(World(grid, UniformSlip(0.3), seed=2), actions)]
    assert k1 != k2


def test_rng_streams_reproducible_and_independent() -> None:
    a, b, c = RngStreams(5), RngStreams(5), RngStreams(6)
    assert np.array_equal(a.outcome.random(50), b.outcome.random(50))
    assert np.array_equal(a.aftershock.random(50), b.aftershock.random(50))
    assert np.array_equal(a.misc.random(50), b.misc.random(50))
    x = RngStreams(5)
    o, af, m = x.outcome.random(20), x.aftershock.random(20), x.misc.random(20)
    assert not np.array_equal(o, af)
    assert not np.array_equal(o, m)
    assert not np.array_equal(af, m)
    assert not np.array_equal(RngStreams(5).outcome.random(20), c.outcome.random(20))
    for gen in (x.outcome, x.aftershock, x.misc):
        assert isinstance(gen, np.random.Generator)


def test_common_random_numbers_across_grids_and_agents() -> None:
    """Same seed => same outcome kinds, whatever the map, the actions or the aftershocks."""
    seed, n = 31, 400
    t_model = UniformSlip(0.3)
    g1 = TrueGrid.from_ascii(open_ascii(9, 9, (4, 4), (0, 0)))
    g2 = _random_grid(6, 6, 0.3, seed=2)
    rng = np.random.default_rng(0)
    w1 = World(g1, t_model, sensor_radius=1, seed=seed)
    w2 = World(g2, t_model, sensor_radius=2, seed=seed, aftershock=AftershockConfig(q=0.5, k=2))
    kinds1 = [w1.step(_steer(w1.pos, (4, 4), t)).kind for t in range(n)]
    kinds2 = [w2.step(ACTIONS[int(rng.integers(4))]).kind for _ in range(n)]
    assert kinds1 == kinds2
    # one outcome uniform per step from the outcome stream (D3)
    us = RngStreams(seed).outcome.random(n)
    assert kinds1 == [t_model.sample_kind((0, 0), float(u)) for u in us]


def test_aftershock_timing_independent_of_actions() -> None:
    """The aftershock stream draws every step, so event timing is shared across agents."""
    grid = TrueGrid.from_ascii(open_ascii(15, 15, (7, 7), (14, 14)))
    cfg = AftershockConfig(q=0.3, k=1)
    w1 = World(grid, UniformSlip(0.2), seed=17, aftershock=cfg)
    w2 = World(grid, UniformSlip(0.2), seed=17, aftershock=cfg)
    rng = np.random.default_rng(5)
    ev1 = [bool(w1.step(_steer(w1.pos, (7, 7), t)).aftershock_cells) for t in range(60)]
    ev2 = [bool(w2.step(ACTIONS[int(rng.integers(4))]).aftershock_cells) for _ in range(60)]
    assert ev1 == ev2
    assert 0 < sum(ev1) < 60


# --- aftershocks -------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "radius"), [(0, 1), (1, 1), (2, 2), (3, 0)])
def test_aftershock_invariants(seed: int, radius: int) -> None:
    grid = _random_grid(10, 10, 0.15, seed=seed)
    original = grid.cells.copy()
    k = 3
    w = World(
        grid,
        UniformSlip(0.2),
        sensor_radius=radius,
        seed=seed,
        aftershock=AftershockConfig(q=1.0, k=k),
    )
    rng = np.random.default_rng(seed + 100)
    events = 0
    for _ in range(80):
        before = _debris_set(w.grid)
        prev_events = w.aftershock_events
        res = w.step(ACTIONS[int(rng.integers(4))])
        after = _debris_set(w.grid)
        cells = res.aftershock_cells
        assert len(cells) <= k
        assert len(set(cells)) == len(cells)
        for p in cells:
            assert w.grid.in_bounds(p)
            assert p not in before  # new debris
            assert p in after
            assert p != res.new_pos  # never on the robot
            assert p != w.grid.survivor  # never on the survivor
            assert p != w.grid.start  # TrueGrid keeps the start OPEN (plan section 3.2)
            assert manhattan(p, res.new_pos) > radius  # outside the current footprint
            assert p not in res.percept.as_dict()
        assert after == before | set(cells)  # nothing else changed
        assert reachable(w.grid.passable_mask(), w.pos, w.grid.survivor)
        # counts only steps that placed >= 1 cell (plan section 3.5), even when q = 1 fires
        if cells:
            assert w.aftershock_events == prev_events + 1
        else:
            assert w.aftershock_events == prev_events
        events = w.aftershock_events
    assert events > 0
    assert np.array_equal(grid.cells, original)  # caller's grid untouched


def test_aftershock_events_count_in_spacious_grid() -> None:
    grid = TrueGrid.from_ascii(open_ascii(20, 20, (10, 10), (19, 19)))
    w = World(grid, UniformSlip(0.2), seed=7, aftershock=AftershockConfig(q=1.0, k=3))
    n = 25
    for t in range(n):
        res = w.step(_steer(w.pos, (10, 10), t))
        assert len(res.aftershock_cells) == 3
    assert w.aftershock_events == n
    assert len(_debris_set(w.grid)) == 3 * n


@pytest.mark.parametrize(
    "cfg", [AftershockConfig(), AftershockConfig(q=0.0, k=5), AftershockConfig(q=1.0, k=0)]
)
def test_aftershocks_disabled_grid_never_changes(cfg: AftershockConfig) -> None:
    grid = _random_grid(8, 8, 0.2, seed=4)
    w = World(grid, UniformSlip(0.3), seed=11, aftershock=cfg)
    rng = np.random.default_rng(2)
    for _ in range(200):
        res = w.step(ACTIONS[int(rng.integers(4))])
        assert res.aftershock_cells == ()
        assert np.array_equal(w.grid.cells, grid.cells)
    assert w.aftershock_events == 0


def test_default_world_has_no_aftershocks(deck_demo: TrueGrid) -> None:
    w = World(deck_demo, UniformSlip(0.2))
    for t in range(50):
        w.step(ACTIONS[t % 4])
    assert w.aftershock_events == 0
    assert np.array_equal(w.grid.cells, deck_demo.cells)


def _reference_aftershock(
    grid: TrueGrid,
    debris: set[tuple[int, int]],
    order: np.ndarray,
    robot: tuple[int, int],
    radius: int,
    k: int,
) -> list[tuple[int, int]]:
    """Spec of D9 / plan section 3.5: the first ``k`` cells in permutation order that are OPEN,
    outside ``footprint(robot, r)``, not the start/survivor, and keep the survivor reachable."""
    rows, cols = grid.shape
    excluded = set(footprint(robot, radius, rows, cols)) | {robot, grid.start, grid.survivor}
    passable = np.ones((rows, cols), dtype=bool)
    for p in debris:
        passable[p] = False
    chosen: list[tuple[int, int]] = []
    for index in order.tolist():
        if len(chosen) >= k:
            break
        p = divmod(int(index), cols)
        if p in excluded or p in debris:
            continue
        passable[p] = False
        if reachable(passable, robot, grid.survivor):
            chosen.append(p)
        else:
            passable[p] = True
    return chosen


@pytest.mark.parametrize("seed", [0, 5, 23])
def test_aftershock_stream_consumption_matches_D9(seed: int) -> None:
    """Every step draws ``(u_event, permutation)`` from the aftershock stream, fired or not."""
    rows = cols = 12
    radius, q, k = 1, 0.3, 2
    grid = TrueGrid.from_ascii(open_ascii(rows, cols, (6, 6), (11, 11)))
    w = World(
        grid,
        UniformSlip(0.2),
        sensor_radius=radius,
        seed=seed,
        aftershock=AftershockConfig(q=q, k=k),
    )
    mirror = RngStreams(seed).aftershock
    rng = np.random.default_rng(seed + 1_000)
    debris: set[tuple[int, int]] = set()
    fired = quiet = 0
    for _ in range(80):
        res = w.step(ACTIONS[int(rng.integers(4))])
        u_event = float(mirror.random())
        order = mirror.permutation(rows * cols)
        assert bool(res.aftershock_cells) == (u_event < q)
        if u_event < q:
            fired += 1
            expected = _reference_aftershock(grid, debris, order, res.new_pos, radius, k)
            assert len(expected) == k  # spacious grid: always enough qualifying cells
            assert list(res.aftershock_cells) == expected
            debris.update(expected)
        else:
            quiet += 1
        assert _debris_set(w.grid) == debris
        assert w.aftershock_events == fired
    assert fired >= 10 and quiet >= 10


@pytest.mark.parametrize("seed", range(5))
def test_aftershock_places_only_qualifying_cells_and_counts_placing_steps(seed: int) -> None:
    """3x3 map, robot parked at the start: exactly 3 of the 5 candidates can ever be filled.

    The footprint of (1,1) at r=1 is {(1,1), (1,2), (2,1)} (1-indexed). The remaining candidates
    are (1,3), (2,2), (2,3), (3,1), (3,2); every route to the survivor (3,3) needs two of them,
    so with k=4 the first event places exactly 3 cells (< k) and later events place none.
    """
    grid = TrueGrid.from_ascii("S..\n...\n..P")
    k = 4
    w = World(
        grid,
        UniformSlip(0.0),
        sensor_radius=1,
        seed=seed,
        aftershock=AftershockConfig(q=1.0, k=k),
    )
    res = w.step(Action.N)  # off-grid: blocked, the robot stays at (0, 0)
    assert res.new_pos == (0, 0) and res.blocked
    assert len(res.aftershock_cells) == 3 < k
    assert w.aftershock_events == 1
    assert reachable(w.grid.passable_mask(), (0, 0), w.grid.survivor)
    saturated = w.grid.cells.copy()
    for _ in range(10):
        res = w.step(Action.N)
        assert res.aftershock_cells == ()  # the event fires (q = 1) but nothing qualifies
        assert w.aftershock_events == 1
        assert np.array_equal(w.grid.cells, saturated)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aftershock_produces_open_to_debris_contradictions(seed: int) -> None:
    """Aftershocks make the belief stale: known-OPEN cells are later sensed as DEBRIS (D8/D9)."""
    grid = _random_grid(10, 10, 0.1, seed=seed)
    original = grid.cells.copy()
    w = World(
        grid,
        UniformSlip(0.2),
        sensor_radius=1,
        seed=seed,
        aftershock=AftershockConfig(q=0.5, k=2),
    )
    b = BeliefMap(grid.rows, grid.cols, grid.survivor)
    first = b.update(w.sense().cells)
    assert all(original[p] == Cell.DEBRIS for p in first.contradictions)
    rng = np.random.default_rng(seed + 50)
    placed: set[tuple[int, int]] = set()
    open_to_debris = 0
    for _ in range(300):
        res = w.step(ACTIONS[int(rng.integers(4))])
        placed.update(res.aftershock_cells)
        prior = b.grid.copy()
        upd = b.update(res.percept.cells)
        for p in upd.contradictions:
            assert w.grid.is_debris(p)
            old = Cell(int(prior[p]))
            assert old in (Cell.UNKNOWN, Cell.OPEN)
            if old is Cell.OPEN:
                # only an earlier aftershock can turn a known-open cell into debris
                assert p in placed
                assert original[p] == Cell.OPEN
                open_to_debris += 1
            elif p not in placed:
                assert original[p] == Cell.DEBRIS
        for p, _ in res.percept.cells:
            if prior[p] == Cell.DEBRIS:  # already-known debris is never reported again
                assert p not in upd.contradictions
    assert open_to_debris > 0
