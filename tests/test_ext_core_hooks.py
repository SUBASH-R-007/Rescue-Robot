"""Core hooks for the optional extensions (plan §3.14): several survivors on a ``TrueGrid``,
a ``BeliefMap`` without a known survivor, ``ProblemModel.from_belief`` on such a belief, and
aftershocks / ``World.on_survivor`` with several survivors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rescue.agents import make_agent
from rescue.belief import BeliefMap
from rescue.config import AftershockConfig
from rescue.grid import Cell, Pos, TrueGrid, manhattan
from rescue.model import ACTIONS, Action, ProblemModel, UniformSlip
from rescue.planners.bfs import reachable
from rescue.world import World

MULTI_ASCII = """\
; three survivors
S..#.
.P...
...#P
P....
"""

#: Rows / cols / debris density / survivor count of the seeded aftershock maps.
AFTERSHOCK_ROWS = 10
AFTERSHOCK_COLS = 10
AFTERSHOCK_DENSITY = 0.15
AFTERSHOCK_SURVIVORS = 3
AFTERSHOCK_STEPS = 120
AFTERSHOCK_SEEDS = range(12)


# --- helpers -----------------------------------------------------------------------------


def _debris_set(grid: TrueGrid) -> set[Pos]:
    rs, cs = np.nonzero(grid.cells == Cell.DEBRIS)
    return {(int(r), int(c)) for r, c in zip(rs, cs, strict=True)}


def _multi_random_grid(seed: int) -> TrueGrid:
    """Seeded rubble map with several survivors, all reachable from the start (0, 0)."""
    rows, cols = AFTERSHOCK_ROWS, AFTERSHOCK_COLS
    for attempt in range(100):
        rng = np.random.default_rng([seed, attempt])
        debris = rng.random((rows, cols)) < AFTERSHOCK_DENSITY
        debris[0, 0] = False
        passable = ~debris
        free = [(r, c) for r in range(rows) for c in range(cols)
                if passable[r, c] and (r, c) != (0, 0) and reachable(passable, (0, 0), (r, c))]
        if len(free) < AFTERSHOCK_SURVIVORS:
            continue
        picks = rng.choice(len(free), size=AFTERSHOCK_SURVIVORS, replace=False)
        survivors = sorted(free[int(i)] for i in picks)
        cells = np.where(debris, int(Cell.DEBRIS), int(Cell.OPEN)).astype(np.int8)
        for p in survivors:
            cells[p] = Cell.SURVIVOR
        return TrueGrid(cells=cells, start=(0, 0), survivor=survivors[0],
                        high_slip=np.zeros((rows, cols), dtype=bool),
                        extra_survivors=tuple(survivors[1:]))
    raise AssertionError("could not build a multi-survivor grid")


def _grid_with(extra: tuple[Pos, ...], *, mark: tuple[Pos, ...] | None = None) -> TrueGrid:
    """3x3 open grid, start (0,0), survivor (2,2); ``mark`` cells are set to SURVIVOR."""
    cells = np.full((3, 3), int(Cell.OPEN), dtype=np.int8)
    cells[2, 2] = Cell.SURVIVOR
    for p in extra if mark is None else mark:
        if 0 <= p[0] < 3 and 0 <= p[1] < 3:
            cells[p] = Cell.SURVIVOR
    return TrueGrid(cells=cells, start=(0, 0), survivor=(2, 2),
                    high_slip=np.zeros((3, 3), dtype=bool), extra_survivors=extra)


# --- TrueGrid: parsing, round trip, invariants ------------------------------------------


def test_single_survivor_defaults(deck_demo: TrueGrid) -> None:
    assert deck_demo.extra_survivors == ()
    assert deck_demo.survivors == (deck_demo.survivor,)


def test_multi_parse_row_major_order() -> None:
    g = TrueGrid.from_ascii(MULTI_ASCII, name="m", multi=True)
    assert g.survivor == (1, 1)
    assert g.extra_survivors == ((2, 4), (3, 0))
    assert g.survivors == ((1, 1), (2, 4), (3, 0))
    assert g.start == (0, 0)
    assert g.name == "m"
    rs, cs = np.nonzero(g.cells == Cell.SURVIVOR)
    assert {(int(r), int(c)) for r, c in zip(rs, cs, strict=True)} == set(g.survivors)


def test_multi_with_one_survivor_equals_strict_parse(deck_demo: TrueGrid) -> None:
    text = deck_demo.to_ascii()
    assert TrueGrid.from_ascii(text, multi=True) == TrueGrid.from_ascii(text)


def test_strict_mode_still_rejects_two_survivors() -> None:
    with pytest.raises(ValueError, match="exactly one survivor 'P'"):
        TrueGrid.from_ascii(MULTI_ASCII)


def test_multi_needs_a_survivor_and_one_start() -> None:
    with pytest.raises(ValueError, match="at least one survivor"):
        TrueGrid.from_ascii("S..\n...\n", multi=True)
    with pytest.raises(ValueError, match="exactly one start"):
        TrueGrid.from_ascii("S.P\nS.P\n", multi=True)
    with pytest.raises(ValueError, match="exactly one start"):
        TrueGrid.from_ascii("..P\n..P\n", multi=True)


def test_multi_round_trip_ascii_and_file(tmp_path: Path) -> None:
    g = TrueGrid.from_ascii(MULTI_ASCII, multi=True)
    text = g.to_ascii()
    assert text.count("P") == 3
    assert TrueGrid.from_ascii(text, multi=True) == g
    path = tmp_path / "multi.txt"
    g.save(path)
    loaded = TrueGrid.load(path, multi=True)
    assert loaded.extra_survivors == g.extra_survivors
    assert loaded.name == "multi"
    with pytest.raises(ValueError, match="exactly one survivor"):
        TrueGrid.load(path)


def test_copy_and_eq_include_extra_survivors() -> None:
    g = TrueGrid.from_ascii(MULTI_ASCII, multi=True)
    c = g.copy()
    assert c == g and c is not g
    assert c.extra_survivors == g.extra_survivors
    c.add_debris((0, 1))
    assert c != g
    one_extra = _grid_with(((0, 2),))
    other_extra = _grid_with(((2, 0),))
    assert one_extra != other_extra  # same survivor, different extras
    assert one_extra != _grid_with((), mark=())


def test_extra_survivors_normalised_to_int_tuples() -> None:
    g = _grid_with((np.array([0, 2]),), mark=((0, 2),))  # type: ignore[arg-type]
    assert g.extra_survivors == ((0, 2),)
    assert all(type(v) is int for v in g.extra_survivors[0])


@pytest.mark.parametrize(
    ("extra", "mark", "message"),
    [
        (((5, 5),), (), "outside"),
        (((0, 0),), ((1, 1),), "cannot be the start"),
        (((2, 2),), (), "distinct"),
        (((0, 2), (0, 2)), None, "distinct"),
        (((0, 2),), (), "must be a SURVIVOR cell"),
        ((), ((0, 2),), "exactly one SURVIVOR"),
        (((0, 2),), ((0, 2), (1, 0)), "exactly 2 SURVIVOR cells"),
    ],
)
def test_extra_survivor_invariants(
    extra: tuple[Pos, ...], mark: tuple[Pos, ...] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _grid_with(extra, mark=mark)


def test_extra_survivor_cannot_be_high_slip() -> None:
    cells = np.full((3, 3), int(Cell.OPEN), dtype=np.int8)
    cells[2, 2] = cells[0, 2] = Cell.SURVIVOR
    slip = np.zeros((3, 3), dtype=bool)
    slip[0, 2] = True
    with pytest.raises(ValueError, match="high-slip"):
        TrueGrid(cells=cells, start=(0, 0), survivor=(2, 2), high_slip=slip,
                 extra_survivors=((0, 2),))


def test_add_debris_refuses_every_survivor() -> None:
    g = TrueGrid.from_ascii(MULTI_ASCII, multi=True)
    for p in g.survivors:
        with pytest.raises(ValueError, match="start/survivor"):
            g.add_debris(p)
    g.add_debris((0, 1))
    assert g.is_debris((0, 1))


# --- BeliefMap(None), survivors_seen, from_belief ----------------------------------------


def test_belief_without_survivor_starts_all_unknown() -> None:
    b = BeliefMap(4, 5, None)
    assert b.survivor is None
    assert b.explored_count() == 0
    assert not b.grid.any()
    assert b.survivors_seen() == ()
    assert b.passable_mask().all()
    assert "survivor=None" in repr(b)
    c = b.copy()
    assert c.survivor is None and c is not b


def test_survivors_seen_row_major_from_observations() -> None:
    b = BeliefMap(4, 5, None)
    b.update({(3, 0): Cell.SURVIVOR, (1, 1): Cell.SURVIVOR, (0, 0): Cell.OPEN})
    assert b.survivors_seen() == ((1, 1), (3, 0))
    assert b.explored_count() == 3
    c = b.copy()
    assert c.survivors_seen() == b.survivors_seen()


def test_survivors_seen_with_known_survivor() -> None:
    b = BeliefMap(3, 3, (2, 1))
    assert b.survivors_seen() == ((2, 1),)
    b.update({(0, 2): Cell.SURVIVOR})
    assert b.survivors_seen() == ((0, 2), (2, 1))


def test_belief_survivor_still_validated() -> None:
    with pytest.raises(ValueError, match="outside"):
        BeliefMap(3, 3, (3, 0))


def test_from_belief_rejects_missing_survivor() -> None:
    t = UniformSlip(0.2)
    with pytest.raises(ValueError, match="without a survivor"):
        ProblemModel.from_belief(BeliefMap(3, 3, None), t)
    model = ProblemModel.from_belief(BeliefMap(3, 3, (2, 2)), t)
    assert model.goal == (2, 2)


def test_core_agent_rejects_belief_without_survivor() -> None:
    agent = make_agent("andor", UniformSlip(0.2))
    with pytest.raises(ValueError, match="belief's survivor none"):
        agent.reset((0, 0), (2, 2), BeliefMap(3, 3, None))


# --- World: aftershocks with several survivors, on_survivor ------------------------------


@pytest.mark.parametrize("seed", AFTERSHOCK_SEEDS)
def test_aftershocks_never_hit_or_disconnect_any_survivor(seed: int) -> None:
    grid = _multi_random_grid(seed)
    assert len(grid.survivors) == AFTERSHOCK_SURVIVORS
    w = World(grid, UniformSlip(0.2), sensor_radius=1, seed=seed,
              aftershock=AftershockConfig(q=1.0, k=3))
    rng = np.random.default_rng([seed, 1])
    placed = 0
    for _ in range(AFTERSHOCK_STEPS):
        before = _debris_set(w.grid)
        res = w.step(ACTIONS[int(rng.integers(len(ACTIONS)))])
        cells = res.aftershock_cells
        placed += len(cells)
        assert _debris_set(w.grid) == before | set(cells)
        for p in cells:
            assert p not in grid.survivors
            assert p != grid.start and p != res.new_pos
            assert manhattan(p, res.new_pos) > 1
        passable = w.grid.passable_mask()
        for s in w.grid.survivors:
            assert w.grid.cells[s] == Cell.SURVIVOR
            assert reachable(passable, w.pos, s)
    assert placed > 0
    assert w.grid.survivors == grid.survivors


def test_aftershocks_never_seal_an_extra_survivor_pocket() -> None:
    """An extra survivor behind a one-cell doorway: the doorway never becomes debris.

    Checking only the primary survivor (the pre-extension rule) would be free to seal it.
    """
    text = """\
S........
.........
.........
.......P.
####.####
P........
.........
"""
    grid = TrueGrid.from_ascii(text, multi=True)
    doorway = (4, 4)
    assert grid.cells[doorway] == Cell.OPEN
    assert grid.extra_survivors == ((5, 0),)
    for seed in range(20):
        w = World(grid, UniformSlip(0.0), sensor_radius=0, seed=seed,
                  aftershock=AftershockConfig(q=1.0, k=3))
        for _ in range(30):
            w.step(Action.W)  # stays near the start corner, far from the doorway
            assert not w.grid.is_debris(doorway)
            assert reachable(w.grid.passable_mask(), w.pos, (5, 0))


def test_aftershocks_stop_when_a_survivor_starts_unreachable() -> None:
    """Documented behaviour (World._aftershock): every survivor in ``grid.survivors`` must stay
    reachable, so one walled-off survivor disables aftershock debris for the whole mission,
    while the (u, permutation) draws are still consumed (streams stay aligned)."""
    walled = TrueGrid.from_ascii("S....\n.....\n..P..\n###..\n.P#..\n", multi=True)
    open_twin = TrueGrid.from_ascii("S....\n.....\n..P..\n###..\n.P...\n", multi=True)
    assert not reachable(walled.passable_mask(), walled.start, (4, 1))
    assert reachable(open_twin.passable_mask(), open_twin.start, (4, 1))
    cfg = AftershockConfig(q=1.0, k=1)
    a = World(walled, UniformSlip(0.0), sensor_radius=0, seed=3, aftershock=cfg)
    b = World(open_twin, UniformSlip(0.0), sensor_radius=0, seed=3, aftershock=cfg)
    placed_b = 0
    for _ in range(10):
        assert a.step(Action.W).aftershock_cells == ()
        placed_b += len(b.step(Action.W).aftershock_cells)
    assert a.aftershock_events == 0 and placed_b > 0
    assert a.rng.aftershock.bit_generator.state == b.rng.aftershock.bit_generator.state


def test_on_survivor_reports_any_survivor() -> None:
    g = TrueGrid.from_ascii("SP.\n...\nP..\n", multi=True)
    w = World(g, UniformSlip(0.0), seed=0)
    assert w.on_survivor is None
    w.step(Action.S)
    assert w.pos == (1, 0) and w.on_survivor is None
    w.step(Action.S)
    assert w.pos == (2, 0)
    assert w.on_survivor == (2, 0)
    assert not w.rescued  # ``rescued`` still means the primary survivor
    w.step(Action.N)
    w.step(Action.N)
    w.step(Action.E)
    assert w.on_survivor == (0, 1) == g.survivor
    assert w.rescued


def test_on_survivor_single_survivor_matches_rescued(deck_demo: TrueGrid) -> None:
    w = World(deck_demo, UniformSlip(0.0), seed=0)
    assert w.on_survivor is None and not w.rescued
