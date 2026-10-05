"""Tests for rescue.belief (docs/plan.md section 3.4, decision D8)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rescue.belief import BeliefMap, BeliefUpdate
from rescue.grid import Cell


def test_initial_belief() -> None:
    b = BeliefMap(3, 4, (2, 3))
    assert (b.rows, b.cols) == (3, 4)
    assert b.survivor == (2, 3)
    assert b.grid.dtype == np.int8
    assert b.grid.shape == (3, 4)
    assert b.get((2, 3)) == Cell.SURVIVOR
    assert b.is_known((2, 3))
    for r in range(3):
        for c in range(4):
            if (r, c) != (2, 3):
                assert b.get((r, c)) == Cell.UNKNOWN
                assert not b.is_known((r, c))
    assert b.explored_count() == 1  # only the survivor cell is known


def test_update_writes_values_and_newly_known() -> None:
    b = BeliefMap(3, 3, (2, 2))
    upd = b.update({(0, 0): Cell.OPEN, (0, 1): Cell.OPEN, (1, 0): Cell.DEBRIS})
    assert isinstance(upd, BeliefUpdate)
    assert set(upd.newly_known) == {(0, 0), (0, 1), (1, 0)}
    assert len(upd.newly_known) == 3
    assert b.get((0, 0)) == Cell.OPEN
    assert b.get((1, 0)) == Cell.DEBRIS
    assert b.explored_count() == 4
    # re-observing the same cells: nothing is new
    upd2 = b.update({(0, 0): Cell.OPEN, (1, 0): Cell.DEBRIS})
    assert upd2.newly_known == ()
    assert upd2.contradictions == ()
    assert b.explored_count() == 4


def test_update_accepts_iterable_of_pairs() -> None:
    b = BeliefMap(2, 2, (1, 1))
    upd = b.update([((0, 0), Cell.OPEN), ((0, 1), Cell.DEBRIS)])
    assert set(upd.newly_known) == {(0, 0), (0, 1)}
    assert b.get((0, 1)) == Cell.DEBRIS
    upd2 = b.update(iter([((1, 0), Cell.OPEN)]))
    assert upd2.newly_known == ((1, 0),)


def test_survivor_observation_is_not_new() -> None:
    b = BeliefMap(2, 2, (1, 1))
    upd = b.update({(1, 1): Cell.SURVIVOR})
    assert upd.newly_known == ()
    assert upd.contradictions == ()
    assert b.get((1, 1)) == Cell.SURVIVOR


def test_unknown_to_debris_is_contradiction() -> None:
    b = BeliefMap(3, 3, (2, 2))
    upd = b.update({(1, 1): Cell.DEBRIS})
    assert upd.contradictions == ((1, 1),)
    assert upd.newly_known == ((1, 1),)


def test_open_to_debris_is_contradiction() -> None:
    b = BeliefMap(3, 3, (2, 2))
    b.update({(1, 1): Cell.OPEN})
    upd = b.update({(1, 1): Cell.DEBRIS})  # aftershock
    assert upd.contradictions == ((1, 1),)
    assert upd.newly_known == ()
    assert b.get((1, 1)) == Cell.DEBRIS


def test_unknown_to_open_is_not_contradiction() -> None:
    b = BeliefMap(3, 3, (2, 2))
    upd = b.update({(1, 1): Cell.OPEN})
    assert upd.contradictions == ()
    assert upd.newly_known == ((1, 1),)


def test_debris_to_debris_is_not_contradiction() -> None:
    b = BeliefMap(3, 3, (2, 2))
    b.update({(1, 1): Cell.DEBRIS})
    upd = b.update({(1, 1): Cell.DEBRIS})
    assert upd.contradictions == ()
    assert upd.newly_known == ()


def test_open_to_open_is_not_contradiction() -> None:
    b = BeliefMap(3, 3, (2, 2))
    b.update({(0, 0): Cell.OPEN})
    assert b.update({(0, 0): Cell.OPEN}).contradictions == ()


def test_mixed_update_reports_only_debris_contradictions() -> None:
    b = BeliefMap(4, 4, (3, 3))
    b.update({(0, 0): Cell.OPEN, (0, 1): Cell.OPEN, (0, 2): Cell.DEBRIS})
    upd = b.update(
        {
            (0, 0): Cell.OPEN,  # OPEN -> OPEN
            (0, 1): Cell.DEBRIS,  # OPEN -> DEBRIS   (contradiction)
            (0, 2): Cell.DEBRIS,  # DEBRIS -> DEBRIS
            (1, 0): Cell.OPEN,  # UNKNOWN -> OPEN
            (1, 1): Cell.DEBRIS,  # UNKNOWN -> DEBRIS (contradiction)
        }
    )
    assert set(upd.contradictions) == {(0, 1), (1, 1)}
    assert len(upd.contradictions) == 2
    assert set(upd.newly_known) == {(1, 0), (1, 1)}
    assert b.explored_count() == 1 + 5


def test_belief_update_is_frozen() -> None:
    upd = BeliefUpdate(newly_known=(), contradictions=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        upd.newly_known = ((0, 0),)  # type: ignore[misc]


def test_passable_mask_is_optimistic() -> None:
    b = BeliefMap(3, 3, (2, 2))
    b.update({(0, 0): Cell.OPEN, (1, 1): Cell.DEBRIS})
    mask = b.passable_mask()
    assert mask.dtype == np.bool_
    expected = np.ones((3, 3), dtype=bool)
    expected[1, 1] = False
    assert np.array_equal(mask, expected)  # UNKNOWN and SURVIVOR are passable


def test_copy_is_independent() -> None:
    b = BeliefMap(3, 3, (2, 2))
    b.update({(0, 0): Cell.OPEN})
    c = b.copy()
    assert np.array_equal(b.grid, c.grid)
    assert c.survivor == b.survivor
    c.update({(1, 1): Cell.DEBRIS})
    assert b.get((1, 1)) == Cell.UNKNOWN
    assert c.get((1, 1)) == Cell.DEBRIS
    assert b.explored_count() == 2
    assert c.explored_count() == 3


def test_explored_count_full_map() -> None:
    b = BeliefMap(2, 3, (0, 2))
    b.update({(r, c): Cell.OPEN for r in range(2) for c in range(3) if (r, c) != (0, 2)})
    assert b.explored_count() == 6


# --- error contract (plan section 3.4 clarification) -------------------------------------


@pytest.mark.parametrize("p", [(-1, 0), (3, 0), (0, 3), (0, -1), (5, 5)])
def test_belief_rejects_off_grid_positions(p: tuple[int, int]) -> None:
    b = BeliefMap(3, 3, (2, 2))
    with pytest.raises(ValueError):
        b.get(p)
    with pytest.raises(ValueError):
        b.is_known(p)
    with pytest.raises(ValueError):
        b.update({p: Cell.OPEN})
    assert b.explored_count() == 1


def test_belief_update_rejects_unknown_and_invalid_values() -> None:
    b = BeliefMap(3, 3, (2, 2))
    with pytest.raises(ValueError):
        b.update({(0, 0): Cell.UNKNOWN})
    with pytest.raises(ValueError):
        b.update([((0, 0), 0)])  # raw UNKNOWN value
    with pytest.raises(ValueError):
        b.update([((0, 0), 7)])  # not a Cell value
    assert b.get((0, 0)) == Cell.UNKNOWN
    assert b.explored_count() == 1


@pytest.mark.parametrize(
    ("rows", "cols", "survivor"),
    [(0, 3, (0, 0)), (3, 0, (0, 0)), (-1, 3, (0, 0)), (3, 3, (3, 3)), (3, 3, (-1, 0))],
)
def test_belief_constructor_rejects_invalid_shape_or_survivor(
    rows: int, cols: int, survivor: tuple[int, int]
) -> None:
    with pytest.raises(ValueError):
        BeliefMap(rows, cols, survivor)


@pytest.mark.parametrize(
    "bad",
    [((9, 9), Cell.OPEN), ((1, 1), Cell.UNKNOWN), ((-1, 0), Cell.DEBRIS)],
)
def test_belief_update_is_all_or_nothing(bad: tuple[tuple[int, int], Cell]) -> None:
    """A rejected update leaves the belief exactly as it was."""
    b = BeliefMap(3, 3, (2, 2))
    b.update({(0, 1): Cell.OPEN})
    before = b.grid.copy()
    with pytest.raises(ValueError):
        b.update([((0, 0), Cell.OPEN), ((0, 1), Cell.DEBRIS), bad])
    assert np.array_equal(b.grid, before)
    assert b.get((0, 0)) == Cell.UNKNOWN
    assert b.get((0, 1)) == Cell.OPEN
