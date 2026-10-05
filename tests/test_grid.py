"""Tests for rescue.grid (docs/plan.md section 3.2, decisions D1, D2, D7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rescue.grid import (
    Cell,
    TrueGrid,
    fmt_pos,
    footprint,
    in_bounds,
    manhattan,
    neighbours4,
)


def _grid_lines(text: str) -> list[str]:
    """Non-blank, non-comment lines of an ASCII map."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith(";"):
            out.append(s)
    return out


def _same_grid(a: TrueGrid, b: TrueGrid) -> bool:
    return (
        a.start == b.start
        and a.survivor == b.survivor
        and a.shape == b.shape
        and np.array_equal(a.cells, b.cells)
        and np.array_equal(a.high_slip, b.high_slip)
    )


# --- Cell, helpers -----------------------------------------------------------------------


def test_cell_values() -> None:
    assert Cell.UNKNOWN == 0
    assert Cell.OPEN == 1
    assert Cell.DEBRIS == 2
    assert Cell.SURVIVOR == 3
    assert int(np.zeros(1, dtype=np.int8)[0]) == Cell.UNKNOWN


@pytest.mark.parametrize(
    ("pos", "text"), [((0, 0), "(1,1)"), ((2, 3), "(3,4)"), ((9, 0), "(10,1)")]
)
def test_fmt_pos_is_one_indexed(pos: tuple[int, int], text: str) -> None:
    assert fmt_pos(pos) == text


def test_manhattan() -> None:
    assert manhattan((0, 0), (0, 0)) == 0
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((5, 1), (2, 3)) == 5
    assert manhattan((2, 3), (5, 1)) == 5


def test_in_bounds() -> None:
    assert in_bounds((0, 0), 3, 4)
    assert in_bounds((2, 3), 3, 4)
    assert not in_bounds((3, 0), 3, 4)
    assert not in_bounds((0, 4), 3, 4)
    assert not in_bounds((-1, 0), 3, 4)
    assert not in_bounds((0, -1), 3, 4)


def test_neighbours4_order_and_clipping() -> None:
    assert neighbours4((1, 1), 3, 3) == [(0, 1), (1, 2), (2, 1), (1, 0)]  # N, E, S, W
    assert neighbours4((0, 0), 3, 3) == [(0, 1), (1, 0)]
    assert neighbours4((2, 2), 3, 3) == [(1, 2), (2, 1)]
    assert neighbours4((0, 1), 3, 3) == [(0, 2), (1, 1), (0, 0)]
    assert neighbours4((0, 0), 1, 1) == []


def test_footprint_sizes() -> None:
    assert footprint((2, 2), 0, 5, 5) == [(2, 2)]
    assert len(footprint((2, 2), 1, 5, 5)) == 5
    assert len(footprint((0, 0), 1, 5, 5)) == 3
    assert len(footprint((0, 2), 1, 5, 5)) == 4
    assert len(footprint((2, 2), 2, 5, 5)) == 13
    assert len(footprint((0, 0), 2, 5, 5)) == 6


def test_footprint_rejects_negative_radius() -> None:
    with pytest.raises(ValueError):
        footprint((0, 0), -1, 3, 3)


def test_footprint_contents_row_major() -> None:
    fp = footprint((1, 1), 1, 3, 3)
    assert fp == [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    fp2 = footprint((2, 2), 2, 5, 5)
    assert fp2 == sorted(fp2)
    assert set(fp2) == {(r, c) for r in range(5) for c in range(5) if abs(r - 2) + abs(c - 2) <= 2}


# --- parsing -----------------------------------------------------------------------------


def test_deck_demo_parse(deck_demo: TrueGrid) -> None:
    g = deck_demo
    assert (g.rows, g.cols) == (4, 4)
    assert g.shape == (4, 4)
    assert g.start == (0, 0)
    assert g.survivor == (3, 3)
    assert g.cells.dtype == np.int8
    assert g.cells.shape == (4, 4)
    assert g.high_slip.dtype == np.bool_
    assert g.high_slip.shape == (4, 4)
    assert not g.high_slip.any()
    debris = {(int(r), int(c)) for r, c in zip(*np.nonzero(g.cells == Cell.DEBRIS), strict=True)}
    assert debris == {(2, 1), (3, 2)}
    assert g.cells[3, 3] == Cell.SURVIVOR
    assert g.cells[0, 0] == Cell.OPEN
    assert g.cells[1, 0] == Cell.OPEN
    assert not (g.cells == Cell.UNKNOWN).any()


def test_parse_terrain_and_markers(grid_from) -> None:
    g = grid_from("S~.\n.#~\n..P")
    assert g.start == (0, 0)
    assert g.survivor == (2, 2)
    assert g.cells[0, 1] == Cell.OPEN  # '~' is open terrain (D2)
    assert g.cells[1, 2] == Cell.OPEN
    assert g.cells[1, 1] == Cell.DEBRIS
    assert g.cells[2, 2] == Cell.SURVIVOR
    assert g.cells[0, 0] == Cell.OPEN
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 1] = expected[1, 2] = True
    assert np.array_equal(g.high_slip, expected)


def test_parse_ignores_comments_blank_lines_and_whitespace(grid_from) -> None:
    text = "; a comment\n\n   S.#  \n; another\n\n..P\n\n"
    g = grid_from(text)
    assert g.shape == (2, 3)
    assert g.start == (0, 0)
    assert g.survivor == (1, 2)
    assert g.cells[0, 2] == Cell.DEBRIS


def test_from_ascii_name(grid_from) -> None:
    g = TrueGrid.from_ascii("SP", name="tiny")
    assert g.name == "tiny"
    assert grid_from("SP").name == ""


@pytest.mark.parametrize(
    "text",
    [
        "S.S\n..P",  # two starts
        "...\n..P",  # missing start
        "S..\n...",  # missing survivor
        "S.P\n..P",  # two survivors
        "S..\n..\n..P",  # ragged rows
        "S...\n..P",  # ragged rows
        "S.x\n..P",  # invalid character
        "S.?\n..P",  # '?' is not allowed in true maps
        "",  # empty
        "; only a comment\n\n",  # no rows
    ],
)
def test_parse_errors(grid_from, text: str) -> None:
    with pytest.raises(ValueError):
        grid_from(text)


@pytest.mark.parametrize(
    ("text", "line", "col"),
    [
        ("S...\n..?.\n...P", 2, 3),
        # comment and blank lines still count towards the 1-indexed line number
        ("; c\nS..\n.x.\n..P", 3, 2),
        ("\n; c\n\nS..\n..P\n.x.", 6, 2),
        # leading whitespace is stripped but still counts towards the column
        ("  S.x\n  ..P", 1, 5),
        ("S.x\n..P", 1, 3),
        ("\tS..\n\t.?P", 2, 3),
    ],
)
def test_parse_error_reports_line_and_col(grid_from, text: str, line: int, col: int) -> None:
    with pytest.raises(ValueError) as info:
        grid_from(text)
    msg = str(info.value)
    assert f"line {line}, col {col}:" in msg, msg


# --- round trip and I/O ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["S...\n....\n.#..\n..#P", "S~.#\n~~#.\n#..P", "P.\n.S", "S\nP", "#S#\n~.~\nP#."],
)
def test_ascii_round_trip(grid_from, text: str) -> None:
    g = grid_from(text)
    out = g.to_ascii()
    assert _grid_lines(out) == _grid_lines(text)
    g2 = TrueGrid.from_ascii(out)
    assert _same_grid(g, g2)


def test_deck_demo_round_trip(deck_demo: TrueGrid, maps_dir: Path) -> None:
    original = (maps_dir / "deck_demo.txt").read_text(encoding="utf-8")
    assert _grid_lines(deck_demo.to_ascii()) == _grid_lines(original)
    assert _same_grid(TrueGrid.from_ascii(deck_demo.to_ascii()), deck_demo)


def test_save_and_load(tmp_path: Path, grid_from) -> None:
    g = grid_from("S~.#\n....\n#..P")
    path = tmp_path / "m.txt"
    g.save(path)
    loaded = TrueGrid.load(path)
    assert _same_grid(g, loaded)
    assert _same_grid(TrueGrid.load(str(path)), g)


def test_deck_figure_loads(maps_dir: Path) -> None:
    g = TrueGrid.load(maps_dir / "deck_figure.txt")
    assert g.start == (0, 0)
    assert g.survivor == (2, 3)
    assert g.cells[1, 0] == Cell.DEBRIS
    assert int((g.cells == Cell.DEBRIS).sum()) == 1


# --- TrueGrid methods --------------------------------------------------------------------


def test_in_bounds_and_is_debris(deck_demo: TrueGrid) -> None:
    g = deck_demo
    assert g.in_bounds((0, 0))
    assert g.in_bounds((3, 3))
    assert not g.in_bounds((4, 0))
    assert not g.in_bounds((0, -1))
    assert g.is_debris((2, 1))
    assert g.is_debris((3, 2))
    assert not g.is_debris((0, 0))
    assert not g.is_debris((3, 3))
    assert not g.is_debris((-1, 0))  # off-grid is not debris
    assert not g.is_debris((0, 4))


def test_passable_mask(deck_demo: TrueGrid) -> None:
    mask = deck_demo.passable_mask()
    assert mask.dtype == np.bool_
    expected = np.ones((4, 4), dtype=bool)
    expected[2, 1] = expected[3, 2] = False
    assert np.array_equal(mask, expected)
    assert mask[3, 3]  # survivor is passable


def test_copy_is_deep(grid_from) -> None:
    g = grid_from("S~.\n...\n..P")
    c = g.copy()
    assert _same_grid(g, c)
    c.cells[1, 1] = Cell.DEBRIS
    c.high_slip[2, 0] = True
    assert g.cells[1, 1] == Cell.OPEN
    assert not g.high_slip[2, 0]


def test_add_debris(grid_from) -> None:
    g = grid_from("S..\n...\n..P")
    g.add_debris((1, 1))
    assert g.is_debris((1, 1))
    assert g.cells[1, 1] == Cell.DEBRIS
    assert not g.passable_mask()[1, 1]


def test_add_debris_on_high_slip_cell_clears_flag_and_round_trips(grid_from) -> None:
    g = grid_from("S~.\n.~.\n..P")
    g.add_debris((0, 1))
    assert g.is_debris((0, 1))
    assert not g.high_slip[0, 1]
    assert g.high_slip[1, 1]  # other terrain untouched
    assert g.to_ascii().splitlines() == ["S#.", ".~.", "..P"]
    assert TrueGrid.from_ascii(g.to_ascii()) == g


@pytest.mark.parametrize("p", [(5, 5), (-1, 0), (0, 3), (3, 0)])
def test_add_debris_off_grid_raises(grid_from, p: tuple[int, int]) -> None:
    g = grid_from("S..\n...\n..P")
    before = g.cells.copy()
    with pytest.raises(ValueError):
        g.add_debris(p)
    assert np.array_equal(g.cells, before)


def test_true_grid_eq(grid_from) -> None:
    g = grid_from("S~.\n.#.\n..P")
    assert g == g.copy()
    assert g == TrueGrid.from_ascii(g.to_ascii())
    assert (g != g.copy()) is False
    renamed = TrueGrid.from_ascii(g.to_ascii(), name="other")
    assert g != renamed
    changed_cell = g.copy()
    changed_cell.add_debris((0, 2))
    assert g != changed_cell
    changed_slip = g.copy()
    changed_slip.high_slip[1, 0] = True
    assert g != changed_slip
    assert g != grid_from(".~.\nS#.\n..P")  # different start
    assert g != grid_from("S~.\n.#P\n...")  # different survivor
    assert g != "S~.\n.#.\n..P"
    with pytest.raises(TypeError):
        hash(g)


def test_constructor_validates_before_int8_cast() -> None:
    """Out-of-range integers must be rejected, not wrapped (258 -> 2 = DEBRIS) by the cast."""
    cells = np.array([[Cell.OPEN, 258], [Cell.OPEN, Cell.SURVIVOR]], dtype=np.int64)
    with pytest.raises(ValueError):
        TrueGrid(cells=cells, start=(0, 0), survivor=(1, 1), high_slip=np.zeros((2, 2), bool))
    with pytest.raises(ValueError):
        TrueGrid(
            cells=np.array([[1, 1], [1, 3 + 256]]),
            start=(0, 0),
            survivor=(1, 1),
            high_slip=np.zeros((2, 2), bool),
        )


def test_constructor_copies_input_arrays() -> None:
    cells = np.array([[Cell.OPEN, Cell.OPEN], [Cell.OPEN, Cell.SURVIVOR]], dtype=np.int8)
    slip = np.zeros((2, 2), dtype=bool)
    g = TrueGrid(cells=cells, start=(0, 0), survivor=(1, 1), high_slip=slip)
    assert g.cells is not cells
    assert g.high_slip is not slip
    cells[0, 1] = Cell.DEBRIS
    slip[0, 1] = True
    assert g.cells[0, 1] == Cell.OPEN
    assert not g.high_slip[0, 1]
    assert g.cells.dtype == np.int8
    assert g.high_slip.dtype == np.bool_


def _arrays(text: str) -> tuple[np.ndarray, np.ndarray]:
    g = TrueGrid.from_ascii(text)
    return g.cells.copy(), g.high_slip.copy()


def test_constructor_accepts_valid_arrays() -> None:
    cells, slip = _arrays("S~.\n.#.\n..P")
    g = TrueGrid(cells=cells, start=(0, 0), survivor=(2, 2), high_slip=slip, name="x")
    assert g.to_ascii().splitlines() == ["S~.", ".#.", "..P"]


@pytest.mark.parametrize(
    "case",
    [
        "unknown_cell",
        "negative_cell",
        "second_survivor",
        "start_not_open",
        "survivor_not_survivor",
        "start_off_grid",
        "start_is_survivor",
        "start_high_slip",
        "survivor_high_slip",
        "debris_high_slip",
        "slip_shape",
        "cells_1d",
        "cells_empty",
        "cell_out_of_int8_range",
    ],
)
def test_constructor_rejects_invalid_grids(case: str) -> None:
    """Every valid TrueGrid must round-trip through ASCII exactly (plan section 3.2)."""
    cells, slip = _arrays("S..\n...\n..P")
    start, survivor = (0, 0), (2, 2)
    if case == "unknown_cell":
        cells[1, 1] = Cell.UNKNOWN
    elif case == "negative_cell":
        cells[1, 1] = -1
    elif case == "second_survivor":
        cells[1, 1] = Cell.SURVIVOR
    elif case == "start_not_open":
        cells[0, 0] = Cell.DEBRIS
    elif case == "survivor_not_survivor":
        survivor = (1, 1)
    elif case == "start_off_grid":
        start = (3, 0)
    elif case == "start_is_survivor":
        start = survivor
    elif case == "start_high_slip":
        slip[0, 0] = True
    elif case == "survivor_high_slip":
        slip[2, 2] = True
    elif case == "debris_high_slip":  # '#' cannot carry the '~' flag in ASCII
        cells[0, 1] = Cell.DEBRIS
        slip[0, 1] = True
    elif case == "slip_shape":
        slip = np.zeros((2, 3), dtype=bool)
    elif case == "cells_1d":
        cells = cells.ravel()
    elif case == "cells_empty":
        cells = np.zeros((0, 3), dtype=np.int8)
    elif case == "cell_out_of_int8_range":
        cells = cells.astype(np.int64)
        cells[1, 1] = 256 + int(Cell.DEBRIS)  # would wrap to DEBRIS under an int8 cast
    with pytest.raises(ValueError):
        TrueGrid(cells=cells, start=start, survivor=survivor, high_slip=slip)


def test_add_debris_on_start_or_survivor_raises(grid_from) -> None:
    g = grid_from("S..\n...\n..P")
    with pytest.raises(ValueError):
        g.add_debris((2, 2))
    with pytest.raises(ValueError):
        g.add_debris((0, 0))
