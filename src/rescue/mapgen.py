"""Scenario library: seeded map generators with a solvability guarantee (brief §6.3).

Every generator takes ``(rows, cols, seed, **params)`` and returns a :class:`TrueGrid` whose
start and survivor are distinct free cells, at least
``MAPGEN_MIN_SEPARATION_FRACTION * (rows + cols - 2)`` apart (Manhattan) and BFS-connected on
the true map. Draws come from ``np.random.default_rng(SeedSequence([seed, MAPGEN_STREAM_KEY]))``,
a stream independent of the World's spawned streams, so the map depends only on
``(rows, cols, seed, params)``. A failed attempt is regenerated with the same generator
(continuing its stream) at most ``MAPGEN_MAX_ATTEMPTS`` times, then
:class:`MapGenerationError` is raised. After ``MAPGEN_RELAX_START_FRACTION`` of the attempts
the separation is relaxed linearly down to 1, which only ever matters on tiny or very
cluttered maps.

Scenarios (:data:`SCENARIOS`):

* ``random_rubble`` - i.i.d. debris at ``density`` (``RUBBLE_DENSITY_RANGE``).
* ``collapsed_building`` - a street around a building of rooms separated by 1-cell walls with
  1-cell doorways and entrances, interior hallways (corridors with a door from every room
  along them and an entrance at both ends) where they fit, some doorways collapsed, a few
  fallen walls and some rubble inside; the robot starts in the street and the survivor is in a
  room. If the survivor is cut off, a collapsed doorway on the boundary of the start's
  component is re-opened (repeatedly), preferring one that leads straight to the survivor.
* ``flooded_zone`` - open terrain with a few smooth pools of high-slip water / mud (``~``,
  thresholded Gaussian-smoothed noise) and sparse debris on dry land; start and survivor are
  placed so that water lies on every shortest route between them.
* ``maze_corridors`` - a perfect maze (iterative recursive backtracker) on the even lattice,
  with ``loop_fraction`` of the remaining inner walls knocked out.
* ``deck_demo`` / ``deck_figure`` - the fixed slide maps in ``maps/`` (size and seed ignored).

CLI: ``python -m rescue.mapgen NAME --rows 15 --cols 15 --seed 0 [--out maps/x.txt]``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import (
    BUILDING_COLLAPSE_FRACTION,
    BUILDING_ENTRANCE_PROB,
    BUILDING_HALL_PROB,
    BUILDING_HALL_WIDTH,
    BUILDING_MERGE_FRACTION,
    BUILDING_MIN_ENTRANCES,
    BUILDING_MIN_ROOM_SIZE,
    BUILDING_ROOM_JITTER,
    BUILDING_ROOM_SIZE,
    BUILDING_RUBBLE_DENSITY,
    BUILDING_STREET_WIDTH,
    DEFAULT_RUBBLE_DENSITY,
    FLOOD_BLOB_SCALE,
    FLOOD_CROSSING_START_TRIES,
    FLOOD_DEBRIS_DENSITY,
    FLOOD_DEBRIS_RANGE,
    FLOOD_MIN_SIGMA,
    FLOOD_PAD_SIGMAS,
    FLOOD_WATER_FRACTION,
    FLOOD_WATER_RANGE,
    MAPGEN_DEFAULT_SEED,
    MAPGEN_DEFAULT_SIZE,
    MAPGEN_MAX_ATTEMPTS,
    MAPGEN_MIN_SEPARATION_FRACTION,
    MAPGEN_MIN_SIZE,
    MAPGEN_RELAX_START_FRACTION,
    MAPGEN_STREAM_KEY,
    MAZE_LOOP_FRACTION,
    RUBBLE_DENSITY_RANGE,
)
from rescue.grid import Cell, Pos, TrueGrid
from rescue.planners.bfs import UNREACHABLE, bfs_distances, reachable

#: Repository root (src/rescue/mapgen.py -> three levels up), for the fixed ``maps/`` files.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: 4-neighbour offsets in the fixed order N, E, S, W.
_DIRS: tuple[Pos, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
#: Float slack when turning a separation fraction into a whole number of steps.
_SEPARATION_EPS = 1e-9

#: One generation attempt: given the stream, the required separation and whether the
#: scenario's soft placement requirements still apply (``strict``), a grid or None.
_Attempt = Callable[[np.random.Generator, int, bool], "TrueGrid | None"]


class MapGenerationError(RuntimeError):
    """No solvable map was found within ``MAPGEN_MAX_ATTEMPTS`` attempts."""


@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario: its generator ``(rows, cols, seed, **params) -> TrueGrid``.

    ``fixed_map`` is the repository-relative map file of the ``deck_*`` scenarios (whose
    generator ignores rows, cols and seed) and ``None`` for generated ones.
    """

    name: str
    description: str
    generator: Callable[..., TrueGrid]
    fixed_map: str | None = None


# --- shared helpers --------------------------------------------------------------------------
def _make_rng(seed: int) -> np.random.Generator:
    """The mapgen stream for ``seed`` (independent of the World's spawned streams)."""
    return np.random.default_rng(np.random.SeedSequence([seed, MAPGEN_STREAM_KEY]))


def _check_size_and_seed(rows: int, cols: int, seed: int) -> None:
    """``ValueError`` unless both sizes are ints >= ``MAPGEN_MIN_SIZE`` and seed is an int >= 0."""
    for label, value in (("rows", rows), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{label} must be an int, got {value!r}")
        if value < MAPGEN_MIN_SIZE:
            raise ValueError(
                f"generated maps need at least {MAPGEN_MIN_SIZE}x{MAPGEN_MIN_SIZE} cells, "
                f"got {label}={value}"
            )
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError(f"seed must be an int >= 0, got {seed!r}")


def _check_range(label: str, value: float, bounds: tuple[float, float]) -> float:
    """``value`` as a float if it is a real number in ``bounds``, else ``ValueError``.

    Bools and strings are rejected like non-int sizes are (``float(True)`` would pass).
    """
    lo, hi = bounds
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{label} must be a number, got {value!r}")
    v = float(value)
    if not lo <= v <= hi:
        raise ValueError(f"{label} must be in [{lo}, {hi}], got {value}")
    return v


def min_separation(rows: int, cols: int) -> int:
    """Required start-survivor Manhattan distance on a ``rows`` x ``cols`` map (unrelaxed)."""
    raw = MAPGEN_MIN_SEPARATION_FRACTION * (rows + cols - 2)
    return max(1, math.ceil(raw - _SEPARATION_EPS))


def _hold(attempts: int) -> int:
    """Attempts made with the full requirements before any relaxation starts."""
    return int(attempts * MAPGEN_RELAX_START_FRACTION)


def _separation(full: int, attempt: int, attempts: int) -> int:
    """Separation required on ``attempt``: ``full``, then relaxed linearly down to 1."""
    hold = _hold(attempts)
    if attempt < hold:
        return full
    progress = (attempt - hold + 1) / max(1, attempts - hold)
    return max(1, math.ceil(full * (1.0 - progress) - _SEPARATION_EPS))


def _components(passable: np.ndarray) -> np.ndarray:
    """4-connected component labels of ``passable`` (0 = blocked, 1.. = components)."""
    labels, _ = ndimage.label(passable)
    return labels


def _place(
    rng: np.random.Generator,
    start_mask: np.ndarray,
    goal_mask: np.ndarray,
    separation: int,
    labels: np.ndarray | None = None,
    accept: Callable[[Pos], np.ndarray] | None = None,
    tries: int = 1,
) -> tuple[Pos, Pos] | None:
    """Pick a start in ``start_mask`` and a survivor in ``goal_mask`` ``separation`` apart.

    The start is uniform over the cells that have at least one partner; the survivor uniform
    over that start's partners. With ``labels`` the two must share a component label. With
    ``accept`` (start -> bool mask of acceptable survivor cells) up to ``tries`` distinct
    starts are tried in random order until one has an accepted partner. Returns ``None`` when
    no pair qualifies.
    """
    starts = np.argwhere(start_mask)
    goals = np.argwhere(goal_mask)
    if len(starts) == 0 or len(goals) == 0:
        return None
    if labels is None:
        s_lab = np.zeros(len(starts), dtype=np.intp)
        g_lab = np.zeros(len(goals), dtype=np.intp)
        n_labels = 1
    else:
        s_lab = labels[starts[:, 0], starts[:, 1]].astype(np.intp)
        g_lab = labels[goals[:, 0], goals[:, 1]].astype(np.intp)
        n_labels = int(labels.max()) + 1
    # Farthest goal of the same label, via the extremes of r + c and r - c (L1 diameter).
    big = 4 * (start_mask.shape[0] + start_mask.shape[1])
    g_sum, g_diff = goals[:, 0] + goals[:, 1], goals[:, 0] - goals[:, 1]
    hi_sum, lo_sum = np.full(n_labels, -big), np.full(n_labels, big)
    hi_diff, lo_diff = np.full(n_labels, -big), np.full(n_labels, big)
    np.maximum.at(hi_sum, g_lab, g_sum)
    np.minimum.at(lo_sum, g_lab, g_sum)
    np.maximum.at(hi_diff, g_lab, g_diff)
    np.minimum.at(lo_diff, g_lab, g_diff)
    s_sum, s_diff = starts[:, 0] + starts[:, 1], starts[:, 0] - starts[:, 1]
    farthest = np.maximum.reduce([
        hi_sum[s_lab] - s_sum, s_sum - lo_sum[s_lab],
        hi_diff[s_lab] - s_diff, s_diff - lo_diff[s_lab],
    ])
    valid = np.flatnonzero(farthest >= separation)
    if valid.size == 0:
        return None
    if accept is None:
        order = [int(valid[rng.integers(valid.size)])]
    else:
        order = [int(i) for i in rng.permutation(valid)[:tries]]
    for i in order:
        start = starts[i]
        start_pos = (int(start[0]), int(start[1]))
        dist = np.abs(goals - start).sum(axis=1)
        ok = (dist >= separation) & (g_lab == s_lab[i])
        if accept is not None:
            ok &= accept(start_pos)[goals[:, 0], goals[:, 1]]
        partners = np.flatnonzero(ok)
        if partners.size == 0:
            continue
        j = int(partners[rng.integers(partners.size)])
        return start_pos, (int(goals[j][0]), int(goals[j][1]))
    return None


def _finish(
    debris: np.ndarray, high_slip: np.ndarray, start: Pos, survivor: Pos, name: str
) -> TrueGrid | None:
    """The grid, or ``None`` if the survivor is not BFS-reachable from the start."""
    if not reachable(~debris, start, survivor):
        return None
    cells = np.where(debris, np.int8(Cell.DEBRIS), np.int8(Cell.OPEN)).astype(np.int8)
    cells[survivor] = Cell.SURVIVOR
    return TrueGrid(cells=cells, start=start, survivor=survivor, high_slip=high_slip, name=name)


def _retry(name: str, rows: int, cols: int, seed: int, attempt: _Attempt) -> TrueGrid:
    """Run ``attempt`` on one stream until it yields a grid (``MapGenerationError`` if never)."""
    rng = _make_rng(seed)
    full = min_separation(rows, cols)
    attempts = MAPGEN_MAX_ATTEMPTS
    hold = _hold(attempts)
    for k in range(attempts):
        grid = attempt(rng, _separation(full, k, attempts), k < hold)
        if grid is not None:
            return grid
    raise MapGenerationError(
        f"{name}: no solvable {rows}x{cols} map for seed {seed} in {attempts} attempts"
    )


# --- random_rubble ---------------------------------------------------------------------------
def random_rubble(
    rows: int, cols: int, seed: int, density: float = DEFAULT_RUBBLE_DENSITY
) -> TrueGrid:
    """Independent debris on each cell with probability ``density`` (``RUBBLE_DENSITY_RANGE``)."""
    _check_size_and_seed(rows, cols, seed)
    p = _check_range("density", density, RUBBLE_DENSITY_RANGE)
    no_slip = np.zeros((rows, cols), dtype=bool)

    def attempt(rng: np.random.Generator, separation: int, strict: bool) -> TrueGrid | None:
        debris = rng.random((rows, cols)) < p
        free = ~debris
        placed = _place(rng, free, free, separation, _components(free))
        if placed is None:
            return None
        return _finish(debris, no_slip, placed[0], placed[1], "random_rubble")

    return _retry("random_rubble", rows, cols, seed, attempt)


# --- collapsed_building ----------------------------------------------------------------------
def _room_count(lo: int, hi: int, room_size: int) -> int:
    """Rooms between outer walls ``lo`` and ``hi``: about ``(hi - lo) / (room_size + 1)``, at
    least two when they fit at ``BUILDING_MIN_ROOM_SIZE``."""
    inner = hi - lo - 1
    k_max = max(1, (inner + 1) // (BUILDING_MIN_ROOM_SIZE + 1))
    k = round((inner + 1) / (room_size + 1))
    return min(k_max, max(k, min(2, k_max)))


def _hall_fits(lo: int, hi: int, room_size: int) -> bool:
    """True if a ``BUILDING_HALL_WIDTH`` hallway (plus its extra wall) fits between ``lo`` and
    ``hi`` beside at least two rooms of ``BUILDING_MIN_ROOM_SIZE``."""
    k = _room_count(lo, hi, room_size)
    room_cells = (hi - lo - 1) - k - BUILDING_HALL_WIDTH
    return k >= 2 and room_cells >= k * BUILDING_MIN_ROOM_SIZE


def _hallways(rng: np.random.Generator, row_fits: bool, col_fits: bool) -> tuple[bool, bool]:
    """Which axes get a hallway: each one that fits with ``BUILDING_HALL_PROB``, and at least
    one (a random fitting axis) whenever any fits. Draws nothing when none fits."""
    fits = [row_fits, col_fits]
    chosen = [bool(f and rng.random() < BUILDING_HALL_PROB) for f in fits]
    if any(fits) and not any(chosen):
        candidates = [i for i, f in enumerate(fits) if f]
        chosen[candidates[int(rng.integers(len(candidates)))]] = True
    return chosen[0], chosen[1]


def _partition(
    rng: np.random.Generator, lo: int, hi: int, room_size: int, hall: bool = False
) -> tuple[list[int], int | None]:
    """Wall lines from outer wall ``lo`` to outer wall ``hi``, and the hallway band (or None).

    Splits the inside into :func:`_room_count` rooms with sizes as equal as possible plus a
    random jitter. With ``hall`` (only when :func:`_hall_fits`) a ``BUILDING_HALL_WIDTH`` band
    is inserted between two random neighbouring rooms; band ``b`` lies between ``lines[b]``
    and ``lines[b + 1]``.
    """
    inner = hi - lo - 1
    k = _room_count(lo, hi, room_size)
    total = inner - (k - 1) - (BUILDING_HALL_WIDTH + 1 if hall else 0)
    sizes = np.full(k, total // k, dtype=int)
    sizes[rng.choice(k, total % k, replace=False)] += 1
    for i in range(k - 1):
        shift = int(rng.integers(-BUILDING_ROOM_JITTER, BUILDING_ROOM_JITTER + 1))
        if min(sizes[i] + shift, sizes[i + 1] - shift) >= min(BUILDING_MIN_ROOM_SIZE, total):
            sizes[i] += shift
            sizes[i + 1] -= shift
    bands = [int(size) for size in sizes]
    hall_band: int | None = None
    if hall:
        hall_band = int(rng.integers(1, k))
        bands.insert(hall_band, BUILDING_HALL_WIDTH)
    lines = [lo]
    for size in bands:
        lines.append(lines[-1] + size + 1)
    return lines, hall_band


def _segments(
    row_lines: Sequence[int],
    col_lines: Sequence[int],
    exterior: bool,
    row_hall: int | None = None,
    col_hall: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Wall segments between adjacent rooms (interior), of the outer walls, and hallway ends.

    Each segment is an ``(n, 2)`` array of the wall cells strictly between two junctions.
    Interior wall lines crossing a hallway band (row band ``row_hall`` / column band
    ``col_hall``) are cut there and yield no segment. Exterior segments are returned only when
    ``exterior`` (there is a street to open onto); the third list holds the indices into the
    exterior list of the segments that close a hallway's ends.
    """
    interior: list[np.ndarray] = []
    outer: list[np.ndarray] = []
    hall_ends: list[int] = []
    kr, kc = len(row_lines) - 1, len(col_lines) - 1
    for a in range(kr):
        rs = np.arange(row_lines[a] + 1, row_lines[a + 1])
        for b in range(kc + 1):
            seg = np.column_stack([rs, np.full(rs.size, col_lines[b])])
            if 0 < b < kc:
                if a != row_hall:
                    interior.append(seg)
            elif exterior:
                if a == row_hall:
                    hall_ends.append(len(outer))
                outer.append(seg)
    for b in range(kc):
        cs = np.arange(col_lines[b] + 1, col_lines[b + 1])
        for a in range(kr + 1):
            seg = np.column_stack([np.full(cs.size, row_lines[a]), cs])
            if 0 < a < kr:
                if b != col_hall:
                    interior.append(seg)
            elif exterior:
                if b == col_hall:
                    hall_ends.append(len(outer))
                outer.append(seg)
    return interior, outer, hall_ends


def _repair(
    rng: np.random.Generator, debris: np.ndarray, collapsed: Sequence[Pos], start: Pos,
    survivor: Pos,
) -> bool:
    """Re-open collapsed doorways on the start component's boundary until connected.

    Only doorways with one side in the start's component and the other in a different
    component qualify; among them, those whose other side is the survivor's component are
    preferred (one re-opened door then reconnects), else one is picked at random. Mutates
    ``debris``; returns False if the survivor stays cut off.
    """
    rows, cols = debris.shape
    labels = _components(~debris)
    while labels[start] != labels[survivor]:
        home, target = int(labels[start]), int(labels[survivor])
        cut: list[Pos] = []
        direct: list[Pos] = []
        for p in collapsed:
            if not debris[p]:
                continue
            sides = {
                int(labels[p[0] + dr, p[1] + dc])
                for dr, dc in _DIRS
                if 0 <= p[0] + dr < rows and 0 <= p[1] + dc < cols
            }
            if home in sides and len(sides - {0, home}) > 0:
                cut.append(p)
                if target in sides:
                    direct.append(p)
        pool = direct or cut
        if not pool:
            return False
        debris[pool[int(rng.integers(len(pool)))]] = False
        labels = _components(~debris)
    return True


def collapsed_building(
    rows: int,
    cols: int,
    seed: int,
    room_size: int = BUILDING_ROOM_SIZE,
    collapse_fraction: float = BUILDING_COLLAPSE_FRACTION,
    rubble_density: float = BUILDING_RUBBLE_DENSITY,
) -> TrueGrid:
    """A street around a building of rooms (about ``room_size`` cells per side) and hallways.

    Rooms are separated by 1-cell walls; every wall between two rooms has a 1-cell doorway
    (``BUILDING_MERGE_FRACTION`` of them have fallen entirely, merging the rooms), some outer
    walls have entrances. Along each axis with room for one (:func:`_hall_fits`), a
    ``BUILDING_HALL_WIDTH`` hallway runs right across the building with probability
    ``BUILDING_HALL_PROB`` (at least one hallway whenever one fits, :func:`_hallways`): every
    room beside it has a doorway onto it and both its ends are entrances.
    ``collapse_fraction`` of all doorways (entrances included) are blocked by debris and the
    rooms and hallways hold ``rubble_density`` rubble away from the doorways. The robot
    starts in the street and the survivor is in a room (never a hallway); if it is cut off,
    collapsed doorways on the cut are re-opened (:func:`_repair`). Maps too small for a street
    (``min(rows, cols) - 2 * BUILDING_STREET_WIDTH < BUILDING_MIN_ROOM_SIZE + 1``) have neither
    street nor hallways; the robot then starts in a room.
    """
    _check_size_and_seed(rows, cols, seed)
    if isinstance(room_size, bool) or not isinstance(room_size, int | np.integer):
        raise ValueError(f"room_size must be an int, got {room_size!r}")
    if room_size < BUILDING_MIN_ROOM_SIZE:
        raise ValueError(f"room_size must be >= {BUILDING_MIN_ROOM_SIZE}, got {room_size}")
    collapse = _check_range("collapse_fraction", collapse_fraction, (0.0, 1.0))
    rubble_p = _check_range("rubble_density", rubble_density, (0.0, 1.0))
    street = BUILDING_STREET_WIDTH
    if min(rows, cols) - 2 * street < BUILDING_MIN_ROOM_SIZE + 1:
        street = 0
    no_slip = np.zeros((rows, cols), dtype=bool)
    size = int(room_size)

    def attempt(rng: np.random.Generator, separation: int, strict: bool) -> TrueGrid | None:
        top, left, bottom, right = street, street, rows - 1 - street, cols - 1 - street
        row_hall, col_hall = _hallways(rng, street > 0 and _hall_fits(top, bottom, size),
                                       street > 0 and _hall_fits(left, right, size))
        row_lines, row_band = _partition(rng, top, bottom, size, row_hall)
        col_lines, col_band = _partition(rng, left, right, size, col_hall)
        walls = np.zeros((rows, cols), dtype=bool)
        walls[row_lines, left:right + 1] = True
        walls[top:bottom + 1, col_lines] = True
        hall = np.zeros((rows, cols), dtype=bool)
        if row_band is not None:  # cut the interior column walls along the hallway
            band = slice(row_lines[row_band] + 1, row_lines[row_band + 1])
            walls[band, col_lines[1:-1]] = False
            hall[band, left + 1:right] = True
        if col_band is not None:
            band = slice(col_lines[col_band] + 1, col_lines[col_band + 1])
            walls[row_lines[1:-1], band] = False
            hall[top + 1:bottom, band] = True
        inside = np.zeros((rows, cols), dtype=bool)
        inside[top:bottom + 1, left:right + 1] = True
        rooms = inside & ~walls
        debris = walls.copy()
        interior, outer, hall_ends = _segments(
            row_lines, col_lines, street > 0, row_band, col_band)
        # Fallen walls: merge a few pairs of rooms into halls.
        n_merge = round(BUILDING_MERGE_FRACTION * len(interior))
        merged = set(rng.choice(len(interior), n_merge, replace=False).tolist())
        for i in merged:
            debris[interior[i][:, 0], interior[i][:, 1]] = False
        doors: list[Pos] = []
        for i, seg in enumerate(interior):
            cell = seg[rng.integers(len(seg))]
            if i not in merged:
                doors.append((int(cell[0]), int(cell[1])))
        if outer:
            chosen = rng.random(len(outer)) < BUILDING_ENTRANCE_PROB
            chosen[hall_ends] = True
            missing = BUILDING_MIN_ENTRANCES - int(chosen.sum())
            if missing > 0:
                extra = rng.choice(np.flatnonzero(~chosen), min(missing, int((~chosen).sum())),
                                   replace=False)
                chosen[extra] = True
            for i in np.flatnonzero(chosen):
                cell = outer[i][rng.integers(len(outer[i]))]
                doors.append((int(cell[0]), int(cell[1])))
        for p in doors:
            debris[p] = False
        n_collapse = round(collapse * len(doors))
        collapsed = [doors[i] for i in rng.choice(len(doors), n_collapse, replace=False)]
        for p in collapsed:
            debris[p] = True
        # Rubble inside the rooms, keeping the cells in front of every doorway clear.
        near_door = np.zeros((rows, cols), dtype=bool)
        for r, c in doors:
            near_door[max(0, r - 1):r + 2, c] = True
            near_door[r, max(0, c - 1):c + 2] = True
        debris |= rooms & ~near_door & (rng.random((rows, cols)) < rubble_p)
        free = ~debris
        start_mask = (free & ~inside) if street > 0 else (free & rooms)
        placed = _place(rng, start_mask, free & rooms & ~hall, separation)
        if placed is None:
            return None
        start, survivor = placed
        if not _repair(rng, debris, collapsed, start, survivor):
            return None
        return _finish(debris, no_slip, start, survivor, "collapsed_building")

    return _retry("collapsed_building", rows, cols, seed, attempt)


# --- flooded_zone ----------------------------------------------------------------------------
def flooded_zone(
    rows: int,
    cols: int,
    seed: int,
    water_fraction: float = FLOOD_WATER_FRACTION,
    debris_density: float = FLOOD_DEBRIS_DENSITY,
) -> TrueGrid:
    """Open terrain with smooth pools of high-slip water / mud and sparse debris on dry land.

    The pools are the top ``water_fraction`` of Gaussian-smoothed white noise (sigma
    ``FLOOD_BLOB_SCALE * sqrt(rows * cols)``, at least ``FLOOD_MIN_SIGMA``), so each map has
    exactly ``round(water_fraction * rows * cols)`` water cells (this per-map threshold makes
    the border slightly wetter than the centre, see ``FLOOD_PAD_SIGMAS``); ``~`` is never on
    debris, the start or the survivor. Start and survivor are placed so that the water cannot
    be skipped for free: every shortest start-survivor route crosses water, i.e. an all-dry
    route is strictly longer or does not exist (up to ``FLOOD_CROSSING_START_TRIES`` starts per
    attempt). Like the separation, this requirement is dropped after
    ``MAPGEN_RELAX_START_FRACTION`` of the attempts (and on maps without water). Missions on it
    use ``TerrainSlip`` through ``runner.make_transition``.
    """
    _check_size_and_seed(rows, cols, seed)
    water_p = _check_range("water_fraction", water_fraction, FLOOD_WATER_RANGE)
    debris_p = _check_range("debris_density", debris_density, FLOOD_DEBRIS_RANGE)
    sigma = max(FLOOD_MIN_SIGMA, FLOOD_BLOB_SCALE * math.sqrt(rows * cols))
    pad = math.ceil(FLOOD_PAD_SIGMAS * sigma)
    n_water = round(water_p * rows * cols)

    def attempt(rng: np.random.Generator, separation: int, strict: bool) -> TrueGrid | None:
        noise = rng.standard_normal((rows + 2 * pad, cols + 2 * pad))
        field = ndimage.gaussian_filter(noise, sigma, mode="wrap")[pad:pad + rows, pad:pad + cols]
        water = np.zeros(rows * cols, dtype=bool)
        if n_water > 0:
            water[np.argsort(-field.ravel(), kind="stable")[:n_water]] = True
        water = water.reshape(rows, cols)
        debris = ~water & (rng.random((rows, cols)) < debris_p)
        free = ~debris
        dry = free & ~water
        accept: Callable[[Pos], np.ndarray] | None = None
        if strict and water.any():

            def accept(start: Pos) -> np.ndarray:
                """Survivor cells whose every shortest route from ``start`` crosses water."""
                dry_dist = bfs_distances(dry, start)
                return (dry_dist == UNREACHABLE) | (dry_dist > bfs_distances(free, start))

        placed = _place(rng, dry, dry, separation, _components(free), accept,
                        FLOOD_CROSSING_START_TRIES)
        if placed is None:
            return None
        return _finish(debris, water, placed[0], placed[1], "flooded_zone")

    return _retry("flooded_zone", rows, cols, seed, attempt)


# --- maze_corridors --------------------------------------------------------------------------
def _carve_maze(
    rng: np.random.Generator, rows: int, cols: int
) -> tuple[np.ndarray, int, int]:
    """Perfect maze (iterative recursive backtracker): ``(walls, row_offset, col_offset)``.

    Lattice cells sit at ``(row_offset + 2i, col_offset + 2j)``. On an even dimension the
    spare wall line goes on a random side (offset 1: top / left, offset 0: bottom / right).
    """
    row_off = int(rng.integers(2)) if rows % 2 == 0 else 0
    col_off = int(rng.integers(2)) if cols % 2 == 0 else 0
    lr, lc = (rows - row_off + 1) // 2, (cols - col_off + 1) // 2
    walls = np.ones((rows, cols), dtype=bool)
    visited = np.zeros((lr, lc), dtype=bool)
    draws = iter(rng.random(lr * lc).tolist())
    i0, j0 = int(rng.integers(lr)), int(rng.integers(lc))
    visited[i0, j0] = True
    walls[row_off + 2 * i0, col_off + 2 * j0] = False
    stack: list[tuple[int, int]] = [(i0, j0)]
    while stack:
        i, j = stack[-1]
        fresh = [
            (i + di, j + dj)
            for di, dj in _DIRS
            if 0 <= i + di < lr and 0 <= j + dj < lc and not visited[i + di, j + dj]
        ]
        if not fresh:
            stack.pop()
            continue
        ni, nj = fresh[int(next(draws) * len(fresh))]
        visited[ni, nj] = True
        walls[row_off + i + ni, col_off + j + nj] = False  # wall between the two cells
        walls[row_off + 2 * ni, col_off + 2 * nj] = False
        stack.append((ni, nj))
    return walls, row_off, col_off


def _inner_walls(walls: np.ndarray, row_off: int, col_off: int) -> np.ndarray:
    """Remaining walls between two lattice cells (knock-out candidates; never the posts)."""
    rows, cols = walls.shape
    r = np.arange(rows)[:, None] - row_off
    c = np.arange(cols)[None, :] - col_off
    lattice_r = (r >= 0) & (r % 2 == 0)
    lattice_c = (c >= 0) & (c % 2 == 0)
    between_r = (r >= 1) & (r % 2 == 1) & (r + row_off + 1 < rows)
    between_c = (c >= 1) & (c % 2 == 1) & (c + col_off + 1 < cols)
    return walls & ((lattice_r & between_c) | (between_r & lattice_c))


def maze_corridors(
    rows: int, cols: int, seed: int, loop_fraction: float = MAZE_LOOP_FRACTION
) -> TrueGrid:
    """A perfect maze of 1-cell corridors with ``loop_fraction`` of inner walls knocked out.

    The recursive backtracker (explicit stack) gives long winding corridors and long dead
    ends; the knocked-out walls add a few loops.
    """
    _check_size_and_seed(rows, cols, seed)
    loops = _check_range("loop_fraction", loop_fraction, (0.0, 1.0))
    no_slip = np.zeros((rows, cols), dtype=bool)

    def attempt(rng: np.random.Generator, separation: int, strict: bool) -> TrueGrid | None:
        debris, row_off, col_off = _carve_maze(rng, rows, cols)
        candidates = np.argwhere(_inner_walls(debris, row_off, col_off))
        n_knock = round(loops * len(candidates))
        if n_knock:
            picked = candidates[rng.choice(len(candidates), n_knock, replace=False)]
            debris[picked[:, 0], picked[:, 1]] = False
        free = ~debris
        placed = _place(rng, free, free, separation, _components(free))
        if placed is None:
            return None
        return _finish(debris, no_slip, placed[0], placed[1], "maze_corridors")

    return _retry("maze_corridors", rows, cols, seed, attempt)


# --- fixed deck maps -------------------------------------------------------------------------
def _resolve_map(path: str) -> Path:
    """``path`` relative to the working directory if it exists, else to the repository."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    return _REPO_ROOT / p


def _fixed_map(name: str, path: str) -> Callable[..., TrueGrid]:
    """Generator that loads ``path`` and ignores rows, cols and seed."""

    def load(rows: int = 0, cols: int = 0, seed: int = 0, **params: Any) -> TrueGrid:
        if params:
            raise ValueError(f"{name} is a fixed map and takes no parameters: {sorted(params)}")
        grid = TrueGrid.load(_resolve_map(path))
        grid.name = name
        return grid

    load.__name__ = name
    load.__doc__ = f"Load the fixed map {path} (rows, cols and seed are ignored)."
    return load


# --- registry --------------------------------------------------------------------------------
SCENARIOS: dict[str, ScenarioSpec] = {
    "random_rubble": ScenarioSpec(
        "random_rubble", "Independent rubble on every cell (density 0.15-0.35).", random_rubble
    ),
    "collapsed_building": ScenarioSpec(
        "collapsed_building",
        "Rooms and hallways behind 1-cell walls with narrow doorways, some collapsed; start in "
        "the street.",
        collapsed_building,
    ),
    "flooded_zone": ScenarioSpec(
        "flooded_zone",
        "Open ground with pools of high-slip water / mud ('~') on every shortest route and "
        "sparse debris.",
        flooded_zone,
    ),
    "maze_corridors": ScenarioSpec(
        "maze_corridors", "Winding 1-cell corridors with long dead ends and a few loops.",
        maze_corridors,
    ),
    "deck_demo": ScenarioSpec(
        "deck_demo", "The 4x4 slide 9 sample-run map.",
        _fixed_map("deck_demo", "maps/deck_demo.txt"), "maps/deck_demo.txt",
    ),
    "deck_figure": ScenarioSpec(
        "deck_figure", "The 4x4 slide 1-2 figure map.",
        _fixed_map("deck_figure", "maps/deck_figure.txt"), "maps/deck_figure.txt",
    ),
}
#: Names of the generated (seeded, sized) scenarios, in registry order.
GENERATED_SCENARIOS: tuple[str, ...] = tuple(n for n, s in SCENARIOS.items() if s.fixed_map is None)


def generate(name: str, rows: int, cols: int, seed: int, **params: Any) -> TrueGrid:
    """Build scenario ``name`` (``ValueError`` if unknown); the grid is named ``name``.

    ``deck_*`` scenarios load their fixed map and ignore ``rows``, ``cols`` and ``seed``.
    """
    spec = SCENARIOS.get(name)
    if spec is None:
        raise ValueError(f"unknown scenario {name!r}; choose from {', '.join(SCENARIOS)}")
    grid = spec.generator(rows, cols, seed, **params)
    grid.name = name
    return grid


# --- CLI -------------------------------------------------------------------------------------
def _parse_param(text: str) -> tuple[str, int | float]:
    """``KEY=VALUE`` with an int or float value."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, int(value)
    except ValueError:
        try:
            return key, float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{key}: not a number: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the scenario generator."""
    rows, cols = MAPGEN_DEFAULT_SIZE
    parser = argparse.ArgumentParser(
        prog="python -m rescue.mapgen",
        description="Generate a scenario map and print it in the ASCII map format.",
    )
    parser.add_argument("name", choices=list(SCENARIOS), help="scenario")
    parser.add_argument("--rows", type=int, default=rows, help="rows (generated scenarios)")
    parser.add_argument("--cols", type=int, default=cols, help="columns (generated scenarios)")
    parser.add_argument("--seed", type=int, default=MAPGEN_DEFAULT_SEED, help="map seed")
    parser.add_argument("--param", action="append", type=_parse_param, default=[],
                        metavar="KEY=VALUE", help="generator parameter, e.g. density=0.3")
    parser.add_argument("--out", metavar="PATH", help="also write the map (with a header)")
    return parser


def map_header(name: str, rows: int, cols: int, seed: int, params: dict[str, Any]) -> str:
    """The ``;`` comment lines recording how a map file was generated."""
    extra = "".join(f" --param {k}={v}" for k, v in params.items())
    spec = SCENARIOS[name]
    if spec.fixed_map is not None:
        return f"; {name}: fixed map {spec.fixed_map}\n"
    return (
        f"; {name} {rows}x{cols}, seed {seed}{', ' if params else ''}"
        f"{', '.join(f'{k}={v}' for k, v in params.items())}\n"
        f"; python -m rescue.mapgen {name} --rows {rows} --cols {cols} --seed {seed}{extra}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m rescue.mapgen``."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    params = dict(args.param)
    try:
        grid = generate(args.name, args.rows, args.cols, args.seed, **params)
    except (ValueError, TypeError, MapGenerationError) as exc:
        build_parser().error(str(exc))
    text = grid.to_ascii()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = map_header(args.name, args.rows, args.cols, args.seed, params)
        out.write_text(header + text, encoding="utf-8", newline="\n")  # LF on every OS
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
