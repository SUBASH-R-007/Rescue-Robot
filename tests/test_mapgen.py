"""Phase 5: the scenario library (``rescue.mapgen``, brief §6.3, plan §1).

Acceptance criterion (brief §9): every scenario of :data:`SCENARIOS` yields a solvable map
for 100 seeds each — the generated ones at several sizes (square, non-square, the 5x5
minimum), the ``deck_*`` ones as fixed maps. Further contract checks: determinism, the
start / survivor rules (distinct free cells, far apart, BFS-connected), generator-specific
structure (rubble density, flood blobs, maze dead ends and corridors, building choke points),
error handling (unknown scenario, too-small sizes, bad parameters, ``MapGenerationError``
after the capped attempts), the CLI and the committed sample maps.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from rescue import config
from rescue.grid import Cell, Pos, TrueGrid, manhattan, neighbours4
from rescue.planners.bfs import reachable, shortest_path_length

mapgen = pytest.importorskip("rescue.mapgen")

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"

EXPECTED_SCENARIOS = (
    "random_rubble",
    "collapsed_building",
    "flooded_zone",
    "maze_corridors",
    "deck_demo",
    "deck_figure",
)
GENERATED = EXPECTED_SCENARIOS[:4]
FIXED = EXPECTED_SCENARIOS[4:]
#: Seeds of the acceptance criterion (100 per scenario and size).
ACCEPT_SEEDS = range(100)
#: (rows, cols) sizes of the acceptance sweep: square, non-square, the minimum.
ACCEPT_SIZES = ((12, 12), (10, 17), (5, 5))
#: Sizes on which the separation rule must hold strictly (not relaxed).
STRICT_SIZES = ((12, 12), (10, 17))
#: Committed sample maps and the scenario each one comes from.
SAMPLE_MAPS = {
    "rubble_01.txt": "random_rubble",
    "building_01.txt": "collapsed_building",
    "flooded_01.txt": "flooded_zone",
    "maze_01.txt": "maze_corridors",
}


# --- helpers ---------------------------------------------------------------------------------
def free_mask(grid: TrueGrid) -> np.ndarray:
    """True where the true map is passable."""
    return grid.passable_mask()


def free_degree(grid: TrueGrid, p: Pos) -> int:
    """Number of free 4-neighbours of ``p``."""
    free = free_mask(grid)
    return sum(1 for q in neighbours4(p, grid.rows, grid.cols) if free[q])


def free_cells(grid: TrueGrid) -> list[Pos]:
    """All passable cells, row-major."""
    return [(int(r), int(c)) for r, c in np.argwhere(free_mask(grid))]


def dead_ends(grid: TrueGrid) -> int:
    """Free cells with exactly one free neighbour."""
    return sum(1 for p in free_cells(grid) if free_degree(grid, p) == 1)


def dead_end_lengths(grid: TrueGrid) -> list[int]:
    """Length of every dead-end branch: cells walked from a dead end until a junction."""
    free = free_mask(grid)
    lengths = []
    for p in free_cells(grid):
        if free_degree(grid, p) != 1:
            continue
        prev, cur, length = None, p, 1
        while True:
            nxt = [q for q in neighbours4(cur, grid.rows, grid.cols) if free[q] and q != prev]
            if len(nxt) != 1 or free_degree(grid, nxt[0]) != 2:
                break
            prev, cur, length = cur, nxt[0], length + 1
        lengths.append(length)
    return lengths


def component(grid: TrueGrid, src: Pos) -> set[Pos]:
    """Free cells 4-connected to ``src``."""
    free = free_mask(grid)
    seen = {src}
    queue = deque([src])
    while queue:
        p = queue.popleft()
        for q in neighbours4(p, grid.rows, grid.cols):
            if free[q] and q not in seen:
                seen.add(q)
                queue.append(q)
    return seen


def articulation_points(grid: TrueGrid, src: Pos) -> set[Pos]:
    """Cut vertices of the free-cell graph of ``src``'s component (iterative Tarjan)."""
    free = free_mask(grid)
    rows, cols = grid.rows, grid.cols
    disc: dict[Pos, int] = {src: 0}
    low: dict[Pos, int] = {src: 0}
    parent: dict[Pos, Pos | None] = {src: None}
    children: dict[Pos, int] = {src: 0}
    points: set[Pos] = set()
    counter = 1
    stack: list[tuple[Pos, list[Pos]]] = [
        (src, [q for q in neighbours4(src, rows, cols) if free[q]])
    ]
    while stack:
        p, pending = stack[-1]
        if pending:
            q = pending.pop()
            if q not in disc:
                disc[q] = low[q] = counter
                counter += 1
                parent[q] = p
                children[q] = 0
                children[p] += 1
                stack.append((q, [n for n in neighbours4(q, rows, cols) if free[n]]))
            elif q != parent[p]:
                low[p] = min(low[p], disc[q])
            continue
        stack.pop()
        par = parent[p]
        if par is not None:
            low[par] = min(low[par], low[p])
            if parent[par] is not None and low[p] >= disc[par]:
                points.add(par)
    if children[src] > 1:
        points.add(src)
    return points


def min_separation(rows: int, cols: int) -> int:
    """The contract's minimum start-survivor Manhattan distance (not relaxed)."""
    return math.ceil(config.MAPGEN_MIN_SEPARATION_FRACTION * (rows + cols - 2) - 1e-9)


def check_valid(grid: TrueGrid, rows: int, cols: int, *, strict_separation: bool) -> None:
    """Every generated map: shape, start/survivor rules and solvability on the true map."""
    assert grid.shape == (rows, cols)
    assert grid.start != grid.survivor
    assert grid.cells[grid.start] == Cell.OPEN
    assert grid.cells[grid.survivor] == Cell.SURVIVOR
    assert int(np.count_nonzero(grid.cells == Cell.SURVIVOR)) == 1
    assert not grid.high_slip[grid.start] and not grid.high_slip[grid.survivor]
    assert not (grid.high_slip & (grid.cells == Cell.DEBRIS)).any()
    assert reachable(grid.passable_mask(), grid.start, grid.survivor)
    if strict_separation:
        assert manhattan(grid.start, grid.survivor) >= min_separation(rows, cols)
    # Round trip through the ASCII format (no '?' or other characters in true maps).
    text = grid.to_ascii()
    assert set(text) <= set(".#SP~\n")
    assert TrueGrid.from_ascii(text, name=grid.name) == grid


# --- registry --------------------------------------------------------------------------------
def test_registry_order_and_specs() -> None:
    assert tuple(mapgen.SCENARIOS) == EXPECTED_SCENARIOS
    assert tuple(mapgen.GENERATED_SCENARIOS) == GENERATED
    for name, spec in mapgen.SCENARIOS.items():
        assert isinstance(spec, mapgen.ScenarioSpec)
        assert spec.name == name
        assert isinstance(spec.description, str) and spec.description
        assert callable(spec.generator)
        if name in FIXED:
            assert spec.fixed_map is not None
            assert Path(spec.fixed_map).name == f"{name}.txt"
        else:
            assert spec.fixed_map is None
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass
        mapgen.SCENARIOS["random_rubble"].name = "x"  # type: ignore[misc]


def test_map_generation_error_is_runtime_error() -> None:
    assert issubclass(mapgen.MapGenerationError, RuntimeError)


def test_config_constants_present() -> None:
    assert config.MAPGEN_MAX_ATTEMPTS >= 1
    assert 0.0 < config.MAPGEN_MIN_SEPARATION_FRACTION <= 1.0
    assert 0.0 < config.FLOOD_WATER_FRACTION < 1.0
    lo, hi = config.RUBBLE_DENSITY_RANGE
    assert lo <= config.DEFAULT_RUBBLE_DENSITY <= hi
    assert isinstance(config.MAPGEN_STREAM_KEY, int)


# --- acceptance: 100 seeds per scenario --------------------------------------------------------
@pytest.mark.parametrize("size", ACCEPT_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", GENERATED)
def test_generated_scenarios_solvable_for_100_seeds(name: str, size: tuple[int, int]) -> None:
    rows, cols = size
    strict = size in STRICT_SIZES
    for seed in ACCEPT_SEEDS:
        grid = mapgen.generate(name, rows, cols, seed)
        assert grid.name == name
        check_valid(grid, rows, cols, strict_separation=strict)


@pytest.mark.parametrize("name", GENERATED)
def test_generated_scenarios_larger_maps(name: str) -> None:
    for seed in range(10):
        grid = mapgen.generate(name, 20, 30, seed)
        check_valid(grid, 20, 30, strict_separation=True)


@pytest.mark.parametrize("name", FIXED)
def test_fixed_scenarios_solvable_for_100_seeds(name: str) -> None:
    expected = TrueGrid.load(MAPS_DIR / f"{name}.txt")
    for seed in ACCEPT_SEEDS:
        # rows / cols / seed are ignored for the deck maps.
        grid = mapgen.generate(name, 12 + seed % 3, 9, seed)
        assert grid.name == name
        assert np.array_equal(grid.cells, expected.cells)
        assert np.array_equal(grid.high_slip, expected.high_slip)
        assert (grid.start, grid.survivor) == (expected.start, expected.survivor)
        assert reachable(grid.passable_mask(), grid.start, grid.survivor)


@pytest.mark.parametrize("name", GENERATED)
def test_tiny_maps_relax_separation_gracefully(name: str) -> None:
    for seed in range(20):
        grid = mapgen.generate(name, 5, 5, seed)
        check_valid(grid, 5, 5, strict_separation=False)
        assert manhattan(grid.start, grid.survivor) >= 2


# --- determinism -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", GENERATED)
def test_same_arguments_same_grid(name: str) -> None:
    for seed in (0, 7, 123):
        a = mapgen.generate(name, 12, 12, seed)
        b = mapgen.generate(name, 12, 12, seed)
        assert a == b
        assert a is not b


@pytest.mark.parametrize("name", GENERATED)
def test_different_seeds_different_grids(name: str) -> None:
    texts = {mapgen.generate(name, 12, 12, seed).to_ascii() for seed in range(20)}
    assert len(texts) == 20


@pytest.mark.parametrize("name", GENERATED)
def test_generators_ignore_global_numpy_state(name: str) -> None:
    np.random.seed(1)
    a = mapgen.generate(name, 12, 12, 5)
    np.random.seed(2)
    _ = np.random.random(100)
    b = mapgen.generate(name, 12, 12, 5)
    assert a == b


def test_generate_matches_direct_generator_call() -> None:
    for name in GENERATED:
        fn = getattr(mapgen, name)
        assert mapgen.SCENARIOS[name].generator is fn
        direct = fn(12, 12, 3)
        via = mapgen.generate(name, 12, 12, 3)
        assert np.array_equal(direct.cells, via.cells)
        assert np.array_equal(direct.high_slip, via.high_slip)
        assert (direct.start, direct.survivor) == (via.start, via.survivor)
        assert via.name == name


def test_generate_passes_params() -> None:
    lo, hi = config.RUBBLE_DENSITY_RANGE
    sparse = mapgen.generate("random_rubble", 30, 30, 4, density=lo)
    dense = mapgen.generate("random_rubble", 30, 30, 4, density=hi)
    assert np.count_nonzero(sparse.cells == Cell.DEBRIS) < np.count_nonzero(
        dense.cells == Cell.DEBRIS
    )
    assert sparse.name == "random_rubble"


def test_size_and_params_change_the_map() -> None:
    a = mapgen.generate("random_rubble", 12, 12, 1)
    b = mapgen.generate("random_rubble", 12, 13, 1)
    assert a.shape != b.shape
    c = mapgen.generate("random_rubble", 12, 12, 1, density=0.3)
    assert not np.array_equal(a.cells, c.cells)


# --- errors ------------------------------------------------------------------------------------
def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError):
        mapgen.generate("volcano", 12, 12, 0)


@pytest.mark.parametrize("name", GENERATED)
@pytest.mark.parametrize("size", [(4, 12), (12, 4), (4, 4), (1, 30), (0, 0)])
def test_too_small_sizes_raise(name: str, size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        mapgen.generate(name, size[0], size[1], 0)
    with pytest.raises(ValueError):
        getattr(mapgen, name)(size[0], size[1], 0)


@pytest.mark.parametrize("density", [0.0, 0.1, 0.149, 0.351, 0.5, 1.0, -0.2])
def test_rubble_density_outside_range_raises(density: float) -> None:
    with pytest.raises(ValueError):
        mapgen.random_rubble(12, 12, 0, density=density)
    with pytest.raises(ValueError):
        mapgen.generate("random_rubble", 12, 12, 0, density=density)


def test_rubble_density_range_endpoints_accepted() -> None:
    lo, hi = config.RUBBLE_DENSITY_RANGE
    for density in (lo, hi):
        check_valid(mapgen.random_rubble(12, 12, 0, density=density), 12, 12,
                    strict_separation=True)


def test_map_generation_error_after_capped_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every solvability check failing, generators give up with MapGenerationError."""
    calls: list[int] = []

    def never(*_args: object, **_kwargs: object) -> bool:
        calls.append(1)
        return False

    patched = False
    for attr in ("reachable", "shortest_path_length", "bfs_distances"):
        if hasattr(mapgen, attr):
            patched = True
            if attr == "reachable":
                monkeypatch.setattr(mapgen, attr, never)
            elif attr == "shortest_path_length":
                monkeypatch.setattr(mapgen, attr, lambda *a, **k: (calls.append(1), None)[1])
            else:
                def unreachable_everywhere(passable: np.ndarray, *_a: object) -> np.ndarray:
                    calls.append(1)
                    return np.full(np.shape(passable), -1, dtype=np.int32)

                monkeypatch.setattr(mapgen, attr, unreachable_everywhere)
    if not patched:
        pytest.skip("mapgen does not import a BFS helper by name")
    monkeypatch.setattr(mapgen, "MAPGEN_MAX_ATTEMPTS", 3, raising=False)
    monkeypatch.setattr(config, "MAPGEN_MAX_ATTEMPTS", 3)
    for name in GENERATED:
        calls.clear()
        with pytest.raises(mapgen.MapGenerationError):
            mapgen.generate(name, 8, 8, 0)
        assert calls, f"{name} never checked solvability"


# --- start / survivor --------------------------------------------------------------------------
@pytest.mark.parametrize("name", GENERATED)
def test_start_and_survivor_far_apart_and_connected(name: str) -> None:
    for seed in range(30):
        grid = mapgen.generate(name, 15, 11, seed)
        assert manhattan(grid.start, grid.survivor) >= min_separation(15, 11)
        assert grid.survivor in component(grid, grid.start)
        length = shortest_path_length(grid.passable_mask(), grid.start, grid.survivor)
        assert length is not None and length >= manhattan(grid.start, grid.survivor)


# --- random_rubble -----------------------------------------------------------------------------
@pytest.mark.parametrize("density", [0.15, 0.25, 0.35])
def test_rubble_density_close_to_requested(density: float) -> None:
    """Law of large numbers on 40x40 maps (solvability rejection biases it only slightly)."""
    fractions = [
        float(np.mean(mapgen.random_rubble(40, 40, seed, density=density).cells == Cell.DEBRIS))
        for seed in range(6)
    ]
    assert abs(float(np.mean(fractions)) - density) < 0.025
    for f in fractions:
        assert abs(f - density) < 0.06


def test_rubble_default_density() -> None:
    grid = mapgen.random_rubble(40, 40, 11)
    frac = float(np.mean(grid.cells == Cell.DEBRIS))
    assert abs(frac - config.DEFAULT_RUBBLE_DENSITY) < 0.06
    assert not grid.high_slip.any()


# --- flooded_zone ------------------------------------------------------------------------------
def test_flooded_zone_has_water_never_on_debris() -> None:
    for seed in range(40):
        grid = mapgen.flooded_zone(12, 12, seed)
        assert grid.high_slip.any(), f"seed {seed}: no '~' cells"
        assert not (grid.high_slip & (grid.cells == Cell.DEBRIS)).any()
        assert not grid.high_slip[grid.start] and not grid.high_slip[grid.survivor]
        assert "~" in grid.to_ascii()


def test_flooded_zone_water_fraction_near_config() -> None:
    target = config.FLOOD_WATER_FRACTION
    fractions = [float(np.mean(mapgen.flooded_zone(30, 30, seed).high_slip)) for seed in range(10)]
    mean = float(np.mean(fractions))
    assert abs(mean - target) <= 0.3 * target + 0.02, (mean, target)


def test_flooded_zone_water_param_and_blobs() -> None:
    low = np.mean([mapgen.flooded_zone(30, 30, s, water_fraction=0.1).high_slip.mean()
                   for s in range(5)])
    high = np.mean([mapgen.flooded_zone(30, 30, s, water_fraction=0.4).high_slip.mean()
                    for s in range(5)])
    assert low < high
    # Blobs, not salt-and-pepper: most water cells touch another water cell.
    for seed in range(5):
        grid = mapgen.flooded_zone(30, 30, seed)
        water = [(int(r), int(c)) for r, c in np.argwhere(grid.high_slip)]
        touching = sum(
            1 for p in water if any(grid.high_slip[q] for q in neighbours4(p, 30, 30))
        )
        assert touching >= 0.8 * len(water)


# --- maze_corridors ----------------------------------------------------------------------------
def test_maze_has_many_dead_ends_and_long_corridors() -> None:
    maze_branch, rubble_branch = [], []
    for seed in range(20):
        maze = mapgen.maze_corridors(15, 15, seed)
        rubble = mapgen.random_rubble(15, 15, seed)
        assert dead_ends(maze) >= 3
        maze_branch.extend(dead_end_lengths(maze))
        rubble_branch.extend(dead_end_lengths(rubble))
        cells = free_cells(maze)
        corridor = sum(1 for p in cells if free_degree(maze, p) == 2)
        assert corridor >= 0.6 * len(cells), "maze free space should be corridors"
        # 1-cell corridors: (almost) no 2x2 block of free cells.
        free = free_mask(maze)
        blocks = free[:-1, :-1] & free[1:, :-1] & free[:-1, 1:] & free[1:, 1:]
        assert int(blocks.sum()) <= 2
    # Long dead ends: maze dead-end branches are much longer than rubble's pockets.
    assert np.mean(maze_branch) > 2 * np.mean(rubble_branch)
    assert max(maze_branch) >= 6
    ratios = []
    for seed in range(20):
        maze = mapgen.maze_corridors(15, 15, seed)
        length = shortest_path_length(maze.passable_mask(), maze.start, maze.survivor)
        assert length is not None
        ratios.append(length / manhattan(maze.start, maze.survivor))
    # Winding: the true path is much longer than the straight-line distance on average.
    assert np.mean(ratios) > 1.2


def test_maze_loop_fraction_reduces_dead_ends() -> None:
    perfect = np.mean([dead_ends(mapgen.maze_corridors(21, 21, s, loop_fraction=0.0))
                       for s in range(8)])
    loopy = np.mean([dead_ends(mapgen.maze_corridors(21, 21, s, loop_fraction=0.5))
                     for s in range(8)])
    assert loopy < perfect


# --- collapsed_building ------------------------------------------------------------------------
def test_collapsed_building_has_choke_points() -> None:
    with_points = 0
    ratios_building, ratios_rubble = [], []
    for seed in range(20):
        grid = mapgen.collapsed_building(15, 15, seed)
        if articulation_points(grid, grid.start):
            with_points += 1
        length = shortest_path_length(grid.passable_mask(), grid.start, grid.survivor)
        assert length is not None
        ratios_building.append(length / manhattan(grid.start, grid.survivor))
        rubble = mapgen.random_rubble(15, 15, seed)
        length_r = shortest_path_length(rubble.passable_mask(), rubble.start, rubble.survivor)
        assert length_r is not None
        ratios_rubble.append(length_r / manhattan(rubble.start, rubble.survivor))
    assert with_points >= 18
    assert np.mean(ratios_building) >= np.mean(ratios_rubble)


def test_collapsed_building_has_walls() -> None:
    """Rooms are separated by wall lines: some row or column is mostly debris."""
    for seed in range(10):
        grid = mapgen.collapsed_building(15, 15, seed)
        debris = grid.cells == Cell.DEBRIS
        best = max(debris.mean(axis=0).max(), debris.mean(axis=1).max())
        assert best >= 0.5


# --- CLI ---------------------------------------------------------------------------------------
def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rescue.mapgen", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )


def test_cli_prints_parseable_map() -> None:
    proc = _run_cli("random_rubble", "--rows", "9", "--cols", "11", "--seed", "3")
    assert proc.returncode == 0, proc.stderr
    grid = TrueGrid.from_ascii(proc.stdout)
    expected = mapgen.generate("random_rubble", 9, 11, 3)
    assert np.array_equal(grid.cells, expected.cells)
    assert (grid.start, grid.survivor) == (expected.start, expected.survivor)


def test_cli_out_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "flood.txt"
    proc = _run_cli("flooded_zone", "--rows", "10", "--cols", "10", "--seed", "2", "--out",
                    str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    grid = TrueGrid.load(out)
    expected = mapgen.generate("flooded_zone", 10, 10, 2)
    assert np.array_equal(grid.cells, expected.cells)
    assert np.array_equal(grid.high_slip, expected.high_slip)


def test_cli_unknown_scenario_fails() -> None:
    proc = _run_cli("volcano")
    assert proc.returncode != 0


# --- committed sample maps ---------------------------------------------------------------------
@pytest.mark.parametrize("filename", sorted(SAMPLE_MAPS))
def test_sample_maps_load_and_are_solvable(filename: str) -> None:
    path = MAPS_DIR / filename
    assert path.exists(), f"missing committed sample map {path}"
    grid = TrueGrid.load(path)
    assert grid.rows >= 5 and grid.cols >= 5
    assert reachable(grid.passable_mask(), grid.start, grid.survivor)
    if SAMPLE_MAPS[filename] == "flooded_zone":
        assert grid.high_slip.any()


@pytest.mark.parametrize("filename", sorted(SAMPLE_MAPS))
def test_sample_maps_reproduce_from_their_header(filename: str) -> None:
    """The recorded ``python -m rescue.mapgen`` command regenerates the committed map."""
    text = (MAPS_DIR / filename).read_text(encoding="utf-8")
    match = re.search(
        r"python -m rescue\.mapgen (\w+) --rows (\d+) --cols (\d+) --seed (\d+)(.*)$",
        text,
        re.MULTILINE,
    )
    if match is None:
        pytest.skip(f"{filename} has no generation header")
    name, rows, cols, seed, extra = match.groups()
    assert name == SAMPLE_MAPS[filename]
    params = {}
    for key, value in re.findall(r"--param (\w+)=(\S+)", extra):
        params[key] = float(value) if "." in value else int(value)
    expected = mapgen.generate(name, int(rows), int(cols), int(seed), **params)
    grid = TrueGrid.from_ascii(text, name=name)
    assert grid == expected


# --- fix pass: parameter types, flooded crossings, hallways, repair, line endings ---------------
@pytest.mark.parametrize(("name", "params"), [
    ("collapsed_building", {"collapse_fraction": True}),
    ("collapsed_building", {"rubble_density": False}),
    ("random_rubble", {"density": "0.3"}),
    ("flooded_zone", {"water_fraction": "0.2"}),
    ("maze_corridors", {"loop_fraction": None}),
])
def test_float_parameters_reject_bools_and_strings(name: str, params: dict) -> None:
    """``float(True)`` would silently run as 1.0: bools / strings are rejected like room_size."""
    with pytest.raises(ValueError, match="must be a number"):
        mapgen.generate(name, 12, 12, 0, **params)


def test_float_parameters_accept_ints_and_numpy_numbers() -> None:
    a = mapgen.collapsed_building(12, 12, 0, collapse_fraction=1)
    b = mapgen.collapsed_building(12, 12, 0, collapse_fraction=np.float64(1.0))
    assert a == b == mapgen.collapsed_building(12, 12, 0, collapse_fraction=1.0)


def _water_on_every_shortest_route(grid: TrueGrid) -> bool:
    """True if no all-dry start-survivor route is as short as the shortest route."""
    free = free_mask(grid)
    full = shortest_path_length(free, grid.start, grid.survivor)
    dry = shortest_path_length(free & ~grid.high_slip, grid.start, grid.survivor)
    assert full is not None
    return dry is None or dry > full


@pytest.mark.parametrize("size", [(12, 12), (10, 17), (30, 30)])
def test_flooded_zone_water_cannot_be_skipped_for_free(size: tuple[int, int]) -> None:
    """Every shortest start-survivor route crosses '~' (an all-dry route is longer or absent):
    flooded_zone exercises the high-slip terrain instead of looking like sparse rubble."""
    rows, cols = size
    for seed in range(40):
        grid = mapgen.flooded_zone(rows, cols, seed)
        assert _water_on_every_shortest_route(grid), (size, seed)


def test_flooded_zone_without_water_still_generates() -> None:
    """No water, no crossing to require: the requirement is skipped, not relaxed slowly."""
    for seed in range(10):
        grid = mapgen.flooded_zone(12, 12, seed, water_fraction=0.0)
        assert not grid.high_slip.any()
        check_valid(grid, 12, 12, strict_separation=True)


def test_hallways_at_least_one_when_it_fits() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        row, col = mapgen._hallways(rng, True, True)
        assert row or col
        assert mapgen._hallways(rng, False, True)[1]
        assert mapgen._hallways(rng, True, False)[0]
    state = rng.bit_generator.state
    assert mapgen._hallways(rng, False, False) == (False, False)
    assert rng.bit_generator.state == state  # nothing drawn when no hallway fits


def test_partition_inserts_one_hallway_band() -> None:
    rng = np.random.default_rng(1)
    lo, hi = 1, 13  # a 15-cell axis with a 1-cell street
    assert mapgen._hall_fits(lo, hi, config.BUILDING_ROOM_SIZE)
    for _ in range(50):
        lines, band = mapgen._partition(rng, lo, hi, config.BUILDING_ROOM_SIZE, True)
        assert lines[0] == lo and lines[-1] == hi
        assert band is not None and 0 < band < len(lines) - 2  # never the first / last band
        widths = np.diff(lines) - 1
        assert widths[band] == config.BUILDING_HALL_WIDTH
        rooms = np.delete(widths, band)
        assert len(rooms) >= 2 and rooms.min() >= config.BUILDING_MIN_ROOM_SIZE
    lines, band = mapgen._partition(rng, lo, hi, config.BUILDING_ROOM_SIZE, False)
    assert band is None and lines[0] == lo and lines[-1] == hi
    assert not mapgen._hall_fits(1, 8, config.BUILDING_ROOM_SIZE)  # 10x10: no room for one


def test_segments_cut_by_a_hallway() -> None:
    row_lines, col_lines = [1, 6, 8, 13], [1, 7, 13]  # row band 1 (row 7) is the hallway
    interior, outer, ends = mapgen._segments(row_lines, col_lines, True, row_hall=1)
    plain_interior, plain_outer, plain_ends = mapgen._segments(row_lines, col_lines, True)
    assert plain_ends == []
    # the column wall at col 7 is cut where it crosses the hallway (row 7)
    assert len(interior) == len(plain_interior) - 1
    assert all(not np.any((seg[:, 0] == 7) & (seg[:, 1] == 7)) for seg in interior)
    # both ends of the hallway are exterior segments
    assert len(outer) == len(plain_outer) and len(ends) == 2
    assert sorted(tuple(map(tuple, outer[i])) for i in ends) == [((7, 1),), ((7, 13),)]


@pytest.mark.parametrize("size", [(15, 15), (15, 20), (20, 20)])
def test_collapsed_building_has_a_hallway(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief §6.3 asks for rooms *and corridors*: wherever one fits, the building has a hallway
    running right across it, and the survivor is never placed in it."""
    rows, cols = size
    calls: list[tuple[list[int], int | None]] = []
    real = mapgen._partition

    def spy(*args: object, **kwargs: object) -> tuple[list[int], int | None]:
        calls.append(real(*args, **kwargs))  # type: ignore[arg-type]
        return calls[-1]

    monkeypatch.setattr(mapgen, "_partition", spy)
    for seed in range(30):
        calls.clear()
        grid = mapgen.collapsed_building(rows, cols, seed)
        (row_lines, row_band), (col_lines, col_band) = calls[-2:]  # the successful attempt
        assert row_band is not None or col_band is not None, (size, seed)
        if row_band is not None:
            assert grid.survivor[0] != row_lines[row_band] + 1
        if col_band is not None:
            assert grid.survivor[1] != col_lines[col_band] + 1


def test_repair_prefers_a_door_into_the_survivors_component() -> None:
    """Two collapsed doors on the start's boundary: one into a sealed dead end, one into the
    survivor's room. The repair opens only the direct one (fewer re-opened collapses)."""
    text = (
        "S.#..\n"
        "..#..\n"
        "#.###\n"
        "..#.P\n"
    )
    base = TrueGrid.from_ascii(text)
    debris = ~free_mask(base)
    dead_end_door, survivor_door = (0, 2), (3, 2)
    for seed in range(20):
        work = debris.copy()
        rng = np.random.default_rng(seed)
        assert mapgen._repair(rng, work, [dead_end_door, survivor_door], base.start,
                              base.survivor)
        assert work[dead_end_door] and not work[survivor_door]


@pytest.mark.parametrize("filename", sorted(SAMPLE_MAPS))
def test_sample_maps_use_lf_line_endings(filename: str) -> None:
    assert b"\r\n" not in (MAPS_DIR / filename).read_bytes()


def test_cli_out_and_save_write_lf_on_every_os(tmp_path: Path) -> None:
    out = tmp_path / "maze.txt"
    proc = _run_cli("maze_corridors", "--rows", "7", "--cols", "9", "--seed", "1", "--out",
                    str(out))
    assert proc.returncode == 0, proc.stderr
    assert b"\r\n" not in out.read_bytes()
    saved = tmp_path / "saved.txt"
    mapgen.generate("maze_corridors", 7, 9, 1).save(saved)
    assert b"\r\n" not in saved.read_bytes()
    a, b = TrueGrid.load(saved), TrueGrid.load(out)
    assert np.array_equal(a.cells, b.cells) and (a.start, a.survivor) == (b.start, b.survivor)
