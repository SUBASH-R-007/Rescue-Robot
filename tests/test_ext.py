"""Optional extensions (plan §3.14, brief §8.3–§8.5): noisy sensing, search for survivors at
unknown locations, several survivors.

Written against the §3.14 contract: ``NoisySensor`` / ``ProbBelief`` (``ext/sensing.py``),
``multi_survivor_grid`` (``ext/scenarios.py``), ``SearchAgent`` (``ext/agents.py``),
``ExtMission`` (``ext/mission.py``) and the ``python -m rescue.ext.demo`` CLI.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from rescue.belief import BeliefMap
from rescue.config import (
    EXT_DEBRIS_THRESHOLD,
    EXT_MAX_BRUTE_FORCE_K,
    EXT_MIN_SURVIVOR_SPACING,
    EXT_PRIOR_DEBRIS,
    WorldConfig,
)
from rescue.ext.agents import SearchAgent
from rescue.ext.mission import ExtMetrics, ExtMission, ExtStepEvent
from rescue.ext.scenarios import multi_survivor_grid
from rescue.ext.sensing import NOISE_STREAM_KEY, NoisySensor, ProbBelief
from rescue.grid import Cell, Pos, TrueGrid, manhattan, neighbours4
from rescue.mapgen import GENERATED_SCENARIOS, SCENARIOS, generate
from rescue.model import ACTIONS, UniformSlip
from rescue.planners.bfs import UNREACHABLE, bfs_distances, shortest_path_length
from rescue.runner import MissionStatus, make_transition
from rescue.world import Percept, World

# --- constants of the tests ----------------------------------------------------------------
#: Generated map size of the mission tests (small enough to keep the module fast).
MISSION_SIZE = 12
#: Seeds per scenario for the unknown-location mission sweep.
SEARCH_SEEDS = range(3)
#: Uniform draws used to check the empirical false-negative rate.
FLIP_DRAWS = 20_000
#: Allowed deviation of the empirical flip rate, in binomial standard deviations.
FLIP_SIGMAS = 4.0
#: Steps of the seeded random walk that compares ProbBelief(eps = 0) with BeliefMap.
WALK_STEPS = 150
#: Number of survivors of the multi-survivor tests.
MULTI_K = 3
#: Seeds of the noisy-sensing missions.
NOISY_SEEDS = range(8)
#: False-negative rate of the noisy missions.
NOISY_EPS = 0.2


# --- helpers -------------------------------------------------------------------------------


def _percept(cells: Sequence[tuple[Pos, Cell]]) -> Percept:
    return Percept(tuple(cells))


#: Grid of the hand-built sensor percepts.
SENSOR_ROWS, SENSOR_COLS = 4, 5


def _reference_flips(seed: int, percepts: Sequence[Percept], eps: float,
                     exempt: Sequence[Sequence[Pos]], rows: int = SENSOR_ROWS,
                     cols: int = SENSOR_COLS) -> list[Percept]:
    """What ``NoisySensor.observe`` must return: per call one uniform per grid cell (row-major);
    a DEBRIS cell (r, c) flips iff its own uniform ``u[r * cols + c] < eps``."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, NOISE_STREAM_KEY]))
    out: list[Percept] = []
    for percept, ex in zip(percepts, exempt, strict=True):
        block = [float(rng.random()) for _ in range(rows * cols)]
        cells: list[tuple[Pos, Cell]] = []
        for p, cell in percept.cells:
            u = block[p[0] * cols + p[1]]
            flip = cell is Cell.DEBRIS and p not in ex and u < eps
            cells.append((p, Cell.OPEN if flip else cell))
        out.append(Percept(tuple(cells)))
    return out


def _bayes_open(p: float, eps: float) -> float:
    return eps * p / (eps * p + (1.0 - p))


def _search_agent(grid: TrueGrid, planner: str, world_config: WorldConfig | None = None
                  ) -> SearchAgent:
    return SearchAgent(make_transition(grid, world_config), planner=planner)


def _run(grid: TrueGrid, planner: str, seed: int, *, world_config: WorldConfig | None = None,
         on_event: Callable[[ExtMission, ExtStepEvent], None] | None = None,
         **kwargs: object) -> ExtMission:
    agent = _search_agent(grid, planner, world_config)
    mission = ExtMission(grid, agent, world_config, seed=seed, **kwargs)  # type: ignore[arg-type]
    if on_event is None:
        mission.run()
    else:
        mission.run(lambda e: on_event(mission, e))
    return mission


def _true_dist(grid: TrueGrid, a: Pos, b: Pos) -> int:
    d = shortest_path_length(grid.passable_mask(), a, b)
    assert d is not None
    return d


def _tour_length(grid: TrueGrid, order: Sequence[Pos]) -> int:
    pos, total = grid.start, 0
    for s in order:
        total += _true_dist(grid, pos, s)
        pos = s
    return total


def _greedy_order(grid: TrueGrid) -> list[Pos]:
    """Nearest-first order on the true map (BFS distance, ties row-major)."""
    left, pos, order = set(grid.survivors), grid.start, []
    while left:
        nxt = min(left, key=lambda s: (_true_dist(grid, pos, s), s))
        order.append(nxt)
        left.remove(nxt)
        pos = nxt
    return order


def _frontier_ok(belief: object, p: Pos) -> bool:
    """``p`` is a known passable cell with at least one UNKNOWN 4-neighbour."""
    get = belief.get  # type: ignore[attr-defined]
    rows, cols = belief.rows, belief.cols  # type: ignore[attr-defined]
    if get(p) is Cell.UNKNOWN or not bool(belief.passable_mask()[p]):  # type: ignore[attr-defined]
        return False
    return any(get(q) is Cell.UNKNOWN for q in neighbours4(p, rows, cols))


# --- NoisySensor ---------------------------------------------------------------------------


def _mixed_percept() -> Percept:
    return _percept([((0, 0), Cell.OPEN), ((0, 1), Cell.DEBRIS), ((1, 0), Cell.SURVIVOR),
                     ((1, 1), Cell.DEBRIS), ((2, 2), Cell.DEBRIS)])


def _sensor(eps: float, seed: int) -> NoisySensor:
    return NoisySensor(eps, seed, SENSOR_ROWS, SENSOR_COLS)


def test_noisy_sensor_matches_reference_stream() -> None:
    """One uniform per grid cell per call, indexed by cell (OPEN, SURVIVOR, exempt included)."""
    eps, seed = 0.5, 11
    percepts = [_mixed_percept()] * 40
    exempt: list[tuple[Pos, ...]] = [((1, 1),) if i % 3 == 0 else () for i in range(40)]
    sensor = _sensor(eps, seed)
    got = [sensor.observe(p, ex) for p, ex in zip(percepts, exempt, strict=True)]
    assert got == _reference_flips(seed, percepts, eps, exempt)
    assert any(g != p for g, p in zip(got, percepts, strict=True))


def test_noisy_sensor_deterministic_per_seed() -> None:
    percepts = [_mixed_percept()] * 30
    a, b, c = _sensor(0.4, 3), _sensor(0.4, 3), _sensor(0.4, 4)
    ra = [a.observe(p) for p in percepts]
    assert ra == [b.observe(p) for p in percepts]
    assert ra != [c.observe(p) for p in percepts]


def test_noisy_sensor_draws_even_when_nothing_can_flip() -> None:
    """A percept without debris still advances the stream by one block of draws (CRN)."""
    eps, seed = 0.5, 5
    open_only = _percept([((0, c), Cell.OPEN) for c in range(4)])
    debris = _percept([((1, c), Cell.DEBRIS) for c in range(SENSOR_COLS)])
    s = _sensor(eps, seed)
    assert s.observe(open_only) == open_only
    got = s.observe(debris)
    ref = _reference_flips(seed, [open_only, debris], eps, [(), ()])
    assert got == ref[1]


def test_noisy_sensor_flip_rate_is_false_negative() -> None:
    eps = 0.1
    cells = [((r, c), Cell.DEBRIS) for r in range(100) for c in range(100)]
    s = NoisySensor(eps, 0, 100, 100)
    flips = 0
    for _ in range(FLIP_DRAWS // len(cells)):
        out = s.observe(_percept(cells))
        flips += sum(cell is Cell.OPEN for _, cell in out.cells)
    n = (FLIP_DRAWS // len(cells)) * len(cells)
    sd = math.sqrt(n * eps * (1 - eps))
    assert abs(flips - n * eps) <= FLIP_SIGMAS * sd


def test_noisy_sensor_never_flips_open_survivor_or_exempt() -> None:
    s = _sensor(1.0 - 1e-12, 2)
    for _ in range(20):
        out = s.observe(_mixed_percept(), exempt=((0, 1), (2, 2)))
        d = out.as_dict()
        assert d[(0, 0)] is Cell.OPEN
        assert d[(1, 0)] is Cell.SURVIVOR
        assert d[(0, 1)] is Cell.DEBRIS and d[(2, 2)] is Cell.DEBRIS  # exempt
        assert [p for p, _ in out.cells] == [p for p, _ in _mixed_percept().cells]
        # No false positives: nothing is ever reported DEBRIS that is not DEBRIS.
        for (_p, cell), (_, true) in zip(out.cells, _mixed_percept().cells, strict=True):
            assert cell is true or (true is Cell.DEBRIS and cell is Cell.OPEN)


def test_noisy_sensor_eps_zero_is_identity() -> None:
    s = _sensor(0.0, 9)
    for _ in range(50):
        assert s.observe(_mixed_percept()) == _mixed_percept()


def test_noisy_sensor_eps_zero_still_consumes_the_stream() -> None:
    """eps = 0 draws exactly as many uniforms as eps > 0 (noise levels share one stream)."""
    clean, noisy = _sensor(0.0, 13), _sensor(0.3, 13)
    for _ in range(7):
        clean.observe(_mixed_percept())
        noisy.observe(_mixed_percept())
    assert clean.rng.random() == noisy.rng.random()


def test_noisy_sensor_draws_are_paired_per_call_and_cell() -> None:
    """Regression: a cell's noise at call n does not depend on the rest of the percept.

    Two sensors on the same seed whose percepts differ (other cells, other order, as for robots
    standing in different places) must give the same verdict for every debris cell both see at
    the same call; percept-order draws gave ~5 % disagreement between agents."""
    eps, seed, rows, cols = 0.5, 21, 6, 6
    a, b = NoisySensor(eps, seed, rows, cols), NoisySensor(eps, seed, rows, cols)
    all_debris = [((r, c), Cell.DEBRIS) for r in range(rows) for c in range(cols)]
    shared = 0
    flipped = False
    for call in range(60):
        subset = [pc for i, pc in enumerate(all_debris) if (i + call) % 3 != 0][::-1]
        exempt = ((call % rows, 0),)
        got_a = a.observe(_percept(all_debris), exempt).as_dict()
        got_b = b.observe(_percept(subset), exempt).as_dict()
        for p, cell in got_b.items():
            assert cell is got_a[p], (call, p)
            shared += 1
        assert got_a[exempt[0]] is Cell.DEBRIS
        flipped = flipped or any(cell is Cell.OPEN for cell in got_a.values())
    assert shared > 0 and flipped


def test_noisy_sensor_rejects_cells_outside_its_grid() -> None:
    with pytest.raises(ValueError, match="outside"):
        _sensor(0.1, 0).observe(_percept([((SENSOR_ROWS, 0), Cell.DEBRIS)]))
    with pytest.raises(ValueError, match="non-empty"):
        NoisySensor(0.1, 0, 0, 3)


def test_noise_is_paired_across_agents_in_missions() -> None:
    """Regression (missions): every (t, cell) both agents observe gets the same noisy verdict."""
    scenario, size, eps = "random_rubble", MISSION_SIZE, 0.2
    shared = 0
    for seed in range(6):
        grid = generate(scenario, size, size, seed)
        logs: list[dict[tuple[int, Pos], Cell]] = []
        for planner in ("andor", "astar"):
            mission = ExtMission(grid, _search_agent(grid, planner), seed=seed,
                                 false_negative=eps)
            log: dict[tuple[int, Pos], Cell] = {}

            def logged(percept: Percept, exempt: Sequence[Pos] = (), *,
                       _observe: Callable[..., Percept] = mission.sensor.observe,
                       _log: dict[tuple[int, Pos], Cell] = log,
                       _m: ExtMission = mission) -> Percept:
                out = _observe(percept, exempt)
                t = _m.world.t if _m.started else 0
                for (p, true), (_, seen) in zip(percept.cells, out.cells, strict=True):
                    if true is Cell.DEBRIS and p not in exempt:
                        _log[(t, p)] = seen
                return out

            mission.sensor.observe = logged  # type: ignore[method-assign]
            mission.run()
            logs.append(log)
        common = set(logs[0]) & set(logs[1])
        shared += len(common)
        assert all(logs[0][key] is logs[1][key] for key in common)
    assert shared > 0


# --- ProbBelief ----------------------------------------------------------------------------


def _pb(eps: float, survivor: Pos | None = None, rows: int = 4, cols: int = 4,
        prior: float = EXT_PRIOR_DEBRIS) -> ProbBelief:
    return ProbBelief(rows, cols, survivor, prior, eps, EXT_DEBRIS_THRESHOLD)


def test_prior_is_optimistic() -> None:
    assert 0.0 < EXT_PRIOR_DEBRIS < EXT_DEBRIS_THRESHOLD <= 1.0


def test_prob_belief_initial_state() -> None:
    b = _pb(0.1)
    assert b.p_debris.dtype == np.float64
    assert np.allclose(b.p_debris, EXT_PRIOR_DEBRIS)
    assert not b.observed.any()
    assert b.passable_mask().all()
    assert b.get((1, 1)) is Cell.UNKNOWN
    assert b.survivors_seen() == ()
    assert b.explored_count() == 0


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3])
@pytest.mark.parametrize("prior", [0.1, EXT_PRIOR_DEBRIS, 0.4])
def test_prob_belief_open_posterior_exact(eps: float, prior: float) -> None:
    b = _pb(eps, prior=prior)
    expected = prior
    for _ in range(4):
        upd = b.update(_percept([((2, 3), Cell.OPEN)]))
        expected = _bayes_open(expected, eps)
        assert b.p_debris[2, 3] == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert upd.contradictions == ()
    assert b.p_debris[0, 0] == pytest.approx(prior)  # untouched cells keep the prior
    assert b.get((2, 3)) is Cell.OPEN
    assert b.observed[2, 3] and not b.observed[0, 0]


def test_prob_belief_open_with_eps_zero_is_certain() -> None:
    b = _pb(0.0)
    b.update(_percept([((1, 2), Cell.OPEN)]))
    assert b.p_debris[1, 2] == 0.0


def test_prob_belief_debris_touched_and_survivor() -> None:
    b = _pb(0.2)
    upd = b.update(_percept([((0, 0), Cell.DEBRIS), ((0, 1), Cell.SURVIVOR)]),
                   touched=((3, 3),))
    assert b.p_debris[0, 0] == 1.0
    assert b.p_debris[3, 3] == 1.0
    assert b.p_debris[0, 1] == 0.0
    assert set(upd.contradictions) == {(0, 0), (3, 3)}
    assert b.get((0, 0)) is Cell.DEBRIS and b.get((3, 3)) is Cell.DEBRIS
    assert b.get((0, 1)) is Cell.SURVIVOR
    assert b.survivors_seen() == ((0, 1),)
    mask = b.passable_mask()
    assert not mask[0, 0] and not mask[3, 3] and mask[0, 1]


def test_prob_belief_no_false_positives_debris_stays_certain() -> None:
    """Noise only hides debris, so one DEBRIS reading is final: later OPEN readings keep 1."""
    b = _pb(0.2)
    b.update(_percept([((1, 1), Cell.DEBRIS)]))
    for _ in range(10):
        upd = b.update(_percept([((1, 1), Cell.OPEN)]))
        assert upd.contradictions == ()
    assert b.p_debris[1, 1] == 1.0
    assert b.get((1, 1)) is Cell.DEBRIS
    assert not b.passable_mask()[1, 1]


def test_prob_belief_contradiction_semantics() -> None:
    b = _pb(0.2)
    # Unknown (prior < threshold, passable) -> DEBRIS: contradiction.
    assert b.update(_percept([((0, 0), Cell.DEBRIS)])).contradictions == ((0, 0),)
    # Already DEBRIS: no new contradiction.
    assert b.update(_percept([((0, 0), Cell.DEBRIS)])).contradictions == ()
    # Seen OPEN (low P, passable) then DEBRIS: contradiction.
    b.update(_percept([((2, 2), Cell.OPEN)]))
    assert b.update(_percept([((2, 2), Cell.DEBRIS)])).contradictions == ((2, 2),)
    # OPEN and SURVIVOR readings never contradict.
    assert b.update(_percept([((3, 0), Cell.OPEN), ((3, 1), Cell.SURVIVOR)])).contradictions \
        == ()
    # A touched cell that was passable is a contradiction; touching known debris is not.
    assert b.update(_percept([]), touched=((1, 3),)).contradictions == ((1, 3),)
    assert b.update(_percept([]), touched=((1, 3),)).contradictions == ()


def test_prob_belief_threshold_drives_categorical_view() -> None:
    """A prior at or above the threshold makes even unobserved cells impassable."""
    b = ProbBelief(3, 3, None, EXT_DEBRIS_THRESHOLD, 0.1, EXT_DEBRIS_THRESHOLD)
    assert not b.passable_mask().any()
    assert b.get((0, 0)) is Cell.UNKNOWN  # never observed
    b.update(_percept([((0, 0), Cell.OPEN)]))
    assert b.passable_mask()[0, 0]
    assert b.get((0, 0)) is Cell.OPEN


def test_prob_belief_known_survivor() -> None:
    b = _pb(0.1, survivor=(3, 2))
    assert b.p_debris[3, 2] == 0.0
    assert b.passable_mask()[3, 2]
    assert b.survivors_seen() == ((3, 2),)


def _walk(grid: TrueGrid, seed: int, steps: int
          ) -> list[tuple[Percept, tuple[Pos, ...]]]:
    """Percepts (and touched cells) of a seeded random walk on ``grid``."""
    world = World(grid, UniformSlip(0.2), sensor_radius=1, seed=seed)
    rng = np.random.default_rng([seed, 99])
    out = [(world.sense(), ())]
    for _ in range(steps):
        res = world.step(ACTIONS[int(rng.integers(len(ACTIONS)))])
        out.append((res.percept, (res.target,) if res.hit_debris else ()))
    return out


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("known", [True, False])
def test_prob_belief_eps_zero_matches_belief_map(seed: int, known: bool) -> None:
    grid = generate("random_rubble", 10, 10, seed)
    survivor = grid.survivor if known else None
    bm = BeliefMap(grid.rows, grid.cols, survivor)
    pb = ProbBelief(grid.rows, grid.cols, survivor, EXT_PRIOR_DEBRIS, 0.0,
                    EXT_DEBRIS_THRESHOLD)
    assert np.array_equal(bm.passable_mask(), pb.passable_mask())
    for percept, touched in _walk(grid, seed, WALK_STEPS):
        u1 = bm.update(percept.cells)
        u2 = pb.update(percept, touched=touched)
        assert set(u2.contradictions) == set(u1.contradictions)
        assert np.array_equal(bm.passable_mask(), pb.passable_mask())
        if not known:
            assert all(bm.get(p) is pb.get(p)
                       for p in np.ndindex(grid.rows, grid.cols))
            assert bm.survivors_seen() == pb.survivors_seen()
            assert bm.explored_count() == pb.explored_count()


# --- multi_survivor_grid -------------------------------------------------------------------


@pytest.mark.parametrize("scenario", GENERATED_SCENARIOS)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_multi_survivor_grid_contract(scenario: str, k: int) -> None:
    for seed in range(3):
        g = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, seed, k)
        base = generate(scenario, MISSION_SIZE, MISSION_SIZE, seed)
        assert len(g.survivors) == k
        assert g.survivor == base.survivor
        assert g.start == base.start
        assert g.start not in g.survivors
        assert len(set(g.survivors)) == k
        for a, b in itertools.combinations(g.survivors, 2):
            assert manhattan(a, b) >= EXT_MIN_SURVIVOR_SPACING
        passable = g.passable_mask()
        dist = bfs_distances(passable, g.start)
        for s in g.survivors:
            assert dist[s] != UNREACHABLE
        # The base map plus the extra survivors, nothing else changed.
        expected = base.cells.copy()
        for s in g.extra_survivors:
            assert base.cells[s] == Cell.OPEN
            expected[s] = Cell.SURVIVOR
        assert np.array_equal(g.cells, expected)
        assert np.array_equal(g.high_slip, base.high_slip)
        assert g == multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, seed, k)
        # Survivors for k are a prefix of those for k + 1 (paired conditions).
        more = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, seed, k + 1)
        assert more.survivors[:k] == g.survivors


def test_multi_survivor_grid_k1_is_base_map() -> None:
    for scenario in GENERATED_SCENARIOS:
        g = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, 5, 1)
        base = generate(scenario, MISSION_SIZE, MISSION_SIZE, 5)
        assert g.extra_survivors == ()
        assert g.survivor == base.survivor
        assert np.array_equal(g.cells, base.cells)


def test_multi_survivor_grid_seed_changes_placement() -> None:
    placements = {multi_survivor_grid("random_rubble", MISSION_SIZE, MISSION_SIZE, s,
                                      MULTI_K).extra_survivors for s in range(6)}
    assert len(placements) > 1


# --- SearchAgent + ExtMission: unknown survivor location ----------------------------------


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_search_andor_unknown_location_every_scenario(scenario: str) -> None:
    for seed in SEARCH_SEEDS:
        grid = generate(scenario, MISSION_SIZE, MISSION_SIZE, seed)
        m = _run(grid, "andor", seed, survivors_known=False)
        met = m.metrics()
        assert met.success, (scenario, seed, met.failure_reason)
        assert met.bumps == 0
        assert met.unplanned_states == 0
        assert met.rescued == 1 and met.k == 1
        assert met.survivors_known is False
        assert met.agent == "search_andor"
        assert 0 <= met.first_detection_step <= met.steps
        assert met.steps >= met.oracle_path == _true_dist(grid, grid.start, grid.survivor)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_search_astar_unknown_location_every_scenario(scenario: str) -> None:
    for seed in SEARCH_SEEDS:
        grid = generate(scenario, MISSION_SIZE, MISSION_SIZE, seed)
        m = _run(grid, "astar", seed, survivors_known=False)
        met = m.metrics()
        assert met.success, (scenario, seed, met.failure_reason)
        assert met.bumps == 0
        assert met.agent == "search_astar"
        assert 0 <= met.first_detection_step <= met.steps


@pytest.mark.parametrize("planner", ["andor", "astar"])
def test_frontier_target_is_valid_when_chosen(planner: str) -> None:
    checked = 0
    for scenario in ("random_rubble", "maze_corridors", "collapsed_building"):
        for seed in range(2):
            grid = generate(scenario, MISSION_SIZE, MISSION_SIZE, seed)
            prev: list[Pos | None] = [None]

            def check(mission: ExtMission, e: ExtStepEvent, prev: list[Pos | None] = prev
                      ) -> None:
                nonlocal checked
                if e.mode == "explore" and e.target is not None and e.target != prev[0]:
                    assert _frontier_ok(mission.agent.belief, e.target), (e.t, e.target)
                    checked += 1
                prev[0] = e.target

            m = _run(grid, planner, seed, survivors_known=False, on_event=check)
            assert m.status is MissionStatus.RESCUED
    assert checked > 0


@pytest.mark.parametrize("planner", ["andor", "astar"])
def test_target_changes_only_on_contract_triggers(planner: str) -> None:
    """The target changes only after a rescue, a new sighting, or when the old one stopped
    being valid (reached, no longer a frontier / unrescued survivor, or unreachable)."""
    changes = 0
    for scenario in GENERATED_SCENARIOS:
        grid = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, 1, 2)
        state: dict[str, object] = {"target": None, "mode": None}
        rescued: set[Pos] = set()

        def check(mission: ExtMission, e: ExtStepEvent, state: dict[str, object] = state,
                  rescued: set[Pos] = rescued) -> None:
            nonlocal changes
            if e.rescued_now:
                rescued.add(e.rescued_now)
            prev = state["target"]
            if e.t > 0 and prev is not None and e.target != prev:
                changes += 1
                belief = mission.agent.belief
                dist = bfs_distances(belief.passable_mask(), e.new_pos)
                invalid = (
                    prev == e.new_pos
                    or dist[prev] == UNREACHABLE  # type: ignore[index]
                    or (state["mode"] == "explore" and not _frontier_ok(belief, prev))
                    or (state["mode"] == "rescue" and prev in rescued)
                )
                assert e.rescued_now or e.found_now or invalid, (e.t, prev, e.target)
            state["target"], state["mode"] = e.target, e.mode

        m = _run(grid, planner, 1, survivors_known=False, on_event=check)
        assert m.status is MissionStatus.RESCUED
    assert changes > 0


def test_modes_switch_from_explore_to_rescue() -> None:
    grid = generate("random_rubble", MISSION_SIZE, MISSION_SIZE, 1)
    m = _run(grid, "andor", 1, survivors_known=False)
    modes = [e.mode for e in m.events]
    assert modes[0] == "explore"
    assert modes[-1] == "rescue" or m.events[-1].rescued_now
    first_found = next(e.t for e in m.events if e.found_now)
    assert first_found == m.metrics().first_detection_step
    assert all(e.mode == "explore" for e in m.events if e.t < first_found)
    known = _run(grid, "andor", 1, survivors_known=True)
    assert known.events[0].mode == "rescue"
    assert known.metrics().first_detection_step == 0


# --- several survivors --------------------------------------------------------------------


@pytest.mark.parametrize("planner", ["andor", "astar"])
@pytest.mark.parametrize("scenario", GENERATED_SCENARIOS)
def test_multi_missions_rescue_all(planner: str, scenario: str) -> None:
    for seed in range(2):
        grid = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, seed, MULTI_K)
        m = _run(grid, planner, seed, survivors_known=True)
        met = m.metrics()
        assert met.success, (scenario, seed, met.failure_reason)
        assert met.rescued == MULTI_K == met.k
        rescued = [e.new_pos for e in m.events if e.rescued_now]
        assert sorted(rescued) == sorted(grid.survivors)
        assert met.bumps == 0
        if planner == "andor":
            assert met.unplanned_states == 0
        assert met.steps >= met.optimal_tour  # the robot physically visits every survivor
        assert m.status is MissionStatus.RESCUED


def test_multi_unknown_location_rescues_all() -> None:
    for seed in range(3):
        grid = multi_survivor_grid("random_rubble", MISSION_SIZE, MISSION_SIZE, seed, MULTI_K)
        met = _run(grid, "andor", seed, survivors_known=False).metrics()
        assert met.success and met.rescued == MULTI_K
        assert met.bumps == 0 and met.unplanned_states == 0
        assert 0 <= met.first_detection_step <= met.steps


@pytest.mark.parametrize("planner", ["andor", "astar"])
def test_greedy_target_order_on_belief(planner: str) -> None:
    """Every change of rescue target picks the nearest unrescued survivor on the belief."""
    checked = 0
    for scenario in GENERATED_SCENARIOS:
        grid = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, 3, MULTI_K)
        rescued: set[Pos] = set()
        prev: list[Pos | None] = [None]

        def check(mission: ExtMission, e: ExtStepEvent, rescued: set[Pos] = rescued,
                  prev: list[Pos | None] = prev, grid: TrueGrid = grid,
                  scenario: str = scenario) -> None:
            nonlocal checked
            if e.rescued_now:
                rescued.add(e.new_pos)
            if e.target is not None and e.target != prev[0] and e.mode == "rescue":
                left = [s for s in grid.survivors if s not in rescued]
                dist = bfs_distances(mission.agent.belief.passable_mask(), e.new_pos)
                reach = [s for s in left if dist[s] != UNREACHABLE]
                best = min(reach, key=lambda s: (int(dist[s]), s))
                assert e.target == best, (scenario, e.t, e.target, best)
                checked += 1
            prev[0] = e.target

        m = _run(grid, planner, 3, survivors_known=True, on_event=check)
        assert m.status is MissionStatus.RESCUED
    # Initial target plus at least one later change per mission (a survivor rescued on the
    # way to another one needs no target change of its own).
    assert checked >= 2 * len(GENERATED_SCENARIOS)


@pytest.mark.parametrize("scenario", GENERATED_SCENARIOS)
def test_tour_metrics(scenario: str) -> None:
    for seed in range(3):
        for k in (1, 2, MULTI_K):
            grid = multi_survivor_grid(scenario, MISSION_SIZE, MISSION_SIZE, seed, k)
            met = _run(grid, "andor", seed, survivors_known=True).metrics()
            farthest = max(_true_dist(grid, grid.start, s) for s in grid.survivors)
            assert met.optimal_tour <= met.greedy_tour
            assert met.optimal_tour >= farthest and met.greedy_tour >= farthest
            # Independent brute force over every visiting order.
            best = min(_tour_length(grid, order) for order in
                       itertools.permutations(grid.survivors))
            assert met.optimal_tour == best
            assert met.greedy_tour == _tour_length(grid, _greedy_order(grid))
            if k == 1:
                assert met.oracle_path == met.greedy_tour == met.optimal_tour
            else:
                assert met.optimal_tour >= sum(
                    sorted(_true_dist(grid, a, b) for a, b in
                           itertools.combinations((grid.start, *grid.survivors), 2))[:k])


def test_optimal_tour_nan_above_brute_force_limit() -> None:
    k = EXT_MAX_BRUTE_FORCE_K + 1
    grid = multi_survivor_grid("random_rubble", 16, 16, 0, k)
    met = _run(grid, "astar", 0, survivors_known=True).metrics()
    assert met.k == k
    assert math.isnan(met.optimal_tour)
    assert not math.isnan(met.greedy_tour)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_budget_scales_with_k(k: int) -> None:
    cfg = WorldConfig()
    grid = multi_survivor_grid("random_rubble", MISSION_SIZE, MISSION_SIZE, 0, k)
    m = ExtMission(grid, _search_agent(grid, "andor"), cfg, seed=0)
    assert m.budget == cfg.step_budget(MISSION_SIZE, MISSION_SIZE) * k


def test_budget_exhaustion_ends_mission() -> None:
    grid = multi_survivor_grid("maze_corridors", MISSION_SIZE, MISSION_SIZE, 0, 2)
    m = ExtMission(grid, _search_agent(grid, "andor"), seed=0, survivors_known=False)
    m.budget = 5  # far too few steps to find and rescue two survivors in a maze
    m.run()
    met = m.metrics()
    assert m.status is MissionStatus.BUDGET
    assert not met.success and met.failure_reason == "budget"
    assert met.steps == 5 and m.events[-1].status is MissionStatus.BUDGET
    assert met.rescued < 2
    with pytest.raises(RuntimeError):
        m.step()


# --- noisy sensing -------------------------------------------------------------------------


def test_noise_free_missions_never_bump_noisy_ones_do() -> None:
    noisy_bumps = 0
    for seed in NOISY_SEEDS:
        grid = generate("random_rubble", MISSION_SIZE, MISSION_SIZE, seed)
        clean = _run(grid, "andor", seed, false_negative=0.0).metrics()
        assert clean.bumps == 0 and clean.unplanned_states == 0
        assert clean.false_negative == 0.0
        noisy = _run(grid, "andor", seed, false_negative=NOISY_EPS).metrics()
        assert noisy.false_negative == NOISY_EPS
        noisy_bumps += noisy.bumps
    assert noisy_bumps > 0


@pytest.mark.parametrize("planner", ["andor", "astar"])
@pytest.mark.parametrize("known", [True, False])
def test_prob_belief_eps_zero_mission_equals_belief_map_mission(planner: str, known: bool
                                                                ) -> None:
    """eps = 0: a ProbBelief makes the same decisions as a BeliefMap, so the events match."""
    for seed in range(3):
        grid = multi_survivor_grid("collapsed_building", MISSION_SIZE, MISSION_SIZE, seed, 2)
        a = _run(grid, planner, seed, survivors_known=known, prob_belief=False)
        b = _run(grid, planner, seed, survivors_known=known, prob_belief=True)
        assert isinstance(a.agent.belief, BeliefMap)
        assert isinstance(b.agent.belief, ProbBelief)
        assert a.events == b.events
        assert a.status is MissionStatus.RESCUED
        assert a.metrics().bumps == 0


# --- determinism, events, metrics ---------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"survivors_known": False},
    {"survivors_known": True},
    {"false_negative": NOISY_EPS},
])
def test_events_deterministic_per_seed(kwargs: dict[str, object]) -> None:
    grid = multi_survivor_grid("random_rubble", MISSION_SIZE, MISSION_SIZE, 2, 2)
    a = _run(grid, "andor", 4, **kwargs)
    b = _run(grid, "andor", 4, **kwargs)
    assert a.events == b.events
    ma, mb = a.metrics(), b.metrics()
    for name in ExtMetrics.__dataclass_fields__:
        if name == "planning_ms":
            continue
        va, vb = getattr(ma, name), getattr(mb, name)
        assert va == vb or (isinstance(va, float) and math.isnan(va) and math.isnan(vb)), name


def test_event_fields_and_metrics_consistency() -> None:
    grid = multi_survivor_grid("random_rubble", MISSION_SIZE, MISSION_SIZE, 1, 2)
    m = _run(grid, "andor", 1, survivors_known=False)
    events = m.events
    assert all(isinstance(e, ExtStepEvent) for e in events)
    assert events[0].t == 0 and events[0].action is None
    assert [e.t for e in events[1:]] == list(range(1, len(events)))
    assert all(e.status is MissionStatus.RUNNING for e in events[:-1])
    assert events[-1].status is MissionStatus.RESCUED
    met = m.metrics()
    assert isinstance(met, ExtMetrics)
    assert met.steps == events[-1].t
    assert met.rescued == sum(1 for e in events if e.rescued_now)
    assert met.bumps == sum(e.bump for e in events)
    assert met.unplanned_states == sum(e.unplanned for e in events)
    assert met.replans == sum(e.replanned for e in events)
    assert met.rows == met.cols == MISSION_SIZE
    assert met.k == 2 and not met.survivors_known
    with pytest.raises(dataclass_error()):
        events[0].t = 5  # type: ignore[misc]


def dataclass_error() -> type[Exception]:
    """Exception raised when assigning to a frozen dataclass field."""
    from dataclasses import FrozenInstanceError

    return FrozenInstanceError


# --- argument validation (fix pass) ------------------------------------------------------


def test_mission_rejects_a_pessimistic_prior() -> None:
    """Regression: prior >= threshold made every unknown cell impassable, so unknown-location
    missions aborted UNREACHABLE on solvable maps (r = 0: 40/40)."""
    grid = generate("random_rubble", MISSION_SIZE, MISSION_SIZE, 0)
    agent = _search_agent(grid, "andor")
    for prior in (EXT_DEBRIS_THRESHOLD, EXT_DEBRIS_THRESHOLD + 0.1):
        with pytest.raises(ValueError, match="below threshold"):
            ExtMission(grid, agent, survivors_known=False, false_negative=0.1,
                       prior_debris=prior)
    with pytest.raises(ValueError, match="below threshold"):
        ExtMission(grid, agent, prior_debris=0.3, threshold=0.3)
    ExtMission(grid, agent, prior_debris=0.29, threshold=0.3)  # just below: accepted


def test_mission_rejects_a_belief_map_under_noise() -> None:
    """Regression: a BeliefMap under false negatives can un-see known debris (outside the
    SearchAgent termination argument); only the Bayes belief is allowed with eps > 0."""
    grid = generate("random_rubble", MISSION_SIZE, MISSION_SIZE, 0)
    agent = _search_agent(grid, "andor")
    with pytest.raises(ValueError, match="prob_belief=False"):
        ExtMission(grid, agent, false_negative=0.1, prob_belief=False)
    assert ExtMission(grid, agent, false_negative=0.0, prob_belief=False).prob_belief is False
    assert ExtMission(grid, agent, false_negative=0.1).prob_belief is True


# --- demo CLI ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["search", "multi", "noisy"])
def test_demo_cli_modes(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    from rescue.ext.demo import main

    assert main(["--mode", mode]) == 0
    out = capsys.readouterr().out
    assert f"extension demo: {mode}" in out
    assert "Execution trace:" in out
    assert "bumps:" in out and "rescued:" in out
    assert main(["--mode", mode]) == 0

    def untimed(text: str) -> list[str]:
        return [line for line in text.splitlines() if "planning_ms" not in line]

    assert untimed(capsys.readouterr().out) == untimed(out)  # deterministic but for timing
