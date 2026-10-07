"""Two-sided sensor noise for the extensions (docs/plan.md §3.15.2): false positives in
``NoisySensor`` / ``ProbBelief``, the ``SearchAgent`` fallback, the ``noisy_fp`` block, X6.

The round-1 identity is checked three ways: the update formulas against the round-1 ones bit for
bit, the sensor and belief against round-1 reference implementations, and whole missions
against event digests recorded with the round-1 code (commit ``a948f27``).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rescue.config import (
    EXT_CLAIM_FP,
    EXT_CLAIM_SUCCESS_TARGET,
    EXT_DEBRIS_THRESHOLD,
    EXT_FALLBACK_MAX_P,
    EXT_MAX_FALSE_POSITIVE,
    EXT_PRIOR_DEBRIS,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.experiments.hypotheses import VERDICT_FAIL, VERDICT_PASS, VERDICT_WARN
from rescue.experiments.spec import task_seed
from rescue.ext import agents as ext_agents
from rescue.ext import claims
from rescue.ext import plots as ext_plots
from rescue.ext.agents import SearchAgent, make_ext_agent, relaxed_passable_mask
from rescue.ext.experiments import (
    NOISY_BLOCK,
    NOISY_FP_BLOCK,
    RAW_COLUMNS,
    ROUND2_COLUMNS,
    SUMMARY_COLUMNS,
    config_from_dict,
    expand_tasks,
    load_ext_config,
    read_raw,
    run_ext_experiments,
)
from rescue.ext.mission import ExtMission, ExtStepEvent, run_ext_mission
from rescue.ext.scenarios import multi_survivor_grid
from rescue.ext.sensing import NoisySensor, ProbBelief, debris_posterior
from rescue.grid import Cell, Pos, TrueGrid
from rescue.planners.andor import PlannerStats, PlanResult
from rescue.planners.astar import AStarResult
from rescue.runner import MissionStatus, make_transition
from rescue.world import Percept

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = ("random_rubble", "collapsed_building", "flooded_zone", "maze_corridors")

# --- constants of the tests ----------------------------------------------------------------
#: Random priors / readings compared bit for bit with the round-1 formulas.
FORMULA_SAMPLES = 2_000
#: Grid of the flip-rate checks and the number of sensor calls on it.
FLIP_SIZE = 60
FLIP_CALLS = 8
#: Allowed deviation of an empirical flip rate, in binomial standard deviations.
FLIP_SIGMAS = 4.0
#: Noise rates of the flip-rate checks.
FLIP_FN = 0.2
FLIP_FP = 0.1
#: Steps of the random belief walk compared with the round-1 filter.
WALK_STEPS = 200
#: A false-positive rate for the two-sided missions.
FP = 0.1
#: Seeds of the fp = 0 fallback check and of the robot-cell check.
FP0_SEEDS = range(4)

#: Round-1 digests (commit a948f27, before the two-sided noise) of 30 missions per scenario:
#: 10×10 maps, p = 0.2, task seeds 0-2 of base seed 2026, both agents, conditions (k, survivor
#: location known, false-negative rate) in ``GOLDEN_CONDITIONS``. The digest covers every
#: event, the non-timing metrics and, for a ``ProbBelief``, the final probabilities' bytes.
GOLDEN_CONDITIONS = ((1, True, 0.0), (1, False, 0.0), (1, True, 0.1), (1, False, 0.2),
                     (2, True, 0.1))
GOLDEN_SIZE = 10
GOLDEN_SLIP = 0.2
GOLDEN_BASE_SEED = 2026
GOLDEN_MAPS = 3
GOLDEN_METRICS = ("success", "failure_reason", "steps", "rescued", "first_detection_step",
                  "cells_explored", "replans", "slips", "skids", "bumps", "unplanned_states",
                  "nodes_expanded", "blocked_moves")
GOLDEN_DIGESTS = {
    "random_rubble": "a7f8b819005bd073eb4525127c311c242e8ed056216e3fdf2eb03a4a98b72ee7",
    "collapsed_building": "48329f11eb8e0aa0fee61138468799e2bcece6b4890ee430ce65187a00308774",
    "flooded_zone": "f71cd7096f7f6cc6cd27ef2cf8ab13ee933bb03395860809b290e1ca0d7d9824",
    "maze_corridors": "2b55bd49f18bdd11bf7a8065068670f38ce9f965bc0d21326f51e8f9b9dd2752",
}


# --- round-1 reference implementations ------------------------------------------------------
def _round1_posterior(prior: float, reads_debris: bool, eps: float) -> float:
    """The round-1 ``ProbBelief`` update of one reading (copied from commit a948f27)."""
    if reads_debris:
        return 1.0
    denominator = eps * prior + (1.0 - prior)
    return eps * prior / denominator if denominator > 0.0 else prior


def _round1_observe(percept: Percept, draws: np.ndarray, cols: int, eps: float,
                    exempt: set[Pos]) -> Percept:
    """The round-1 one-sided sensor given its draws (false negatives only)."""
    if eps <= 0.0:
        return percept
    cells = [(p, Cell.OPEN if cell is Cell.DEBRIS and p not in exempt
              and draws[p[0] * cols + p[1]] < eps else cell) for p, cell in percept.cells]
    return Percept(tuple(cells))


def _random_percept(rng: np.random.Generator, rows: int, cols: int, n: int,
                    survivor_share: float = 0.05) -> Percept:
    cells: dict[Pos, Cell] = {}
    for _ in range(n):
        p = (int(rng.integers(rows)), int(rng.integers(cols)))
        u = rng.random()
        cells[p] = (Cell.SURVIVOR if u < survivor_share
                    else Cell.DEBRIS if u < 0.5 else Cell.OPEN)
    return Percept(tuple(cells.items()))


# --- Bayes math ------------------------------------------------------------------------------
def test_two_sided_posterior_values() -> None:
    fn, fp, prior = 0.1, 0.2, 0.25
    debris = (1 - fn) * prior / ((1 - fn) * prior + fp * (1 - prior))
    free = fn * prior / (fn * prior + (1 - fp) * (1 - prior))
    assert debris_posterior(prior, True, fn, fp) == pytest.approx(debris)
    assert debris_posterior(prior, False, fn, fp) == pytest.approx(free)
    assert debris == pytest.approx(0.6) and free == pytest.approx(0.025 / 0.625)
    # Odds form: each reading multiplies the prior odds by its likelihood ratio.
    odds = prior / (1 - prior)
    after = debris_posterior(prior, True, fn, fp)
    assert after / (1 - after) == pytest.approx(odds * (1 - fn) / fp)
    # Certain states stay certain; impossible readings keep the prior.
    assert debris_posterior(1.0, False, fn, fp) == 1.0
    assert debris_posterior(1.0, True, fn, fp) == 1.0
    assert debris_posterior(0.0, True, fn, fp) == 0.0
    assert debris_posterior(0.0, False, fn, fp) == 0.0
    assert debris_posterior(1.0, False, 0.0, fp) == 1.0  # OPEN on known debris, fn = 0
    # fn = 0: an OPEN reading proves the cell free.
    assert debris_posterior(0.4, False, 0.0, fp) == 0.0


def test_fp0_posterior_is_round1_bit_for_bit() -> None:
    rng = np.random.default_rng(0)
    priors = np.concatenate([rng.random(FORMULA_SAMPLES), [0.0, 1.0, EXT_PRIOR_DEBRIS]])
    for eps in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        for prior in priors:
            for reads_debris in (False, True):
                got = debris_posterior(float(prior), reads_debris, eps, 0.0)
                want = _round1_posterior(float(prior), reads_debris, eps)
                assert got == want, (prior, reads_debris, eps)  # exact, not approx


def test_fp0_debris_formula_is_one_where_defined() -> None:
    """The general DEBRIS formula at fp = 0 is x / (x + 0) = 1 exactly whenever it is defined,
    so returning the round-1 constant 1 is the formula, not a special case."""
    rng = np.random.default_rng(1)
    for prior in rng.random(FORMULA_SAMPLES):
        for fn in (0.0, 0.05, 0.1, 0.2, 0.9):
            numerator = (1.0 - fn) * float(prior)
            if numerator > 0.0:
                assert numerator / (numerator + 0.0 * (1.0 - float(prior))) == 1.0


def test_probbelief_fp0_is_round1_bit_for_bit() -> None:
    """A random walk of percepts, touches and robot cells through ProbBelief(fp = 0) and the
    round-1 filter: identical probabilities, observed cells and reports at every step."""
    rows = cols = 9
    for eps in (0.0, 0.1, 0.3):
        rng = np.random.default_rng(7)
        belief = ProbBelief(rows, cols, (4, 4), EXT_PRIOR_DEBRIS, eps, EXT_DEBRIS_THRESHOLD)
        ref = np.full((rows, cols), EXT_PRIOR_DEBRIS)
        ref[4, 4] = 0.0
        for _ in range(WALK_STEPS):
            percept = _random_percept(rng, rows, cols, 6)
            touched = [(int(rng.integers(rows)), int(rng.integers(cols)))] if rng.random() < 0.2 \
                else []
            here = (int(rng.integers(rows)), int(rng.integers(cols)))
            before = ref.copy()
            update = belief.update(percept, touched, here)  # here ignored at fp = 0
            for p, cell in percept.cells:
                ref[p] = 0.0 if cell is Cell.SURVIVOR else _round1_posterior(
                    float(ref[p]), cell is Cell.DEBRIS, eps)
            for p in touched:
                ref[p] = 1.0
            assert np.array_equal(belief.p_debris, ref)
            crossed = {p for p in {q for q, _ in percept.cells} | set(touched)
                       if before[p] < EXT_DEBRIS_THRESHOLD <= ref[p]}
            assert set(update.contradictions) == crossed


def test_probbelief_two_sided_update() -> None:
    fn, fp = 0.1, 0.2
    b = ProbBelief(3, 3, None, EXT_PRIOR_DEBRIS, fn, EXT_DEBRIS_THRESHOLD, fp)
    up = b.update([((0, 0), Cell.DEBRIS)])
    first = debris_posterior(EXT_PRIOR_DEBRIS, True, fn, fp)
    assert b.p_debris[0, 0] == first and first < 1.0
    assert up.contradictions == (((0, 0),) if first >= EXT_DEBRIS_THRESHOLD else ())
    # A phantom is un-seen by OPEN readings (impossible without false positives).
    b.update([((0, 0), Cell.OPEN)])
    b.update([((0, 0), Cell.OPEN)])
    assert b.get((0, 0)) is Cell.OPEN and b.p_debris[0, 0] < EXT_PRIOR_DEBRIS
    # Contact is certain, and stays certain under noisy readings.
    b.update([], touched=[(1, 1)])
    b.update([((1, 1), Cell.OPEN)])
    assert b.p_debris[1, 1] == 1.0
    # The robot's own cell is free (fp > 0); survivors are free.
    b.update([((2, 2), Cell.DEBRIS)], here=(2, 2))
    assert b.p_debris[2, 2] == 0.0 and b.is_known((2, 2))
    b.update([((2, 1), Cell.SURVIVOR)])
    assert b.p_debris[2, 1] == 0.0 and b.get((2, 1)) is Cell.SURVIVOR
    with pytest.raises(ValueError):
        b.update([], here=(5, 5))
    c = b.copy()
    assert c.false_positive == fp and np.array_equal(c.p_debris, b.p_debris)
    assert "false_positive=0.2" in repr(b)


@pytest.mark.parametrize("bad", [-0.01, EXT_MAX_FALSE_POSITIVE, 0.7, 1.0])
def test_false_positive_range(bad: float) -> None:
    with pytest.raises(ValueError, match="false_positive"):
        NoisySensor(0.1, 0, 4, 4, bad)
    with pytest.raises(ValueError, match="false_positive"):
        ProbBelief(4, 4, None, EXT_PRIOR_DEBRIS, 0.1, EXT_DEBRIS_THRESHOLD, bad)
    grid = multi_survivor_grid("random_rubble", 8, 8, 0, 1)
    with pytest.raises(ValueError, match="false_positive"):
        ExtMission(grid, _agent("search_andor", grid), false_positive=bad)


def test_mission_rejects_beliefmap_with_false_positives() -> None:
    grid = multi_survivor_grid("random_rubble", 8, 8, 0, 1)
    with pytest.raises(ValueError, match="prob_belief"):
        ExtMission(grid, _agent("search_andor", grid), false_positive=FP, prob_belief=False)
    m = ExtMission(grid, _agent("search_andor", grid), false_positive=FP)
    assert m.prob_belief  # fp alone selects the Bayes belief


# --- sensor ----------------------------------------------------------------------------------
def test_sensor_fp0_matches_round1() -> None:
    rows, cols = 7, 9
    for eps in (0.0, 0.1, 0.5):
        new = NoisySensor(eps, 3, rows, cols, 0.0)
        default = NoisySensor(eps, 3, rows, cols)
        twin = NoisySensor(eps, 3, rows, cols)  # supplies the draws of the reference
        rng = np.random.default_rng(5)
        for _ in range(50):
            percept = _random_percept(rng, rows, cols, 12)
            exempt = {(int(rng.integers(rows)), int(rng.integers(cols)))}
            draws = twin.rng.random(rows * cols)
            want = _round1_observe(percept, draws, cols, eps, exempt)
            assert new.observe(percept, exempt).cells == want.cells
            assert default.observe(percept, exempt).cells == want.cells
    noiseless = NoisySensor(0.0, 0, rows, cols, 0.0)
    percept = _random_percept(np.random.default_rng(0), rows, cols, 10)
    assert noiseless.observe(percept) is percept


def test_sensor_draws_do_not_depend_on_noise_rates() -> None:
    """One uniform per grid cell per call whatever the rates (common random numbers), and the
    flipped cells are nested in the rate: a flip at fp = 0.05 is a flip at fp = 0.1."""
    rows = cols = 12
    percept = Percept(tuple(((r, c), Cell.OPEN) for r in range(rows) for c in range(cols)))
    low, high, none = (NoisySensor(0.1, 9, rows, cols, fp) for fp in (0.05, 0.1, 0.0))
    for _ in range(5):
        a, b, _ = (s.observe(percept) for s in (low, high, none))
        flips_a = {p for p, cell in a.cells if cell is Cell.DEBRIS}
        flips_b = {p for p, cell in b.cells if cell is Cell.DEBRIS}
        assert flips_a <= flips_b
    assert low.rng.random() == high.rng.random() == none.rng.random()


def test_flip_rates_both_directions() -> None:
    rows = cols = FLIP_SIZE
    cells = tuple(((r, c), Cell.DEBRIS if (r + c) % 2 else Cell.OPEN)
                  for r in range(rows) for c in range(cols))
    percept = Percept(cells)
    sensor = NoisySensor(FLIP_FN, 21, rows, cols, FLIP_FP)
    fn_hits = fp_hits = n_debris = n_open = 0
    for _ in range(FLIP_CALLS):
        out = dict(sensor.observe(percept).cells)
        for p, truth in cells:
            if truth is Cell.DEBRIS:
                n_debris += 1
                fn_hits += out[p] is Cell.OPEN
            else:
                n_open += 1
                fp_hits += out[p] is Cell.DEBRIS
    for hits, n, rate in ((fn_hits, n_debris, FLIP_FN), (fp_hits, n_open, FLIP_FP)):
        sigma = math.sqrt(rate * (1 - rate) / n)
        assert abs(hits / n - rate) <= FLIP_SIGMAS * sigma, (hits, n, rate)


def test_survivors_and_exempt_cells_never_flip() -> None:
    rows = cols = 10
    cells = tuple(((r, c), Cell.SURVIVOR if r == 0 else Cell.OPEN if r % 2 else Cell.DEBRIS)
                  for r in range(rows) for c in range(cols))
    exempt = {(5, 5), (4, 4)}
    sensor = NoisySensor(0.45, 1, rows, cols, EXT_MAX_FALSE_POSITIVE - 0.01)
    for _ in range(40):
        out = dict(sensor.observe(Percept(cells), exempt).cells)
        for p, truth in cells:
            if truth is Cell.SURVIVOR or p in exempt:
                assert out[p] is truth


def _check_robot_cell(seed: int) -> None:
    grid = multi_survivor_grid("random_rubble", 10, 10, seed, 1)
    agent = _agent("search_andor", grid)
    m = ExtMission(grid, agent, WorldConfig(slip_p=0.2), seed, false_negative=0.2,
                   false_positive=EXT_MAX_FALSE_POSITIVE - 0.01)
    calls: list[tuple[set[Pos], Percept, Percept]] = []
    observe = m.sensor.observe

    def spy(percept: Percept, exempt: Sequence[Pos] = ()) -> Percept:
        out = observe(percept, exempt)
        calls.append(({(int(p[0]), int(p[1])) for p in exempt}, percept, out))
        return out

    m.sensor.observe = spy  # type: ignore[method-assign]
    positions: list[Pos] = []

    def record(event: ExtStepEvent) -> None:
        if event.action is not None or event.t == 0:  # every event that sensed
            positions.append(event.new_pos)

    m.run(record)
    assert len(calls) == len(positions)
    for (exempt, truth, out), pos in zip(calls, positions, strict=True):
        assert pos in exempt
        assert dict(out.cells)[pos] is dict(truth.cells)[pos]
        assert dict(truth.cells)[pos] is not Cell.DEBRIS
    assert agent.belief.p_debris[m.world.pos] == 0.0


@pytest.mark.parametrize("seed", FP0_SEEDS)
def test_robot_cell_never_flips_in_a_mission(seed: int) -> None:
    """Every sensor call of a two-sided mission exempts the robot's cell and reports it as it
    is, and the belief holds the robot's cell as free."""
    _check_robot_cell(seed)


# --- missions at fp = 0 ----------------------------------------------------------------------
def _agent(name: str, grid: TrueGrid, cfg: WorldConfig | None = None, **kwargs: Any) -> Any:
    cfg = cfg if cfg is not None else WorldConfig()
    return make_ext_agent(name, make_transition(grid, cfg), PlannerConfig(), AgentConfig(),
                          **kwargs)


def _digest(scenario: str) -> str:
    h = hashlib.sha256()
    for i in range(GOLDEN_MAPS):
        seed = task_seed(GOLDEN_BASE_SEED, scenario, GOLDEN_SIZE, GOLDEN_SIZE, i)
        for k, known, fn in GOLDEN_CONDITIONS:
            grid = multi_survivor_grid(scenario, GOLDEN_SIZE, GOLDEN_SIZE, seed, k)
            cfg = WorldConfig(slip_p=GOLDEN_SLIP)
            for name in ("search_andor", "search_astar"):
                agent = _agent(name, grid, cfg)
                m = ExtMission(grid, agent, cfg, seed, survivors_known=known, false_negative=fn,
                               false_positive=0.0)
                m.run()
                for e in m.events:
                    h.update(repr(dataclasses.astuple(e)).encode())
                met = m.metrics()
                h.update(repr([getattr(met, f) for f in GOLDEN_METRICS]).encode())
                assert met.fallback_replans == 0 and met.false_positive == 0.0
                if isinstance(agent.belief, ProbBelief):
                    h.update(np.ascontiguousarray(agent.belief.p_debris).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fp0_missions_match_round1_traces(scenario: str) -> None:
    logging.disable(logging.WARNING)
    try:
        assert _digest(scenario) == GOLDEN_DIGESTS[scenario]
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("name", ["search_andor", "search_astar"])
def test_fallback_never_engages_without_false_positives(name: str) -> None:
    """fp = 0: after every update the relaxed and strict masks are equal (the docstring
    argument), so the fallback is never entered."""
    for seed in FP0_SEEDS:
        _check_masks_equal(name, seed)


def _check_masks_equal(name: str, seed: int) -> None:
    grid = multi_survivor_grid("maze_corridors", 10, 10, seed, 2)
    agent = _agent(name, grid, WorldConfig(slip_p=0.3))

    def check(_event: ExtStepEvent) -> None:
        assert np.array_equal(relaxed_passable_mask(agent.belief), agent.belief.passable_mask())
        assert not agent.relaxed

    m = run_ext_mission(grid, agent, WorldConfig(slip_p=0.3), seed, survivors_known=False,
                        false_negative=0.2, on_event=check)
    assert m.metrics().fallback_replans == 0


# --- fallback --------------------------------------------------------------------------------
class PhantomSensor(NoisySensor):
    """No random noise; each cell in ``phantoms`` reads DEBRIS the first time it is sensed
    (unless exempt), and truthfully afterwards."""

    def __init__(self, rows: int, cols: int, phantoms: Sequence[Pos]) -> None:
        super().__init__(0.0, 0, rows, cols)
        self.pending = set(phantoms)

    def observe(self, percept: Percept, exempt: Any = ()) -> Percept:
        out = super().observe(percept, exempt)
        cells = []
        for p, cell in out.cells:
            if p in self.pending and p not in set(exempt):
                self.pending.discard(p)
                cell = Cell.DEBRIS if cell is Cell.OPEN else cell
            cells.append((p, cell))
        return Percept(tuple(cells))


def _phantom_mission(text: str, phantoms: Sequence[Pos], name: str, *, fallback: bool,
                     slip_p: float = 0.0) -> ExtMission:
    grid = TrueGrid.from_ascii(text)
    cfg = WorldConfig(slip_p=slip_p)
    m = ExtMission(grid, _agent(name, grid, cfg, fallback=fallback), cfg, 0, false_positive=FP)
    m.sensor = PhantomSensor(grid.rows, grid.cols, phantoms)
    m.run()
    return m


@pytest.mark.parametrize("name", ["search_andor", "search_astar"])
def test_fallback_crosses_a_phantom_wall(name: str) -> None:
    """A one-row corridor whose middle cell reads DEBRIS once: under the threshold the survivor
    is unreachable and no frontier is left; the fallback replans through the suspected cell."""
    corridor = "S...P"
    with_fb = _phantom_mission(corridor, [(0, 2)], name, fallback=True)
    assert with_fb.status is MissionStatus.RESCUED
    assert with_fb.metrics().fallback_replans >= 1
    assert any(e.contradictions == ((0, 2),) for e in with_fb.events)
    without = _phantom_mission(corridor, [(0, 2)], name, fallback=False)
    assert without.status is MissionStatus.UNREACHABLE
    assert without.metrics().fallback_replans == 0
    # With slips the outcome is the same, only slower.
    slippery = _phantom_mission(corridor, [(0, 2)], name, fallback=True, slip_p=0.3)
    assert slippery.status is MissionStatus.RESCUED


@pytest.mark.parametrize("name", ["search_andor", "search_astar"])
def test_fallback_stops_at_certain_debris(name: str) -> None:
    """The relaxed plan tries a suspected cell that is real debris: the bump makes it certain,
    the plan is blocked, the strict and the relaxed search both fail, and the mission aborts
    at once instead of bumping until the step budget (regression: a suspected cell going
    certain is no threshold contradiction, so it must trigger the replan itself)."""
    m = _phantom_mission("S.#.P", [], name, fallback=True)
    assert m.status is MissionStatus.UNREACHABLE
    met = m.metrics()
    assert met.blocked_moves == 1 and met.bumps == 0  # anticipated: believed DEBRIS
    assert met.steps < m.budget and met.fallback_replans == 1
    assert m.agent.belief.p_debris[0, 2] == 1.0


def test_two_sided_missions_run_and_report() -> None:
    for name in ("search_andor", "search_astar"):
        for seed in FP0_SEEDS:
            grid = multi_survivor_grid("collapsed_building", 10, 10, seed, 1)
            cfg = WorldConfig(slip_p=0.2)
            m = run_ext_mission(grid, _agent(name, grid, cfg), cfg, seed, false_negative=0.1,
                                false_positive=FP)
            met = m.metrics()
            assert met.false_positive == FP and met.fallback_replans >= 0
            assert m.done and met.steps <= m.budget


# --- harness ---------------------------------------------------------------------------------
def test_ext_configs_have_the_noisy_fp_block() -> None:
    for path, size, n_maps in (("ext.yaml", 15, 200), ("ext_quick.yaml", 10, 10)):
        cfg = load_ext_config(REPO / "experiments" / path)
        block = next(b for b in cfg.blocks if b.name == NOISY_FP_BLOCK)
        assert block.scenarios == SCENARIOS and block.sizes == (size,)
        assert block.slip_p == (0.2,) and block.survivors_known == (True,)
        assert block.false_negative == (0.1,)
        assert block.false_positive == (0.0, 0.05, 0.1, 0.2)
        assert cfg.block_agents(block) == ("search_andor", "search_astar")
        assert cfg.block_n_maps(block) == n_maps
        assert all(b.false_positive == (0.0,) for b in cfg.blocks if b.name != NOISY_FP_BLOCK)
        tasks = [t for t in expand_tasks(cfg) if t.experiment == NOISY_FP_BLOCK]
        assert len(tasks) == 4 * 4 * n_maps
        # Same base maps and streams as the noisy block (seed per scenario and replicate).
        noisy = {(t.scenario, t.replicate): t.seed for t in expand_tasks(cfg)
                 if t.experiment == NOISY_BLOCK}
        assert all(noisy[(t.scenario, t.replicate)] == t.seed for t in tasks)


def test_block_validation_of_false_positive() -> None:
    base = {"name": "x", "base_seed": 1, "n_maps": 1, "agents": ["search_andor"]}
    block = {"name": "b", "scenarios": ["random_rubble"], "sizes": [8], "slip_p": [0.2]}
    for bad in ([0.5], [-0.1], [0.1, 0.1], [True]):
        with pytest.raises(ValueError):
            config_from_dict({**base, "blocks": [{**block, "false_positive": bad}]})
    cfg = config_from_dict({**base, "blocks": [{**block, "false_positive": 0.1}]})
    assert cfg.blocks[0].false_positive == (0.1,)


def test_raw_columns_append_round2_and_old_files_read_back(tmp_path: Path) -> None:
    assert RAW_COLUMNS[-len(ROUND2_COLUMNS):] == ROUND2_COLUMNS
    old = tmp_path / "raw.csv"
    with old.open("w", encoding="utf-8", newline="") as fh:
        fields = [c for c in RAW_COLUMNS if c not in ROUND2_COLUMNS]
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({c: _row()[c] for c in fields})
    (row,) = read_raw(old)
    assert row["false_positive"] == 0.0 and row["fallback_replans"] == 0


def _tiny_fp_config() -> Any:
    return config_from_dict({
        "name": "tiny_fp", "base_seed": 5, "n_maps": 2,
        "agents": ["search_andor", "search_astar"],
        "blocks": [{"name": NOISY_FP_BLOCK, "scenarios": ["random_rubble", "maze_corridors"],
                    "sizes": [8], "slip_p": [0.2], "false_negative": [0.1],
                    "false_positive": [0.0, 0.1]}],
    })


def test_tiny_noisy_fp_run_claims_and_plot(tmp_path: Path) -> None:
    raw = run_ext_experiments(_tiny_fp_config(), tmp_path, workers=1, progress=False)
    rows = read_raw(raw)
    assert len(rows) == 2 * 2 * 2 * 2
    assert {r["false_positive"] for r in rows} == {0.0, 0.1}
    assert all(r["fallback_replans"] == 0 for r in rows if r["false_positive"] == 0.0)
    x6 = claims.claim_x6(rows)
    assert x6.label == "X6" and x6.verdict in (VERDICT_PASS, VERDICT_WARN, VERDICT_FAIL)
    assert "Rule:" in x6.detail and "steps cost" in x6.measured
    paths = ext_plots.make_ext_plots(rows, tmp_path / "plots")
    assert [p.name for p in paths] == ["noisy_fp.png"] and paths[0].stat().st_size > 0


# --- X6 on synthetic rows --------------------------------------------------------------------
def _row(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "experiment": NOISY_FP_BLOCK, "scenario": "random_rubble", "rows": 10, "cols": 10,
        "slip_p": 0.2, "k": 1, "survivors_known": True, "false_negative": 0.1, "replicate": 0,
        "seed": 1, "agent": "search_andor", "success": True, "failure_reason": "", "steps": 10,
        "rescued": 1, "first_detection_step": 0.0, "cells_explored": 20, "replans": 1,
        "slips": 0, "skids": 0, "bumps": 0, "blocked_moves": 0, "unplanned_states": 0,
        "planning_ms": 1.0, "nodes_expanded": 5, "oracle_path": 8.0, "greedy_tour": 8.0,
        "optimal_tour": 8.0, "plans_made": 1, "elapsed_ms": 2.0, "false_positive": 0.0,
        "fallback_replans": 0,
    }
    base.update(fields)
    return base


def _fp_rows(n: int, fails: dict[float, int], extra_steps: float = 0.0,
             agents: Sequence[str] = ("search_andor",)) -> list[dict[str, Any]]:
    rows = []
    for agent in agents:
        for fp in (0.0, 0.05, 0.1, 0.2):
            for i in range(n):
                ok = i >= fails.get(fp, 0)
                rows.append(_row(agent=agent, false_positive=fp, replicate=i, seed=100 + i,
                                 success=ok, failure_reason="" if ok else "budget",
                                 steps=10 + i % 3 + (extra_steps if fp > 0 else 0.0),
                                 fallback_replans=1 if fp > 0 and i % 4 == 0 else 0))
    return rows


def test_x6_rules() -> None:
    n = 40
    assert claims.claim_x6(_fp_rows(n, {})).verdict == VERDICT_PASS
    # Failures above EXT_CLAIM_FP do not count; failures at or below it do.
    assert claims.claim_x6(_fp_rows(n, {0.2: 30})).verdict == VERDICT_PASS
    assert claims.claim_x6(_fp_rows(n, {0.1: 12})).verdict == VERDICT_FAIL
    assert claims.claim_x6(_fp_rows(n, {0.05: 12})).verdict == VERDICT_FAIL
    # 2 / 40 failures = exactly 95 %: passes; 3 / 40 is below but the Wilson bound reaches it.
    assert claims.claim_x6(_fp_rows(n, {0.1: 2})).verdict == VERDICT_PASS
    assert 1 - 3 / n < EXT_CLAIM_SUCCESS_TARGET
    assert claims.claim_x6(_fp_rows(n, {0.1: 3})).verdict == VERDICT_WARN
    # fp = 0 is not part of the claim (it is X4's question).
    assert claims.claim_x6(_fp_rows(n, {0.0: 30})).verdict == VERDICT_PASS


def test_x6_not_measured_without_the_claim_level() -> None:
    rows = [r for r in _fp_rows(10, {}) if r["false_positive"] != EXT_CLAIM_FP]
    x6 = claims.claim_x6(rows)
    assert x6.verdict == VERDICT_WARN and "not measured" in x6.measured
    assert claims.claim_x6([]).verdict == VERDICT_WARN


def test_x6_reports_paired_steps_cost_and_fallbacks() -> None:
    rows = _fp_rows(20, {}, extra_steps=5.0, agents=("search_andor", "search_astar"))
    x6 = claims.claim_x6(rows)
    assert "+5.00 steps [+5.00, +5.00] over 20 maps" in x6.measured
    assert "fallback in 5/20 missions" in x6.measured
    assert "`search_astar`" in x6.detail and "steps(ε_fp) − steps(0)" in x6.detail


def test_x6_leaves_x1_to_x5_unchanged() -> None:
    noisy = [_row(experiment=NOISY_BLOCK, false_negative=eps, agent=a, replicate=i, seed=i,
                  bumps=(i % 2) if eps > 0 else 0)
             for a in ("search_andor", "search_astar") for eps in (0.0, 0.1) for i in range(6)]
    before = claims.evaluate_claims(noisy)
    after = claims.evaluate_claims(noisy + _fp_rows(10, {0.1: 1}))
    assert [c.label for c in after] == ["X1", "X2", "X3", "X4", "X5", "X6"]
    assert before[:5] == after[:5]
    assert before[5].verdict == VERDICT_WARN  # no noisy_fp block: not measured


def test_claim_constant_and_fallback_constant() -> None:
    assert 0.0 < EXT_CLAIM_FP < EXT_MAX_FALSE_POSITIVE
    assert EXT_DEBRIS_THRESHOLD < EXT_FALLBACK_MAX_P < 1.0


# --- fix pass: aftershocks with false positives ----------------------------------------------
#: Aftershocks of the dynamic-map missions (frequent, two cells each), their grid and seeds.
AFTERSHOCK_Q = 0.3
AFTERSHOCK_K = 2
AFTERSHOCK_SIZE = 12
AFTERSHOCK_SEEDS = range(3)
#: Noise rates of the dynamic-map checks.
DYN_FN = 0.1
DYN_FP = 0.1


def test_debris_floor_lets_a_certainly_free_cell_turn_debris() -> None:
    """The static filter is deaf at P = 0 when fp > 0 ((1 - fn)·0 / (0 + fp) = 0); with
    ``debris_floor`` a DEBRIS reading starts from the floor, so buried debris can be sensed."""
    p = (0, 1)
    static = ProbBelief(1, 3, None, EXT_PRIOR_DEBRIS, DYN_FN, EXT_DEBRIS_THRESHOLD, DYN_FP)
    dynamic = ProbBelief(1, 3, None, EXT_PRIOR_DEBRIS, DYN_FN, EXT_DEBRIS_THRESHOLD, DYN_FP,
                         debris_floor=EXT_PRIOR_DEBRIS)
    updates = []
    for belief in (static, dynamic):
        belief.update([], here=p)  # the robot stood there: certainly free
        assert belief.p_debris[p] == 0.0
        updates.append(belief.update([(p, Cell.DEBRIS)]))
    assert static.p_debris[p] == 0.0 and updates[0].contradictions == ()
    expected = debris_posterior(EXT_PRIOR_DEBRIS, True, DYN_FN, DYN_FP)
    assert dynamic.p_debris[p] == expected >= EXT_DEBRIS_THRESHOLD
    assert updates[1].contradictions == (p,)
    assert dynamic.copy().debris_floor == EXT_PRIOR_DEBRIS
    # OPEN readings, and DEBRIS readings of cells above the floor, are the plain Bayes update.
    dynamic.update([(p, Cell.OPEN)])
    assert dynamic.p_debris[p] == debris_posterior(expected, False, DYN_FN, DYN_FP)
    high = (0, 2)
    dynamic.update([(high, Cell.DEBRIS)])
    dynamic.update([(high, Cell.DEBRIS)])
    first = debris_posterior(EXT_PRIOR_DEBRIS, True, DYN_FN, DYN_FP)
    assert dynamic.p_debris[high] == debris_posterior(first, True, DYN_FN, DYN_FP)


def test_debris_floor_has_no_effect_without_false_positives() -> None:
    plain = ProbBelief(1, 2, None, EXT_PRIOR_DEBRIS, DYN_FN, EXT_DEBRIS_THRESHOLD)
    floored = ProbBelief(1, 2, None, EXT_PRIOR_DEBRIS, DYN_FN, EXT_DEBRIS_THRESHOLD,
                         debris_floor=EXT_PRIOR_DEBRIS)
    for reading in (Cell.OPEN, Cell.OPEN, Cell.DEBRIS, Cell.OPEN):
        for belief in (plain, floored):
            belief.update([((0, 0), reading)], here=(0, 1))
        assert np.array_equal(plain.p_debris, floored.p_debris)


@pytest.mark.parametrize("bad", [-0.01, EXT_DEBRIS_THRESHOLD, 1.0])
def test_debris_floor_range(bad: float) -> None:
    with pytest.raises(ValueError, match="debris_floor"):
        ProbBelief(2, 2, None, EXT_PRIOR_DEBRIS, DYN_FN, EXT_DEBRIS_THRESHOLD, DYN_FP,
                   debris_floor=bad)


def _aftershock_cfg(q: float = AFTERSHOCK_Q) -> WorldConfig:
    return WorldConfig(slip_p=0.2, aftershock=AftershockConfig(q=q, k=AFTERSHOCK_K))


def test_mission_sets_the_floor_only_with_aftershocks_and_false_positives() -> None:
    grid = multi_survivor_grid("random_rubble", AFTERSHOCK_SIZE, AFTERSHOCK_SIZE, 0, 1)
    cases = (
        (_aftershock_cfg(), DYN_FP, EXT_PRIOR_DEBRIS),
        (_aftershock_cfg(), 0.0, 0.0),
        (_aftershock_cfg(0.0), DYN_FP, 0.0),
    )
    for cfg, fp, floor in cases:
        m = ExtMission(grid, _agent("search_astar", grid, cfg), cfg, 0, false_negative=DYN_FN,
                       false_positive=fp)
        m.start()
        assert m.debris_floor == floor
        assert isinstance(m.agent.belief, ProbBelief) and m.agent.belief.debris_floor == floor


@pytest.mark.parametrize("name", ["search_andor", "search_astar"])
def test_aftershock_debris_on_free_cells_is_sensed(monkeypatch: pytest.MonkeyPatch,
                                                   name: str) -> None:
    """Regression (fix pass): with aftershocks and fp > 0 a DEBRIS reading of a cell held at
    P = 0 (one the robot stood on, which an aftershock may bury) used to leave P = 0, so that
    debris was found only by bumping into it. Every DEBRIS reading now leaves P > 0."""
    original = ProbBelief.update
    deaf_readings: list[Pos] = []

    def spy(self: ProbBelief, percept: Any, touched: Any = (), here: Any = None) -> Any:
        cells = list(percept)
        zero_before = [p for p, cell in cells if cell is Cell.DEBRIS and self.p_debris[p] == 0.0]
        update = original(self, cells, touched, here)
        deaf_readings.extend(zero_before)
        for p, cell in cells:
            if cell is Cell.DEBRIS:
                assert self.p_debris[p] > 0.0, f"DEBRIS reading of {p} left P = 0"
        return update

    monkeypatch.setattr(ProbBelief, "update", spy)
    cfg = _aftershock_cfg()
    for seed in AFTERSHOCK_SEEDS:
        grid = multi_survivor_grid("random_rubble", AFTERSHOCK_SIZE, AFTERSHOCK_SIZE, seed, 1)
        m = run_ext_mission(grid, _agent(name, grid, cfg), cfg, seed, false_negative=DYN_FN,
                            false_positive=DYN_FP)
        assert m.done and m.metrics().aftershock_events > 0
    assert deaf_readings, "no DEBRIS reading hit a P = 0 cell: the check exercised nothing"


# --- fix pass: fallback episodes, one planning call per update -------------------------------
@pytest.mark.parametrize(("scenario", "name", "seed"), [
    ("collapsed_building", "search_andor", 0),
    ("maze_corridors", "search_andor", 1),
    ("maze_corridors", "search_astar", 0),
])
def test_fallback_counts_episodes_not_reentries(monkeypatch: pytest.MonkeyPatch, scenario: str,
                                                name: str, seed: int) -> None:
    """``fallback_replans`` counts strict -> relaxed switches. A re-selection made while relaxed
    that falls back again (frequent for AND-OR, whose envelope re-triggers it) is not counted
    (regression: it was, inflating search_andor's count against search_astar's)."""
    entries: list[bool] = []
    original = SearchAgent._enter_fallback

    def spy(self: SearchAgent, *, new_episode: bool) -> bool:
        switched = original(self, new_episode=new_episode)
        if switched:
            entries.append(new_episode)
        return switched

    monkeypatch.setattr(SearchAgent, "_enter_fallback", spy)
    grid = multi_survivor_grid(scenario, 10, 10, seed, 1)
    cfg = WorldConfig(slip_p=0.2)
    agent = _agent(name, grid, cfg)
    state = {"relaxed": False, "switches": 0}

    def track(_event: ExtStepEvent) -> None:
        if agent.relaxed and not state["relaxed"]:
            state["switches"] += 1
        state["relaxed"] = agent.relaxed

    m = run_ext_mission(grid, agent, cfg, seed, false_negative=0.1, false_positive=0.2,
                        on_event=track)
    assert entries.count(False) >= 1, "no re-entry happened: the check exercised nothing"
    assert m.metrics().fallback_replans == entries.count(True) == state["switches"] >= 1


@pytest.mark.parametrize("name", ["search_andor", "search_astar"])
def test_no_plan_aborts_with_one_planning_call_per_update(monkeypatch: pytest.MonkeyPatch,
                                                          name: str) -> None:
    """A planner that returns no plan (impossible for a reachable target, but not trusted)
    makes the agent abort ``unreachable``; the update plans once, so ``stats.replans`` still
    equals the number of updates with ``replanned=True`` (agents/base.py)."""
    grid = TrueGrid.from_ascii("S...P")
    cfg = WorldConfig(slip_p=0.0)
    agent = _agent(name, grid, cfg)
    calls = [0]
    if name == "search_andor":
        real_plan = agent._andor.plan

        def plan(model: Any, start: Pos) -> PlanResult:
            calls[0] += 1
            if calls[0] == 1:
                return real_plan(model, start)
            return PlanResult(plan=None, stats=PlannerStats(0, 0, 0, 0, 0.0, 0))

        monkeypatch.setattr(agent._andor, "plan", plan)
    else:
        real_astar = ext_agents.astar

        def fake_astar(passable: Any, start: Pos, goal: Pos, tie_break: Any = "g") -> Any:
            calls[0] += 1
            if calls[0] == 1:
                return real_astar(passable, start, goal, tie_break)
            return AStarResult(path=None, nodes_expanded=0)

        monkeypatch.setattr(ext_agents, "astar", fake_astar)
    m = ExtMission(grid, agent, cfg, 0, false_positive=FP)
    m.sensor = PhantomSensor(grid.rows, grid.cols, [(0, 2)])  # forces a fallback replan
    m.run()
    assert m.status is MissionStatus.UNREACHABLE
    assert calls[0] == 2 and agent.stats.plans_made == 2
    assert agent.stats.replans == sum(e.replanned for e in m.events) == 1
    assert m.metrics().fallback_replans == 1


def test_summary_keeps_round1_column_positions() -> None:
    """Round-2 summary columns (the ``false_positive`` key, ``fallback_replans_*``) come after
    every round-1 column, as in ``raw.csv``."""
    round2 = ("false_positive", "fallback_replans_mean", "fallback_replans_lo",
              "fallback_replans_hi")
    assert SUMMARY_COLUMNS[-len(round2):] == round2
    assert SUMMARY_COLUMNS.index("agent") == SUMMARY_COLUMNS.index("false_negative") + 1
