"""Round 2 stretch planner: LAO* (``planners/lao_star.py``) and the ``lao_expected`` agent.

* Optimality: on 200+ seeded random maps (sizes 3..15, p in [0, 0.4], a third with high-slip
  terrain) the LAO* start value equals value iteration's optimum within 1e-6, and so does the
  exact expected cost of the LAO* policy (linear solve of its absorbing Markov chain on the
  closed envelope); the policy is strong cyclic and closed.
* Heuristic: admissible (``h <= V*`` wherever finite); with p = 0 the value is the BFS distance.
* Search effort: expanded states never exceed the free states reachable from the start, and
  are strictly fewer than all free states on most 20x20 maps (ratios printed with ``-s``).
* Agent: registry, missions with 0 bumps / 0 unplanned states (r = 1), determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rescue import mapgen
from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES, STRETCH_AGENT_NAMES, make_agent
from rescue.agents.lao_expected import LaoExpectedAgent
from rescue.config import AftershockConfig, PlannerConfig, WorldConfig
from rescue.grid import Pos, TrueGrid
from rescue.model import Action, ProblemModel, UniformSlip
from rescue.planners.bfs import UNREACHABLE, bfs_distances
from rescue.planners.lao_star import LaoResult, LaoStarPlanner, heuristic, lao_star
from rescue.planners.validate import is_strong_cyclic
from rescue.planners.value_iteration import value_iteration
from rescue.runner import Mission, make_transition, run_mission
from tests.helpers_maps import random_solvable_grid

#: Random optimality cases and the agreement tolerance with value iteration.
N_RANDOM_CASES = 210
VALUE_TOL = 1e-6
#: Most of the 20x20 maps must leave some free state unexpanded.
EFFORT_MAPS = 20
EFFORT_MIN_STRICT = 15


# --- helpers ---------------------------------------------------------------------------------
def outcome_probs(model: ProblemModel, s: Pos, a: Action) -> list[tuple[Pos, float]]:
    """``(next state, probability)`` of every outcome of ``a`` at ``s`` (merged labels summed)."""
    dist = dict(model.transition.distribution(s))
    return [(o.state, sum(dist[k] for k in o.labels)) for o in model.results(s, a)]


def envelope(policy: dict[Pos, Action], model: ProblemModel, start: Pos) -> set[Pos]:
    """States reachable from ``start`` under ``policy`` (KeyError if one has no action)."""
    reach = {start}
    frontier = [start]
    while frontier:
        s = frontier.pop()
        if s == model.goal:
            continue
        for t, prob in outcome_probs(model, s, policy[s]):
            if prob > 0 and t not in reach:
                reach.add(t)
                frontier.append(t)
    return reach


def exact_cost(policy: dict[Pos, Action], model: ProblemModel, start: Pos) -> float:
    """Exact expected steps from ``start``: solve ``(I - Q) x = 1`` on the transient states."""
    states = sorted(s for s in envelope(policy, model, start) if s != model.goal)
    if not states:
        return 0.0
    index = {s: i for i, s in enumerate(states)}
    a_mat = np.eye(len(states))
    for s in states:
        for t, prob in outcome_probs(model, s, policy[s]):
            if t != model.goal:
                a_mat[index[s], index[t]] -= prob
    return float(np.linalg.solve(a_mat, np.ones(len(states)))[index[start]])


def model_of(grid: TrueGrid, p: float, terrain_p: float = 0.4) -> ProblemModel:
    """Planner model on the grid's true map with the world's dynamics."""
    world = WorldConfig(slip_p=p, terrain_high_p=terrain_p if p > 0 else 0.0)
    return ProblemModel(grid.passable_mask(), grid.survivor, make_transition(grid, world))


def random_cases(count: int) -> list[tuple[TrueGrid, float]]:
    """Seeded maps of 3..15 cells a side, p uniform in [0, 0.4], every third with terrain."""
    cases = []
    for i in range(count):
        rng = np.random.default_rng([8101, i])
        rows, cols = int(rng.integers(3, 16)), int(rng.integers(3, 16))
        terrain = 0.2 if i % 3 == 0 else 0.0
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.0, 0.35)), 5000 + i, terrain)
        cases.append((grid, float(rng.uniform(0.0, 0.4))))
    return cases


# --- optimality ------------------------------------------------------------------------------
def test_lao_matches_value_iteration_on_random_maps() -> None:
    terrain_cases = 0
    for i, (grid, p) in enumerate(random_cases(N_RANDOM_CASES)):
        model = model_of(grid, p)
        start, goal = grid.start, grid.survivor
        assert start != goal
        terrain_cases += bool(np.any(grid.high_slip))
        result = lao_star(model, start)
        vi = value_iteration(model)
        assert result.converged, i
        assert result.policy is not None, i
        assert result.values[start] == pytest.approx(vi.values[start], abs=VALUE_TOL), i
        # Every state of the solution graph has its optimal value, not just the start.
        reach = envelope(result.policy, model, start)
        assert set(result.policy) == reach - {goal}, i  # closed, no extra states
        for s in reach:
            assert result.values[s] == pytest.approx(vi.values[s], abs=VALUE_TOL), (i, s)
        assert exact_cost(result.policy, model, start) == pytest.approx(
            vi.values[start], abs=VALUE_TOL), i
        assert is_strong_cyclic(result.policy, model, start), i
    assert terrain_cases >= N_RANDOM_CASES // 4  # TerrainSlip is really exercised


def test_heuristic_is_admissible() -> None:
    for grid, p in random_cases(60):
        model = model_of(grid, p)
        h = heuristic(model)
        vi = value_iteration(model)
        for s in model.states():
            if math.isfinite(h[s]):
                assert h[s] <= vi.values[s] + VALUE_TOL
            else:
                assert math.isinf(vi.values[s])


def test_p0_value_is_bfs_distance() -> None:
    for i in range(30):
        rng = np.random.default_rng([8102, i])
        rows, cols = int(rng.integers(3, 16)), int(rng.integers(3, 16))
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.0, 0.35)), 6000 + i)
        model = ProblemModel(grid.passable_mask(), grid.survivor, UniformSlip(0.0))
        result = lao_star(model, grid.start)
        d = int(bfs_distances(grid.passable_mask(), grid.survivor)[grid.start])
        assert result.values[grid.start] == pytest.approx(d, abs=VALUE_TOL)
        assert result.policy is not None and len(result.policy) == d  # one path, no branches


def test_corridor_closed_form() -> None:
    n, p = 9, 0.3
    model = ProblemModel(np.ones((1, n), dtype=bool), (0, n - 1), UniformSlip(p))
    result = lao_star(model, (0, 0))
    # Skids hit the walls, so each step advances with probability 1 - p.
    assert result.values[(0, 0)] == pytest.approx((n - 1) / (1 - p), abs=VALUE_TOL)
    assert result.policy == {(0, c): Action.E for c in range(n - 1)}


def test_unreachable_and_trivial_starts() -> None:
    passable = np.ones((1, 5), dtype=bool)
    passable[0, 2] = False
    blocked = ProblemModel(passable, (0, 4), UniformSlip(0.2))
    result = lao_star(blocked, (0, 0))
    assert result.policy is None and result.expanded_states == 0
    assert LaoStarPlanner().plan(blocked, (0, 0)).plan is None
    at_goal = lao_star(blocked, (0, 4))
    assert at_goal == LaoResult({(0, 4): 0.0}, {}, 0, 0, 0, True)
    # A start on debris has no policy either.
    assert lao_star(blocked, (0, 2)).policy is None


def test_max_iterations_cap_reports_no_convergence() -> None:
    grid = random_solvable_grid(10, 10, 0.2, 7)
    result = lao_star(model_of(grid, 0.2), grid.start, max_iterations=1)
    assert not result.converged
    assert result.iterations == 1
    assert result.policy is None  # the best graph still has tips


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0.0},
        {"tolerance": -1e-9},
        {"tolerance": math.inf},
        {"tolerance": math.nan},
        {"expansion_tolerance": 0.0},
        {"expansion_tolerance": -0.5},
        {"expansion_tolerance": math.nan},
        {"max_iterations": -1},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    # A zero tolerance can never be met (every sweep loop stops on residual < tolerance), so
    # it used to run VI_MAX_ITERS sweeps per round instead of failing fast.
    model = ProblemModel(np.ones((4, 4), dtype=bool), (3, 3), UniformSlip(0.2))
    with pytest.raises(ValueError):
        lao_star(model, (0, 0), **kwargs)  # type: ignore[arg-type]


def test_infinite_expansion_tolerance_and_zero_cap_are_valid() -> None:
    model = ProblemModel(np.ones((4, 4), dtype=bool), (3, 3), UniformSlip(0.2))
    vi = value_iteration(model)
    sweep_once = lao_star(model, (0, 0), expansion_tolerance=math.inf)
    assert sweep_once.converged
    assert sweep_once.values[(0, 0)] == pytest.approx(vi.values[(0, 0)], abs=VALUE_TOL)
    no_rounds = lao_star(model, (0, 0), max_iterations=0)
    assert not no_rounds.converged and no_rounds.policy is None and no_rounds.iterations == 0


def test_convergence_tests_have_their_own_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import rescue.planners.lao_star as lao_module

    model = ProblemModel(np.ones((1, 9), dtype=bool), (0, 8), UniformSlip(0.3))
    # Expansion rounds are capped by max_iterations alone: a convergence-test cap of 1 does
    # not stop the 8 rounds this corridor needs (the old loop shared one VI_MAX_ITERS budget).
    monkeypatch.setattr(lao_module, "LAO_MAX_CONVERGENCE_TESTS", 1)
    capped_tests = lao_star(model, (0, 0))
    assert capped_tests.converged and capped_tests.iterations > 1
    # With no convergence test allowed the search reports no convergence, but the expansion
    # rounds still ran to a closed best graph.
    monkeypatch.setattr(lao_module, "LAO_MAX_CONVERGENCE_TESTS", 0)
    no_tests = lao_star(model, (0, 0))
    assert not no_tests.converged
    assert no_tests.iterations == capped_tests.iterations
    assert no_tests.policy == {(0, c): Action.E for c in range(8)}


def test_planner_construction_loads_csgraph_outside_timed_plans() -> None:
    # Regression: the lazy csgraph import (0.1–0.5 s) used to land in the first plan's
    # planning_ms of every harness worker. A fresh interpreter shows where it loads now.
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from rescue.planners.lao_star import LaoStarPlanner, warm_up\n"
        "before = 'scipy.sparse.csgraph' in sys.modules\n"
        "LaoStarPlanner()\n"
        "print(before, 'scipy.sparse.csgraph' in sys.modules, warm_up())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["False", "True", "True"]


# --- search effort ---------------------------------------------------------------------------
def test_expanded_states_bounded_by_reachable_states() -> None:
    for grid, p in random_cases(80):
        model = model_of(grid, p)
        result = lao_star(model, grid.start)
        reach = int(np.count_nonzero(bfs_distances(grid.passable_mask(), grid.start)
                                     != UNREACHABLE))
        assert 0 < result.expanded_states <= reach


@pytest.mark.parametrize("scenario", ["open", "random_rubble"])
def test_lao_expands_fewer_states_than_value_iteration_sweeps(scenario: str) -> None:
    ratios = []
    strict = 0
    for seed in range(EFFORT_MAPS):
        if scenario == "open":
            rng = np.random.default_rng([8103, seed])
            cells = [(r, c) for r in range(20) for c in range(20)]
            start, goal = (cells[i] for i in rng.choice(len(cells), size=2, replace=False))
            model = ProblemModel(np.ones((20, 20), dtype=bool), goal, UniformSlip(0.2))
        else:
            grid = mapgen.generate(scenario, 20, 20, seed)
            start = grid.start
            model = ProblemModel(grid.passable_mask(), grid.survivor, UniformSlip(0.2))
        result = lao_star(model, start)
        free = len(model.states())
        ratios.append(result.expanded_states / free)
        strict += result.expanded_states < free
    print(f"\n{scenario} 20x20 p=0.2: expanded/free mean {np.mean(ratios):.3f}, "
          f"min {min(ratios):.3f}, max {max(ratios):.3f}, strictly fewer on {strict}/"
          f"{EFFORT_MAPS}")
    assert strict >= EFFORT_MIN_STRICT


# --- planner and agent -----------------------------------------------------------------------
def test_planner_interface_and_validation() -> None:
    grid = random_solvable_grid(8, 8, 0.25, 3, 0.2)
    model = model_of(grid, 0.3)
    planner = LaoStarPlanner(PlannerConfig(validate=True))
    result = planner.plan(model, grid.start)
    assert result.success and result.plan is not None
    assert result.stats.validated is True and not result.stats.fallback_used
    assert planner.last_result is not None
    assert result.stats.nodes_expanded == planner.last_result.expanded_states
    assert result.plan.to_policy() == planner.last_result.policy
    assert result.stats.plan_size == len(planner.last_result.policy or {})
    assert planner.last_values[grid.start] == pytest.approx(
        value_iteration(model).values[grid.start], abs=VALUE_TOL)


def test_registry() -> None:
    assert "lao_expected" in STRETCH_AGENT_NAMES and "lao_expected" in ALL_AGENT_NAMES
    assert "lao_expected" not in AGENT_NAMES
    agent = make_agent("lao_expected", UniformSlip(0.2))
    assert isinstance(agent, LaoExpectedAgent)
    assert agent.planner is agent.lao_planner
    assert math.isinf(agent.expected_steps)


def test_agent_rescues_without_bumps_or_unplanned_states() -> None:
    for i in range(24):
        rng = np.random.default_rng([8104, i])
        rows, cols = int(rng.integers(5, 13)), int(rng.integers(5, 13))
        terrain = 0.2 if i % 4 == 0 else 0.0
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.1, 0.35)), 7000 + i, terrain)
        p = (0.0, 0.1, 0.2, 0.3, 0.4)[i % 5]
        shock = AftershockConfig(q=0.1, k=1) if i % 3 == 2 else AftershockConfig()
        world = WorldConfig(slip_p=p, sensor_radius=1, aftershock=shock)
        agent = make_agent("lao_expected", make_transition(grid, world))
        mission = run_mission(grid, agent, world, seed=i)
        m = mission.metrics()
        assert m.success, (i, m.failure_reason)
        assert m.bumps == 0
        assert m.unplanned_states == 0
        assert m.planner_fallbacks == 0
        assert math.isfinite(agent.expected_steps)


def test_agent_expected_steps_match_vi_agent_on_first_plan() -> None:
    grid = random_solvable_grid(10, 10, 0.25, 11, 0.2)
    world = WorldConfig(slip_p=0.3)
    lao = make_agent("lao_expected", make_transition(grid, world))
    vi = make_agent("vi_expected", make_transition(grid, world))
    for agent in (lao, vi):
        Mission(grid, agent, world_config=world, seed=3).start()
    assert math.isfinite(lao.expected_steps)
    assert lao.expected_steps == pytest.approx(vi.expected_steps, abs=VALUE_TOL)


def test_agent_deterministic() -> None:
    grid = random_solvable_grid(9, 9, 0.25, 5, 0.2)
    world = WorldConfig(slip_p=0.3, aftershock=AftershockConfig(q=0.2, k=1))
    runs = [
        run_mission(grid, make_agent("lao_expected", make_transition(grid, world)), world,
                    seed=11)
        for _ in range(2)
    ]
    assert runs[0].events == runs[1].events
    a, b = (r.metrics().as_dict() for r in runs)
    a.pop("planning_ms")
    b.pop("planning_ms")
    assert a == b
    model = model_of(grid, 0.3)
    assert lao_star(model, grid.start) == lao_star(model_of(grid, 0.3), grid.start)


def test_agent_expected_steps_nan_when_reference_policy_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``validate`` a rejected LAO* policy is replaced by the fixpoint reference policy;
    the LAO* values then do not describe the plan in force, so ``expected_steps`` is NaN."""
    import rescue.planners.lao_star as lao_mod

    grid = random_solvable_grid(6, 6, 0.2, 4)
    world = WorldConfig(slip_p=0.2)
    calls = {"n": 0}
    real = lao_mod.is_strong_cyclic

    def reject_first(policy, model, start):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return False if calls["n"] == 1 else real(policy, model, start)

    monkeypatch.setattr(lao_mod, "is_strong_cyclic", reject_first)
    agent = make_agent("lao_expected", make_transition(grid, world),
                       PlannerConfig(validate=True))
    Mission(grid, agent, world_config=world, seed=0).start()
    assert agent.stats.planner_fallbacks == 1
    assert agent.snapshots[0].plan is not None
    assert math.isnan(agent.expected_steps)
    monkeypatch.setattr(lao_mod, "is_strong_cyclic", real)
    agent._replan()
    assert agent.stats.planner_fallbacks == 1  # this plan is LAO*'s own
    assert agent.expected_steps == pytest.approx(agent.lao_planner.last_values[agent.pos])
