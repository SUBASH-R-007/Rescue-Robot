"""Phase 5 stretch agents: ``dstar_lite`` (D* Lite) and ``vi_expected`` (value iteration).

* Registry: ``make_agent`` builds both; ``AGENT_NAMES`` keeps the three compared agents and
  ``ALL_AGENT_NAMES`` lists all five.
* Value iteration: exact closed forms on 1xN corridors (each step advances with probability
  ``1 - p_cell``, otherwise the robot stays — slips and wall-blocked skids — so the expected
  steps are ``Σ 1 / (1 - p_cell)``), the Bellman optimality equation, a strong-cyclic greedy
  policy on random maps, and expected cost ≤ the AND-OR policy's (both evaluated exactly by
  a linear solve of the policy's Markov chain).
* D* Lite: path lengths equal BFS on random maps under random incremental debris additions
  and moving starts.
* Missions: both agents rescue with 0 bumps (r = 1), ``vi_expected`` with 0 unplanned
  states; same seed ⇒ identical events.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from rescue.agents import AGENT_NAMES, make_agent
from rescue.config import AftershockConfig, PlannerConfig, WorldConfig
from rescue.grid import Pos, TrueGrid
from rescue.model import ACTIONS, Action, ProblemModel, TerrainSlip, UniformSlip
from rescue.planners.andor import and_or_search
from rescue.planners.bfs import reachable, shortest_path_length
from rescue.planners.validate import is_strong_cyclic
from rescue.runner import Mission, make_transition, run_mission
from tests.helpers_maps import random_solvable_grid

vi_mod = pytest.importorskip("rescue.planners.value_iteration")

SLIP_PS = (0.0, 0.1, 0.2, 0.3, 0.4)
STRETCH = ("dstar_lite", "vi_expected")


# --- helpers ---------------------------------------------------------------------------------
def outcome_probs(model: ProblemModel, s: Pos, a: Action) -> list[tuple[Pos, float]]:
    """``(next state, probability)`` of every outcome of ``a`` at ``s`` (merged labels summed)."""
    dist = dict(model.transition.distribution(s))
    return [(o.state, sum(dist[k] for k in o.labels)) for o in model.results(s, a)]


def policy_cost(policy: dict[Pos, Action], model: ProblemModel, start: Pos) -> dict[Pos, float]:
    """Exact expected steps to the goal under ``policy`` (linear solve of its Markov chain)."""
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
    states = sorted(s for s in reach if s != model.goal)
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    if n == 0:
        return {model.goal: 0.0}
    a_mat = np.eye(n)
    b = np.ones(n)
    for s in states:
        for t, prob in outcome_probs(model, s, policy[s]):
            if t != model.goal:
                a_mat[index[s], index[t]] -= prob
    v = np.linalg.solve(a_mat, b)
    out = {s: float(v[index[s]]) for s in states}
    out[model.goal] = 0.0
    return out


def model_of(grid: TrueGrid, p: float, terrain_p: float = 0.4) -> ProblemModel:
    """Planner model on the grid's true map with the world's dynamics."""
    world = WorldConfig(slip_p=p, terrain_high_p=terrain_p if p > 0 else 0.0)
    return ProblemModel(grid.passable_mask(), grid.survivor, make_transition(grid, world))


# --- registry --------------------------------------------------------------------------------
def test_registry_names() -> None:
    from rescue import agents

    assert AGENT_NAMES == ("andor", "astar_fixed", "astar_replan")
    assert tuple(agents.ALL_AGENT_NAMES) == (*AGENT_NAMES, "dstar_lite", "vi_expected")
    for name in agents.ALL_AGENT_NAMES:
        agent = make_agent(name, UniformSlip(0.2))
        assert agent.name == name
    with pytest.raises(ValueError):
        make_agent("lao_star", UniformSlip(0.2))


# --- value iteration: closed forms -----------------------------------------------------------
@pytest.mark.parametrize("p", SLIP_PS)
@pytest.mark.parametrize("n", [2, 3, 7, 12])
def test_vi_corridor_closed_form(n: int, p: float) -> None:
    passable = np.ones((1, n), dtype=bool)
    model = ProblemModel(passable, (0, n - 1), UniformSlip(p))
    values, policy, iterations = vi_mod.value_iteration(model)
    assert iterations >= 1
    for c in range(n):
        assert values[(0, c)] == pytest.approx((n - 1 - c) / (1 - p), rel=1e-7, abs=1e-9)
    assert (0, n - 1) not in policy
    for c in range(n - 1):
        assert policy[(0, c)] is Action.E


def test_vi_corridor_terrain_chain() -> None:
    """Terrain-dependent slip (D4): expected steps add up 1 / (1 - p(cell)) along the chain."""
    n = 9
    high = np.zeros((1, n), dtype=bool)
    high[0, 2:5] = True
    base, hi = 0.1, 0.4
    model = ProblemModel(np.ones((1, n), dtype=bool), (0, n - 1), TerrainSlip(base, hi, high))
    values, _, _ = vi_mod.value_iteration(model)
    for c in range(n):
        expected = sum(1 / (1 - (hi if high[0, j] else base)) for j in range(c, n - 1))
        assert values[(0, c)] == pytest.approx(expected, rel=1e-7)


def test_vi_vertical_corridor_and_blocked_cells() -> None:
    """Goal at the top of a 6x1 column; debris splits off an unreachable part elsewhere."""
    p = 0.3
    model = ProblemModel(np.ones((6, 1), dtype=bool), (0, 0), UniformSlip(p))
    values, policy, _ = vi_mod.value_iteration(model)
    for r in range(6):
        assert values[(r, 0)] == pytest.approx(r / (1 - p), rel=1e-7)
    assert all(a is Action.N for a in policy.values())
    passable = np.ones((1, 5), dtype=bool)
    passable[0, 2] = False
    model = ProblemModel(passable, (0, 4), UniformSlip(p))
    values, policy, _ = vi_mod.value_iteration(model)
    assert math.isinf(values[(0, 0)]) and math.isinf(values[(0, 1)])
    assert values[(0, 3)] == pytest.approx(1 / (1 - p))
    assert (0, 0) not in policy and (0, 1) not in policy


def test_vi_p0_equals_bfs_distance() -> None:
    for seed in range(10):
        grid = random_solvable_grid(8, 9, 0.25, seed)
        model = model_of(grid, 0.0)
        values, _, _ = vi_mod.value_iteration(model)
        for s in model.states():
            d = shortest_path_length(grid.passable_mask(), s, grid.survivor)
            if d is None:
                assert math.isinf(values[s])
            else:
                assert values[s] == pytest.approx(d)


# --- value iteration: optimality and strong cyclicity on random maps ---------------------------
def _random_cases(count: int) -> list[tuple[TrueGrid, float]]:
    cases = []
    for i in range(count):
        rng = np.random.default_rng([4242, i])
        rows, cols = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        terrain = 0.2 if i % 3 == 0 else 0.0
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.1, 0.35)), 900 + i, terrain)
        cases.append((grid, SLIP_PS[1 + i % 4]))
    return cases


def test_vi_bellman_strong_cyclic_and_beats_andor() -> None:
    better = 0
    for grid, p in _random_cases(40):
        model = model_of(grid, p)
        values, policy, _ = vi_mod.value_iteration(model)
        start, goal = grid.start, grid.survivor
        assert values[goal] == 0.0
        assert math.isfinite(values[start])
        assert is_strong_cyclic(policy, model, start)
        # Bellman optimality at every finite state; the greedy policy attains the minimum.
        for s in model.states():
            if s == goal or math.isinf(values[s]):
                continue
            q = {
                a: 1 + sum(prob * values[t] for t, prob in outcome_probs(model, s, a) if prob > 0)
                for a in ACTIONS
            }
            assert values[s] == pytest.approx(min(q.values()), rel=1e-6, abs=1e-6)
            assert q[policy[s]] == pytest.approx(values[s], rel=1e-6, abs=1e-6)
        # Exact evaluation of the VI policy agrees with its values.
        exact = policy_cost(policy, model, start)
        for s, v in exact.items():
            assert v == pytest.approx(values[s], rel=1e-6, abs=1e-6)
        # The AND-OR plan's policy can only be as good or worse in expectation.
        andor = and_or_search(model, start)
        assert andor.plan is not None
        andor_cost = policy_cost(andor.plan.to_policy(), model, start)
        assert values[start] <= andor_cost[start] + 1e-6
        if values[start] < andor_cost[start] - 1e-6:
            better += 1
    # Non-vacuous: the AND-OR plan (first strong-cyclic plan found) is sometimes suboptimal.
    assert better >= 1


def test_vi_planner_interface() -> None:
    grid = random_solvable_grid(7, 7, 0.25, 3)
    model = model_of(grid, 0.2)
    planner = vi_mod.ValueIterationPlanner(PlannerConfig(validate=True))
    result = planner.plan(model, grid.start)
    assert result.success and result.plan is not None
    assert result.stats.validated is True and not result.stats.fallback_used
    assert is_strong_cyclic(result.plan.to_policy(), model, grid.start)
    assert planner.last_values[grid.start] == pytest.approx(
        vi_mod.value_iteration(model).values[grid.start]
    )
    # Unreachable goal: no plan.
    passable = np.ones((1, 5), dtype=bool)
    passable[0, 2] = False
    blocked = ProblemModel(passable, (0, 4), UniformSlip(0.2))
    assert vi_mod.ValueIterationPlanner().plan(blocked, (0, 0)).plan is None


def test_vi_deterministic() -> None:
    grid = random_solvable_grid(8, 8, 0.25, 17, 0.2)
    model = model_of(grid, 0.3)
    a = vi_mod.value_iteration(model)
    b = vi_mod.value_iteration(model_of(grid, 0.3))
    assert a.policy == b.policy
    assert a.values == b.values


# --- D* Lite -----------------------------------------------------------------------------------
def _check_path(path: list[Pos], passable: np.ndarray, start: Pos, goal: Pos) -> None:
    """A 4-connected path of passable cells from ``start`` to ``goal``."""
    assert path[0] == start and path[-1] == goal
    for a, b in itertools.pairwise(path):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    for p in path:
        assert passable[p]


def test_dstar_lite_matches_bfs_under_incremental_changes() -> None:
    dstar_mod = pytest.importorskip("rescue.planners.dstar_lite")
    checks = 0
    for i in range(30):
        rng = np.random.default_rng([5150, i])
        rows, cols = int(rng.integers(5, 13)), int(rng.integers(5, 13))
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.1, 0.3)), 600 + i)
        goal = grid.survivor
        passable = grid.passable_mask().copy()
        pos = grid.start
        search = dstar_mod.DStarLite(passable, pos, goal)
        for _ in range(12):
            result = search.plan(passable, pos)
            expected = shortest_path_length(passable, pos, goal)
            if expected is None:
                assert result.path is None
                break
            assert result.path is not None
            assert len(result.path) - 1 == expected
            assert search.rhs(pos) == expected  # g(start) may stay inf (optimized version)
            _check_path(result.path, passable, pos, goal)
            checks += 1
            # Move the start a few cells along the path (sometimes jump: a skid or a slip).
            if rng.random() < 0.2:
                free = [tuple(map(int, q)) for q in np.argwhere(passable)]
                pos = free[int(rng.integers(len(free)))]
            else:
                pos = result.path[min(len(result.path) - 1, int(rng.integers(1, 4)))]
            if pos == goal:
                break
            # Add random debris (never on the robot or the goal); sometimes clear some.
            for _ in range(int(rng.integers(1, 4))):
                q = (int(rng.integers(rows)), int(rng.integers(cols)))
                if q not in (pos, goal):
                    passable[q] = False
            if rng.random() < 0.15:
                blocked = np.argwhere(~passable)
                if len(blocked):
                    r, c = blocked[int(rng.integers(len(blocked)))]
                    passable[r, c] = True
    assert checks >= 100


def test_dstar_lite_unreachable_then_reopened() -> None:
    dstar_mod = pytest.importorskip("rescue.planners.dstar_lite")
    passable = np.ones((3, 5), dtype=bool)
    search = dstar_mod.DStarLite(passable, (1, 0), (1, 4))
    assert len(search.plan(passable, (1, 0)).path) == 5
    passable[:, 2] = False
    assert search.plan(passable, (1, 0)).path is None
    passable[0, 2] = True
    path = search.plan(passable, (1, 0)).path
    assert path is not None and len(path) - 1 == 6
    _check_path(path, passable, (1, 0), (1, 4))


# --- missions ----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", STRETCH)
def test_stretch_agents_rescue_without_bumps(name: str) -> None:
    for i in range(24):
        rng = np.random.default_rng([77, i])
        rows, cols = int(rng.integers(5, 11)), int(rng.integers(5, 11))
        terrain = 0.2 if i % 4 == 0 else 0.0
        grid = random_solvable_grid(rows, cols, float(rng.uniform(0.1, 0.35)), 300 + i, terrain)
        p = SLIP_PS[i % len(SLIP_PS)]
        shock = AftershockConfig(q=0.1, k=1) if i % 3 == 2 else AftershockConfig()
        world = WorldConfig(slip_p=p, sensor_radius=1, aftershock=shock)
        agent = make_agent(name, make_transition(grid, world))
        mission = Mission(grid, agent, world_config=world, seed=i)
        event = mission.start()
        while True:
            # Every step, aftershocks included, leaves the survivor reachable from the robot.
            passable = mission.world.grid.passable_mask()
            assert reachable(passable, event.new_pos, grid.survivor), (name, i, event.t)
            if mission.done:
                break
            event = mission.step()
        m = mission.metrics()
        assert m.success, (name, i, m.failure_reason)
        assert m.bumps == 0
        if name == "vi_expected":
            assert m.unplanned_states == 0
            assert m.planner_fallbacks == 0


@pytest.mark.parametrize("name", STRETCH)
def test_stretch_agents_deterministic(name: str) -> None:
    grid = random_solvable_grid(9, 9, 0.25, 5, 0.2)
    world = WorldConfig(slip_p=0.3, aftershock=AftershockConfig(q=0.2, k=1))
    runs = [
        run_mission(grid, make_agent(name, make_transition(grid, world)), world, seed=11)
        for _ in range(2)
    ]
    assert runs[0].events == runs[1].events
    a, b = (r.metrics().as_dict() for r in runs)
    for key in ("planning_ms",):
        a.pop(key)
        b.pop(key)
    assert a == b


# --- fix pass: vectorised tables, agent planner, expected steps on fallback --------------------
def test_transition_tables_match_problem_model_results() -> None:
    """The vectorised tables give every (s, a) the same successor distribution as
    ``ProblemModel.results`` with merged labels summed (support and probabilities)."""
    for i in range(40):
        rng = np.random.default_rng([515, i])
        rows, cols = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        passable = rng.random((rows, cols)) > rng.uniform(0.0, 0.4)
        goal = (int(rng.integers(rows)), int(rng.integers(cols)))
        passable[goal] = True
        if i % 2:
            transition = TerrainSlip(float(rng.uniform(0, 0.4)), float(rng.uniform(0, 0.4)),
                                     rng.random((rows, cols)) < 0.4)
        else:
            transition = UniformSlip(SLIP_PS[i % len(SLIP_PS)])
        model = ProblemModel(passable, goal, transition)
        tables = vi_mod.transition_tables(model)
        assert tables.states == model.states()
        assert tables.states[tables.goal_index] == goal
        for si, s in enumerate(tables.states):
            for ai, a in enumerate(ACTIONS):
                got: dict[Pos, float] = {}
                for k in range(tables.succ.shape[2]):
                    prob = float(tables.prob[ai, si, k])
                    if prob > 0.0:
                        t = tables.states[int(tables.succ[ai, si, k])]
                        got[t] = got.get(t, 0.0) + prob
                want = dict(outcome_probs(model, s, a))
                assert got.keys() == want.keys(), (i, s, a)
                for t, prob in want.items():
                    assert got[t] == pytest.approx(prob, abs=1e-15)


def test_vi_agent_planner_is_value_iteration() -> None:
    grid = random_solvable_grid(6, 6, 0.2, 3)
    agent = make_agent("vi_expected", make_transition(grid, WorldConfig()))
    assert isinstance(agent.planner, vi_mod.ValueIterationPlanner)
    assert agent.vi_planner is agent.planner
    assert math.isinf(agent.expected_steps)
    mission = run_mission(grid, agent, WorldConfig(), seed=1)
    assert mission.metrics().success
    assert agent.snapshots and agent.planner.last_iterations > 0


def test_vi_agent_expected_steps_nan_when_reference_policy_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``validate`` a rejected VI policy is replaced by the fixpoint reference policy; the
    VI values then do not describe the plan in force, so ``expected_steps`` is NaN."""
    grid = random_solvable_grid(6, 6, 0.2, 4)
    world = WorldConfig(slip_p=0.2)
    calls = {"n": 0}
    real = vi_mod.is_strong_cyclic

    def reject_first(policy, model, start):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return False if calls["n"] == 1 else real(policy, model, start)

    monkeypatch.setattr(vi_mod, "is_strong_cyclic", reject_first)
    agent = make_agent("vi_expected", make_transition(grid, world),
                       PlannerConfig(validate=True))
    mission = Mission(grid, agent, world_config=world, seed=0)
    mission.start()
    assert agent.stats.planner_fallbacks == 1
    assert agent.snapshots[0].plan is not None
    assert math.isnan(agent.expected_steps)
    monkeypatch.setattr(vi_mod, "is_strong_cyclic", real)
    agent._replan()
    assert agent.stats.planner_fallbacks == 1  # this plan is value iteration's own
    assert agent.expected_steps == pytest.approx(agent.vi_planner.last_values[agent.pos])
