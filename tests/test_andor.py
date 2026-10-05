"""Tests for rescue.planners.andor (docs/plan.md sections 3.7 and 4, decisions D13/D14)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import rescue.planners.andor as andor_mod
from rescue.config import PlannerConfig
from rescue.grid import Pos
from rescue.model import Action, OutcomeKind, ProblemModel, TerrainSlip, UniformSlip
from rescue.plan import Act, Branch, Goal, Loop, Plan, Ref
from rescue.planners.andor import (
    AndOrPlanner,
    AndOrSearch,
    PlannerStats,
    PlanResult,
    and_or_search,
)
from rescue.planners.bfs import reachable, shortest_path_length
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy
from rescue.planners.validate import is_strong_cyclic, validate_policy
from tests.helpers_phase2 import (
    HOSTILE_CLASSES,
    AllStayFirst,
    BumpFirst,
    BumpOnly,
    Textbook,
    all_stay_actions,
    ascii_model,
    bump_actions,
    check_structure,
    deck_model,
    deck_start,
    farthest_cell,
    random_instance,
    run_deterministic,
    run_search,
    serpentine_mask,
    walk_branches,
    walk_nodes,
)

ARR = OutcomeKind.ARRIVED
SLIP = OutcomeKind.SLIPPED
SKL = OutcomeKind.SKID_L
SKR = OutcomeKind.SKID_R

DECK_ROOT = "E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}"
#: Corridor with a dead-end alcove at (1,2): an E-skid from (0,2) enters it, and its only exit
#: is back to (0,2), so every branch of the alcove's action is a Loop.
ALCOVE = "S...P\n##.##"
#: Goal walled off from the start: (3,4) and (4,4) form a closed pocket.
WALLED = "S....\n.....\n...##\n...#.\n...#P"
#: Start enclosed by debris; goal elsewhere.
ENCLOSED = "S#..\n#...\n...P"
#: 4x1 corridor of docs/algorithm.md §5.3: with the textbook order, x = (0,0) tries E, whose
#: skid-R reaches y = (1,0), and y's first action N leads only back to x or stays (mutual loop).
MUTUAL_LOOP = "S\n.\n.\nP"
#: The dead plan the relaxed syntactic rule R accepts on MUTUAL_LOOP (docs/algorithm.md §5.3).
MUTUAL_LOOP_RULE_R = "E → {slipped: retry E, skid→(2,1): N-plan}"


# --- search subclasses with forced action orderings ------------------------------------------
# BumpFirst, BumpOnly, AllStayFirst, Reversed, Textbook and the seeded permutations live in
# tests/helpers_phase2.py (shared with the property tests).
class RelaxedRule(Textbook):
    """Rule R of docs/algorithm.md §5.1: some branch is not a retry of ``s`` itself."""

    def has_progress(self, s: Pos) -> bool:
        return any(o.state != s for o in self.model.results(s, self.solved[s]))


class AllStayFirstRecorded(AllStayFirst):
    """AllStayFirst that records every HAS_PROGRESS verdict."""

    def __init__(self, model: ProblemModel, config: PlannerConfig | None = None) -> None:
        super().__init__(model, config)
        self.verdicts: list[tuple[Pos, Action, bool]] = []

    def has_progress(self, s: Pos) -> bool:
        accepted = super().has_progress(s)
        self.verdicts.append((s, self.solved[s], accepted))
        return accepted


def check_chain(plan: Plan) -> int:
    """Assert the plan is a simple chain of single-outcome Acts ending in Goal; its length."""
    node = plan.root
    steps = 0
    while not isinstance(node, Goal):
        assert isinstance(node, Act)
        assert len(node.branches) == 1
        node = node.branches[0].node
        steps += 1
    return steps


# --- deck_demo root plan (D14) -----------------------------------------------------------------
def test_deck_root_pretty() -> None:
    res = and_or_search(deck_model(0.2), deck_start())
    assert isinstance(res, PlanResult)
    assert res.success
    assert res.plan is not None
    assert res.plan.pretty() == DECK_ROOT


def test_deck_root_structure() -> None:
    model = deck_model(0.2)
    res = and_or_search(model, deck_start())
    assert res.plan is not None
    root = res.plan.root
    assert isinstance(root, Act)
    assert root.state == (0, 0)
    assert root.action is Action.E
    arrived, slipped, skid = root.branches
    assert arrived.state == (0, 1)
    assert arrived.labels == (ARR,)
    assert isinstance(arrived.node, Act)
    assert arrived.node.action is Action.S
    assert slipped.state == (0, 0)
    assert slipped.labels == (SLIP, SKL)
    assert slipped.node == Loop((0, 0))
    assert skid.state == (1, 0)
    assert skid.labels == (SKR,)
    assert isinstance(skid.node, Act | Ref)
    assert res.plan.to_policy()[(1, 0)] is Action.E
    check_structure(res.plan, model)
    assert is_strong_cyclic(res.plan.to_policy(), model, deck_start())


@pytest.mark.parametrize("p", [0.05, 0.1, 0.3, 0.4])
def test_deck_root_shape_other_slip_values(p: float) -> None:
    res = and_or_search(deck_model(p), deck_start())
    assert res.plan is not None
    assert res.plan.pretty() == DECK_ROOT


# --- p = 0 reduces to ordinary search ----------------------------------------------------------
def test_p0_deck_is_shortest_chain() -> None:
    model = deck_model(0.0)
    start = deck_start()
    res = and_or_search(model, start)
    assert res.plan is not None
    length = shortest_path_length(model.passable, start, model.goal)
    assert check_chain(res.plan) == length
    assert not any(isinstance(n, Loop) for n in walk_nodes(res.plan.root))
    visited = run_deterministic(res.plan.to_policy(), model, start, max_steps=100)
    assert visited[-1] == model.goal
    assert len(visited) - 1 == length


def test_p0_random_maps_are_shortest_chains() -> None:
    rng = np.random.default_rng(31)
    for _ in range(80):
        rows, cols = int(rng.integers(2, 11)), int(rng.integers(2, 11))
        passable, start, goal = random_instance(rng, rows, cols, 0.25, connected=True)
        model = ProblemModel(passable, goal, UniformSlip(0.0))
        for outcomes in (model.results(s, a) for s in model.states() for a in Action):
            assert len(outcomes) == 1
        res = and_or_search(model, start)
        assert res.plan is not None
        length = shortest_path_length(passable, start, goal)
        assert check_chain(res.plan) == length
        assert not any(isinstance(n, Loop | Ref) for n in walk_nodes(res.plan.root))
        policy = res.plan.to_policy()
        assert len(policy) == length
        visited = run_deterministic(policy, model, start, max_steps=rows * cols)
        assert visited[-1] == goal
        assert len(visited) - 1 == length


# --- wall-bump regression ----------------------------------------------------------------------
def test_wall_bump_rejected_p0() -> None:
    model, _ = ascii_model("S...\n....\n....\n...P", UniformSlip(0.0))
    start = (0, 0)
    assert bump_actions(model, start) == [Action.N, Action.W]
    plan, search = run_search(BumpFirst, model, start)
    assert plan is not None
    assert search.rollbacks >= 1
    assert isinstance(plan.root, Act)
    assert plan.root.action not in (Action.N, Action.W)
    policy = plan.to_policy()
    for s, a in policy.items():
        assert a not in bump_actions(model, s)
    assert is_strong_cyclic(policy, model, start)
    assert check_chain(plan) == shortest_path_length(model.passable, start, model.goal)
    assert search.solved == policy


@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
def test_bump_first_still_sound_with_slip(p: float) -> None:
    model, start = ascii_model("S...\n.#..\n....\n...P", UniformSlip(p))
    plan, search = run_search(BumpFirst, model, start)
    assert plan is not None
    check_structure(plan, model)
    assert is_strong_cyclic(plan.to_policy(), model, start)
    assert search.solved == plan.to_policy()


@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
def test_all_stay_bump_rejected_with_slip(p: float) -> None:
    # The alcove (2,3) is a dead end: S runs into the bottom wall and both of its skids (E, W)
    # hit debris, so even with p > 0 every outcome of S is "stay". HAS_PROGRESS must reject
    # it (the plan would loop forever) and roll back, although skids exist elsewhere.
    model, start = ascii_model(ALCOVE, UniformSlip(p))
    pocket = (1, 2)
    assert all_stay_actions(model, pocket) == [Action.S]
    assert [o.state for o in model.results(pocket, Action.S)] == [pocket]
    assert len(model.results(start, Action.E)) > 1  # skids do exist elsewhere
    plan, search = run_search(AllStayFirstRecorded, model, start)
    assert plan is not None
    assert (pocket, Action.S, False) in search.verdicts
    assert search.rollbacks >= 1
    policy = plan.to_policy()
    assert policy[pocket] is Action.N
    for s, a in policy.items():
        assert a not in all_stay_actions(model, s)
    assert is_strong_cyclic(policy, model, start)
    check_structure(plan, model)
    assert search.solved == policy


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_bump_only_actions_fail(p: float) -> None:
    # Goal in the centre of an open 3x3: bump actions only ever slide along the wall.
    model, start = ascii_model("S..\n.P.\n...", UniformSlip(p))
    plan, search = run_search(BumpOnly, model, start)
    assert plan is None
    assert search.solved == {}
    assert search.rollbacks >= 1
    # The domain itself is solvable: the failure is due to the restricted action set.
    assert strong_cyclic_policy(model, start) is not None


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_enclosed_start_fails(p: float) -> None:
    model, start = ascii_model(ENCLOSED, UniformSlip(p))
    res = and_or_search(model, start)
    assert res.plan is None
    assert not res.success
    assert res.stats.nodes_expanded <= 1
    planned = AndOrPlanner().plan(model, start)
    assert planned.plan is None
    assert not planned.stats.fallback_used


# --- semantic progress: alcove completeness ----------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
def test_alcove_is_planned(p: float) -> None:
    model, start = ascii_model(ALCOVE, UniformSlip(p))
    res = and_or_search(model, start)
    assert res.plan is not None
    plan = res.plan
    policy = plan.to_policy()
    assert is_strong_cyclic(policy, model, start)
    check_structure(plan, model)
    alcove = (1, 2)
    assert policy.get(alcove) is Action.N
    assert any(br.state == alcove for _, br in walk_branches(plan.root))
    alcove_acts = [n for n in walk_nodes(plan.root) if isinstance(n, Act) and n.state == alcove]
    assert len(alcove_acts) == 1
    # Every branch of the alcove's action loops back: syntactically "no progress".
    assert all(isinstance(br.node, Loop) for br in alcove_acts[0].branches)


# --- semantic progress: circular pending regression (docs/algorithm.md §5.3) -------------------
@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
def test_circular_pending_rejected(p: float) -> None:
    model, start = ascii_model(MUTUAL_LOOP, UniformSlip(p))
    x, y = start, (1, 0)
    # The scenario really is a mutual loop: rule R accepts it and returns a dead plan.
    bad, _ = run_search(RelaxedRule, model, start)
    assert bad is not None
    assert bad.pretty() == MUTUAL_LOOP_RULE_R
    report = validate_policy(bad.to_policy(), model, start)
    assert not report.ok
    assert report.dead == frozenset({x, y})
    # HAS_PROGRESS accepts y: N only as pending on its proper ancestor x, then rejects x: E
    # (x reaches nothing but {x, y}) and the rollback erases y: N as well.
    plan, search = run_search(Textbook, model, start)
    assert search.verdicts
    assert search.verdicts[0] == (x, Action.N, False)  # wall bump
    assert (y, Action.N, True) in search.verdicts
    assert (x, Action.E, False) in search.verdicts
    # Every textbook-order attempt at x is rejected: the search fails and rolls back fully.
    assert plan is None
    assert search.solved == {}
    assert search.rollbacks >= 1
    # The default ordering (D13) plans the corridor directly.
    res = and_or_search(model, start)
    assert res.plan is not None
    assert is_strong_cyclic(res.plan.to_policy(), model, start)


# --- soundness and rollback under adversarial orderings --------------------------------------
@pytest.mark.parametrize("cls", HOSTILE_CLASSES, ids=lambda c: c.__name__)
def test_hostile_orderings_are_sound(cls: type[AndOrSearch]) -> None:
    rng = np.random.default_rng(4242)
    returned = 0
    for _ in range(50):
        rows, cols = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        p = float(rng.choice([0.0, 0.1, 0.2, 0.4]))
        density = float(rng.uniform(0.1, 0.35))
        passable, start, goal = random_instance(rng, rows, cols, density)
        model = ProblemModel(passable, goal, UniformSlip(p))
        plan, search = run_search(cls, model, start)
        reference = strong_cyclic_policy(model, start)
        # Incompleteness under a hostile order is allowed (the planner falls back); the default
        # ordering must be complete on the same map.
        assert and_or_search(model, start).success == (reference is not None)
        if plan is None:
            assert search.solved == {}
            continue
        returned += 1
        policy = plan.to_policy()
        assert is_strong_cyclic(policy, model, start), (cls.__name__, start, goal, p)
        assert reference is not None
        check_structure(plan, model)
        # Rollback: nothing solved only during a rejected attempt survives.
        assert search.solved == policy
    assert returned > 0


def test_rollback_after_rejected_action() -> None:
    # BumpFirst at p = 0 forces a rejected attempt at every state on the path.
    model, start = ascii_model("S....\n.....\n....P", UniformSlip(0.0))
    plan, search = run_search(BumpFirst, model, start)
    assert plan is not None
    assert search.rollbacks >= 1
    assert search.solved == plan.to_policy()
    assert all(a not in bump_actions(model, s) for s, a in search.solved.items())


def test_default_search_solved_table_matches_policy() -> None:
    model = deck_model(0.2)
    search = AndOrSearch(model, PlannerConfig())
    root = search.run(deck_start())
    assert root is not None
    plan = Plan(root=root, start=deck_start(), goal=model.goal)
    assert search.solved == plan.to_policy()


# --- unreachable goal / start == goal ----------------------------------------------------------
@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_unreachable_goal(p: float) -> None:
    model, start = ascii_model(WALLED, UniformSlip(p))
    assert not reachable(model.passable, start, model.goal)
    res = and_or_search(model, start)
    assert res.plan is None
    assert res.stats.nodes_expanded <= 1
    assert res.stats.plan_size == 0
    for config in (PlannerConfig(), PlannerConfig(validate=True)):
        planned = AndOrPlanner(config).plan(model, start)
        assert planned.plan is None
        assert not planned.success
        assert planned.stats.fallback_used is False


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_start_equals_goal(p: float) -> None:
    model = deck_model(p)
    res = and_or_search(model, model.goal)
    assert res.plan is not None
    assert res.plan.root == Goal(model.goal)
    assert res.plan.to_policy() == {}
    assert res.plan.pretty() == "goal"
    assert res.stats.plan_size == 0
    planned = AndOrPlanner(PlannerConfig(validate=True)).plan(model, model.goal)
    assert planned.plan is not None
    assert isinstance(planned.plan.root, Goal)
    assert not planned.stats.fallback_used


# --- instrumentation ---------------------------------------------------------------------------
def _check_stats(res: PlanResult) -> None:
    st = res.stats
    assert isinstance(st, PlannerStats)
    assert res.plan is not None
    assert st.or_nodes >= st.nodes_expanded >= 1
    assert st.and_nodes >= 1
    assert st.rollbacks >= 0
    assert st.time_ms >= 0.0
    assert st.plan_size == len(res.plan.to_policy())
    assert st.fallback_used is False
    assert st.validated is None


def test_stats_deck_and_random() -> None:
    _check_stats(and_or_search(deck_model(0.2), deck_start()))
    rng = np.random.default_rng(8)
    for _ in range(20):
        passable, start, goal = random_instance(rng, 7, 7, 0.25, connected=True)
        _check_stats(and_or_search(ProblemModel(passable, goal, UniformSlip(0.2)), start))


# --- terrain-dependent slip --------------------------------------------------------------------
def test_terrain_slip_low_cells_single_outcome() -> None:
    rng = np.random.default_rng(17)
    checked = 0
    for _ in range(40):
        rows, cols = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        passable, start, goal = random_instance(rng, rows, cols, 0.2, connected=True)
        high = rng.random((rows, cols)) < 0.35
        model = ProblemModel(passable, goal, TerrainSlip(0.0, 0.4, high))
        res = and_or_search(model, start)
        assert res.plan is not None
        assert is_strong_cyclic(res.plan.to_policy(), model, start)
        check_structure(res.plan, model)
        for node in walk_nodes(res.plan.root):
            if isinstance(node, Act) and not high[node.state]:
                assert len(node.branches) == 1
                checked += 1
    assert checked > 0


def test_terrain_slip_on_deck() -> None:
    model = deck_model(0.0)
    high = np.zeros((model.rows, model.cols), dtype=bool)
    high[1, :] = True
    tmodel = ProblemModel(model.passable, model.goal, TerrainSlip(0.0, 0.4, high))
    res = and_or_search(tmodel, deck_start())
    assert res.plan is not None
    assert is_strong_cyclic(res.plan.to_policy(), tmodel, deck_start())
    for node in walk_nodes(res.plan.root):
        if isinstance(node, Act):
            assert (len(node.branches) == 1) == (not high[node.state])


# --- AndOrPlanner ------------------------------------------------------------------------------
def test_planner_validates_deck() -> None:
    res = AndOrPlanner(PlannerConfig(validate=True)).plan(deck_model(0.2), deck_start())
    assert res.plan is not None
    assert res.plan.pretty() == DECK_ROOT
    assert res.stats.validated is True
    assert res.stats.fallback_used is False


def test_planner_validates_random_maps() -> None:
    rng = np.random.default_rng(2025)
    planner = AndOrPlanner(PlannerConfig(validate=True))
    for _ in range(60):
        rows, cols = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        p = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]))
        passable, start, goal = random_instance(rng, rows, cols, 0.25, connected=True)
        model = ProblemModel(passable, goal, UniformSlip(p))
        res = planner.plan(model, start)
        assert res.plan is not None
        assert res.stats.validated is True
        assert res.stats.fallback_used is False
        assert is_strong_cyclic(res.plan.to_policy(), model, start)


def test_planner_without_validation_leaves_validated_none() -> None:
    res = AndOrPlanner().plan(deck_model(0.2), deck_start())
    assert res.plan is not None
    assert res.stats.validated is None
    assert res.stats.fallback_used is False


def _failed(model: ProblemModel, start: Pos, config: PlannerConfig | None = None) -> PlanResult:
    stats = PlannerStats(
        nodes_expanded=0, or_nodes=1, and_nodes=0, rollbacks=0, time_ms=0.0, plan_size=0
    )
    return PlanResult(plan=None, stats=stats)


def _bogus(model: ProblemModel, start: Pos, config: PlannerConfig | None = None) -> PlanResult:
    # A wall-bump "plan": N at the start with its single stay outcome looping (p = 0).
    root = Act(start, Action.N, (Branch(start, (ARR,), Loop(start)),))
    stats = PlannerStats(
        nodes_expanded=1, or_nodes=2, and_nodes=1, rollbacks=0, time_ms=0.0, plan_size=1
    )
    return PlanResult(plan=Plan(root=root, start=start, goal=model.goal), stats=stats)


def test_planner_falls_back_when_search_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(andor_mod, "and_or_search", _failed)
    model = deck_model(0.2)
    res = AndOrPlanner().plan(model, deck_start())
    assert res.plan is not None
    assert res.stats.fallback_used is True
    assert res.stats.validated is None
    assert res.stats.plan_size == len(res.plan.to_policy())
    assert res.stats.time_ms > 0.0  # includes the fallback, not only the (stubbed) search
    assert is_strong_cyclic(res.plan.to_policy(), model, deck_start())
    no_fallback = AndOrPlanner(PlannerConfig(fallback_to_reference=False)).plan(model, deck_start())
    assert no_fallback.plan is None
    assert no_fallback.stats.plan_size == 0
    assert no_fallback.stats.fallback_used is False


def test_planner_validates_the_fallback_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: with validate=True a fallback after a failed search used to stay unchecked
    # (validated=None).
    monkeypatch.setattr(andor_mod, "and_or_search", _failed)
    model = deck_model(0.2)
    res = AndOrPlanner(PlannerConfig(validate=True)).plan(model, deck_start())
    assert res.plan is not None
    assert res.stats.fallback_used is True
    assert res.stats.validated is True
    assert res.stats.plan_size == len(res.plan.to_policy())


def test_planner_rejects_an_invalid_fallback_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    model, start = ascii_model("S..\n...\n..P", UniformSlip(0.0))
    monkeypatch.setattr(andor_mod, "and_or_search", _failed)
    monkeypatch.setattr(andor_mod, "strong_cyclic_policy", lambda m, s: {s: Action.N})
    res = AndOrPlanner(PlannerConfig(validate=True)).plan(model, start)
    assert res.plan is None
    assert res.stats.validated is False
    assert res.stats.fallback_used is False
    assert res.stats.plan_size == 0
    # Without validation the (bogus) reference plan would have been returned.
    unchecked = AndOrPlanner().plan(model, start)
    assert unchecked.plan is not None
    assert unchecked.stats.fallback_used is True


def test_planner_falls_back_on_invalid_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(andor_mod, "and_or_search", _bogus)
    model, start = ascii_model("S..\n...\n..P", UniformSlip(0.0))
    res = AndOrPlanner(PlannerConfig(validate=True)).plan(model, start)
    assert res.plan is not None
    assert res.stats.fallback_used is True
    # False flags the rejected AND-OR plan although the returned (fallback) plan is valid.
    assert res.stats.validated is False
    assert res.stats.plan_size == len(res.plan.to_policy())
    assert is_strong_cyclic(res.plan.to_policy(), model, start)
    strict = AndOrPlanner(PlannerConfig(validate=True, fallback_to_reference=False))
    rejected = strict.plan(model, start)
    assert rejected.plan is None
    assert rejected.stats.validated is False
    assert rejected.stats.fallback_used is False
    assert rejected.stats.plan_size == 0  # regression: was the rejected plan's size


def _counting(
    monkeypatch: pytest.MonkeyPatch, name: str, calls: dict[str, int]
) -> Callable[..., Any]:
    real = getattr(andor_mod, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[name] = calls.get(name, 0) + 1
        return real(*args, **kwargs)

    monkeypatch.setattr(andor_mod, name, wrapper)
    return wrapper


def test_planner_success_runs_one_bfs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the planner recomputed the whole distance map on every call just to decide
    # whether a fallback could help; now that check only runs when a fallback is considered.
    calls: dict[str, int] = {}
    _counting(monkeypatch, "bfs_distances", calls)
    _counting(monkeypatch, "reachable", calls)
    for config in (PlannerConfig(), PlannerConfig(validate=True)):
        calls.clear()
        res = AndOrPlanner(config).plan(deck_model(0.2), deck_start())
        assert res.plan is not None
        assert calls == {"bfs_distances": 1}
    calls.clear()
    model, start = ascii_model(WALLED, UniformSlip(0.2))
    assert AndOrPlanner().plan(model, start).plan is None
    assert calls == {"bfs_distances": 1, "reachable": 1}


# --- configuration -----------------------------------------------------------------------------
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 5, 6, 7])
def test_depth_limit_p0(limit: int) -> None:
    # p = 0 deck: the shortest path has 6 steps, i.e. non-goal states at depths 0..5.
    model = deck_model(0.0)
    res = and_or_search(model, deck_start(), PlannerConfig(depth_limit=limit))
    length = shortest_path_length(model.passable, deck_start(), model.goal)
    assert length == 6
    assert res.success == (length - 1 <= limit)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_fixed_tie_break_is_valid(p: float) -> None:
    model = deck_model(p)
    res = and_or_search(model, deck_start(), PlannerConfig(tie_break="fixed"))
    assert res.plan is not None
    assert is_strong_cyclic(res.plan.to_policy(), model, deck_start())
    check_structure(res.plan, model)


def test_deterministic() -> None:
    model = deck_model(0.2)
    first = and_or_search(model, deck_start())
    again = and_or_search(deck_model(0.2), deck_start())
    assert first.plan == again.plan
    assert first.plan is not None and again.plan is not None
    assert first.plan.to_policy() == again.plan.to_policy()
    assert first.stats.nodes_expanded == again.stats.nodes_expanded


# --- recursion limit (process-wide) ----------------------------------------------------------
#: Seconds a thread waits for its partner before the test gives up (never reached when green).
THREAD_WAIT_S = 10.0


def test_recursion_headroom_restored_after_overlapping_threads() -> None:
    # Regression: each search saved and restored the global limit itself, so when two
    # overlapping searches exited in the wrong order the limit was lowered under the one still
    # running (RecursionError risk) and left raised for good afterwards.
    base = sys.getrecursionlimit()
    a_inside, b_inside, a_left = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, int] = {}
    errors: list[BaseException] = []

    def thread_a() -> None:
        try:
            with andor_mod._recursion_headroom(1_000):
                a_inside.set()
                assert b_inside.wait(THREAD_WAIT_S)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            a_left.set()

    def thread_b() -> None:
        try:
            assert a_inside.wait(THREAD_WAIT_S)
            with andor_mod._recursion_headroom(40):
                seen["inside"] = sys.getrecursionlimit()
                b_inside.set()
                assert a_left.wait(THREAD_WAIT_S)
                seen["after_a_left"] = sys.getrecursionlimit()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(2 * THREAD_WAIT_S)
    assert not errors
    assert seen["inside"] >= base + 1_000
    assert seen["after_a_left"] >= base + 40  # B is still protected
    assert sys.getrecursionlimit() == base  # and nothing leaks once both are done


def test_recursion_headroom_nests_in_one_thread() -> None:
    base = sys.getrecursionlimit()
    with andor_mod._recursion_headroom(100):
        assert sys.getrecursionlimit() >= base + 100
        with andor_mod._recursion_headroom(50):
            assert sys.getrecursionlimit() >= base + 150
        assert sys.getrecursionlimit() >= base + 100
    assert sys.getrecursionlimit() == base


def test_concurrent_deep_searches() -> None:
    mask = serpentine_mask(30)
    goal = farthest_cell(mask, (0, 0))
    base = sys.getrecursionlimit()
    n_threads = 4
    barrier = threading.Barrier(n_threads)
    plans: list[Plan | None] = [None] * n_threads
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            model = ProblemModel(mask, goal, UniformSlip(0.2))
            barrier.wait(THREAD_WAIT_S)
            plans[i] = and_or_search(model, (0, 0)).plan
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(6 * THREAD_WAIT_S)
    assert not errors
    first = plans[0]
    assert first is not None
    assert first.stats().depth > 400
    assert all(plan == first for plan in plans)
    assert sys.getrecursionlimit() == base


# --- scale -------------------------------------------------------------------------------------
MapInstance = tuple[np.ndarray, Pos, Pos]


def _rubble_30() -> MapInstance:
    rng = np.random.default_rng(30)
    return random_instance(rng, 30, 30, 0.25, connected=True)


def _serpentine_30() -> MapInstance:
    # The deepest recursion a 30x30 map allows: one long corridor (plan depth > 400).
    mask = serpentine_mask(30)
    return mask, (0, 0), farthest_cell(mask, (0, 0))


@pytest.mark.slow
@pytest.mark.parametrize("make", [_rubble_30, _serpentine_30], ids=["rubble", "serpentine"])
def test_thirty_by_thirty_under_two_seconds(make: Callable[[], MapInstance]) -> None:
    passable, start, goal = make()
    model = ProblemModel(passable, goal, UniformSlip(0.2))
    t0 = time.perf_counter()
    res = and_or_search(model, start)
    elapsed = time.perf_counter() - t0
    assert res.plan is not None
    assert elapsed < 2.0
    assert res.stats.rollbacks == 0
    assert res.stats.nodes_expanded <= int(passable.sum())
    assert is_strong_cyclic(res.plan.to_policy(), model, start)
