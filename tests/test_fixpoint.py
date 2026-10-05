"""Tests for rescue.planners.strong_cyclic_fixpoint (docs/plan.md section 3.7)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from rescue.config import PlannerConfig
from rescue.grid import Pos, TrueGrid
from rescue.model import (
    ACTIONS,
    Action,
    Outcome,
    OutcomeKind,
    ProblemModel,
    TerrainSlip,
    TransitionModel,
    UniformSlip,
    step_pos,
)
from rescue.planners.andor import AndOrPlanner, and_or_search
from rescue.planners.bfs import UNREACHABLE, bfs_distances
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy, winning_region
from rescue.planners.validate import is_strong_cyclic
from tests.helpers_phase2 import (
    ascii_model,
    bump_actions,
    deck_model,
    deck_start,
    expected_steps,
    random_instance,
)

SLIP_VALUES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
#: Map for the pit domain: the free cell (3,4) is sealed off by debris, so it cannot reach P.
PIT_MAP = "S..#\n..P#\n..#."
PIT: Pos = (2, 3)
TRAP: Pos = (0, 0)
#: The reference policy may be slower than AND-OR's by at most this factor (expected steps).
EXPECTED_STEPS_FACTOR = 1.25


def component(model: ProblemModel) -> frozenset[tuple[int, int]]:
    """Cells with a finite BFS distance to the goal (goal included)."""
    dist = bfs_distances(model.passable, model.goal)
    rows, cols = np.nonzero(dist != UNREACHABLE)
    return frozenset((int(r), int(c)) for r, c in zip(rows, cols, strict=True))


@pytest.mark.parametrize("p", SLIP_VALUES)
def test_winning_region_open_grid(p: float) -> None:
    model, _ = ascii_model("S...\n....\n...P", UniformSlip(p))
    region = winning_region(model)
    assert isinstance(region, frozenset)
    assert region == frozenset(model.states())
    assert model.goal in region


@pytest.mark.parametrize("p", SLIP_VALUES)
def test_winning_region_excludes_walled_pocket(p: float) -> None:
    model, start = ascii_model("S....\n.....\n...##\n...#.\nP..#.", UniformSlip(p))
    region = winning_region(model)
    assert (3, 4) not in region
    assert (4, 4) not in region
    assert start in region
    assert region == component(model)


@pytest.mark.parametrize("p", SLIP_VALUES)
def test_winning_region_equals_bfs_component_random(p: float) -> None:
    rng = np.random.default_rng(1234 + int(p * 100))
    for _ in range(40):
        rows, cols = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        density = float(rng.uniform(0.0, 0.5))
        passable, _start, goal = random_instance(rng, rows, cols, density, distinct=False)
        model = ProblemModel(passable, goal, UniformSlip(p))
        assert winning_region(model) == component(model)


def test_winning_region_equals_component_with_terrain() -> None:
    rng = np.random.default_rng(99)
    for _ in range(40):
        rows, cols = int(rng.integers(2, 9)), int(rng.integers(2, 9))
        passable, _start, goal = random_instance(rng, rows, cols, 0.3)
        high = rng.random((rows, cols)) < 0.4
        base_p = float(rng.choice([0.0, 0.1, 0.2]))
        model = ProblemModel(passable, goal, TerrainSlip(base_p, 0.4, high))
        assert winning_region(model) == component(model)


@pytest.mark.parametrize("p", SLIP_VALUES)
def test_strong_cyclic_policy_on_deck(p: float) -> None:
    model = deck_model(p)
    start = deck_start()
    policy = strong_cyclic_policy(model, start)
    assert policy is not None
    assert model.goal not in policy
    assert is_strong_cyclic(policy, model, start)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_strong_cyclic_policy_random_maps(p: float) -> None:
    rng = np.random.default_rng(777 + int(p * 10))
    found = 0
    for _ in range(60):
        rows, cols = int(rng.integers(2, 9)), int(rng.integers(2, 9))
        passable, start, goal = random_instance(rng, rows, cols, float(rng.uniform(0.1, 0.4)))
        model = ProblemModel(passable, goal, UniformSlip(p))
        policy = strong_cyclic_policy(model, start)
        solvable = start in component(model)
        assert (policy is not None) == solvable
        if policy is not None:
            found += 1
            assert is_strong_cyclic(policy, model, start)
    assert found > 0


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_unreachable_start_gives_none(p: float) -> None:
    model, _ = ascii_model("S....\n.....\n...##\n...#.\nP..#.", UniformSlip(p))
    assert strong_cyclic_policy(model, (4, 4)) is None
    assert strong_cyclic_policy(model, (3, 4)) is None


def test_goal_enclosed_gives_none() -> None:
    model, start = ascii_model("S.#P\n..#.", UniformSlip(0.2))
    assert strong_cyclic_policy(model, start) is None
    assert start not in winning_region(model)


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_start_equals_goal_gives_empty_policy(p: float) -> None:
    model = deck_model(p)
    assert strong_cyclic_policy(model, model.goal) == {}


# --- a domain where strong cyclic != weakly solvable ------------------------------------------
class PitModel(ProblemModel):
    """Grid model plus one extra outcome: some actions at ``trap`` may also drop into ``pit``.

    ``pit`` is a free cell walled off from the goal, so a state whose every action can fall
    into it is *weakly* solvable (a path exists) but has no strong-cyclic solution. On plain
    grid maps the two notions coincide, so only such a domain tells the fixpoint reference
    apart from BFS connectivity.
    """

    def __init__(
        self,
        passable: np.ndarray,
        goal: Pos,
        transition: TransitionModel,
        trapped: Iterable[Action],
    ) -> None:
        super().__init__(passable, goal, transition)
        self.trapped = frozenset(trapped)

    def results(self, s: Pos, a: Action) -> tuple[Outcome, ...]:
        base = super().results(s, a)
        if s == TRAP and a in self.trapped:
            return (*base, Outcome(PIT, (OutcomeKind.SLIPPED,)))
        return base


def pit_model(p: float, trapped: Iterable[Action]) -> PitModel:
    grid = TrueGrid.from_ascii(PIT_MAP)
    return PitModel(grid.passable_mask(), grid.survivor, UniformSlip(p), trapped)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.4])
def test_weak_but_not_strong_cyclic(p: float) -> None:
    model = pit_model(p, ACTIONS)
    assert all(PIT in {o.state for o in model.results(TRAP, a)} for a in ACTIONS)
    region = winning_region(model)
    # TRAP has a path to the goal, but every action risks the pit: weak, not strong cyclic.
    assert TRAP in component(model)
    assert TRAP not in region
    assert PIT not in region
    assert region < component(model)
    assert strong_cyclic_policy(model, TRAP) is None
    assert and_or_search(model, TRAP).plan is None
    planned = AndOrPlanner(PlannerConfig(validate=True)).plan(model, TRAP)
    assert planned.plan is None
    assert planned.stats.fallback_used is False
    # AND-OR <=> reference from every state of this domain, and every AND-OR plan is sound
    # (validate_policy sees the pit outcome too).
    for s in model.states():
        result = and_or_search(model, s)
        reference = strong_cyclic_policy(model, s)
        assert result.success == (reference is not None) == (s in region), s
        if result.plan is not None and reference is not None:
            assert is_strong_cyclic(result.plan.to_policy(), model, s)
            assert is_strong_cyclic(reference, model, s)
            assert TRAP not in result.plan.to_policy()
            assert TRAP not in reference


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.4])
def test_contaminated_action_is_avoided(p: float) -> None:
    # Only E (the action D13 tries first at TRAP) can fall into the pit: TRAP stays winning,
    # and both planners must pick another action there.
    model = pit_model(p, (Action.E,))
    assert TRAP in winning_region(model)
    reference = strong_cyclic_policy(model, TRAP)
    assert reference is not None
    assert reference[TRAP] is not Action.E
    assert is_strong_cyclic(reference, model, TRAP)
    result = and_or_search(model, TRAP)
    assert result.plan is not None
    assert result.plan.to_policy()[TRAP] is not Action.E
    assert result.stats.rollbacks >= 1  # E was tried and rejected
    assert is_strong_cyclic(result.plan.to_policy(), model, TRAP)


# --- RemoveNonProgress picks a useful action ---------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("n", [3, 5, 8])
def test_reference_policy_is_not_a_wall_bump(n: int, p: float) -> None:
    # Regression: the first progressing action in N,E,S,W order was kept, which in open cells
    # is usually a wall bump whose only progress is a skid (probability p/8). On an 8x8 grid
    # at p = 0.2 that policy needed 2240 expected steps (step budget 256) against 17.5.
    model = ProblemModel(np.ones((n, n), dtype=bool), (n - 1, n - 1), UniformSlip(p))
    start = (0, 0)
    reference = strong_cyclic_policy(model, start)
    assert reference is not None
    assert reference[start] not in bump_actions(model, start)
    plan = and_or_search(model, start).plan
    assert plan is not None
    ref_steps = expected_steps(reference, model, start)
    andor_steps = expected_steps(plan.to_policy(), model, start)
    assert ref_steps <= EXPECTED_STEPS_FACTOR * andor_steps


@pytest.mark.parametrize("p", [0.0, 0.2, 0.4])
def test_reference_policy_follows_the_distance_gradient(p: float) -> None:
    # On grid maps the layers are BFS distances, so the preferred action's intended move
    # always lands one step closer to the goal.
    rng = np.random.default_rng(606 + int(p * 10))
    checked = 0
    for _ in range(40):
        rows, cols = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        passable, start, goal = random_instance(rng, rows, cols, 0.25, connected=True)
        model = ProblemModel(passable, goal, UniformSlip(p))
        dist = bfs_distances(passable, goal)
        reference = strong_cyclic_policy(model, start)
        assert reference is not None
        for s, a in reference.items():
            target = step_pos(s, a)
            assert model.is_free(target)
            assert dist[target] == dist[s] - 1
            checked += 1
    assert checked > 0
