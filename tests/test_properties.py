"""Property tests (hypothesis): AND-OR vs the fixpoint reference planner (brief §9, Phase 2).

Over random maps up to 8x8 (start != goal; start == goal has deterministic tests elsewhere),
random debris density and random slip (uniform or terrain-dependent):

(a) AND-OR finds a plan  <=>  the strong-cyclic fixpoint planner does;
(b) every AND-OR plan passes ``is_strong_cyclic``;
(c) the plan tree is well formed: every Act's branches are exactly ``Results(s, a)`` with
    merged labels, every Ref/Loop target has an action, ``to_policy()`` keys are exactly the
    Act states, and ``pretty``/``to_tree``/``stats`` agree;
(d) ``AndOrPlanner(PlannerConfig(validate=True))`` never needs the fallback;
(e) with the default ordering (D13) the search is backtrack-free (no rollbacks) and expands
    each free cell at most once, so a blow-up fails fast instead of hanging;
(f) the reference policy, unfolded by ``plan_from_policy`` (the production fallback path), is
    strong cyclic and keeps exactly the reference's actions.

A second property runs AND-OR under adversarial action orderings, where the search *does*
backtrack: HAS_PROGRESS, the rollback journal and dead-state pruning are exercised there, and
every plan returned must still be sound (completeness is not required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from rescue.config import MAX_SLIP_P, PlannerConfig
from rescue.grid import Pos
from rescue.model import Action, ProblemModel, TerrainSlip, TransitionModel, UniformSlip
from rescue.plan import Plan, plan_from_policy
from rescue.planners.andor import AndOrPlanner, AndOrSearch, and_or_search
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy
from rescue.planners.validate import is_strong_cyclic
from tests.helpers_phase2 import HOSTILE_CLASSES, check_structure, run_search

MAX_SIDE = 8
#: Cell pools for the random mask, one per debris density (1/4, 1/3, 2/5, 1/2 debris).
CELL_POOLS: tuple[tuple[bool, ...], ...] = (
    (True, True, True, False),
    (True, True, False),
    (True, True, True, False, False),
    (True, False),
)
N_EXAMPLES = 500
#: Hostile orderings backtrack, so each example costs more: fewer examples.
N_HOSTILE_EXAMPLES = 300
#: Correct code needs under 1,000 expansions per hostile search on these maps; a regression
#: that makes the search blow up fails at this budget instead of hanging the suite.
HOSTILE_EXPANSION_CAP = 20_000


@dataclass(frozen=True)
class Instance:
    """One random planning problem."""

    passable: np.ndarray
    start: Pos
    goal: Pos
    transition: TransitionModel

    def model(self) -> ProblemModel:
        return ProblemModel(self.passable, self.goal, self.transition)


slip = st.floats(min_value=0.0, max_value=MAX_SLIP_P, allow_nan=False, allow_infinity=False)


@st.composite
def instances(draw: st.DrawFn, max_side: int = MAX_SIDE) -> Instance:
    """Random map with at least two cells and distinct, free start and goal."""
    # Mix in a strategy biased towards the largest sizes, so big maps are common.
    side = st.integers(1, max_side) | st.integers(max(1, max_side - 3), max_side)
    rows = draw(side)
    cols = max(draw(side), 2 if rows == 1 else 1)
    n = rows * cols
    pool = draw(st.sampled_from(CELL_POOLS))
    cells = draw(st.lists(st.sampled_from(pool), min_size=n, max_size=n))
    passable = np.array(cells, dtype=bool).reshape(rows, cols)
    start_i = draw(st.integers(0, n - 1))
    goal_i = draw(st.integers(0, n - 2))
    goal_i += goal_i >= start_i  # uniform over the other n - 1 cells
    start = divmod(start_i, cols)
    goal = divmod(goal_i, cols)
    passable[start] = True
    passable[goal] = True
    transition: TransitionModel
    if draw(st.booleans()):
        transition = UniformSlip(draw(slip))
    else:
        high = draw(st.lists(st.booleans(), min_size=n, max_size=n))
        mask = np.array(high, dtype=bool).reshape(rows, cols)
        transition = TerrainSlip(draw(slip), draw(slip), mask)
    return Instance(passable, start, goal, transition)


def _mask(text: str) -> np.ndarray:
    return np.array([[ch != "#" for ch in line] for line in text.split()], dtype=bool)


#: 8x8 rubble with a winding route from (0,0) to (7,7).
OPEN_8X8 = _mask(
    "..#....# ..#.##.. ....#... ##.#..#. ...#.#.. .#...#.# .#.#.... ...#.#.."
)
#: Same size, but the goal corner is sealed off by debris.
SEALED_8X8 = _mask(
    "........ ........ ........ ........ ........ ........ ......## ......#."
)


def _capped(cls: type[AndOrSearch]) -> type[AndOrSearch]:
    """``cls`` with an expansion budget: exceeding it fails the test immediately."""

    class Capped(cls):  # type: ignore[valid-type,misc]
        def ordered_actions(self, s: Pos) -> list[Action]:
            assert self.expanded <= HOSTILE_EXPANSION_CAP, f"{cls.__name__} search blew up"
            return super().ordered_actions(s)

    Capped.__name__ = Capped.__qualname__ = cls.__name__
    return Capped


CAPPED_HOSTILE_CLASSES = tuple(_capped(cls) for cls in HOSTILE_CLASSES)


def _tree_size(tree: dict[str, Any]) -> int:
    count, stack = 0, [tree]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node["children"])
    return count


@settings(
    max_examples=N_EXAMPLES,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instances())
@example(Instance(OPEN_8X8, (0, 0), (7, 7), UniformSlip(0.3)))
@example(Instance(SEALED_8X8, (0, 0), (7, 7), UniformSlip(0.2)))
def test_andor_matches_reference_and_is_sound(inst: Instance) -> None:
    model = inst.model()
    result = and_or_search(model, inst.start)
    reference = strong_cyclic_policy(model, inst.start)
    rows, cols = inst.passable.shape
    event(f"{'solvable' if reference is not None else 'unsolvable'}")
    event(f"size {'>= 5x5' if min(rows, cols) >= 5 else 'small'}")
    event(f"8x8: {(rows, cols) == (MAX_SIDE, MAX_SIDE)}")

    # (a) completeness both ways
    assert result.success == (reference is not None)
    # (e) backtrack-free under D13, and no state is expanded twice
    assert result.stats.rollbacks == 0
    assert result.stats.nodes_expanded <= int(inst.passable.sum())
    if result.plan is None:
        assert result.stats.plan_size == 0
        # (d) nothing for the fallback to find either
        planned = AndOrPlanner(PlannerConfig(validate=True)).plan(model, inst.start)
        assert planned.plan is None
        assert planned.stats.fallback_used is False
        assert planned.stats.plan_size == 0
        return
    assert reference is not None
    plan = result.plan
    policy = plan.to_policy()

    # (b) soundness
    assert is_strong_cyclic(policy, model, inst.start)

    # (c) tree structure and the views on it
    check_structure(plan, model)
    assert result.stats.plan_size == len(policy)
    stats = plan.stats()
    assert stats.act_nodes == len(policy)
    assert isinstance(plan.pretty(), str)
    assert _tree_size(plan.to_tree(stats.depth)) == stats.nodes
    assert plan.envelope() >= set(policy) | {inst.start, model.goal}

    # (d) the validated planner never needs the reference fallback
    planned = AndOrPlanner(PlannerConfig(validate=True)).plan(model, inst.start)
    assert planned.plan == plan
    assert planned.stats.fallback_used is False
    assert planned.stats.validated is True

    # (f) the fallback path: unfolding the reference policy
    unfolded = plan_from_policy(reference, model, inst.start)
    assert isinstance(unfolded, Plan)
    assert unfolded.to_policy() == reference
    assert is_strong_cyclic(reference, model, inst.start)
    check_structure(unfolded, model)


@settings(
    max_examples=N_HOSTILE_EXAMPLES,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(instances())
def test_hostile_orderings_stay_sound(inst: Instance) -> None:
    model = inst.model()
    solvable = strong_cyclic_policy(model, inst.start) is not None
    for cls in CAPPED_HOSTILE_CLASSES:
        plan, search = run_search(cls, model, inst.start)
        if search.rollbacks:
            event("hostile search backtracked")
        if plan is None:
            # Every rejected attempt was rolled back, down to the root.
            assert search.solved == {}
            continue
        policy = plan.to_policy()
        assert solvable, cls.__name__
        assert is_strong_cyclic(policy, model, inst.start), cls.__name__
        check_structure(plan, model)
        # Rollback: nothing solved only during a rejected attempt survives.
        assert search.solved == policy, cls.__name__
