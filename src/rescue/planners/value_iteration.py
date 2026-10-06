"""Expected-cost optimal policies by value iteration (stretch planner ``vi_expected``).

AND-OR search finds *a* strong-cyclic plan; it ignores how likely each outcome is. Value
iteration instead minimises the expected number of steps to the survivor, using the *true*
outcome probabilities of the robot's :class:`~rescue.model.TransitionModel` (``TerrainSlip``
aware: the slip parameter of the cell the robot stands on, D4) on the same frozen belief
snapshot the AND-OR planner sees (``ProblemModel``: unknown assumed open). It is a stochastic
shortest path problem with unit step costs::

    V(goal) = 0
    V(s)    = min_a [ 1 + Σ_o P(o | s, a) · V(o) ]      o ∈ ProblemModel.results(s, a)

where the probability of a merged outcome is the sum over its labels (e.g. a skid into a wall
merged with "slipped"). Implementation:

1. Index the free states (row-major) and build one compact table per action with numpy index
   arithmetic: ``succ[a, s, k]`` (state index where outcome kind ``OUTCOME_ORDER[k]`` lands)
   and ``prob[a, s, k]`` (its probability); kinds outside the support point at the goal with
   probability 0. Kinds landing on the same cell keep separate slots, which gives the same
   backups as one merged outcome with the summed probability.
2. Proper region: the states that can reach the goal with probability 1 under some policy
   (the strong-cyclic winning region), computed as a vectorised fixpoint — repeatedly keep the
   state-action pairs whose outcomes all stay in the region, then keep the states from which
   such pairs can reach the goal. Every other state is unreachable and gets ``inf``.
3. Iterate the Bellman backup on the region, starting from ``V = 0`` (a lower bound, so the
   values rise monotonically to the optimum), until the largest change is below
   ``config.VI_TOLERANCE`` or ``config.VI_MAX_ITERS`` sweeps have run.
4. Greedy policy: at every non-goal state of the region the first action in ``ACTIONS`` order
   whose Q-value is within ``config.VI_TIE_TOLERANCE`` of the minimum. With unit costs a
   greedy policy for the optimal values is proper, hence strong cyclic.

:class:`ValueIterationPlanner` wraps this in the :class:`~rescue.planners.andor.AndOrPlanner`
interface (``plan(model, start) -> PlanResult``) so agents and the UI can treat it like the
AND-OR planner: the plan is the policy unfolded from the start with
:func:`~rescue.plan.plan_from_policy`.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import numpy as np

from rescue.config import (
    PROB_EPS,
    VI_MAX_ITERS,
    VI_TIE_TOLERANCE,
    VI_TOLERANCE,
    PlannerConfig,
)
from rescue.grid import Pos
from rescue.model import ACTIONS, OUTCOME_ORDER, Action, ProblemModel, kind_direction
from rescue.plan import plan_from_policy
from rescue.planners.andor import MS_PER_S, PlannerStats, PlanResult
from rescue.planners.astar import STEP_COST
from rescue.planners.bfs import reachable
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy
from rescue.planners.validate import is_strong_cyclic

log = logging.getLogger(__name__)

#: Most distinct outcomes one action can have (one per outcome kind).
MAX_OUTCOMES: int = len(OUTCOME_ORDER)


class ValueIterationResult(NamedTuple):
    """``(values, policy, iterations)`` of :func:`value_iteration`.

    ``values``: expected steps to the goal from every free state under the optimal policy
    (``inf`` outside the proper region, 0 at the goal). ``policy``: the greedy action of every
    non-goal state in the proper region. ``iterations``: Bellman sweeps performed.
    """

    values: dict[Pos, float]
    policy: dict[Pos, Action]
    iterations: int


class TransitionTables(NamedTuple):
    """Compact per-action transition tables over the indexed free states."""

    states: list[Pos]
    goal_index: int
    #: ``(len(ACTIONS), n, MAX_OUTCOMES)`` int: successor state indices (padding → goal).
    succ: np.ndarray
    #: ``(len(ACTIONS), n, MAX_OUTCOMES)`` float: outcome probabilities (padding → 0).
    prob: np.ndarray


def transition_tables(model: ProblemModel) -> TransitionTables:
    """Index the free states of ``model`` and tabulate ``P(o | s, a)`` per action.

    Vectorised over the states: slot ``k`` of ``succ[a, s]`` / ``prob[a, s]`` is outcome kind
    ``OUTCOME_ORDER[k]`` with the probability ``model.transition.distribution(s)`` gives it; a
    blocked or off-grid target collapses to ``s`` and a kind with probability at most
    ``PROB_EPS`` (outside the support) is padding (goal, probability 0), exactly as
    ``ProblemModel.results``. Kinds that land on the same cell (a merged outcome there) keep
    separate slots: ``Σ_k P_k · V(succ_k)`` and the support test are the same as with their
    probabilities summed into one slot. Raises ``ValueError`` if the goal is not a free state.
    """
    states = model.states()
    n = len(states)
    index = np.full((model.rows, model.cols), -1, dtype=np.intp)
    coords = np.array(states, dtype=np.intp).reshape(n, 2)
    index[coords[:, 0], coords[:, 1]] = np.arange(n)
    gr, gc = model.goal
    if not (0 <= gr < model.rows and 0 <= gc < model.cols) or index[gr, gc] < 0:
        raise ValueError(f"goal {model.goal} is not a free state of the model")
    goal_index = int(index[gr, gc])
    kind_prob = np.array(
        [[p for _, p in model.transition.distribution(s)] for s in states], dtype=float
    ).reshape(n, MAX_OUTCOMES)
    kind_prob = np.where(kind_prob > PROB_EPS, kind_prob, 0.0)
    possible = kind_prob > 0.0
    own = np.arange(n)
    succ = np.empty((len(ACTIONS), n, MAX_OUTCOMES), dtype=np.intp)
    prob = np.empty((len(ACTIONS), n, MAX_OUTCOMES), dtype=float)
    for ai, a in enumerate(ACTIONS):
        for k, kind in enumerate(OUTCOME_ORDER):
            direction = kind_direction(kind, a)
            if direction is None:
                target = own
            else:
                dr, dc = direction.value
                r, c = coords[:, 0] + dr, coords[:, 1] + dc
                inside = (r >= 0) & (r < model.rows) & (c >= 0) & (c < model.cols)
                hit = np.full(n, -1, dtype=np.intp)
                hit[inside] = index[r[inside], c[inside]]
                target = np.where(hit >= 0, hit, own)
            succ[ai, :, k] = np.where(possible[:, k], target, goal_index)
            prob[ai, :, k] = kind_prob[:, k]
    return TransitionTables(states, goal_index, succ, prob)


def proper_region(tables: TransitionTables) -> np.ndarray:
    """Bool mask of the states with a policy reaching the goal with probability 1.

    Greatest fixpoint: keep only state-action pairs whose possible outcomes all stay in the
    region, then only states from which those pairs can reach the goal; repeat until stable.
    """
    succ, prob, goal = tables.succ, tables.prob, tables.goal_index
    possible = prob > PROB_EPS
    region = np.ones(len(tables.states), dtype=bool)
    while True:
        # (A, n): pairs whose every possible outcome stays in the region.
        closed = np.all(region[succ] | ~possible, axis=2) & region[np.newaxis, :]
        reach = np.zeros_like(region)
        reach[goal] = True
        while True:
            hits = np.any(reach[succ] & possible, axis=2) & closed
            grown = reach | np.any(hits, axis=0)
            if np.array_equal(grown, reach):
                break
            reach = grown
        if np.array_equal(reach, region):
            return region
        region = reach


def _q_values(values: np.ndarray, tables: TransitionTables) -> np.ndarray:
    """``(A, n)`` Q-values ``1 + Σ_k P · V(succ)`` (``inf`` when an outcome has ``V = inf``)."""
    with np.errstate(invalid="ignore"):
        expected = np.where(tables.prob > 0.0, tables.prob * values[tables.succ], 0.0)
    return STEP_COST + expected.sum(axis=2)


def value_iteration(model: ProblemModel) -> ValueIterationResult:
    """Expected-cost optimal values and greedy policy on ``model`` (see the module docstring).

    Ties between actions are broken by ``ACTIONS`` order (within ``VI_TIE_TOLERANCE``).
    """
    tables = transition_tables(model)
    states, goal = tables.states, tables.goal_index
    region = proper_region(tables)
    values = np.where(region, 0.0, np.inf)
    iterations = 0
    converged = False
    while iterations < VI_MAX_ITERS:
        iterations += 1
        new = _q_values(values, tables).min(axis=0)
        new[goal] = 0.0
        new[~region] = np.inf
        delta = float(np.max(np.abs(new[region] - values[region]))) if region.any() else 0.0
        values = new
        if delta < VI_TOLERANCE:
            converged = True
            break
    if not converged:
        log.warning("value iteration did not converge in %d sweeps", VI_MAX_ITERS)

    q = _q_values(values, tables)
    best = q.min(axis=0)
    # First action (ACTIONS order) within the tie tolerance of the minimum.
    choice = np.argmax(q <= best[np.newaxis, :] + VI_TIE_TOLERANCE, axis=0)
    policy: dict[Pos, Action] = {}
    for i in np.flatnonzero(region):
        if i != goal:
            policy[states[i]] = ACTIONS[int(choice[i])]
    value_map = {s: float(values[i]) for i, s in enumerate(states)}
    return ValueIterationResult(value_map, policy, iterations)


class ValueIterationPlanner:
    """Planner interface around :func:`value_iteration` (mirrors ``AndOrPlanner.plan``).

    Statistics: ``nodes_expanded`` is the number of Bellman backups (sweeps × proper-region
    size); ``plan_size`` the number of states with an action that are reachable from the start
    under the policy. With ``config.validate`` the policy is checked with
    :func:`is_strong_cyclic` and, if it failed and ``config.fallback_to_reference`` is set, the
    fixpoint reference policy is used instead (``fallback_used``). ``last_values`` keeps the
    value map of the most recent call (expected steps to the goal).
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.last_values: dict[Pos, float] = {}
        self.last_iterations: int = 0

    def plan(self, model: ProblemModel, start: Pos) -> PlanResult:
        """Plan from ``start`` on ``model``."""
        t0 = time.perf_counter()
        values, policy, iterations = value_iteration(model)
        self.last_values = values
        self.last_iterations = iterations
        backups = iterations * sum(1 for v in values.values() if v != np.inf)
        validated: bool | None = None
        fallback_used = False
        chosen: dict[Pos, Action] | None = None
        if start == model.goal or values.get(start, np.inf) != np.inf:
            chosen = policy
            if self.config.validate:
                validated = is_strong_cyclic(policy, model, start)
                if not validated:
                    log.error("value-iteration policy from %s is not strong-cyclic", start)
                    chosen = None
        if (
            chosen is None
            and validated is False
            and self.config.fallback_to_reference
            and reachable(model.passable, start, model.goal)
        ):
            reference = strong_cyclic_policy(model, start)
            if reference is not None and is_strong_cyclic(reference, model, start):
                chosen = reference
                fallback_used = True
        plan = plan_from_policy(chosen, model, start) if chosen is not None else None
        plan_size = len(plan.to_policy()) if plan is not None else 0
        stats = PlannerStats(
            nodes_expanded=backups,
            or_nodes=0,
            and_nodes=0,
            rollbacks=0,
            time_ms=(time.perf_counter() - t0) * MS_PER_S,
            plan_size=plan_size,
            fallback_used=fallback_used,
            validated=validated,
        )
        return PlanResult(plan=plan, stats=stats)
