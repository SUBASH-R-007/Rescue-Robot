"""LAO* (Hansen & Zilberstein 2001): heuristic search for expected-steps optimal policies.

Stretch planner of the ``lao_expected`` agent (brief §8.1, docs/plan.md §3.15.1). It solves the
same stochastic shortest path problem as :mod:`rescue.planners.value_iteration` — unit step
costs, the goal absorbing at 0, the *true* outcome probabilities of the robot's
:class:`~rescue.model.TransitionModel` (``TerrainSlip`` aware) on the frozen belief snapshot
(``ProblemModel``: unknown assumed open)::

    V(goal) = 0
    V(s)    = min_a [ 1 + Σ_o P(o | s, a) · V(o) ]

but, like A* for deterministic problems, it only builds the part of the state space that the
best policy from the start can reach under an admissible heuristic, instead of sweeping every
free cell.

Heuristic: ``h(s)`` = BFS distance from ``s`` to the goal on the planning passability
(:func:`~rescue.planners.bfs.bfs_distances`, computed once per call). Every outcome moves the
robot at most one cell, so at least ``h(s)`` steps are needed whatever happens: ``h`` is
admissible, and consistent (``h(s) ≤ 1 + Σ P · h(s')`` because ``h(s') ≥ h(s) − 1`` for
every successor), so backed-up values only rise toward the optimum. A cell with no BFS route
to the goal has ``h = inf`` (a dead end); the start's ``h = inf`` means there is no policy.

Algorithm (the original LAO* of the paper, with all tips expanded per round):

1. Tabulate the implicit graph once with
   :func:`~rescue.planners.value_iteration.transition_tables` (vectorised; slot ``k`` =
   outcome kind ``OUTCOME_ORDER[k]``, padding → goal with probability 0) and set ``V = h``.
   The explicit graph starts as ``{start}``; expanding a state adds its outcomes under every
   action (their values stay ``h`` until backed up).
2. Repeat: (a) build the best partial solution graph — the states reachable from the start
   along the positive-probability outcomes of each expanded state's greedy action (first
   action in ``ACTIONS`` order within ``VI_TIE_TOLERANCE`` of the minimum, value iteration's
   rule; traversed with ``scipy.sparse.csgraph``); (b) if it has unexpanded non-goal tips,
   expand **all** of them; (c) value-iterate (Jacobi sweeps) on ``Z`` = every non-goal state
   of that best partial graph — the newly expanded tips and all expanded states of the graph,
   a superset of the tips' ancestors and a subset of all expanded states — until the Bellman
   residual on ``Z`` is below ``LAO_EXPANSION_TOLERANCE``; go to (a).
3. Convergence test (when the best graph has no tips): value-iterate on the states of the
   best solution graph until the residual is below ``tolerance`` (``VI_TOLERANCE``), then
   rebuild the best graph; if it gained a tip, go back to step 2; if its state set changed,
   repeat the test on the new graph; otherwise stop. Expansion rounds and convergence tests
   have separate caps (``max_iterations`` and ``LAO_MAX_CONVERGENCE_TESTS``).

At termination the best solution graph is closed (every outcome of every policy state is a
policy state or the goal), its values satisfy the Bellman equation within the tolerance, and
they are lower bounds (consistent ``h``), so they equal the optimal values on that graph: the
start value equals value iteration's optimum and the policy is proper (strong cyclic).
``LAO_EXPANSION_TOLERANCE`` only changes which states get expanded, never the result.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from typing import NamedTuple

import numpy as np

from rescue.config import (
    LAO_EXPANSION_TOLERANCE,
    LAO_MAX_CONVERGENCE_TESTS,
    LAO_MAX_ITERATIONS,
    VI_MAX_ITERS,
    VI_TIE_TOLERANCE,
    VI_TOLERANCE,
    PlannerConfig,
)
from rescue.grid import Pos
from rescue.model import ACTIONS, Action, ProblemModel
from rescue.plan import plan_from_policy
from rescue.planners.andor import MS_PER_S, PlannerStats, PlanResult
from rescue.planners.astar import STEP_COST
from rescue.planners.bfs import UNREACHABLE, bfs_distances, reachable
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy
from rescue.planners.validate import is_strong_cyclic
from rescue.planners.value_iteration import transition_tables

log = logging.getLogger(__name__)


class LaoResult(NamedTuple):
    """Result of :func:`lao_star`.

    ``values``: the value of every state of the explicit graph (expanded states and their
    successors; 0 at the goal). On the final best solution graph they are the optimal expected
    steps to the goal; elsewhere they are lower bounds (``h`` or a partial backup).
    ``policy``: the greedy action of every non-goal state of the final best solution graph
    (``{}`` when the start is the goal), or ``None`` when the start cannot reach the goal (or
    the search stopped before the best graph was closed). ``expanded_states``: states
    expanded. ``backups``: single-state Bellman backups. ``iterations``: expansion rounds
    (step 2 passes that expanded tips). ``converged``: the convergence test passed.
    """

    values: dict[Pos, float]
    policy: dict[Pos, Action] | None
    expanded_states: int
    backups: int
    iterations: int
    converged: bool


@functools.cache
def warm_up() -> bool:
    """Load (and run once on a one-state graph) the ``scipy.sparse`` routines LAO* uses.

    Importing ``scipy.sparse.csgraph`` costs about 0.1–0.5 s per process. ``_Graph`` imports it
    lazily (the browser simulator has no scipy), so without a warm-up the first plan of every
    process would count that import in its ``planning_ms``. :class:`LaoStarPlanner` calls this
    on construction, i.e. outside every timed plan; later calls are free (cached). Returns
    False, without raising, when scipy is not installed.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order
    except ImportError:
        return False
    breadth_first_order(csr_matrix((1, 1), dtype=np.int8), 0, directed=True,
                        return_predecessors=False)
    return True


class _Graph:
    """The explicit graph over the indexed free states of the model.

    The implicit graph (every state's outcomes and their probabilities) is tabulated up front
    by the vectorised :func:`~rescue.planners.value_iteration.transition_tables` (one numpy
    pass, about the cost of a single value-iteration sweep); the search reads the rows of the
    states it expands and backs up only those states.
    """

    def __init__(self, model: ProblemModel, h: np.ndarray) -> None:
        """Graph over ``model`` with heuristic ``h`` (per cell); nothing expanded yet."""
        from scipy.sparse import csr_matrix  # lazy: the browser simulator has no scipy
        from scipy.sparse.csgraph import breadth_first_order

        self._csr_matrix = csr_matrix
        self._bfs = breadth_first_order
        tables = transition_tables(model)
        self.states = tables.states
        self.goal = tables.goal_index
        self.index = {s: i for i, s in enumerate(tables.states)}
        n = len(tables.states)
        coords = np.array(tables.states, dtype=np.intp).reshape(n, 2)
        self.values = h[coords[:, 0], coords[:, 1]].astype(float)
        self.values[self.goal] = 0.0
        self.succ = tables.succ
        self.prob = tables.prob
        self.expanded = np.zeros(n, dtype=bool)
        self.discovered = np.zeros(n, dtype=bool)
        self.best = np.zeros(n, dtype=np.intp)

    def expand(self, ids: np.ndarray) -> None:
        """Add the outcomes of the states ``ids`` (all actions) to the explicit graph."""
        self.expanded[ids] = True
        self.discovered[ids] = True
        succ = self.succ[:, ids, :]
        self.discovered[succ[self.prob[:, ids, :] > 0.0]] = True

    def value_iterate(self, z: np.ndarray, tolerance: float) -> tuple[int, bool]:
        """Jacobi Bellman sweeps over the states ``z`` until the largest value change is below
        ``tolerance`` (at most ``VI_MAX_ITERS``): (backups, converged).

        Greedy action: the first in ``ACTIONS`` order within ``VI_TIE_TOLERANCE`` of the
        minimum (value iteration's rule). Padding slots point at the goal (``V = 0``) with
        probability 0, so ``P · V`` is never ``0 · inf``; callers run under
        ``np.errstate(invalid="ignore")`` for the ``inf − inf`` change of a dead end.
        """
        if not len(z):
            return 0, True
        prob = self.prob[:, z, :]
        succ = self.succ[:, z, :]
        values = self.values
        for sweep in range(1, VI_MAX_ITERS + 1):
            q = STEP_COST + (prob * values[succ]).sum(axis=2)
            best = q.min(axis=0)
            old = values[z]
            change = np.abs(best - old)
            change[old == best] = 0.0
            values[z] = best
            if change.max() < tolerance:
                self.best[z] = (q <= best + VI_TIE_TOLERANCE).argmax(axis=0)
                return sweep * len(z), True
        self.best[z] = (q <= best + VI_TIE_TOLERANCE).argmax(axis=0)
        return VI_MAX_ITERS * len(z), False

    def best_graph(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        """Best partial solution graph from ``start``: (its expanded non-goal states, its
        unexpanded non-goal tips), in breadth-first order.

        Edges: every outcome with positive probability of each expanded state's greedy
        action; the traversal runs in compiled code (``scipy.sparse.csgraph``).
        """
        ids = np.flatnonzero(self.expanded)
        a = self.best[ids]
        possible = (self.prob[a, ids, :] > 0.0).ravel()
        rows = np.repeat(ids, self.succ.shape[2])[possible]
        cols = self.succ[a, ids, :].ravel()[possible]
        n = len(self.states)
        adjacency = self._csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        order = self._bfs(adjacency, start, directed=True, return_predecessors=False)
        order = order[order != self.goal]
        inside = self.expanded[order]
        return order[inside], order[~inside]


def heuristic(model: ProblemModel) -> np.ndarray:
    """``h`` of every cell: BFS steps to the goal on ``model.passable`` (``inf`` if none)."""
    dist = bfs_distances(model.passable, model.goal)
    return np.where(dist == UNREACHABLE, math.inf, dist.astype(float))


def _search(
    graph: _Graph, root: int, tolerance: float, max_iterations: int, expansion_tolerance: float
) -> tuple[int, int, bool]:
    """The LAO* main loop (steps 2–3 of the module docstring): (backups, rounds, converged).

    Every pass is either an expansion round (capped by ``max_iterations``) or a convergence
    test (capped by ``LAO_MAX_CONVERGENCE_TESTS``), so the loop always ends.
    """
    backups = 0
    iterations = 0
    tests = 0
    while True:
        interior, tips = graph.best_graph(root)
        if len(tips):
            if iterations >= max_iterations:
                log.warning("LAO* stopped after %d expansion rounds", max_iterations)
                return backups, iterations, False
            iterations += 1
            graph.expand(tips)
            z = np.concatenate([interior, tips])
            done, _ = graph.value_iterate(z, expansion_tolerance)
            backups += done
            continue
        # Convergence test: value-iterate on the closed best solution graph, then check that
        # the greedy actions still define the same graph (else continue with the new one).
        if tests >= LAO_MAX_CONVERGENCE_TESTS:
            log.warning("LAO* stopped after %d convergence tests", LAO_MAX_CONVERGENCE_TESTS)
            return backups, iterations, False
        tests += 1
        done, settled = graph.value_iterate(interior, tolerance)
        backups += done
        after, new_tips = graph.best_graph(root)
        if not len(new_tips) and np.array_equal(np.sort(after), np.sort(interior)):
            return backups, iterations, settled


def lao_star(
    model: ProblemModel,
    start: Pos,
    tolerance: float = VI_TOLERANCE,
    max_iterations: int = LAO_MAX_ITERATIONS,
    expansion_tolerance: float = LAO_EXPANSION_TOLERANCE,
) -> LaoResult:
    """Expected-steps optimal policy from ``start`` on ``model`` (see the module docstring).

    ``tolerance``: Bellman residual of the convergence test (default ``VI_TOLERANCE``, as
    value iteration). ``expansion_tolerance``: residual of the value iteration on ``Z`` after
    each expansion round. ``max_iterations``: cap on the expansion rounds; the result then has
    ``converged`` False (and the policy of the best graph if it happens to be closed, else
    ``None``). Both tolerances must be positive (``math.inf`` is allowed for
    ``expansion_tolerance``: one sweep per round) and ``max_iterations`` non-negative, else
    ``ValueError``: every sweep loop stops on a residual *strictly* below its tolerance, so a
    zero tolerance would run to ``VI_MAX_ITERS`` sweeps.
    """
    if not tolerance > 0.0 or math.isinf(tolerance):
        raise ValueError(f"tolerance must be positive and finite, got {tolerance}")
    if not expansion_tolerance > 0.0:
        raise ValueError(f"expansion_tolerance must be positive, got {expansion_tolerance}")
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    goal = model.goal
    if start == goal:
        return LaoResult({goal: 0.0}, {}, 0, 0, 0, True)
    h = heuristic(model)
    if not model.is_free(start) or math.isinf(h[start]):
        return LaoResult({}, None, 0, 0, 0, True)
    graph = _Graph(model, h)
    root = graph.index[start]
    graph.discovered[root] = True
    with np.errstate(invalid="ignore"):  # inf − inf residuals of dead ends (never chosen)
        backups, iterations, converged = _search(
            graph, root, tolerance, max_iterations, expansion_tolerance
        )
    if not converged:
        log.warning("LAO* did not converge from %s", start)
    graph.discovered[graph.goal] = True
    values = {graph.states[i]: float(graph.values[i]) for i in np.flatnonzero(graph.discovered)}
    interior, tips = graph.best_graph(root)
    policy = (
        None if len(tips)
        else {graph.states[i]: ACTIONS[int(graph.best[i])] for i in sorted(interior.tolist())}
    )
    return LaoResult(values, policy, int(np.count_nonzero(graph.expanded)), backups,
                     iterations, converged)


class LaoStarPlanner:
    """Planner interface around :func:`lao_star` (mirrors ``ValueIterationPlanner.plan``).

    Statistics: ``nodes_expanded`` is the number of states LAO* expanded; ``plan_size`` the
    number of policy states reachable from the start. With ``config.validate`` the policy is
    checked with :func:`is_strong_cyclic` and, if it failed and
    ``config.fallback_to_reference`` is set, the fixpoint reference policy is used instead
    (``fallback_used``). ``last_values`` / ``last_result`` keep the most recent call's values
    (expected steps to the goal on the solution graph) and full result. Construction calls
    :func:`warm_up`, so no plan's ``time_ms`` includes loading ``scipy.sparse.csgraph``.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        warm_up()
        self.config = config or PlannerConfig()
        self.last_values: dict[Pos, float] = {}
        self.last_result: LaoResult | None = None

    def plan(self, model: ProblemModel, start: Pos) -> PlanResult:
        """Plan from ``start`` on ``model``."""
        t0 = time.perf_counter()
        result = lao_star(model, start)
        self.last_result = result
        self.last_values = result.values
        validated: bool | None = None
        fallback_used = False
        chosen = result.policy
        if chosen is not None and self.config.validate:
            validated = is_strong_cyclic(chosen, model, start)
            if not validated:
                log.error("LAO* policy from %s is not strong-cyclic", start)
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
            nodes_expanded=result.expanded_states,
            or_nodes=0,
            and_nodes=0,
            rollbacks=0,
            time_ms=(time.perf_counter() - t0) * MS_PER_S,
            plan_size=plan_size,
            fallback_used=fallback_used,
            validated=validated,
        )
        return PlanResult(plan=plan, stats=stats)
