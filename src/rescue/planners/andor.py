"""AND-OR graph search for strong-cyclic contingency plans.

This is AIMA Fig. 4.11 (AND-OR-GRAPH-SEARCH) with the cyclic extension of AIMA §4.3.3: an
outcome that repeats a state on the current path becomes a ``Loop`` label ("retry") instead
of a failure, and a global ``solved`` table turns the tree search into a graph search
(``Ref`` nodes point at subplans solved elsewhere).

Two refinements make the search sound (see ``docs/algorithm.md`` for the argument):

* **Transactional solved table.** Every entry added while trying an action at a state is
  journaled and rolled back if that action is rejected. Without this, a state solved with a
  ``Loop`` back to an ancestor that later fails would stay cached and could be reused through
  a ``Ref`` — a plan that points at a state with no action.
* **Semantic progress test** (``HAS_PROGRESS``). An action is accepted at ``s`` only if, in the
  provisional policy graph (solved table plus ``s``), ``s`` can reach the goal, a state already
  proven to reach the goal ("grounded"), or a *proper* ancestor on the current path (whose own
  test, when it completes, closes the argument). This is AIMA's requirement that "a leaf is
  reachable from every point in the plan", checked incrementally. It rejects the wall-bump
  "plan" (an action whose only outcome is staying put) and pairs of states that only loop
  into each other, while still accepting a dead-end alcove whose only exit is back to its
  parent. The brief's purely syntactic rule ("at least one branch is not a Loop") is sound
  with a transactional table but incomplete: it rejects the alcove. Letting a Loop to an
  ancestor count as progress accepts the alcove but becomes unsound (the mutual loop). See
  docs/algorithm.md §5.

Action ordering (``ORDERED_ACTIONS``) is by BFS distance-to-goal of the intended target on the
belief map, then a diagonal tie-break, then the fixed order N, E, S, W (docs/plan.md D13).
With this ordering the first action tried at every state has an "arrived" outcome strictly
closer to the goal, which makes the search backtrack-free and complete on grid maps; any
other failure is caught by the fixpoint reference planner in :class:`AndOrPlanner`.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace

from rescue.config import PlannerConfig
from rescue.grid import Pos
from rescue.model import ACTIONS, Action, Outcome, ProblemModel, step_pos
from rescue.plan import Act, Branch, Goal, Loop, Plan, PlanNode, Ref, plan_from_policy
from rescue.planners.bfs import UNREACHABLE, bfs_distances, reachable
from rescue.planners.strong_cyclic_fixpoint import strong_cyclic_policy
from rescue.planners.validate import is_strong_cyclic

log = logging.getLogger(__name__)

#: Python frames used per OR level of the recursion (OR_SEARCH + AND_SEARCH, with headroom).
FRAMES_PER_LEVEL = 3
#: Extra recursion headroom for the caller's own stack.
RECURSION_SLACK = 1_000
#: Milliseconds per second, for timing.
MS_PER_S = 1_000.0


@dataclass(frozen=True)
class PlannerStats:
    """Instrumentation for one planning call.

    ``time_ms`` is the wall time of the whole call (for :meth:`AndOrPlanner.plan`: search,
    validation and fallback). ``plan_size`` is the number of states with an action in the
    returned plan (0 when no plan is returned). ``fallback_used`` is True when the returned
    plan came from the fixpoint reference planner. ``validated`` is None when validation was
    off or no plan was produced; otherwise True if every plan produced in the call passed
    ``is_strong_cyclic`` and False if one was rejected (a soundness bug worth reporting even
    when the fallback then succeeded). With validation on, an invalid plan is never returned.
    """

    nodes_expanded: int
    or_nodes: int
    and_nodes: int
    rollbacks: int
    time_ms: float
    plan_size: int
    fallback_used: bool = False
    validated: bool | None = None


@dataclass(frozen=True)
class PlanResult:
    """Outcome of a planning call: a plan (``None`` means FAIL) plus statistics."""

    plan: Plan | None
    stats: PlannerStats

    @property
    def success(self) -> bool:
        """True when a plan was found."""
        return self.plan is not None


class _RecursionHeadroom:
    """Reference-counted, thread-safe raise of the process-wide recursion limit.

    ``sys.setrecursionlimit`` is global, so overlapping searches in several threads must not
    each save and restore it (the first to finish would lower the limit under the others).
    The first entrant records the original limit (the *base*); every entrant raises the limit
    to at least ``base + frames`` plus whatever its own thread has already reserved (nested
    use); the limit is restored to the base only when the last entrant leaves. While searches
    overlap, the limit only grows, bounded by ``base`` plus the largest per-thread reservation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._base = 0
        self._local = threading.local()

    @contextmanager
    def reserve(self, frames: int) -> Iterator[None]:
        """Guarantee ``frames`` extra Python frames for the duration of the block."""
        reserved: int = getattr(self._local, "frames", 0)
        with self._lock:
            if self._active == 0:
                self._base = sys.getrecursionlimit()
            self._active += 1
            wanted = self._base + reserved + frames
            if wanted > sys.getrecursionlimit():
                sys.setrecursionlimit(wanted)
        self._local.frames = reserved + frames
        try:
            yield
        finally:
            self._local.frames = reserved
            with self._lock:
                self._active -= 1
                if self._active == 0:
                    sys.setrecursionlimit(self._base)


_HEADROOM = _RecursionHeadroom()


def _recursion_headroom(frames: int) -> AbstractContextManager[None]:
    """Context manager: make room for ``frames`` more Python frames (thread-safe)."""
    return _HEADROOM.reserve(frames)


class AndOrSearch:
    """State of one AND-OR-GRAPH-SEARCH call.

    The recursion mirrors the textbook pseudocode; search state (solved table, current path,
    undo journal, counters) lives on the instance instead of being threaded through calls.
    Subclasses may override :meth:`ordered_actions` (tests use this to force bad orderings).
    """

    def __init__(self, model: ProblemModel, config: PlannerConfig | None = None) -> None:
        config = config or PlannerConfig()
        self.model = model
        self.goal: Pos = model.goal
        n_cells = model.rows * model.cols
        self.depth_limit = n_cells if config.depth_limit is None else config.depth_limit
        self.diagonal = config.tie_break == "diagonal"
        self.dist: list[list[int]] = bfs_distances(model.passable, model.goal).tolist()
        self.solved: dict[Pos, Action] = {}
        self.grounded: set[Pos] = {model.goal}
        # Undo log: (True, s) = s was marked grounded, (False, s) = solved[s] was set.
        self.journal: list[tuple[bool, Pos]] = []
        self.path: list[Pos] = []
        self.on_path: set[Pos] = set()
        self.or_nodes = 0
        self.and_nodes = 0
        self.expanded = 0
        self.rollbacks = 0

    # --- heuristics ------------------------------------------------------------------------
    def distance(self, p: Pos) -> float:
        """BFS distance from ``p`` to the goal on the belief map (``inf`` if unreachable)."""
        d = self.dist[p[0]][p[1]]
        return math.inf if d == UNREACHABLE else d

    def ordered_actions(self, s: Pos) -> list[Action]:
        """ORDERED_ACTIONS(s): intended-target distance, then diagonal tie-break, then N,E,S,W."""
        goal_r, goal_c = self.goal
        keyed: list[tuple[float, int, int, Action]] = []
        for index, action in enumerate(ACTIONS):
            target = step_pos(s, action)
            if not self.model.is_free(target):
                target = s
            diag = abs(abs(goal_r - target[0]) - abs(goal_c - target[1])) if self.diagonal else 0
            keyed.append((self.distance(target), diag, index, action))
        keyed.sort(key=lambda item: item[:3])
        return [item[3] for item in keyed]

    # --- AND-OR recursion ------------------------------------------------------------------
    def or_search(self, s: Pos, depth: int) -> PlanNode | None:
        """OR_SEARCH: pick one action at ``s`` whose every outcome is solved. ``None`` = FAIL."""
        self.or_nodes += 1
        if s == self.goal:
            return Goal(s)
        if s in self.solved:
            return Ref(s)
        if s in self.on_path:
            return Loop(s)
        # Depth cut-off, and dead-state pruning: a state with no path to the goal on the belief
        # map cannot appear in any strong-cyclic plan.
        if depth > self.depth_limit or self.distance(s) == math.inf:
            return None
        self.expanded += 1
        self.path.append(s)
        self.on_path.add(s)
        try:
            for action in self.ordered_actions(s):
                mark = len(self.journal)
                branches = self.and_search(self.model.results(s, action), depth + 1)
                if branches is not None:
                    self._record_solved(s, action)
                    if self.has_progress(s):
                        return Act(s, action, branches)
                self._rollback(mark)
            return None
        finally:
            self.path.pop()
            self.on_path.discard(s)

    def and_search(self, outcomes: Sequence[Outcome], depth: int) -> tuple[Branch, ...] | None:
        """AND_SEARCH: every outcome of the chosen action needs a subplan. ``None`` = FAIL."""
        self.and_nodes += 1
        branches: list[Branch] = []
        for outcome in outcomes:
            node = self.or_search(outcome.state, depth)
            if node is None:
                return None
            branches.append(Branch(outcome.state, outcome.labels, node))
        return tuple(branches)

    def has_progress(self, s: Pos) -> bool:
        """HAS_PROGRESS(s): can ``s`` reach the goal, a grounded state or a proper ancestor?

        Breadth-first search over the provisional policy graph. An unsolved state reached by
        the search is necessarily on the current path (every outcome of a solved state is the
        goal, solved, or was on the path when it was solved), i.e. a proper ancestor of ``s``:
        ``s`` is then *pending* on it. Reaching the goal or a grounded state grounds ``s``.
        """
        frontier: deque[Pos] = deque([s])
        seen = {s}
        reaches_ancestor = False
        while frontier:
            x = frontier.popleft()
            action = self.solved.get(x)
            if action is None:
                reaches_ancestor = True
                continue
            for outcome in self.model.results(x, action):
                y = outcome.state
                if y in self.grounded:
                    self._mark_grounded(s)
                    return True
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)
        return reaches_ancestor

    # --- transactional solved table ---------------------------------------------------------
    def _record_solved(self, s: Pos, action: Action) -> None:
        self.solved[s] = action
        self.journal.append((False, s))

    def _mark_grounded(self, s: Pos) -> None:
        if s not in self.grounded:
            self.grounded.add(s)
            self.journal.append((True, s))

    def _rollback(self, mark: int) -> None:
        """Undo every solved/grounded entry made since journal position ``mark``."""
        self.rollbacks += 1
        while len(self.journal) > mark:
            is_grounded, state = self.journal.pop()
            if is_grounded:
                self.grounded.discard(state)
            else:
                del self.solved[state]

    # --- driver ----------------------------------------------------------------------------
    def run(self, start: Pos) -> PlanNode | None:
        """AND_OR_SEARCH(problem): search from ``start``; returns the root node or ``None``.

        Safe to call from several threads at once (each on its own instance): the recursion
        limit is raised through a shared, reference-counted guard.
        """
        levels = min(self.depth_limit, self.model.rows * self.model.cols) + 1
        with _recursion_headroom(FRAMES_PER_LEVEL * levels + RECURSION_SLACK):
            return self.or_search(start, 0)


def and_or_search(
    model: ProblemModel, start: Pos, config: PlannerConfig | None = None
) -> PlanResult:
    """Run AND-OR graph search from ``start``; ``result.plan is None`` means FAIL."""
    t0 = time.perf_counter()
    search = AndOrSearch(model, config)
    root = search.run(start)
    elapsed_ms = (time.perf_counter() - t0) * MS_PER_S
    plan = None if root is None else Plan(root=root, start=start, goal=model.goal)
    stats = PlannerStats(
        nodes_expanded=search.expanded,
        or_nodes=search.or_nodes,
        and_nodes=search.and_nodes,
        rollbacks=search.rollbacks,
        time_ms=elapsed_ms,
        plan_size=len(search.solved) if plan is not None else 0,
    )
    return PlanResult(plan=plan, stats=stats)


class AndOrPlanner:
    """AND-OR planner with optional validation and a reference-planner fallback.

    ``plan()`` runs :func:`and_or_search`. With ``config.validate`` every plan is checked with
    :func:`is_strong_cyclic` (the AND-OR plan and, if one is used, the fallback plan); a plan
    that fails is never returned. If the search fails (other than for a start that provably
    cannot reach the goal) or its plan is rejected, and ``config.fallback_to_reference`` is
    set, the fixpoint strong-cyclic planner is consulted; when it finds a plan, that plan is
    returned with ``stats.fallback_used`` set so the event is counted as a
    ``planner_fallback``. See :class:`PlannerStats` for the meaning of each statistic.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, model: ProblemModel, start: Pos) -> PlanResult:
        """Plan from ``start`` on ``model``."""
        t0 = time.perf_counter()
        result = and_or_search(model, start, self.config)
        plan = result.plan
        plan_size = result.stats.plan_size
        validated: bool | None = None
        fallback_used = False
        if plan is not None and self.config.validate:
            validated = is_strong_cyclic(plan.to_policy(), model, start)
            if not validated:
                log.warning("AND-OR produced a plan that is not strong-cyclic from %s", start)
                plan = None
        # The reachability check runs only when a fallback is actually considered: a start
        # with no path to the goal has no strong-cyclic plan, so the reference is skipped.
        if (
            plan is None
            and self.config.fallback_to_reference
            and reachable(model.passable, start, model.goal)
        ):
            policy = strong_cyclic_policy(model, start)
            if policy is not None:
                log.warning("AND-OR failed from %s; using the fixpoint reference plan", start)
                ok = True
                if self.config.validate:
                    ok = is_strong_cyclic(policy, model, start)
                    if not ok:
                        log.error("fixpoint reference plan from %s is not strong-cyclic", start)
                    validated = ok if validated is None else (validated and ok)
                if ok:
                    plan = plan_from_policy(policy, model, start)
                    plan_size = len(policy)
                    fallback_used = True
        stats = replace(
            result.stats,
            time_ms=(time.perf_counter() - t0) * MS_PER_S,
            plan_size=plan_size if plan is not None else 0,
            fallback_used=fallback_used,
            validated=validated,
        )
        return PlanResult(plan=plan, stats=stats)
