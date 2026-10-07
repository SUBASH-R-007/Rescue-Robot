"""Search-and-rescue agents for unknown and multiple survivors (brief §8.4–§8.5, plan §3.14).

:class:`SearchAgent` extends the core plan → act → observe → update belief → replan loop with
*target selection*. The agent knows how many survivors there are (a radio count) but, in the
search setting, not where they are. Its current target is either the nearest known unrescued
survivor (``mode == "rescue"``) or, while none is known, the nearest frontier cell
(``mode == "explore"``). It reaches the target with the planner it was built with:

* ``planner="andor"`` (``search_andor``): an AND-OR strong-cyclic plan on
  ``ProblemModel(passable, goal=target, transition)``, followed as a policy (the core
  :class:`~rescue.agents.contingency.ContingencyAgent` with a moving goal);
* ``planner="astar"`` (``search_astar``): an A* path, replanned on any deviation (the core
  ``astar_replan`` with a moving goal).

The belief can be a :class:`~rescue.belief.BeliefMap` (noise-free sensing) or a
:class:`~rescue.ext.sensing.ProbBelief` (noisy sensing); only ``get``, ``passable_mask``,
``survivors_seen``, ``update`` and the known-cell mask are used.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection
from typing import ClassVar, Literal

import numpy as np

from rescue.agents.base import (
    NO_REPLAN,
    Abort,
    Agent,
    AgentStats,
    AgentUpdate,
    Decision,
    Observation,
    ReplanInfo,
    direction,
)
from rescue.belief import BeliefMap, BeliefUpdate
from rescue.config import EXT_FALLBACK_MAX_P, AgentConfig, PlannerConfig
from rescue.ext.sensing import ProbBelief
from rescue.grid import Cell, Pos, fmt_pos, neighbours4
from rescue.model import Action, ProblemModel, TransitionModel, step_pos
from rescue.plan import Plan
from rescue.planners.andor import MS_PER_S, AndOrPlanner
from rescue.planners.astar import astar
from rescue.planners.bfs import UNREACHABLE, bfs_distances
from rescue.world import Percept

log = logging.getLogger(__name__)

SearchPlanner = Literal["andor", "astar"]
SearchMode = Literal["explore", "rescue"]
#: Planners a :class:`SearchAgent` accepts.
SEARCH_PLANNERS: tuple[str, ...] = ("andor", "astar")
#: Belief types a :class:`SearchAgent` works with.
AnyBelief = BeliefMap | ProbBelief


def known_mask(belief: AnyBelief) -> np.ndarray:
    """Bool array of the cells the belief has observed (or knows in advance)."""
    if isinstance(belief, ProbBelief):
        return belief.observed.copy()
    return belief.grid != Cell.UNKNOWN


def relaxed_passable_mask(belief: AnyBelief) -> np.ndarray:
    """The fallback passability of :class:`SearchAgent`: every cell whose debris probability is
    below ``max(threshold, EXT_FALLBACK_MAX_P)`` (a :class:`ProbBelief`), so only (near-)certain
    debris blocks; a :class:`~rescue.belief.BeliefMap` holds only certain debris, so its relaxed
    mask is its ordinary one."""
    if isinstance(belief, ProbBelief):
        return belief.p_debris < max(belief.threshold, EXT_FALLBACK_MAX_P)
    return belief.passable_mask()


def frontier_mask(belief: AnyBelief, passable: np.ndarray | None = None) -> np.ndarray:
    """Frontier cells: known, passable, with at least one UNKNOWN 4-neighbour (``passable``
    defaults to the belief's own passable mask)."""
    if passable is None:
        passable = belief.passable_mask()
    known = known_mask(belief)
    unknown = ~known
    near_unknown = np.zeros_like(unknown)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    return known & passable & near_unknown


class SearchAgent(Agent):
    """Explore-then-rescue agent for ``n_survivors`` survivors (``search_andor`` /
    ``search_astar``).

    **Target selection** happens at t = 0 and is re-evaluated *only* after a rescue, when the
    current target stops being valid (a rescue target must still be a known unrescued survivor,
    an explore target still a frontier cell, and either must be reachable on the optimistic
    belief and differ from the robot's cell), or when a new survivor is seen:

    1. if a known unrescued survivor is reachable, the nearest one by BFS distance on the
       agent's optimistic passability (ties row-major) — greedy nearest-first order;
    2. otherwise the nearest reachable frontier cell (known, passable, with an UNKNOWN
       4-neighbour; ties row-major). With sensor radius 0 the robot can stand on a frontier
       cell without seeing its unknown neighbours; the target is then that cell's first
       passable unknown neighbour in N, E, S, W order (entering or bumping it reveals it);
    3. otherwise ``choose_action`` returns ``Abort("unreachable", …)``.

    Only cells the belief holds as ``SURVIVOR`` — sensed, or given in advance when the
    locations are known — are ever rescue targets; an UNKNOWN cell is never assumed to hide a
    survivor. Rescues are reported by the mission (:meth:`on_rescue`).

    **Replanning** (always inside ``on_outcome``, as in the core): at t = 0 (not a replan),
    after a target change, on an unplanned state (D10), on contradictions (``eager``: any;
    ``lazy``: in the plan envelope / on the remaining path), and for ``andor`` when the
    position has no policy entry, for ``astar`` when the robot is off the path or the next path
    cell is known debris.

    **Termination of exploration** (informal proof). Between two target changes the agent
    follows a strong-cyclic plan (or a replanned A* path) to a fixed, reachable target, so it
    reaches it with probability 1 unless the belief changes first. Every target change in
    explore mode has one of these causes, each possible only finitely often: (a) the target
    stopped being a frontier cell — with radius r ≥ 1 this happens at the latest on arrival,
    because the robot then senses every 4-neighbour of the target, so at least one UNKNOWN cell
    became known; (b) the target became unreachable or impassable — a cell became debris in the
    belief, which happens at most once per cell (no false positives: debris is never un-seen);
    (c) a survivor was seen or rescued — at most twice per survivor. The grid is finite, so
    after finitely many target changes either every survivor is rescued or no frontier is left.
    On a solvable map no frontier with an unfound survivor is impossible: take a true path from
    the robot to an unfound survivor; its cells are truly passable, hence believed passable
    (sensing has no false positives), and the survivor cell itself is UNKNOWN, so the last
    known cell before the first UNKNOWN cell on the path is a reachable frontier cell.

    **False positives and the fallback** (round 2, plan §3.15.2). With a false-positive rate
    ``fp > 0`` (a :class:`~rescue.ext.sensing.ProbBelief`) a truly free cell can read DEBRIS and
    cross the threshold: a *phantom* wall. Both arguments above then fail: debris can be un-seen
    (a later OPEN reading lowers ``P``), so (b) can repeat, and a phantom cell on every true path
    can leave no reachable target. So when the strict search (``P < threshold``) finds no
    reachable target, the agent (``fallback=True``, the default) recomputes the target and its
    plan on the relaxed passability of :func:`relaxed_passable_mask` (``P <
    EXT_FALLBACK_MAX_P``: only (near-)certain debris, e.g. a cell touched by a bump, stays
    blocked) before giving up. The relaxed mode lasts until the next target re-selection, which
    tries the strict passability first again. A fallback is entered only when the relaxed mask
    differs from the strict one. :attr:`fallback_replans` counts fallback *episodes*: switches
    from strict to relaxed mode. A re-selection made while relaxed that falls back again (the
    strict search still finds nothing) continues the episode and is not counted, so the count
    does not depend on how often the plan in force triggers a re-selection (an AND-OR envelope,
    which holds skid cells off the main path, triggers it more often than an A* path).
    Termination with ``fp > 0`` is no longer guaranteed; the mission's step budget bounds every
    run.

    A target that is reachable on the passability in force always has a plan: A* finds a path
    to any reachable cell, and AND-OR (sound and complete) finds a strong-cyclic plan, because every
    outcome of a move stays in the robot's connected component (a skid or slip lands on a
    passable 4-neighbour of a reachable cell, or stays put), from which the target is reachable.
    So "no plan to a reachable target" is not a fallback trigger: if a planner ever returned
    none, ``choose_action`` aborts (``"unreachable"``) as in round 1.

    *The fallback cannot change anything when* ``fp = 0``. With a
    :class:`~rescue.belief.BeliefMap` the relaxed mask is the strict mask. With a
    :class:`~rescue.ext.sensing.ProbBelief` and ``fp = 0`` every cell is in one of three states:
    never observed (``P`` = the prior, below the threshold, as the mission enforces), read
    DEBRIS or touched at least once and not un-seen (``P = 1`` exactly: without false positives
    a DEBRIS reading sets 1 and an OPEN reading of a ``P = 1`` cell keeps 1 when ``fn > 0``; it
    is impossible when ``fn = 0``), or only ever read OPEN (``P`` ≤ the prior: an OPEN reading
    never raises ``P``). Hence ``P ≥ threshold ⟺ P = 1 ⟺ P ≥ max(threshold,
    EXT_FALLBACK_MAX_P)``: the two masks are equal, the fallback is never entered, and every
    decision, plan and event is the round-1 one (on solvable and unsolvable maps alike; tested
    against recorded traces in ``tests/test_ext_fp.py``).
    """

    #: Placeholder; every instance sets ``name`` to ``"search_<planner>"``.
    name: ClassVar[str] = "search"

    def __init__(
        self,
        transition: TransitionModel,
        planner: SearchPlanner = "andor",
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
        *,
        fallback: bool = True,
    ) -> None:
        if planner not in SEARCH_PLANNERS:
            raise ValueError(f"unknown planner {planner!r}; choose from {SEARCH_PLANNERS}")
        self.planner_name: SearchPlanner = planner
        #: Recompute targets / plans on relaxed passability before aborting (class docstring).
        self.fallback: bool = bool(fallback)
        self._contact: tuple[tuple[Pos, ...], Pos | None] = ((), None)
        self.name = f"search_{planner}"  # type: ignore[misc]  # per-instance registry id
        # Deviations are routine for A*, a bug for AND-OR (as in the core agents).
        self.unplanned_log_level = (  # type: ignore[misc]
            logging.WARNING if planner == "andor" else logging.DEBUG
        )
        self._reset_search_state(1)
        super().__init__(transition, planner_config, agent_config)
        self._andor = AndOrPlanner(self.planner_config)

    # --- lifecycle -------------------------------------------------------------------------
    def _reset_search_state(self, n_survivors: int) -> None:
        self.n_survivors: int = int(n_survivors)
        self.rescued: set[Pos] = set()
        self.target: Pos | None = None
        self.mode: SearchMode = "explore"
        self._rescue_pending = False
        self._passable: np.ndarray | None = None
        self._pass_version = 0
        self._dist_key: tuple[int, Pos] | None = None
        self._dist: np.ndarray | None = None
        self._target_checked_version = -1
        #: True while targets and plans use the relaxed (fallback) passability.
        self.relaxed: bool = False
        #: Fallback episodes: switches from strict to relaxed passability (class docstring).
        self.fallback_replans: int = 0

    def reset_search(self, start: Pos, belief: AnyBelief, n_survivors: int) -> None:
        """Start a mission at ``start`` with ``belief`` (used in place) and ``n_survivors``."""
        if n_survivors < 1:
            raise ValueError(f"n_survivors must be >= 1, got {n_survivors}")
        self.belief = belief  # type: ignore[assignment]
        self.pos = (int(start[0]), int(start[1]))
        self.goal = self.pos
        self.stats = AgentStats()
        self.snapshots = []
        self.current_plan_id = -1
        self.t = 0
        self._observed = False
        self._last_pos = None
        self._last_action = None
        self._reset_search_state(n_survivors)
        self._reset_plan()
        # With false positives an unplanned state is an expected model mismatch (the robot
        # skids into a cell believed to be debris that is free), not a planner bug.
        noisy_fp = isinstance(belief, ProbBelief) and belief.false_positive > 0.0
        self.unplanned_log_level = (  # type: ignore[misc]
            logging.WARNING if self.planner_name == "andor" and not noisy_fp else logging.DEBUG
        )

    def reset(self, start: Pos, goal: Pos, belief: BeliefMap) -> None:
        """Core-compatible reset: one survivor, known at ``goal`` (the belief's survivor)."""
        goal = (int(goal[0]), int(goal[1]))
        if belief.get(goal) is not Cell.SURVIVOR:
            raise ValueError(f"goal {fmt_pos(goal)} is not a survivor in the belief")
        self.reset_search(start, belief, 1)

    def on_rescue(self, pos: Pos) -> None:
        """The mission rescued the survivor at ``pos``; the target is re-selected next update."""
        self.rescued.add((int(pos[0]), int(pos[1])))
        self._rescue_pending = True

    @property
    def done(self) -> bool:
        """True once every survivor has been rescued."""
        return len(self.rescued) >= self.n_survivors

    # --- plan state ------------------------------------------------------------------------
    def _reset_plan(self) -> None:
        self._model: ProblemModel | None = None
        self._plan: Plan | None = None
        self._policy: dict[Pos, Action] = {}
        self._envelope: frozenset[Pos] = frozenset()
        self._path: tuple[Pos, ...] | None = None
        self._cursor = 0

    @property
    def plan(self) -> Plan | None:
        """The AND-OR plan in force (``search_andor`` only; None otherwise)."""
        return self._plan

    @property
    def path(self) -> tuple[Pos, ...] | None:
        """The A* path in force (``search_astar`` only; None otherwise)."""
        return self._path

    # --- belief-derived caches -------------------------------------------------------------
    def _mask(self) -> np.ndarray:
        """The passability in force: strict (``P < threshold``) or, in fallback mode, relaxed."""
        if self.relaxed:
            return relaxed_passable_mask(self.belief)
        return self.belief.passable_mask()

    def _refresh_passability(self) -> None:
        """Bump the passability version when the passable mask in force changed."""
        mask = self._mask()
        if self._passable is None or not np.array_equal(mask, self._passable):
            self._passable = mask
            self._pass_version += 1

    def _distances(self) -> np.ndarray:
        """BFS distances from the robot on the optimistic belief (cached per version, pos)."""
        key = (self._pass_version, self.pos)
        if self._dist_key != key or self._dist is None:
            assert self._passable is not None
            self._dist = bfs_distances(self._passable, self.pos)
            self._dist_key = key
        return self._dist

    def _is_frontier(self, p: Pos) -> bool:
        belief = self.belief
        if belief.get(p) is Cell.UNKNOWN or not bool(self._passable_at(p)):
            return False
        return any(belief.get(q) is Cell.UNKNOWN for q in neighbours4(p, belief.rows,
                                                                      belief.cols))

    def _passable_at(self, p: Pos) -> bool:
        assert self._passable is not None
        return bool(self._passable[p])

    # --- target selection ------------------------------------------------------------------
    def _target_valid(self) -> bool:
        """Is the current target still a valid target (see the class docstring)?"""
        target = self.target
        if target is None or target == self.pos or not self._passable_at(target):
            return False
        cell = self.belief.get(target)
        if self.mode == "rescue":
            if target in self.rescued or cell is not Cell.SURVIVOR:
                return False
        elif cell is Cell.UNKNOWN:
            # Radius-0 exploration step onto an unknown neighbour: valid until it is seen.
            pass
        elif not self._is_frontier(target):
            return False
        # Reachability changes only when passability does.
        if self._target_checked_version != self._pass_version:
            if self._distances()[target] == UNREACHABLE:
                return False
            self._target_checked_version = self._pass_version
        return True

    def _select_target(self) -> None:
        """Pick the nearest known unrescued survivor, else the nearest frontier cell."""
        dist = self._distances()
        survivors = [
            s for s in self.belief.survivors_seen()
            if s not in self.rescued and s != self.pos and dist[s] != UNREACHABLE
        ]
        self._target_checked_version = self._pass_version
        if survivors:
            self.target = min(survivors, key=lambda s: (int(dist[s]), s))
            self.mode = "rescue"
            return
        self.mode = "explore"
        frontier = frontier_mask(self.belief, self._passable) & (dist != UNREACHABLE)
        if frontier[self.pos]:
            # Radius 0: the robot stands on a frontier cell it cannot see past.
            belief = self.belief
            for q in neighbours4(self.pos, belief.rows, belief.cols):
                if belief.get(q) is Cell.UNKNOWN and self._passable_at(q):
                    self.target = q
                    return
            frontier[self.pos] = False
        cells = np.argwhere(frontier)
        if cells.size == 0:
            self.target = None
            return
        d = dist[frontier]  # row-major, aligned with ``cells``
        best = int(np.argmin(d))  # first minimum = row-major tie-break
        self.target = (int(cells[best][0]), int(cells[best][1]))

    def _choose_target(self) -> None:
        """Target selection with the fallback: strict passability first, then (if no target is
        reachable) the relaxed one."""
        was_relaxed = self.relaxed
        if was_relaxed:
            self.relaxed = False
            self._refresh_passability()
        self._select_target()
        if self.target is None and self._enter_fallback(new_episode=not was_relaxed):
            self._select_target()

    def _enter_fallback(self, *, new_episode: bool) -> bool:
        """Switch to the relaxed passability if allowed and different; True if switched.
        ``new_episode`` (the agent was in strict mode) counts one :attr:`fallback_replans`."""
        if not self.fallback or self.relaxed:
            return False
        assert self._passable is not None
        if np.array_equal(relaxed_passable_mask(self.belief), self._passable):
            return False
        self.relaxed = True
        if new_episode:
            self.fallback_replans += 1
            log.info("%s: no reachable target under the debris threshold at %s (t=%d); "
                     "falling back to relaxed passability", self.name, fmt_pos(self.pos),
                     self.t)
        else:
            log.debug("%s: still no reachable target under the debris threshold at %s (t=%d); "
                      "staying relaxed", self.name, fmt_pos(self.pos), self.t)
        self._refresh_passability()
        return True

    def _relaxed_plan_blocked(self) -> bool:
        """Fallback mode only: a cell of the plan in force (AND-OR envelope / remaining A*
        path) is impassable under the relaxed passability (always False in strict mode)."""
        if not self.relaxed:
            return False
        assert self._passable is not None
        return any(not bool(self._passable[c]) for c in self._plan_cells())

    # --- planning --------------------------------------------------------------------------
    def _plan_to_target(self) -> bool:
        """Plan from ``self.pos`` to ``self.target``; True if it was a replan (D12)."""
        target = self.target
        assert target is not None
        self.goal = target
        passable = self._mask()
        if self.planner_name == "andor":
            model = ProblemModel(passable, target, self.transition)
            result = self._andor.plan(model, self.pos)
            plan = result.plan
            self._model = model
            self._plan = plan
            self._policy = plan.to_policy() if plan is not None else {}
            self._envelope = plan.envelope() if plan is not None else frozenset()
            if plan is None:
                log.info("%s: no strong-cyclic plan from %s to %s (t=%d)", self.name,
                         fmt_pos(self.pos), fmt_pos(target), self.t)
            stats = result.stats
            return self._record_plan(
                plan=plan,
                path=None,
                envelope=self._envelope,
                elapsed_ms=stats.time_ms,
                nodes_expanded=stats.nodes_expanded,
                plan_size=stats.plan_size,
                fallback_used=stats.fallback_used,
            )
        t0 = time.perf_counter()
        found = astar(passable, self.pos, target, self.planner_config.astar_tie_break)
        elapsed_ms = (time.perf_counter() - t0) * MS_PER_S
        path = tuple(found.path) if found.path is not None else None
        self._path = path
        self._cursor = 0
        if path is None:
            log.info("%s: A* found no path from %s to %s (t=%d)", self.name, fmt_pos(self.pos),
                     fmt_pos(target), self.t)
        return self._record_plan(
            plan=None,
            path=path,
            envelope=frozenset(path) if path is not None else frozenset(),
            elapsed_ms=elapsed_ms,
            nodes_expanded=found.nodes_expanded,
            plan_size=len(path) - 1 if path is not None else 0,
        )

    def _expected_cell(self) -> Pos | None:
        if self._path is None or self._cursor >= len(self._path):
            return None
        return self._path[self._cursor]

    def _next_cell(self) -> Pos | None:
        if self._path is None or self._cursor + 1 >= len(self._path):
            return None
        return self._path[self._cursor + 1]

    def _plan_cells(self) -> frozenset[Pos]:
        """Cells a lazy replan watches: the AND-OR envelope or the remaining A* path."""
        if self.planner_name == "andor":
            return self._envelope
        if self._path is None:
            return frozenset()
        return frozenset(self._path[self._cursor + 1:])

    def _plan_stale(self) -> bool:
        """The plan in force cannot be followed from ``self.pos``."""
        if self.planner_name == "andor":
            return self._plan is None or self.pos not in self._policy
        nxt = self._next_cell()
        if self._path is None or self._expected_cell() != self.pos or nxt is None:
            return True
        if self.relaxed:
            # Fallback: a suspected (not certain) debris cell on the path may be tried.
            return not bool(relaxed_passable_mask(self.belief)[nxt])
        return self.belief.get(nxt) is Cell.DEBRIS

    # --- agent interface -------------------------------------------------------------------
    def on_outcome(self, new_pos: Pos, percept: Percept,
                   touched: Collection[Pos] = ()) -> AgentUpdate:
        """The core ``on_outcome`` plus what the mission knows for certain: the cells a bump
        ``touched`` (debris) and the robot's own cell ``new_pos`` (free). A
        :class:`~rescue.ext.sensing.ProbBelief` uses them (``update(..., touched, here)``); a
        :class:`~rescue.belief.BeliefMap` ignores them (the percept already holds the touched
        cell as DEBRIS)."""
        self._contact = (tuple((int(p[0]), int(p[1])) for p in touched),
                         (int(new_pos[0]), int(new_pos[1])))
        try:
            return super().on_outcome(new_pos, percept)
        finally:
            self._contact = ((), None)

    def _sense(self, percept: Percept) -> BeliefUpdate:
        belief = self.belief
        if not isinstance(belief, ProbBelief):
            return super()._sense(percept)
        touched, here = self._contact
        update = belief.update(percept.cells, touched, here)
        if update.contradictions:
            self.stats.contradictions += len(update.contradictions)
        return update

    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        if self.done:
            return NO_REPLAN
        initial = not self.snapshots
        if not initial and self.planner_name == "astar" and self._path is not None:
            self._cursor += 1  # one executed action: the robot is expected one cell further
        self._refresh_passability()
        new_survivor = any(self.belief.get(p) is Cell.SURVIVOR for p in update.newly_known)
        target_changed = False
        # Fallback mode: a contradiction is judged against the threshold, so a suspected cell
        # (P >= threshold) the relaxed plan crosses going certain (a bump) is no contradiction;
        # such a cell re-selects the target (strict passability first) and replans instead.
        relaxed_blocked = self._relaxed_plan_blocked()
        if (initial or self._rescue_pending or new_survivor or relaxed_blocked
                or not self._target_valid()):
            previous = self.target
            self._choose_target()
            target_changed = self.target != previous
        self._rescue_pending = False
        if self.target is None:
            self._reset_plan()
            return NO_REPLAN
        # Judged against the plan in force, before any replan.
        triggers = self._relevant_contradictions(update.contradictions, self._plan_cells())
        if not (initial or target_changed or unplanned or triggers or relaxed_blocked
                or self._plan_stale()):
            return NO_REPLAN
        # The target is reachable on the passability in force, so a plan exists (class
        # docstring); one planning call per update keeps ``stats.replans`` equal to the number
        # of updates with ``replanned=True`` (agents/base.py).
        replanned = self._plan_to_target()
        return ReplanInfo(replanned=replanned, triggers=triggers)

    def choose_action(self, obs: Observation) -> Decision:
        """Next action towards the target, or ``Abort("unreachable")``.

        ``RuntimeError`` before the first ``on_outcome``, once every survivor is rescued, or
        when the plan is stale (``on_outcome`` should have replanned); ``ValueError`` if
        ``obs.pos`` is not the agent's position.
        """
        if not self._observed:
            raise RuntimeError("on_outcome() must be called (at t=0) before choose_action()")
        pos = (int(obs.pos[0]), int(obs.pos[1]))
        if pos != self.pos:
            raise ValueError(
                f"observation position {fmt_pos(pos)} != agent position {fmt_pos(self.pos)}"
            )
        if self.done:
            raise RuntimeError(f"{self.name}: every survivor is rescued; the mission is over")
        if self.target is None:
            return Abort("unreachable", f"no reachable survivor or frontier from {fmt_pos(pos)}")
        if self.planner_name == "andor":
            if self._plan is None:
                return Abort("unreachable", f"no strong-cyclic plan from {fmt_pos(pos)} to "
                             f"{fmt_pos(self.target)}")
            action = self._policy.get(pos)
            if action is None:
                raise RuntimeError(f"{self.name}: no policy entry at {fmt_pos(pos)} "
                                   f"(t={self.t}); on_outcome() should have replanned")
            return self._begin_action(pos, action)
        if self._path is None:
            return Abort("unreachable", f"A* found no path from {fmt_pos(pos)} to "
                         f"{fmt_pos(self.target)}")
        if self._plan_stale():
            raise RuntimeError(f"{self.name}: stale path at {fmt_pos(pos)} (t={self.t}); "
                               "on_outcome() should have replanned")
        nxt = self._next_cell()
        assert nxt is not None
        return self._begin_action(pos, direction(pos, nxt))

    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]:
        """``Results(pos, action)`` of the AND-OR snapshot model, or the next A* path cell."""
        if self.planner_name == "andor":
            if self._model is None:
                return frozenset()
            return frozenset(o.state for o in self._model.results(pos, action))
        if self._expected_cell() == pos:
            nxt = self._next_cell()
            if nxt is not None:
                return frozenset((nxt,))
        return frozenset((step_pos(pos, action),))

    def policy_arrows(self) -> dict[Pos, Action]:
        """The AND-OR policy, or the arrows along the remaining A* path (a fresh dict)."""
        if self.planner_name == "andor":
            return dict(self._policy)
        path = self._path
        if path is None:
            return {}
        return {path[i]: direction(path[i], path[i + 1])
                for i in range(self._cursor, len(path) - 1)}


class SearchAndOrAgent(SearchAgent):
    """:class:`SearchAgent` with the AND-OR planner (registry id ``"search_andor"``)."""

    name: ClassVar[str] = "search_andor"

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
        *,
        fallback: bool = True,
    ) -> None:
        super().__init__(transition, "andor", planner_config, agent_config,
                         fallback=fallback)


class SearchAStarAgent(SearchAgent):
    """:class:`SearchAgent` with A* + replan (registry id ``"search_astar"``)."""

    name: ClassVar[str] = "search_astar"

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
        *,
        fallback: bool = True,
    ) -> None:
        super().__init__(transition, "astar", planner_config, agent_config,
                         fallback=fallback)


#: Registry: extension agent id → agent class.
EXT_AGENTS: dict[str, type[SearchAgent]] = {
    SearchAndOrAgent.name: SearchAndOrAgent,
    SearchAStarAgent.name: SearchAStarAgent,
}
#: Extension agent ids in display order.
EXT_AGENT_NAMES: tuple[str, ...] = tuple(EXT_AGENTS)


def make_ext_agent(
    name: str,
    transition: TransitionModel,
    planner_config: PlannerConfig | None = None,
    agent_config: AgentConfig | None = None,
    *,
    fallback: bool = True,
) -> SearchAgent:
    """Instantiate the extension agent registered as ``name`` (ValueError for unknown names);
    ``fallback`` as in :class:`SearchAgent`."""
    try:
        cls = EXT_AGENTS[name]
    except KeyError:
        raise ValueError(
            f"unknown extension agent {name!r}; choose from {', '.join(EXT_AGENT_NAMES)}"
        ) from None
    return cls(transition, planner_config, agent_config,  # type: ignore[call-arg]
               fallback=fallback)
