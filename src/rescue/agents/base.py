"""Common agent interface and shared machinery (docs/plan.md §3.8, §3.10).

Every agent runs the same loop, driven by the runner:

1. ``reset(start, goal, belief)`` once per mission;
2. ``on_outcome(start, percept)`` at t = 0 (first sensing, initial plan);
3. repeatedly ``choose_action(obs)`` → world step → ``on_outcome(new_pos, percept)``.

The base class owns the parts every agent shares: the belief update and contradiction
bookkeeping (D8), the unplanned-state check (D10, against :meth:`Agent.anticipated` of the plan
in force when the action was chosen), planning statistics, replan counting (D12: only calls
after the first count) and the :class:`PlanSnapshot` history used by the UI and replays.
Bumps (D6) are computed by the runner, not the agent.

Every replan happens inside ``on_outcome`` and is reported in its :class:`AgentUpdate`, so
``stats.replans`` always equals the number of updates with ``replanned=True``.
``choose_action`` never plans: a state where ``on_outcome`` should have replanned but did not
is a logic error and raises ``RuntimeError``. Calling ``choose_action`` on the goal (the
mission is over) also raises ``RuntimeError``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from rescue.belief import BeliefMap, BeliefUpdate
from rescue.config import AgentConfig, PlannerConfig
from rescue.grid import Cell, Pos, fmt_pos
from rescue.model import ACTIONS, Action, TransitionModel, step_pos
from rescue.plan import Plan
from rescue.planners.andor import MS_PER_S
from rescue.planners.astar import astar
from rescue.world import Percept

log = logging.getLogger(__name__)

AbortReason = Literal["aborted", "unreachable"]

#: ``(d_row, d_col)`` → the action that produces it.
_DELTA_TO_ACTION: dict[tuple[int, int], Action] = {a.delta: a for a in ACTIONS}


@dataclass(frozen=True)
class Observation:
    """What the runner tells the agent before it acts: its position and the time step."""

    pos: Pos
    t: int


@dataclass(frozen=True)
class Abort:
    """The agent gives up: ``"aborted"`` (baseline-specific) or ``"unreachable"`` (no plan)."""

    reason: AbortReason
    detail: str


#: What :meth:`Agent.choose_action` returns.
Decision = Action | Abort


@dataclass
class AgentStats:
    """Per-mission counters, updated in place by the agent.

    ``plans_made`` counts every planning call (including failed ones); ``replans`` those after
    the first (D12). ``plan_size`` is the size (states with an action) of the most recent plan,
    0 if that call failed; ``max_plan_size`` the largest so far. ``contradictions`` counts
    contradicted cells over the whole mission, t = 0 included.
    """

    replans: int = 0
    plans_made: int = 0
    planning_ms: float = 0.0
    nodes_expanded: int = 0
    plan_size: int = 0
    max_plan_size: int = 0
    unplanned_states: int = 0
    planner_fallbacks: int = 0
    contradictions: int = 0


@dataclass(frozen=True)
class AgentUpdate:
    """What :meth:`Agent.on_outcome` reports to the runner (bumps are the runner's job)."""

    newly_known: tuple[Pos, ...]
    contradictions: tuple[Pos, ...]
    #: A planning call after the initial plan happened in this update.
    replanned: bool
    #: The new position was not anticipated by the plan in force (D10).
    unplanned: bool
    #: The contradicted cells that demanded this update's replan: all of them in ``eager``
    #: mode, those in the plan envelope / on the remaining path in ``lazy`` mode. Empty when
    #: there was no replan or it had other causes only (deviation, unplanned state).
    triggering_contradictions: tuple[Pos, ...] = ()


@dataclass(frozen=True)
class ReplanInfo:
    """What :meth:`Agent._after_outcome` did: whether it replanned, and the contradicted cells
    that demanded a replan (see :attr:`AgentUpdate.triggering_contradictions`)."""

    replanned: bool = False
    triggers: tuple[Pos, ...] = ()


#: :meth:`Agent._after_outcome` result when nothing was replanned.
NO_REPLAN = ReplanInfo()


@dataclass(frozen=True)
class PlanSnapshot:
    """The plan in force from time ``t`` on, made at ``pos``.

    Contingency agents set ``plan`` (``path`` is None); path agents set ``path`` (``plan`` is
    None). Both are None for a failed planning call. ``envelope`` is every state the plan can
    visit (the path cells for path agents; empty when planning failed).
    """

    plan_id: int
    t: int
    pos: Pos
    plan: Plan | None
    path: tuple[Pos, ...] | None
    envelope: frozenset[Pos]


def direction(a: Pos, b: Pos) -> Action:
    """The action that moves from ``a`` to the 4-adjacent cell ``b`` (ValueError otherwise)."""
    action = _DELTA_TO_ACTION.get((b[0] - a[0], b[1] - a[1]))
    if action is None:
        raise ValueError(f"{fmt_pos(a)} and {fmt_pos(b)} are not 4-adjacent")
    return action


class Agent(ABC):
    """Abstract rescue agent (plan → act → observe → update belief → replan).

    Subclasses implement :meth:`choose_action`, :meth:`anticipated`, :meth:`policy_arrows`,
    :meth:`_reset_plan` (clear plan state) and :meth:`_after_outcome` (replanning logic).
    """

    #: Registry id ("andor", "astar_fixed", ...).
    name: ClassVar[str]
    #: Log level for unplanned states (a warning for the contingency agent, where one is a bug).
    unplanned_log_level: ClassVar[int] = logging.WARNING

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.transition = transition
        self.planner_config = planner_config or PlannerConfig()
        self.agent_config = agent_config or AgentConfig()
        self.stats = AgentStats()
        self.snapshots: list[PlanSnapshot] = []
        self.current_plan_id: int = -1
        self.t: int = 0
        self._observed = False
        self._last_pos: Pos | None = None
        self._last_action: Action | None = None
        self.belief: BeliefMap
        self.pos: Pos
        self.goal: Pos
        self._reset_plan()

    @property
    def eager(self) -> bool:
        """True when any contradiction triggers a replan (``replan_mode == "eager"``)."""
        return self.agent_config.replan_mode == "eager"

    # --- lifecycle -------------------------------------------------------------------------
    def reset(self, start: Pos, goal: Pos, belief: BeliefMap) -> None:
        """Start a new mission at ``start``; ``belief`` is used (and updated) in place."""
        goal = (int(goal[0]), int(goal[1]))
        if goal != belief.survivor:
            raise ValueError(
                f"goal {fmt_pos(goal)} differs from the belief's survivor "
                f"{fmt_pos(belief.survivor)}"
            )
        self.belief = belief
        self.pos = (int(start[0]), int(start[1]))
        self.goal = goal
        self.stats = AgentStats()
        self.snapshots = []
        self.current_plan_id = -1
        self.t = 0
        self._observed = False
        self._last_pos = None
        self._last_action = None
        self._reset_plan()

    def on_outcome(self, new_pos: Pos, percept: Percept) -> AgentUpdate:
        """Observe the result of the last action (or the start at t = 0) and maybe replan.

        Order: belief update → unplanned check (against the plan the action came from) →
        move to ``new_pos`` → subclass replanning (:meth:`_after_outcome`).
        """
        if self._observed:
            self.t += 1
        self._observed = True
        update = self._sense(percept)
        new_pos = (int(new_pos[0]), int(new_pos[1]))
        unplanned = self._check_unplanned(new_pos)
        self.pos = new_pos
        info = self._after_outcome(update, unplanned)
        self._last_pos = None
        self._last_action = None
        return AgentUpdate(
            newly_known=update.newly_known,
            contradictions=update.contradictions,
            replanned=info.replanned,
            unplanned=unplanned,
            triggering_contradictions=info.triggers if info.replanned else (),
        )

    @abstractmethod
    def choose_action(self, obs: Observation) -> Decision:
        """Next action from ``obs.pos``, or an :class:`Abort`.

        Raises ``RuntimeError`` before the first :meth:`on_outcome` or on the goal, and
        ``ValueError`` if ``obs.pos`` is not the agent's position.
        """

    @abstractmethod
    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]:
        """Positions the current plan expects after doing ``action`` at ``pos`` (D10)."""

    @abstractmethod
    def policy_arrows(self) -> dict[Pos, Action]:
        """State → action arrows of the plan in force (for the UI; a fresh dict)."""

    @abstractmethod
    def _reset_plan(self) -> None:
        """Forget any plan (called by :meth:`reset`)."""

    @abstractmethod
    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        """Replanning logic after an outcome at ``self.pos`` (did a *replan* happen, and why)."""

    # --- shared helpers --------------------------------------------------------------------
    def _sense(self, percept: Percept) -> BeliefUpdate:
        """Write the percept into the belief and count contradictions (D8)."""
        update = self.belief.update(percept.cells)
        if update.contradictions:
            self.stats.contradictions += len(update.contradictions)
        return update

    def _check_unplanned(self, new_pos: Pos) -> bool:
        """D10: was ``new_pos`` outside what the plan anticipated for the last action?"""
        prev, action = self._last_pos, self._last_action
        if prev is None or action is None:
            return False
        if new_pos in self.anticipated(prev, action):
            return False
        self.stats.unplanned_states += 1
        log.log(
            self.unplanned_log_level,
            "%s: unplanned state %s after %s from %s (t=%d)",
            self.name,
            fmt_pos(new_pos),
            action,
            fmt_pos(prev),
            self.t,
        )
        return True

    def _begin_action(self, pos: Pos, action: Action) -> Action:
        """Remember the action being taken (for the unplanned check) and return it."""
        self._last_pos = pos
        self._last_action = action
        return action

    def _check_obs(self, obs: Observation) -> Pos:
        """The observed position, which must be the agent's own and not the goal.

        ``RuntimeError`` before the first :meth:`on_outcome` or on the goal (the mission is
        over: there is nothing to choose); ``ValueError`` on a position mismatch.
        """
        if not self._observed:
            raise RuntimeError("on_outcome() must be called (at t=0) before choose_action()")
        pos = (int(obs.pos[0]), int(obs.pos[1]))
        if pos != self.pos:
            raise ValueError(
                f"observation position {fmt_pos(pos)} != agent position {fmt_pos(self.pos)}"
            )
        if pos == self.goal:
            raise RuntimeError(
                f"{self.name}: choose_action() called on the goal {fmt_pos(pos)}; "
                "the mission is already over"
            )
        return pos

    def _relevant_contradictions(
        self, contradictions: tuple[Pos, ...], plan_cells: frozenset[Pos]
    ) -> tuple[Pos, ...]:
        """Contradictions that demand a replan: all (``eager``) or those in ``plan_cells``."""
        if self.eager:
            return tuple(contradictions)
        return tuple(c for c in contradictions if c in plan_cells)

    def _record_plan(
        self,
        *,
        plan: Plan | None,
        path: tuple[Pos, ...] | None,
        envelope: frozenset[Pos],
        elapsed_ms: float,
        nodes_expanded: int,
        plan_size: int,
        fallback_used: bool = False,
    ) -> bool:
        """Account one planning call and append its snapshot; True if it was a replan (D12)."""
        is_replan = bool(self.snapshots)
        stats = self.stats
        stats.plans_made += 1
        if is_replan:
            stats.replans += 1
        stats.planning_ms += elapsed_ms
        stats.nodes_expanded += nodes_expanded
        stats.plan_size = plan_size
        stats.max_plan_size = max(stats.max_plan_size, plan_size)
        if fallback_used:
            stats.planner_fallbacks += 1
        plan_id = len(self.snapshots)
        self.snapshots.append(
            PlanSnapshot(
                plan_id=plan_id, t=self.t, pos=self.pos, plan=plan, path=path, envelope=envelope
            )
        )
        self.current_plan_id = plan_id
        return is_replan


class PathAgent(Agent):
    """Shared machinery of the A* baselines: one path, followed cell by cell.

    ``_cursor`` indexes the path cell the robot is expected to occupy; it advances after every
    executed action, whatever the outcome, so after a deviation ``path[_cursor]`` is the cell
    the robot *should* be on. ``anticipated(pos, a)`` is the single expected next cell (D10).
    """

    #: Deviations are routine for A* agents, so they are logged at debug level.
    unplanned_log_level: ClassVar[int] = logging.DEBUG

    def _reset_plan(self) -> None:
        self._path: tuple[Pos, ...] | None = None
        self._index: dict[Pos, int] = {}
        self._cursor = 0

    def _plan_path(self) -> bool:
        """Run A* from ``self.pos`` on the optimistic belief; True if it was a replan.

        The f-tie-break is ``planner_config.astar_tie_break`` (docs/plan.md §3.7).
        """
        t0 = time.perf_counter()
        result = astar(self.belief.passable_mask(), self.pos, self.goal,
                       self.planner_config.astar_tie_break)
        elapsed_ms = (time.perf_counter() - t0) * MS_PER_S
        path = tuple(result.path) if result.path is not None else None
        self._path = path
        self._index = {p: i for i, p in enumerate(path)} if path is not None else {}
        self._cursor = 0
        if path is None:
            log.info("%s: A* found no path from %s (t=%d)", self.name, fmt_pos(self.pos), self.t)
        return self._record_plan(
            plan=None,
            path=path,
            envelope=frozenset(self._index),
            elapsed_ms=elapsed_ms,
            nodes_expanded=result.nodes_expanded,
            plan_size=len(path) - 1 if path is not None else 0,
        )

    def _expected_cell(self) -> Pos | None:
        """Path cell the robot should be on now (None without a path)."""
        if self._path is None or self._cursor >= len(self._path):
            return None
        return self._path[self._cursor]

    def _next_cell(self) -> Pos | None:
        """Path cell after the expected one (None at the end or without a path)."""
        if self._path is None or self._cursor + 1 >= len(self._path):
            return None
        return self._path[self._cursor + 1]

    def _next_is_known_debris(self) -> bool:
        nxt = self._next_cell()
        return nxt is not None and self.belief.get(nxt) is Cell.DEBRIS

    def _remaining_path(self) -> frozenset[Pos]:
        """Path cells still to be entered (those after the expected cell)."""
        if self._path is None:
            return frozenset()
        return frozenset(self._path[self._cursor + 1 :])

    def _advance(self) -> None:
        """Called after every executed action: the robot is now expected one cell further."""
        if self._path is not None:
            self._cursor += 1

    def _unreachable(self) -> Abort:
        return Abort("unreachable", f"A* found no path from {fmt_pos(self.pos)}")

    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]:
        """The single expected next cell: the following path cell (or ``pos`` + ``action``)."""
        if self._expected_cell() == pos:
            nxt = self._next_cell()
            if nxt is not None:
                return frozenset((nxt,))
        return frozenset((step_pos(pos, action),))

    def policy_arrows(self) -> dict[Pos, Action]:
        """Arrows along the remaining path (from the expected cell to the goal)."""
        path = self._path
        if path is None:
            return {}
        return {
            path[i]: direction(path[i], path[i + 1]) for i in range(self._cursor, len(path) - 1)
        }
