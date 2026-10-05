"""The mission loop shared by the CLI demo, the simulator UI and the experiment harness.

A :class:`Mission` wires a :class:`~rescue.world.World` to an agent and runs the
plan → act → observe → update belief → replan loop one step at a time (docs/plan.md §3.10).
Every step produces a frozen :class:`StepEvent`; the UI only renders these events, and
:class:`MissionMetrics` summarises a finished (or running) mission.

Text output: :func:`format_trace` renders the deck's execution trace
(``(1,1) E ok → (1,2) S slip → … → (4,4)``), :func:`format_summary` the two-line summary and
:func:`format_metrics_table` an aligned ``key: value`` table for CLIs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import numpy as np

from rescue.agents.base import Abort, Agent, Observation
from rescue.belief import BeliefMap
from rescue.config import PROB_EPS, WorldConfig
from rescue.grid import Cell, Pos, TrueGrid, fmt_pos
from rescue.model import Action, OutcomeKind, TerrainSlip, TransitionModel, UniformSlip
from rescue.planners.bfs import shortest_path_length
from rescue.world import World

log = logging.getLogger(__name__)

#: Separator between trace segments (and around the ellipsis of a shortened trace).
TRACE_SEPARATOR = " → "
#: Placeholder for the segments dropped from a shortened trace.
TRACE_ELLIPSIS = "…"
#: Trace label of each outcome kind.
TRACE_LABELS: dict[OutcomeKind, str] = {
    OutcomeKind.ARRIVED: "ok",
    OutcomeKind.SLIPPED: "slip",
    OutcomeKind.SKID_L: "skid-L",
    OutcomeKind.SKID_R: "skid-R",
}
#: Suffix of a trace segment whose outcome was blocked (wall or debris).
TRACE_BLOCKED_SUFFIX = " (blocked)"
#: Spacing between the fields of the summary line.
SUMMARY_GAP = "   "


class MissionStatus(Enum):
    """Lifecycle state of a mission; everything but ``RUNNING`` is terminal."""

    RUNNING = "running"
    RESCUED = "rescued"
    ABORTED = "aborted"
    BUDGET = "budget"
    UNREACHABLE = "unreachable"

    def __str__(self) -> str:
        return self.name


#: Failure reason recorded in the metrics for each terminal failure status.
FAILURE_REASONS: dict[MissionStatus, str] = {
    MissionStatus.ABORTED: "aborted",
    MissionStatus.UNREACHABLE: "unreachable",
    MissionStatus.BUDGET: "budget",
}
#: Mission status for each ``Abort.reason`` an agent can return.
_ABORT_STATUS: dict[str, MissionStatus] = {
    "aborted": MissionStatus.ABORTED,
    "unreachable": MissionStatus.UNREACHABLE,
}


@dataclass(frozen=True)
class StepEvent:
    """Everything the outside world needs to know about one mission step.

    The t = 0 event (and the event ending a mission on an agent abort) has ``action`` and
    ``outcome`` set to None and ``pos == new_pos``. ``t`` is the world time after the step.
    ``bump`` is an unexpected collision (D6); ``blocked`` is True for every outcome that tried
    to move but stayed, bumps included. ``new_cells`` are the cells that were UNKNOWN in the
    belief before this percept, with their observed values. ``triggering_contradictions`` are
    the contradicted cells that demanded this step's replan (``AgentUpdate``; empty without a
    replan or when it had other causes only). ``note`` is a short human-readable description
    for event logs (empty when nothing notable happened).
    """

    t: int
    pos: Pos
    action: Action | None
    outcome: OutcomeKind | None
    new_pos: Pos
    blocked: bool
    bump: bool
    new_cells: tuple[tuple[Pos, Cell], ...]
    contradictions: tuple[Pos, ...]
    triggering_contradictions: tuple[Pos, ...]
    replanned: bool
    unplanned: bool
    plan_id: int
    aftershock_cells: tuple[Pos, ...]
    status: MissionStatus
    note: str = ""


@dataclass(frozen=True)
class MissionMetrics:
    """Per-mission metrics (one row of the experiment CSV).

    ``failure_reason`` is None on success, else ``aborted`` / ``unreachable`` / ``budget``
    (None as well while the mission is still running). ``wasted_moves`` is ``steps −
    shortest_path`` for successful missions and NaN otherwise (D16); ``shortest_path`` is the
    BFS length on the *initial* true map (None if the survivor was unreachable there).
    ``slips_handled`` (D11) is ``slips + skids`` minus a final non-arrived outcome the agent
    answered by giving up (``Abort("aborted")``, e.g. ``astar_fixed``'s first deviation).
    """

    agent: str
    scenario: str
    seed: int
    rows: int
    cols: int
    slip_p: float
    success: bool
    failure_reason: str | None
    steps: int
    shortest_path: int | None
    wasted_moves: float
    replans: int
    slips: int
    skids: int
    slips_handled: int
    bumps: int
    blocked_moves: int
    unplanned_states: int
    cells_explored: int
    planning_ms: float
    nodes_expanded: int
    plan_size: int
    planner_fallbacks: int
    aftershock_events: int

    def as_dict(self) -> dict[str, Any]:
        """Field name → value, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionMetrics:
        """Inverse of :meth:`as_dict` after a JSON round trip (``None`` floats → NaN).

        Unknown or missing keys raise ``ValueError``.
        """
        names = [f.name for f in fields(cls)]
        unknown = set(data) - set(names)
        missing = [name for name in names if name not in data]
        if unknown or missing:
            raise ValueError(
                f"bad MissionMetrics keys: unknown {sorted(unknown)}, missing {missing}"
            )
        values = dict(data)
        for f in fields(cls):
            if f.type in ("float", float) and values[f.name] is None:
                values[f.name] = math.nan
        return cls(**values)


def make_transition(grid: TrueGrid, world_config: WorldConfig | None = None) -> TransitionModel:
    """Transition model for ``grid``: terrain-dependent slip iff the map has '~' cells."""
    cfg = world_config if world_config is not None else WorldConfig()
    high = np.asarray(grid.high_slip, dtype=bool)
    if high.any():
        return TerrainSlip(cfg.slip_p, cfg.terrain_high_p, high)
    return UniformSlip(cfg.slip_p)


def _replan_note(contradictions: Sequence[Pos]) -> str:
    """Event-log text for a replan, e.g. ``debris at (3,2), replanned``."""
    if contradictions:
        return f"debris at {', '.join(fmt_pos(p) for p in contradictions)}, replanned"
    return "replanned"


def check_agent_transition(agent: Agent, world: TransitionModel, rows: int, cols: int) -> None:
    """D5: the agent must be given the world's dynamics (``ValueError`` otherwise).

    Compares the slip parameter of every cell (so the outcome supports agree too); a mismatch
    would silently break the andor invariants (``unplanned_states == 0``).
    """
    mine = agent.transition
    for r in range(rows):
        for c in range(cols):
            p_agent, p_world = mine.slip_p((r, c)), world.slip_p((r, c))
            if abs(p_agent - p_world) > PROB_EPS:
                raise ValueError(
                    f"agent {agent.name!r} was built with {mine!r} but the world uses "
                    f"{world!r}: slip p at {fmt_pos((r, c))} is {p_agent:g} vs {p_world:g} "
                    "(build the agent with make_transition(grid, world_config), D5)"
                )


class Mission:
    """One rescue mission: a world, an agent and the events produced so far.

    Call :meth:`start` once (it emits the t = 0 event), then :meth:`step` until :attr:`done`.
    ``ValueError`` if the agent's transition model differs from the world's (D5).
    """

    def __init__(
        self,
        grid: TrueGrid,
        agent: Agent,
        world_config: WorldConfig | None = None,
        seed: int = 0,
        scenario: str = "",
    ) -> None:
        self.grid: TrueGrid = grid.copy()  # the initial true map (the world mutates its own)
        self.agent = agent
        self.world_config: WorldConfig = (
            world_config if world_config is not None else WorldConfig()
        )
        self.seed = int(seed)
        self.scenario = scenario
        self.transition: TransitionModel = make_transition(self.grid, self.world_config)
        check_agent_transition(agent, self.transition, self.grid.rows, self.grid.cols)
        self.budget: int = self.world_config.step_budget(self.grid.rows, self.grid.cols)
        self.events: list[StepEvent] = []
        self.status: MissionStatus = MissionStatus.RUNNING
        self._world: World | None = None
        self._slips = 0
        self._skids = 0
        #: 1 if the mission ended by an ``aborted`` decision right after a slip / skid (D11).
        self._unhandled_slips = 0
        self._bumps = 0
        self._blocked = 0
        self._shortest: int | None = shortest_path_length(
            self.grid.passable_mask(), self.grid.start, self.grid.survivor
        )

    # --- state -----------------------------------------------------------------------------
    @property
    def started(self) -> bool:
        """True once :meth:`start` has run."""
        return self._world is not None

    @property
    def done(self) -> bool:
        """True once the mission reached a terminal status."""
        return self.status is not MissionStatus.RUNNING

    @property
    def world(self) -> World:
        """The simulated true world (available after :meth:`start`)."""
        if self._world is None:
            raise RuntimeError("mission not started; call start() first")
        return self._world

    @property
    def shortest_path(self) -> int | None:
        """BFS shortest-path length start → survivor on the initial true map."""
        return self._shortest

    # --- loop ------------------------------------------------------------------------------
    def start(self) -> StepEvent:
        """Build the world and belief, let the agent sense at t = 0 and plan; emit event 0.

        Calling it again returns the existing t = 0 event without restarting anything.
        """
        if self._world is not None:
            return self.events[0]
        cfg = self.world_config
        world = World(
            self.grid,
            self.transition,
            sensor_radius=cfg.sensor_radius,
            seed=self.seed,
            aftershock=cfg.aftershock,
        )
        self._world = world
        belief = BeliefMap(self.grid.rows, self.grid.cols, self.grid.survivor)
        self.agent.reset(self.grid.start, self.grid.survivor, belief)
        percept = world.sense()
        update = self.agent.on_outcome(world.pos, percept)
        new_cells = self._new_cells(update.newly_known, percept.as_dict())
        note = _replan_note(update.triggering_contradictions) if update.replanned else ""
        event = StepEvent(
            t=world.t,
            pos=world.pos,
            action=None,
            outcome=None,
            new_pos=world.pos,
            blocked=False,
            bump=False,
            new_cells=new_cells,
            contradictions=tuple(update.contradictions),
            triggering_contradictions=tuple(update.triggering_contradictions),
            replanned=bool(update.replanned),
            unplanned=bool(update.unplanned),
            plan_id=self.agent.current_plan_id,
            aftershock_cells=(),
            status=self.status,
            note=note,
        )
        self.events.append(event)
        log.debug("mission start: agent=%s seed=%d plan_id=%d", self.agent.name, self.seed,
                  event.plan_id)
        return event

    def step(self) -> StepEvent:
        """Ask the agent for a decision and execute it; returns the new event.

        Raises ``RuntimeError`` before :meth:`start` or once the mission is done.
        """
        world = self.world
        if self.done:
            raise RuntimeError(f"mission already finished ({self.status.name})")
        decision = self.agent.choose_action(Observation(pos=world.pos, t=world.t))
        if isinstance(decision, Abort):
            return self._finish_abort(decision)
        if not isinstance(decision, Action):
            raise TypeError(f"agent returned an unknown decision: {decision!r}")

        result = world.step(decision)
        # D6: a bump is judged on the belief the agent held when it acted.
        bump = bool(
            result.hit_debris and self.agent.belief.get(result.target) is not Cell.DEBRIS
        )
        update = self.agent.on_outcome(result.new_pos, result.percept)

        if result.kind is OutcomeKind.SLIPPED:
            self._slips += 1
        elif result.kind in (OutcomeKind.SKID_L, OutcomeKind.SKID_R):
            self._skids += 1
        if bump:
            self._bumps += 1
            log.warning("bump at %s (t=%d)", fmt_pos(result.target), world.t)
        elif result.blocked:
            self._blocked += 1

        if result.new_pos == self.grid.survivor:
            self.status = MissionStatus.RESCUED
        elif world.t >= self.budget:
            self.status = MissionStatus.BUDGET

        notes: list[str] = []
        if bump:
            notes.append(f"bumped into debris at {fmt_pos(result.target)}")
        if update.replanned:
            notes.append(_replan_note(update.triggering_contradictions))
        if self.status is MissionStatus.BUDGET:
            notes.append(f"step budget {self.budget} exhausted")
        event = StepEvent(
            t=world.t,
            pos=result.prev,
            action=result.action,
            outcome=result.kind,
            new_pos=result.new_pos,
            blocked=result.blocked,
            bump=bump,
            new_cells=self._new_cells(update.newly_known, result.percept.as_dict()),
            contradictions=tuple(update.contradictions),
            triggering_contradictions=tuple(update.triggering_contradictions),
            replanned=bool(update.replanned),
            unplanned=bool(update.unplanned),
            plan_id=self.agent.current_plan_id,
            aftershock_cells=tuple(result.aftershock_cells),
            status=self.status,
            note="; ".join(notes),
        )
        self.events.append(event)
        if self.done:
            log.info("mission %s after %d steps (agent=%s, seed=%d)", self.status.name,
                     world.t, self.agent.name, self.seed)
        return event

    def run(self, on_event: Callable[[StepEvent], None] | None = None) -> None:
        """Start (if needed) and step until done, calling ``on_event`` for every new event."""
        if not self.started:
            event = self.start()
            if on_event is not None:
                on_event(event)
        while not self.done:
            event = self.step()
            if on_event is not None:
                on_event(event)

    def _finish_abort(self, decision: Abort) -> StepEvent:
        """End the mission on an agent ``Abort`` decision."""
        world = self.world
        reason, detail = str(decision.reason), str(decision.detail)
        status = _ABORT_STATUS.get(reason)
        if status is None:
            raise ValueError(f"unknown abort reason {reason!r} in {decision!r}")
        last = self.events[-1].outcome if self.events else None
        if status is MissionStatus.ABORTED and last not in (None, OutcomeKind.ARRIVED):
            self._unhandled_slips = 1  # D11: the agent gave up on this slip / skid
        self.status = status
        log.info("mission %s at t=%d: %s", status.name, world.t, detail)
        event = StepEvent(
            t=world.t,
            pos=world.pos,
            action=None,
            outcome=None,
            new_pos=world.pos,
            blocked=False,
            bump=False,
            new_cells=(),
            contradictions=(),
            triggering_contradictions=(),
            replanned=False,
            unplanned=False,
            plan_id=self.agent.current_plan_id,
            aftershock_cells=(),
            status=status,
            note=f"{reason}: {detail}" if detail else reason,
        )
        self.events.append(event)
        return event

    @staticmethod
    def _new_cells(
        newly_known: Sequence[Pos], observed: dict[Pos, Cell]
    ) -> tuple[tuple[Pos, Cell], ...]:
        """``(pos, cell)`` pairs for the cells the agent learned in this percept."""
        return tuple((p, Cell(observed[p])) for p in newly_known if p in observed)

    # --- metrics ---------------------------------------------------------------------------
    def metrics(self) -> MissionMetrics:
        """Metrics of the mission so far (final once :attr:`done`)."""
        world = self.world
        stats = self.agent.stats
        success = self.status is MissionStatus.RESCUED
        steps = world.t
        wasted = (
            float(steps - self._shortest)
            if success and self._shortest is not None
            else math.nan
        )
        return MissionMetrics(
            agent=self.agent.name,
            scenario=self.scenario,
            seed=self.seed,
            rows=self.grid.rows,
            cols=self.grid.cols,
            slip_p=self.world_config.slip_p,
            success=success,
            failure_reason=FAILURE_REASONS.get(self.status),
            steps=steps,
            shortest_path=self._shortest,
            wasted_moves=wasted,
            replans=int(stats.replans),
            slips=self._slips,
            skids=self._skids,
            slips_handled=self._slips + self._skids - self._unhandled_slips,
            bumps=self._bumps,
            blocked_moves=self._blocked,
            unplanned_states=int(stats.unplanned_states),
            cells_explored=self.agent.belief.explored_count(),
            planning_ms=float(stats.planning_ms),
            nodes_expanded=int(stats.nodes_expanded),
            plan_size=int(stats.plan_size),
            planner_fallbacks=int(stats.planner_fallbacks),
            aftershock_events=world.aftershock_events,
        )


def run_mission(
    grid: TrueGrid,
    agent: Agent,
    world_config: WorldConfig | None = None,
    seed: int = 0,
    scenario: str = "",
    on_event: Callable[[StepEvent], None] | None = None,
) -> Mission:
    """Run a mission to completion and return it (events, metrics, world, agent)."""
    mission = Mission(grid, agent, world_config=world_config, seed=seed, scenario=scenario)
    mission.run(on_event)
    return mission


# --- text output ---------------------------------------------------------------------------
def _trace_note(event: StepEvent) -> str:
    """Annotation of the position an event ends on (docs/plan.md §3.9).

    Only a replan triggered by contradictions is annotated, as
    ``" [debris at (r,c), …, replanned]"`` listing the triggering cells
    (``event.triggering_contradictions``). Replans after a mere deviation (A* + replan) and
    contradictions that did not demand a replan (lazy mode: off the envelope / remaining path)
    leave the position bare: the outcome label already shows the deviation.
    """
    if not (event.replanned and event.triggering_contradictions):
        return ""
    cells = ", ".join(fmt_pos(p) for p in event.triggering_contradictions)
    return f" [debris at {cells}, replanned]"


def _trace_segments(events: Sequence[StepEvent]) -> list[str]:
    """Trace segments, the final position included as the last segment."""
    if not events:
        return []
    segments: list[str] = []
    note = ""
    last_pos = events[0].new_pos
    for event in events:
        if event.action is None or event.outcome is None:
            # t = 0 or abort event: no move, but a replan here annotates the position.
            note = note or _trace_note(event)
            last_pos = event.new_pos
            continue
        label = TRACE_LABELS[event.outcome]
        if event.blocked:
            label += TRACE_BLOCKED_SUFFIX
        segments.append(f"{fmt_pos(event.pos)}{note} {event.action} {label}")
        note = _trace_note(event)
        last_pos = event.new_pos
    segments.append(f"{fmt_pos(last_pos)}{note}")
    return segments


def format_trace(events: Sequence[StepEvent], max_segments: int | None = None) -> str:
    """Deck-style execution trace, e.g. ``(1,1) E ok → (1,2) S slip → … → (4,4)``.

    Each move is ``"{pos}{note} {action} {label}"`` (labels ``ok``/``slip``/``skid-L``/
    ``skid-R``, plus ``" (blocked)"`` when the robot stayed), where ``note`` is
    ``" [debris at (r,c), replanned]"`` on a position where contradictions triggered a replan
    (listing the triggering cells; other replans are not annotated). The final position ends
    the trace. With ``max_segments`` (final position included), longer traces keep their first
    and last segments around ``" → … → "``.
    """
    segments = _trace_segments(events)
    if max_segments is not None and max_segments > 0 and len(segments) > max_segments:
        tail = max(1, max_segments // 2)
        head = max(1, max_segments - tail)
        segments = [*segments[:head], TRACE_ELLIPSIS, *segments[-tail:]]
    return TRACE_SEPARATOR.join(segments)


def format_summary(metrics: MissionMetrics) -> str:
    """Two-line deck summary: counters, then ``Rescue successful`` / ``Rescue failed (…)``."""
    counters = SUMMARY_GAP.join(
        (
            f"Cells explored: {metrics.cells_explored}",
            f"Replans: {metrics.replans}",
            f"Slips handled: {metrics.slips_handled}",
            f"Collisions: {metrics.bumps}",
        )
    )
    if metrics.success:
        verdict = "Rescue successful"
    elif metrics.failure_reason is None:
        verdict = "Rescue in progress"
    else:
        verdict = f"Rescue failed ({metrics.failure_reason})"
    return f"{counters}\n{verdict}"


def _format_value(value: Any) -> str:
    """Human-readable metric value (NaN → ``n/a``, floats with two decimals)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.2f}"
    return str(value)


def format_metrics_table(metrics: MissionMetrics) -> str:
    """All metrics as aligned ``key: value`` lines (for CLIs)."""
    items = metrics.as_dict()
    width = max(len(key) for key in items)
    return "\n".join(f"{key + ':':<{width + 1}} {_format_value(v)}" for key, v in items.items())
