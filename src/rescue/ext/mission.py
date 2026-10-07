"""The extension mission loop: several survivors, unknown locations and noisy sensing
(brief §8.3–§8.5, docs/plan.md §3.14).

:class:`ExtMission` mirrors :class:`rescue.runner.Mission` but drives a
:class:`~rescue.ext.agents.SearchAgent`. The core :class:`~rescue.world.World` simulates motion
and aftershocks (so the outcome stream is the core's); every percept passes through a
:class:`~rescue.ext.sensing.NoisySensor` (always, even without noise, so the noise draws line
up across conditions); a bump's touched cell and the robot's own cell are exempt from the noise
(both are known for certain: contact, and the cell the robot stands on), and both are passed to
the agent's belief as certain (``SearchAgent.on_outcome(..., touched)``). When the robot stands
on an unrescued survivor that survivor is rescued. The mission ends ``RESCUED``
when all ``k`` survivors are rescued, ``BUDGET`` after ``world_config.step_budget(rows, cols)
* k`` steps, or on an agent ``Abort``.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

from rescue.agents.base import Abort, Observation
from rescue.belief import BeliefMap
from rescue.config import (
    EXT_DEBRIS_THRESHOLD,
    EXT_MAX_BRUTE_FORCE_K,
    EXT_PRIOR_DEBRIS,
    WorldConfig,
)
from rescue.ext.agents import AnyBelief, SearchAgent, SearchMode
from rescue.ext.sensing import NoisySensor, ProbBelief
from rescue.grid import Cell, Pos, TrueGrid, fmt_pos
from rescue.model import Action, OutcomeKind
from rescue.planners.bfs import UNREACHABLE, bfs_distances
from rescue.runner import (
    FAILURE_REASONS,
    TRACE_BLOCKED_SUFFIX,
    TRACE_ELLIPSIS,
    TRACE_LABELS,
    TRACE_SEPARATOR,
    MissionStatus,
    check_agent_transition,
    make_transition,
)
from rescue.world import World

log = logging.getLogger(__name__)

#: Mission status for each ``Abort.reason`` an agent can return (as in the core runner).
ABORT_STATUS: dict[str, MissionStatus] = {
    "aborted": MissionStatus.ABORTED,
    "unreachable": MissionStatus.UNREACHABLE,
}

__all__ = [
    "ExtMetrics",
    "ExtMission",
    "ExtStepEvent",
    "format_ext_trace",
    "oracle_tours",
    "run_ext_mission",
]


@dataclass(frozen=True)
class ExtStepEvent:
    """One step of an extension mission (the t = 0 / abort events have no action).

    ``target`` and ``mode`` are the agent's target and mode *after* the step (``None`` target
    when it has none). ``rescued_now`` is the survivor rescued on this step, ``found_now`` the
    survivors first sensed on it (row-major). ``triggering_contradictions`` are the contradicted
    cells that demanded this step's replan (as in the core ``StepEvent``).
    """

    t: int
    pos: Pos
    action: Action | None
    outcome: OutcomeKind | None
    new_pos: Pos
    blocked: bool
    bump: bool
    contradictions: tuple[Pos, ...]
    replanned: bool
    unplanned: bool
    target: Pos | None
    mode: SearchMode
    rescued_now: Pos | None
    found_now: tuple[Pos, ...]
    status: MissionStatus
    triggering_contradictions: tuple[Pos, ...] = ()
    aftershock_cells: tuple[Pos, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ExtMetrics:
    """Per-mission metrics of an extension mission (one row of ``results/ext/raw.csv``).

    ``mode`` is the caller's block label (``search`` / ``multi`` / ``noisy`` / ``noisy_fp`` or
    empty). ``false_positive`` is the sensor's false-positive rate and ``fallback_replans`` the
    agent's :attr:`~rescue.ext.agents.SearchAgent.fallback_replans` (round 2, appended).
    ``first_detection_step`` is the step at which the first survivor was sensed (0 when the
    locations are known; NaN if none was ever found). ``oracle_path`` is the true-map shortest
    path start → survivor (k = 1; NaN otherwise); ``greedy_tour`` the true-map BFS length of
    the greedy nearest-first visiting order from the start (ties row-major) and
    ``optimal_tour`` the shortest over all orders (k ≤ ``EXT_MAX_BRUTE_FORCE_K``; NaN
    otherwise). All three use the *initial* true map; NaN if a survivor is unreachable there.
    """

    agent: str
    mode: str
    scenario: str
    seed: int
    rows: int
    cols: int
    slip_p: float
    k: int
    survivors_known: bool
    false_negative: float
    success: bool
    failure_reason: str | None
    steps: int
    rescued: int
    first_detection_step: float
    cells_explored: int
    replans: int
    slips: int
    skids: int
    bumps: int
    unplanned_states: int
    planning_ms: float
    nodes_expanded: int
    oracle_path: float
    greedy_tour: float
    optimal_tour: float
    blocked_moves: int = 0
    aftershock_events: int = 0
    false_positive: float = 0.0
    fallback_replans: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Field name → value, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def oracle_tours(grid: TrueGrid) -> tuple[float, float, float]:
    """``(oracle_path, greedy_tour, optimal_tour)`` on the true map (see :class:`ExtMetrics`)."""
    survivors = list(grid.survivors)
    k = len(survivors)
    passable = grid.passable_mask()
    from_survivor = {s: bfs_distances(passable, s) for s in survivors}

    def d(a: Pos, b: Pos) -> int | None:
        value = int(from_survivor[b][a])
        return None if value == UNREACHABLE else value

    if any(d(grid.start, s) is None for s in survivors):
        return math.nan, math.nan, math.nan
    oracle = float(d(grid.start, survivors[0]) or 0) if k == 1 else math.nan
    left, pos, greedy = set(survivors), grid.start, 0
    while left:
        nxt = min(left, key=lambda s: (d(pos, s), s))
        greedy += d(pos, nxt) or 0
        left.remove(nxt)
        pos = nxt
    optimal = math.nan
    if k <= EXT_MAX_BRUTE_FORCE_K:
        best: int | None = None
        for order in itertools.permutations(survivors):
            total, here = 0, grid.start
            for s in order:
                total += d(here, s) or 0
                here = s
                if best is not None and total >= best:
                    break
            else:
                best = total if best is None else min(best, total)
        optimal = float(best) if best is not None else math.nan
    return oracle, float(greedy), optimal


class ExtMission:
    """One extension mission: a world, a noisy sensor, a :class:`SearchAgent` and its events.

    ``survivors_known`` pre-marks every survivor in the belief (the core setting); otherwise the
    belief starts all UNKNOWN and survivors are found by sensing. The belief is a
    :class:`~rescue.belief.BeliefMap` when ``false_negative == 0`` and ``false_positive == 0``,
    and a :class:`~rescue.ext.sensing.ProbBelief` (``prior_debris``, ``threshold``, both noise
    rates) otherwise; ``prob_belief`` forces one or the other. With aftershocks enabled and
    ``false_positive > 0`` the ``ProbBelief`` gets ``debris_floor = prior_debris``
    (:attr:`debris_floor`): a DEBRIS reading of a cell believed free counts as much as one of a
    never-seen cell, because an aftershock may have buried it since (see
    :class:`~rescue.ext.sensing.ProbBelief`); otherwise the floor is 0 (the static-map filter).

    ``ValueError`` if the agent's dynamics differ from the world's (D5); if ``prior_debris >=
    threshold`` (every never-observed cell would count as impassable, so frontier search could
    never step into unknown territory and is no longer complete); or if ``prob_belief=False``
    is combined with ``false_negative > 0`` or ``false_positive > 0`` (a
    :class:`~rescue.belief.BeliefMap` takes a noisy reading at face value: it can "un-see" known
    debris, which the termination argument of :class:`~rescue.ext.agents.SearchAgent` excludes,
    or hold a phantom wall as certain). ``false_positive`` must be in ``[0,
    EXT_MAX_FALSE_POSITIVE)``.
    """

    def __init__(
        self,
        grid: TrueGrid,
        agent: SearchAgent,
        world_config: WorldConfig | None = None,
        seed: int = 0,
        *,
        survivors_known: bool = True,
        false_negative: float = 0.0,
        false_positive: float = 0.0,
        prior_debris: float = EXT_PRIOR_DEBRIS,
        threshold: float = EXT_DEBRIS_THRESHOLD,
        scenario: str = "",
        mode: str = "",
        prob_belief: bool | None = None,
    ) -> None:
        if not isinstance(agent, SearchAgent):
            raise TypeError(f"ExtMission needs a SearchAgent, got {type(agent).__name__}")
        self.grid: TrueGrid = grid.copy()
        self.agent = agent
        self.world_config: WorldConfig = (
            world_config if world_config is not None else WorldConfig()
        )
        self.seed = int(seed)
        self.survivors_known = bool(survivors_known)
        self.false_negative = float(false_negative)
        self.false_positive = float(false_positive)
        self.prior_debris = float(prior_debris)
        self.threshold = float(threshold)
        self.scenario = scenario or grid.name
        self.mode = mode
        noisy = self.false_negative > 0.0 or self.false_positive > 0.0
        self.prob_belief = noisy if prob_belief is None else bool(prob_belief)
        if self.prior_debris >= self.threshold:
            raise ValueError(
                f"prior_debris ({self.prior_debris:g}) must be below threshold "
                f"({self.threshold:g}): unknown cells must stay optimistically passable"
            )
        if not self.prob_belief and noisy:
            raise ValueError(
                f"prob_belief=False needs false_negative = 0 and false_positive = 0 (got "
                f"{self.false_negative:g}, {self.false_positive:g}): a BeliefMap would take a "
                "noisy reading at face value"
            )
        #: Floor of the debris probability a DEBRIS reading starts from (ProbBelief only).
        self.debris_floor: float = (
            self.prior_debris
            if self.world_config.aftershock.enabled and self.false_positive > 0.0 else 0.0
        )
        self.k: int = len(self.grid.survivors)
        self.transition = make_transition(self.grid, self.world_config)
        check_agent_transition(agent, self.transition, self.grid.rows, self.grid.cols)
        self.budget: int = self.world_config.step_budget(self.grid.rows, self.grid.cols) * self.k
        self.sensor = NoisySensor(self.false_negative, self.seed, self.grid.rows,
                                  self.grid.cols, self.false_positive)
        self.events: list[ExtStepEvent] = []
        self.status: MissionStatus = MissionStatus.RUNNING
        self.rescued: list[Pos] = []
        self.first_detection_step: int | None = 0 if self.survivors_known else None
        self._found: set[Pos] = set(self.grid.survivors) if self.survivors_known else set()
        self._world: World | None = None
        self._slips = 0
        self._skids = 0
        self._bumps = 0
        self._blocked = 0
        self._oracle = oracle_tours(self.grid)

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

    def _make_belief(self) -> AnyBelief:
        rows, cols = self.grid.rows, self.grid.cols
        primary = self.grid.survivor if self.survivors_known else None
        belief: AnyBelief
        if self.prob_belief:
            belief = ProbBelief(rows, cols, primary, self.prior_debris, self.false_negative,
                                self.threshold, self.false_positive, self.debris_floor)
            if self.survivors_known:
                for s in self.grid.extra_survivors:
                    belief.mark_survivor(s)
        else:
            belief = BeliefMap(rows, cols, primary)
            if self.survivors_known:
                for s in self.grid.extra_survivors:
                    belief.grid[s] = Cell.SURVIVOR
        return belief

    def _found_now(self, newly_known: Sequence[Pos]) -> tuple[Pos, ...]:
        """Survivors sensed for the first time in this update (row-major)."""
        belief = self.agent.belief
        found = sorted(p for p in newly_known
                       if p not in self._found and belief.get(p) is Cell.SURVIVOR)
        self._found.update(found)
        if found and self.first_detection_step is None:
            self.first_detection_step = self.world.t
        return tuple(found)

    # --- loop ------------------------------------------------------------------------------
    def start(self) -> ExtStepEvent:
        """Build the world and belief, sense at t = 0 and let the agent plan; emit event 0."""
        if self._world is not None:
            return self.events[0]
        cfg = self.world_config
        world = World(self.grid, self.transition, sensor_radius=cfg.sensor_radius,
                      seed=self.seed, aftershock=cfg.aftershock)
        self._world = world
        self.agent.reset_search(self.grid.start, self._make_belief(), self.k)
        percept = self.sensor.observe(world.sense(), (world.pos,))
        update = self.agent.on_outcome(world.pos, percept)
        event = ExtStepEvent(
            t=world.t,
            pos=world.pos,
            action=None,
            outcome=None,
            new_pos=world.pos,
            blocked=False,
            bump=False,
            contradictions=tuple(update.contradictions),
            replanned=bool(update.replanned),
            unplanned=bool(update.unplanned),
            target=self.agent.target,
            mode=self.agent.mode,
            rescued_now=None,
            found_now=self._found_now(update.newly_known),
            status=self.status,
            triggering_contradictions=tuple(update.triggering_contradictions),
        )
        self.events.append(event)
        return event

    def step(self) -> ExtStepEvent:
        """Ask the agent for a decision and execute it; returns the new event."""
        world = self.world
        if self.done:
            raise RuntimeError(f"mission already finished ({self.status.name})")
        agent = self.agent
        decision = agent.choose_action(Observation(pos=world.pos, t=world.t))
        if isinstance(decision, Abort):
            return self._finish_abort(decision)
        if not isinstance(decision, Action):
            raise TypeError(f"agent returned an unknown decision: {decision!r}")
        result = world.step(decision)
        # D6: a bump is judged on the belief the agent held when it acted.
        bump = bool(result.hit_debris and agent.belief.get(result.target) is not Cell.DEBRIS)
        touched = (result.target,) if result.hit_debris else ()
        percept = self.sensor.observe(result.percept, (*touched, result.new_pos))
        rescued_now: Pos | None = None
        on = world.on_survivor
        if on is not None and on not in self.rescued:
            self.rescued.append(on)
            rescued_now = on
            agent.on_rescue(on)
        update = agent.on_outcome(result.new_pos, percept, touched)

        if result.kind is OutcomeKind.SLIPPED:
            self._slips += 1
        elif result.kind in (OutcomeKind.SKID_L, OutcomeKind.SKID_R):
            self._skids += 1
        if bump:
            self._bumps += 1
        elif result.blocked:
            self._blocked += 1
        if len(self.rescued) >= self.k:
            self.status = MissionStatus.RESCUED
        elif world.t >= self.budget:
            self.status = MissionStatus.BUDGET

        notes: list[str] = []
        if bump:
            notes.append(f"bumped into debris at {fmt_pos(result.target)}")
        if rescued_now is not None:
            notes.append(f"rescued {fmt_pos(rescued_now)}")
        if self.status is MissionStatus.BUDGET:
            notes.append(f"step budget {self.budget} exhausted")
        event = ExtStepEvent(
            t=world.t,
            pos=result.prev,
            action=result.action,
            outcome=result.kind,
            new_pos=result.new_pos,
            blocked=result.blocked,
            bump=bump,
            contradictions=tuple(update.contradictions),
            replanned=bool(update.replanned),
            unplanned=bool(update.unplanned),
            target=agent.target,
            mode=agent.mode,
            rescued_now=rescued_now,
            found_now=self._found_now(update.newly_known),
            status=self.status,
            triggering_contradictions=tuple(update.triggering_contradictions),
            aftershock_cells=tuple(result.aftershock_cells),
            note="; ".join(notes),
        )
        self.events.append(event)
        return event

    def run(self, on_event: Callable[[ExtStepEvent], None] | None = None) -> None:
        """Start (if needed) and step until done, calling ``on_event`` for every new event."""
        if not self.started:
            event = self.start()
            if on_event is not None:
                on_event(event)
        while not self.done:
            event = self.step()
            if on_event is not None:
                on_event(event)

    def _finish_abort(self, decision: Abort) -> ExtStepEvent:
        world = self.world
        status = ABORT_STATUS.get(str(decision.reason))
        if status is None:
            raise ValueError(f"unknown abort reason {decision.reason!r} in {decision!r}")
        self.status = status
        log.info("ext mission %s at t=%d: %s", status.name, world.t, decision.detail)
        event = ExtStepEvent(
            t=world.t,
            pos=world.pos,
            action=None,
            outcome=None,
            new_pos=world.pos,
            blocked=False,
            bump=False,
            contradictions=(),
            replanned=False,
            unplanned=False,
            target=self.agent.target,
            mode=self.agent.mode,
            rescued_now=None,
            found_now=(),
            status=status,
            note=f"{decision.reason}: {decision.detail}",
        )
        self.events.append(event)
        return event

    # --- metrics ---------------------------------------------------------------------------
    def metrics(self) -> ExtMetrics:
        """Metrics of the mission so far (final once :attr:`done`)."""
        world = self.world
        stats = self.agent.stats
        oracle, greedy, optimal = self._oracle
        first = self.first_detection_step
        return ExtMetrics(
            agent=self.agent.name,
            mode=self.mode,
            scenario=self.scenario,
            seed=self.seed,
            rows=self.grid.rows,
            cols=self.grid.cols,
            slip_p=self.world_config.slip_p,
            k=self.k,
            survivors_known=self.survivors_known,
            false_negative=self.false_negative,
            success=self.status is MissionStatus.RESCUED,
            failure_reason=FAILURE_REASONS.get(self.status),
            steps=world.t,
            rescued=len(self.rescued),
            first_detection_step=math.nan if first is None else float(first),
            cells_explored=self.agent.belief.explored_count(),
            replans=int(stats.replans),
            slips=self._slips,
            skids=self._skids,
            bumps=self._bumps,
            unplanned_states=int(stats.unplanned_states),
            planning_ms=float(stats.planning_ms),
            nodes_expanded=int(stats.nodes_expanded),
            oracle_path=oracle,
            greedy_tour=greedy,
            optimal_tour=optimal,
            blocked_moves=self._blocked,
            aftershock_events=world.aftershock_events,
            false_positive=self.false_positive,
            fallback_replans=int(self.agent.fallback_replans),
        )


def run_ext_mission(
    grid: TrueGrid,
    agent: SearchAgent,
    world_config: WorldConfig | None = None,
    seed: int = 0,
    *,
    on_event: Callable[[ExtStepEvent], None] | None = None,
    **kwargs: Any,
) -> ExtMission:
    """Run an extension mission to completion and return it (``kwargs`` → :class:`ExtMission`)."""
    mission = ExtMission(grid, agent, world_config, seed, **kwargs)
    mission.run(on_event)
    return mission


# --- text output ---------------------------------------------------------------------------
def _ext_notes(event: ExtStepEvent) -> str:
    """Annotations of the position an event ends on: rescue, sightings, contradiction replan."""
    notes: list[str] = []
    if event.rescued_now is not None:
        notes.append(f" [rescued {fmt_pos(event.rescued_now)}]")
    notes.extend(f" [found survivor at {fmt_pos(p)}]" for p in event.found_now)
    if event.replanned and event.triggering_contradictions:
        cells = ", ".join(fmt_pos(p) for p in event.triggering_contradictions)
        notes.append(f" [debris at {cells}, replanned]")
    return "".join(notes)


def format_ext_trace(events: Sequence[ExtStepEvent], max_segments: int | None = None) -> str:
    """Execution trace in the core style (``(1,1) E ok → (1,2) S slip → … → (4,4)``) with
    ``[rescued (r,c)]`` and ``[found survivor at (r,c)]`` notes besides the core's
    ``[debris at …, replanned]``. ``max_segments`` shortens it like ``runner.format_trace``.
    """
    if not events:
        return ""
    segments: list[str] = []
    note = ""
    last_pos = events[0].new_pos
    for event in events:
        if event.action is None or event.outcome is None:
            note += _ext_notes(event)
            last_pos = event.new_pos
            continue
        label = TRACE_LABELS[event.outcome]
        if event.blocked:
            label += TRACE_BLOCKED_SUFFIX
        segments.append(f"{fmt_pos(event.pos)}{note} {event.action} {label}")
        note = _ext_notes(event)
        last_pos = event.new_pos
    segments.append(f"{fmt_pos(last_pos)}{note}")
    if max_segments is not None and 0 < max_segments < len(segments):
        tail = max(1, max_segments // 2)
        head = max(1, max_segments - tail)
        segments = [*segments[:head], TRACE_ELLIPSIS, *segments[-tail:]]
    return TRACE_SEPARATOR.join(segments)
