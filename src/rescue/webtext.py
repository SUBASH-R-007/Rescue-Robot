"""Texts of the web app (docs/plan.md §3.16): event-log lines and step explanations.

* :func:`core_log_lines` / :func:`ext_log_lines` give the event-log lines of one mission event
  with exactly the Pygame simulator's wording (``viz.sim_app.event_messages`` and
  ``viz.ext_sim_app.ext_event_messages``; tested line for line). They are spelled out here
  because the browser has no pygame.
* :func:`explain_event` describes one step in plain English for a first-year student: what the
  robot tried, what happened, and how this agent reacted and why.

Events are duck-typed: a core :class:`~rescue.runner.StepEvent` or an extension
:class:`~rescue.ext.mission.ExtStepEvent`. Every position in a text is 1-indexed
(:func:`~rescue.grid.fmt_pos`). Pure functions; nothing here imports pygame or scipy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rescue.config import SKID_FRACTION, SLIP_STAY_FRACTION
from rescue.grid import Pos, fmt_pos, in_bounds
from rescue.model import Action, OutcomeKind, kind_direction, step_pos
from rescue.runner import FAILURE_REASONS, MissionStatus

__all__ = [
    "LOG_KINDS",
    "TONES",
    "ExplainContext",
    "core_header",
    "core_log_lines",
    "explain_event",
    "explain_event_toned",
    "ext_header",
    "ext_log_lines",
    "friendly_header",
]

#: Kinds of event-log lines (the Pygame ``EventLog`` kinds).
LOG_KINDS: tuple[str, ...] = ("info", "slip", "skid", "replan", "success", "failure", "warn")
#: Short labels of the skid outcomes in log lines.
_SKID_LABELS: dict[OutcomeKind, str] = {
    OutcomeKind.SKID_L: "Skid-L",
    OutcomeKind.SKID_R: "Skid-R",
}
#: Log line with a false-positive rate: how possible phantom debris is drawn (Pygame wording).
PHANTOM_HINT_TEXT = "Debris drawn lighter with a '?' badge may be a phantom (P(debris) < 1)"
#: Decimal places of the probabilities quoted in explanations (in percent).
PERCENT_DECIMALS = 2
#: Text every explanation sentence that numbers a replan contains.
REPLAN_MARK = "(replan #"
#: Tones of explanation sentences (the UI picks a glyph and colour per tone; never parses the
#: text): a plain move, the outcome kinds, what the sensors showed, how the agent reacted,
#: search-mode events, the mission's end and the next move.
TONE_MOVE, TONE_SLIP, TONE_SKID, TONE_BLOCKED, TONE_BUMP = "move", "slip", "skid", "blocked", "bump"
TONE_AFTERSHOCK, TONE_SENSE, TONE_REPLAN, TONE_SEARCH = "aftershock", "sense", "replan", "search"
TONE_FOUND, TONE_GOOD, TONE_BAD, TONE_NEXT, TONE_INFO = "found", "good", "bad", "next", "info"
TONES: tuple[str, ...] = (
    TONE_MOVE, TONE_SLIP, TONE_SKID, TONE_BLOCKED, TONE_BUMP, TONE_AFTERSHOCK, TONE_SENSE,
    TONE_REPLAN, TONE_SEARCH, TONE_FOUND, TONE_GOOD, TONE_BAD, TONE_NEXT, TONE_INFO,
)
#: Agent kinds (catalog ``kind``).
KIND_CONTINGENCY, KIND_PATH, KIND_EXPECTED = "contingency", "path", "expected"
#: How explanations name the planner of each path agent.
PATH_PLANNER_NAMES: dict[str, str] = {
    "astar_fixed": "Fixed A*",
    "astar_replan": "A*",
    "dstar_lite": "D* Lite",
    "search_astar": "A*",
}


def _cells(cells: Sequence[Pos]) -> str:
    return ", ".join(fmt_pos(p) for p in cells)


def _attempted_target(event: Any) -> Pos | None:
    """The cell a move event tried to enter (None for SLIPPED / non-move events)."""
    if event.action is None or event.outcome is None:
        return None
    direction = kind_direction(event.outcome, event.action)
    return None if direction is None else step_pos(event.pos, direction)


# --- event log ------------------------------------------------------------------------------
def core_header(agent: str, scenario: str, rows: int, cols: int, seed: int,
                slip_p: float) -> str:
    """First log line of a core mission (the Pygame simulator's)."""
    return f"{agent} on {scenario} {rows}x{cols}, seed {seed}, p = {slip_p:.2f}"


def ext_header(agent: str, mode: str, scenario: str, rows: int, cols: int, seed: int,
               slip_p: float, k: int, survivors_known: bool, false_negative: float,
               false_positive: float) -> list[tuple[str, str]]:
    """First log lines of an extension mission (the Pygame extension simulator's)."""
    where = "known" if survivors_known else "unknown"
    noise = f", ε = {false_negative:g}" if false_negative > 0 else ""
    if false_positive > 0:
        noise += f", fp = {false_positive:g}"
    lines = [(f"{agent} · {mode} on {scenario} {rows}x{cols}, seed {seed}, p = {slip_p:.2f}, "
              f"k = {k} ({where}){noise}", "info")]
    if false_positive > 0:
        lines.append((PHANTOM_HINT_TEXT, "info"))
    return lines


def core_log_lines(event: Any, *, bump: bool | None = None,
                   status: MissionStatus | None = None) -> list[tuple[str, str]]:
    """Event-log lines ``(text, kind)`` of one event (``viz.sim_app.event_messages``).

    ``bump`` / ``status`` override the event's (the extension log words bumps and the mission
    end itself).
    """
    bump = event.bump if bump is None else bump
    status = event.status if status is None else status
    lines: list[tuple[str, str]] = []
    target = _attempted_target(event)
    if event.outcome is OutcomeKind.SLIPPED:
        lines.append((f"Slip at {fmt_pos(event.pos)}", "slip"))
    elif event.outcome in _SKID_LABELS:
        label = _SKID_LABELS[event.outcome]
        if event.blocked and target is not None:
            lines.append((f"{label} to {fmt_pos(target)} (blocked)", "skid"))
        else:
            lines.append((f"{label} to {fmt_pos(event.new_pos)}", "skid"))
    elif event.outcome is OutcomeKind.ARRIVED and event.blocked and target is not None:
        lines.append((f"Move {event.action} into {fmt_pos(target)} blocked", "info"))
    if bump and target is not None:
        lines.append((f"Bumped into debris at {fmt_pos(target)}", "warn"))
    if event.aftershock_cells:
        lines.append((f"Aftershock: debris at {_cells(event.aftershock_cells)}", "warn"))
    triggering = tuple(event.triggering_contradictions)
    if event.replanned and triggering:
        lines.append((f"Debris sensed at {_cells(triggering)} → replanning", "replan"))
    ignored = tuple(p for p in event.contradictions if p not in triggering)
    if ignored and event.action is None:  # t = 0: sensed before the initial plan
        lines.append((f"Debris sensed at {_cells(ignored)}", "info"))
    elif ignored:
        lines.append((f"Debris sensed at {_cells(ignored)} (no replan needed)", "info"))
    if event.replanned and not triggering:
        lines.append((f"Replanned at {fmt_pos(event.new_pos)}", "replan"))
    if event.unplanned:
        lines.append((f"Unplanned state at {fmt_pos(event.new_pos)}!", "warn"))
    if status is MissionStatus.RESCUED:
        lines.append(("Rescue successful", "success"))
    elif status is not MissionStatus.RUNNING:
        reason = FAILURE_REASONS.get(status, status.name.lower())
        lines.append((f"Mission failed ({reason})", "failure"))
        if event.action is None and event.note:
            lines.append((event.note, "info"))
    return lines


def ext_log_lines(event: Any, *, previous_target: Pos | None, rescued_count: int,
                  k: int) -> list[tuple[str, str]]:
    """Event-log lines of an extension event (``viz.ext_sim_app.ext_event_messages``).

    ``rescued_count`` counts the rescues so far, this event's included.
    """
    lines = core_log_lines(event, bump=False, status=MissionStatus.RUNNING)
    if event.bump and event.action is not None and event.outcome is not None:
        direction = kind_direction(event.outcome, event.action) or event.action
        bump = (f"Bump into unseen debris at {fmt_pos(step_pos(event.pos, direction))}", "warn")
        if event.outcome is OutcomeKind.ARRIVED and lines:
            lines[0] = bump  # replaces the core's "Move … blocked" line
        else:
            lines.insert(min(1, len(lines)), bump)  # after the skid line
    lines.extend((f"Survivor found at {fmt_pos(p)}", "success") for p in event.found_now)
    if event.rescued_now is not None:
        lines.append((f"Rescued {fmt_pos(event.rescued_now)} — {rescued_count}/{k}", "success"))
    if (event.status is MissionStatus.RUNNING and event.target is not None
            and event.target != previous_target):
        if event.mode == "explore":
            lines.append((f"Exploring frontier {fmt_pos(event.target)}", "info"))
        else:
            lines.append((f"Heading to survivor {fmt_pos(event.target)}", "info"))
    if event.status is MissionStatus.RESCUED:
        text = f"All {k} survivors rescued" if k > 1 else "Rescue successful"
        lines.append((text, "success"))
    elif event.status is not MissionStatus.RUNNING:
        reason = FAILURE_REASONS.get(event.status, event.status.name.lower())
        lines.append((f"Mission failed ({reason})", "failure"))
        if event.action is None and event.note:
            lines.append((event.note, "info"))
    return lines


# --- explanations ---------------------------------------------------------------------------
#: A sentence and its tone (one of :data:`TONES`).
Toned = tuple[str, str]


@dataclass(frozen=True)
class ExplainContext:
    """What :func:`explain_event` needs besides the event (the agent's state *after* it).

    ``slip_p`` is the slip parameter of the cell the robot acted from and ``high_slip`` whether
    that cell is '~' terrain. ``replans`` counts the replans so far (this one included).
    ``plan_pretty`` is the deck one-liner of the plan in force when this step made a new
    contingency plan (else None) and ``path_moves`` the length of a new path. ``next_action``
    is the plan's action at the robot's new position (None when the agent will not move next,
    e.g. Fixed A* about to give up). ``path_ahead`` holds the path cells a path agent still
    has to enter and ``blocked_ahead`` the next of them when it is known debris (Fixed A* then
    gives up on its next decision). ``previous_target`` is the extension agent's target before
    the step; ``fallback_started`` says whether this step began a relaxed-passability fallback
    episode.
    """

    agent: str
    kind: str
    mode: str
    rows: int
    cols: int
    replan_mode: str
    sensor_radius: int
    slip_p: float
    high_slip: bool
    replans: int
    budget: int
    goal: Pos | None
    plan_pretty: str | None = None
    path_moves: int | None = None
    expected_steps: float | None = None
    next_action: Action | None = None
    has_plan: bool = True
    k: int = 1
    rescued_count: int = 0
    survivors_known: bool = True
    previous_target: Pos | None = None
    false_negative: float = 0.0
    false_positive: float = 0.0
    fallback_started: bool = False
    path_ahead: tuple[Pos, ...] = ()
    blocked_ahead: Pos | None = None

    @property
    def extension(self) -> bool:
        """True for the extension modes (search / multi / noisy)."""
        return self.mode != "core"

    @property
    def noisy(self) -> bool:
        """True when the debris sensor makes mistakes."""
        return self.false_negative > 0 or self.false_positive > 0

    @property
    def planner_name(self) -> str:
        """How the explanations name the planner of a path agent."""
        return PATH_PLANNER_NAMES.get(self.agent, "A*")


def _pct(probability: float) -> str:
    return f"{round(probability * 100, PERCENT_DECIMALS):g} %"


def _blocker(cell: Pos, ctx: ExplainContext) -> str:
    return "debris" if in_bounds(cell, ctx.rows, ctx.cols) else "the edge of the map"


def _start_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    """The t = 0 event: first sensing and the initial plan."""
    pos = fmt_pos(event.new_pos)
    if ctx.sensor_radius > 0:
        sense = (f"The robot starts at {pos} and its sensors read every cell within "
                 f"{ctx.sensor_radius} step{'s' if ctx.sensor_radius > 1 else ''} of it.")
    else:
        sense = f"The robot starts at {pos}; its sensor (radius 0) sees only its own cell."
    out: list[Toned] = [(sense, TONE_INFO)]
    if event.contradictions:
        out.append((f"It spots debris at {_cells(event.contradictions)}.", TONE_SENSE))
    if ctx.extension:
        out.extend(_ext_start(event, ctx))
        return out
    goal = fmt_pos(ctx.goal) if ctx.goal is not None else "?"
    if not ctx.has_plan:
        out.append((f"No plan exists: on the robot's map the survivor at {goal} cannot be "
                    f"reached from {pos}.", TONE_BAD))
        return out
    unknown = "Unexplored cells are assumed open."
    if ctx.kind == KIND_CONTINGENCY:
        out.append((f"The survivor is at {goal}. AND-OR search builds a contingency plan that "
                    f"covers every slip and skid in advance: {ctx.plan_pretty}. {unknown}",
                    TONE_INFO))
    elif ctx.kind == KIND_EXPECTED:
        how = "LAO* heuristic search" if ctx.agent == "lao_expected" else "Value iteration"
        expected = ""
        if ctx.expected_steps is not None:
            expected = f" ({ctx.expected_steps:.2f} expected steps)"
        out.append((f"The survivor is at {goal}. {how} computes the policy with the fewest "
                    f"expected steps{expected}: {ctx.plan_pretty}. {unknown}", TONE_INFO))
    else:
        moves = ctx.path_moves if ctx.path_moves is not None else 0
        out.append((f"{ctx.planner_name if ctx.agent != 'astar_fixed' else 'A*'} finds a "
                    f"shortest path of {moves} move{'s' if moves != 1 else ''} to the survivor "
                    f"at {goal}. {unknown}", TONE_INFO))
        if ctx.agent == "astar_fixed":
            out.append(("Fixed A* will follow this path without ever replanning: it gives up "
                        "at the first surprise.", TONE_INFO))
    return out


def _ext_start(event: Any, ctx: ExplainContext) -> list[Toned]:
    out: list[Toned] = []
    target = event.target
    if target is None:
        out.append(("No survivor or frontier cell can be reached: the robot is walled in.",
                    TONE_BAD))
    elif event.mode == "explore":
        count = "one survivor" if ctx.k == 1 else f"{ctx.k} survivors"
        out.append((f"The robot knows there is {count} but not where. It explores: its target "
                    f"is the frontier cell {fmt_pos(target)}, the nearest known open cell next "
                    "to unexplored territory.", TONE_SEARCH))
    elif ctx.k > 1:
        out.append((f"All {ctx.k} survivors are known. The robot rescues them nearest-first, "
                    f"starting with {fmt_pos(target)}.", TONE_INFO))
    else:
        out.append((f"The survivor is at {fmt_pos(target)}.", TONE_INFO))
    for p in event.found_now:
        if not ctx.survivors_known:
            out.append((f"Survivor spotted at {fmt_pos(p)}!", TONE_FOUND))
    if ctx.agent == "search_andor":
        out.append(("Each leg is planned with AND-OR search, so slips and skids are covered "
                    "in advance.", TONE_INFO))
    else:
        out.append(("Each leg is planned with A*, which replans whenever the robot leaves its "
                    "path.", TONE_INFO))
    if ctx.noisy:
        noise = f"it misses debris with probability {ctx.false_negative:g}"
        if ctx.false_positive > 0:
            noise += f" and reports phantom debris with probability {ctx.false_positive:g}"
        out.append((f"Its debris sensor is noisy: {noise}, so the robot keeps a debris "
                    "probability for every cell.", TONE_SENSE))
    return out


def _move_sentence(event: Any, ctx: ExplainContext) -> Toned:
    action, outcome = event.action, event.outcome
    pos, new = fmt_pos(event.pos), fmt_pos(event.new_pos)
    attempted = _attempted_target(event)
    if outcome is OutcomeKind.SLIPPED:
        tone = TONE_SLIP
        text = (f"The robot tried to move {action} from {pos} but slipped and stayed in place "
                f"(chance {_pct(SLIP_STAY_FRACTION * ctx.slip_p)}).")
    elif outcome in (OutcomeKind.SKID_L, OutcomeKind.SKID_R):
        tone = TONE_SKID
        side = "left" if outcome is OutcomeKind.SKID_L else "right"
        chance = _pct(SKID_FRACTION * ctx.slip_p / 2.0)
        if event.blocked and attempted is not None:
            text = (f"The robot tried to move {action} from {pos} but skidded {side} towards "
                    f"{fmt_pos(attempted)}, where {_blocker(attempted, ctx)} stopped it, so it "
                    f"stayed at {pos} (chance {chance}).")
        else:
            text = (f"The robot tried to move {action} from {pos} but skidded {side} to {new} "
                    f"(chance {chance}).")
    elif event.blocked and attempted is not None:
        tone = TONE_BLOCKED
        text = (f"The robot tried to move {action} from {pos} into {fmt_pos(attempted)}, but "
                f"{_blocker(attempted, ctx)} blocked it, so it stayed at {pos}.")
    else:
        tone = TONE_MOVE
        text = f"The robot moved {action} from {pos} to {new}."
    if ctx.high_slip and outcome is not OutcomeKind.ARRIVED:
        text += (f" It was standing on slippery terrain (water or mud), where p = "
                 f"{ctx.slip_p:g}.")
    return text, tone


def _outcome_tone(event: Any) -> str:
    """Tone of a 'the plan already covers this' sentence: the outcome it covers."""
    if event.outcome is OutcomeKind.SLIPPED:
        return TONE_SLIP
    if event.outcome in (OutcomeKind.SKID_L, OutcomeKind.SKID_R):
        return TONE_SKID
    return TONE_BLOCKED


def _reaction_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    """How the agent reacted to the outcome (no contradiction-triggered replan)."""
    intended = step_pos(event.pos, event.action)
    deviated = event.new_pos != intended
    new = fmt_pos(event.new_pos)
    running = event.status is MissionStatus.RUNNING
    out: list[Toned] = []
    if ctx.kind == KIND_PATH:
        if ctx.agent == "astar_fixed":
            if deviated and running:
                out.append((f"Fixed A* expected to reach {fmt_pos(intended)} but is at {new}; "
                            "it cannot recover from a deviation, so it will give up.", TONE_BAD))
            return out
        if event.replanned and deviated:
            out.append((f"{ctx.planner_name} expected {fmt_pos(intended)}; it must replan from "
                        f"{new} (replan #{ctx.replans}).", TONE_REPLAN))
        elif event.replanned and not ctx.extension:
            out.append((f"{ctx.planner_name} replans from {new} (replan #{ctx.replans}).",
                        TONE_REPLAN))
        return out
    # Contingency and expected-cost agents follow a policy that covers every outcome.
    if event.unplanned and event.replanned:
        out.append((f"{new} was not anticipated by the plan (an unplanned state), so the robot "
                    f"replans (replan #{ctx.replans}).", TONE_REPLAN))
    elif event.replanned:
        if not ctx.extension:
            out.append((f"The robot replans from {new} (replan #{ctx.replans}).", TONE_REPLAN))
    elif running:
        tone = _outcome_tone(event)
        covered = ("The policy already covers this" if ctx.kind == KIND_EXPECTED
                   else "The plan already contains a")
        if event.outcome is OutcomeKind.SLIPPED:
            if ctx.kind == KIND_EXPECTED:
                out.append((f"{covered} slip (it simply tries again), so no replanning was "
                            "needed.", tone))
            else:
                out.append((f"{covered} retry branch for this slip, so no replanning was "
                            "needed.", tone))
        elif event.blocked:
            out.append(("The plan already anticipated this blocked move (the robot just tries "
                        "again), so no replanning was needed.", tone))
        elif deviated:
            if ctx.kind == KIND_EXPECTED:
                out.append((f"{covered} skid to {new}, so no replanning was needed.", tone))
            else:
                out.append((f"{covered} branch for a skid to {new}, so no replanning was "
                            "needed.", tone))
    return out


def _fixed_astar_sentence(cells: Sequence[Pos], ctx: ExplainContext) -> Toned:
    """What sensed debris means for Fixed A*, which never replans: definite when it can tell."""
    text = f"Sensors revealed debris at {_cells(cells)}."
    on_path = [p for p in cells if p in ctx.path_ahead]
    if ctx.blocked_ahead is not None and ctx.blocked_ahead in cells:
        return (f"{text} {fmt_pos(ctx.blocked_ahead)} is the next cell of its path and Fixed "
                "A* never replans, so it will give up on its next decision.", TONE_BAD)
    if on_path:
        return (f"{text} Debris at {_cells(on_path)} lies on its path and Fixed A* never "
                "replans, so it will give up when it gets there.", TONE_BAD)
    return (f"{text} None of it is on the path, and Fixed A* never replans, so it carries on.",
            TONE_SENSE)


def _contradiction_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    out: list[Toned] = []
    triggering = tuple(event.triggering_contradictions)
    if event.replanned and triggering:
        touched = _attempted_target(event) if event.bump else None
        source = "Sensors"
        if touched in triggering:
            source = "The collision" if len(triggering) == 1 else "The collision and sensors"
        out.append((f"{source} revealed debris at {_cells(triggering)}, which the robot's map "
                    f"had assumed open — the plan is recomputed (replan #{ctx.replans}).",
                    TONE_REPLAN))
    ignored = tuple(p for p in event.contradictions if p not in triggering)
    if not ignored:
        return out
    cells = _cells(ignored)
    if ctx.agent == "astar_fixed":
        out.append(_fixed_astar_sentence(ignored, ctx))
    elif ctx.replan_mode == "lazy" and event.status is MissionStatus.RUNNING:
        if event.replanned:
            out.append((f"Sensors also revealed debris at {cells}, which is not on the plan "
                        "(lazy replanning alone would ignore it).", TONE_SENSE))
        else:
            out.append((f"Sensors revealed debris at {cells}, but it is not on the current "
                        "plan, so lazy replanning keeps the plan.", TONE_SENSE))
    else:
        out.append((f"Sensors revealed debris at {cells}.", TONE_SENSE))
    return out


def _ext_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    out: list[Toned] = [(f"Survivor spotted at {fmt_pos(p)}!", TONE_FOUND)
                        for p in event.found_now]
    if event.rescued_now is not None:
        out.append((f"The robot reached the survivor at {fmt_pos(event.rescued_now)}: "
                    f"{ctx.rescued_count} of {ctx.k} rescued.", TONE_GOOD))
    if ctx.fallback_started:
        out.append(("No target can be reached while suspected debris counts as blocked, so the "
                    "robot temporarily treats uncertain (possibly phantom) debris as passable.",
                    TONE_REPLAN))
    if event.status is not MissionStatus.RUNNING:
        return out
    target = event.target
    if target is None:
        out.append(("No survivor or frontier cell can be reached any more.", TONE_BAD))
    elif target != ctx.previous_target:
        if event.mode == "explore":
            out.append((f"No survivor is known yet, so the robot explores: its new target is "
                        f"the frontier cell {fmt_pos(target)}.", TONE_SEARCH))
        else:
            out.append((f"Next target: the survivor at {fmt_pos(target)}, the nearest known "
                        "one.", TONE_SEARCH))
    elif event.replanned and not event.triggering_contradictions and not event.unplanned:
        if ctx.kind == KIND_PATH and event.new_pos != step_pos(event.pos, event.action):
            pass  # the deviation sentence already says it
        else:
            out.append((f"The robot replans its route to {fmt_pos(target)} "
                        f"(replan #{ctx.replans}).", TONE_REPLAN))
    return out


def _plan_sentence(event: Any, ctx: ExplainContext, counted: bool) -> list[Toned]:
    """The new plan or path of a replan; ``counted`` says whether an earlier sentence already
    gave the replan number."""
    if not event.replanned or event.status is not MissionStatus.RUNNING:
        return []
    number = "" if counted else f" (replan #{ctx.replans})"
    if ctx.plan_pretty is not None:
        return [(f"New plan from {fmt_pos(event.new_pos)}{number}: {ctx.plan_pretty}.",
                 TONE_REPLAN)]
    if ctx.path_moves is not None:
        moves = ctx.path_moves
        return [(f"New path{number}: {moves} move{'s' if moves != 1 else ''} to "
                 f"{fmt_pos(event.target if ctx.extension else ctx.goal)}.", TONE_REPLAN)]
    return []


def _end_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    status = event.status
    if status is MissionStatus.RESCUED:
        if ctx.extension and ctx.k > 1:
            return [(f"All {ctx.k} survivors rescued in {event.t} steps.", TONE_GOOD)]
        return [(f"Rescue successful: the robot reached the survivor at "
                 f"{fmt_pos(event.new_pos)} in {event.t} steps.", TONE_GOOD)]
    if status is MissionStatus.BUDGET:
        return [(f"The step budget of {ctx.budget} steps is used up, so the mission stops. (A "
                 "strong-cyclic plan reaches the goal with probability 1 only when time is "
                 "unlimited.)", TONE_BAD)]
    return []


def _abort_sentences(event: Any, ctx: ExplainContext) -> list[Toned]:
    detail = str(event.note)
    reason = FAILURE_REASONS.get(event.status, "")
    prefix = f"{reason}: "
    if detail.startswith(prefix):
        detail = detail[len(prefix):]
    pos = fmt_pos(event.new_pos)
    if event.status is MissionStatus.ABORTED and ctx.agent == "astar_fixed":
        return [(f"Fixed A* gives up: {detail}. It never replans, which is why it fails after "
                 "a slip, a skid or a blocked path.", TONE_BAD)]
    if event.status is MissionStatus.UNREACHABLE:
        if ctx.extension:
            return [(f"The mission ends: {detail} on the robot's map.", TONE_BAD)]
        return [(f"The mission ends: no plan reaches the survivor from {pos} on the robot's "
                 f"map ({detail}).", TONE_BAD)]
    return [(f"The mission ends ({reason or event.status.value}): {detail}.", TONE_BAD)]


def explain_event_toned(event: Any, ctx: ExplainContext) -> list[Toned]:
    """Plain-English sentences describing one mission event, each with its tone.

    The t = 0 event describes the first sensing and plan; an abort event (no action, terminal
    status) why the agent gave up; every move event the attempted move and its outcome,
    collisions, aftershocks, sightings and rescues, what the sensors revealed, how the agent
    reacted (a covered branch, a replan and why, a new target) and how the mission ended.
    """
    if event.action is None or event.outcome is None:
        if event.status is not MissionStatus.RUNNING:
            return _abort_sentences(event, ctx)
        return _start_sentences(event, ctx)
    out: list[Toned] = [_move_sentence(event, ctx)]
    attempted = _attempted_target(event)
    if event.bump and attempted is not None:
        bump = f"It bumped into debris at {fmt_pos(attempted)} that it did not know about."
        if ctx.noisy:
            bump += (" Its noisy sensor had missed this debris; touching it makes the debris "
                     "certain.")
        elif ctx.sensor_radius == 0:
            bump += " With sensor radius 0 it cannot see the cell in front of it."
        out.append((bump, TONE_BUMP))
    if event.aftershock_cells:
        out.append((f"An aftershock dropped new debris at {_cells(event.aftershock_cells)}, "
                    "outside the robot's sensor range.", TONE_AFTERSHOCK))
    out.extend(_contradiction_sentences(event, ctx))
    if not (event.replanned and event.triggering_contradictions):
        out.extend(_reaction_sentences(event, ctx))
    if ctx.extension:
        out.extend(_ext_sentences(event, ctx))
    out.extend(_plan_sentence(event, ctx, any(REPLAN_MARK in text for text, _ in out)))
    out.extend(_end_sentences(event, ctx))
    if event.status is MissionStatus.RUNNING:
        if ctx.next_action is not None:
            out.append((f"Next: move {ctx.next_action} from {fmt_pos(event.new_pos)}.",
                        TONE_NEXT))
        elif (ctx.blocked_ahead is not None
              and not any(fmt_pos(ctx.blocked_ahead) in text for text, _ in out)):
            out.append((f"The next cell of its path, {fmt_pos(ctx.blocked_ahead)}, is known "
                        "debris and Fixed A* never replans, so it will give up on its next "
                        "decision.", TONE_BAD))
    return out


def explain_event(event: Any, ctx: ExplainContext) -> list[str]:
    """The sentences of :func:`explain_event_toned` without their tones."""
    return [text for text, _ in explain_event_toned(event, ctx)]


def friendly_header(agent_label: str, map_title: str, rows: int, cols: int, seed: int,
                    slip_p: float, extra: str = "") -> str:
    """The first event-log line as the web app shows it (labels instead of identifiers)."""
    return (f"{agent_label} on {map_title} ({rows}×{cols}), luck seed {seed}, wheel slip "
            f"p = {slip_p:.2f}{extra}")
