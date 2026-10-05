"""Tests for rescue.runner (docs/plan.md §3.9, §3.10 "Mission loop"; D6, D11, D12, D16)."""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from rescue.agents import make_agent
from rescue.agents.base import NO_REPLAN, Abort, Agent, Decision, Observation, ReplanInfo
from rescue.belief import BeliefUpdate
from rescue.config import AftershockConfig, AgentConfig, PlannerConfig, WorldConfig
from rescue.grid import Cell, Pos, TrueGrid
from rescue.model import Action, OutcomeKind, TerrainSlip, UniformSlip, step_pos
from rescue.planners.bfs import shortest_path_length
from rescue.runner import (
    Mission,
    MissionMetrics,
    MissionStatus,
    StepEvent,
    format_metrics_table,
    format_summary,
    format_trace,
    make_transition,
    run_mission,
)
from tests.conftest import MAPS_DIR
from tests.helpers_maps import random_solvable_grid

AGENT_IDS = ("andor", "astar_fixed", "astar_replan")


def agent_for(name: str, grid: TrueGrid, cfg: WorldConfig, mode: str = "eager") -> object:
    """Fresh agent of type ``name`` given the true dynamics of ``grid`` (D5)."""
    return make_agent(
        name, make_transition(grid, cfg), PlannerConfig(), AgentConfig(replan_mode=mode)
    )


def mission(
    name: str, grid: TrueGrid, cfg: WorldConfig | None = None, seed: int = 0
) -> Mission:
    """Completed mission of agent ``name`` on ``grid``."""
    cfg = cfg or WorldConfig()
    return run_mission(grid, agent_for(name, grid, cfg), cfg, seed)


def deck() -> TrueGrid:
    return TrueGrid.load(MAPS_DIR / "deck_demo.txt")


# --- lifecycle -----------------------------------------------------------------------------------


def test_mission_start_event() -> None:
    grid = deck()
    cfg = WorldConfig()
    m = Mission(grid, agent_for("andor", grid, cfg), cfg, seed=3, scenario="deck")
    ev = m.start()
    assert isinstance(ev, StepEvent)
    assert ev.t == 0
    assert ev.action is None
    assert ev.outcome is None
    assert ev.pos == ev.new_pos == grid.start
    assert not ev.bump
    assert not ev.unplanned
    assert ev.status is MissionStatus.RUNNING
    assert m.status is MissionStatus.RUNNING
    assert not m.done
    assert list(m.events) == [ev]
    assert ev.plan_id == m.agent.current_plan_id
    known = {p for p, _ in ev.new_cells}
    assert {(0, 0), (0, 1), (1, 0)} <= known
    assert m.world.pos == grid.start


def test_mission_steps_until_done_then_raises() -> None:
    grid = deck()
    cfg = WorldConfig()
    m = Mission(grid, agent_for("andor", grid, cfg), cfg, seed=0)
    m.start()
    steps = 0
    while not m.done:
        ev = m.step()
        steps += 1
        assert ev is m.events[-1]
        assert steps <= cfg.step_budget(grid.rows, grid.cols)
    assert m.status is MissionStatus.RESCUED
    assert m.events[-1].status is MissionStatus.RESCUED
    assert m.events[-1].new_pos == grid.survivor
    with pytest.raises(RuntimeError):
        m.step()


def test_mission_rescued_on_first_step() -> None:
    """start == survivor cannot be expressed (TrueGrid rejects it); one arrival is the minimum."""
    grid = TrueGrid.from_ascii("SP")
    with pytest.raises(ValueError):
        TrueGrid(cells=grid.cells, start=grid.survivor, survivor=grid.survivor,
                 high_slip=grid.high_slip)
    m = mission("andor", grid, WorldConfig(slip_p=0.0))
    assert m.status is MissionStatus.RESCUED
    assert [e.action for e in m.events] == [None, Action.E]
    metrics = m.metrics()
    assert metrics.success
    assert metrics.failure_reason is None
    assert metrics.steps == 1
    assert metrics.shortest_path == 1
    assert metrics.wasted_moves == 0


#: World variants of the event-chain test: plain slips, a blind sensor (bumps, D6) and
#: aftershocks (stale belief).
CHAIN_VARIANTS = {
    "slip": WorldConfig(slip_p=0.3),
    "blind": WorldConfig(slip_p=0.3, sensor_radius=0),
    "aftershock": WorldConfig(slip_p=0.3, aftershock=AftershockConfig(q=0.2, k=1)),
}


@pytest.mark.parametrize("name", AGENT_IDS)
def test_event_chain_and_metrics_consistency(name: str) -> None:
    totals = dict.fromkeys(("bumps", "aftershocks", "replans", "blocked"), 0)
    for variant, cfg in CHAIN_VARIANTS.items():
        for seed in range(6):
            m = mission(name, random_solvable_grid(8, 8, 0.25, seed), cfg, seed=seed)
            _check_event_chain(m, name, seed, cfg)
            metrics = m.metrics()
            if variant == "slip":
                assert metrics.bumps == 0  # r = 1 (D6)
            totals["bumps"] += metrics.bumps
            totals["aftershocks"] += metrics.aftershock_events
            totals["replans"] += metrics.replans
            totals["blocked"] += metrics.blocked_moves
    # Non-vacuity: the checks above ran on bumps, aftershocks and blocked moves.
    assert totals["bumps"] > 0
    assert totals["aftershocks"] > 0
    assert totals["blocked"] > 0
    if name != "astar_fixed":
        assert totals["replans"] > 0


def _check_event_chain(m: Mission, name: str, seed: int, cfg: WorldConfig) -> None:
    """Every link between events, agent state and metrics (docs/plan.md §3.9, D6, D11, D12)."""
    grid = m.grid
    events = m.events
    ctx = (name, seed, cfg)
    assert events[0].action is None and events[0].t == 0, ctx
    moves = [e for e in events if e.action is not None]
    for prev, ev in itertools.pairwise(events):
        # plan ids only advance on a replan, by exactly one (D12)
        assert ev.plan_id - prev.plan_id == int(ev.replanned), ctx
        if ev.action is None:
            continue
        assert ev.pos == prev.new_pos, ctx
        assert ev.t == prev.t + 1, ctx
    for ev in events:
        assert set(ev.triggering_contradictions) <= set(ev.contradictions), ctx
        assert ev.replanned or not ev.triggering_contradictions, ctx
        if ev.replanned or ev.t == 0:
            snap = m.agent.snapshots[ev.plan_id]
            assert (snap.plan_id, snap.pos, snap.t) == (ev.plan_id, ev.new_pos, ev.t), ctx
    assert m.done and m.status is not MissionStatus.RUNNING, ctx
    assert events[-1].status is m.status, ctx
    assert all(e.status is MissionStatus.RUNNING for e in events[:-1]), ctx
    metrics = m.metrics()
    assert isinstance(metrics, MissionMetrics)
    assert metrics.agent == name
    assert metrics.seed == seed
    assert (metrics.rows, metrics.cols) == (grid.rows, grid.cols)
    assert metrics.slip_p == pytest.approx(cfg.slip_p)
    assert metrics.steps == len(moves) == m.world.t, ctx
    assert metrics.slips == sum(e.outcome is OutcomeKind.SLIPPED for e in moves), ctx
    assert metrics.skids == sum(
        e.outcome in (OutcomeKind.SKID_L, OutcomeKind.SKID_R) for e in moves
    ), ctx
    gave_up = (
        m.status is MissionStatus.ABORTED
        and bool(moves)
        and events[-1].action is None
        and moves[-1].outcome is not OutcomeKind.ARRIVED
    )
    assert metrics.slips_handled == metrics.slips + metrics.skids - int(gave_up), ctx
    assert metrics.bumps == sum(e.bump for e in moves), ctx
    assert all(e.blocked for e in moves if e.bump), ctx
    assert metrics.blocked_moves == sum(e.blocked and not e.bump for e in moves), ctx
    assert metrics.unplanned_states == m.agent.stats.unplanned_states, ctx
    assert metrics.unplanned_states == sum(e.unplanned for e in moves), ctx
    assert metrics.replans == m.agent.stats.replans == sum(e.replanned for e in events), ctx
    assert metrics.cells_explored == m.agent.belief.explored_count(), ctx
    assert metrics.cells_explored == 1 + sum(len(e.new_cells) for e in events), ctx
    assert metrics.planner_fallbacks == m.agent.stats.planner_fallbacks, ctx
    assert metrics.aftershock_events == m.world.aftershock_events, ctx
    assert metrics.aftershock_events == sum(bool(e.aftershock_cells) for e in events), ctx
    assert metrics.success == (m.status is MissionStatus.RESCUED), ctx
    if metrics.success:
        assert metrics.failure_reason is None
        assert events[-1].new_pos == grid.survivor
    else:
        assert metrics.failure_reason in ("aborted", "unreachable", "budget"), ctx


def test_status_matches_failure_reason() -> None:
    reasons = {
        MissionStatus.ABORTED: "aborted",
        MissionStatus.UNREACHABLE: "unreachable",
        MissionStatus.BUDGET: "budget",
    }
    m = mission("andor", TrueGrid.from_ascii("S#P"))
    assert m.status is MissionStatus.UNREACHABLE
    assert m.metrics().failure_reason == reasons[m.status]
    assert m.metrics().steps == 0
    assert math.isnan(m.metrics().wasted_moves)
    assert m.metrics().shortest_path is None  # unreachable on the true map
    m = mission("astar_fixed", TrueGrid.from_ascii("S..#P\n....."), WorldConfig(slip_p=0.0))
    assert m.status is MissionStatus.ABORTED
    assert m.metrics().failure_reason == "aborted"
    assert m.metrics().steps == 2


# --- determinism and common random numbers -----------------------------------------------------


@pytest.mark.parametrize("name", AGENT_IDS)
def test_same_seed_identical_events_and_trace(name: str) -> None:
    for seed in (0, 7, 42):
        grid = random_solvable_grid(8, 8, 0.25, seed, terrain_fraction=0.2)
        cfg = WorldConfig(slip_p=0.3)
        a = mission(name, grid, cfg, seed=seed)
        b = mission(name, grid, cfg, seed=seed)
        assert a.events == b.events
        assert format_trace(a.events) == format_trace(b.events)
        assert _nan_equal(a.metrics(), b.metrics())


def _nan_equal(a: MissionMetrics, b: MissionMetrics) -> bool:
    """Field-wise equality treating NaN == NaN and ignoring wall-clock planning time."""
    for f in dataclasses.fields(a):
        if f.name == "planning_ms":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


@pytest.mark.parametrize("name", ["andor", "astar_replan"])
def test_different_seeds_differ(name: str) -> None:
    grid = random_solvable_grid(8, 8, 0.2, 5)
    cfg = WorldConfig(slip_p=0.3)
    traces = {format_trace(mission(name, grid, cfg, seed=s).events) for s in range(5)}
    assert len(traces) >= 4


@pytest.mark.parametrize("terrain", [0.0, 0.3])
def test_common_random_numbers_across_agents(terrain: float) -> None:
    """Every agent consumes exactly one outcome uniform per step (D3): equal slip p at the
    robot's cell at step t => equal outcome kind at step t."""
    compared = 0
    for seed in range(8):
        grid = random_solvable_grid(8, 8, 0.25, seed, terrain_fraction=terrain)
        cfg = WorldConfig(slip_p=0.3)
        transition = make_transition(grid, cfg)
        runs = {name: mission(name, grid, cfg, seed=seed).events for name in AGENT_IDS}
        base = runs["andor"]
        for other in ("astar_replan", "astar_fixed"):
            for ea, eb in zip(base[1:], runs[other][1:], strict=False):
                if ea.outcome is None or eb.outcome is None:
                    break
                if transition.slip_p(ea.pos) != transition.slip_p(eb.pos):
                    continue
                assert ea.outcome is eb.outcome, (seed, other, ea.t)
                compared += 1
    assert compared > 20


# --- budget, bumps, wasted moves -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["andor", "astar_replan"])
def test_budget_exhaustion(name: str) -> None:
    grid = TrueGrid.from_ascii("S..........P")  # 1x12: shortest path 11, budget 12
    cfg = WorldConfig(slip_p=0.4, budget_factor=1)
    budget = cfg.step_budget(grid.rows, grid.cols)
    assert budget == 12
    statuses = []
    rescued_on_last_step = False
    for seed in range(40):
        m = mission(name, grid, cfg, seed=seed)
        statuses.append(m.status)
        metrics = m.metrics()
        assert metrics.steps <= budget
        if m.status is MissionStatus.BUDGET:
            assert metrics.steps == budget == m.world.t
            assert not metrics.success
            assert metrics.failure_reason == "budget"
            assert math.isnan(metrics.wasted_moves)
            assert m.events[-1].status is MissionStatus.BUDGET
            assert m.events[-1].new_pos != grid.survivor
        else:
            assert m.status is MissionStatus.RESCUED
            # Arriving on the very last budget step is a rescue, not a budget failure.
            rescued_on_last_step |= m.world.t == budget
    assert MissionStatus.BUDGET in statuses
    assert rescued_on_last_step


def test_bumps_happen_only_without_lookahead() -> None:
    bumps_r0 = 0
    for seed in range(10):
        grid = random_solvable_grid(8, 8, 0.3, seed)
        m0 = mission("andor", grid, WorldConfig(slip_p=0.2, sensor_radius=0), seed=seed)
        m1 = mission("andor", grid, WorldConfig(slip_p=0.2, sensor_radius=1), seed=seed)
        for m in (m0, m1):
            moves = [e for e in m.events if e.action is not None]
            assert m.metrics().bumps == sum(e.bump for e in moves)
            assert all(e.blocked for e in moves if e.bump)
        assert m1.metrics().bumps == 0
        bumps_r0 += m0.metrics().bumps
        assert m0.status in (MissionStatus.RESCUED, MissionStatus.BUDGET)
    assert bumps_r0 > 0


def test_wasted_moves() -> None:
    seen_success = seen_failure = False
    for seed in range(8):
        grid = random_solvable_grid(7, 7, 0.25, seed)
        sp = shortest_path_length(grid.passable_mask(), grid.start, grid.survivor)
        for name in AGENT_IDS:
            metrics = mission(name, grid, WorldConfig(slip_p=0.3), seed=seed).metrics()
            assert metrics.shortest_path == sp
            if metrics.success:
                seen_success = True
                assert metrics.wasted_moves == metrics.steps - sp
                assert metrics.wasted_moves >= 0
            else:
                seen_failure = True
                assert math.isnan(metrics.wasted_moves)
    assert seen_success and seen_failure


# --- formatting ----------------------------------------------------------------------------------


def _hand_events() -> list[StepEvent]:
    grid = deck()
    cfg = WorldConfig()
    base = Mission(grid, agent_for("andor", grid, cfg), cfg, seed=0).start()

    def ev(t: int, pos: tuple[int, int], action: Action | None, outcome: OutcomeKind | None,
           new_pos: tuple[int, int], **kw: object) -> StepEvent:
        fields = {
            "t": t, "pos": pos, "action": action, "outcome": outcome, "new_pos": new_pos,
            "blocked": False, "bump": False, "contradictions": (),
            "triggering_contradictions": (), "replanned": False,
            "unplanned": False, "note": "", "status": MissionStatus.RUNNING,
        }
        fields.update(kw)
        return dataclasses.replace(base, **fields)

    A, SL, KL, KR = (OutcomeKind.ARRIVED, OutcomeKind.SLIPPED, OutcomeKind.SKID_L,
                     OutcomeKind.SKID_R)
    return [
        ev(0, (0, 0), None, None, (0, 0)),
        ev(1, (0, 0), Action.E, A, (0, 1)),
        ev(2, (0, 1), Action.S, SL, (0, 1)),
        ev(3, (0, 1), Action.S, A, (1, 1), contradictions=((2, 1),),
           triggering_contradictions=((2, 1),), replanned=True),
        ev(4, (1, 1), Action.W, KL, (1, 1), blocked=True),
        ev(5, (1, 1), Action.E, A, (1, 2), contradictions=((2, 2),)),  # lazy: no replan
        ev(6, (1, 2), Action.S, KR, (1, 1)),
    ]


HAND_TRACE = (
    "(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned] W skid-L "
    "(blocked) → (2,2) E ok → (2,3) S skid-R → (2,2)"
)


def test_format_trace_exact() -> None:
    assert format_trace(_hand_events()) == HAND_TRACE


def test_format_trace_max_segments() -> None:
    events = _hand_events()
    assert format_trace(events, max_segments=7) == HAND_TRACE  # 6 moves + final position
    assert format_trace(events, max_segments=100) == HAND_TRACE
    assert "…" not in HAND_TRACE
    short = format_trace(events, max_segments=2)
    assert short.startswith("(1,1) E ok → ")
    assert "…" in short
    assert "S ok" not in short
    assert short.endswith("(2,2)")
    assert len(short) < len(HAND_TRACE)


def test_format_trace_multiple_contradictions() -> None:
    events = _hand_events()
    events[3] = dataclasses.replace(
        events[3], contradictions=((2, 1), (2, 2)), triggering_contradictions=((2, 1), (2, 2))
    )
    trace = format_trace(events)
    assert "(2,2) [debris at (3,2)" in trace
    assert "(3,3), replanned] W skid-L (blocked)" in trace


def test_format_trace_annotates_only_contradiction_replans() -> None:
    """§3.9: the note marks a replan triggered by contradictions; a replan after a mere
    deviation (A* + replan) and a contradiction without a replan (lazy) stay unannotated."""
    events = _hand_events()
    events[2] = dataclasses.replace(events[2], replanned=True)  # slip at (1,2), no debris
    trace = format_trace(events)
    assert trace.startswith("(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2)")
    assert "replanned]" in trace
    assert trace.count("[") == 1
    assert "(2,3) S skid-R" in trace  # the lazy contradiction at step 5 adds no note


def test_format_trace_ignores_contradictions_that_did_not_trigger_the_replan() -> None:
    """Lazy mode: a deviation replan in the same step as an off-path contradiction is not a
    debris replan, and only the triggering cells are listed."""
    events = _hand_events()
    # step 5: deviation replan + off-path contradiction (nothing triggering) -> no note
    events[5] = dataclasses.replace(events[5], replanned=True)
    # step 3: two contradictions, only (3,2) on the plan -> only (3,2) listed
    events[3] = dataclasses.replace(events[3], contradictions=((2, 1), (0, 3)))
    trace = format_trace(events)
    assert trace == HAND_TRACE


def test_trace_notes_of_lazy_astar_replan_list_only_triggers() -> None:
    """Regression (real missions): lazy A* + replan replans on deviations; contradictions off
    the remaining path in the same step must not be reported as a debris replan."""
    misleading = 0
    for seed in range(40):
        grid = random_solvable_grid(8, 8, 0.3, seed)
        cfg = WorldConfig(slip_p=0.3)
        m = run_mission(grid, agent_for("astar_replan", grid, cfg, mode="lazy"), cfg, seed)
        trace = format_trace(m.events)
        for ev in m.events:
            if ev.replanned and ev.contradictions and not ev.triggering_contradictions:
                misleading += 1
                assert "debris" not in ev.note
        noted = [ev for ev in m.events if ev.replanned and ev.triggering_contradictions]
        assert trace.count("[debris at") == len(noted)
    assert misleading > 0  # the scenario of the regression really occurs


def test_format_trace_of_real_mission_starts_and_ends_right() -> None:
    grid = deck()
    m = mission("andor", grid, WorldConfig(), seed=1)
    trace = format_trace(m.events)
    assert trace.startswith("(1,1) ")
    assert trace.endswith("(4,4)")
    assert trace.count(" → ") == m.metrics().steps


def test_format_summary_success_and_failure() -> None:
    m = mission("andor", deck(), WorldConfig(), seed=0).metrics()
    ok = dataclasses.replace(
        m, success=True, failure_reason=None, cells_explored=11, replans=2, slips=2, skids=1,
        slips_handled=3, bumps=0,
    )
    assert format_summary(ok).rstrip("\n") == (
        "Cells explored: 11   Replans: 2   Slips handled: 3   Collisions: 0\n"
        "Rescue successful"
    )
    bad = dataclasses.replace(
        m, success=False, failure_reason="budget", cells_explored=9, replans=0, slips=0,
        skids=0, slips_handled=0, bumps=4, wasted_moves=math.nan,
    )
    assert format_summary(bad).rstrip("\n") == (
        "Cells explored: 9   Replans: 0   Slips handled: 0   Collisions: 4\n"
        "Rescue failed (budget)"
    )
    aborted = dataclasses.replace(bad, failure_reason="aborted", bumps=0)
    assert format_summary(aborted).rstrip("\n").endswith("Rescue failed (aborted)")
    running = dataclasses.replace(bad, failure_reason=None)
    assert format_summary(running).splitlines()[-1] == "Rescue in progress"


# --- make_transition -----------------------------------------------------------------------------


def test_make_transition_uniform_without_terrain() -> None:
    cfg = WorldConfig(slip_p=0.3)
    t = make_transition(TrueGrid.from_ascii("S..P"), cfg)
    assert isinstance(t, UniformSlip)
    assert not isinstance(t, TerrainSlip)
    assert t.slip_p((0, 1)) == pytest.approx(0.3)


def test_make_transition_terrain_with_tilde() -> None:
    cfg = WorldConfig(slip_p=0.1, terrain_high_p=0.35)
    t = make_transition(TrueGrid.from_ascii("S~.P"), cfg)
    assert isinstance(t, TerrainSlip)
    assert t.slip_p((0, 1)) == pytest.approx(0.35)
    assert t.slip_p((0, 2)) == pytest.approx(0.1)
    assert t.slip_p((0, 0)) == pytest.approx(0.1)


# --- public API and error paths (F15) -----------------------------------------------------------


class _StubAgent(Agent):
    """Agent whose decision is fixed (for the runner's error paths)."""

    name = "stub"

    def __init__(self, decision: object) -> None:
        super().__init__(UniformSlip(WorldConfig().slip_p))
        self.decision = decision

    def _reset_plan(self) -> None:
        pass

    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        return NO_REPLAN

    def choose_action(self, obs: Observation) -> Decision:
        return self.decision  # type: ignore[return-value]

    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]:
        return frozenset({step_pos(pos, action)})

    def policy_arrows(self) -> dict[Pos, Action]:
        return {}


def test_run_mission_on_event_receives_every_event() -> None:
    grid = deck()
    cfg = WorldConfig()
    got: list[StepEvent] = []
    m = run_mission(grid, agent_for("andor", grid, cfg), cfg, seed=3, on_event=got.append)
    assert got == m.events
    assert got[0].t == 0 and got[0].action is None
    assert got[-1].status is MissionStatus.RESCUED


def test_step_world_and_metrics_before_start_raise() -> None:
    grid = deck()
    cfg = WorldConfig()
    m = Mission(grid, agent_for("andor", grid, cfg), cfg)
    assert not m.started
    with pytest.raises(RuntimeError):
        m.step()
    with pytest.raises(RuntimeError):
        _ = m.world
    with pytest.raises(RuntimeError):
        m.metrics()


def test_start_is_idempotent() -> None:
    grid = deck()
    cfg = WorldConfig()
    m = Mission(grid, agent_for("andor", grid, cfg), cfg, seed=2)
    first = m.start()
    assert m.start() is first
    assert m.events == [first]
    assert m.agent.stats.plans_made == 1


def test_unknown_abort_reason_and_bad_decision() -> None:
    grid = deck()
    m = Mission(grid, _StubAgent(Abort("weird", "x")))  # type: ignore[arg-type]
    m.start()
    with pytest.raises(ValueError, match="weird"):
        m.step()
    m = Mission(grid, _StubAgent(None))
    m.start()
    with pytest.raises(TypeError):
        m.step()


def test_mission_rejects_agent_with_other_dynamics() -> None:
    """D5: the agent is given the world's transition model; a mismatch is a ValueError."""
    grid = deck()
    cfg = WorldConfig(slip_p=0.2)
    with pytest.raises(ValueError, match="D5"):
        Mission(grid, make_agent("andor", UniformSlip(0.0)), cfg)
    terrain = TrueGrid.from_ascii("S~.P")
    with pytest.raises(ValueError, match="D5"):
        Mission(terrain, make_agent("andor", UniformSlip(0.2)), cfg)  # '~' needs TerrainSlip
    Mission(terrain, make_agent("andor", make_transition(terrain, cfg)), cfg)  # accepted


# --- D11: slips handled --------------------------------------------------------------------------


def test_slips_handled_excludes_the_slip_the_agent_gave_up_on() -> None:
    """astar_fixed aborts on its first deviation: that slip was experienced, not handled."""
    grid = deck()
    m = mission("astar_fixed", grid, WorldConfig(slip_p=0.3, sensor_radius=0), seed=3)
    assert format_trace(m.events) == "(1,1) E ok → (1,2) E ok → (1,3) E slip → (1,3)"
    metrics = m.metrics()
    assert m.status is MissionStatus.ABORTED
    assert (metrics.slips, metrics.skids, metrics.slips_handled) == (1, 0, 0)
    assert "Slips handled: 0" in format_summary(metrics)
    for seed in range(6):  # the contingency agent handles every slip and skid
        metrics = mission("andor", grid, WorldConfig(slip_p=0.3), seed=seed).metrics()
        assert metrics.slips_handled == metrics.slips + metrics.skids


# --- D6: known debris is blocked, unknown debris is a bump --------------------------------------


def _target(ev: StepEvent) -> Pos:
    """The cell an event's outcome tried to enter."""
    assert ev.action is not None and ev.outcome is not None
    turn = {
        OutcomeKind.ARRIVED: ev.action,
        OutcomeKind.SKID_L: ev.action.left,
        OutcomeKind.SKID_R: ev.action.right,
    }
    return step_pos(ev.pos, turn[ev.outcome]) if ev.outcome in turn else ev.pos


def test_known_debris_is_blocked_not_bump() -> None:
    grid = TrueGrid.from_ascii("S.#P\n....")
    debris = (0, 2)
    # r = 0, p = 0: the first contact is an unexpected collision that reveals the cell.
    m = mission("andor", grid, WorldConfig(slip_p=0.0, sensor_radius=0))
    bump = next(e for e in m.events if e.bump)
    assert bump.blocked and bump.new_pos == bump.pos and _target(bump) == debris
    assert (debris, Cell.DEBRIS) in bump.new_cells  # touch sensing (D6)
    assert debris in bump.contradictions
    # With slips, a later outcome into the now-known debris is blocked, not a bump.
    later: list[StepEvent] = []
    for seed in range(100):
        m = mission("andor", grid, WorldConfig(slip_p=0.2, sensor_radius=0), seed=seed)
        moves = [e for e in m.events if e.action is not None]
        first = next(i for i, e in enumerate(moves) if e.bump)
        later = [e for e in moves[first + 1:] if e.blocked and _target(e) == debris]
        if later:
            break
    assert later, "expected a seed that hits the known debris again"
    assert not any(e.bump for e in later)
    metrics = m.metrics()
    assert metrics.bumps == 1  # only the first, unexpected contact
    assert metrics.blocked_moves >= len(later)


# --- formatting edge cases (F21) ----------------------------------------------------------------


def test_format_trace_of_aborted_mission() -> None:
    m = mission("astar_fixed", TrueGrid.from_ascii("S..#P\n....."), WorldConfig(slip_p=0.0))
    assert m.status is MissionStatus.ABORTED
    assert m.events[-1].action is None
    assert format_trace(m.events) == "(1,1) E ok → (1,2) E ok → (1,3)"


def test_format_metrics_table() -> None:
    m = mission("astar_fixed", TrueGrid.from_ascii("S..#P\n....."), WorldConfig(slip_p=0.0))
    metrics = m.metrics()
    lines = format_metrics_table(metrics).splitlines()
    assert len(lines) == len(dataclasses.fields(MissionMetrics))
    table = {line.split(":")[0]: line.split(":", 1)[1].strip() for line in lines}
    assert list(table) == list(metrics.as_dict())
    assert table["wasted_moves"] == "n/a"  # NaN on failure (D16)
    assert table["failure_reason"] == "aborted"
    assert table["slip_p"] == "0.00"
    assert table["steps"] == "2"
    # values are aligned: they all start in the same column
    assert len({len(line) - len(line.split(":", 1)[1].lstrip()) for line in lines}) == 1
