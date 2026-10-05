"""Tests for rescue.agents (docs/plan.md §3.8, §3.10; decisions D6, D8, D10, D12).

The agents are driven here with a hand-written loop over :class:`rescue.world.World` (not the
runner), so these tests pin the agent contract on its own: ``reset`` → ``on_outcome`` at t=0
→ (``choose_action`` → ``World.step`` → ``on_outcome``)*.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import pytest

from rescue.agents import AGENTS, make_agent
from rescue.agents.base import Abort, Agent, AgentUpdate, Observation, PathAgent
from rescue.belief import BeliefMap
from rescue.config import AgentConfig, PlannerConfig, WorldConfig
from rescue.grid import Cell, Pos, TrueGrid, footprint
from rescue.model import Action, OutcomeKind, ProblemModel, TransitionModel, UniformSlip, step_pos
from rescue.planners.andor import PlanResult
from rescue.runner import make_transition, run_mission
from rescue.world import Percept, StepResult, World
from tests.conftest import MAPS_DIR, open_ascii
from tests.helpers_maps import random_solvable_grid

AGENT_IDS = ("andor", "astar_fixed", "astar_replan")
DECK_ROOT = "E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}"
DECK_P = 0.2
MAX_STEPS = 400

#: 2x5, p = 0: the shortest path runs along row 1; the debris at (2,2) is sensed from (1,2)
#: and is NOT in the plan envelope.
MAP_OFF_PATH = "S...P\n.#..."
#: 2x5, p = 0: the debris at (1,4) is ON the planned row-1 path, sensed only from (1,3).
MAP_ON_PATH = "S..#P\n....."


# --- driver ------------------------------------------------------------------------------------


@dataclass
class StepLog:
    """One iteration of the hand-written mission loop."""

    pos: Pos
    decision: Action | Abort
    anticipated: frozenset[Pos] = frozenset()
    result: StepResult | None = None
    update: AgentUpdate | None = None
    #: Contradictions recomputed independently from the belief before ``on_outcome`` (D8).
    contradictions: frozenset[Pos] = frozenset()
    #: Envelope of the plan in force when the action was chosen.
    envelope: frozenset[Pos] = frozenset()
    bump: bool = False
    #: ``agent.stats.replans`` right after ``choose_action`` of this step.
    replans_at_choice: int = 0
    #: Path agents: path cells still to be entered once this action is executed (the cells
    #: after the next expected one), captured when the action was chosen.
    remaining: frozenset[Pos] = frozenset()


@dataclass
class Run:
    """Everything recorded while driving an agent."""

    agent: Agent
    grid: TrueGrid
    world: World
    initial: AgentUpdate
    steps: list[StepLog] = field(default_factory=list)

    @property
    def abort(self) -> Abort | None:
        last = self.steps[-1].decision if self.steps else None
        return last if isinstance(last, Abort) else None

    @property
    def moves(self) -> list[StepLog]:
        return [s for s in self.steps if s.result is not None]

    @property
    def rescued(self) -> bool:
        return self.world.pos == self.grid.survivor


def new_agent(
    name: str,
    transition: TransitionModel,
    *,
    mode: str = "eager",
    validate: bool = True,
) -> Agent:
    """Agent from the registry with validation on (soundness bugs surface as fallbacks)."""
    return make_agent(
        name, transition, PlannerConfig(validate=validate), AgentConfig(replan_mode=mode)
    )


def drive(
    agent: Agent,
    grid: TrueGrid,
    transition: TransitionModel,
    *,
    radius: int = 1,
    seed: int = 0,
    max_steps: int = MAX_STEPS,
) -> Run:
    """Run the plan → act → observe loop with the agent's own decisions."""
    world = World(grid, transition, sensor_radius=radius, seed=seed)
    belief = BeliefMap(grid.rows, grid.cols, grid.survivor)
    agent.reset(grid.start, grid.survivor, belief)
    initial = agent.on_outcome(grid.start, world.sense())
    run = Run(agent=agent, grid=grid, world=world, initial=initial)
    for t in range(max_steps):
        if world.pos == grid.survivor:
            break
        pos = world.pos
        envelope = _current_envelope(agent)
        decision = agent.choose_action(Observation(pos, t))
        log = StepLog(pos=pos, decision=decision, replans_at_choice=agent.stats.replans)
        run.steps.append(log)
        if isinstance(decision, Abort):
            break
        log.envelope = _current_envelope(agent) or envelope
        log.anticipated = frozenset(agent.anticipated(pos, decision))
        log.remaining = _remaining_after_step(agent)
        before = agent.belief.copy()
        result = world.step(decision)
        log.result = result
        log.bump = result.hit_debris and before.get(result.target) != Cell.DEBRIS
        log.contradictions = frozenset(
            p for p, cell in result.percept.cells
            if cell == Cell.DEBRIS and before.get(p) != Cell.DEBRIS
        )
        log.update = agent.on_outcome(result.new_pos, result.percept)
    return run


def _current_envelope(agent: Agent) -> frozenset[Pos]:
    return agent.snapshots[-1].envelope if agent.snapshots else frozenset()


def _remaining_after_step(agent: Agent) -> frozenset[Pos]:
    """Path cells after the next expected one (the lazy-mode replan zone after the step)."""
    if not isinstance(agent, PathAgent) or agent.snapshots[-1].path is None:
        return frozenset()
    path = agent.snapshots[-1].path
    cursor = path.index(agent.pos)  # the robot is on the expected cell when it acts
    return frozenset(path[cursor + 2 :])


def load_deck() -> TrueGrid:
    return TrueGrid.load(MAPS_DIR / "deck_demo.txt")


def start_agent(name: str, grid: TrueGrid, transition: TransitionModel, **kw: object) -> Agent:
    """Agent after reset + the t=0 sensing at the start (initial plan made)."""
    agent = new_agent(name, transition, **kw)  # type: ignore[arg-type]
    world = World(grid, transition, sensor_radius=1, seed=0)
    agent.reset(grid.start, grid.survivor, BeliefMap(grid.rows, grid.cols, grid.survivor))
    agent.on_outcome(grid.start, world.sense())
    return agent


# --- registry ------------------------------------------------------------------------------------


def test_registry_has_the_three_agents() -> None:
    for name in AGENT_IDS:
        assert name in AGENTS


@pytest.mark.parametrize("name", AGENT_IDS)
def test_make_agent_builds_named_agent(name: str) -> None:
    agent = make_agent(name, UniformSlip(DECK_P), PlannerConfig(), AgentConfig())
    assert isinstance(agent, Agent)
    assert agent.name == name
    assert type(agent).name == name


def test_make_agent_unknown_name() -> None:
    with pytest.raises(ValueError):
        make_agent("no_such_agent", UniformSlip(DECK_P), PlannerConfig(), AgentConfig())


@pytest.mark.parametrize("name", AGENT_IDS)
def test_reset_and_initial_plan_state(name: str) -> None:
    grid = load_deck()
    agent = new_agent(name, UniformSlip(DECK_P))
    agent.reset(grid.start, grid.survivor, BeliefMap(grid.rows, grid.cols, grid.survivor))
    assert agent.pos == grid.start
    assert agent.goal == grid.survivor
    assert agent.current_plan_id == -1
    world = World(grid, UniformSlip(DECK_P), sensor_radius=1, seed=0)
    update = agent.on_outcome(grid.start, world.sense())
    assert not update.unplanned
    assert agent.stats.replans == 0  # D12: the initial plan is not a replan
    assert agent.stats.unplanned_states == 0
    # The t=0 sensing reached the agent's belief.
    for p in footprint(grid.start, 1, grid.rows, grid.cols):
        assert agent.belief.is_known(p)
    # A plan exists once the first action is chosen.
    decision = agent.choose_action(Observation(grid.start, 0))
    assert isinstance(decision, Action)
    assert agent.current_plan_id >= 0
    assert agent.snapshots and agent.snapshots[-1].plan_id == agent.current_plan_id
    assert agent.stats.plans_made >= 1


# --- andor ---------------------------------------------------------------------------------------


def test_andor_initial_plan_on_deck_demo() -> None:
    grid = load_deck()
    agent = start_agent("andor", grid, UniformSlip(DECK_P))
    assert agent.current_plan_id >= 0
    snap = agent.snapshots[-1]
    assert snap.plan is not None
    assert snap.plan.pretty() == DECK_ROOT
    assert snap.pos == grid.start
    assert snap.t == 0
    assert snap.envelope == snap.plan.envelope()
    assert agent.stats.plans_made == 1
    assert agent.stats.replans == 0
    assert agent.stats.plan_size == len(snap.plan.to_policy())
    assert agent.policy_arrows()[grid.start] is Action.E
    assert agent.choose_action(Observation(grid.start, 0)) is Action.E


def test_andor_anticipated_is_results_on_planning_snapshot() -> None:
    grid = load_deck()
    transition = UniformSlip(DECK_P)
    agent = start_agent("andor", grid, transition)
    model = ProblemModel.from_belief(agent.belief, transition)  # belief == planning belief now
    for s in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2)):
        for a in Action:
            want = frozenset(o.state for o in model.results(s, a))
            assert agent.anticipated(s, a) == want, (s, a)
    # At (1,1) going E the N-skid hits the wall and merges with "stay" (deck).
    assert agent.anticipated((0, 0), Action.E) == frozenset({(0, 0), (0, 1), (1, 0)})


def test_andor_anticipated_uses_snapshot_not_current_belief() -> None:
    """Lazy mode ignores an off-envelope contradiction, so the snapshot stays the old one."""
    grid = TrueGrid.from_ascii(MAP_OFF_PATH)
    run = drive(new_agent("andor", UniformSlip(0.0), mode="lazy"), grid, UniformSlip(0.0),
                max_steps=1)
    agent = run.agent
    assert run.moves[0].result.new_pos == (0, 1)
    assert agent.belief.get((1, 1)) == Cell.DEBRIS
    assert agent.stats.replans == 0
    # The planning model believed (2,2) open: moving S from (1,2) anticipates arriving there.
    assert agent.anticipated((0, 1), Action.S) == frozenset({(1, 1)})


def test_andor_eager_replans_on_off_envelope_contradiction() -> None:
    grid = TrueGrid.from_ascii(MAP_OFF_PATH)
    run = drive(new_agent("andor", UniformSlip(0.0), mode="eager"), grid, UniformSlip(0.0))
    first = run.moves[0]
    assert first.result.new_pos == (0, 1)
    assert first.contradictions == {(1, 1)}
    assert (1, 1) not in first.envelope
    assert first.update is not None and first.update.replanned
    assert set(first.update.contradictions) == {(1, 1)}
    assert run.rescued
    assert run.agent.stats.replans == 1
    assert run.agent.stats.unplanned_states == 0


def test_andor_lazy_ignores_off_envelope_contradiction() -> None:
    grid = TrueGrid.from_ascii(MAP_OFF_PATH)
    run = drive(new_agent("andor", UniformSlip(0.0), mode="lazy"), grid, UniformSlip(0.0))
    first = run.moves[0]
    assert first.contradictions == {(1, 1)}
    assert (1, 1) not in first.envelope
    assert first.update is not None and not first.update.replanned
    assert set(first.update.contradictions) == {(1, 1)}
    assert run.rescued
    assert run.agent.stats.replans == 0
    assert run.agent.stats.contradictions >= 1


def test_andor_lazy_replans_on_envelope_contradiction() -> None:
    grid = TrueGrid.from_ascii(MAP_ON_PATH)
    run = drive(new_agent("andor", UniformSlip(0.0), mode="lazy"), grid, UniformSlip(0.0))
    hits = [s for s in run.moves if (0, 3) in s.contradictions]
    assert len(hits) == 1
    hit = hits[0]
    assert hit.result.new_pos == (0, 2)
    assert (0, 3) in hit.envelope
    assert hit.update is not None and hit.update.replanned
    # No replan before the contradiction.
    assert all(not s.update.replanned for s in run.moves[: run.moves.index(hit)])
    assert run.rescued
    assert run.agent.stats.replans == 1
    assert run.agent.stats.unplanned_states == 0


def test_andor_never_replans_on_a_mere_slip() -> None:
    """All-open map: no contradictions ever, so slips and skids never cause a replan."""
    grid = TrueGrid.from_ascii(open_ascii(6, 6, (0, 0), (5, 5)))
    transition = UniformSlip(0.4)
    non_arrived = 0
    for seed in range(8):
        run = drive(new_agent("andor", transition), grid, transition, seed=seed)
        assert run.rescued
        assert run.agent.stats.replans == 0
        assert run.agent.stats.unplanned_states == 0
        assert all(not s.update.replanned for s in run.moves)
        non_arrived += sum(s.result.kind is not OutcomeKind.ARRIVED for s in run.moves)
    assert non_arrived > 0


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_andor_replan_rule_on_random_maps(mode: str) -> None:
    """§3.10: with r >= 1 the andor agent replans exactly on (relevant) contradictions."""
    replans_seen = 0
    for seed in range(20):
        grid = random_solvable_grid(7, 7, 0.3, seed)
        p = (0.0, 0.1, 0.2, 0.3, 0.4)[seed % 5]
        transition = UniformSlip(p)
        run = drive(new_agent("andor", transition, mode=mode), grid, transition, seed=seed)
        assert run.rescued, seed
        stats = run.agent.stats
        assert stats.unplanned_states == 0
        assert stats.planner_fallbacks == 0
        for s in run.moves:
            assert s.result.new_pos in s.anticipated  # D10: never unplanned with r >= 1
            assert not s.bump  # D6
            assert not s.update.unplanned
            assert set(s.update.contradictions) == s.contradictions
            if s.result.new_pos == grid.survivor:
                continue
            relevant = s.contradictions if mode == "eager" else s.contradictions & s.envelope
            assert s.update.replanned == bool(relevant), (seed, s.pos, s.result)
            assert set(s.update.triggering_contradictions) == relevant
        replans_seen += stats.replans
        assert stats.replans == sum(s.update.replanned for s in run.moves)
        assert stats.plans_made == stats.replans + 1
        ids = [snap.plan_id for snap in run.agent.snapshots]
        assert ids == sorted(set(ids))
    assert replans_seen > 0


def test_andor_unplanned_outcome_triggers_safety_net() -> None:
    """r = 0, p = 0: a bump is an unplanned stay; the agent counts it and replans."""
    grid = TrueGrid.from_ascii("S.#P\n....")
    transition = UniformSlip(0.0)
    run = drive(new_agent("andor", transition), grid, transition, radius=0)
    bumps = [s for s in run.moves if s.bump]
    assert bumps, "the robot should bump into the unseen debris at (1,3)"
    first = bumps[0]
    assert first.result.new_pos == first.pos
    assert first.result.new_pos not in first.anticipated
    assert first.update.unplanned
    assert first.update.replanned
    assert run.agent.stats.unplanned_states >= 1
    assert run.rescued


# --- astar_fixed ---------------------------------------------------------------------------------


def _first_deviation(run: Run) -> int | None:
    for i, s in enumerate(run.moves):
        if s.result.new_pos not in s.anticipated:
            return i
    return None


def test_astar_fixed_aborts_exactly_at_first_deviation() -> None:
    grid = TrueGrid.from_ascii(open_ascii(6, 6, (0, 0), (5, 5)))
    transition = UniformSlip(0.4)
    first_step_slip_seen = False
    for seed in range(40):
        run = drive(new_agent("astar_fixed", transition), grid, transition, seed=seed)
        dev = _first_deviation(run)
        for i, s in enumerate(run.moves):
            # Deviation <=> the outcome is not an unblocked "arrived".
            arrived = s.result.kind is OutcomeKind.ARRIVED and not s.result.blocked
            assert (s.result.new_pos in s.anticipated) == arrived
            assert len(s.anticipated) == 1
            if dev is None or i < dev:
                assert arrived
        if dev is None:
            assert run.rescued
            assert run.agent.stats.unplanned_states == 0
            continue
        assert len(run.moves) == dev + 1  # no move after the deviation
        assert run.agent.stats.unplanned_states == 1
        if run.moves[dev].result.new_pos == grid.survivor:
            continue  # a lucky skid onto the survivor ends the mission first
        assert run.abort is not None
        assert run.abort.reason == "aborted"
        assert isinstance(run.abort.detail, str)
        assert run.steps[-1].pos == run.moves[dev].result.new_pos
        if dev == 0 and run.moves[0].result.kind is OutcomeKind.SLIPPED:
            first_step_slip_seen = True
            assert run.steps[-1].pos == grid.start
    assert first_step_slip_seen, "expected a seed whose very first outcome is a slip"


def test_astar_fixed_aborts_when_next_cell_is_sensed_debris() -> None:
    grid = TrueGrid.from_ascii(MAP_ON_PATH)
    transition = UniformSlip(0.0)
    run = drive(new_agent("astar_fixed", transition), grid, transition)
    assert [s.result.new_pos for s in run.moves] == [(0, 1), (0, 2)]
    assert run.abort is not None
    assert run.abort.reason == "aborted"
    assert run.steps[-1].pos == (0, 2)
    assert run.agent.stats.unplanned_states == 0  # no deviation, just new debris
    assert run.agent.stats.replans == 0


def test_astar_fixed_plans_once_and_ignores_off_path_debris() -> None:
    grid = TrueGrid.from_ascii(MAP_OFF_PATH)
    transition = UniformSlip(0.0)
    run = drive(new_agent("astar_fixed", transition), grid, transition)
    assert run.rescued
    assert run.agent.stats.replans == 0
    assert run.agent.stats.plans_made == 1
    assert len(run.moves) == 4


@pytest.mark.parametrize("name", ["astar_fixed", "astar_replan"])
def test_astar_anticipated_is_expected_next_cell(name: str) -> None:
    grid = load_deck()
    transition = UniformSlip(DECK_P)
    agent = start_agent(name, grid, transition)
    decision = agent.choose_action(Observation(grid.start, 0))
    assert isinstance(decision, Action)
    snap = agent.snapshots[-1]
    assert snap.path is not None
    assert snap.path[0] == grid.start and snap.path[-1] == grid.survivor
    assert len(snap.path) == 7  # optimistic shortest path (1,1) -> (4,4)
    assert snap.path[1] == step_pos(grid.start, decision)
    assert agent.anticipated(grid.start, decision) == frozenset({snap.path[1]})
    assert set(agent.policy_arrows()) == set(snap.path[:-1])


# --- astar_replan --------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_astar_replan_replans_on_deviation_and_contradiction(mode: str) -> None:
    """§3.10: exactly one replan per step that deviated or saw a relevant contradiction (any
    in eager mode, one on the remaining path in lazy mode), and none otherwise."""
    deviations_seen = relevant_seen = ignored_seen = 0
    for seed in range(20):
        grid = random_solvable_grid(7, 7, 0.3, seed)
        transition = UniformSlip((0.1, 0.2, 0.3, 0.4)[seed % 4])
        run = drive(new_agent("astar_replan", transition, mode=mode), grid, transition,
                    seed=seed)
        assert run.rescued, seed
        moves = run.moves
        deviations = sum(s.result.new_pos not in s.anticipated for s in moves)
        assert run.agent.stats.unplanned_states == deviations
        assert all(len(s.anticipated) == 1 for s in moves)
        assert not any(s.bump for s in moves)
        assert run.agent.stats.replans == sum(s.update.replanned for s in moves)
        for i in range(len(moves) - 1):
            s = moves[i]
            delta = moves[i + 1].replans_at_choice - s.replans_at_choice
            deviated = s.result.new_pos not in s.anticipated
            relevant = s.contradictions if mode == "eager" else s.contradictions & s.remaining
            triggered = deviated or bool(relevant)
            assert delta == int(triggered), (seed, i)
            assert s.update.replanned == triggered, (seed, i)
            assert set(s.update.triggering_contradictions) == (relevant if triggered else set())
            deviations_seen += deviated
            relevant_seen += bool(relevant)
            ignored_seen += bool(s.contradictions - relevant)
    assert deviations_seen > 0
    assert relevant_seen > 0
    if mode == "lazy":
        assert ignored_seen > 0  # off-path contradictions were really ignored


@pytest.mark.parametrize(("mode", "replans"), [("eager", 1), ("lazy", 0)])
def test_astar_replan_contradiction_modes(mode: str, replans: int) -> None:
    grid = TrueGrid.from_ascii(MAP_OFF_PATH)
    transition = UniformSlip(0.0)
    run = drive(new_agent("astar_replan", transition, mode=mode), grid, transition)
    assert run.rescued
    assert run.agent.stats.replans == replans


def test_astar_replan_lazy_replans_when_path_blocked() -> None:
    grid = TrueGrid.from_ascii(MAP_ON_PATH)
    transition = UniformSlip(0.0)
    run = drive(new_agent("astar_replan", transition, mode="lazy"), grid, transition)
    assert run.rescued
    assert run.agent.stats.replans == 1
    assert run.agent.stats.unplanned_states == 0


# --- unreachable ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", AGENT_IDS)
def test_unreachable_when_walled_off_at_start(name: str) -> None:
    grid = TrueGrid.from_ascii("S#P")
    transition = UniformSlip(DECK_P)
    run = drive(new_agent(name, transition), grid, transition)
    assert run.moves == []
    assert run.abort is not None
    assert run.abort.reason == "unreachable"


@pytest.mark.parametrize("name", ["andor", "astar_replan"])
def test_unreachable_discovered_while_exploring(name: str) -> None:
    grid = TrueGrid.from_ascii("S..\n.##\n.#P")
    transition = UniformSlip(DECK_P)
    for seed in range(3):
        run = drive(new_agent(name, transition), grid, transition, seed=seed)
        assert run.abort is not None, seed
        assert run.abort.reason == "unreachable"
        assert not any(s.bump for s in run.moves)
        for p in ((1, 1), (1, 2), (2, 1)):
            assert run.agent.belief.get(p) == Cell.DEBRIS


def test_on_outcome_accepts_hand_made_percept() -> None:
    """The agent only relies on the Percept contract (pairs of position and true cell)."""
    grid = load_deck()
    agent = new_agent("andor", UniformSlip(DECK_P))
    agent.reset(grid.start, grid.survivor, BeliefMap(grid.rows, grid.cols, grid.survivor))
    percept = Percept(((grid.start, Cell.OPEN), ((0, 1), Cell.OPEN), ((1, 0), Cell.DEBRIS)))
    update = agent.on_outcome(grid.start, percept)
    assert set(update.newly_known) == {(0, 0), (0, 1), (1, 0)}
    assert set(update.contradictions) == {(1, 0)}
    assert agent.choose_action(Observation(grid.start, 0)) is Action.E


# --- lifecycle contract (F14) -------------------------------------------------------------------


@pytest.mark.parametrize("name", AGENT_IDS)
def test_reset_rejects_goal_not_survivor(name: str) -> None:
    agent = new_agent(name, UniformSlip(DECK_P))
    with pytest.raises(ValueError):
        agent.reset((0, 0), (1, 1), BeliefMap(3, 3, (2, 2)))


@pytest.mark.parametrize("name", AGENT_IDS)
def test_choose_action_before_on_outcome_raises(name: str) -> None:
    grid = load_deck()
    agent = new_agent(name, UniformSlip(DECK_P))
    agent.reset(grid.start, grid.survivor, BeliefMap(grid.rows, grid.cols, grid.survivor))
    with pytest.raises(RuntimeError):
        agent.choose_action(Observation(grid.start, 0))


@pytest.mark.parametrize("name", AGENT_IDS)
def test_choose_action_rejects_wrong_position(name: str) -> None:
    grid = load_deck()
    agent = start_agent(name, grid, UniformSlip(DECK_P))
    with pytest.raises(ValueError):
        agent.choose_action(Observation((1, 1), 0))


@pytest.mark.parametrize("name", AGENT_IDS)
def test_choose_action_on_goal_raises_without_planning(name: str) -> None:
    """On the goal the mission is over: a clear error, not a misleading Abort("unreachable"),
    and no planning call (which would have counted a phantom replan)."""
    # Start on the goal (only reachable by driving the agent directly; TrueGrid forbids it).
    agent = new_agent(name, UniformSlip(DECK_P))
    agent.reset((1, 1), (1, 1), BeliefMap(2, 2, (1, 1)))
    agent.on_outcome((1, 1), Percept((((1, 1), Cell.SURVIVOR),)))
    with pytest.raises(RuntimeError, match="goal"):
        agent.choose_action(Observation((1, 1), 0))
    assert agent.stats.plans_made == 0
    # Arrive on the goal mid-mission.
    grid = load_deck()
    cfg = WorldConfig(slip_p=0.0)
    m = run_mission(grid, new_agent(name, make_transition(grid, cfg)), cfg, seed=0)
    assert m.world.pos == grid.survivor
    stats = dataclasses.replace(m.agent.stats)
    with pytest.raises(RuntimeError, match="goal"):
        m.agent.choose_action(Observation(grid.survivor, m.world.t))
    assert m.agent.stats == stats


def test_andor_choose_action_never_plans() -> None:
    """A position without a policy entry is replanned in on_outcome; reaching choose_action
    without one is a logic error (no silent replan that no event would report)."""
    agent = start_agent("andor", load_deck(), UniformSlip(DECK_P))
    plans = agent.stats.plans_made
    del agent._policy[agent.pos]  # simulate the impossible state
    with pytest.raises(RuntimeError, match="policy"):
        agent.choose_action(Observation(agent.pos, 0))
    assert agent.stats.plans_made == plans


def test_astar_replan_choose_action_never_plans() -> None:
    agent = start_agent("astar_replan", load_deck(), UniformSlip(DECK_P))
    plans = agent.stats.plans_made
    agent._cursor += 1  # simulate a stale path that on_outcome failed to replan
    with pytest.raises(RuntimeError, match="stale"):
        agent.choose_action(Observation(agent.pos, 0))
    assert agent.stats.plans_made == plans


@pytest.mark.parametrize("name", AGENT_IDS)
def test_agent_reuse_matches_fresh_agent(name: str) -> None:
    """reset() clears everything: a reused agent behaves exactly like a fresh one (UI R / S)."""
    cfg = WorldConfig(slip_p=0.3)
    deck = load_deck()
    other = random_solvable_grid(7, 7, 0.25, 3)
    reused = new_agent(name, make_transition(deck, cfg))
    run_mission(other, reused, cfg, seed=1)
    second = run_mission(deck, reused, cfg, seed=5)
    fresh = run_mission(deck, new_agent(name, make_transition(deck, cfg)), cfg, seed=5)
    assert second.events == fresh.events
    assert dataclasses.replace(second.agent.stats, planning_ms=0.0) == dataclasses.replace(
        fresh.agent.stats, planning_ms=0.0
    )
    assert second.agent.snapshots == fresh.agent.snapshots
    assert second.agent.policy_arrows() == fresh.agent.policy_arrows()


# --- planner fallbacks (F11) --------------------------------------------------------------------


def test_andor_counts_planner_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner_fallbacks counter really increments (so '== 0' assertions can fail)."""
    grid = load_deck()
    cfg = WorldConfig()
    agent = new_agent("andor", make_transition(grid, cfg))
    real_plan = agent.planner.plan

    def fake_plan(model: ProblemModel, start: Pos) -> PlanResult:
        result = real_plan(model, start)
        return dataclasses.replace(
            result, stats=dataclasses.replace(result.stats, fallback_used=True)
        )

    monkeypatch.setattr(agent.planner, "plan", fake_plan)
    m = run_mission(grid, agent, cfg, seed=3)
    stats = agent.stats
    assert stats.plans_made >= 2  # the initial plan and at least one replan
    assert stats.planner_fallbacks == stats.plans_made
    assert m.metrics().planner_fallbacks == stats.plans_made


# --- PathAgent branches (F22) -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["astar_fixed", "astar_replan"])
def test_path_agent_anticipated_off_path(name: str) -> None:
    """Off the expected cell, the anticipated cell is simply ``pos + action``."""
    agent = start_agent(name, load_deck(), UniformSlip(DECK_P))
    path = agent.snapshots[-1].path
    assert path is not None and path[0] == (0, 0)  # the expected cell is the start
    assert (3, 0) not in path
    assert agent.anticipated((3, 0), Action.E) == frozenset({(3, 1)})
    assert agent.anticipated((3, 0), Action.N) == frozenset({(2, 0)})


@pytest.mark.parametrize("name", ["astar_fixed", "astar_replan"])
def test_path_agent_policy_arrows_without_path(name: str) -> None:
    agent = start_agent(name, TrueGrid.from_ascii("S#P"), UniformSlip(DECK_P))
    assert agent.snapshots[-1].path is None
    assert agent.policy_arrows() == {}
