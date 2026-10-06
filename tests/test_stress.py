"""Phase 5 stress simulations (brief §6.4, plan D4, D9).

* Aftershocks never make the survivor unreachable: missions of ``andor`` and ``astar_replan``
  on every generated scenario with ``AftershockConfig(q ∈ {0.1, 0.3}, k ∈ {1, 3})``; after
  *every* step the survivor is BFS-reachable from the robot on the world's current true map,
  and no aftershock cell lies in the sensor footprint, on the robot, survivor or start.
  ``andor`` keeps its invariants (0 unplanned states, 0 bumps) and is rescued.
* Terrain-dependent slip: on ``flooded_zone`` maps the empirical non-arrived frequency is
  ``terrain_high_p`` on '~' cells and ``slip_p`` elsewhere (exact binomial / chi-square
  tests over many aggregated steps).
* Sensor radius sweep r = 1 vs 2 and slip sweep p ∈ {0, 0.1, 0.2, 0.3, 0.4}.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy import stats

from rescue.agents import make_agent
from rescue.config import AftershockConfig, WorldConfig
from rescue.grid import Cell, TrueGrid, footprint
from rescue.model import OUTCOME_ORDER, OutcomeKind, outcome_distribution
from rescue.planners.bfs import reachable
from rescue.runner import Mission, MissionStatus, make_transition, run_mission

mapgen = pytest.importorskip("rescue.mapgen")

SIZE = (12, 12)
GENERATED = ("random_rubble", "collapsed_building", "flooded_zone", "maze_corridors")
AFTERSHOCK_GRID = [(q, k) for q in (0.1, 0.3) for k in (1, 3)]
STRESS_SEEDS = (0, 1, 2)
STRESS_AGENTS = ("andor", "astar_replan")
#: Significance level of the statistical checks (fixed seeds: deterministic outcome).
ALPHA = 1e-4


def make_mission(grid: TrueGrid, agent_name: str, world: WorldConfig, seed: int) -> Mission:
    """A mission with the agent built on the world's transition model (D5)."""
    agent = make_agent(agent_name, make_transition(grid, world))
    return Mission(grid, agent, world_config=world, seed=seed, scenario=grid.name)


# --- aftershocks -------------------------------------------------------------------------------
@pytest.mark.parametrize("agent_name", STRESS_AGENTS)
@pytest.mark.parametrize("scenario", GENERATED)
def test_aftershocks_never_disconnect_survivor(scenario: str, agent_name: str) -> None:
    rows, cols = SIZE
    shocks = 0
    for q, k in AFTERSHOCK_GRID:
        for seed in STRESS_SEEDS:
            grid = mapgen.generate(scenario, rows, cols, seed)
            world_cfg = WorldConfig(aftershock=AftershockConfig(q=q, k=k))
            mission = make_mission(grid, agent_name, world_cfg, seed=1000 + seed)
            mission.start()
            world = mission.world
            survivor, start = grid.survivor, grid.start
            assert reachable(world.grid.passable_mask(), world.pos, survivor)
            while not mission.done:
                before = world.grid.cells.copy()
                event = mission.step()
                robot = world.pos
                assert event.new_pos == robot
                assert reachable(world.grid.passable_mask(), robot, survivor), (
                    f"{scenario}/{agent_name} q={q} k={k} seed={seed} t={event.t}: "
                    "survivor disconnected"
                )
                shocked = event.aftershock_cells
                assert len(shocked) <= k
                assert len(set(shocked)) == len(shocked)
                sensed = set(footprint(robot, world_cfg.sensor_radius, rows, cols))
                for p in shocked:
                    assert p not in sensed
                    assert p not in (robot, survivor, start)
                    assert before[p] == Cell.OPEN
                    assert world.grid.cells[p] == Cell.DEBRIS
                    assert not world.grid.high_slip[p]
                shocks += len(shocked)
            m = mission.metrics()
            assert m.bumps == 0
            assert m.success, f"{scenario}/{agent_name} q={q} k={k} seed={seed}: {m}"
            if agent_name == "andor":
                assert m.unplanned_states == 0
            assert m.aftershock_events == sum(1 for e in mission.events if e.aftershock_cells)
    assert shocks > 0, "no aftershock ever fired: the test is vacuous"


def test_aftershocks_reach_belief_as_contradictions() -> None:
    """Some aftershock debris is later sensed and contradicts an OPEN belief (D8)."""
    contradictions = 0
    for seed in range(6):
        grid = mapgen.generate("random_rubble", 12, 12, seed)
        world_cfg = WorldConfig(aftershock=AftershockConfig(q=0.3, k=3))
        mission = run_mission(grid, make_agent("andor", make_transition(grid, world_cfg)),
                              world_cfg, seed=seed)
        contradictions += mission.agent.stats.contradictions
        assert mission.status is MissionStatus.RESCUED
    assert contradictions > 0


# --- terrain-dependent slip ----------------------------------------------------------------------
def test_terrain_dependent_slip_frequencies() -> None:
    base_p, high_p = 0.1, 0.4
    world_cfg = WorldConfig(slip_p=base_p, terrain_high_p=high_p)
    counts = {True: Counter(), False: Counter()}
    for seed in range(120):
        grid = mapgen.generate("flooded_zone", 12, 12, seed)
        assert grid.high_slip.any()
        transition = make_transition(grid, world_cfg)
        mission = run_mission(grid, make_agent("andor", transition), world_cfg, seed=seed)
        assert mission.status is MissionStatus.RESCUED
        for event in mission.events:
            if event.action is None or event.outcome is None:
                continue
            on_water = bool(grid.high_slip[event.pos])  # D4: the cell the robot stands on
            counts[on_water][event.outcome] += 1
    for on_water, p in ((True, high_p), (False, base_p)):
        c = counts[on_water]
        n = sum(c.values())
        assert n >= 300, f"too few steps on {'water' if on_water else 'land'}: {n}"
        non_arrived = n - c[OutcomeKind.ARRIVED]
        test = stats.binomtest(non_arrived, n, p)
        assert test.pvalue > ALPHA, (on_water, non_arrived / n, p)
        # Full four-way split (arrived / slipped / skid-L / skid-R) by chi-square.
        expected = np.array([prob for _, prob in outcome_distribution(p)]) * n
        observed = np.array([c[kind] for kind in OUTCOME_ORDER])
        chi2 = stats.chisquare(observed, expected)
        assert chi2.pvalue > ALPHA, (on_water, observed, expected)
    # The two empirical rates are clearly different.
    rate = {w: 1 - counts[w][OutcomeKind.ARRIVED] / sum(counts[w].values()) for w in counts}
    assert rate[True] > rate[False] + 0.15


def test_flooded_missions_use_terrain_slip_model() -> None:
    grid = mapgen.generate("flooded_zone", 12, 12, 0)
    world_cfg = WorldConfig(slip_p=0.1, terrain_high_p=0.4)
    transition = make_transition(grid, world_cfg)
    water = tuple(int(x) for x in np.argwhere(grid.high_slip)[0])
    assert transition.slip_p(water) == pytest.approx(0.4)
    assert transition.slip_p(grid.start) == pytest.approx(0.1)


# --- sensor radius sweep -------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", GENERATED)
def test_sensor_radius_two_explores_more(scenario: str) -> None:
    explored = {1: [], 2: []}
    for seed in range(5):
        grid = mapgen.generate(scenario, 12, 12, seed)
        for radius in (1, 2):
            world_cfg = WorldConfig(sensor_radius=radius)
            mission = run_mission(grid, make_agent("andor", make_transition(grid, world_cfg)),
                                  world_cfg, seed=seed)
            m = mission.metrics()
            assert m.success and m.bumps == 0 and m.unplanned_states == 0
            explored[radius].append(m.cells_explored)
    per_mission = [b >= a for a, b in zip(explored[1], explored[2], strict=True)]
    assert sum(per_mission) >= len(per_mission) - 1, explored
    assert sum(explored[2]) > sum(explored[1])


def test_sensor_radius_initial_footprint() -> None:
    """At t = 0 alone, r = 2 already knows at least as much as r = 1 (superset footprint)."""
    grid = mapgen.generate("random_rubble", 12, 12, 3)
    known = {}
    for radius in (1, 2):
        world_cfg = WorldConfig(sensor_radius=radius)
        mission = make_mission(grid, "andor", world_cfg, seed=0)
        mission.start()
        known[radius] = {p for p, _ in mission.events[0].new_cells}
    assert known[1] <= known[2]
    assert len(known[2]) > len(known[1])


# --- slip sweep ----------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.3, 0.4])
def test_slip_sweep_andor_succeeds(p: float) -> None:
    for scenario in GENERATED:
        for seed in range(3):
            grid = mapgen.generate(scenario, 12, 12, seed)
            world_cfg = WorldConfig(slip_p=p, terrain_high_p=0.4 if p else 0.0)
            mission = run_mission(grid, make_agent("andor", make_transition(grid, world_cfg)),
                                  world_cfg, seed=seed)
            m = mission.metrics()
            assert m.success, (scenario, seed, p, m.failure_reason)
            assert m.bumps == 0 and m.unplanned_states == 0
            if p == 0.0:
                assert m.slips == 0 and m.skids == 0
                # deterministic moves: every move arrives where it was aimed (r = 1 senses the
                # target before the move, so nothing is ever blocked)
                moves = [e for e in mission.events if e.action is not None]
                assert moves
                assert all(e.outcome is OutcomeKind.ARRIVED and not e.blocked for e in moves)


@pytest.mark.parametrize("agent_name", ["andor", "astar_fixed", "astar_replan", "dstar_lite",
                                        "vi_expected"])
def test_p0_open_map_takes_the_shortest_path(agent_name: str) -> None:
    """p = 0 and no debris: nothing can surprise any agent, so steps == the BFS distance."""
    rows, cols = 9, 12
    for seed in range(4):
        rng = np.random.default_rng([31, seed])
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        a, b = rng.choice(len(cells), 2, replace=False)
        lines = [["."] * cols for _ in range(rows)]
        lines[cells[a][0]][cells[a][1]] = "S"
        lines[cells[b][0]][cells[b][1]] = "P"
        grid = TrueGrid.from_ascii("\n".join("".join(row) for row in lines) + "\n")
        world_cfg = WorldConfig(slip_p=0.0)
        mission = run_mission(grid, make_agent(agent_name, make_transition(grid, world_cfg)),
                              world_cfg, seed=seed)
        m = mission.metrics()
        assert m.success and m.replans == 0
        assert m.steps == m.shortest_path == abs(cells[a][0] - cells[b][0]) + abs(
            cells[a][1] - cells[b][1])
