"""Phase 3 acceptance: invariants over 1,000 seeded missions (brief §2.4, §9; plan D6, D10).

* ``andor`` (r >= 1): every mission is rescued with ``unplanned_states == 0``, ``bumps == 0``
  and ``planner_fallbacks == 0`` — across slip p, sensor radius, replan mode, aftershocks and
  high-slip terrain. Plans are validated (``PlannerConfig(validate=True)``), so a plan that is
  not strong-cyclic would be replaced by the reference planner and show up as a fallback.
* ``astar_replan`` on the same missions: no bumps, always rescued.
* ``astar_fixed``: every failure is ``aborted`` and happens at the first deviation.
* Same seed => identical trace (a re-run sample).
* With r = 0 the safety net is real: bumps / unplanned states happen, nothing crashes.

The 1,000 missions are split into chunks so pytest-xdist can spread them. Each chunk also
checks that it was not vacuous: aftershocks fired, andor replanned and slipped, and
``astar_fixed`` both aborted (on a deviation and on sensed debris ahead) and rescued.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rescue.agents import make_agent
from rescue.config import AftershockConfig, AgentConfig, PlannerConfig, WorldConfig
from rescue.grid import TrueGrid
from rescue.model import OutcomeKind
from rescue.runner import Mission, MissionStatus, format_trace, make_transition, run_mission
from tests.helpers_maps import random_solvable_grid

N_MISSIONS = 1_000
N_CHUNKS = 10
CHUNK = N_MISSIONS // N_CHUNKS
MASTER_SEED = 20_261_006
SLIP_PS = (0.0, 0.1, 0.2, 0.3, 0.4)
AFTERSHOCK_QS = (0.05, 0.2)
TERRAIN_FRACTION = 0.2
#: Every RERUN_EVERY-th mission is re-run to check same seed => same trace.
RERUN_EVERY = 25


@dataclass(frozen=True)
class Knobs:
    """The configuration of mission ``index`` (everything but the map itself)."""

    index: int
    rows: int
    cols: int
    density: float
    terrain: float
    world: WorldConfig
    mode: str
    seed: int


@dataclass(frozen=True)
class Spec:
    """One seeded mission configuration."""

    index: int
    grid: TrueGrid
    world: WorldConfig
    mode: str
    seed: int


def mission_knobs(i: int) -> Knobs:
    """Deterministic knobs of mission ``i``: categorical ones cycle, continuous ones are drawn."""
    rng = np.random.default_rng([MASTER_SEED, i])
    rows = int(rng.integers(5, 11))
    cols = int(rng.integers(5, 11))
    density = float(rng.uniform(0.1, 0.35))
    p = SLIP_PS[i % len(SLIP_PS)]
    radius = 1 + (i // len(SLIP_PS)) % 2
    mode = ("eager", "lazy")[(i // 10) % 2]
    shock = (i // 20) % 4  # 0, 1: none; 2: q = 0.05; 3: q = 0.2
    q = AFTERSHOCK_QS[shock - 2] if shock >= 2 else 0.0
    terrain = TERRAIN_FRACTION if (i // 80) % 3 == 0 else 0.0
    world = WorldConfig(slip_p=p, sensor_radius=radius, aftershock=AftershockConfig(q=q, k=1))
    return Knobs(index=i, rows=rows, cols=cols, density=density, terrain=terrain, world=world,
                 mode=mode, seed=i)


def knobs_grid(k: Knobs) -> TrueGrid:
    """The seeded random map of mission ``k.index``."""
    return random_solvable_grid(k.rows, k.cols, k.density, seed=MASTER_SEED + k.index,
                                terrain_fraction=k.terrain)


def mission_spec(i: int) -> Spec:
    """Deterministic mission ``i`` (knobs plus its map)."""
    k = mission_knobs(i)
    return Spec(index=i, grid=knobs_grid(k), world=k.world, mode=k.mode, seed=k.seed)


def run(spec: Spec, name: str, *, world: WorldConfig | None = None) -> Mission:
    """Run agent ``name`` on ``spec`` (optionally overriding the world config).

    andor plans are validated (``is_strong_cyclic``); the A* agents ignore the planner config.
    """
    cfg = world or spec.world
    agent = make_agent(
        name,
        make_transition(spec.grid, cfg),
        PlannerConfig(validate=True),
        AgentConfig(replan_mode=spec.mode),
    )
    return run_mission(spec.grid, agent, cfg, spec.seed, scenario="invariants")


def chunk_specs(chunk: int) -> list[Spec]:
    return [mission_spec(i) for i in range(chunk * CHUNK, (chunk + 1) * CHUNK)]


def test_specs_cover_the_matrix() -> None:
    """The knobs cover the whole matrix (no maps built: the chunks build and run them)."""
    knobs = [mission_knobs(i) for i in range(N_MISSIONS)]
    assert {k.world.slip_p for k in knobs} == set(SLIP_PS)
    assert {k.world.sensor_radius for k in knobs} == {1, 2}
    assert {k.mode for k in knobs} == {"eager", "lazy"}
    assert {k.world.aftershock.q for k in knobs} == {0.0, *AFTERSHOCK_QS}
    assert {k.terrain for k in knobs} == {0.0, TERRAIN_FRACTION}
    assert {k.rows for k in knobs} == set(range(5, 11))
    assert {k.cols for k in knobs} == set(range(5, 11))
    # Terrain knobs really produce '~' cells (and the grids honour the drawn size).
    terrain = [k for k in knobs if k.terrain > 0][:5]
    grids = [knobs_grid(k) for k in terrain]
    assert any(g.high_slip.any() for g in grids)
    assert all((g.rows, g.cols) == (k.rows, k.cols) for g, k in zip(grids, terrain, strict=True))
    # Every chunk mixes aftershock and quiet missions.
    for chunk in range(N_CHUNKS):
        qs = {knobs[i].world.aftershock.q for i in range(chunk * CHUNK, (chunk + 1) * CHUNK)}
        assert qs == {0.0, *AFTERSHOCK_QS}, chunk


@pytest.mark.parametrize("chunk", range(N_CHUNKS))
def test_invariants(chunk: int) -> None:
    totals = dict.fromkeys(
        ("aftershocks", "replans", "slips", "fixed_deviation_aborts", "fixed_debris_aborts",
         "fixed_rescues"),
        0,
    )
    for spec in chunk_specs(chunk):
        ctx = (spec.index, spec.world, spec.mode)

        # --- andor: the Phase 3 acceptance criterion --------------------------------------
        m = run(spec, "andor")
        metrics = m.metrics()
        assert m.status is MissionStatus.RESCUED, ctx
        assert metrics.success, ctx
        assert metrics.unplanned_states == 0, ctx
        assert metrics.bumps == 0, ctx
        assert metrics.planner_fallbacks == 0, ctx  # validated: no unsound plan either
        assert metrics.wasted_moves == metrics.steps - metrics.shortest_path, ctx
        totals["aftershocks"] += metrics.aftershock_events
        totals["replans"] += metrics.replans
        totals["slips"] += metrics.slips + metrics.skids
        if spec.index % RERUN_EVERY == 0:
            again = run(spec, "andor")
            assert again.events == m.events, ctx
            assert format_trace(again.events) == format_trace(m.events), ctx

        # --- astar_replan: never bumps, always gets there -----------------------------------
        r = run(spec, "astar_replan")
        assert r.status is MissionStatus.RESCUED, ctx
        assert r.metrics().bumps == 0, ctx

        # --- astar_fixed: fails only by aborting, at the first deviation --------------------
        f = run(spec, "astar_fixed")
        fm = f.metrics()
        assert fm.bumps == 0, ctx
        assert f.status in (MissionStatus.RESCUED, MissionStatus.ABORTED), ctx
        if not fm.success:
            assert fm.failure_reason == "aborted", ctx
            totals["fixed_deviation_aborts" if fm.unplanned_states else "fixed_debris_aborts"] += 1
        else:
            totals["fixed_rescues"] += 1
        moves = [e for e in f.events if e.action is not None]
        deviations = [
            k for k, e in enumerate(moves) if e.outcome is not OutcomeKind.ARRIVED or e.blocked
        ]
        assert fm.unplanned_states == min(1, len(deviations)), ctx
        if deviations:
            first = deviations[0]
            # Nothing is executed after the first deviation.
            assert len(moves) == first + 1, ctx
            if moves[first].new_pos != spec.grid.survivor:
                assert f.status is MissionStatus.ABORTED, ctx
        if spec.index % RERUN_EVERY == 0:
            assert run(spec, "astar_fixed").events == f.events, ctx

    # Non-vacuity: the invariants above were exercised by what they are about.
    for key, total in totals.items():
        assert total > 0, (chunk, key, totals)


def test_safety_net_is_real_with_blind_sensor() -> None:
    """r = 0 (testing only, D7): unseen debris causes bumps / unplanned states, yet the andor
    agent never crashes and every mission ends rescued or out of budget."""
    bumps = unplanned = 0
    for i in range(60):
        spec = mission_spec(i)
        cfg = WorldConfig(slip_p=spec.world.slip_p, sensor_radius=0)
        m = run(spec, "andor", world=cfg)
        assert m.status in (MissionStatus.RESCUED, MissionStatus.BUDGET), spec.index
        bumps += m.metrics().bumps
        unplanned += m.metrics().unplanned_states
    assert bumps > 0
    assert unplanned > 0
