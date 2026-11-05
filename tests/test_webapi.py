"""The web app's Python bridge (``rescue.webapi``, ``rescue.webmaps``, ``rescue.webtext``;
docs/plan.md §3.16).

1. Fidelity: a :class:`WebSession` runs exactly the mission the CLI builds (same events, same
   metrics) for many maps, seeds, agents and modes; the slide 9 preset reproduces the demo.
2. The state is JSON-serializable, 0-indexed, and every text in it is 1-indexed.
3. ``state_at(t)`` equals the state returned live at ``t`` (property over seeds).
4. Explanations exist for every step and name the right cells and agents.
5. Map editing and validation.
6. The pre-generated map library equals ``mapgen.generate``.
7. ``rescue.webapi`` imports and runs library sessions with pygame, matplotlib, scipy and
   PyYAML blocked.
8. Performance: a 20x20 AND-OR session steps in well under 50 ms (median).
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rescue import mapgen, webapi, webmaps
from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES, make_agent
from rescue.config import (
    DEMO_SEED,
    EXT_DEMO_MULTI_K,
    EXT_DEMO_SCENARIO,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
    MAPGEN_DEFAULT_SIZE,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.ext.agents import EXT_AGENT_NAMES, make_ext_agent
from rescue.ext.demo import MODES as EXT_DEMO_MODES
from rescue.ext.mission import ExtMission
from rescue.ext.scenarios import add_survivors, multi_survivor_grid
from rescue.grid import TrueGrid, fmt_pos
from rescue.runner import Mission, MissionStatus, make_transition
from rescue.webapi import WebSession, catalog, dispatch, normalize_config

#: Repository root, maps and the recorded ``python -m rescue.demo`` output.
REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
#: Library subset generated for the tests (the full library takes the build a while).
TEST_LIBRARY_SIZES = (10, EXT_DEMO_SIZE[0])
TEST_LIBRARY_SEEDS = (0, 1, 2, EXT_DEMO_SEED)
#: World seeds of the fidelity sweeps.
FIDELITY_SEEDS = (0, 1, 2)
#: Seeds found by search (see the tests using them).
BUDGET_SEED = 5582  # "SP" at p = 0.4: eight failed moves in a row exhaust the budget of 8
FIXED_ABORT_SEED = 0  # astar_fixed on deck_demo slips at t = 1 and gives up
SKID_SEED = 4  # andor on deck_demo skids (not blocked) at t = 3
BLOCKED_SKID_SEED = 28  # andor on deck_demo skids into a wall at t = 7
LAZY_SEED = 0  # lazy andor on maze_01 senses debris off its envelope
NOISY_BUMP_SEED = 0  # search_andor, noisy mode: bumps into missed debris
#: Performance target (median per-step time of a 20x20 AND-OR session in CPython).
STEP_MEDIAN_LIMIT_S = 0.050
PERF_SIZE = 20
PERF_STEPS = 60
#: Most identical replays of the timed session: the per-step minimum over replays filters the
#: wall-clock noise of a loaded machine (pytest-xdist, other processes); the test stops at the
#: first replay that meets the target, as test_viz's frame-rate test does.
PERF_REPLAYS = 6
#: 1-indexed positions in texts, e.g. "(3,2)".
POS_PATTERN = re.compile(r"\((\d+),(\d+)\)")
#: Hypothesis health checks off for the state_at properties (whole missions per example; the
#: library fixture is the same for every example).
SLOW_CHECKS = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
#: Modules the web bridge must not need.
BLOCKED_MODULES = ("pygame", "matplotlib", "scipy", "yaml")


# --- fixtures and helpers ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def library_doc() -> dict[str, Any]:
    """A small library document (generated with mapgen)."""
    return webmaps.build_map_library(sizes=TEST_LIBRARY_SIZES, seeds=TEST_LIBRARY_SEEDS)


@pytest.fixture(autouse=True)
def _library(library_doc: dict[str, Any]) -> Iterator[None]:
    """Every test runs with the test library installed (and forgets it afterwards)."""
    webmaps.set_map_library(library_doc)
    yield
    webmaps.set_map_library(None)


@pytest.fixture(autouse=True)
def _quiet_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Bumps and replans log warnings by design; keep the test output readable."""
    caplog.set_level(logging.ERROR)


def run_session(config: dict[str, Any]) -> tuple[WebSession, dict[int, str], list[dict]]:
    """Run a session to the end; the live states (JSON, last per t) and every step result."""
    session = WebSession(config)
    live = {0: json.dumps(session.state(), sort_keys=True, allow_nan=False)}
    results: list[dict] = []
    while not session.done:
        result = session.step()
        results.append(result)
        live[result["state"]["t"]] = json.dumps(result["state"], sort_keys=True,
                                                allow_nan=False)
    return session, live, results


def metrics_without_timing(data: dict[str, Any]) -> dict[str, Any]:
    """Metrics with NaN normalised and the timing column dropped."""
    out = {}
    for key, value in data.items():
        if key == "planning_ms":
            continue
        if isinstance(value, float) and math.isnan(value):
            value = None
        out[key] = value
    return out


def cli_core_mission(config: dict[str, Any]) -> Mission:
    """The mission ``python -m rescue.sim`` / ``rescue.demo`` builds for a core config."""
    cfg = normalize_config(config)
    spec = cfg["map"]
    if spec["source"] == "file":
        grid = TrueGrid.load(MAPS_DIR / f"{spec['name']}.txt")
    elif spec["source"] == "scenario":
        grid = mapgen.generate(spec["scenario"], spec["rows"], spec["cols"], spec["seed"])
    else:
        grid = TrueGrid.from_ascii(spec["text"], name=spec["name"])
    world = WorldConfig(slip_p=cfg["slip_p"], sensor_radius=cfg["sensor_radius"],
                        aftershock=AftershockConfig(q=cfg["aftershock_q"],
                                                    k=cfg["aftershock_k"]))
    agent = make_agent(cfg["agent"], make_transition(grid, world), PlannerConfig(),
                       AgentConfig(replan_mode=cfg["replan_mode"]))
    mission = Mission(grid, agent, world, seed=cfg["seed"], scenario=grid.name)
    mission.run()
    return mission


def cli_ext_mission(config: dict[str, Any]) -> ExtMission:
    """The mission ``python -m rescue.sim --ext`` / ``rescue.ext.demo`` builds."""
    cfg = normalize_config(config)
    spec = cfg["map"]
    mode = EXT_DEMO_MODES[cfg["mode"]]
    if spec["source"] == "scenario":
        grid = multi_survivor_grid(spec["scenario"], spec["rows"], spec["cols"], spec["seed"],
                                   cfg["k"])
    else:
        grid = TrueGrid.load(MAPS_DIR / f"{spec['name']}.txt")
        if cfg["k"] > len(grid.survivors):
            grid = add_survivors(grid, cfg["seed"], cfg["k"])
    world = WorldConfig(slip_p=cfg["slip_p"], sensor_radius=cfg["sensor_radius"],
                        aftershock=AftershockConfig(q=cfg["aftershock_q"],
                                                    k=cfg["aftershock_k"]))
    agent = make_ext_agent(cfg["agent"], make_transition(grid, world), PlannerConfig(),
                           AgentConfig(replan_mode=cfg["replan_mode"]))
    mission = ExtMission(grid, agent, world, cfg["seed"],
                         survivors_known=mode.survivors_known,
                         false_negative=cfg["false_negative"],
                         false_positive=cfg["false_positive"],
                         scenario=grid.name, mode=cfg["mode"])
    mission.run()
    return mission


def assert_same_events(session: WebSession, events: list[Any]) -> None:
    """The session's events equal ``events``, and its JSON events describe them."""
    assert session.mission.events == events
    for i, event in enumerate(events):
        data = session._frames[i].event
        assert data["t"] == event.t
        assert data["pos"] == list(event.pos)
        assert data["new_pos"] == list(event.new_pos)
        assert data["action"] == (None if event.action is None else event.action.name)
        assert data["outcome"] == (None if event.outcome is None else event.outcome.value)
        assert data["status"] == event.status.value


def all_texts(state: dict[str, Any]) -> Iterator[str]:
    """Every human-readable text of a state."""
    yield from state["explanation"]
    yield from (line["text"] for line in state["log"])
    if state["plan_message"]:
        yield state["plan_message"]
    plan = state["plan"]
    if plan is not None:
        yield plan["pretty"]
        stack = [plan["tree"]]
        while stack:
            node = stack.pop()
            yield node["text"]
            if node["edge"]:
                yield node["edge"]
            stack.extend(node["children"])


def find_step(session_config: dict[str, Any], predicate: Any) -> dict[str, Any]:
    """The first step result of a fresh session satisfying ``predicate``."""
    session = WebSession(session_config)
    while not session.done:
        result = session.step()
        if predicate(result):
            return result
    raise AssertionError(f"no step matched in {session_config}")


# --- 1. fidelity -------------------------------------------------------------------------------
CORE_CASES: list[dict[str, Any]] = [
    {"map": {"source": "file", "name": name}, "agent": agent, "seed": seed}
    for name in ("deck_demo", "rubble_01")
    for agent in ALL_AGENT_NAMES
    for seed in FIDELITY_SEEDS
] + [
    {"map": {"source": "file", "name": "flooded_01"}, "agent": "andor", "seed": 3,
     "replan_mode": "lazy"},
    {"map": {"source": "file", "name": "maze_01"}, "agent": "astar_replan", "seed": 4,
     "sensor_radius": 2, "slip_p": 0.35},
    {"map": {"source": "scenario", "scenario": "collapsed_building", "rows": 10, "cols": 10,
             "seed": 1}, "agent": "andor", "aftershock_q": 0.3, "aftershock_k": 2},
    {"map": {"source": "scenario", "scenario": "random_rubble", "rows": 10, "cols": 10,
             "seed": 2}, "agent": "dstar_lite", "sensor_radius": 0},
    {"map": {"source": "scenario", "scenario": "flooded_zone", "rows": 12, "cols": 12,
             "seed": 0}, "agent": "vi_expected"},
]


@pytest.mark.parametrize("config", CORE_CASES, ids=lambda c: json.dumps(c, sort_keys=True))
def test_core_session_equals_cli_mission(config: dict[str, Any]) -> None:
    session, _, _ = run_session(config)
    direct = cli_core_mission(config)
    assert_same_events(session, direct.events)
    assert metrics_without_timing(session.mission.metrics().as_dict()) == \
        metrics_without_timing(direct.metrics().as_dict())
    assert metrics_without_timing(session.state()["metrics"]) == \
        metrics_without_timing(direct.metrics().as_dict())


EXT_CASES: list[dict[str, Any]] = [
    {"mode": mode, "agent": agent, "seed": seed,
     "map": {"source": "scenario", "scenario": scenario, "rows": 10, "cols": 10, "seed": seed}}
    for mode in ("search", "multi", "noisy")
    for agent in EXT_AGENT_NAMES
    for seed, scenario in zip(FIDELITY_SEEDS, ("collapsed_building", "random_rubble",
                                               "maze_corridors"), strict=True)
] + [
    {"mode": "noisy", "agent": "search_andor", "false_negative": 0.2, "false_positive": 0.1,
     "map": {"source": "scenario", "scenario": "flooded_zone", "rows": 12, "cols": 12,
             "seed": EXT_DEMO_SEED}, "seed": EXT_DEMO_SEED},
    {"mode": "multi", "agent": "search_astar", "k": 4, "aftershock_q": 0.2},
    {"mode": "multi", "agent": "search_andor", "map": {"source": "file", "name": "rubble_01"}},
    {"mode": "search", "agent": "search_astar", "replan_mode": "lazy", "sensor_radius": 2},
]


@pytest.mark.parametrize("config", EXT_CASES, ids=lambda c: json.dumps(c, sort_keys=True))
def test_ext_session_equals_cli_mission(config: dict[str, Any]) -> None:
    session, _, _ = run_session(config)
    direct = cli_ext_mission(config)
    assert_same_events(session, direct.events)
    assert metrics_without_timing(session.mission.metrics().as_dict()) == \
        metrics_without_timing(direct.metrics().as_dict())


def test_slide9_preset_reproduces_the_demo() -> None:
    preset = next(p for p in catalog()["presets"] if p["id"] == "slide9")
    session, _, _ = run_session(preset["config"])
    report = session.report()
    demo = subprocess.run([sys.executable, "-m", "rescue.demo"], capture_output=True,
                          text=True, encoding="utf-8", cwd=REPO_ROOT, check=True,
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).stdout
    lines = demo.splitlines()
    assert report["trace"] in lines
    assert report["trace"] == lines[lines.index("Execution trace:") + 1]
    assert report["summary"] in demo
    assert f"Initial plan from (1,1): {report['initial_plan']}" in lines
    assert report["map"] in demo
    assert session.state()["status"] == "rescued"
    assert session.config["seed"] == DEMO_SEED


def test_presets_and_defaults_build() -> None:
    for preset in catalog()["presets"]:
        session = WebSession(preset["config"])
        assert session.state()["status"] == "running", preset["id"]
    for mode in webapi.MODES:
        state = WebSession({"mode": mode}).state()
        assert state["mode"] == mode
    ext = WebSession({"mode": "multi"})
    assert ext.config["map"] == {"source": "scenario", "scenario": EXT_DEMO_SCENARIO,
                                 "rows": EXT_DEMO_SIZE[0], "cols": EXT_DEMO_SIZE[1],
                                 "seed": EXT_DEMO_SEED}
    assert ext.config["k"] == EXT_DEMO_MULTI_K == len(ext.grid.survivors)


# --- 2. JSON and coordinates -------------------------------------------------------------------
JSON_CASES = [
    {},
    {"agent": "astar_fixed", "seed": FIXED_ABORT_SEED},
    {"agent": "astar_replan", "map": {"source": "file", "name": "building_01"}, "seed": 2},
    {"mode": "search", "map": {"source": "scenario", "scenario": "maze_corridors", "rows": 10,
                               "cols": 10, "seed": 1}},
    {"mode": "noisy", "false_positive": 0.1, "map": {"source": "scenario",
                                                     "scenario": "random_rubble",
                                                     "rows": 10, "cols": 10, "seed": 0}},
]


@pytest.mark.parametrize("config", JSON_CASES, ids=lambda c: json.dumps(c, sort_keys=True))
def test_state_is_json_and_texts_are_one_indexed(config: dict[str, Any]) -> None:
    session = WebSession(config)
    states = [session.state()]
    while not session.done:
        states.append(session.step()["state"])
    rows, cols = session.grid.rows, session.grid.cols
    for state in states:
        text = json.dumps(state, allow_nan=False)
        assert json.loads(text) == state  # only JSON types (no tuples, numpy, NaN)
        assert state["api"] == webapi.API_VERSION
        assert len(state["cells"]) == len(state["belief"]) == rows
        assert all(len(row) == cols for row in state["cells"])
        r, c = state["robot"]
        assert 0 <= r < rows and 0 <= c < cols
        for key in ("robot", "start"):
            assert isinstance(state[key], list)
        for cell in state["envelope"] + state["survivors"] + state["rescued"]:
            assert 0 <= cell[0] < rows and 0 <= cell[1] < cols
        for key in state["policy"]:
            pr, pc = (int(v) for v in key.split(","))
            assert 0 <= pr < rows and 0 <= pc < cols
        for line in all_texts(state):
            for match in POS_PATTERN.finditer(line):
                tr, tc = int(match.group(1)), int(match.group(2))
                assert 1 <= tr <= rows and 1 <= tc <= cols, line
    final = states[-1]
    assert final["robot"] == list(session.mission.world.pos)


def test_deck_demo_state_at_t3_is_0_indexed_with_1_indexed_texts() -> None:
    session = WebSession({})
    for _ in range(3):
        session.step()
    state = session.state_at(3)
    assert state["robot"] == [1, 1]  # (2,2) for humans
    assert state["start"] == [0, 0] and state["survivors"] == [[3, 3]]
    assert state["event"]["triggering_contradictions"] == [[2, 1]]
    assert "Sensors revealed debris at (3,2)" in " ".join(state["explanation"])
    assert state["plan"]["tree"]["text"] == "E @ (2,2)"
    assert state["plan"]["tree"]["state"] == [1, 1]
    assert state["policy"]["1,1"] == "E"
    assert state["log"][-1] == {"t": 3, "kind": "replan", "tag": "replan",
                                "text": "Debris sensed at (3,2) → replanning"}


# --- 3. state_at -------------------------------------------------------------------------------
def check_state_at(config: dict[str, Any]) -> None:
    """``state_at(t)`` equals the live state at every t, and the live state matches the
    mission objects."""
    session = WebSession(config)
    mission = session.mission
    live = {0: json.dumps(session.state(), sort_keys=True)}

    def check_against_mission(state: dict[str, Any]) -> None:
        agent = mission.agent
        assert state["robot"] == list(mission.world.pos)
        assert state["belief"] == np.asarray(agent.belief.grid).tolist()
        assert state["cells"] == mission.world.grid.cells.tolist()
        expected_policy = {f"{r},{c}": a.name for (r, c), a in agent.policy_arrows().items()}
        assert state["policy"] == expected_policy
        p_debris = getattr(agent.belief, "p_debris", None)
        if p_debris is None:
            assert state["p_debris"] is None
        else:  # float32 in the frames, rounded in the JSON
            assert np.allclose(state["p_debris"], p_debris, atol=1e-6)
        assert metrics_without_timing(state["metrics"]) == \
            metrics_without_timing(mission.metrics().as_dict())
        assert state["plan_id"] == agent.current_plan_id

    check_against_mission(session.state())
    while not session.done:
        state = session.step()["state"]
        check_against_mission(state)
        live[state["t"]] = json.dumps(state, sort_keys=True)
    for t, text in live.items():
        assert json.dumps(session.state_at(t), sort_keys=True) == text, t
    assert session.state_at(session.t) == session.state()


@settings(max_examples=12, deadline=None, suppress_health_check=SLOW_CHECKS)
@given(seed=st.integers(min_value=0, max_value=200),
       agent=st.sampled_from(ALL_AGENT_NAMES),
       aftershock=st.booleans())
def test_state_at_equals_live_state_core(seed: int, agent: str, aftershock: bool) -> None:
    check_state_at({"map": {"source": "file", "name": "rubble_01"}, "agent": agent,
                    "seed": seed, "aftershock_q": 0.3 if aftershock else 0.0})


@settings(max_examples=12, deadline=None, suppress_health_check=SLOW_CHECKS)
@given(seed=st.sampled_from(TEST_LIBRARY_SEEDS),
       mode=st.sampled_from(("search", "multi", "noisy")),
       agent=st.sampled_from(EXT_AGENT_NAMES))
def test_state_at_equals_live_state_ext(seed: int, mode: str, agent: str) -> None:
    check_state_at({"mode": mode, "agent": agent, "seed": seed, "false_positive": 0.1,
                    "map": {"source": "scenario", "scenario": "collapsed_building",
                            "rows": 10, "cols": 10, "seed": seed}})


def test_state_at_on_an_abort_returns_the_final_state() -> None:
    session, live, _ = run_session({"agent": "astar_fixed", "seed": FIXED_ABORT_SEED})
    assert session.state()["status"] == "aborted"
    assert session.frame_count() == session.t + 2  # the abort event shares the last t
    assert json.dumps(session.state_at(session.t), sort_keys=True) == live[session.t]
    with pytest.raises(ValueError, match="t must be between"):
        session.state_at(session.t + 1)


# --- 4. explanations ---------------------------------------------------------------------------
def joined(result: dict[str, Any]) -> str:
    return " ".join(result["explanation"])


def test_every_step_is_explained() -> None:
    for config in [*CORE_CASES[:: len(FIDELITY_SEEDS)], *EXT_CASES[:6]]:
        session = WebSession(config)
        assert session.state()["explanation"]
        while not session.done:
            result = session.step()
            assert result["explanation"]
            assert all(isinstance(s, str) and s for s in result["explanation"])
            assert result["explanation"] == result["state"]["explanation"]


def test_explains_the_slide9_run() -> None:
    session = WebSession({})
    first = " ".join(session.state()["explanation"])
    assert "AND-OR" in first and "(4,4)" in first and "skid→(2,1)" in first
    results = [session.step() for _ in range(9)]
    slip = joined(results[1])
    assert "slipped" in slip and "(1,2)" in slip
    assert "retry branch" in slip and "no replanning was needed" in slip
    contradiction = joined(results[2])
    assert "Sensors revealed debris at (3,2)" in contradiction
    assert "replan #1" in contradiction
    assert "replan #2" in joined(results[5]) and "(4,3)" in joined(results[5])
    assert "Rescue successful" in joined(results[-1]) and "(4,4)" in joined(results[-1])


def test_explains_skids() -> None:
    skid = find_step({"seed": SKID_SEED}, lambda r: str(r["event"]["outcome"]).startswith("skid")
                     and not r["event"]["blocked"])
    to = fmt_pos(tuple(skid["event"]["new_pos"]))
    assert "skidded" in joined(skid) and f"branch for a skid to {to}" in joined(skid)
    blocked = find_step({"seed": BLOCKED_SKID_SEED},
                        lambda r: str(r["event"]["outcome"]).startswith("skid")
                        and r["event"]["blocked"])
    assert "the edge of the map" in joined(blocked) or "debris" in joined(blocked)


def test_explains_a_star_replanning_after_a_slip() -> None:
    for agent, name in (("astar_replan", "A*"), ("dstar_lite", "D* Lite")):
        result = find_step({"agent": agent, "seed": 1},
                           lambda r: r["event"]["outcome"] == "slipped")
        pos = tuple(result["event"]["pos"])
        intended = (pos[0] + 1, pos[1]) if result["event"]["action"] == "S" else None
        text = joined(result)
        assert f"{name} expected" in text
        assert f"it must replan from {fmt_pos(pos)}" in text
        if intended is not None:
            assert f"expected {fmt_pos(intended)}" in text


def test_explains_fixed_a_star_giving_up() -> None:
    _, _, results = run_session({"agent": "astar_fixed", "seed": FIXED_ABORT_SEED})
    deviation = joined(results[-2])
    assert "cannot recover from a deviation" in deviation
    final = joined(results[-1])
    assert "Fixed A* gives up" in final and "deviated from the path" in final
    assert results[-1]["state"]["status"] == "aborted"


def test_explains_budget_exhaustion_and_unreachable_maps() -> None:
    _, _, results = run_session({"map": {"source": "ascii", "text": "SP\n"}, "slip_p": 0.4,
                                 "seed": BUDGET_SEED})
    assert results[-1]["state"]["status"] == "budget"
    assert "step budget of 8 steps is used up" in joined(results[-1])
    walled = WebSession({"map": {"source": "ascii", "text": "S#.P\n#...\n"}})
    start = " ".join(walled.state()["explanation"])
    assert "No plan exists" in start and "(1,4)" in start
    assert walled.state()["plan_message"] is not None
    final = walled.step()
    assert final["state"]["status"] == "unreachable"
    assert "no plan reaches the survivor from (1,1)" in joined(final)


def test_explains_lazy_replanning_and_aftershocks() -> None:
    lazy_map = {"seed": LAZY_SEED, "replan_mode": "lazy",
                "map": {"source": "file", "name": "maze_01"}}
    lazy = find_step(lazy_map, lambda r: r["event"]["contradictions"]
                     and not r["event"]["replanned"] and r["state"]["status"] == "running")
    cells = ", ".join(fmt_pos(tuple(p)) for p in lazy["event"]["contradictions"])
    assert f"debris at {cells}, but it is not on the current plan" in joined(lazy)
    assert "lazy replanning keeps the plan" in joined(lazy)
    both = find_step(lazy_map, lambda r: r["event"]["replanned"]
                     and r["state"]["status"] == "running"
                     and len(r["event"]["contradictions"])
                     > len(r["event"]["triggering_contradictions"]))
    assert "the plan is recomputed" in joined(both)
    assert "lazy replanning alone would ignore it" in joined(both)
    shock = find_step({"aftershock_q": 0.5, "map": {"source": "file", "name": "rubble_01"}},
                      lambda r: r["event"]["aftershock_cells"])
    cells = ", ".join(fmt_pos(tuple(p)) for p in shock["event"]["aftershock_cells"])
    assert f"aftershock dropped new debris at {cells}" in joined(shock)


def test_explains_frontier_search_and_sightings() -> None:
    session = WebSession({"mode": "search"})
    state = session.state()
    target = fmt_pos(tuple(state["target"]))
    assert state["target_mode"] == "explore"
    assert f"frontier cell {target}" in " ".join(state["explanation"])
    found = None
    while not session.done:
        result = session.step()
        if result["event"]["found_now"]:
            found = result
            break
    assert found is not None
    where = fmt_pos(tuple(found["event"]["found_now"][0]))
    assert f"Survivor spotted at {where}" in joined(found)


def test_explains_multi_survivor_rescues() -> None:
    _, _, results = run_session({"mode": "multi", "agent": "search_astar"})
    rescues = [r for r in results if r["event"]["rescued_now"]]
    assert len(rescues) == EXT_DEMO_MULTI_K
    first = joined(rescues[0])
    assert f"at {fmt_pos(tuple(rescues[0]['event']['rescued_now']))}: 1 of 3 rescued" in first
    assert "Next target: the survivor at" in first
    assert f"All {EXT_DEMO_MULTI_K} survivors rescued" in joined(results[-1])


def test_explains_bumps_under_noise() -> None:
    bump = find_step({"mode": "noisy", "seed": NOISY_BUMP_SEED}, lambda r: r["event"]["bump"])
    event = bump["event"]
    action = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}[event["action"]]
    if event["outcome"] == "arrived":
        cell = (event["pos"][0] + action[0], event["pos"][1] + action[1])
        assert f"bumped into debris at {fmt_pos(cell)}" in joined(bump)
    assert "noisy sensor had missed" in joined(bump)
    assert any(line["text"].startswith("Bump into unseen debris")
               for line in bump["state"]["log"])


def test_explains_the_false_positive_fallback() -> None:
    config = {"mode": "noisy", "seed": 0, "false_positive": 0.3,
              "map": {"source": "scenario", "scenario": "maze_corridors", "rows": 12,
                      "cols": 12, "seed": 0}}
    result = find_step(config, lambda r: "temporarily treats uncertain" in joined(r))
    assert result["state"]["metrics"]["fallback_replans"] >= 1


# --- 5. map editing ----------------------------------------------------------------------------
BLANK = "S...\n....\n....\n...P\n"


def test_edit_ops() -> None:
    out = webapi.edit_map(BLANK, {"op": "toggle_debris", "cell": [1, 1]})
    assert out["ascii"] == "S...\n.#..\n....\n...P\n" and out["changed"] and out["solvable"]
    back = webapi.edit_map(out["ascii"], {"op": "toggle_debris", "cell": [1, 1]})
    assert back["ascii"] == BLANK
    refused = webapi.edit_map(BLANK, {"op": "toggle_debris", "cell": [0, 0]})
    assert not refused["changed"]
    assert refused["warnings"] == ["No debris on the start / survivor cell (1,1)"]
    moved = webapi.edit_map(BLANK, {"op": "set_start", "cell": [2, 2]})
    assert moved["ascii"] == "....\n....\n..S.\n...P\n"
    survivor = webapi.edit_map(BLANK, {"op": "set_survivor", "cell": [0, 3]})
    assert survivor["ascii"] == "S..P\n....\n....\n....\n"
    added = webapi.edit_map(BLANK, {"op": "add_survivor", "cell": [3, 0]})
    assert added["ascii"].count("P") == 2 and added["valid"]
    assert webapi.validate_map(added["ascii"])["ok"] is False  # core: one survivor only
    assert webapi.validate_map(added["ascii"], "multi")["ok"] is True
    removed = webapi.edit_map(added["ascii"], {"op": "remove_survivor", "cell": [3, 3]})
    assert removed["ascii"] == "S...\n....\n....\nP...\n"
    last = webapi.edit_map(BLANK, {"op": "remove_survivor", "cell": [3, 3]})
    assert not last["changed"] and "(4,4)" in last["warnings"][0]
    slip = webapi.edit_map(BLANK, {"op": "toggle_high_slip", "cell": [1, 2]})
    assert slip["ascii"] == "S...\n..~.\n....\n...P\n"
    painted = webapi.edit_map(BLANK, {"op": "paint", "cell": [2, 0], "value": "#"})
    assert painted["ascii"] == "S...\n....\n#...\n...P\n"
    bigger = webapi.edit_map(BLANK, {"op": "resize", "rows": 5, "cols": 6})
    assert bigger["ascii"] == "S.....\n......\n......\n...P..\n......\n"
    smaller = webapi.edit_map(BLANK, {"op": "resize", "rows": 2, "cols": 2})
    assert smaller["ascii"].count("S") == 1 and smaller["ascii"].count("P") == 1
    assert smaller["rows"] == 2 and smaller["valid"]
    cleared = webapi.edit_map("S#~\n##P\n", {"op": "clear"})
    assert cleared["ascii"] == "S..\n..P\n"


def test_edit_reports_unsolvable_maps_and_rejects_bad_requests() -> None:
    walled = webapi.edit_map("S.#.\n..#P\n", {"op": "toggle_debris", "cell": [0, 1]})
    assert walled["solvable"] is False
    assert walled["warnings"] == ["The survivor at (2,4) cannot be reached from the start (1,1)"]
    with pytest.raises(ValueError, match="unknown edit"):
        webapi.edit_map(BLANK, {"op": "explode", "cell": [0, 0]})
    with pytest.raises(ValueError, match="outside"):
        webapi.edit_map(BLANK, {"op": "toggle_debris", "cell": [9, 9]})
    with pytest.raises(ValueError, match="unknown map character"):
        webapi.edit_map("S.x\n..P\n", {"op": "clear"})
    with pytest.raises(ValueError, match="paint value"):
        webapi.edit_map(BLANK, {"op": "paint", "cell": [1, 1], "value": "P"})


def test_validate_map() -> None:
    ok = webapi.validate_map(BLANK)
    assert ok == {"ok": True, "errors": [], "warnings": [], "rows": 4, "cols": 4,
                  "survivors": [[3, 3]], "solvable": True, "problems": []}
    missing = webapi.validate_map("S...\n....\n")
    assert not missing["ok"] and missing["solvable"] is None and missing["rows"] == 2
    assert "survivor" in missing["errors"][0]
    walled = webapi.validate_map("S#P\n.#.\n")
    assert walled["ok"] and walled["solvable"] is False and walled["warnings"]
    assert walled["problems"] == [[0, 2]]  # the walled-in survivor, outlined by the editor
    typo = webapi.validate_map("; comment\nS..X\n...P\n")
    assert not typo["ok"] and typo["problems"] == [[0, 3]]
    assert "line 2, column 4" in typo["errors"][0]
    huge = "S" + "." * 59 + "\n" + ("." * 59 + "P\n")
    assert not webapi.validate_map(huge)["ok"]  # 60 columns: more than WEB_MAX_MAP_SIZE
    with pytest.raises(ValueError, match="at most"):
        WebSession({"map": {"source": "ascii", "text": huge}})
    with pytest.raises(ValueError, match="too long"):
        WebSession({"map": {"source": "ascii", "text": "S.P\n" + ";" * 9000}})
    # An edited map runs as a session; an unsolvable one ends "unreachable".
    session, _, _ = run_session({"map": {"source": "ascii", "text": "S#P\n.#.\n"}})
    assert session.state()["status"] == "unreachable"


def test_map_ascii_round_trips_bundled_and_generated_maps() -> None:
    text = webapi.map_ascii({"source": "file", "name": "deck_demo"})
    assert text == TrueGrid.load(MAPS_DIR / "deck_demo.txt").to_ascii()
    generated = webapi.map_ascii({"source": "scenario", "scenario": "random_rubble",
                                  "rows": 10, "cols": 10, "seed": 1})
    assert generated == mapgen.generate("random_rubble", 10, 10, 1).to_ascii()


# --- 6. map library ----------------------------------------------------------------------------
def test_library_entries_equal_mapgen(library_doc: dict[str, Any]) -> None:
    library = webmaps.MapLibrary.from_document(library_doc)
    assert set(library.scenarios) == set(mapgen.GENERATED_SCENARIOS)
    assert len(library.maps) == (len(mapgen.GENERATED_SCENARIOS) * len(TEST_LIBRARY_SIZES)
                                 * len(TEST_LIBRARY_SEEDS))
    for scenario in mapgen.GENERATED_SCENARIOS:
        for n in TEST_LIBRARY_SIZES:
            for seed in TEST_LIBRARY_SEEDS:
                assert library.grid(scenario, n, n, seed) == mapgen.generate(scenario, n, n,
                                                                             seed)


def test_library_constants_cover_the_defaults() -> None:
    sizes = webmaps.LIBRARY_SIZES
    assert {10, 15, 20} <= set(sizes)
    assert EXT_DEMO_SIZE[0] in sizes and MAPGEN_DEFAULT_SIZE[0] in sizes
    assert EXT_DEMO_SEED in webmaps.LIBRARY_SEEDS
    assert tuple(range(30)) == webmaps.LIBRARY_SEEDS


def test_library_multi_survivor_maps_equal_multi_survivor_grid(
        library_doc: dict[str, Any]) -> None:
    library = webmaps.MapLibrary.from_document(library_doc)
    for scenario in mapgen.GENERATED_SCENARIOS:
        for seed in TEST_LIBRARY_SEEDS:
            for k in (1, 2, 3, 5):
                base = library.grid(scenario, 10, 10, seed)
                assert base is not None
                grid = base if k == 1 else add_survivors(base, seed, k)
                assert grid == multi_survivor_grid(scenario, 10, 10, seed, k)
                session = WebSession({"mode": "multi", "k": k, "map": {
                    "source": "scenario", "scenario": scenario, "rows": 10, "cols": 10,
                    "seed": seed}})
                assert session.grid == grid


def test_library_file_round_trip_and_discovery(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    path = webmaps.write_map_library(tmp_path / webmaps.LIBRARY_FILENAME,
                                     scenarios=("maze_corridors",), sizes=(10,), seeds=(5,))
    webmaps.set_map_library(None)
    monkeypatch.chdir(tmp_path)
    library = webmaps.get_map_library()
    assert library is not None and library.has("maze_corridors", 10, 10, 5)
    assert not webmaps.needs_generator("maze_corridors", 10, 10, 5)
    assert webmaps.needs_generator("maze_corridors", 10, 10, 6)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    with pytest.raises(ValueError, match="unsupported map library version"):
        webmaps.MapLibrary.from_document({"version": 99, "maps": {}})


# --- catalog, config and dispatch --------------------------------------------------------------
def test_catalog() -> None:
    data = catalog()
    json.dumps(data, allow_nan=False)
    assert data["api"] == webapi.API_VERSION
    names = [a["name"] for a in data["agents"]]
    assert names == [*ALL_AGENT_NAMES, *EXT_AGENT_NAMES]
    kinds = {a["name"]: a["kind"] for a in data["agents"]}
    assert kinds["andor"] == "contingency" and kinds["astar_replan"] == "path"
    assert kinds["vi_expected"] == kinds["lao_expected"] == "expected"
    assert [a["name"] for a in data["agents"] if a["needs_scipy"]] == ["lao_expected"]
    assert [a["name"] for a in data["agents"] if a["main"]] == [*AGENT_NAMES, *EXT_AGENT_NAMES]
    maps = {m["name"]: m for m in data["maps"]}
    assert set(maps) == {p.stem for p in MAPS_DIR.glob("*.txt")}
    assert maps["deck_demo"]["rows"] == 4 and "slide 9" in maps["deck_demo"]["description"]
    assert maps["flooded_01"]["high_slip"]
    scenarios = {s["name"]: s for s in data["scenarios"]}
    assert set(scenarios) == set(mapgen.GENERATED_SCENARIOS)
    for name, entry in scenarios.items():
        assert entry["description"] == mapgen.SCENARIOS[name].description
    modes = {m["name"]: m for m in data["modes"]}
    assert list(modes) == ["core", "search", "multi", "noisy"]
    for name, spec in EXT_DEMO_MODES.items():
        defaults = modes[name]["defaults"]
        assert modes[name]["description"] == spec.description
        assert (defaults["k"], defaults["false_negative"], defaults["survivors_known"]) == \
            (spec.k, spec.noise, spec.survivors_known)
    assert data["params"]["slip_p"]["max"] == 0.4
    assert {p["id"] for p in data["presets"]} == {
        "slide9", "head_to_head", "hidden_survivor", "several_survivors", "noisy_sensors",
        "blank_canvas"}
    assert data["library"]["loaded"]


def test_config_validation() -> None:
    assert normalize_config({})["seed"] == DEMO_SEED
    generated = normalize_config({"map": {"source": "scenario", "scenario": "maze_corridors",
                                          "rows": 10, "cols": 10, "seed": 9}})
    assert generated["seed"] == 9  # the world seed defaults to the map seed
    assert normalize_config({"mode": "search"})["seed"] == EXT_DEMO_SEED
    bad: list[tuple[dict[str, Any], str]] = [
        ({"colour": "red"}, "unknown config keys"),
        ({"mode": "party"}, "unknown mode"),
        ({"agent": "search_andor"}, "not available in the core mode"),
        ({"mode": "multi", "agent": "andor"}, "not available in the multi mode"),
        ({"slip_p": 0.9}, "slip_p"),
        ({"k": 3}, "need an extension mode"),
        ({"mode": "noisy", "false_positive": 0.5}, "false_positive"),
        ({"map": {"source": "file", "name": "../secret"}}, "invalid map name"),
        ({"map": {"source": "file", "name": "nope"}}, "unknown map"),
        ({"map": {"source": "scenario", "scenario": "volcano"}}, "unknown scenario"),
        ({"map": {"source": "scenario", "scenario": "random_rubble", "rows": 3}}, "rows"),
        ({"map": {"source": "web"}}, "map source"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "whole number"),
        ({"sensor_radius": True}, "whole number"),
    ]
    for config, message in bad:
        with pytest.raises(ValueError, match=re.escape(message)):
            WebSession(config)


def test_needs_scipy() -> None:
    assert webapi.needs_scipy({}) == {"scipy": False, "reasons": [], "agent": None,
                                      "map": False}
    lao = webapi.needs_scipy({"agent": "lao_expected"})
    assert lao["scipy"] and lao["agent"] == "LAO* (expected cost)" and not lao["map"]
    off_library = {"map": {"source": "scenario", "scenario": "random_rubble", "rows": 11,
                           "cols": 11, "seed": 0}}
    assert webapi.needs_scipy(off_library)["scipy"]
    assert not webapi.needs_scipy({"map": {"source": "scenario", "scenario": "random_rubble",
                                           "rows": 10, "cols": 10, "seed": 0}})["scipy"]


def call(method: str, **params: Any) -> dict[str, Any]:
    return json.loads(dispatch(json.dumps({"method": method, "params": params})))


def test_dispatch_round_trip() -> None:
    opened = call("open", config={})
    assert opened["ok"]
    sid = opened["result"]["id"]
    step = call("step", id=sid)
    assert step["ok"] and step["result"]["state"]["t"] == 1
    rest = call("run", id=sid)
    assert rest["result"]["state"]["status"] == "rescued"
    assert len(rest["result"]["events"]) == rest["result"]["state"]["t"] - 1
    assert call("state_at", id=sid, t=3)["result"]["robot"] == [1, 1]
    assert call("state", id=sid)["result"]["status"] == "rescued"
    assert call("report", id=sid)["result"]["summary"].endswith("Rescue successful")
    again = call("step", id=sid)
    assert not again["ok"] and "mission is over" in again["error"]["message"]
    assert call("close", id=sid)["result"] == {"closed": True}
    missing = call("state", id=sid)
    assert not missing["ok"] and missing["error"]["type"] == "ValueError"
    assert call("catalog")["ok"]
    assert call("edit_map", ascii=BLANK, op={"op": "clear"})["result"]["ascii"] == BLANK
    assert call("validate_map", ascii=BLANK)["result"]["ok"]
    assert call("needs_scipy", config={"agent": "lao_expected"})["result"]["scipy"]
    assert call("map_ascii", spec={"source": "file", "name": "deck_demo"})["ok"]
    assert call("version")["result"]["api"] == webapi.API_VERSION
    unknown = json.loads(dispatch('{"method": "fly"}'))
    assert not unknown["ok"] and "unknown method" in unknown["error"]["message"]
    garbage = json.loads(dispatch("not json"))
    assert not garbage["ok"] and garbage["error"]["details"]


def test_dispatch_keeps_a_bounded_number_of_sessions() -> None:
    ids = [call("open", config={})["result"]["id"] for _ in range(webapi.WEB_MAX_SESSIONS + 2)]
    assert not call("state", id=ids[0])["ok"]
    assert call("state", id=ids[-1])["ok"]
    for sid in ids:
        call("close", id=sid)


def test_run_with_a_step_limit_and_step_after_the_end() -> None:
    session = WebSession({})
    partial = session.run(max_steps=4)
    assert len(partial["events"]) == 4 and partial["state"]["t"] == 4
    session.run()
    with pytest.raises(RuntimeError, match="mission is over"):
        session.step()
    assert session.run()["events"] == []


def test_status_values_match_mission_status() -> None:
    assert {s.value for s in MissionStatus} == {"running", "rescued", "aborted", "budget",
                                                "unreachable"}


# --- 7. no heavy imports -----------------------------------------------------------------------
BLOCKER_SCRIPT = textwrap.dedent("""
    import importlib.abc, json, sys
    BLOCKED = {blocked!r}

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}} (blocked)", name=name)
            return None

    sys.meta_path.insert(0, Blocker())
    from rescue import webapi, webmaps
    webmaps.set_map_library({library!r})
    assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
    json.dumps(webapi.catalog(), allow_nan=False)
    configs = [
        {{}},
        {{"agent": "vi_expected"}},
        {{"agent": "dstar_lite", "map": {{"source": "file", "name": "flooded_01"}}}},
        {{"map": {{"source": "scenario", "scenario": "collapsed_building", "rows": 10,
                   "cols": 10, "seed": 1}}, "agent": "astar_replan"}},
        {{"mode": "search", "map": {{"source": "scenario", "scenario": "maze_corridors",
                                     "rows": 10, "cols": 10, "seed": 2}}}},
        {{"mode": "multi", "k": 3, "agent": "search_astar"}},
        {{"mode": "noisy", "false_positive": 0.1}},
    ]
    for config in configs:
        assert not webapi.needs_scipy(config)["scipy"], config
        reply = json.loads(webapi.dispatch(json.dumps({{"method": "open",
                                                       "params": {{"config": config}}}})))
        assert reply["ok"], reply
        sid = reply["result"]["id"]
        done = json.loads(webapi.dispatch(json.dumps({{"method": "run",
                                                      "params": {{"id": sid}}}})))
        assert done["ok"] and done["result"]["state"]["done"], done
    # Off the library, the generator (scipy) is needed: the error names the missing module.
    off = {{"map": {{"source": "scenario", "scenario": "random_rubble", "rows": 11,
                     "cols": 11, "seed": 0}}}}
    assert webapi.needs_scipy(off)["scipy"]
    reply = json.loads(webapi.dispatch(json.dumps({{"method": "open",
                                                   "params": {{"config": off}}}})))
    assert not reply["ok"] and "scipy" in reply["error"]["message"], reply
    assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
    print("OK")
""")


def test_webapi_needs_no_pygame_matplotlib_scipy_or_yaml(library_doc: dict[str, Any]) -> None:
    script = BLOCKER_SCRIPT.format(blocked=BLOCKED_MODULES, library=library_doc)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=REPO_ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


# --- 8. performance ----------------------------------------------------------------------------
def _timed_steps(config: dict[str, Any]) -> list[float]:
    """Wall-clock seconds of each of the first ``PERF_STEPS`` steps of a fresh session."""
    session = WebSession(config)
    durations = []
    while not session.done and len(durations) < PERF_STEPS:
        t0 = time.perf_counter()
        session.step()
        durations.append(time.perf_counter() - t0)
    return durations


def test_20x20_andor_steps_quickly(record_property: Any) -> None:
    webmaps.set_map_library(None)
    config = {"map": {"source": "scenario", "scenario": "collapsed_building",
                      "rows": PERF_SIZE, "cols": PERF_SIZE, "seed": 0},
              "agent": "andor"}
    best: list[float] = []
    medians: list[float] = []
    for _ in range(PERF_REPLAYS):  # deterministic: step i does the same work in every replay
        durations = _timed_steps(config)
        assert durations
        best = durations if not best else [min(a, b) for a, b in zip(best, durations,
                                                                      strict=True)]
        medians.append(statistics.median(durations))
        if statistics.median(best) < STEP_MEDIAN_LIMIT_S:
            break
    median = statistics.median(best)
    record_property("step_ms_median", round(1000 * median, 2))
    assert median < STEP_MEDIAN_LIMIT_S, (
        f"median step {1000 * median:.1f} ms over {len(best)} steps (best of {len(medians)} "
        f"replays; replay medians {[round(1000 * m, 1) for m in medians]} ms)")
