"""What the web front end relies on beyond the state contract (docs/plan.md §3.16).

* Explanation sentences carry a tone (the UI never guesses it from the English text), and the
  sentences are true to the events: "Next: move X" is the move the agent really makes next,
  also for Fixed A*, which gives up instead of moving when its next path cell is debris.
* Lean states (what the worker sends) put back together equal the full states, and their size
  does not grow with the mission's length.
* Event-log lines carry the timeline tag; the header line has a friendly display text.
* The catalog gives the front end the constants it must not copy (outcome split, size limits).
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import pytest

from rescue.config import SKID_FRACTION, SLIP_STAY_FRACTION
from rescue.webapi import WebSession, catalog, dispatch
from rescue.webmaps import WEB_MAX_MAP_SIZE
from rescue.webtext import TONES

#: A long core mission (168 steps) and a long search mission (380 steps).
LONG_CORE = {"map": {"source": "scenario", "scenario": "maze_corridors", "rows": 20, "cols": 20,
                     "seed": 0}, "agent": "andor", "slip_p": 0.4}
LONG_SEARCH = {"mode": "search", "map": {"source": "scenario", "scenario": "collapsed_building",
                                         "rows": 20, "cols": 20, "seed": 3},
               "agent": "search_andor", "slip_p": 0.4}
#: A lean step near the end may be at most this many times the size of the first one.
LEAN_GROWTH_LIMIT = 2.0
#: The fidelity sweep: Fixed A* on generated maps (map seeds), where it often gives up.
FIXED_ASTAR_SEEDS = range(12)
COVERED_TEXT = "so no replanning was needed"


def _run(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every state of a mission, t = 0 first."""
    session = WebSession(config)
    states = [session.state()]
    while not session.done:
        states.append(session.step()["state"])
    return states


# --- tones -------------------------------------------------------------------------------------
@pytest.mark.parametrize("config", [
    {"agent": "andor", "seed": 76},
    {"agent": "astar_replan", "seed": 3, "slip_p": 0.4},
    LONG_SEARCH,
])
def test_every_sentence_has_a_known_tone(config: dict[str, Any]) -> None:
    for state in _run(config):
        assert len(state["explanation_tones"]) == len(state["explanation"])
        assert set(state["explanation_tones"]) <= set(TONES)


def test_covered_slips_get_the_outcome_tone_not_replan() -> None:
    seen = 0
    for state in _run({"map": {"source": "file", "name": "rubble_01"}, "agent": "andor",
                       "seed": 4, "slip_p": 0.4}):
        for text, tone in zip(state["explanation"], state["explanation_tones"], strict=True):
            if COVERED_TEXT in text:
                seen += 1
                assert tone in {"slip", "skid", "blocked"}, text
            if "(replan #" in text:
                assert tone == "replan", text
    assert seen


# --- explanation fidelity ----------------------------------------------------------------------
def _check_next_moves(states: list[dict[str, Any]]) -> int:
    """'Next: move X' equals the next event's action; a 'will give up on its next decision'
    is followed by an abort. Returns the number of checked 'Next' sentences."""
    checked = 0
    for state, following in pairwise(states):
        text = " ".join(state["explanation"])
        event = following["event"]
        if "Next: move " in text:
            action = text.split("Next: move ")[1].split(" ")[0]
            assert event["action"] == action, (state["t"], text, event)
            checked += 1
        if "give up on its next decision" in text:
            assert event["action"] is None and following["status"] == "aborted", text
    return checked


def test_fixed_astar_regression_collapsed_building_seed_7() -> None:
    states = _run({"map": {"source": "scenario", "scenario": "collapsed_building", "rows": 15,
                           "cols": 15, "seed": 7}, "agent": "astar_fixed"})
    assert _check_next_moves(states)


@pytest.mark.parametrize("scenario", ["collapsed_building", "random_rubble", "maze_corridors"])
def test_next_move_sentences_are_true_for_every_agent(scenario: str) -> None:
    aborted_with_warning = 0
    for seed in FIXED_ASTAR_SEEDS:
        for agent in ("astar_fixed", "astar_replan", "andor"):
            states = _run({"map": {"source": "scenario", "scenario": scenario, "rows": 10,
                                   "cols": 10, "seed": seed}, "agent": agent, "slip_p": 0.3})
            _check_next_moves(states)
            if states[-1]["status"] == "aborted" and "next path cell" in str(
                    states[-1]["event"]["note"]):
                warned = " ".join(states[-2]["explanation"])
                assert "will give up" in warned, warned
                aborted_with_warning += 1
    assert aborted_with_warning > 0  # Fixed A* meets known debris ahead in every scenario


# --- lean states -------------------------------------------------------------------------------
class Client:
    """The front end's MissionSession bookkeeping (webapp/js/session.js), in Python."""

    def __init__(self, first: dict[str, Any]) -> None:
        self.trace: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.grids: dict[int, tuple[Any, Any]] = {}
        self.plans: dict[int, Any] = {}
        self.config = first["config"]

    def full(self, lean: dict[str, Any]) -> dict[str, Any]:
        self.trace.extend(lean["trace_new"])
        self.log.extend(lean["log_new"])
        if "cells" in lean:
            self.grids[lean["grid_version"]] = (lean["cells"], lean["high_slip"])
        if "plan" in lean:
            self.plans[lean["plan_id"]] = lean["plan"]
        state = {k: v for k, v in lean.items()
                 if k not in {"trace_new", "log_new", "lean", "has_plan"}}
        state["cells"], state["high_slip"] = self.grids[lean["grid_version"]]
        state["trace"] = self.trace[:lean["trace_end"]]
        state["log"] = self.log[:lean["log_end"]]
        state["plan"] = self.plans.get(lean["plan_id"]) if lean["has_plan"] else None
        state["config"] = self.config
        return state


def _call(method: str, **params: Any) -> Any:
    reply = json.loads(dispatch(json.dumps({"method": method, "params": params})))
    assert reply["ok"], reply
    return reply["result"]


@pytest.mark.parametrize("config", [
    LONG_SEARCH,
    {"map": {"source": "file", "name": "flooded_01"}, "agent": "andor", "seed": 5,
     "aftershock_q": 0.3, "aftershock_k": 2},
    {"mode": "noisy", "agent": "search_astar", "seed": 2, "false_negative": 0.2,
     "false_positive": 0.1},
])
def test_lean_states_rebuild_the_full_states(config: dict[str, Any]) -> None:
    opened = _call("open", config=config, lean=True)
    sid = opened["id"]
    client = Client(opened["state"])
    live = [client.full(opened["state"])]
    while live[-1]["status"] == "running":
        live.append(client.full(_call("step", id=sid, lean=True)["state"]))
    for state in live:
        full = _call("state_at", id=sid, t=state["t"])
        rebuilt = client.full(_call("state_at", id=sid, t=state["t"], lean=True))
        assert rebuilt == full, state["t"]
    # Finish in one call after a reset, then scrub back: still equal.
    second = _call("open", config=config, lean=True)
    other = Client(second["state"])
    other.full(second["state"])
    final = other.full(_call("run", id=second["id"], lean=True)["state"])
    assert final == _call("state_at", id=second["id"], t=final["t"])
    for t in range(0, final["t"] + 1, max(1, final["t"] // 7)):
        assert other.full(_call("state_at", id=second["id"], t=t, lean=True)) == _call(
            "state_at", id=second["id"], t=t)


@pytest.mark.parametrize("config", [LONG_CORE, LONG_SEARCH])
def test_lean_payload_does_not_grow_with_the_mission(config: dict[str, Any]) -> None:
    session = WebSession(config)
    session.lean_state()
    sizes = []
    while not session.done:
        sizes.append(len(json.dumps(session.step(lean=True)["state"])))
    assert len(sizes) > 100
    assert sizes[-1] < LEAN_GROWTH_LIMIT * sizes[0], (sizes[0], sizes[-1])
    assert sizes[-1] < len(json.dumps(session.state())) / 4


# --- log tags and header -----------------------------------------------------------------------
def test_log_tags_mark_found_survivors_rescues_and_replans() -> None:
    log = _run(LONG_SEARCH)[-1]["log"]
    tags = {entry.get("tag") for entry in log}
    assert {"found", "rescue"} <= tags
    for entry in log:
        if entry["text"].startswith("Survivor found at"):
            assert entry["tag"] == "found"
        if entry["kind"] == "replan":
            assert entry["tag"] == "replan"


def test_header_line_keeps_the_simulator_text_and_adds_a_display_text() -> None:
    first = WebSession({}).state()["log"][0]
    assert first["text"] == "andor on deck_demo 4x4, seed 76, p = 0.20"
    assert first["display"] == ("AND-OR contingency on Quick demo (4×4), luck seed 76, "
                                "wheel slip p = 0.20")


# --- catalog -----------------------------------------------------------------------------------
def test_catalog_gives_the_front_end_its_constants() -> None:
    params = catalog()["params"]
    assert params["outcome_fractions"] == {"stay": SLIP_STAY_FRACTION, "skid": SKID_FRACTION}
    assert params["map_size"]["max"] == WEB_MAX_MAP_SIZE
    assert params["map_size"]["library_seeds"] == len(catalog()["library"]["seeds"])
    for entry in catalog()["maps"]:
        assert "python -m" not in entry["description"], entry["name"]
        assert "source" in entry
    for agent in catalog()["agents"]:
        assert "scipy" not in agent["description"], agent["name"]
