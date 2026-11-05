"""Record the web app's test fixtures: ``webapp/tests/fixtures/missions.json``.

The front end's Node tests (``node --test webapp/tests``) and the development-only mock
backend (``?mock=1`` while serving ``webapp/`` itself) replay missions recorded from the real
Python core through :class:`rescue.webapi.WebSession`. This script writes them; a pytest test
(``tests/test_web_fixtures.py``) re-records them and fails when the committed file is stale.

Format: ``{"catalog": webapi.catalog(), "missions": [recording, ...]}``; a recording is
``{"config", "base", "plans", "frames", "complete"}`` where ``base`` holds the state keys that
never change, ``plans`` the plan JSON by id, and each frame the per-step keys, with ``belief``,
``p_debris``, ``policy`` and ``envelope`` only when they changed, ``plan`` the plan id (or
null) and ``trace`` / ``log`` only the new entries (``webapp/js/mock.js`` expands them).
Wall-clock timings are zeroed so the file is reproducible.

Usage (from the repository root)::

    python scripts/make_web_fixtures.py            # rewrite the fixture
    python scripts/make_web_fixtures.py --check    # exit 1 if the committed fixture is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rescue.webapi import WebSession, catalog

#: Repository root (scripts/make_web_fixtures.py -> one level up) and the fixture.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "webapp" / "tests" / "fixtures" / "missions.json"
#: The recorded missions: the slide 9 demo with the three main agents, the same agents on a
#: generated map, and the three extension modes.
_GENERATED = {"source": "scenario", "scenario": "collapsed_building", "rows": 12, "cols": 12,
              "seed": 7}
MISSIONS: tuple[dict[str, Any], ...] = (
    {"mode": "core", "map": {"source": "file", "name": "deck_demo"}, "agent": "andor",
     "seed": 76},
    {"mode": "core", "map": {"source": "file", "name": "deck_demo"}, "agent": "astar_fixed",
     "seed": 76},
    {"mode": "core", "map": {"source": "file", "name": "deck_demo"}, "agent": "astar_replan",
     "seed": 76},
    {"mode": "core", "map": dict(_GENERATED), "agent": "andor"},
    {"mode": "core", "map": dict(_GENERATED), "agent": "astar_fixed"},
    {"mode": "core", "map": dict(_GENERATED), "agent": "astar_replan"},
    {"mode": "search", "map": dict(_GENERATED), "agent": "search_andor"},
    {"mode": "multi", "map": dict(_GENERATED), "agent": "search_andor", "k": 3},
    {"mode": "noisy", "map": dict(_GENERATED), "agent": "search_andor", "false_negative": 0.1},
)
#: State keys that never change during a mission (stored once, in ``base``).
BASE_KEYS: tuple[str, ...] = ("agent", "api", "budget", "cells", "cols", "config", "high_slip",
                              "mode", "rows", "scenario", "seed", "start", "survivors",
                              "survivors_known")
#: Per-frame keys stored only when they changed from the previous frame.
SPARSE_KEYS: tuple[str, ...] = ("belief", "p_debris", "policy", "envelope")
#: Per-frame keys stored in every frame.
FRAME_KEYS: tuple[str, ...] = ("frame", "t", "status", "done", "robot", "rescued", "target",
                               "target_mode", "plan_id", "path", "plan_message", "metrics",
                               "panel", "explanation", "explanation_tones", "event")
#: Timing fields zeroed for a reproducible file (metrics key, panel label).
TIMING_METRIC = "planning_ms"
TIMING_PANEL_LABEL = "Plan ms"
ZERO_MS = 0.0


def _untimed(state: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(state["metrics"])
    if TIMING_METRIC in metrics:
        metrics[TIMING_METRIC] = ZERO_MS
    panel = [[label, f"{ZERO_MS:.1f}" if label == TIMING_PANEL_LABEL else value]
             for label, value in state["panel"]]
    return {**state, "metrics": metrics, "panel": panel}


def record(config: dict[str, Any]) -> dict[str, Any]:
    """One mission's recording (see the module docstring)."""
    session = WebSession(config)
    states = [session.state()]
    while not session.done:
        states.append(session.step()["state"])
    states = [json.loads(json.dumps(_untimed(s))) for s in states]  # JSON types throughout
    base = {key: states[0][key] for key in BASE_KEYS}
    plans: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    trace_seen = log_seen = 0
    for state in states:
        frame = {key: state.get(key) for key in FRAME_KEYS}
        for key in SPARSE_KEYS:
            if not frames or state[key] != previous.get(key):
                frame[key] = state[key]
        plan = state["plan"]
        if plan is not None:
            plans[str(plan["id"])] = plan
        frame["plan"] = None if plan is None else plan["id"]
        frame["trace"] = state["trace"][trace_seen:]
        frame["log"] = state["log"][log_seen:]
        trace_seen, log_seen = len(state["trace"]), len(state["log"])
        frames.append(frame)
        previous = state
    return {"config": config, "base": base, "plans": plans, "frames": frames,
            "complete": bool(states[-1]["done"])}


def build_fixture() -> dict[str, Any]:
    """The whole fixture document."""
    return {"catalog": catalog(), "missions": [record(dict(c)) for c in MISSIONS]}


def dump(document: dict[str, Any]) -> str:
    """The file's text (compact, stable)."""
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write (or with ``--check`` verify) the fixture; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed fixture differs from a fresh recording")
    args = parser.parse_args(argv)
    text = dump(build_fixture())
    if args.check:
        current = FIXTURE.read_text(encoding="utf-8") if FIXTURE.is_file() else ""
        if current != text:
            print(f"{FIXTURE} is stale: run python scripts/make_web_fixtures.py", file=sys.stderr)
            return 1
        print(f"{FIXTURE} is up to date")
        return 0
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE} ({len(text) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
