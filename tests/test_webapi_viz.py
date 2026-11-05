"""The web app shows exactly what the Pygame simulator shows (docs/plan.md §3.16).

A :class:`~rescue.webapi.WebSession` and the headless simulator (``python -m rescue.sim``,
with or without ``--ext``) run the same mission event for event, and the web state's event log
and panel metrics are the simulator's lines and labels, word for word (``rescue.webtext``
spells them out because the browser has no pygame).
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import logging
from pathlib import Path
from typing import Any

import pytest

pygame = pytest.importorskip("pygame")

from rescue import webmaps  # noqa: E402
from rescue.viz.ext_sim_app import ExtSimApp, ExtSimOptions  # noqa: E402
from rescue.viz.sim_app import SimApp, SimOptions  # noqa: E402
from rescue.webapi import WebSession  # noqa: E402

pytestmark = pytest.mark.viz

#: Repository maps.
MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"
#: Panel rows that hold wall-clock timings (differ between two runs of one mission).
TIMING_LABELS = frozenset({"Plan ms"})

CORE_CASES: list[dict[str, Any]] = [
    {"map": "deck_demo", "agent": "andor", "seed": 76},
    {"map": "deck_demo", "agent": "astar_fixed", "seed": 0},
    {"map": "rubble_01", "agent": "astar_replan", "seed": 3, "replan_mode": "lazy"},
    {"map": "flooded_01", "agent": "andor", "seed": 5, "aftershock_q": 0.3, "aftershock_k": 2},
    {"map": "maze_01", "agent": "andor", "seed": 1, "sensor_radius": 0, "slip_p": 0.3},
]
EXT_CASES: list[dict[str, Any]] = [
    {"mode": "search", "agent": "search_andor", "seed": 7},
    {"mode": "multi", "agent": "search_astar", "seed": 7, "k": 4},
    {"mode": "noisy", "agent": "search_andor", "seed": 2, "noise": 0.2, "false_positive": 0.1},
    {"mode": "multi", "agent": "search_andor", "seed": 1, "aftershock_q": 0.2},
]


@pytest.fixture(autouse=True)
def _quiet(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)


@pytest.fixture(autouse=True)
def _no_library() -> Any:
    """Generate maps with mapgen (no library) in this module."""
    webmaps.set_map_library(None)
    yield
    webmaps.set_map_library(None)


def app_options(tmp_path: Path, cls: type, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "headless": True,
        "autoplay": False,
        "maps_dir": str(MAPS_DIR),
        "recordings_dir": str(tmp_path / "recordings"),
        "runs_dir": str(tmp_path / "runs"),
    }
    settings.update(overrides)
    return cls(**settings)


def run_app(app: SimApp) -> None:
    while not app.mission.done:
        assert app.step_once()


def web_lines(session: WebSession) -> list[tuple[str, str]]:
    return [(line["text"], line["kind"]) for line in session.state()["log"]]


def panel(rows: list[list[str]] | dict[str, str]) -> dict[str, str]:
    items = rows.items() if isinstance(rows, dict) else rows
    return {label: value for label, value in items if label not in TIMING_LABELS}


@pytest.mark.parametrize("case", CORE_CASES, ids=lambda c: f"{c['map']}-{c['agent']}")
def test_core_session_matches_the_simulator(case: dict[str, Any], tmp_path: Path) -> None:
    app = SimApp(app_options(
        tmp_path, SimOptions, map_path=str(MAPS_DIR / f"{case['map']}.txt"),
        agent=case["agent"], seed=case["seed"], replan_mode=case.get("replan_mode", "eager"),
        slip_p=case.get("slip_p", 0.2), sensor_radius=case.get("sensor_radius", 1),
        aftershock_q=case.get("aftershock_q", 0.0), aftershock_k=case.get("aftershock_k", 1)))
    try:
        session = WebSession({
            "map": {"source": "file", "name": case["map"]}, "agent": case["agent"],
            "seed": case["seed"], "replan_mode": case.get("replan_mode", "eager"),
            "slip_p": case.get("slip_p", 0.2), "sensor_radius": case.get("sensor_radius", 1),
            "aftershock_q": case.get("aftershock_q", 0.0),
            "aftershock_k": case.get("aftershock_k", 1)})
        assert panel(session.state()["panel"]) == panel(app._metric_values())
        run_app(app)
        session.run()
        assert session.mission.events == app.mission.events
        assert web_lines(session)[-len(app.log.lines):] == app.log.lines
        assert panel(session.state()["panel"]) == panel(app._metric_values())
    finally:
        app.close()


@pytest.mark.parametrize("case", EXT_CASES, ids=lambda c: f"{c['mode']}-{c['agent']}")
def test_ext_session_matches_the_extension_simulator(case: dict[str, Any],
                                                     tmp_path: Path) -> None:
    rows = cols = 10
    app = ExtSimApp(app_options(
        tmp_path, ExtSimOptions, mode=case["mode"], agent=case["agent"], seed=case["seed"],
        scenario="collapsed_building", rows=rows, cols=cols, k=case.get("k"),
        noise=case.get("noise"), false_positive=case.get("false_positive"),
        aftershock_q=case.get("aftershock_q", 0.0)))
    try:
        config: dict[str, Any] = {
            "mode": case["mode"], "agent": case["agent"], "seed": case["seed"],
            "map": {"source": "scenario", "scenario": "collapsed_building", "rows": rows,
                    "cols": cols, "seed": case["seed"]},
            "aftershock_q": case.get("aftershock_q", 0.0),
        }
        for key, web_key in (("k", "k"), ("noise", "false_negative"),
                             ("false_positive", "false_positive")):
            if case.get(key) is not None:
                config[web_key] = case[key]
        session = WebSession(config)
        assert session.grid == app.base_grid
        run_app(app)
        session.run()
        assert session.mission.events == app.mission.events
        assert web_lines(session)[-len(app.log.lines):] == app.log.lines
        assert panel(session.state()["panel"]) == panel(app._metric_values())
    finally:
        app.close()
