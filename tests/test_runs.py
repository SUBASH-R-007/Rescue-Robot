"""Tests for rescue.runs: JSON run files for replay (brief §6.5, docs/plan.md §1)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from rescue import runs
from rescue.agents import make_agent
from rescue.config import AftershockConfig, AgentConfig, PlannerConfig, WorldConfig
from rescue.grid import TrueGrid
from rescue.runner import Mission, MissionMetrics, MissionStatus, make_transition, run_mission
from rescue.runs import RunRecord, load_run, save_run
from tests.conftest import MAPS_DIR
from tests.helpers_maps import random_solvable_grid

#: Default run-file name: ``<timestamp>_<agent>_<seed>.json`` (brief §6.5).
DECK_ROOT = "E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}"
NAME_RE = re.compile(r"^\d[\d_-]*_(?P<agent>[a-z_]+)_(?P<seed>\d+)(-\d+)?\.json$")


def finished(
    name: str,
    grid: TrueGrid,
    cfg: WorldConfig,
    seed: int,
    planner_config: PlannerConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> Mission:
    agent = make_agent(
        name,
        make_transition(grid, cfg),
        planner_config or PlannerConfig(),
        agent_config or AgentConfig(),
    )
    return run_mission(grid, agent, cfg, seed, scenario="unit")


def same_value(a: Any, b: Any) -> bool:
    """Exact equality where NaN matches NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def assert_same_metrics(loaded: MissionMetrics, original: MissionMetrics) -> None:
    """The loaded metrics are a ``MissionMetrics`` equal to the original (NaN included)."""
    assert isinstance(loaded, MissionMetrics)
    for key, value in original.as_dict().items():
        assert same_value(getattr(loaded, key), value), key


def test_round_trip_success(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    m = finished("andor", grid, WorldConfig(), 3)
    assert m.status is MissionStatus.RESCUED
    path = save_run(m, runs_dir=tmp_path)
    assert path.exists()
    rec = load_run(path)
    assert isinstance(rec, RunRecord)
    assert rec.events == list(m.events)
    assert rec.grid == m.grid == grid
    assert rec.agent == "andor"
    assert rec.seed == 3
    assert rec.scenario == "unit"
    assert rec.world_config == WorldConfig()
    assert rec.metrics == m.metrics()  # no NaN on success: plain equality
    assert len(rec.snapshots) == len(m.agent.snapshots) >= 1
    for snap, orig in zip(rec.snapshots, m.agent.snapshots, strict=True):
        assert {"policy", "pretty", "tree"} <= set(snap)
        assert snap["plan_id"] == orig.plan_id
        assert snap["pretty"] == orig.plan.pretty()
        assert len(snap["policy"]) == len(orig.plan.to_policy())
        assert snap["path"] is None
        assert isinstance(snap["tree"], dict)
        assert snap["tree"]["kind"] == "act"
    assert rec.snapshots[0]["pretty"] == DECK_ROOT


def test_round_trip_failure_restores_nan(tmp_path: Path) -> None:
    """A failed path-agent mission: NaN wasted moves come back as NaN, A* snapshots keep the
    path (and no plan)."""
    grid = random_solvable_grid(7, 7, 0.25, 11, terrain_fraction=0.2)
    cfg = WorldConfig(slip_p=0.3)
    for seed in range(10):
        m = finished("astar_fixed", grid, cfg, seed)
        if not m.metrics().success:
            break
    assert not m.metrics().success
    assert math.isnan(m.metrics().wasted_moves)
    rec = load_run(save_run(m, path=tmp_path / "sub" / "fail.json"))
    assert rec.path == tmp_path / "sub" / "fail.json"
    assert rec.events == list(m.events)
    assert rec.grid == grid
    assert rec.world_config == cfg
    assert math.isnan(rec.metrics.wasted_moves)  # not None: replays use metrics like live ones
    assert_same_metrics(rec.metrics, m.metrics())
    assert len(rec.snapshots) == len(m.agent.snapshots) == 1
    snap, orig = rec.snapshots[0], m.agent.snapshots[0]
    assert orig.path is not None
    assert snap["path"] == [list(p) for p in orig.path]
    assert snap["pretty"] is None
    assert snap["tree"] is None
    assert snap["policy"] == {}
    assert snap["envelope"] == [list(p) for p in sorted(orig.envelope)]


def test_round_trip_with_aftershocks(tmp_path: Path) -> None:
    """Aftershock cells survive the round trip, and the run file keeps the INITIAL map."""
    grid = random_solvable_grid(7, 7, 0.25, 11)
    cfg = WorldConfig(slip_p=0.3, aftershock=AftershockConfig(q=0.2, k=1))
    for seed in range(20):
        m = finished("andor", grid, cfg, seed)
        if m.metrics().aftershock_events > 0:
            break
    assert m.metrics().aftershock_events > 0
    assert any(e.aftershock_cells for e in m.events)
    rec = load_run(save_run(m, path=tmp_path / "shock.json"))
    assert rec.events == list(m.events)
    assert any(e.aftershock_cells for e in rec.events)
    assert rec.world_config == cfg
    assert rec.grid == m.grid == grid
    assert rec.grid != m.world.grid  # the world's map has the aftershock debris
    for cell in (c for e in rec.events for c in e.aftershock_cells):
        assert not rec.grid.is_debris(cell)
        assert m.world.grid.is_debris(cell)
    assert_same_metrics(rec.metrics, m.metrics())


def test_round_trip_non_default_configs(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig(slip_p=0.1, sensor_radius=2, terrain_high_p=0.3, budget_factor=3)
    planner_config = PlannerConfig(validate=True, tie_break="fixed", depth_limit=40)
    agent_config = AgentConfig(replan_mode="lazy")
    m = finished("andor", grid, cfg, 4, planner_config, agent_config)
    rec = load_run(save_run(m, path=tmp_path / "cfg.json"))
    assert rec.world_config == cfg
    assert rec.planner_config == planner_config
    assert rec.agent_config == agent_config


def test_metrics_from_dict_rejects_bad_keys() -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    data = finished("andor", grid, WorldConfig(), 3).metrics().as_dict()
    assert MissionMetrics.from_dict(data).as_dict() == data
    with pytest.raises(ValueError):
        MissionMetrics.from_dict({**data, "extra": 1})
    missing = dict(data)
    del missing["steps"]
    with pytest.raises(ValueError):
        MissionMetrics.from_dict(missing)


def test_load_run_rejects_other_version(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    path = save_run(finished("andor", grid, WorldConfig(), 3), path=tmp_path / "v.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["version"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_run(path)


def test_save_unstarted_mission_raises(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig()
    mission = Mission(grid, make_agent("andor", make_transition(grid, cfg)), cfg)
    with pytest.raises(RuntimeError):
        save_run(mission, runs_dir=tmp_path)
    assert not any(tmp_path.iterdir())


class _FrozenClock(datetime):
    """``datetime`` whose ``now()`` is fixed, so default file names collide deterministically."""

    @classmethod
    def now(cls, tz: Any = None) -> _FrozenClock:  # type: ignore[override]
        return cls(2026, 10, 6, 12, 0, 0)


def test_default_file_naming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runs, "datetime", _FrozenClock)
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    m = finished("astar_replan", grid, WorldConfig(), 17)
    saved = [save_run(m, runs_dir=tmp_path) for _ in range(3)]
    assert [p.name for p in saved] == [
        "20261006-120000_astar_replan_17.json",
        "20261006-120000_astar_replan_17-1.json",  # never overwrite an existing run file
        "20261006-120000_astar_replan_17-2.json",
    ]
    for p in saved:
        assert p.parent == tmp_path
        match = NAME_RE.match(p.name)
        assert match, p.name
        assert match["agent"] == "astar_replan"
        assert match["seed"] == "17"
    assert load_run(saved[1]).events == load_run(saved[0]).events


def test_metrics_dataclass_fields_are_all_saved(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    m = finished("andor", grid, WorldConfig(), 3)
    document = json.loads(save_run(m, path=tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(document["metrics"]) == [f.name for f in dataclasses.fields(MissionMetrics)]
