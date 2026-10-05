"""Run files: save a mission to JSON and load it back for replay (brief §6.5).

A run file holds everything a replay needs without re-simulating: the initial true map, the
configs, the seed, every :class:`~rescue.runner.StepEvent` and every plan snapshot (policy,
path, envelope and a depth-limited plan tree), plus the final metrics. Positions are stored
0-indexed as ``[r, c]``; enums are stored as strings, cells as ints, NaN as ``null`` (turned
back into NaN when the metrics are loaded, so events and metrics round-trip exactly).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rescue.agents.base import PlanSnapshot
from rescue.config import (
    AgentConfig,
    PlannerConfig,
    WorldConfig,
    agent_config_from_dict,
    planner_config_from_dict,
    world_config_from_dict,
)
from rescue.grid import Cell, Pos, TrueGrid
from rescue.model import Action, OutcomeKind
from rescue.runner import Mission, MissionMetrics, MissionStatus, StepEvent

log = logging.getLogger(__name__)

#: Version of the run-file format written by :func:`save_run`.
RUN_FORMAT_VERSION = 1
#: Default directory for run files.
DEFAULT_RUNS_DIR = "runs"
#: Timestamp format of default run-file names.
RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
#: Act depth of the plan trees stored with each snapshot.
SNAPSHOT_TREE_DEPTH = 8


@dataclass
class RunRecord:
    """A loaded run file.

    ``grid`` is the *initial* true map (before any aftershock); ``metrics`` are the final
    metrics exactly as :meth:`Mission.metrics` returned them (NaN restored); ``snapshots`` stay
    JSON dicts (``plan_id``, ``t``, ``pos``, ``pretty``, ``policy``, ``path``, ``envelope``,
    ``tree``).
    """

    grid: TrueGrid
    agent: str
    seed: int
    scenario: str
    world_config: WorldConfig
    planner_config: PlannerConfig
    agent_config: AgentConfig
    events: list[StepEvent]
    metrics: MissionMetrics
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    version: int = RUN_FORMAT_VERSION
    path: Path | None = None


# --- encoding ------------------------------------------------------------------------------
def _pos(p: Pos) -> list[int]:
    return [int(p[0]), int(p[1])]


def _jsonable(value: Any) -> Any:
    """Recursively convert tuples to lists and NaN/inf floats to None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def event_to_dict(event: StepEvent) -> dict[str, Any]:
    """JSON-ready dict of a :class:`StepEvent`."""
    return {
        "t": event.t,
        "pos": _pos(event.pos),
        "action": None if event.action is None else event.action.name,
        "outcome": None if event.outcome is None else event.outcome.value,
        "new_pos": _pos(event.new_pos),
        "blocked": event.blocked,
        "bump": event.bump,
        "new_cells": [[_pos(p), int(cell)] for p, cell in event.new_cells],
        "contradictions": [_pos(p) for p in event.contradictions],
        "triggering_contradictions": [_pos(p) for p in event.triggering_contradictions],
        "replanned": event.replanned,
        "unplanned": event.unplanned,
        "plan_id": event.plan_id,
        "aftershock_cells": [_pos(p) for p in event.aftershock_cells],
        "status": event.status.name,
        "note": event.note,
    }


def _as_pos(raw: Any) -> Pos:
    return (int(raw[0]), int(raw[1]))


def event_from_dict(data: dict[str, Any]) -> StepEvent:
    """Inverse of :func:`event_to_dict`."""
    action = data.get("action")
    outcome = data.get("outcome")
    return StepEvent(
        t=int(data["t"]),
        pos=_as_pos(data["pos"]),
        action=None if action is None else Action[action],
        outcome=None if outcome is None else OutcomeKind(outcome),
        new_pos=_as_pos(data["new_pos"]),
        blocked=bool(data["blocked"]),
        bump=bool(data["bump"]),
        new_cells=tuple((_as_pos(p), Cell(int(c))) for p, c in data["new_cells"]),
        contradictions=tuple(_as_pos(p) for p in data["contradictions"]),
        triggering_contradictions=tuple(
            _as_pos(p) for p in data["triggering_contradictions"]
        ),
        replanned=bool(data["replanned"]),
        unplanned=bool(data["unplanned"]),
        plan_id=int(data["plan_id"]),
        aftershock_cells=tuple(_as_pos(p) for p in data["aftershock_cells"]),
        status=MissionStatus[data["status"]],
        note=str(data.get("note", "")),
    )


def _snapshot_to_dict(snapshot: PlanSnapshot) -> dict[str, Any]:
    """JSON-ready dict of an agent ``PlanSnapshot``."""
    plan = snapshot.plan
    path = snapshot.path
    policy: dict[str, str] = {}
    if plan is not None:
        policy = {f"{r},{c}": a.name for (r, c), a in sorted(plan.to_policy().items())}
    return {
        "plan_id": int(snapshot.plan_id),
        "t": int(snapshot.t),
        "pos": _pos(snapshot.pos),
        "pretty": None if plan is None else plan.pretty(),
        "policy": policy,
        "path": None if path is None else [_pos(p) for p in path],
        "envelope": [_pos(p) for p in sorted(snapshot.envelope)],
        "tree": None if plan is None else _jsonable(plan.to_tree(SNAPSHOT_TREE_DEPTH)),
    }


def _agent_configs(mission: Mission) -> tuple[PlannerConfig, AgentConfig]:
    """The agent's planner/agent configs (defaults if the agent does not expose them)."""
    agent = mission.agent
    planner_config = getattr(agent, "planner_config", None)
    agent_config = getattr(agent, "agent_config", None)
    if not isinstance(planner_config, PlannerConfig):
        planner_config = PlannerConfig()
    if not isinstance(agent_config, AgentConfig):
        agent_config = AgentConfig()
    return planner_config, agent_config


def run_to_dict(mission: Mission) -> dict[str, Any]:
    """The JSON document :func:`save_run` writes for ``mission``."""
    planner_config, agent_config = _agent_configs(mission)
    return {
        "version": RUN_FORMAT_VERSION,
        "agent": mission.agent.name,
        "seed": mission.seed,
        "scenario": mission.scenario,
        "map": mission.grid.to_ascii(),
        "map_name": mission.grid.name,
        "world_config": _jsonable(asdict(mission.world_config)),
        "planner_config": _jsonable(asdict(planner_config)),
        "agent_config": _jsonable(asdict(agent_config)),
        "events": [event_to_dict(e) for e in mission.events],
        "snapshots": [_snapshot_to_dict(s) for s in mission.agent.snapshots],
        "metrics": _jsonable(mission.metrics().as_dict()),
    }


def default_run_path(mission: Mission, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """``runs/<YYYYmmdd-HHMMSS>_<agent>_<seed>.json`` (suffixed ``-1``, ``-2`` … if taken)."""
    stamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    stem = f"{stamp}_{mission.agent.name}_{mission.seed}"
    directory = Path(runs_dir)
    candidate = directory / f"{stem}.json"
    n = 0
    while candidate.exists():
        n += 1
        candidate = directory / f"{stem}-{n}.json"
    return candidate


def save_run(
    mission: Mission, path: str | Path | None = None, runs_dir: str | Path = DEFAULT_RUNS_DIR
) -> Path:
    """Write ``mission`` (started; usually finished) to a JSON run file and return its path."""
    if not mission.started:
        raise RuntimeError("cannot save a mission that has not started")
    target = Path(path) if path is not None else default_run_path(mission, runs_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = run_to_dict(mission)
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=1)
    target.write_text(text + "\n", encoding="utf-8")
    log.info("saved run to %s", target)
    return target


# --- decoding ------------------------------------------------------------------------------
def load_run(path: str | Path) -> RunRecord:
    """Read a run file written by :func:`save_run` (``ValueError`` for another version)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    version = int(data.get("version", 0))
    if version != RUN_FORMAT_VERSION:
        raise ValueError(f"{p}: unsupported run-file version {version}")
    grid = TrueGrid.from_ascii(data["map"], name=str(data.get("map_name", "")))
    return RunRecord(
        grid=grid,
        agent=str(data["agent"]),
        seed=int(data["seed"]),
        scenario=str(data.get("scenario", "")),
        world_config=world_config_from_dict(data.get("world_config")),
        planner_config=planner_config_from_dict(data.get("planner_config")),
        agent_config=agent_config_from_dict(data.get("agent_config")),
        events=[event_from_dict(e) for e in data["events"]],
        metrics=MissionMetrics.from_dict(data["metrics"]),
        snapshots=list(data.get("snapshots", [])),
        version=version,
        path=p,
    )
