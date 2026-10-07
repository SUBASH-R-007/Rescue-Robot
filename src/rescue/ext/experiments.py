"""Monte Carlo harness of the extensions: ``python -m rescue.ext.experiments`` (plan §3.14).

Three blocks, defined in ``experiments/ext.yaml`` (full) and ``experiments/ext_quick.yaml``
(``--quick``):

* ``search`` (brief §8.4) — survivor location known vs unknown, ``search_andor`` vs
  ``search_astar``;
* ``multi`` (§8.5) — k ∈ {1, 2, 3, 4} known survivors; greedy vs optimal tour, steps per rescue;
* ``noisy`` (§8.3) — debris false-negative rate ε ∈ {0, 0.05, 0.1, 0.2}.

Seeding follows the core harness (:func:`rescue.experiments.spec.task_seed`): a task's seed
depends only on ``(base_seed, scenario, rows, cols, replicate)``, so known vs unknown location,
every k (the survivors for k are a prefix of those for k + 1), every noise level and every agent
replay the same base map, the same outcome stream and the same noise draws (the sensor draws
one uniform per grid cell per step, so a given (step, cell) gets the same draw whatever path
led there). The seed is the map seed, the World seed and the NoisySensor seed.

Pipeline: :func:`run_ext_experiments` (``raw.csv`` + ``run_meta.json``; spawn pool with BLAS
thread caps, rows sorted by task then agent, row-loss guard) → :func:`write_summary`
(``summary.csv``) → :func:`rescue.ext.claims.write_extensions` (``EXTENSIONS.md``) →
:func:`rescue.ext.plots.make_ext_plots` (``plots/*.png``). Two runs of one config give
identical ``raw.csv`` files except for the timing columns (``planning_ms``, ``elapsed_ms``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import MAPGEN_MIN_SIZE, MAX_SLIP_P, AgentConfig, PlannerConfig, load_yaml
from rescue.experiments.run import (
    BLAS_THREAD_LIMITS,
    META_FILE,
    PROGRESS_FRACTION,
    TIMING_COLUMNS,
    TIMING_DECIMALS,
    check_complete,
    chunksize_for,
    default_workers,
    git_commit,
)
from rescue.experiments.spec import REPO_ROOT, task_seed
from rescue.experiments.stats import mean_ci, wilson_interval

# --- files and names -------------------------------------------------------------------------
#: Configs of a full and a ``--quick`` run.
EXT_CONFIG: Path = Path("experiments") / "ext.yaml"
EXT_QUICK_CONFIG: Path = Path("experiments") / "ext_quick.yaml"
#: Output directories (a quick run never overwrites the committed full results).
EXT_OUT_DIR: Path = Path("results") / "ext"
EXT_QUICK_OUT_DIR: Path = EXT_OUT_DIR / "quick"
#: Output file names.
RAW_FILE: str = "raw.csv"
SUMMARY_FILE: str = "summary.csv"
EXTENSIONS_FILE: str = "EXTENSIONS.md"
PLOTS_DIR: str = "plots"
#: Block names the claims and plots look for.
SEARCH_BLOCK: str = "search"
MULTI_BLOCK: str = "multi"
NOISY_BLOCK: str = "noisy"
#: Scenario label of the pooled summary rows.
ALL_SCENARIOS: str = "all"

#: Task fields copied into every raw row.
TASK_COLUMNS: tuple[str, ...] = (
    "experiment", "scenario", "rows", "cols", "slip_p", "k", "survivors_known",
    "false_negative", "replicate", "seed",
)
#: ``ExtMetrics`` fields in ``raw.csv`` (overlapping task fields not repeated).
METRIC_COLUMNS: tuple[str, ...] = (
    "agent", "success", "failure_reason", "steps", "rescued", "first_detection_step",
    "cells_explored", "replans", "slips", "skids", "bumps", "blocked_moves", "unplanned_states",
    "planning_ms", "nodes_expanded", "oracle_path", "greedy_tour", "optimal_tour",
)
#: Planning calls (initial plan included) and mission wall time.
EXTRA_COLUMNS: tuple[str, ...] = ("plans_made", "elapsed_ms")
#: Fixed column order of ``raw.csv``.
RAW_COLUMNS: tuple[str, ...] = TASK_COLUMNS + METRIC_COLUMNS + EXTRA_COLUMNS
#: Columns identifying one task (map + condition); paired comparisons of agents join on them.
EXT_TASK_KEYS: tuple[str, ...] = TASK_COLUMNS
#: Columns identifying one base map (and its outcome / noise streams) inside a block: the unit
#: of analysis of paired comparisons pooled over conditions.
EXT_MAP_KEYS: tuple[str, ...] = ("experiment", "scenario", "rows", "cols", "replicate", "seed")
#: Columns identifying one survivor layout (base map + k): the true-map tours depend on these
#: only, not on slip p, the location knowledge or the noise level.
EXT_LAYOUT_KEYS: tuple[str, ...] = (*EXT_MAP_KEYS, "k")

_STR_COLUMNS = frozenset({"experiment", "scenario", "agent", "failure_reason"})
_BOOL_COLUMNS = frozenset({"success", "survivors_known"})
_INT_COLUMNS = frozenset({
    "rows", "cols", "k", "replicate", "seed", "steps", "rescued", "cells_explored", "replans",
    "slips", "skids", "bumps", "blocked_moves", "unplanned_states", "nodes_expanded",
    "plans_made",
})
_TRUE_TEXT = frozenset({"true", "1", "yes"})
_FALSE_TEXT = frozenset({"false", "0", "no"})

_CONFIG_KEYS = frozenset({"name", "base_seed", "n_maps", "agents", "blocks", "workers"})
_REQUIRED_CONFIG_KEYS = frozenset({"name", "base_seed", "n_maps", "agents", "blocks"})
_BLOCK_KEYS = frozenset({
    "name", "scenarios", "sizes", "slip_p", "survivors_known", "k", "false_negative",
    "agents", "n_maps",
})
_REQUIRED_BLOCK_KEYS = frozenset({"name", "scenarios", "sizes", "slip_p"})


# --- specification ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtBlock:
    """One factorial block: scenarios × sizes × slip p × k × location known × ε × replicates.

    ``agents`` / ``n_maps`` of ``None`` inherit the config's values.
    """

    name: str
    scenarios: tuple[str, ...]
    sizes: tuple[int, ...]
    slip_p: tuple[float, ...]
    survivors_known: tuple[bool, ...] = (True,)
    k: tuple[int, ...] = (1,)
    false_negative: tuple[float, ...] = (0.0,)
    agents: tuple[str, ...] | None = None
    n_maps: int | None = None

    def __post_init__(self) -> None:
        _validate_block(self)


@dataclass(frozen=True)
class ExtConfig:
    """A whole extension experiment (``workers`` of ``None``: CPUs − 2)."""

    name: str
    base_seed: int
    n_maps: int
    agents: tuple[str, ...]
    blocks: tuple[ExtBlock, ...]
    workers: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must not be empty")
        _check_count("base_seed", self.base_seed, 0)
        _check_count("n_maps", self.n_maps, 1)
        _check_agents("agents", self.agents)
        if not self.blocks:
            raise ValueError("an experiment needs at least one block")
        _unique("block names", tuple(b.name for b in self.blocks))
        if self.workers is not None:
            _check_count("workers", self.workers, 1)

    def block_agents(self, block: ExtBlock) -> tuple[str, ...]:
        """Agents of ``block`` (its own list, else the top-level one)."""
        return block.agents if block.agents is not None else self.agents

    def block_n_maps(self, block: ExtBlock) -> int:
        """Maps per parameter combination of ``block``."""
        return block.n_maps if block.n_maps is not None else self.n_maps


@dataclass(frozen=True)
class ExtTask:
    """One base map under one condition; :func:`run_task` runs every agent on it."""

    experiment: str
    scenario: str
    rows: int
    cols: int
    slip_p: float
    k: int
    survivors_known: bool
    false_negative: float
    replicate: int
    seed: int
    agents: tuple[str, ...]


def _unique(label: str, values: tuple[Any, ...]) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} has duplicates: {list(values)}")


def _check_count(label: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")


def _check_agents(label: str, agents: tuple[str, ...]) -> None:
    from rescue.ext.agents import EXT_AGENT_NAMES

    _unique(label, agents)
    unknown = [a for a in agents if a not in EXT_AGENT_NAMES]
    if unknown:
        raise ValueError(f"{label}: unknown agents {unknown}; "
                         f"choose from {', '.join(EXT_AGENT_NAMES)}")


def _validate_block(block: ExtBlock) -> None:
    from rescue.mapgen import SCENARIOS

    where = f"block {block.name!r}"
    if not block.name:
        raise ValueError("block name must not be empty")
    _unique(f"{where} scenarios", block.scenarios)
    unknown = [s for s in block.scenarios if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"{where}: unknown scenarios {unknown}; "
                         f"choose from {', '.join(SCENARIOS)}")
    _unique(f"{where} sizes", block.sizes)
    for size in block.sizes:
        _check_count(f"{where} size", size, MAPGEN_MIN_SIZE)
    _unique(f"{where} slip_p", block.slip_p)
    if any(not 0.0 <= p <= MAX_SLIP_P for p in block.slip_p):
        raise ValueError(f"{where}: slip_p must be in [0, {MAX_SLIP_P}], got {block.slip_p}")
    _unique(f"{where} survivors_known", block.survivors_known)
    _unique(f"{where} k", block.k)
    for k in block.k:
        _check_count(f"{where} k", k, 1)
    _unique(f"{where} false_negative", block.false_negative)
    if any(not 0.0 <= e <= 1.0 for e in block.false_negative):
        raise ValueError(f"{where}: false_negative must be in [0, 1], got "
                         f"{block.false_negative}")
    if block.agents is not None:
        _check_agents(f"{where} agents", block.agents)
    if block.n_maps is not None:
        _check_count(f"{where} n_maps", block.n_maps, 1)


def _as_tuple(value: Any, kind: type, label: str) -> tuple[Any, ...]:
    """A YAML scalar or list as a tuple of ``kind`` (ints accepted for floats; bools only
    where bools are wanted)."""
    items = list(value) if isinstance(value, list | tuple) else [value]
    out: list[Any] = []
    for item in items:
        if kind is bool:
            if not isinstance(item, bool):
                raise ValueError(f"{label}: expected true/false, got {item!r}")
            out.append(item)
        elif isinstance(item, bool):
            raise ValueError(f"{label}: expected {kind.__name__}, got {item!r}")
        elif kind is float and isinstance(item, int | float):
            out.append(float(item))
        elif isinstance(item, kind):
            out.append(item)
        else:
            raise ValueError(f"{label}: expected {kind.__name__}, got {item!r}")
    return tuple(out)


def _mapping(label: str, data: Any, allowed: frozenset[str],
             required: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a mapping, got {type(data).__name__}")
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{label}: unknown keys {sorted(unknown)}")
    missing = required - set(data)
    if missing:
        raise ValueError(f"{label}: missing keys {sorted(missing)}")
    return data


#: Optional block factors: YAML key → element type.
_FACTOR_TYPES: dict[str, type] = {"survivors_known": bool, "k": int, "false_negative": float}


def block_from_dict(data: dict[str, Any]) -> ExtBlock:
    """An :class:`ExtBlock` from a parsed YAML mapping (unknown keys → ValueError)."""
    data = _mapping("block", data, _BLOCK_KEYS, _REQUIRED_BLOCK_KEYS)
    name = str(data["name"])
    label = f"block {name!r}"
    kwargs: dict[str, Any] = {
        "name": name,
        "scenarios": _as_tuple(data["scenarios"], str, f"{label} scenarios"),
        "sizes": _as_tuple(data["sizes"], int, f"{label} sizes"),
        "slip_p": _as_tuple(data["slip_p"], float, f"{label} slip_p"),
    }
    for key, kind in _FACTOR_TYPES.items():
        if key in data:
            kwargs[key] = _as_tuple(data[key], kind, f"{label} {key}")
    if data.get("agents") is not None:
        kwargs["agents"] = _as_tuple(data["agents"], str, f"{label} agents")
    if data.get("n_maps") is not None:
        kwargs["n_maps"] = data["n_maps"]
    return ExtBlock(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExtConfig:
    """An :class:`ExtConfig` from a parsed YAML mapping (unknown keys → ValueError)."""
    data = _mapping("experiment config", data, _CONFIG_KEYS, _REQUIRED_CONFIG_KEYS)
    if not isinstance(data["blocks"], list):
        raise ValueError("blocks must be a list of mappings")
    return ExtConfig(
        name=str(data["name"]),
        base_seed=data["base_seed"],
        n_maps=data["n_maps"],
        agents=_as_tuple(data["agents"], str, "agents"),
        blocks=tuple(block_from_dict(b) for b in data["blocks"]),
        workers=data.get("workers"),
    )


def load_ext_config(path: str | Path) -> ExtConfig:
    """Load and validate an extension experiment YAML (bad keys or values → ValueError)."""
    try:
        return config_from_dict(load_yaml(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def resolve_path(path: str | Path) -> Path:
    """``path`` if it exists (or is absolute), else relative to the repository root."""
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    fallback = REPO_ROOT / candidate
    return fallback if fallback.exists() else candidate


def expand_tasks(cfg: ExtConfig) -> list[ExtTask]:
    """Every task of ``cfg`` in a deterministic order: block, scenario, size, slip p, k,
    location known, ε (each as listed), replicate. ``raw.csv`` follows this order, then the
    block's agents in config order. Seeds: :func:`rescue.experiments.spec.task_seed`."""
    tasks: list[ExtTask] = []
    for block in cfg.blocks:
        agents = cfg.block_agents(block)
        n_maps = cfg.block_n_maps(block)
        for scenario in block.scenarios:
            for size in block.sizes:
                seeds = [task_seed(cfg.base_seed, scenario, size, size, i) for i in range(n_maps)]
                for p in block.slip_p:
                    for k in block.k:
                        for known in block.survivors_known:
                            for eps in block.false_negative:
                                tasks.extend(
                                    ExtTask(block.name, scenario, size, size, p, k, known, eps,
                                            replicate, seed, agents)
                                    for replicate, seed in enumerate(seeds)
                                )
    return tasks


def _map_of(task: ExtTask) -> tuple[Any, ...]:
    """The base map of a task (its block, scenario, size and replicate)."""
    return (task.experiment, task.scenario, task.rows, task.cols, task.replicate)


def subsample_maps(tasks: Sequence[ExtTask], max_tasks: int | None) -> list[ExtTask]:
    """An evenly spaced sample of whole maps with at most ``max_tasks`` tasks (order kept).

    In this harness the paired conditions (known / unknown location, every k, every ε) are
    separate tasks of one map, so a sample keeps or drops all of a map's tasks together and the
    paired claims stay paired. At least one map is kept, even if it alone exceeds
    ``max_tasks``; ``None`` (or a large enough value) keeps every task.
    """
    if max_tasks is None or max_tasks >= len(tasks):
        return list(tasks)
    if max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
    groups: dict[tuple[Any, ...], int] = {}
    for task in tasks:
        groups[_map_of(task)] = groups.get(_map_of(task), 0) + 1
    maps = list(groups)
    chosen: set[tuple[Any, ...]] = {maps[0]}
    for m in range(min(len(maps), max_tasks), 0, -1):
        picks = {maps[i] for i in np.unique(np.linspace(0, len(maps) - 1, m).round().astype(int))}
        if sum(groups[g] for g in picks) <= max_tasks:
            chosen = picks
            break
    return [t for t in tasks if _map_of(t) in chosen]


# --- running ---------------------------------------------------------------------------------
def run_task(task: ExtTask) -> list[dict[str, Any]]:
    """Build the task's map once and run every agent on it (same seed); one row per agent.

    Map: :func:`rescue.ext.scenarios.multi_survivor_grid` (the core map for k = 1). World:
    ``WorldConfig(slip_p)`` with the core defaults (sensor radius, no aftershocks). Each row =
    the task fields + the ``ExtMetrics`` columns (``failure_reason`` '' on success) +
    ``plans_made`` + ``elapsed_ms``.
    """
    from rescue.config import WorldConfig
    from rescue.ext.agents import make_ext_agent
    from rescue.ext.mission import run_ext_mission
    from rescue.ext.scenarios import multi_survivor_grid
    from rescue.runner import make_transition

    grid = multi_survivor_grid(task.scenario, task.rows, task.cols, task.seed, task.k)
    world_cfg = WorldConfig(slip_p=task.slip_p)
    base = {name: getattr(task, name) for name in TASK_COLUMNS}
    rows: list[dict[str, Any]] = []
    for name in task.agents:
        t0 = time.perf_counter()
        agent = make_ext_agent(name, make_transition(grid, world_cfg), PlannerConfig(),
                               AgentConfig())
        mission = run_ext_mission(
            grid, agent, world_cfg, task.seed,
            survivors_known=task.survivors_known, false_negative=task.false_negative,
            scenario=task.scenario, mode=task.experiment,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = mission.metrics().as_dict()
        row = dict(base)
        for column in METRIC_COLUMNS:
            row[column] = metrics[column]
        row["failure_reason"] = row["failure_reason"] or ""
        row["plans_made"] = int(agent.stats.plans_made)
        row["elapsed_ms"] = elapsed_ms
        rows.append(row)
    return rows


def _run_indexed(item: tuple[int, ExtTask]) -> tuple[int, list[dict[str, Any]]]:
    """Pool entry point: ``(index, task)`` → ``(index, rows)``."""
    index, task = item
    try:
        return index, run_task(task)
    except Exception as exc:  # re-raised in the parent with the failing task attached
        raise RuntimeError(f"task {index} failed: {task!r}: {exc!r}") from exc


def _init_worker() -> None:
    """Pool initialiser: import the heavy modules once per worker."""
    import rescue.ext.mission
    import rescue.ext.scenarios  # noqa: F401


def _submission_order(tasks: Sequence[ExtTask]) -> list[tuple[int, ExtTask]]:
    """Indexed tasks, expensive ones first (map area × k, unknown locations before known);
    stable, and without effect on the output (rows are sorted by index)."""
    indexed = list(enumerate(tasks))
    indexed.sort(key=lambda item: (-(item[1].rows * item[1].cols * item[1].k),
                                   item[1].survivors_known))
    return indexed


def _progress_printer(total: int, enabled: bool) -> Any:
    """A ``done → None`` callback printing progress to stderr about every 5 %."""
    step = max(1, math.ceil(total * PROGRESS_FRACTION))
    state = {"next": step, "t0": time.perf_counter()}

    def update(done: int) -> None:
        if not enabled or (done < state["next"] and done != total):
            return
        while state["next"] <= done:
            state["next"] += step
        elapsed = time.perf_counter() - state["t0"]
        eta = elapsed / done * (total - done) if done else math.nan
        print(f"[ext] {done}/{total} tasks ({100.0 * done / total:.0f}%), {elapsed:.1f} s "
              f"elapsed, ETA {eta:.0f} s", file=sys.stderr, flush=True)

    return update


def _execute(indexed: list[tuple[int, ExtTask]], workers: int,
             progress: Any) -> dict[int, list[dict[str, Any]]]:
    """Run the tasks (in-process for one worker, else a spawn pool) → rows by task index."""
    results: dict[int, list[dict[str, Any]]] = {}
    if workers <= 1:
        for item in indexed:
            index, rows = _run_indexed(item)
            results[index] = rows
            progress(len(results))
        return results
    for name, value in BLAS_THREAD_LIMITS.items():
        os.environ.setdefault(name, value)  # spawned children inherit the environment
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for index, rows in pool.imap_unordered(_run_indexed, indexed,
                                               chunksize=chunksize_for(len(indexed), workers)):
            results[index] = rows
            progress(len(results))
    return results


def _format_cell(column: str, value: Any) -> Any:
    if value is None:
        return ""
    if column in TIMING_COLUMNS and isinstance(value, float) and math.isfinite(value):
        return round(value, TIMING_DECIMALS)
    return value


def write_raw(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write rows as CSV with :data:`RAW_COLUMNS` (LF endings, atomic replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RAW_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _format_cell(c, row[c]) for c in RAW_COLUMNS})
    os.replace(tmp, target)
    return target


def _parse(column: str, text: str) -> Any:
    if column in _STR_COLUMNS:
        return text
    if column in _BOOL_COLUMNS:
        low = text.strip().lower()
        if low in _TRUE_TEXT:
            return True
        if low in _FALSE_TEXT:
            return False
        raise ValueError(f"bad boolean {text!r} in column {column}")
    if column in _INT_COLUMNS:
        return int(text)
    return float(text) if text.strip() else math.nan


def read_raw(path: str | Path) -> list[dict[str, Any]]:
    """``raw.csv`` back as typed rows (bools, ints, floats with NaN, strings)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [{c: _parse(c, t) for c, t in row.items()} for row in csv.DictReader(fh)]


def run_ext_experiments(
    cfg: ExtConfig,
    out_dir: str | Path,
    workers: int | None = None,
    progress: bool = True,
    *,
    max_tasks: int | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Run every task of ``cfg``; write ``out_dir/raw.csv`` and ``out_dir/run_meta.json``.

    ``workers``: explicit, else ``cfg.workers``, else CPUs − 2; 1 runs in-process.
    ``max_tasks`` runs an evenly spaced sample of whole maps (:func:`subsample_maps`: every
    condition of a sampled map, so paired claims stay paired). ``RuntimeError`` if any task or
    row is missing.
    Returns the path of ``raw.csv``.
    """
    out = Path(out_dir)
    all_tasks = expand_tasks(cfg)
    tasks = subsample_maps(all_tasks, max_tasks)
    if workers is None:
        workers = cfg.workers if cfg.workers is not None else default_workers()
    workers = max(1, min(int(workers), len(tasks)))
    started = _dt.datetime.now().astimezone()
    t0 = time.perf_counter()
    if progress:
        sample = f" (sample of {len(all_tasks)})" if len(tasks) < len(all_tasks) else ""
        print(f"[ext] {cfg.name}: {len(tasks)} tasks{sample}, "
              f"{sum(len(t.agents) for t in tasks)} missions, {workers} worker(s)",
              file=sys.stderr, flush=True)
    results = _execute(_submission_order(tasks), workers,
                       _progress_printer(len(tasks), progress))
    ordered = [row for index in sorted(results) for row in results[index]]
    check_complete(tasks, results, ordered)  # type: ignore[arg-type]
    raw_path = write_raw(ordered, out / RAW_FILE)
    wall_s = time.perf_counter() - t0
    meta = {
        "config_name": cfg.name,
        "config_path": str(config_path) if config_path is not None else None,
        "base_seed": cfg.base_seed,
        "n_maps": cfg.n_maps,
        "agents": list(cfg.agents),
        "blocks": [b.name for b in cfg.blocks],
        "n_tasks": len(tasks),
        "n_tasks_total": len(all_tasks),
        "max_tasks": max_tasks,
        "n_runs": len(ordered),
        "workers": workers,
        "wall_time_s": round(wall_s, 3),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "date": started.date().isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if progress:
        print(f"[ext] wrote {raw_path} ({len(ordered)} runs) in {wall_s:.1f} s",
              file=sys.stderr, flush=True)
    return raw_path


# --- derived values and summary --------------------------------------------------------------
def steps_per_rescue(row: dict[str, Any]) -> float:
    """``steps / k`` of a successful mission (NaN on failure)."""
    return row["steps"] / row["k"] if row["success"] else math.nan


def unique_layouts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per survivor layout (:data:`EXT_LAYOUT_KEYS`, first row kept, order kept): the
    true-map tours are properties of the layout, so slip p, agents and other conditions of the
    same layout must not count as extra maps."""
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        seen.setdefault(tuple(r[k] for k in EXT_LAYOUT_KEYS), r)
    return list(seen.values())


def tour_ratio(row: dict[str, Any]) -> float:
    """Greedy / optimal true-map tour length (NaN when either is undefined or 0)."""
    greedy, optimal = float(row["greedy_tour"]), float(row["optimal_tour"])
    if not (math.isfinite(greedy) and math.isfinite(optimal)) or optimal <= 0.0:
        return math.nan
    return greedy / optimal


#: Summary grouping (one row per condition and agent; ``scenario = 'all'`` pools scenarios).
GROUP_KEYS: tuple[str, ...] = (
    "experiment", "scenario", "rows", "slip_p", "k", "survivors_known", "false_negative", "agent",
)
#: Metrics summarised as mean / lo / hi; the first two over successful missions only.
SUCCESS_ONLY: tuple[str, ...] = ("steps", "steps_per_rescue")
SUMMARY_METRICS: tuple[str, ...] = (
    "steps", "steps_per_rescue", "rescued", "first_detection_step", "cells_explored", "replans",
    "slips", "skids", "bumps", "blocked_moves", "unplanned_states", "planning_ms",
    "nodes_expanded", "greedy_tour", "optimal_tour", "tour_ratio",
)
#: Failure reasons counted per summary row.
FAILURE_REASONS: tuple[str, ...] = ("aborted", "budget", "unreachable")
SUMMARY_COLUMNS: tuple[str, ...] = (
    *GROUP_KEYS, "n", "successes", "success_rate", "success_lo", "success_hi", "missions_bumped",
    *(f"n_{r}" for r in FAILURE_REASONS),
    *(f"{m}_{s}" for m in SUMMARY_METRICS for s in ("mean", "lo", "hi")),
)
_FLOAT_FORMAT: str = ".6g"


def _metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "steps_per_rescue":
        return steps_per_rescue(row)
    if metric == "tour_ratio":
        return tour_ratio(row)
    if metric in SUCCESS_ONLY and not row["success"]:
        return math.nan
    return float(row[metric])


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per condition × agent, per scenario and pooled (``scenario='all'``),
    in first-appearance order of ``rows`` (pooled rows after each block's scenarios)."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for pooled in (False, True):
        for row in rows:
            key = tuple(ALL_SCENARIOS if pooled and g == "scenario" else row[g]
                        for g in GROUP_KEYS)
            groups.setdefault(key, []).append(row)
    order = {name: i for i, name in enumerate(dict.fromkeys(r["experiment"] for r in rows))}
    keys = sorted(groups, key=lambda key: (order[key[0]], key[1] == ALL_SCENARIOS))
    out: list[dict[str, Any]] = []
    for key in keys:
        group = groups[key]
        n = len(group)
        ok = sum(bool(r["success"]) for r in group)
        lo, hi = wilson_interval(ok, n)
        summary: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        summary.update(n=n, successes=ok, success_rate=ok / n, success_lo=lo, success_hi=hi,
                       missions_bumped=sum(r["bumps"] > 0 for r in group))
        for reason in FAILURE_REASONS:
            summary[f"n_{reason}"] = sum(r["failure_reason"] == reason for r in group)
        for metric in SUMMARY_METRICS:
            mean, mlo, mhi = mean_ci([_metric_value(r, metric) for r in group])
            summary.update({f"{metric}_mean": mean, f"{metric}_lo": mlo, f"{metric}_hi": mhi})
        out.append(summary)
    return out


def write_summary(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write :func:`summarize` as CSV (floats ``.6g``; NaN as ``nan``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in summarize(rows):
            writer.writerow({c: format(v, _FLOAT_FORMAT) if isinstance(v, float) else v
                             for c, v in row.items()})
    return target


# --- CLI -------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the extension harness."""
    parser = argparse.ArgumentParser(
        prog="python -m rescue.ext.experiments",
        description="Extension experiments (unknown survivor location, several survivors, noisy "
        "sensing); writes raw.csv, summary.csv, EXTENSIONS.md and plots.",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--config", metavar="PATH", default=None,
                       help=f"experiment YAML (default {EXT_CONFIG.as_posix()})")
    which.add_argument("--quick", action="store_true",
                       help=f"use {EXT_QUICK_CONFIG.as_posix()} (< 60 s) and write to "
                            f"{EXT_QUICK_OUT_DIR.as_posix()}/ by default")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="worker processes (default: config value, else CPUs - 2; "
                             "1 = in-process)")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help=f"output directory (default {EXT_OUT_DIR.as_posix()}, or "
                             f"{EXT_QUICK_OUT_DIR.as_posix()} with --quick)")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    parser.add_argument("--max-tasks", type=int, default=None, metavar="N",
                        help="run an evenly spaced sample of whole maps, at most N tasks "
                             "(every condition of a sampled map is kept, so the paired claims "
                             "stay paired; at least one map)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code (2 on a bad config or argument)."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.workers is not None and args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2
    if args.max_tasks is not None and args.max_tasks < 1:
        print("error: --max-tasks must be >= 1", file=sys.stderr)
        return 2
    config_path = resolve_path(EXT_QUICK_CONFIG if args.quick else (args.config or EXT_CONFIG))
    try:
        cfg = load_ext_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else (EXT_QUICK_OUT_DIR if args.quick else EXT_OUT_DIR)

    from rescue.ext.claims import evaluate_claims, write_extensions

    t0 = time.perf_counter()
    raw_path = run_ext_experiments(cfg, out_dir, workers=args.workers, progress=not args.quiet,
                                   max_tasks=args.max_tasks, config_path=config_path.as_posix())
    rows = read_raw(raw_path)
    write_summary(rows, out_dir / SUMMARY_FILE)
    write_extensions(rows, out_dir / EXTENSIONS_FILE)
    written = [raw_path, out_dir / SUMMARY_FILE, out_dir / EXTENSIONS_FILE]
    if not args.no_plots:
        from rescue.ext.plots import make_ext_plots

        written.extend(make_ext_plots(rows, out_dir / PLOTS_DIR))
    for claim in evaluate_claims(rows):
        print(f"{claim.label} {claim.verdict} {claim.claim}")
    print(f"{len(rows)} runs, total wall time {time.perf_counter() - t0:.1f} s")
    print(f"Wrote {len(written)} files to {out_dir.as_posix()}/")
    return 0


__all__ = [
    "ALL_SCENARIOS",
    "EXTENSIONS_FILE",
    "EXT_CONFIG",
    "EXT_LAYOUT_KEYS",
    "EXT_MAP_KEYS",
    "EXT_OUT_DIR",
    "EXT_QUICK_CONFIG",
    "EXT_QUICK_OUT_DIR",
    "EXT_TASK_KEYS",
    "MULTI_BLOCK",
    "NOISY_BLOCK",
    "RAW_COLUMNS",
    "SEARCH_BLOCK",
    "SUMMARY_COLUMNS",
    "ExtBlock",
    "ExtConfig",
    "ExtTask",
    "block_from_dict",
    "config_from_dict",
    "expand_tasks",
    "load_ext_config",
    "main",
    "read_raw",
    "run_ext_experiments",
    "run_task",
    "steps_per_rescue",
    "subsample_maps",
    "summarize",
    "tour_ratio",
    "unique_layouts",
    "write_raw",
    "write_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
