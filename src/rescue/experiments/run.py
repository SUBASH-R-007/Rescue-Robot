"""Monte Carlo runner: every task of an experiment, in parallel, into ``raw.csv``.

:func:`run_task` generates one map and runs every agent on it with the same world seed (common
random numbers); :func:`run_experiments` farms the tasks out to a ``multiprocessing`` pool
(spawn context), writes ``raw.csv`` in a fixed row and column order and ``run_meta.json``
describing the run. Two runs of the same config give byte-identical ``raw.csv`` files except
for the timing columns (``TIMING_COLUMNS``).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rescue.config import AftershockConfig, AgentConfig, WorldConfig
from rescue.experiments.spec import (
    REPO_ROOT,
    ExperimentConfig,
    Task,
    expand_tasks,
    subsample_tasks,
)

#: Task fields copied into every raw row (``agents`` is replaced by the per-row ``agent``).
TASK_COLUMNS: tuple[str, ...] = (
    "experiment",
    "scenario",
    "rows",
    "cols",
    "slip_p",
    "sensor_radius",
    "replan_mode",
    "planner",
    "aftershock_q",
    "aftershock_k",
    "replicate",
    "seed",
)
#: ``MissionMetrics`` fields in ``raw.csv`` (the overlapping task fields are not repeated).
METRIC_COLUMNS: tuple[str, ...] = (
    "agent",
    "success",
    "failure_reason",
    "steps",
    "shortest_path",
    "wasted_moves",
    "replans",
    "slips",
    "skids",
    "slips_handled",
    "bumps",
    "blocked_moves",
    "unplanned_states",
    "cells_explored",
    "planning_ms",
    "nodes_expanded",
    "plan_size",
    "planner_fallbacks",
    "aftershock_events",
)
#: Extra per-run columns: planning calls (initial plan included) and mission wall time.
EXTRA_COLUMNS: tuple[str, ...] = ("plans_made", "elapsed_ms")
#: Fixed column order of ``raw.csv``.
RAW_COLUMNS: tuple[str, ...] = TASK_COLUMNS + METRIC_COLUMNS + EXTRA_COLUMNS
#: Columns that hold wall-clock measurements (the only ones that differ between two runs).
TIMING_COLUMNS: tuple[str, ...] = ("planning_ms", "elapsed_ms")
#: Decimals kept for the timing columns (sub-microsecond digits are noise).
TIMING_DECIMALS: int = 3
#: File names written by :func:`run_experiments` into the output directory.
RAW_FILE: str = "raw.csv"
META_FILE: str = "run_meta.json"
#: Logical CPUs left free when choosing the default worker count.
RESERVED_CPUS: int = 2
#: Target number of chunks per worker for ``imap_unordered`` (small chunks balance the very
#: unequal task costs), and the largest chunk.
CHUNKS_PER_WORKER: int = 16
MAX_CHUNKSIZE: int = 8
#: Progress is printed every this fraction of the tasks.
PROGRESS_FRACTION: float = 0.05
#: BLAS / OpenMP thread caps for the workers (see tests/conftest.py: each spawned worker would
#: otherwise commit ~0.6 GB for OpenBLAS thread pools and exhaust the Windows commit limit).
BLAS_THREAD_LIMITS: dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
#: Seconds allowed for ``git rev-parse`` when recording the commit in ``run_meta.json``.
GIT_TIMEOUT_S: float = 5.0


# --- one task --------------------------------------------------------------------------------
def world_config_for(task: Task) -> WorldConfig:
    """World parameters of ``task`` (terrain high-slip p and budget at their defaults)."""
    return WorldConfig(
        slip_p=task.slip_p,
        sensor_radius=task.sensor_radius,
        aftershock=AftershockConfig(q=task.aftershock_q, k=task.aftershock_k),
    )


def _task_fields(task: Task) -> dict[str, Any]:
    return {name: getattr(task, name) for name in TASK_COLUMNS}


def run_task(task: Task) -> list[dict[str, Any]]:
    """Generate the task's map once and run every agent on it; one row per agent.

    Every agent gets ``PlannerConfig(**task's planner overrides)`` (the block's ``planner``).
    Each row = the task fields + every ``MissionMetrics`` field (``failure_reason`` '' on
    success, ``wasted_moves`` NaN on failure) + ``plans_made`` + ``elapsed_ms`` (wall time of
    the mission, construction included). Rows follow ``task.agents`` order.
    """
    from rescue import mapgen
    from rescue.agents import make_agent
    from rescue.runner import Mission, make_transition

    grid = mapgen.generate(task.scenario, task.rows, task.cols, task.seed)
    world_cfg = world_config_for(task)
    planner_cfg = task.planner_config()
    agent_cfg = AgentConfig(replan_mode=task.replan_mode)  # type: ignore[arg-type]
    base = _task_fields(task)
    rows: list[dict[str, Any]] = []
    for name in task.agents:
        t0 = time.perf_counter()
        agent = make_agent(name, make_transition(grid, world_cfg), planner_cfg, agent_cfg)
        mission = Mission(grid, agent, world_cfg, seed=task.seed, scenario=task.scenario)
        mission.run()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = dataclasses.asdict(mission.metrics())
        row = dict(base)
        for column in METRIC_COLUMNS:
            row[column] = metrics[column]
        if row["failure_reason"] is None:
            row["failure_reason"] = ""
        if not row["success"]:
            row["wasted_moves"] = math.nan
        row["plans_made"] = int(agent.stats.plans_made)
        row["elapsed_ms"] = elapsed_ms
        rows.append(row)
    return rows


def _run_indexed(item: tuple[int, Task]) -> tuple[int, list[dict[str, Any]]]:
    """Pool entry point: ``(task index, task)`` → ``(task index, rows)``."""
    index, task = item
    try:
        return index, run_task(task)
    except Exception as exc:  # re-raised in the parent with the failing task attached
        raise RuntimeError(f"task {index} failed: {task!r}: {exc!r}") from exc


def _init_worker() -> None:
    """Pool initialiser: import the heavy modules once per worker, not inside the first task."""
    import rescue.agents
    import rescue.mapgen
    import rescue.planners.value_iteration
    import rescue.runner  # noqa: F401


# --- the whole experiment --------------------------------------------------------------------
def default_workers() -> int:
    """``os.cpu_count() - RESERVED_CPUS``, at least 1."""
    return max(1, (os.cpu_count() or 1) - RESERVED_CPUS)


def chunksize_for(n_tasks: int, workers: int) -> int:
    """``imap_unordered`` chunk size: about ``CHUNKS_PER_WORKER`` chunks per worker, capped."""
    return max(1, min(MAX_CHUNKSIZE, n_tasks // max(1, workers * CHUNKS_PER_WORKER)))


def _submission_order(tasks: Sequence[Task]) -> list[tuple[int, Task]]:
    """Indexed tasks, largest maps first (longest-processing-time first shortens the tail).

    Python's sort is stable, so equal sizes keep their deterministic order. The submission
    order never affects the output: rows are sorted by task index before writing.
    """
    indexed = list(enumerate(tasks))
    indexed.sort(key=lambda item: -(item[1].rows * item[1].cols))
    return indexed


class _Progress:
    """Prints ``done/total`` with elapsed time and an ETA to stderr every ~5 %."""

    def __init__(self, total: int, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.step = max(1, math.ceil(total * PROGRESS_FRACTION))
        self.next = self.step
        self.t0 = time.perf_counter()

    def update(self, done: int) -> None:
        if not self.enabled or (done < self.next and done != self.total):
            return
        while self.next <= done:
            self.next += self.step
        elapsed = time.perf_counter() - self.t0
        eta = elapsed / done * (self.total - done) if done else math.nan
        print(
            f"[experiments] {done}/{self.total} tasks ({100.0 * done / self.total:.0f}%), "
            f"{elapsed:.1f} s elapsed, ETA {eta:.0f} s",
            file=sys.stderr,
            flush=True,
        )


def _execute(
    indexed: list[tuple[int, Task]], workers: int, progress: _Progress
) -> dict[int, list[dict[str, Any]]]:
    """Run the indexed tasks (in-process for one worker) and collect rows by task index."""
    results: dict[int, list[dict[str, Any]]] = {}
    if workers <= 1:
        for item in indexed:
            index, rows = _run_indexed(item)
            results[index] = rows
            progress.update(len(results))
        return results
    for name, value in BLAS_THREAD_LIMITS.items():
        os.environ.setdefault(name, value)  # children inherit the parent's environment
    ctx = mp.get_context("spawn")
    chunksize = chunksize_for(len(indexed), workers)
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for index, rows in pool.imap_unordered(_run_indexed, indexed, chunksize=chunksize):
            results[index] = rows
            progress.update(len(results))
    return results


def check_complete(
    tasks: Sequence[Task],
    results: dict[int, list[dict[str, Any]]],
    ordered: Sequence[dict[str, Any]],
) -> None:
    """Raise ``RuntimeError`` unless every task came back with one row per agent.

    A guard against silent row loss (a dropped chunk, a swallowed worker exception): a
    ``raw.csv`` with missing tasks would bias every rate and break the pairing.
    """
    missing = sorted(set(range(len(tasks))) - set(results))
    extra = sorted(set(results) - set(range(len(tasks))))
    if missing or extra:
        raise RuntimeError(
            f"{len(missing)} task(s) missing (first: {missing[:5]}) and {len(extra)} unexpected "
            f"task index(es) (first: {extra[:5]}) in the pool results"
        )
    short = [i for i, task in enumerate(tasks) if len(results[i]) != len(task.agents)]
    expected = sum(len(task.agents) for task in tasks)
    if short or len(ordered) != expected:
        raise RuntimeError(
            f"expected {expected} rows ({len(tasks)} tasks × their agents), got {len(ordered)}; "
            f"tasks with the wrong row count: {short[:5]}"
        )


def _format_cell(column: str, value: Any) -> Any:
    """CSV text of one value: timing columns rounded, None → ''."""
    if value is None:
        return ""
    if column in TIMING_COLUMNS and isinstance(value, float) and math.isfinite(value):
        return round(value, TIMING_DECIMALS)
    return value


def write_raw(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write rows to ``path`` as CSV with ``RAW_COLUMNS`` (LF line endings, atomic replace)."""
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


def git_commit() -> str | None:
    """Short hash of the checked-out commit (None when git or the repository is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def read_run_meta(out_dir: str | Path) -> dict[str, Any] | None:
    """``run_meta.json`` of an output directory, or None if it is missing or unreadable."""
    path = Path(out_dir) / META_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_experiments(
    cfg: ExperimentConfig,
    out_dir: str | Path,
    workers: int | None = None,
    progress: bool = True,
    *,
    max_tasks: int | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Run every task of ``cfg`` and write ``out_dir/raw.csv`` and ``out_dir/run_meta.json``.

    ``workers``: explicit count, else ``cfg.workers``, else :func:`default_workers`; 1 runs
    in-process (tests). ``max_tasks`` runs an evenly spaced subsample of the tasks
    (:func:`~rescue.experiments.spec.subsample_tasks`). Rows are sorted by task order
    (experiment, scenario, size, slip p, sensor radius, replan mode, aftershock q, replicate)
    then by agent in config order. Returns the path of ``raw.csv``.
    """
    out = Path(out_dir)
    all_tasks = expand_tasks(cfg)
    tasks = subsample_tasks(all_tasks, max_tasks)
    if workers is None:
        workers = cfg.workers if cfg.workers is not None else default_workers()
    workers = max(1, min(int(workers), len(tasks)))
    started = _dt.datetime.now().astimezone()
    t0 = time.perf_counter()
    if progress:
        print(
            f"[experiments] {cfg.name}: {len(tasks)} tasks"
            + (f" (sample of {len(all_tasks)})" if len(tasks) < len(all_tasks) else "")
            + f", {sum(len(t.agents) for t in tasks)} missions, {workers} worker(s)",
            file=sys.stderr,
            flush=True,
        )
    results = _execute(_submission_order(tasks), workers, _Progress(len(tasks), progress))
    ordered = [row for index in sorted(results) for row in results[index]]
    check_complete(tasks, results, ordered)
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
        print(f"[experiments] wrote {raw_path} ({len(ordered)} runs) in {wall_s:.1f} s",
              file=sys.stderr, flush=True)
    return raw_path


__all__ = [
    "META_FILE",
    "RAW_COLUMNS",
    "RAW_FILE",
    "TIMING_COLUMNS",
    "check_complete",
    "chunksize_for",
    "default_workers",
    "git_commit",
    "read_run_meta",
    "run_experiments",
    "run_task",
    "world_config_for",
    "write_raw",
]
