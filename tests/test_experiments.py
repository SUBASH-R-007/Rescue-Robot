"""Phase 6 tests of the Monte Carlo experiment harness (brief §6.6, §9; docs/plan.md §1, §5).

Covers the shared data contract of ``rescue.experiments``: config loading and task expansion
(``spec``), common-random-number seeds, running tasks and whole experiments (``run``), the
statistics (``stats``), the hypothesis verdicts (``hypotheses``), the plots (``plots``) and
the ``python -m rescue.experiments`` CLI. Everything runs on tiny matrices (6x6 maps, a few
replicates) or synthetic rows, so the module stays well under a minute; the real
``experiments/quick.yaml`` is only parsed, never run.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy import stats as sps

from rescue import mapgen
from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES, make_agent
from rescue.config import (
    MAPGEN_MIN_SIZE,
    MAX_SLIP_P,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.experiments import hypotheses, plots, run, spec, stats
from rescue.experiments.spec import (
    ExperimentBlock,
    ExperimentConfig,
    Task,
    expand_tasks,
    load_experiment_config,
)
from rescue.runner import MissionMetrics, make_transition, run_mission
from tests.conftest import REPO_ROOT

EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"
MAIN_CONFIG: Path = EXPERIMENTS_DIR / "main.yaml"
QUICK_CONFIG: Path = EXPERIMENTS_DIR / "quick.yaml"

#: Columns a run writes besides the MissionMetrics fields (the task description + extras).
TASK_COLUMNS: tuple[str, ...] = (
    "experiment",
    "scenario",
    "rows",
    "cols",
    "slip_p",
    "sensor_radius",
    "replan_mode",
    "aftershock_q",
    "aftershock_k",
    "replicate",
    "seed",
)
EXTRA_COLUMNS: tuple[str, ...] = ("plans_made", "elapsed_ms")
#: Wall-clock columns: the only ones allowed to differ between two runs of one config.
TIMING_COLUMNS: tuple[str, ...] = ("planning_ms", "elapsed_ms")
#: Size of the tiny maps used by the end-to-end tests (mapgen accepts >= MAPGEN_MIN_SIZE).
TINY_SIZE: int = 6
#: Scenarios of the tiny matrix, in alphabetical order (so "sorted by scenario" is unambiguous).
TINY_SCENARIOS: tuple[str, ...] = ("flooded_zone", "random_rubble")
TINY_N_MAPS: int = 3
TINY_BASE_SEED: int = 11
PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"


# --- helpers -----------------------------------------------------------------------------------


def make_block(name: str = "tiny", **overrides: Any) -> ExperimentBlock:
    """An ExperimentBlock of the tiny matrix, with keyword overrides."""
    values: dict[str, Any] = {
        "name": name,
        "scenarios": TINY_SCENARIOS,
        "sizes": (TINY_SIZE,),
        "slip_p": (0.0, 0.3),
    }
    values.update(overrides)
    return ExperimentBlock(**values)


def make_config(
    blocks: Sequence[ExperimentBlock] | None = None,
    *,
    agents: Sequence[str] = AGENT_NAMES,
    n_maps: int = TINY_N_MAPS,
    base_seed: int = TINY_BASE_SEED,
    workers: int | None = None,
    name: str = "tiny",
) -> ExperimentConfig:
    """A tiny ExperimentConfig (default: one block, 2 scenarios x 6x6 x 2 p x 3 maps)."""
    return ExperimentConfig(
        name=name,
        base_seed=base_seed,
        n_maps=n_maps,
        agents=tuple(agents),
        blocks=tuple(blocks) if blocks is not None else (make_block(),),
        workers=workers,
    )


def expected_task_count(cfg: ExperimentConfig) -> int:
    """Σ over blocks of |scenarios|·|sizes|·|p|·|radius|·|mode|·|q|·n_maps."""
    total = 0
    for b in cfg.blocks:
        n_maps = b.n_maps if b.n_maps is not None else cfg.n_maps
        total += (
            len(b.scenarios)
            * len(b.sizes)
            * len(b.slip_p)
            * len(b.sensor_radius)
            * len(b.replan_mode)
            * len(b.aftershock_q)
            * n_maps
        )
    return total


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Header and rows (as raw strings) of a CSV file."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        return header, list(reader)


def without(rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> list[dict[str, Any]]:
    """Copies of ``rows`` without ``columns``."""
    drop = set(columns)
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


def same_value(a: Any, b: Any) -> bool:
    """Equality that treats two NaNs as equal."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def synth_row(**values: Any) -> dict[str, Any]:
    """A raw.csv row (typed, like ``stats.read_raw``) with plausible defaults."""
    row: dict[str, Any] = {column: 0 for column in run.RAW_COLUMNS}
    row.update(
        experiment="slip_sweep",
        scenario="random_rubble",
        rows=10,
        cols=10,
        slip_p=0.2,
        sensor_radius=1,
        replan_mode="eager",
        aftershock_q=0.0,
        aftershock_k=2,
        replicate=0,
        seed=0,
        agent="andor",
        success=True,
        failure_reason="",
        steps=20,
        shortest_path=18,
        wasted_moves=2.0,
        cells_explored=30,
        planning_ms=1.0,
        elapsed_ms=2.0,
        plans_made=1,
        plan_size=5,
        nodes_expanded=10,
    )
    row.update(values)
    if not row["success"]:
        row["wasted_moves"] = math.nan
        row["failure_reason"] = row["failure_reason"] or "budget"
    return row


def pair_rows(
    diffs: Sequence[float],
    *,
    metric: str = "wasted_moves",
    agent_a: str = "andor",
    agent_b: str = "astar_replan",
    base: float = 10.0,
    **common: Any,
) -> list[dict[str, Any]]:
    """Two rows per replicate i: agent_b has ``base``, agent_a ``base + diffs[i]``."""
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(diffs):
        values = {metric: base, "replicate": i, "seed": 1000 + i, **common}
        rows.append(synth_row(agent=agent_b, **values))
        rows.append(synth_row(agent=agent_a, **{**values, metric: base + d}))
    return rows


# --- spec: config loading ----------------------------------------------------------------------


@pytest.mark.parametrize("path", [MAIN_CONFIG, QUICK_CONFIG], ids=["main", "quick"])
def test_shipped_configs_parse(path: Path) -> None:
    cfg = load_experiment_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name
    assert isinstance(cfg.base_seed, int)
    assert cfg.n_maps >= 1
    assert cfg.agents
    assert set(cfg.agents) <= set(ALL_AGENT_NAMES)
    assert cfg.blocks
    assert all(isinstance(b, ExperimentBlock) for b in cfg.blocks)
    assert len({b.name for b in cfg.blocks}) == len(cfg.blocks)
    for b in cfg.blocks:
        assert set(b.scenarios) <= set(mapgen.SCENARIOS)
        assert all(size >= MAPGEN_MIN_SIZE for size in b.sizes)
        assert all(0.0 <= p <= MAX_SLIP_P for p in b.slip_p)
        assert all(r >= 0 for r in b.sensor_radius)
        assert set(b.replan_mode) <= {"eager", "lazy"}
        assert all(0.0 <= q <= 1.0 for q in b.aftershock_q)
        assert b.agents is None or set(b.agents) <= set(ALL_AGENT_NAMES)
        assert b.n_maps is None or b.n_maps >= 1
    tasks = expand_tasks(cfg)
    assert len(tasks) == expected_task_count(cfg)


def test_main_config_covers_the_brief() -> None:
    """main.yaml: the three compared agents, a slip sweep with p = 0 and p = 0.3, sizes 10-20."""
    cfg = load_experiment_config(MAIN_CONFIG)
    assert set(AGENT_NAMES) <= set(cfg.agents)
    sweeps = [b for b in cfg.blocks if {0.0, 0.3} <= set(b.slip_p)]
    assert sweeps, "no block sweeps slip p over 0 and 0.3 (H1, H4)"
    sizes = {s for b in cfg.blocks for s in b.sizes}
    assert {10, 15, 20} <= sizes


def test_quick_config_is_smaller_than_main() -> None:
    main_cfg = load_experiment_config(MAIN_CONFIG)
    quick_cfg = load_experiment_config(QUICK_CONFIG)
    assert quick_cfg.n_maps <= main_cfg.n_maps
    assert len(expand_tasks(quick_cfg)) < len(expand_tasks(main_cfg))


def _first_block(data: dict[str, Any]) -> dict[str, Any]:
    """The first block mapping of a parsed config file (blocks as a list or a mapping)."""
    blocks = data["blocks"]
    if isinstance(blocks, dict):
        return next(iter(blocks.values()))
    return blocks[0]


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    data = yaml.safe_load(QUICK_CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "copy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_experiment_config(path) == load_experiment_config(QUICK_CONFIG)


@pytest.mark.parametrize("where", ["top", "block"])
def test_unknown_config_keys_are_rejected(tmp_path: Path, where: str) -> None:
    data = yaml.safe_load(QUICK_CONFIG.read_text(encoding="utf-8"))
    target = data if where == "top" else _first_block(data)
    target["definitely_not_a_key"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


# --- spec: task expansion ----------------------------------------------------------------------


def _seed_test_config(base_seed: int = TINY_BASE_SEED) -> ExperimentConfig:
    """Two blocks varying every axis that must not change a task's seed."""
    sweep = make_block(
        "sweep",
        scenarios=("random_rubble", "maze_corridors"),
        sizes=(6, 7),
        slip_p=(0.0, 0.2),
        sensor_radius=(1, 2),
        replan_mode=("eager", "lazy"),
        aftershock_q=(0.0, 0.05),
        n_maps=2,
    )
    other = make_block(
        "other",
        scenarios=("random_rubble",),
        sizes=(6,),
        slip_p=(0.4,),
        agents=("andor",),
        n_maps=3,
    )
    return make_config((sweep, other), agents=ALL_AGENT_NAMES, base_seed=base_seed)


def test_expand_tasks_count_order_and_fields() -> None:
    cfg = _seed_test_config()
    tasks = expand_tasks(cfg)
    assert len(tasks) == expected_task_count(cfg) == 2 * 2 * 2 * 2 * 2 * 2 * 2 + 3
    assert tasks == expand_tasks(cfg)  # deterministic order
    assert len(set(tasks)) == len(tasks)  # frozen, hashable, no duplicates
    assert all(isinstance(t, Task) for t in tasks)
    by_block = defaultdict(list)
    for t in tasks:
        by_block[t.experiment].append(t)
    assert set(by_block) == {"sweep", "other"}
    assert all(t.agents == ALL_AGENT_NAMES for t in by_block["sweep"])  # top-level agents
    assert all(t.agents == ("andor",) for t in by_block["other"])  # block override
    assert {t.replicate for t in by_block["sweep"]} == {0, 1}
    assert {t.replicate for t in by_block["other"]} == {0, 1, 2}
    for t in tasks:
        assert t.rows == t.cols
        assert t.aftershock_k == 2
    sweep = by_block["sweep"]
    combos = {(t.scenario, t.rows, t.slip_p, t.sensor_radius, t.replan_mode, t.aftershock_q)
              for t in sweep}
    assert len(combos) == 2**6
    assert {t.replan_mode for t in sweep} == {"eager", "lazy"}


def test_expand_tasks_uses_block_n_maps_override() -> None:
    cfg = make_config((make_block("a"), make_block("b", n_maps=1)), n_maps=4)
    tasks = expand_tasks(cfg)
    per_block = {name: sum(t.experiment == name for t in tasks) for name in ("a", "b")}
    assert per_block == {"a": 2 * 2 * 4, "b": 2 * 2 * 1}
    assert len(tasks) == expected_task_count(cfg)


def test_seed_depends_only_on_base_seed_scenario_size_and_replicate() -> None:
    tasks = expand_tasks(_seed_test_config())
    seeds: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    for t in tasks:
        assert isinstance(t.seed, int)
        assert 0 <= t.seed < 2**32
        seeds[(t.scenario, t.rows, t.cols, t.replicate)].add(t.seed)
    # Same map + outcome stream across p, radius, mode, q, blocks and agents (CRN) ...
    assert all(len(s) == 1 for s in seeds.values()), seeds
    # ... and different maps for different (scenario, size, replicate).
    distinct = [next(iter(s)) for s in seeds.values()]
    assert len(set(distinct)) == len(distinct)
    # The base seed changes every seed.
    other = {(t.scenario, t.rows, t.cols, t.replicate): t.seed
             for t in expand_tasks(_seed_test_config(base_seed=TINY_BASE_SEED + 1))}
    assert all(other[key] not in s for key, s in seeds.items())


_SEED_SCRIPT = """
import json
from rescue.experiments.spec import ExperimentBlock, ExperimentConfig, expand_tasks
block = ExperimentBlock(name="b", scenarios={scenarios!r}, sizes={sizes!r}, slip_p=(0.0,))
cfg = ExperimentConfig(name="x", base_seed={base_seed!r}, n_maps={n_maps!r},
                       agents=("andor",), blocks=(block,), workers=None)
print(json.dumps([[t.scenario, t.rows, t.cols, t.replicate, int(t.seed)]
                  for t in expand_tasks(cfg)]))
"""


def test_seeds_are_stable_across_processes() -> None:
    """Seeds must not depend on ``hash()`` (PYTHONHASHSEED) or anything process-local."""
    scenarios, sizes, base_seed, n_maps = ("random_rubble", "flooded_zone"), (6, 9), 123, 3
    cfg = make_config(
        (make_block("b", scenarios=scenarios, sizes=sizes, slip_p=(0.0,)),),
        agents=("andor",),
        base_seed=base_seed,
        n_maps=n_maps,
    )
    here = [[t.scenario, t.rows, t.cols, t.replicate, int(t.seed)] for t in expand_tasks(cfg)]
    script = _SEED_SCRIPT.format(
        scenarios=scenarios, sizes=sizes, base_seed=base_seed, n_maps=n_maps
    )
    for hash_seed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            timeout=120,
            check=True,
        )
        assert json.loads(out.stdout) == here


# --- run: one task -------------------------------------------------------------------------------


def _single_task(block: ExperimentBlock, agents: Sequence[str] = ALL_AGENT_NAMES) -> Task:
    tasks = expand_tasks(make_config((block,), agents=agents, n_maps=1))
    assert len(tasks) == 1
    return tasks[0]


def test_raw_columns_cover_task_metrics_and_extras() -> None:
    columns = list(run.RAW_COLUMNS)
    assert len(columns) == len(set(columns))
    metric_fields = [f.name for f in dataclasses.fields(MissionMetrics)]
    assert set(TASK_COLUMNS) | set(metric_fields) | set(EXTRA_COLUMNS) <= set(columns)


@pytest.mark.parametrize(
    "block",
    [
        make_block("terrain", scenarios=("flooded_zone",), slip_p=(0.3,)),
        make_block(
            "stress",
            scenarios=("random_rubble",),
            slip_p=(0.2,),
            sensor_radius=(2,),
            replan_mode=("lazy",),
            aftershock_q=(0.3,),
            aftershock_k=2,
        ),
    ],
    ids=["terrain", "stress"],
)
def test_run_task_runs_every_agent_on_one_map(block: ExperimentBlock) -> None:
    task = _single_task(block)
    rows = run.run_task(task)
    assert [row["agent"] for row in rows] == list(task.agents)
    for row in rows:
        assert set(run.RAW_COLUMNS) <= set(row)
        for column in TASK_COLUMNS:
            assert same_value(row[column], getattr(task, column)), column
        assert row["elapsed_ms"] >= 0.0
        assert row["planning_ms"] >= 0.0
    # Shared map and outcome stream (common random numbers).
    assert len({row["shortest_path"] for row in rows}) == 1
    assert len({row["seed"] for row in rows}) == 1
    # The fixed seeds exercise both outcomes (astar_fixed aborts) and the stress features.
    assert any(row["success"] for row in rows)
    assert not all(row["success"] for row in rows)
    if task.aftershock_q > 0:
        assert any(row["aftershock_events"] > 0 for row in rows)

    # Re-running every mission by hand with the documented recipe gives the same metrics.
    grid = mapgen.generate(task.scenario, task.rows, task.cols, task.seed)
    if task.scenario == "flooded_zone":
        assert grid.high_slip.any()  # TerrainSlip through make_transition
    world_cfg = WorldConfig(
        slip_p=task.slip_p,
        sensor_radius=task.sensor_radius,
        aftershock=AftershockConfig(q=task.aftershock_q, k=task.aftershock_k),
    )
    for row in rows:
        agent = make_agent(
            row["agent"],
            make_transition(grid, world_cfg),
            PlannerConfig(),
            AgentConfig(replan_mode=task.replan_mode),
        )
        mission = run_mission(grid, agent, world_cfg, seed=task.seed)
        metrics = mission.metrics()
        for name, value in metrics.as_dict().items():
            if name in ("planning_ms", "scenario"):
                continue
            if name == "failure_reason":
                assert row[name] == (value or ""), row["agent"]
            else:
                assert same_value(row[name], value), (row["agent"], name, row[name], value)
        assert row["plans_made"] == agent.stats.plans_made
        if metrics.success:
            assert row["failure_reason"] == ""
            assert row["wasted_moves"] == row["steps"] - row["shortest_path"]
        else:
            assert row["failure_reason"] in ("aborted", "unreachable", "budget")
            assert math.isnan(row["wasted_moves"])


# --- run: whole experiments --------------------------------------------------------------------


def _sort_key(row: dict[str, Any], agents: Sequence[str]) -> tuple[Any, ...]:
    return (
        row["experiment"],
        row["scenario"],
        row["rows"],
        row["slip_p"],
        row["sensor_radius"],
        row["replan_mode"],
        row["aftershock_q"],
        row["replicate"],
        list(agents).index(row["agent"]),
    )


def test_run_experiments_is_deterministic_across_worker_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config()
    tasks = expand_tasks(cfg)
    one = run.run_experiments(cfg, tmp_path / "w1", workers=1, progress=True)
    assert capsys.readouterr().err.strip(), "progress goes to stderr"
    two = run.run_experiments(cfg, tmp_path / "w2", workers=2, progress=False)
    assert one == tmp_path / "w1" / "raw.csv"
    assert two == tmp_path / "w2" / "raw.csv"

    header_one, rows_one = read_csv(one)
    header_two, rows_two = read_csv(two)
    assert header_one == header_two == list(run.RAW_COLUMNS)
    assert len(rows_one) == len(tasks) * len(cfg.agents)
    assert without(rows_one, TIMING_COLUMNS) == without(rows_two, TIMING_COLUMNS)

    typed = stats.read_raw(one)
    assert len(typed) == len(rows_one)
    keys = [_sort_key(row, cfg.agents) for row in typed]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    # Every agent ran every task, and one task's rows share the seed (paired design).
    per_task: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in typed:
        per_task[_sort_key(row, cfg.agents)[:-1]].append(row)
    assert len(per_task) == len(tasks)
    for group in per_task.values():
        assert [row["agent"] for row in group] == list(cfg.agents)
        assert len({row["seed"] for row in group}) == 1
        assert len({row["shortest_path"] for row in group}) == 1

    # The CSV holds exactly what run_task returns (types restored by read_raw).
    direct = sorted(
        (row for task in tasks for row in run.run_task(task)),
        key=lambda row: _sort_key(row, cfg.agents),
    )
    for got, want in zip(typed, direct, strict=True):
        for column in run.RAW_COLUMNS:
            if column in TIMING_COLUMNS:
                continue
            assert same_value(got[column], want[column]), (column, got[column], want[column])

    for out, workers in ((tmp_path / "w1", 1), (tmp_path / "w2", 2)):
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["base_seed"] == cfg.base_seed
        assert meta["n_tasks"] == len(tasks)
        assert meta["n_runs"] == len(rows_one)
        assert meta["workers"] == workers
        assert meta["wall_time_s"] >= 0.0
        assert "git_commit" in meta
        assert "config_path" in meta


def test_read_raw_types(tmp_path: Path) -> None:
    cfg = make_config(
        (make_block(scenarios=("random_rubble",), slip_p=(0.4,)),),
        agents=("astar_fixed", "andor"),
        n_maps=2,
    )
    path = run.run_experiments(cfg, tmp_path, workers=1, progress=False)
    rows = stats.read_raw(path)
    assert [row["agent"] for row in rows] == ["astar_fixed", "andor"] * 2
    assert {row["success"] for row in rows} == {True, False}  # both NaN and finite wasted moves
    for row in rows:
        assert isinstance(row["success"], bool)
        assert isinstance(row["rows"], int) and not isinstance(row["rows"], bool)
        assert isinstance(row["steps"], int)
        assert isinstance(row["replicate"], int)
        assert isinstance(row["seed"], int)
        assert isinstance(row["slip_p"], float)
        assert isinstance(row["wasted_moves"], float)
        assert isinstance(row["planning_ms"], float)
        assert isinstance(row["failure_reason"], str)
        assert row["success"] == (row["failure_reason"] == "")
        assert math.isnan(row["wasted_moves"]) == (not row["success"])


# --- stats: confidence intervals ---------------------------------------------------------------


def _wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Reference Wilson score interval (no continuity correction)."""
    z = sps.norm.ppf(0.5 + confidence / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (50, 100, 0.4038, 0.5962),
        (19, 20, 0.7639, 0.9911),
    ],
)
def test_wilson_interval_known_values(k: int, n: int, lo: float, hi: float) -> None:
    got_lo, got_hi = stats.wilson_interval(k, n)
    assert got_lo == pytest.approx(lo, abs=5e-4)
    assert got_hi == pytest.approx(hi, abs=5e-4)
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert got_lo == pytest.approx(ref.low, abs=1e-9)
    assert got_hi == pytest.approx(ref.high, abs=1e-9)


def test_wilson_interval_properties() -> None:
    for n in (1, 7, 20, 200):
        bounds = [stats.wilson_interval(k, n) for k in range(n + 1)]
        for k, (lo, hi) in enumerate(bounds):
            assert 0.0 <= lo <= k / n <= hi <= 1.0
            ref_lo, ref_hi = _wilson(k, n)
            assert lo == pytest.approx(ref_lo, abs=1e-9)
            assert hi == pytest.approx(ref_hi, abs=1e-9)
        los, his = zip(*bounds, strict=True)
        assert all(a < b for a, b in itertools.pairwise(los[1:]))  # strictly increasing in k
        assert all(a < b for a, b in itertools.pairwise(his[:-1]))
        # Symmetry: interval(k) mirrors interval(n - k).
        for k in range(n + 1):
            lo, hi = bounds[k]
            mlo, mhi = bounds[n - k]
            assert lo == pytest.approx(1 - mhi, abs=1e-12)
            assert hi == pytest.approx(1 - mlo, abs=1e-12)
    # More data → narrower; more confidence → wider.
    width = [hi - lo for lo, hi in (stats.wilson_interval(n // 2, n) for n in (10, 100, 1000))]
    assert width[0] > width[1] > width[2]
    lo90, hi90 = stats.wilson_interval(30, 50, confidence=0.90)
    lo99, hi99 = stats.wilson_interval(30, 50, confidence=0.99)
    assert lo99 < lo90 and hi99 > hi90


def test_mean_ci_matches_scipy_t_interval() -> None:
    rng = np.random.default_rng(3)
    for n in (2, 5, 40):
        values = rng.normal(10.0, 3.0, size=n)
        mean, lo, hi = stats.mean_ci(list(values))
        ref_lo, ref_hi = sps.t.interval(
            0.95, n - 1, loc=float(np.mean(values)), scale=float(sps.sem(values))
        )
        assert mean == pytest.approx(float(np.mean(values)))
        assert lo == pytest.approx(ref_lo)
        assert hi == pytest.approx(ref_hi)
        lo99 = stats.mean_ci(list(values), confidence=0.99)[1]
        assert lo99 < lo


def test_mean_ci_drops_nans_and_needs_two_values() -> None:
    values = [1.0, math.nan, 3.0, math.nan, 5.0]
    mean, lo, hi = stats.mean_ci(values)
    ref_lo, ref_hi = sps.t.interval(0.95, 2, loc=3.0, scale=float(sps.sem([1.0, 3.0, 5.0])))
    assert mean == pytest.approx(3.0)
    assert (lo, hi) == (pytest.approx(ref_lo), pytest.approx(ref_hi))
    mean, lo, hi = stats.mean_ci([4.0, math.nan])
    assert mean == pytest.approx(4.0)
    assert math.isnan(lo) and math.isnan(hi)


# --- stats: paired differences -----------------------------------------------------------------

PAIR_KEYS: tuple[str, ...] = (
    "experiment",
    "scenario",
    "rows",
    "slip_p",
    "sensor_radius",
    "replan_mode",
    "aftershock_q",
    "replicate",
)


def test_paired_difference_pairs_on_task_keys() -> None:
    diffs = [-3.0, -1.0, -2.0, -4.0, -2.5, -1.5, -2.0, -3.5]
    rows = pair_rows(diffs, metric="replans")
    rows.append(synth_row(agent="andor", replans=99, replicate=50, seed=50))  # no partner
    rows.append(synth_row(agent="astar_replan", replans=-99, replicate=51, seed=51))  # no partner
    rows.append(synth_row(agent="astar_fixed", replans=500, replicate=0, seed=1000))  # other agent
    rng = np.random.default_rng(0)
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    mean, lo, hi, n_pairs, p_value = stats.paired_difference(
        shuffled, "replans", "andor", "astar_replan", PAIR_KEYS
    )
    d = np.asarray(diffs)
    ref_lo, ref_hi = sps.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sps.sem(d))
    assert n_pairs == len(diffs)
    assert mean == pytest.approx(d.mean())
    assert (lo, hi) == (pytest.approx(ref_lo), pytest.approx(ref_hi))
    assert hi < 0
    assert p_value == pytest.approx(float(sps.ttest_1samp(d, 0.0).pvalue))
    assert 0.0 <= p_value < 0.05
    result = stats.paired_difference(shuffled, "replans", "andor", "astar_replan", PAIR_KEYS)
    p_wilcoxon = getattr(result, "p_wilcoxon", None)
    assert p_wilcoxon == pytest.approx(float(sps.wilcoxon(d).pvalue))
    # Swapping the agents flips the sign.
    mean_ba, lo_ba, hi_ba, n_ba, p_ba = stats.paired_difference(
        shuffled, "replans", "astar_replan", "andor", PAIR_KEYS
    )
    assert n_ba == n_pairs
    assert mean_ba == pytest.approx(-mean)
    assert (lo_ba, hi_ba) == (pytest.approx(-hi), pytest.approx(-lo))
    assert p_ba == pytest.approx(p_value)


def test_paired_difference_keys_separate_conditions() -> None:
    """Rows only pair within identical keys: a p = 0.1 andor run never meets a p = 0.2 A* run."""
    rows = pair_rows([1.0, 1.0, 1.0], metric="steps", slip_p=0.1)
    rows += pair_rows([-5.0, -5.0], metric="steps", slip_p=0.2)
    rows += [synth_row(agent="andor", steps=0, slip_p=0.3, replicate=0, seed=1000)]
    mean, _lo, _hi, n_pairs, _p = stats.paired_difference(
        rows, "steps", "andor", "astar_replan", PAIR_KEYS
    )
    assert n_pairs == 5
    assert mean == pytest.approx((3 * 1.0 - 2 * 5.0) / 5)


def test_paired_wasted_moves_needs_both_successes() -> None:
    rows = pair_rows([-1.0, -2.0, -3.0, -2.0])
    # A failed andor run and a failed A* run: both pairs must be dropped.
    rows += [
        synth_row(agent="astar_replan", wasted_moves=4.0, replicate=10, seed=10),
        synth_row(agent="andor", success=False, replicate=10, seed=10),
        synth_row(agent="astar_replan", success=False, replicate=11, seed=11),
        synth_row(agent="andor", wasted_moves=1.0, replicate=11, seed=11),
    ]
    mean, lo, hi, n_pairs, _p = stats.paired_difference(
        rows, "wasted_moves", "andor", "astar_replan", PAIR_KEYS
    )
    assert n_pairs == 4
    assert mean == pytest.approx(-2.0)
    assert lo < mean < hi < 0


def test_paired_difference_with_too_few_pairs() -> None:
    mean, lo, hi, n_pairs, _p = stats.paired_difference(
        pair_rows([2.0]), "wasted_moves", "andor", "astar_replan", PAIR_KEYS
    )
    assert n_pairs == 1
    assert mean == pytest.approx(2.0)
    assert math.isnan(lo) and math.isnan(hi)
    mean, lo, hi, n_pairs, _p = stats.paired_difference(
        [], "wasted_moves", "andor", "astar_replan", PAIR_KEYS
    )
    assert n_pairs == 0
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


# --- stats: summary ------------------------------------------------------------------------------


def _summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p, agent in itertools.product((0.0, 0.3), ("andor", "astar_fixed")):
        for i in range(5):
            success = not (agent == "astar_fixed" and p > 0 and i >= 2)
            rows.append(
                synth_row(
                    agent=agent,
                    slip_p=p,
                    replicate=i,
                    seed=i,
                    success=success,
                    steps=20 + i,
                    wasted_moves=float(i),
                    replans=i,
                    failure_reason="" if success else "aborted",
                )
            )
    rows += [synth_row(scenario="maze_corridors", replicate=i, seed=i) for i in range(3)]
    return rows


def test_summarize_groups_and_columns(tmp_path: Path) -> None:
    rows = _summary_rows()
    summary = stats.summarize(rows)
    assert len(summary) == 5  # 2 p x 2 agents on rubble + 1 maze group
    for group in summary:
        assert list(group) == list(stats.SUMMARY_COLUMNS)
    assert set(stats.SUMMARY_METRICS) >= {
        "steps", "wasted_moves", "replans", "unplanned_states", "planning_ms", "bumps",
    }
    group_keys = (
        "experiment", "scenario", "rows", "slip_p", "sensor_radius", "replan_mode",
        "aftershock_q", "agent",
    )
    columns = list(stats.SUMMARY_COLUMNS)
    assert len(columns) == len(set(columns))
    assert columns[: len(group_keys)] == list(group_keys)
    assert {"n", "successes", "success_rate", "success_lo", "success_hi"} <= set(columns)
    for metric in (*stats.SUMMARY_METRICS, "steps_all"):
        for suffix in ("mean", "lo", "hi"):
            assert f"{metric}_{suffix}" in columns
    keys = [tuple(group[k] for k in group_keys) for group in summary]
    assert len(set(keys)) == len(keys)

    def find(**where: Any) -> dict[str, Any]:
        (match,) = [g for g in summary if all(g[k] == v for k, v in where.items())]
        return match

    fixed = find(agent="astar_fixed", slip_p=0.3, scenario="random_rubble")
    assert fixed["n"] == 5
    assert fixed["successes"] == 2
    assert fixed["success_rate"] == pytest.approx(0.4)
    lo, hi = stats.wilson_interval(2, 5)
    assert fixed["success_lo"] == pytest.approx(lo)
    assert fixed["success_hi"] == pytest.approx(hi)
    andor = find(agent="andor", slip_p=0.3, scenario="random_rubble")
    assert andor["n"] == andor["successes"] == 5
    assert andor["success_rate"] == pytest.approx(1.0)
    maze = find(scenario="maze_corridors")
    assert maze["n"] == 3

    # Success-only metrics: wasted moves and steps over the 2 successful runs (i = 0, 1);
    # steps_all and every other metric over all 5 runs.
    assert fixed["wasted_moves_mean"] == pytest.approx(0.5)
    assert fixed["steps_mean"] == pytest.approx(20.5)
    assert fixed["steps_all_mean"] == pytest.approx(22.0)
    assert fixed["replans_mean"] == pytest.approx(2.0)
    ref_lo, ref_hi = stats.mean_ci([0, 1, 2, 3, 4])[1:]
    assert (fixed["replans_lo"], fixed["replans_hi"]) == (
        pytest.approx(ref_lo), pytest.approx(ref_hi),
    )
    assert maze["replans_mean"] == maze["replans_lo"] == maze["replans_hi"] == 0.0  # no spread

    path = tmp_path / "summary.csv"
    stats.write_summary(rows, path)
    header, written = read_csv(path)
    assert header == list(stats.SUMMARY_COLUMNS)
    assert len(written) == len(summary)
    first = written[0]
    assert int(first["n"]) == summary[0]["n"]
    assert float(first["success_rate"]) == pytest.approx(summary[0]["success_rate"])


# --- hypotheses ----------------------------------------------------------------------------------

PASS, FAIL, WARN = "✅", "❌", "⚠️"


def _claims(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``evaluate(rows)`` keyed H1..H6 (by label, else by position)."""
    results = hypotheses.evaluate(rows)
    assert len(results) == 6
    out = {}
    for i, h in enumerate(results):
        assert h.verdict in (PASS, FAIL, WARN)
        assert h.claim and h.measured and h.detail
        out[getattr(h, "label", "") or f"H{i + 1}"] = h
    assert list(out) == [f"H{i}" for i in range(1, 7)]
    return out


def _sweep_rows(
    *,
    n: int = 100,
    ps: Sequence[float] = (0.0, 0.1, 0.3),
    andor_failures: dict[float, int] | None = None,
    wasted_diff: float = -1.0,
    spread: float = 0.5,
    p0_diff: float = 0.0,
    unplanned_runs: int = 0,
    agents: Sequence[str] = AGENT_NAMES,
    experiment: str = spec.SLIP_SWEEP_EXPERIMENT,
) -> list[dict[str, Any]]:
    """Slip-sweep rows with controlled outcomes (deterministic, no RNG).

    Per p and replicate i: ``andor`` fails for i < andor_failures[p]; its wasted moves are
    ``astar_replan``'s plus ``wasted_diff ± spread`` (alternating in pairs of replicates, so the
    mean difference is exactly ``wasted_diff`` over a multiple of 4 pairs) at p > 0 and
    ``p0_diff`` at p = 0; ``andor`` replans once, ``astar_replan`` 4 times; ``astar_fixed``
    succeeds only at p = 0 and otherwise aborts after a slip. The first ``unplanned_runs``
    andor runs at the largest p have one unplanned state.
    """
    failures = andor_failures or {}
    scenarios = ("random_rubble", "maze_corridors")
    rows: list[dict[str, Any]] = []
    for p in ps:
        for i in range(n):
            common = {
                "experiment": experiment,
                "slip_p": p,
                "replicate": i,
                "seed": i,
                "scenario": scenarios[i % 2],
            }
            base = 5.0 + i % 4
            sign = 1.0 if (i // 2) % 2 == 0 else -1.0
            diff = wasted_diff + sign * spread if p > 0 else p0_diff
            if "andor" in agents:
                unplanned = 1 if (p == max(ps) and i < unplanned_runs) else 0
                rows.append(synth_row(agent="andor", success=i >= failures.get(p, 0),
                                      wasted_moves=base + diff, replans=1,
                                      unplanned_states=unplanned, **common))
            if "astar_replan" in agents:
                rows.append(synth_row(agent="astar_replan", wasted_moves=base, replans=4,
                                      slips=1 if p > 0 else 0, **common))
            if "astar_fixed" in agents:
                ok = p == 0
                rows.append(synth_row(agent="astar_fixed", success=ok,
                                      failure_reason="" if ok else "aborted",
                                      slips=0 if ok else 1, **common))
    return rows


@pytest.mark.parametrize(
    ("failures", "verdict", "rate"),
    [(0, PASS, "100.0%"), (5, PASS, "95.0%"), (7, WARN, "93.0%"), (20, FAIL, "80.0%")],
)
def test_h1_verdict_rule(failures: int, verdict: str, rate: str) -> None:
    rows = _sweep_rows(andor_failures={0.3: failures})
    # Failures in another block (aftershocks) must not count against the slip sweep.
    rows += [synth_row(experiment="aftershocks", slip_p=0.3, aftershock_q=0.2, success=False,
                       replicate=i, seed=i) for i in range(30)]
    h1 = _claims(rows)["H1"]
    assert h1.verdict == verdict
    assert rate in h1.measured
    lo, hi = stats.wilson_interval(100 - failures, 100)
    assert verdict != WARN or (lo < 0.95 <= hi)
    for scenario in ("random_rubble", "maze_corridors"):  # per-scenario rates
        assert scenario in h1.detail


def test_h2_counts_every_unplanned_state() -> None:
    claims = _claims(_sweep_rows())
    assert claims["H2"].verdict == PASS
    assert "0" in claims["H2"].measured
    assert "bump" in claims["H2"].measured.lower()
    # A single unplanned state anywhere (even outside the slip sweep) refutes the claim.
    rows = _sweep_rows()
    rows.append(synth_row(experiment="sensor_range", sensor_radius=2, unplanned_states=1))
    assert _claims(rows)["H2"].verdict == FAIL
    assert _claims(_sweep_rows(unplanned_runs=1))["H2"].verdict == FAIL


@pytest.mark.parametrize(("wasted_diff", "verdict"), [(-1.0, PASS), (0.0, WARN), (1.0, FAIL)])
def test_h3_verdict_rule(wasted_diff: float, verdict: str) -> None:
    rows = _sweep_rows(wasted_diff=wasted_diff, spread=0.5 if wasted_diff else 2.0)
    h3 = _claims(rows)["H3"]
    assert h3.verdict == verdict
    pd = stats.paired_difference(
        [r for r in rows if r["slip_p"] > 0], "wasted_moves", "andor", "astar_replan",
        stats.TASK_KEYS,
    )
    assert pd[3] == 200
    assert pd[0] == pytest.approx(wasted_diff)
    assert f"{wasted_diff:+.2f}" in h3.measured  # the verdict states the number
    for p in ("0.1", "0.3"):  # per-p rows
        assert p in h3.detail


def test_h3_ignores_p0_and_failed_pairs() -> None:
    # A huge andor penalty at p = 0 would flip the verdict if p = 0 were pooled.
    rows = _sweep_rows(wasted_diff=-1.0, p0_diff=50.0, andor_failures={0.1: 4, 0.3: 4})
    h3 = _claims(rows)["H3"]
    assert h3.verdict == PASS
    assert "192 pairs" in h3.measured  # 2 x (100 - 4) both-successful pairs at p > 0


def test_h4_and_h5_on_clear_data() -> None:
    claims = _claims(_sweep_rows())
    assert claims["H4"].verdict == PASS
    assert "100.0%" in claims["H4"].measured and "0.0%" in claims["H4"].measured
    assert "aborted" in claims["H4"].detail
    assert claims["H5"].verdict == PASS  # andor replans 1 vs 4 on every task
    assert "-3.00" in claims["H5"].measured
    # astar_fixed doing as well at p >= 0.1 as at p = 0 refutes the deck narrative.
    rows = [
        synth_row(**{**r, "success": True, "failure_reason": "", "wasted_moves": 0.0})
        if r["agent"] == "astar_fixed" else r
        for r in _sweep_rows()
    ]
    assert _claims(rows)["H4"].verdict == FAIL


def test_missing_data_is_not_measured() -> None:
    claims = _claims([])
    for h in claims.values():
        assert h.verdict == WARN
        assert "not measured" in h.measured
    # Without astar_replan, H3 and H5 cannot be measured; without p = 0.3, neither can H1.
    rows = _sweep_rows(ps=(0.0, 0.1), agents=("andor", "astar_fixed"))
    claims = _claims(rows)
    for tag in ("H1", "H3", "H5", "H6"):
        assert claims[tag].verdict == WARN, tag
        assert "not measured" in claims[tag].measured, tag
    assert claims["H2"].verdict == PASS
    assert claims["H4"].verdict == PASS


def test_write_hypotheses_report(tmp_path: Path) -> None:
    rows = _sweep_rows(andor_failures={0.3: 7})
    meta = {
        "config_name": "tiny_config",
        "config_path": "experiments/tiny.yaml",
        "base_seed": 77,
        "n_tasks": 300,
        "workers": 3,
        "wall_time_s": 12.5,
        "date": "2026-10-06",
        "git_commit": "abc1234",
    }
    (tmp_path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    path = hypotheses.write_hypotheses(rows, tmp_path / "HYPOTHESES.md")
    assert path == tmp_path / "HYPOTHESES.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# ")
    for needle in ("tiny_config", "2026-10-06", str(len(rows)), "abc1234", "77"):
        assert needle in text, needle
    claims = hypotheses.evaluate(rows)
    table = [line for line in text.splitlines() if line.startswith("|")]
    table = table[: 2 + len(claims)]  # the claims table comes first
    for h, line in zip(claims, table[2:], strict=True):
        assert h.verdict in line
    assert WARN in table[2]  # H1 at 93 %
    for h in claims:
        assert h.detail.splitlines()[0] in text
    # Without metadata the report still renders.
    bare = hypotheses.write_hypotheses(rows, tmp_path / "bare" / "HYPOTHESES.md")
    assert bare.read_text(encoding="utf-8").startswith("# ")


# --- plots ---------------------------------------------------------------------------------------

#: Figures that need only the slip sweep.
SLIP_PLOTS: frozenset[str] = frozenset(
    {
        "success_vs_p.png",
        "steps_vs_p.png",
        "wasted_moves_vs_p.png",
        "replans_vs_p.png",
        "cells_explored_vs_p.png",
        "paired_wasted_moves_diff.png",
        "paired_replans_diff.png",
    }
)
#: Every figure make_all_plots draws when all the blocks are present.
ALL_PLOTS: frozenset[str] = SLIP_PLOTS | {
    "planning_time_vs_size.png",
    "success_under_aftershocks.png",
    "sensor_radius.png",
    "replan_mode.png",
}


def _is_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and path.read_bytes()[:8] == PNG_SIGNATURE


#: Replicates per synthetic plot condition (plots draw no mean of fewer than 5 missions).
PLOT_N: int = 8


def _plot_rows(with_blocks: bool) -> list[dict[str, Any]]:
    """Synthetic raw rows: a slip sweep, plus (optionally) every stress block."""
    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []

    def add(experiment: str, agents: Sequence[str], n: int, **conditions: Any) -> None:
        p = conditions.get("slip_p", 0.2)
        for i in range(n):
            for agent in agents:
                success = bool(rng.random() > 0.15 * p)
                if agent == "astar_fixed" and p > 0:
                    success = bool(rng.random() < 0.3)
                steps = int(18 + rng.integers(0, 8))
                rows.append(synth_row(
                    experiment=experiment, agent=agent, replicate=i, seed=i, success=success,
                    failure_reason="" if success else "aborted", steps=steps,
                    wasted_moves=float(steps - 18), replans=int(rng.integers(0, 5)),
                    cells_explored=int(30 + rng.integers(0, 20)),
                    planning_ms=float(rng.uniform(0.5, 5.0) * conditions.get("rows", 10)),
                    **conditions,
                ))

    for scenario in ("random_rubble", "maze_corridors"):
        for p in (0.0, 0.2, 0.4):
            add("slip_sweep", AGENT_NAMES, PLOT_N, scenario=scenario, slip_p=p)
    if with_blocks:
        for size in (10, 15):
            add("scaling", ("andor", "astar_replan"), PLOT_N, rows=size, cols=size)
        for q in (0.0, 0.1):
            add("aftershocks", AGENT_NAMES, PLOT_N, aftershock_q=q)
        for radius in (1, 2):
            add("sensor_range", AGENT_NAMES, PLOT_N, sensor_radius=radius)
        for mode in ("eager", "lazy"):
            add("replan_mode", ("andor", "astar_replan"), PLOT_N, replan_mode=mode)
    return rows


def test_agent_style_is_fixed() -> None:
    expected = {
        "andor": ("#2a78d6", "o"),
        "astar_fixed": ("#eb6834", "s"),
        "astar_replan": ("#1baf7a", "^"),
        "dstar_lite": ("#eda100", "D"),
        "vi_expected": ("#e87ba4", "v"),
    }
    got = {
        agent: (style.colour.lower(), style.marker) for agent, style in plots.AGENT_STYLE.items()
    }
    assert got == expected


def test_make_all_plots_writes_every_figure(tmp_path: Path) -> None:
    out = tmp_path / "plots"
    paths = [Path(p) for p in plots.make_all_plots(_plot_rows(with_blocks=True), out)]
    assert {p.name for p in paths} == ALL_PLOTS
    assert len(paths) == len(ALL_PLOTS)
    assert {p.name for p in out.iterdir()} == ALL_PLOTS
    for path in paths:
        assert path.parent == out
        assert _is_png(path), path


def test_make_all_plots_skips_missing_blocks(tmp_path: Path) -> None:
    out = tmp_path / "plots"
    paths = [Path(p) for p in plots.make_all_plots(_plot_rows(with_blocks=False), out)]
    assert {p.name for p in paths} == SLIP_PLOTS
    assert all(_is_png(p) for p in paths)
    assert {p.name for p in out.iterdir()} == SLIP_PLOTS
    assert plots.make_all_plots([], tmp_path / "empty" / "plots") == []


# --- CLI -----------------------------------------------------------------------------------------


def _cli_module() -> Any:
    import importlib

    return importlib.import_module("rescue.experiments.__main__")


def test_cli_quick_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--quick --no-plots --out DIR`` on a monkeypatched tiny config (never the real one)."""
    cli = _cli_module()
    tiny = make_config(
        (make_block(spec.SLIP_SWEEP_EXPERIMENT),), n_maps=2, workers=1, name="tiny_cli"
    )
    loaded: list[Path] = []

    def fake_load(path: str | Path) -> ExperimentConfig:
        loaded.append(Path(path))
        return tiny

    monkeypatch.setattr(cli, "load_experiment_config", fake_load)
    out = tmp_path / "out"
    code = cli.main(["--quick", "--no-plots", "--quiet", "--out", str(out)])
    assert code == 0
    assert len(loaded) == 1 and loaded[0].name == "quick.yaml"

    header, rows = read_csv(out / "raw.csv")
    assert header == list(run.RAW_COLUMNS)
    assert len(rows) == len(expand_tasks(tiny)) * len(tiny.agents)
    summary_header, summary = read_csv(out / "summary.csv")
    assert summary_header == list(stats.SUMMARY_COLUMNS)
    assert len(summary) == 2 * 2 * len(tiny.agents)  # scenarios x p x agents
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["workers"] == 1  # the config's worker count
    assert str(meta["config_path"]).endswith("quick.yaml")
    report = (out / "HYPOTHESES.md").read_text(encoding="utf-8")
    assert "tiny_cli" in report
    assert not (out / "plots").exists()

    captured = capsys.readouterr()
    assert "[experiments]" not in captured.err  # --quiet
    assert "andor" in captured.out


def test_cli_rejects_bad_arguments(tmp_path: Path) -> None:
    cli = _cli_module()
    assert cli.main(["--config", str(tmp_path / "missing.yaml"), "--out", str(tmp_path)]) == 2
    assert cli.main(["--quick", "--workers", "0", "--out", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        cli.main(["--quick", "--config", str(MAIN_CONFIG)])
    assert not (tmp_path / "raw.csv").exists()


# --- fix-pass regressions (statistics review, plots QA) ------------------------------------------

#: Slip probabilities > 0 of the regression sweeps (one map reused at each, as in the harness).
CRN_PS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)


def _crn_pair_rows(
    map_diffs: Sequence[float], *, metric: str = "wasted_moves", scenario: str = "random_rubble",
    seed0: int = 1000,
) -> list[dict[str, Any]]:
    """Paired andor / astar_replan rows: map i has difference ``map_diffs[i]`` at every p > 0
    (common random numbers make a map's differences across p near-identical)."""
    rows: list[dict[str, Any]] = []
    for p in CRN_PS:
        for i, d in enumerate(map_diffs):
            rows += pair_rows([d], metric=metric, slip_p=p, scenario=scenario, replicate=i,
                              seed=seed0 + i)
    return rows


def test_paired_difference_clusters_on_maps() -> None:
    """Regression: a map reused at 4 p values is one unit, not 4 (pseudo-replication)."""
    map_diffs = [-3.0, 1.0, -2.0, 0.5, -1.0, 2.0, -2.5, 0.0, -1.5, 1.0]
    rows = _crn_pair_rows(map_diffs)
    pooled = stats.paired_difference(rows, "wasted_moves", "andor", "astar_replan")
    clustered = stats.paired_difference(rows, "wasted_moves", "andor", "astar_replan",
                                        cluster_keys=stats.MAP_KEYS)
    d = np.asarray(map_diffs)
    assert pooled.n_pairs == clustered.n_pairs == 4 * len(map_diffs)
    assert pooled.n_units == 4 * len(map_diffs)
    assert clustered.n_units == len(map_diffs)
    assert clustered.mean_diff == pytest.approx(d.mean())
    ref_lo, ref_hi = sps.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sps.sem(d))
    assert (clustered.lo, clustered.hi) == (pytest.approx(ref_lo), pytest.approx(ref_hi))
    assert clustered.p_value == pytest.approx(float(sps.ttest_1samp(d, 0.0).pvalue))
    assert clustered.n_nonzero == int(np.count_nonzero(d))
    # The pooled interval is about half as wide: it counts each map four times.
    assert (clustered.hi - clustered.lo) > 1.8 * (pooled.hi - pooled.lo)


def test_wilcoxon_keeps_zero_differences_in_the_ranking() -> None:
    """Regression: zero_method='pratt', so a mostly-tied sample is not overstated."""
    diffs = [0.0] * 12 + [-1.0, -2.0, -1.0, -3.0, 1.0, -1.0, -2.0]
    pd = stats.paired_difference(pair_rows(diffs, metric="replans"), "replans", "andor",
                                 "astar_replan")
    assert stats.WILCOXON_ZERO_METHOD == "pratt"
    expected = float(sps.wilcoxon(np.asarray(diffs), zero_method="pratt").pvalue)
    assert pd.p_wilcoxon == pytest.approx(expected)
    assert pd.p_wilcoxon > float(sps.wilcoxon(np.asarray(diffs), zero_method="wilcox").pvalue)
    assert pd.n_nonzero == 7 and pd.n_units == len(diffs)


def test_sign_test() -> None:
    assert stats.sign_test([1.0, 1.0, 1.0, 0.0, -1.0, math.nan]) == (
        3, 1, pytest.approx(float(sps.binomtest(3, 4, 0.5).pvalue)))
    assert stats.sign_test([0.0, 0.0]) == (0, 0, 1.0)


def test_h3_uses_maps_not_pairs_as_units() -> None:
    """Regression: 30 maps x 4 p. Pooled over 120 'pairs' the CI excludes 0 (old ✅); over the
    30 maps it contains 0, so the verdict is ⚠️ and the report states the unit."""
    map_diffs = [-0.3 + (1.0 if i % 2 else -1.0) for i in range(30)]
    rows = _crn_pair_rows(map_diffs)
    pooled = stats.paired_difference(rows, "wasted_moves", "andor", "astar_replan")
    assert pooled.hi < 0  # the old, pseudo-replicated verdict would have been ✅
    h3 = _claims(rows)["H3"]
    assert h3.verdict == WARN
    assert "over 30 maps (120 pairs)" in h3.measured
    assert "Unit of analysis: the map" in h3.detail


def test_h3_reports_scenarios_against_the_claim() -> None:
    """Regression: a scenario where andor wastes more moves is named, not hidden by pooling."""
    good = [-2.0 + (0.5 if i % 2 else -0.5) for i in range(20)]
    bad = [2.0 + (0.5 if i % 2 else -0.5) for i in range(20)]
    rows = _crn_pair_rows(good) + _crn_pair_rows(bad, scenario="maze_corridors", seed0=5000)
    h3 = _claims(rows)["H3"]
    assert h3.verdict == WARN  # pooled over both scenarios: mean 0
    assert "CI above 0 (claim refuted) in maze_corridors" in h3.measured
    assert "By scenario" in h3.detail and "By grid size" in h3.detail
    scenario_lines = [line for line in h3.detail.splitlines()
                      if line.startswith(("| random_rubble", "| maze_corridors"))]
    assert any(line.startswith("| random_rubble") and PASS in line for line in scenario_lines)
    assert any(line.startswith("| maze_corridors") and FAIL in line for line in scenario_lines)


def test_h1_states_the_lower_bound_and_low_scenarios() -> None:
    """Regression: 19/20 passes on the point estimate; the report says the CI does not."""
    h1 = _claims(_sweep_rows(n=20, andor_failures={0.3: 1}))["H1"]
    assert h1.verdict == PASS
    lo, _ = stats.wilson_interval(19, 20)
    assert f"Wilson lower bound {100 * lo:.1f}%" in h1.measured
    assert "not statistically established" in h1.measured
    assert "random_rubble 90.0%" in h1.measured  # replicate 0 is a rubble map
    clean = _claims(_sweep_rows(n=100))["H1"]
    assert "≥ 95.0%" in clean.measured and "below" not in clean.measured


def test_h2_counts_unique_missions() -> None:
    """Regression: default-condition runs of another block replay slip-sweep missions."""
    rows = _sweep_rows(n=10, ps=(0.0, 0.1, 0.3))
    andor = [r for r in rows if r["agent"] == "andor"]
    replay = [{**r, "experiment": "aftershocks", "aftershock_k": 5} for r in andor
              if r["slip_p"] == 0.1]  # q = 0: k is irrelevant, the mission is identical
    shaken = [{**r, "experiment": "aftershocks", "aftershock_q": 0.2} for r in andor
              if r["slip_p"] == 0.1]
    h2 = _claims(rows + replay + shaken)["H2"]
    assert h2.verdict == PASS
    assert "over 50 `andor` runs (40 unique missions)" in h2.measured


def test_h4_pairs_success_on_maps() -> None:
    """Regression: 4 maps reused at 4 p values. The pooled Wilson CIs (4/4 vs 0/16) are
    disjoint, but only 4 independent maps changed: the sign test (p = 0.125) is not
    significant, so H4 is ⚠️ rather than ✅."""
    rows = _sweep_rows(n=4, ps=(0.0, *CRN_PS))
    h4 = _claims(rows)["H4"]
    assert h4.verdict == WARN
    assert "sign test p = 0.12" in h4.measured
    assert "4 maps worse, 0 better" in h4.measured
    assert _claims(_sweep_rows(n=40, ps=(0.0, *CRN_PS)))["H4"].verdict == PASS


def test_check_complete_rejects_lost_rows() -> None:
    tasks = expand_tasks(make_config())
    full: dict[int, list[dict[str, Any]]] = {i: [{}] * len(t.agents) for i, t in enumerate(tasks)}
    ordered = [row for i in sorted(full) for row in full[i]]
    run.check_complete(tasks, full, ordered)  # complete: no error
    lost_task = {i: rows for i, rows in full.items() if i != 3}
    with pytest.raises(RuntimeError, match="missing"):
        run.check_complete(tasks, lost_task, [r for i in sorted(lost_task) for r in lost_task[i]])
    short = {**full, 0: full[0][1:]}
    with pytest.raises(RuntimeError, match="expected"):
        run.check_complete(tasks, short, [r for i in sorted(short) for r in short[i]])


def test_run_experiments_refuses_to_write_lost_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def lossy(indexed: list[tuple[int, Task]], workers: int, progress: Any) -> dict[int, Any]:
        return {i: [{}] * len(t.agents) for i, t in indexed[1:]}  # one task silently dropped

    monkeypatch.setattr(run, "_execute", lossy)
    with pytest.raises(RuntimeError, match="missing"):
        run.run_experiments(make_config(), tmp_path, workers=1, progress=False)
    assert not (tmp_path / run.RAW_FILE).exists()


def test_legend_handles_follow_the_marks() -> None:
    """Regression: unconnected series get marker-only legend handles; hollow key on demand."""
    connected = plots._legend_handles(["andor", "dstar_lite"])
    assert all(h.get_linestyle() not in ("None", "none") for h in connected)
    loose = plots._legend_handles(["andor", "dstar_lite"], connect=False, hollow=True)
    assert [h.get_linestyle() for h in loose] == ["None"] * 3
    assert "hollow" in str(loose[-1].get_label())


def test_geometric_mean_interval_stays_positive_on_log_axes() -> None:
    """Regression: a skewed timing sample gives a t-interval below 0; on the log axis the
    planning-time chart uses the back-transformed log-scale interval instead."""
    values = [1.0, 1.0, 1.0, 1.0, 100.0]
    assert stats.mean_ci(values)[1] < 0
    rows = [synth_row(agent="andor", planning_ms=v, replicate=i, seed=i)
            for i, v in enumerate(values)]
    (point,) = plots._mean_points(rows, "rows", "planning_ms", successful_only=False,
                                  geometric=True)["andor"]
    assert point.y == pytest.approx(math.exp(np.mean(np.log(values))))
    assert 0 < point.lo < point.y < point.hi


def test_draw_agents_puts_the_proposed_agent_on_top_and_reports_drawn() -> None:
    _fig, (ax,) = plots._new_figure(1, sharey=False)
    series = {
        "andor": [plots._Point(0.1, 100.0, 90.0, 100.0, 20, 1.0)],
        "astar_fixed": [plots._Point(0.1, 0.0, 0.0, 1.0, 20, 0.0)],
        "vi_expected": [plots._Point(0.1, 100.0, 90.0, 100.0, 20, 1.0)],
        "dstar_lite": [],
    }
    agents = ["andor", "astar_fixed", "dstar_lite", "vi_expected"]
    drawn = plots._draw_agents(ax, series, agents, [0.1],
                               hollow_below=plots.LOW_SUCCESS_FRACTION)
    assert drawn.agents == {"andor", "astar_fixed", "vi_expected"}
    assert drawn.hollow  # astar_fixed succeeded in 0 % of its missions
    assert ax.lines[-1].get_color() == plots.AGENT_STYLE["andor"].colour


def test_metric_charts_show_one_grid_size() -> None:
    block = [synth_row(rows=s, cols=s, replicate=i, seed=i) for s in (10, 15, 20)
             for i in range(3)]
    rows, note = plots._metric_rows(block)
    assert {r["rows"] for r in rows} == {15}
    assert "15×15" in note and "10×10" in note and "20×20" in note
    single, no_note = plots._metric_rows([r for r in block if r["rows"] == 10])
    assert len(single) == 3 and no_note == ""
