"""Extension experiment harness (docs/plan.md §3.14): config, seeding, runs, claims, plots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from rescue.config import (
    EXT_CLAIM_BUMP_MARGIN,
    EXT_CLAIM_NOISE,
    EXT_CLAIM_SUCCESS_TARGET,
    EXT_CLAIM_TOUR_RATIO_MAX,
)
from rescue.experiments import plots as core_plots
from rescue.experiments.hypotheses import VERDICT_FAIL, VERDICT_PASS, VERDICT_WARN
from rescue.experiments.spec import task_seed
from rescue.ext import claims, experiments
from rescue.ext import plots as ext_plots
from rescue.ext.experiments import (
    RAW_COLUMNS,
    ExtConfig,
    config_from_dict,
    expand_tasks,
    load_ext_config,
    read_raw,
    run_ext_experiments,
)

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = ["random_rubble", "collapsed_building", "flooded_zone", "maze_corridors"]
TIMING = ("planning_ms", "elapsed_ms")


def tiny_config(n_maps: int = 2) -> ExtConfig:
    """Three small blocks on 8×8 maps (fast enough for workers=1 and 2). The multi and noisy
    blocks share ``collapsed_building`` at p = 0.2, so multi k = 1 and noisy ε = 0 are the same
    missions (seed chain check)."""
    return config_from_dict({
        "name": "tiny", "base_seed": 11, "n_maps": n_maps,
        "agents": ["search_andor", "search_astar"],
        "blocks": [
            {"name": "search", "scenarios": ["random_rubble", "maze_corridors"], "sizes": [8],
             "slip_p": [0.2], "survivors_known": [True, False]},
            {"name": "multi", "scenarios": ["collapsed_building"], "sizes": [8],
             "slip_p": [0.2], "k": [1, 3], "agents": ["search_andor"]},
            {"name": "noisy", "scenarios": ["random_rubble", "collapsed_building"],
             "sizes": [8], "slip_p": [0.2], "false_negative": [0.0, 0.1]},
        ],
    })


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    out = tmp_path_factory.mktemp("ext_tiny")
    raw = run_ext_experiments(tiny_config(), out, workers=1, progress=False)
    return out, read_raw(raw)


def _strip_timing(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: v for k, v in row.items() if k not in TIMING} for row in csv.DictReader(fh)]


# --- config ----------------------------------------------------------------------------------
def test_full_and_quick_configs_parse() -> None:
    full = load_ext_config(REPO / "experiments" / "ext.yaml")
    quick = load_ext_config(REPO / "experiments" / "ext_quick.yaml")
    for cfg in (full, quick):
        # Round 2 appended the noisy_fp block (tests/test_ext_fp.py checks it).
        assert [b.name for b in cfg.blocks] == ["search", "multi", "noisy", "noisy_fp"]
        search, multi, noisy = cfg.blocks[:3]
        assert list(search.scenarios) == SCENARIOS
        assert search.slip_p == (0.1, 0.3) and search.survivors_known == (True, False)
        assert cfg.block_agents(search) == ("search_andor", "search_astar")
        assert multi.k == (1, 2, 3, 4) and cfg.block_agents(multi) == ("search_andor",)
        assert multi.slip_p == (0.2,) and multi.survivors_known == (True,)
        assert noisy.false_negative == (0.0, 0.05, 0.1, 0.2) and noisy.survivors_known == (True,)
        assert cfg.block_agents(noisy) == ("search_andor", "search_astar")
    assert full.blocks[0].sizes == (15,) and quick.blocks[0].sizes == (10,)
    assert full.block_n_maps(full.blocks[0]) == 200 and full.block_n_maps(full.blocks[1]) == 100
    assert all(quick.block_n_maps(b) == 10 for b in quick.blocks)
    assert len(expand_tasks(full)) == 4 * 2 * 2 * 200 + 4 * 4 * 100 + 4 * 4 * 200 + 4 * 4 * 200
    assert len(expand_tasks(quick)) == 4 * 2 * 2 * 10 + 4 * 4 * 10 + 4 * 4 * 10 + 4 * 4 * 10


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"bogus": 1}, "unknown keys"),
        ({"agents": ["andor"]}, "unknown agents"),
        ({"blocks": [{"name": "b", "scenarios": ["nowhere"], "sizes": [8], "slip_p": [0.1]}]},
         "unknown scenarios"),
        ({"blocks": [{"name": "b", "scenarios": ["random_rubble"], "sizes": [8],
                      "slip_p": [0.1], "survivors_known": ["yes"]}]}, "true/false"),
        ({"blocks": [{"name": "b", "scenarios": ["random_rubble"], "sizes": [8],
                      "slip_p": [0.1], "k": [0]}]}, "k must be"),
        ({"blocks": [{"name": "b", "scenarios": ["random_rubble"], "sizes": [8],
                      "slip_p": [0.1], "false_negative": [1.5]}]}, "false_negative"),
        ({"blocks": [{"name": "b", "scenarios": ["random_rubble"], "sizes": [8],
                      "slip_p": [0.1], "extra": 1}]}, "unknown keys"),
    ],
)
def test_bad_configs_are_rejected(patch: dict[str, Any], message: str) -> None:
    data = {"name": "x", "base_seed": 1, "n_maps": 1, "agents": ["search_andor"],
            "blocks": [{"name": "b", "scenarios": ["random_rubble"], "sizes": [8],
                        "slip_p": [0.1]}]}
    data.update(patch)
    with pytest.raises(ValueError, match=message):
        config_from_dict(data)


def test_load_reports_the_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nbase_seed: 1\nn_maps: 1\nagents: [nobody]\nblocks: []\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        load_ext_config(path)


# --- seeding ---------------------------------------------------------------------------------
def test_seeds_are_paired_across_factors_and_blocks() -> None:
    cfg = load_ext_config(REPO / "experiments" / "ext_quick.yaml")
    tasks = expand_tasks(cfg)
    by_map: dict[tuple[str, int, int], set[int]] = {}
    for t in tasks:
        assert t.seed == task_seed(cfg.base_seed, t.scenario, t.rows, t.cols, t.replicate)
        by_map.setdefault((t.scenario, t.rows, t.replicate), set()).add(t.seed)
    # one seed per (scenario, size, replicate): known/unknown, every k, every eps, every block
    assert all(len(seeds) == 1 for seeds in by_map.values())
    for block in cfg.blocks:
        own = [t for t in tasks if t.experiment == block.name]
        conditions = {(t.slip_p, t.k, t.survivors_known, t.false_negative, t.false_positive)
                      for t in own}
        assert len(conditions) == (len(block.slip_p) * len(block.k)
                                   * len(block.survivors_known) * len(block.false_negative)
                                   * len(block.false_positive))
        for cond in conditions:
            seeds = [t.seed for t in own
                     if (t.slip_p, t.k, t.survivors_known, t.false_negative,
                         t.false_positive) == cond]
            assert len(seeds) == len(set(seeds)) == len(block.scenarios) * cfg.block_n_maps(block)


# --- running ---------------------------------------------------------------------------------
def test_tiny_run_is_identical_for_one_and_two_workers(
    tiny_run: tuple[Path, list[dict[str, Any]]], tmp_path: Path
) -> None:
    out1, rows = tiny_run
    raw2 = run_ext_experiments(tiny_config(), tmp_path, workers=2, progress=False)
    assert _strip_timing(out1 / "raw.csv") == _strip_timing(raw2)
    header = (out1 / "raw.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(RAW_COLUMNS)
    # rows: task order, then the block's agents in config order
    tasks = expand_tasks(tiny_config())
    expected = [(t.experiment, t.scenario, t.k, t.survivors_known, t.false_negative,
                 t.replicate, a) for t in tasks for a in t.agents]
    got = [(r["experiment"], r["scenario"], r["k"], r["survivors_known"], r["false_negative"],
            r["replicate"], r["agent"]) for r in rows]
    assert got == expected
    meta = json.loads((out1 / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["n_runs"] == len(rows) and meta["workers"] == 1


def test_raw_rows_are_typed_and_consistent(tiny_run: tuple[Path, list[dict[str, Any]]]) -> None:
    _, rows = tiny_run
    for r in rows:
        assert isinstance(r["success"], bool) and isinstance(r["survivors_known"], bool)
        assert isinstance(r["steps"], int) and isinstance(r["slip_p"], float)
        assert r["rescued"] <= r["k"]
        assert r["success"] == (r["rescued"] == r["k"])
        assert (r["failure_reason"] == "") == r["success"]
        if r["survivors_known"]:
            assert r["first_detection_step"] == 0.0
        if r["false_negative"] == 0.0:
            assert r["bumps"] == 0
        if r["agent"] == "search_andor" and r["false_negative"] == 0.0:
            assert r["unplanned_states"] == 0
        if r["k"] <= 2 and not math.isnan(r["optimal_tour"]):
            assert r["greedy_tour"] == r["optimal_tour"]
    # agents on one task share the map: identical oracle tours
    by_task: dict[tuple[Any, ...], set[tuple[float, float]]] = {}
    for r in rows:
        key = tuple(r[k] for k in experiments.EXT_TASK_KEYS)
        by_task.setdefault(key, set()).add((r["greedy_tour"], r["optimal_tour"]))
    assert all(len(v) == 1 for v in by_task.values())


#: Columns that must agree between two runs of the same mission (timing aside).
MISSION_COLUMNS = ("success", "steps", "slips", "skids", "bumps", "blocked_moves", "replans",
                   "plans_made", "cells_explored", "nodes_expanded", "unplanned_states",
                   "oracle_path", "greedy_tour", "optimal_tour")


def test_task_seed_reaches_map_world_and_sensor(
    tiny_run: tuple[Path, list[dict[str, Any]]]
) -> None:
    """Regression (seed chain): the task seed, not the whole task, drives the map, the World
    and the noise. Multi k = 1 and noisy ε = 0 on one map are the same mission, field for field;
    the oracle tours of one map agree across blocks and location knowledge."""
    _, rows = tiny_run

    def own(block: str, **cond: Any) -> dict[int, dict[str, Any]]:
        return {r["replicate"]: r for r in rows if r["experiment"] == block
                and r["scenario"] == "collapsed_building" and r["agent"] == "search_andor"
                and all(r[k] == v for k, v in cond.items())}

    multi, noisy = own("multi", k=1), own("noisy", false_negative=0.0)
    assert multi and multi.keys() == noisy.keys()
    for rep_ in multi:
        assert multi[rep_]["seed"] == noisy[rep_]["seed"]
        assert ({c: multi[rep_][c] for c in MISSION_COLUMNS}
                == {c: noisy[rep_][c] for c in MISSION_COLUMNS})
    oracle: dict[tuple[Any, ...], set[float]] = {}
    for r in rows:
        if r["k"] == 1:
            oracle.setdefault((r["scenario"], r["rows"], r["replicate"]), set()).add(
                r["oracle_path"])
    assert all(len(v) == 1 for v in oracle.values())
    search = [r for r in rows if r["experiment"] == "search"]
    tours: dict[tuple[Any, ...], set[tuple[float, float]]] = {}
    for r in search:
        tours.setdefault((r["scenario"], r["replicate"]), set()).add(
            (r["greedy_tour"], r["optimal_tour"]))
    assert all(len(v) == 1 for v in tours.values())
    assert {r["survivors_known"] for r in search} == {True, False}


def test_max_tasks_samples_whole_maps() -> None:
    """Regression: a --max-tasks sample keeps every condition of a sampled map (pairs intact)."""
    cfg = load_ext_config(REPO / "experiments" / "ext_quick.yaml")
    tasks = expand_tasks(cfg)

    def per_map(ts: list[Any]) -> dict[tuple[Any, ...], int]:
        out: dict[tuple[Any, ...], int] = {}
        for t in ts:
            key = (t.experiment, t.scenario, t.rows, t.replicate)
            out[key] = out.get(key, 0) + 1
        return out

    full = per_map(tasks)
    for limit in (1, 7, 60, 200):
        sample = experiments.subsample_maps(tasks, limit)
        counts = per_map(sample)
        assert counts and all(counts[m] == full[m] for m in counts)
        assert len(sample) <= max(limit, max(full.values()))
        assert sample == [t for t in tasks if t in sample]  # order kept
    sample = experiments.subsample_maps(tasks, 200)
    assert {t.experiment for t in sample} == {"search", "multi", "noisy", "noisy_fp"}
    assert len(per_map(sample)) == len(per_map(experiments.subsample_maps(tasks, 200)))
    assert experiments.subsample_maps(tasks, None) == tasks
    with pytest.raises(ValueError, match="max_tasks"):
        experiments.subsample_maps(tasks, 0)


def test_row_loss_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = experiments._execute

    def lossy(indexed: Any, workers: int, progress: Any) -> dict[int, Any]:
        results = real(indexed, workers, progress)
        results.pop(min(results))
        return results

    monkeypatch.setattr(experiments, "_execute", lossy)
    with pytest.raises(RuntimeError, match="missing"):
        run_ext_experiments(tiny_config(1), tmp_path, workers=1, progress=False, max_tasks=3)
    assert not (tmp_path / "raw.csv").exists()


def test_summary_has_pooled_rows(tiny_run: tuple[Path, list[dict[str, Any]]],
                                 tmp_path: Path) -> None:
    _, rows = tiny_run
    summary = experiments.summarize(rows)
    pooled = [s for s in summary if s["scenario"] == experiments.ALL_SCENARIOS]
    assert pooled and all(s["n"] >= 1 for s in summary)
    search_pooled = [s for s in pooled if s["experiment"] == "search"]
    assert sum(s["n"] for s in search_pooled) == sum(r["experiment"] == "search" for r in rows)
    path = experiments.write_summary(rows, tmp_path / "summary.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == experiments.SUMMARY_COLUMNS
        assert len(list(reader)) == len(summary)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "name: cli\nbase_seed: 3\nn_maps: 1\nagents: [search_andor, search_astar]\nblocks:\n"
        "  - {name: search, scenarios: [random_rubble], sizes: [8], slip_p: [0.1],\n"
        "     survivors_known: [true, false]}\n", encoding="utf-8")
    out = tmp_path / "out"
    code = experiments.main(["--config", str(config), "--workers", "1", "--out", str(out),
                             "--no-plots", "--quiet"])
    assert code == 0
    for name in ("raw.csv", "summary.csv", "EXTENSIONS.md", "run_meta.json"):
        assert (out / name).is_file()
    assert not (out / "plots").exists()
    text = capsys.readouterr().out
    assert "X1" in text and "X5" in text
    assert experiments.main(["--workers", "0"]) == 2
    assert experiments.main(["--config", str(tmp_path / "missing.yaml")]) == 2


# --- claims ----------------------------------------------------------------------------------
def _row(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "experiment": "search", "scenario": "random_rubble", "rows": 10, "cols": 10,
        "slip_p": 0.1, "k": 1, "survivors_known": True, "false_negative": 0.0, "replicate": 0,
        "seed": 1, "agent": "search_andor", "success": True, "failure_reason": "", "steps": 10,
        "rescued": 1, "first_detection_step": 0.0, "bumps": 0, "unplanned_states": 0,
        "greedy_tour": 10.0, "optimal_tour": 10.0, "false_positive": 0.0, "fallback_replans": 0,
    }
    base.update(fields)
    return base


def _search_rows(n: int, n_fail: int, extra_unknown: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        ok = i >= n_fail
        rows.append(_row(replicate=i, seed=i, steps=20 + i % 3))
        rows.append(_row(replicate=i, seed=i, survivors_known=False, success=ok,
                         failure_reason="" if ok else "budget",
                         steps=20 + i % 3 + extra_unknown + (1 if i % 2 else -1)))
    return rows


def test_x1_rule() -> None:
    n = 100
    assert claims.claim_x1(_search_rows(n, 0, 5)).verdict == VERDICT_PASS
    fails = round(n * (1 - EXT_CLAIM_SUCCESS_TARGET)) + 2  # just below the target
    assert claims.claim_x1(_search_rows(n, fails, 5)).verdict == VERDICT_WARN
    assert claims.claim_x1(_search_rows(n, 30, 5)).verdict == VERDICT_FAIL
    assert claims.claim_x1([]).verdict == VERDICT_WARN


def test_x2_uses_the_map_as_unit_and_both_successes() -> None:
    """Regression: two slip p per map → n_units = maps, n_pairs = 2 × maps; a pair whose known
    twin failed is dropped."""
    n = 12
    rows = []
    for p in (0.1, 0.3):
        rows += [dict(r, slip_p=p) for r in _search_rows(n, 0, 6)]
    pd, ratio = claims._location_diff(rows, "search_andor")
    assert pd.n_units == n and pd.n_pairs == 2 * n
    assert ratio > 1.0
    for r in rows:
        if r["survivors_known"] and r["replicate"] == 0 and r["slip_p"] == 0.3:
            r.update(success=False, failure_reason="budget")
    pd, _ = claims._location_diff(rows, "search_andor")
    assert pd.n_units == n and pd.n_pairs == 2 * n - 1
    assert claims.claim_x2(rows).verdict == VERDICT_PASS


def test_x2_rule() -> None:
    passing = claims.claim_x2(_search_rows(30, 0, 8))
    assert passing.verdict == VERDICT_PASS and "ratio" in passing.measured
    assert claims.claim_x2(_search_rows(30, 0, -8)).verdict == VERDICT_FAIL
    assert claims.claim_x2(_search_rows(30, 0, 0)).verdict == VERDICT_WARN
    known_only = [r for r in _search_rows(5, 0, 1) if r["survivors_known"]]
    assert claims.claim_x2(known_only).verdict == VERDICT_WARN


def _multi_rows(ratios: dict[int, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for k, values in ratios.items():
        for i, ratio in enumerate(values):
            rows.append(_row(experiment="multi", k=k, replicate=i, seed=i, rescued=k,
                             optimal_tour=20.0, greedy_tour=20.0 * ratio, slip_p=0.2))
    return rows


def test_x3_rule() -> None:
    limit = EXT_CLAIM_TOUR_RATIO_MAX
    ok = {1: [1.0] * 10, 3: [1.0, 1.02, 1.0, 1.04] * 5}
    assert claims.claim_x3(_multi_rows(ok)).verdict == VERDICT_PASS
    bad = {1: [1.0] * 10, 4: [limit + 0.1, limit + 0.12] * 10}
    assert claims.claim_x3(_multi_rows(bad)).verdict == VERDICT_FAIL
    unsure = {3: [1.0, limit + 0.15] * 3}
    assert claims.claim_x3(_multi_rows(unsure)).verdict == VERDICT_WARN
    assert claims.claim_x3([]).verdict == VERDICT_WARN


def test_x3_a_k_without_ci_is_inconclusive() -> None:
    """Regression: a k with a single map (no CI) used to pass on its point estimate."""
    single = {1: [1.0] * 10, 3: [1.0]}
    assert claims.claim_x3(_multi_rows(single)).verdict == VERDICT_WARN


def test_x3_and_plots_count_each_layout_once() -> None:
    """Regression: the same maps at two slip p are the same layouts, not twice the maps."""
    ratios = {3: [1.0, 1.02, 1.12, 1.0, 1.08, 1.0]}
    one_p = _multi_rows(ratios)
    two_p = one_p + [dict(r, slip_p=0.3, agent="search_andor") for r in one_p]
    assert len(experiments.unique_layouts(two_p)) == len(one_p)
    assert claims.claim_x3(two_p).measured == claims.claim_x3(one_p).measured
    assert ext_plots._ratio_points(two_p) == ext_plots._ratio_points(one_p)
    assert ext_plots._optimal_per_survivor(two_p) == ext_plots._optimal_per_survivor(one_p)


def _noisy_rows(bumps: dict[float, tuple[int, int]], fails_at_claim: int = 0,
                n: int = 40) -> list[dict[str, Any]]:
    """``bumps[eps] = (andor bumps, astar bumps)`` per mission."""
    rows = []
    for eps, (b_andor, b_astar) in bumps.items():
        for i in range(n):
            for agent, b in (("search_andor", b_andor), ("search_astar", b_astar)):
                ok = not (agent == "search_andor" and eps == EXT_CLAIM_NOISE
                          and i < fails_at_claim)
                rows.append(_row(experiment="noisy", slip_p=0.2, false_negative=eps,
                                 replicate=i, seed=i, agent=agent, bumps=b + i % 2,
                                 success=ok, failure_reason="" if ok else "budget"))
    return rows


def test_x4_rule() -> None:
    clean = {0.0: (0, 0), EXT_CLAIM_NOISE: (1, 1)}
    rows = _noisy_rows(clean)
    for r in rows:  # the clean condition must stay collision-free
        if r["false_negative"] == 0.0:
            r["bumps"] = 0
    assert claims.claim_x4(rows).verdict == VERDICT_PASS
    no_bumps = [dict(r, bumps=0) for r in rows]
    assert claims.claim_x4(no_bumps).verdict == VERDICT_FAIL
    dirty = [dict(r, bumps=1) for r in rows]  # bumps at eps = 0 break the core invariant
    assert claims.claim_x4(dirty).verdict == VERDICT_FAIL
    no_clean = [r for r in rows if r["false_negative"] > 0.0]
    assert claims.claim_x4(no_clean).verdict == VERDICT_WARN
    failing = _noisy_rows(clean, fails_at_claim=20)
    for r in failing:
        if r["false_negative"] == 0.0:
            r["bumps"] = 0
    assert claims.claim_x4(failing).verdict == VERDICT_FAIL
    assert claims.claim_x4(_noisy_rows({0.0: (0, 0)})).verdict == VERDICT_WARN


def test_x5_rule() -> None:
    assert claims.claim_x5(_noisy_rows({0.1: (1, 1), 0.2: (2, 2)})).verdict == VERDICT_PASS
    assert claims.claim_x5(_noisy_rows({0.1: (2, 1)})).verdict == VERDICT_FAIL
    noisy = _noisy_rows({0.1: (1, 1)})
    for r in noisy:  # small mean difference, wide spread → inconclusive
        if r["agent"] == "search_andor":
            r["bumps"] = 0 if r["replicate"] % 2 else 4
    pd_hi = claims.paired_difference(noisy, "bumps", "search_andor", "search_astar",
                                     keys=experiments.EXT_TASK_KEYS, require_success=False).hi
    assert pd_hi > EXT_CLAIM_BUMP_MARGIN
    assert claims.claim_x5(noisy).verdict == VERDICT_WARN
    assert claims.claim_x5(_noisy_rows({0.0: (0, 0)})).verdict == VERDICT_WARN


def test_x5_small_but_significant_excess_fails() -> None:
    """A constant excess below the margin is still significantly > 0: ❌ beats ✅."""
    rows = _noisy_rows({0.1: (1, 1)})
    small = EXT_CLAIM_BUMP_MARGIN / 2
    for r in rows:
        if r["agent"] == "search_andor":
            r["bumps"] = r["bumps"] + small
    pd = claims._bump_diff(rows)
    assert pd.lo > 0.0 and pd.hi <= EXT_CLAIM_BUMP_MARGIN
    assert claims.claim_x5(rows).verdict == VERDICT_FAIL


def test_x5_pooling_cannot_hide_one_noise_level() -> None:
    """Regression: search_andor better at one eps and worse at another pools to 0 per map; the
    per-eps rule must still refute it."""
    rows = _noisy_rows({0.1: (0, 1), 0.2: (2, 1)})
    pooled = claims._bump_diff(rows)
    assert pooled.mean_diff == 0.0 and pooled.hi <= EXT_CLAIM_BUMP_MARGIN
    hypothesis = claims.claim_x5(rows)
    assert hypothesis.verdict == VERDICT_FAIL
    assert "ε = 0.2" in hypothesis.measured
    assert "non-inferiority" in hypothesis.claim


def test_extensions_report(tiny_run: tuple[Path, list[dict[str, Any]]], tmp_path: Path) -> None:
    out, rows = tiny_run
    (tmp_path / "run_meta.json").write_text((out / "run_meta.json").read_text(encoding="utf-8"),
                                            encoding="utf-8")
    path = claims.write_extensions(rows, tmp_path / "EXTENSIONS.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Extension claims — `tiny`")
    for label in ("X1", "X2", "X3", "X4", "X5"):
        assert f"| {label} |" in text and f"### {label}." in text
    assert text.count("Rule") >= 5
    bare = claims.write_extensions([], tmp_path / "bare" / "EXTENSIONS.md")
    assert bare.read_text(encoding="utf-8").count(VERDICT_WARN) >= 5


# --- plots -----------------------------------------------------------------------------------
def test_search_plot_averages_per_map_first() -> None:
    """Regression: one point per map (pooled over p), so n counts maps, not missions."""
    rows = []
    for p in (0.1, 0.3):
        rows += [dict(r, slip_p=p, location="unknown" if not r["survivors_known"] else "known")
                 for r in _search_rows(6, 0, 4)]
    per_map = ext_plots._per_map_steps(rows)
    assert len(per_map) == 2 * 6  # known + unknown, 6 maps, one agent
    by_key = {(r["location"], r["replicate"]): r["steps"] for r in per_map}
    for (loc, rep_), steps in by_key.items():
        own = [r["steps"] for r in rows
               if r["location"] == loc and r["replicate"] == rep_ and r["success"]]
        assert steps == sum(own) / len(own)


def test_plots_are_written(tiny_run: tuple[Path, list[dict[str, Any]]], tmp_path: Path) -> None:
    _, rows = tiny_run
    before = dict(core_plots.AGENT_STYLE)
    paths = ext_plots.make_ext_plots(rows, tmp_path / "plots")
    assert sorted(p.name for p in paths) == ["multi_tours.png", "noisy.png", "search_steps.png"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    assert before == core_plots.AGENT_STYLE  # the extension styles were only borrowed
    only_noisy = [r for r in rows if r["experiment"] == "noisy"]
    assert [p.name for p in ext_plots.make_ext_plots(only_noisy, tmp_path / "x")] == ["noisy.png"]
    assert ext_plots.make_ext_plots([], tmp_path / "y") == []
