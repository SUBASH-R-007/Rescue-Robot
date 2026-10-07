"""LAO* experiment analysis (brief §8.1, docs/plan.md §3.15.1): ``LAO.md`` from ``raw.csv``.

``python -m rescue.experiments.lao_report results/lao`` reads the ``raw.csv`` (and
``run_meta.json``, if present) of an ``experiments/lao.yaml`` run and writes ``LAO.md`` next to
them. The harness's ``HYPOTHESES.md`` covers the core claims H1–H7 only; this report answers
the LAO\\* experiment's own questions, descriptively (no verdicts):

1. **Plan quality.** Paired ``lao_expected − vi_expected`` and ``lao_expected − andor`` steps
   (both successful; the map is the unit, as in every paired test of the harness), success
   rates, and the share of LAO\\*/VI mission pairs identical on every non-timing metric
   (``nodes_expanded`` excepted: it counts different things). Both planners compute the same
   optimum, but they may break exact ties differently (values agree to ~1e-7, the tie
   tolerance), so identical missions are an observation, not a guarantee.
2. **Planning effort.** Mean planning time per plan (``planning_ms / plans_made``) of each
   agent by scenario and size with 95 % CIs, the LAO\\*/VI ratio of the means, and the
   paired per-mission ratio. These are wall-clock times measured with ``workers`` processes
   running at once (``run_meta.json``); run with ``--workers 1`` for unloaded times.
3. **LAO\\* expansions.** States expanded per plan, also as a share of the true map's free
   cells (the planning belief treats unknown cells as open, so early plans may expand more
   cells than the true map has free; the share can exceed 100 %).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rescue.agents.contingency import ContingencyAgent
from rescue.agents.lao_expected import LaoExpectedAgent
from rescue.agents.vi_expected import ValueIterationAgent
from rescue.cli_utils import ensure_utf8_stdout
from rescue.experiments.run import METRIC_COLUMNS, TIMING_COLUMNS
from rescue.experiments.stats import MAP_KEYS, TASK_KEYS, mean_ci, paired_difference, read_raw

#: Output file, written next to ``raw.csv``.
LAO_REPORT_FILE: str = "LAO.md"
#: Input files of a harness run.
RAW_FILE: str = "raw.csv"
RUN_META_FILE: str = "run_meta.json"
#: Default run directory (``experiments/lao.yaml`` header: ``--out results/lao``).
DEFAULT_LAO_DIR: Path = Path("results") / "lao"

LAO = LaoExpectedAgent.name
VI = ValueIterationAgent.name
ANDOR = ContingencyAgent.name
#: Agents of the planning-effort table, in column order.
EFFORT_AGENTS: tuple[str, ...] = (ANDOR, VI, LAO)
#: Metrics two missions must share to count as identical (the agent id, timing and the
#: planner-specific ``nodes_expanded`` excluded).
IDENTITY_METRICS: tuple[str, ...] = (
    *(m for m in METRIC_COLUMNS
      if m not in TIMING_COLUMNS and m not in ("agent", "nodes_expanded")),
    "plans_made",
)
#: Decimal places of the times (ms), ratios and shares in the tables.
MS_DECIMALS: int = 1
RATIO_DECIMALS: int = 2
STEP_DECIMALS: int = 3
PERCENT: float = 100.0
#: Label of the pooled (all scenarios) rows.
ALL_SCENARIOS: str = "all"


# --- per-mission quantities -------------------------------------------------------------------
def ms_per_plan(row: dict[str, Any]) -> float:
    """Mean planning time of one mission's plans, ms (NaN without a plan)."""
    plans = int(row["plans_made"])
    return float(row["planning_ms"]) / plans if plans > 0 else math.nan


def expanded_per_plan(row: dict[str, Any]) -> float:
    """Mean ``nodes_expanded`` per plan of one mission (NaN without a plan)."""
    plans = int(row["plans_made"])
    return float(row["nodes_expanded"]) / plans if plans > 0 else math.nan


def free_cells(scenario: str, rows: int, cols: int, seed: int) -> int:
    """Free cells of the task's true map (regenerated from its seed)."""
    from rescue import mapgen

    return int(np.count_nonzero(mapgen.generate(scenario, rows, cols, seed).passable_mask()))


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[k] for k in TASK_KEYS)


def identical_share(rows: Sequence[dict[str, Any]], agent_a: str,
                    agent_b: str) -> tuple[int, int]:
    """(paired missions identical on :data:`IDENTITY_METRICS`, all paired missions)."""
    side_b = {_key(r): r for r in rows if r["agent"] == agent_b}
    same = total = 0
    for ra in (r for r in rows if r["agent"] == agent_a):
        rb = side_b.get(_key(ra))
        if rb is None:
            continue
        total += 1
        same += all(ra[m] == rb[m] or (_nan(ra[m]) and _nan(rb[m])) for m in IDENTITY_METRICS)
    return same, total


def _nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def paired_ratios(rows: Sequence[dict[str, Any]], agent_a: str, agent_b: str,
                  value: Callable[[dict[str, Any]], float]) -> list[float]:
    """Per-mission ``value(agent_a) / value(agent_b)`` over paired missions (finite, b > 0)."""
    side_b = {_key(r): r for r in rows if r["agent"] == agent_b}
    out = []
    for ra in (r for r in rows if r["agent"] == agent_a):
        rb = side_b.get(_key(ra))
        if rb is None:
            continue
        a, b = value(ra), value(rb)
        if math.isfinite(a) and math.isfinite(b) and b > 0.0:
            out.append(a / b)
    return out


# --- formatting -------------------------------------------------------------------------------
def _ci(mean: float, lo: float, hi: float, decimals: int) -> str:
    if math.isnan(mean):
        return "–"
    if math.isnan(lo):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return lines


def _groups(rows: Sequence[dict[str, Any]]) -> list[tuple[str, int, list[dict[str, Any]]]]:
    """(scenario or ``ALL_SCENARIOS``, size, rows) per scenario × size, then pooled per size."""
    scenarios = sorted({r["scenario"] for r in rows})
    sizes = sorted({int(r["rows"]) for r in rows})
    out = []
    for scenario in (*scenarios, ALL_SCENARIOS):
        for size in sizes:
            sub = [r for r in rows if int(r["rows"]) == size
                   and (scenario == ALL_SCENARIOS or r["scenario"] == scenario)]
            if sub:
                out.append((scenario, size, sub))
    return out


# --- the report -------------------------------------------------------------------------------
def _quality(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = ["## 1. Plan quality", ""]
    body = []
    for other in (VI, ANDOR):
        pd = paired_difference(rows, "steps", LAO, other, cluster_keys=MAP_KEYS)
        body.append([f"`{LAO}` − `{other}`", _ci(pd[0], pd[1], pd[2], STEP_DECIMALS),
                     str(pd.n_units), str(pd.n_nonzero)])
    lines += _table(["steps (both successful)", "mean diff [95 % CI]", "maps", "maps ≠ 0"], body)
    lines.append("")
    rates = []
    for agent in EFFORT_AGENTS:
        runs = [r for r in rows if r["agent"] == agent]
        ok = sum(1 for r in runs if r["success"])
        rates.append(f"`{agent}` {ok}/{len(runs)}")
    same, total = identical_share(rows, LAO, VI)
    lines += [
        "Success: " + ", ".join(rates) + ".",
        "",
        f"Identical `{LAO}` / `{VI}` missions (every metric except timing and "
        f"`nodes_expanded`): {same}/{total}. Both planners compute the same optimum but may "
        "break exact ties differently, so this is an observation, not a guarantee.",
        "",
    ]
    return lines


def _effort(rows: Sequence[dict[str, Any]], workers: Any) -> list[str]:
    lines = [
        "## 2. Planning time per plan (ms)",
        "",
        f"Wall-clock means with 95 % CIs, measured with {workers} worker process(es) running "
        "at once (`run_meta.json`); `LAO*/VI` = ratio of the means, `paired` = mean of the "
        "per-mission ratios.",
        "",
    ]
    body = []
    for scenario, size, sub in _groups(rows):
        cells = [scenario, f"{size}×{size}"]
        means = {}
        for agent in EFFORT_AGENTS:
            m = mean_ci(ms_per_plan(r) for r in sub if r["agent"] == agent)
            means[agent] = m[0]
            cells.append(_ci(*m, MS_DECIMALS))
        ratio = means[LAO] / means[VI] if means[VI] > 0 else math.nan
        paired = mean_ci(paired_ratios(sub, LAO, VI, ms_per_plan))
        cells += [f"{ratio:.{RATIO_DECIMALS}f}", _ci(*paired, RATIO_DECIMALS)]
        body.append(cells)
    header = ["scenario", "size", *(f"`{a}`" for a in EFFORT_AGENTS), "LAO*/VI", "paired"]
    return lines + _table(header, body) + [""]


def _expansions(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"## 3. `{LAO}` states expanded per plan",
        "",
        "Share = expanded per plan / free cells of the true map (unknown cells are planned as "
        "open, so it can exceed 100 %).",
        "",
    ]
    lao_rows = [r for r in rows if r["agent"] == LAO]
    free = {_key(r): free_cells(r["scenario"], int(r["rows"]), int(r["cols"]), int(r["seed"]))
            for r in lao_rows}
    body = []
    for scenario, size, sub in _groups(lao_rows):
        expanded = mean_ci(expanded_per_plan(r) for r in sub)
        share = mean_ci(PERCENT * expanded_per_plan(r) / free[_key(r)] for r in sub)
        body.append([scenario, f"{size}×{size}", _ci(*expanded, MS_DECIMALS),
                     _ci(*share, MS_DECIMALS)])
    return lines + _table(["scenario", "size", "expanded / plan", "share of free cells (%)"],
                          body) + [""]


def lao_report(rows: Sequence[dict[str, Any]], meta: dict[str, Any] | None = None) -> str:
    """The Markdown of ``LAO.md`` for the typed ``raw.csv`` rows of a LAO\\* run."""
    meta = meta or {}
    agents = {r["agent"] for r in rows}
    missing = [a for a in EFFORT_AGENTS if a not in agents]
    if missing:
        raise ValueError(f"raw.csv has no rows for {missing}; run experiments/lao.yaml")
    workers = meta.get("workers", "an unknown number of")
    lines = [
        "# LAO\\* experiment (plan §3.15.1)",
        "",
        f"Config `{meta.get('config_name', '?')}`, {meta.get('n_tasks', '?')} tasks, "
        f"{len(rows)} missions, workers {workers}, commit `{meta.get('git_commit', '?')}`. "
        "Descriptive only: no claim of the deck depends on these numbers.",
        "",
    ]
    lines += _quality(rows) + _effort(rows, workers) + _expansions(rows)
    return "\n".join(lines)


def write_lao_report(run_dir: str | Path) -> Path:
    """Write ``LAO.md`` for the run in ``run_dir``; returns its path."""
    run_dir = Path(run_dir)
    rows = read_raw(run_dir / RAW_FILE)
    meta_path = run_dir / RUN_META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    out = run_dir / LAO_REPORT_FILE
    out.write_text(lao_report(rows, meta), encoding="utf-8", newline="\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m rescue.experiments.lao_report [DIR]``; returns the exit code."""
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m rescue.experiments.lao_report",
                                     description="Write LAO.md for an experiments/lao.yaml run.")
    parser.add_argument("run_dir", nargs="?", default=str(DEFAULT_LAO_DIR),
                        help=f"run directory with raw.csv (default {DEFAULT_LAO_DIR.as_posix()})")
    args = parser.parse_args(argv)
    try:
        path = write_lao_report(args.run_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
