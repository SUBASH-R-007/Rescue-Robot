"""Statistics for the Monte Carlo harness: typed raw rows, intervals, paired tests, summaries.

Brief §6.6. ``read_raw`` turns ``results/raw.csv`` back into typed rows;
``wilson_interval`` (success rates) and ``mean_ci`` (Student t) give 95 % intervals;
``paired_difference`` compares two agents on identical tasks (common random numbers: same map,
same outcome stream) with a paired t-test and a Wilcoxon signed-rank test, optionally with the
map as the unit of analysis (``cluster_keys``: a map reused at several slip p values is one
independent unit, not several); ``summarize`` /
``write_summary`` produce ``results/summary.csv``.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from rescue.agents import ALL_AGENT_NAMES
from rescue.mapgen import SCENARIOS

#: Default confidence level of every interval in the harness.
DEFAULT_CONFIDENCE: float = 0.95

#: Columns that identify a task (one map + one outcome stream); paired comparisons join on them.
TASK_KEYS: tuple[str, ...] = (
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

#: Columns that identify one map (and its outcome stream) inside an experiment. Task seeds depend
#: only on (base seed, scenario, size, replicate), so every condition of a block reuses the map:
#: differences from one map at several slip p values are correlated, not independent.
MAP_KEYS: tuple[str, ...] = ("experiment", "scenario", "rows", "cols", "replicate", "seed")

#: How the Wilcoxon signed-rank test treats zero differences (see ``_wilcoxon_p``).
WILCOXON_ZERO_METHOD: str = "pratt"

#: Summary grouping: one summary row per (condition, agent).
GROUP_KEYS: tuple[str, ...] = (
    "experiment",
    "scenario",
    "rows",
    "slip_p",
    "sensor_radius",
    "replan_mode",
    "aftershock_q",
    "agent",
)

#: Metrics summarised as mean / lo / hi (Student-t 95 % CI) in ``summary.csv``.
SUMMARY_METRICS: tuple[str, ...] = (
    "steps",
    "wasted_moves",
    "replans",
    "slips",
    "skids",
    "slips_handled",
    "bumps",
    "unplanned_states",
    "cells_explored",
    "planning_ms",
    "nodes_expanded",
    "plan_size",
    "planner_fallbacks",
    "aftershock_events",
)

#: Metrics averaged over successful runs only (undefined or truncated on failure, D16).
SUCCESS_ONLY_METRICS: frozenset[str] = frozenset({"steps", "wasted_moves"})

#: Failure reasons counted per summary row (``MissionMetrics.failure_reason``).
FAILURE_REASONS: tuple[str, ...] = ("aborted", "budget", "unreachable")

#: Fixed column order of ``summary.csv``.
SUMMARY_COLUMNS: tuple[str, ...] = (
    *GROUP_KEYS,
    "n",
    "successes",
    "success_rate",
    "success_lo",
    "success_hi",
    *(f"n_{reason}" for reason in FAILURE_REASONS),
    *(
        col
        for metric in SUMMARY_METRICS
        for col in (
            (f"{metric}_mean", f"{metric}_lo", f"{metric}_hi")
            + (("steps_all_mean", "steps_all_lo", "steps_all_hi") if metric == "steps" else ())
        )
    ),
)

# --- raw.csv column types ---------------------------------------------------------------------
_STR_COLUMNS: frozenset[str] = frozenset(
    {"experiment", "scenario", "replan_mode", "agent", "failure_reason"}
)
_INT_COLUMNS: frozenset[str] = frozenset(
    {
        "rows", "cols", "sensor_radius", "aftershock_k", "replicate", "seed", "steps", "replans",
        "slips", "skids", "slips_handled", "bumps", "blocked_moves", "unplanned_states",
        "cells_explored", "nodes_expanded", "plan_size", "planner_fallbacks",
        "aftershock_events", "plans_made",
    }
)
_OPTIONAL_INT_COLUMNS: frozenset[str] = frozenset({"shortest_path"})
_FLOAT_COLUMNS: frozenset[str] = frozenset(
    {"slip_p", "aftershock_q", "wasted_moves", "planning_ms", "elapsed_ms"}
)
_BOOL_COLUMNS: frozenset[str] = frozenset({"success"})
_TRUE_TEXT: frozenset[str] = frozenset({"true", "1", "yes"})
_FALSE_TEXT: frozenset[str] = frozenset({"false", "0", "no"})
_FLOAT_FORMAT: str = ".6g"


def _parse_value(column: str, text: str) -> Any:
    """Typed value of one raw.csv cell (unknown columns: int, then float, else str)."""
    text = text.strip()
    if column in _STR_COLUMNS:
        return text
    if column in _BOOL_COLUMNS:
        low = text.lower()
        if low in _TRUE_TEXT:
            return True
        if low in _FALSE_TEXT:
            return False
        raise ValueError(f"bad boolean {text!r} in column {column!r}")
    if column in _OPTIONAL_INT_COLUMNS:
        return None if text in ("", "None") else int(float(text))
    if column in _INT_COLUMNS:
        return int(float(text))
    if column in _FLOAT_COLUMNS:
        return math.nan if text in ("", "None") else float(text)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def read_raw(path: str | Path) -> list[dict[str, Any]]:
    """Read ``raw.csv`` into typed rows (bools, ints, floats with NaN, strings)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [
            {col: _parse_value(col, text) for col, text in row.items()} for row in reader
        ]


# --- intervals --------------------------------------------------------------------------------
def _z(confidence: float) -> float:
    """Two-sided standard-normal quantile for ``confidence``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(sps.norm.ppf(0.5 + confidence / 2.0))


def wilson_interval(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (``(nan, nan)`` when ``n == 0``).

    Exact at the edges: ``lo == 0`` when ``successes == 0`` and ``hi == 1`` when
    ``successes == n``.
    """
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return (math.nan, math.nan)
    z = _z(confidence)
    phat = successes / n
    z2n = z * z / n
    centre = (phat + z2n / 2.0) / (1.0 + z2n)
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2n / (4.0 * n)) / (1.0 + z2n)
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


def _finite(values: Iterable[Any]) -> np.ndarray:
    """Values as a float array with NaN / None dropped."""
    arr = np.asarray([math.nan if v is None else float(v) for v in values], dtype=float)
    return arr[~np.isnan(arr)]


def mean_ci(
    values: Iterable[Any], confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` with a Student-t interval; NaNs dropped.

    No values → all NaN; one value → ``(mean, nan, nan)``; zero spread → ``(mean, mean, mean)``.
    """
    arr = _finite(values)
    n = arr.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(arr.mean())
    if n < 2:
        return (mean, math.nan, math.nan)
    _z(confidence)  # validates the level
    se = float(arr.std(ddof=1)) / math.sqrt(n)
    half = float(sps.t.ppf(0.5 + confidence / 2.0, n - 1)) * se
    return (mean, mean - half, mean + half)


# --- paired comparison ------------------------------------------------------------------------
class PairedDifference(tuple):
    """``(mean_diff, lo, hi, n_pairs, p_value)`` plus extra attributes.

    ``mean_diff`` = mean of ``agent_a − agent_b`` over the analysis units (the pairs, or the
    per-map means of the pairs with ``cluster_keys``); ``lo``/``hi`` its Student-t CI;
    ``n_pairs`` = number of paired runs; ``p_value`` = two-sided paired t-test over the units.
    Attributes: ``p_wilcoxon`` = two-sided Wilcoxon signed-rank test over the units
    (``zero_method='pratt'``: zero differences are ranked, then dropped; 1.0 when every
    difference is zero); ``n_units`` = number of independent units (= ``n_pairs`` without
    clustering); ``n_nonzero`` = units with a non-zero difference.
    """

    p_wilcoxon: float
    n_units: int
    n_nonzero: int

    def __new__(
        cls,
        mean_diff: float,
        lo: float,
        hi: float,
        n_pairs: int,
        p_value: float,
        p_wilcoxon: float = math.nan,
        n_units: int | None = None,
        n_nonzero: int = 0,
    ) -> PairedDifference:
        obj = super().__new__(cls, (mean_diff, lo, hi, n_pairs, p_value))
        obj.p_wilcoxon = p_wilcoxon
        obj.n_units = n_pairs if n_units is None else n_units
        obj.n_nonzero = n_nonzero
        return obj

    def __getnewargs__(self) -> tuple[Any, ...]:  # pickling / copying keep the attributes
        return (*self, self.p_wilcoxon, self.n_units, self.n_nonzero)

    @property
    def mean_diff(self) -> float:
        """Mean of ``agent_a − agent_b``."""
        return self[0]

    @property
    def lo(self) -> float:
        """Lower CI bound of the mean difference."""
        return self[1]

    @property
    def hi(self) -> float:
        """Upper CI bound of the mean difference."""
        return self[2]

    @property
    def n_pairs(self) -> int:
        """Number of paired tasks."""
        return self[3]

    @property
    def p_value(self) -> float:
        """Paired t-test p-value (two-sided)."""
        return self[4]

    def __repr__(self) -> str:
        return (
            f"PairedDifference(mean_diff={self[0]!r}, lo={self[1]!r}, hi={self[2]!r}, "
            f"n_pairs={self[3]!r}, p_value={self[4]!r}, p_wilcoxon={self.p_wilcoxon!r}, "
            f"n_units={self.n_units!r}, n_nonzero={self.n_nonzero!r})"
        )


def _paired_t_p(diffs: np.ndarray) -> float:
    """Two-sided paired t-test p-value of the differences (degenerate spreads handled)."""
    if diffs.size < 2:
        return math.nan
    if float(diffs.std(ddof=1)) == 0.0:
        return 1.0 if float(diffs.mean()) == 0.0 else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = sps.ttest_rel(diffs, np.zeros_like(diffs))
    return float(result.pvalue)


def _wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if every difference is zero.

    ``zero_method='pratt'``: zero differences take part in the ranking (then are dropped), so
    a sample that is mostly ties is not read as strong evidence; ``'wilcox'`` would discard
    the zeros first and answer a question about the non-zero pairs only.
    """
    if diffs.size < 1:
        return math.nan
    if not np.any(diffs != 0.0):
        return 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (RuntimeWarning, UserWarning))
            result = sps.wilcoxon(diffs, zero_method=WILCOXON_ZERO_METHOD)
    except ValueError:
        return math.nan
    return float(result.pvalue)


def paired_difference(
    rows: Sequence[dict[str, Any]],
    metric: str,
    agent_a: str,
    agent_b: str,
    keys: Sequence[str] = TASK_KEYS,
    *,
    require_success: bool | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    cluster_keys: Sequence[str] | None = None,
) -> PairedDifference:
    """Mean of ``metric(agent_a) − metric(agent_b)`` over tasks identical on ``keys``.

    A task contributes only when both agents ran it (and, with ``require_success``, both
    succeeded; the default requires success for :data:`SUCCESS_ONLY_METRICS`) and both values
    are finite. Returns a :class:`PairedDifference` 5-tuple ``(mean_diff, lo, hi, n_pairs,
    p_value)`` (paired t-test) whose ``p_wilcoxon`` attribute holds the Wilcoxon p-value.
    Duplicate rows for one agent on one key raise ``ValueError``.

    ``cluster_keys`` (e.g. :data:`MAP_KEYS`) makes the cluster the unit of analysis: the
    differences of each cluster are averaged first, and the mean, CI and both tests use the
    cluster means (``n_units`` clusters). Use it whenever the rows reuse one map at several
    conditions (common random numbers), since those differences are correlated.
    """
    if require_success is None:
        require_success = metric in SUCCESS_ONLY_METRICS
    keys = tuple(keys)
    by_agent: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {agent_a: {}, agent_b: {}}
    for row in rows:
        agent = row.get("agent")
        if agent not in by_agent:
            continue
        key = tuple(row[k] for k in keys)
        if key in by_agent[agent]:
            raise ValueError(f"duplicate {agent} row for task {dict(zip(keys, key, strict=True))}")
        by_agent[agent][key] = row
    diffs: list[float] = []
    clusters: list[tuple[Any, ...]] = []
    for key, ra in by_agent[agent_a].items():
        rb = by_agent[agent_b].get(key)
        if rb is None:
            continue
        if require_success and not (ra["success"] and rb["success"]):
            continue
        va, vb = ra.get(metric), rb.get(metric)
        if va is None or vb is None:
            continue
        a, b = float(va), float(vb)
        if math.isnan(a) or math.isnan(b):
            continue
        diffs.append(a - b)
        if cluster_keys is not None:
            clusters.append(tuple(ra[k] for k in cluster_keys))
    n_pairs = len(diffs)
    if cluster_keys is not None:
        grouped: dict[tuple[Any, ...], list[float]] = {}
        for cluster, d in zip(clusters, diffs, strict=True):
            grouped.setdefault(cluster, []).append(d)
        diffs = [float(np.mean(values)) for values in grouped.values()]
    arr = np.asarray(diffs, dtype=float)
    mean, lo, hi = mean_ci(arr, confidence)
    if arr.size >= 2 and float(arr.std(ddof=1)) == 0.0:
        lo = hi = mean
    return PairedDifference(mean, lo, hi, n_pairs, _paired_t_p(arr), _wilcoxon_p(arr),
                            int(arr.size), int(np.count_nonzero(arr)))


def sign_test(diffs: Iterable[Any]) -> tuple[int, int, float]:
    """Exact two-sided sign test of the differences: ``(n_positive, n_negative, p_value)``.

    Zeros and NaNs are dropped; ``p_value`` is 1.0 when no non-zero difference remains. With
    0/1 outcomes per unit (success at one condition minus success at another) this is the
    exact McNemar test.
    """
    arr = _finite(diffs)
    n_pos, n_neg = int(np.sum(arr > 0.0)), int(np.sum(arr < 0.0))
    if n_pos + n_neg == 0:
        return (n_pos, n_neg, 1.0)
    return (n_pos, n_neg, float(sps.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue))


# --- summary ----------------------------------------------------------------------------------
def _order_index(order: Sequence[str], value: str) -> tuple[int, str]:
    """Sort key: position in ``order`` (unknown values last, alphabetically)."""
    try:
        return (order.index(value), "")
    except ValueError:
        return (len(order), value)


def summarize(
    rows: Sequence[dict[str, Any]], confidence: float = DEFAULT_CONFIDENCE
) -> list[dict[str, Any]]:
    """One summary row per :data:`GROUP_KEYS` group, keyed by :data:`SUMMARY_COLUMNS`.

    ``success_rate`` is a fraction in [0, 1] with Wilson bounds ``success_lo`` /
    ``success_hi``; each metric of :data:`SUMMARY_METRICS` has ``_mean`` / ``_lo`` / ``_hi``
    (Student t). ``steps`` and ``wasted_moves`` use successful runs only; ``steps_all`` uses
    every run. Groups are ordered by experiment (first appearance), scenario (registry order),
    size, slip p, radius, replan mode, aftershock q and agent (display order).
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in GROUP_KEYS), []).append(row)
    experiments: list[str] = []
    for row in rows:
        if row["experiment"] not in experiments:
            experiments.append(row["experiment"])
    scenario_order = list(SCENARIOS)
    agent_order = list(ALL_AGENT_NAMES)

    def sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        g = dict(zip(GROUP_KEYS, key, strict=True))
        return (
            experiments.index(g["experiment"]),
            _order_index(scenario_order, g["scenario"]),
            g["rows"],
            g["slip_p"],
            g["sensor_radius"],
            g["replan_mode"],
            g["aftershock_q"],
            _order_index(agent_order, g["agent"]),
        )

    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=sort_key):
        members = groups[key]
        n = len(members)
        ok = [r for r in members if r["success"]]
        lo, hi = wilson_interval(len(ok), n, confidence)
        summary: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        summary.update(
            n=n,
            successes=len(ok),
            success_rate=len(ok) / n,
            success_lo=lo,
            success_hi=hi,
        )
        for reason in FAILURE_REASONS:
            summary[f"n_{reason}"] = sum(
                1 for r in members if not r["success"] and r.get("failure_reason") == reason
            )
        for metric in SUMMARY_METRICS:
            source = ok if metric in SUCCESS_ONLY_METRICS else members
            mean, mlo, mhi = mean_ci((r.get(metric) for r in source), confidence)
            summary.update({f"{metric}_mean": mean, f"{metric}_lo": mlo, f"{metric}_hi": mhi})
            if metric == "steps":
                mean, mlo, mhi = mean_ci((r.get(metric) for r in members), confidence)
                summary.update(steps_all_mean=mean, steps_all_lo=mlo, steps_all_hi=mhi)
        out.append({col: summary[col] for col in SUMMARY_COLUMNS})
    return out


def _format_cell(value: Any) -> str:
    """CSV text of one summary value (floats with 6 significant digits, NaN as 'nan')."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else format(value, _FLOAT_FORMAT)
    return str(value)


def write_summary(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Path:
    """Write :func:`summarize` of ``rows`` to ``path`` (``SUMMARY_COLUMNS`` order); returns it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        for summary in summarize(rows, confidence):
            writer.writerow([_format_cell(summary[col]) for col in SUMMARY_COLUMNS])
    return out


__all__ = [
    "DEFAULT_CONFIDENCE",
    "FAILURE_REASONS",
    "GROUP_KEYS",
    "MAP_KEYS",
    "SUCCESS_ONLY_METRICS",
    "SUMMARY_COLUMNS",
    "SUMMARY_METRICS",
    "TASK_KEYS",
    "WILCOXON_ZERO_METHOD",
    "PairedDifference",
    "mean_ci",
    "paired_difference",
    "read_raw",
    "sign_test",
    "summarize",
    "wilson_interval",
    "write_summary",
]
