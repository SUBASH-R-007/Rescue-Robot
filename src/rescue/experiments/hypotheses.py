"""Verdicts on the proposal deck's claims, from the Monte Carlo rows → ``results/HYPOTHESES.md``.

Brief §6.6. :func:`evaluate` checks each claim against typed raw rows (``stats.read_raw``) and
returns one :class:`Hypothesis` per claim with a verdict (✅ supported, ❌ refuted, ⚠️
inconclusive or not measured) and the numbers behind it; :func:`write_hypotheses` renders them
as Markdown with a header from ``run_meta.json``. Claims are evaluated as stated — thresholds
come from the deck, never from the data — and failed claims are reported as failed.

Claims (H1–H3 from the deck, H4 deck narrative, H5 derived, H6 reference):

* **H1** ``andor`` succeeds in ≥ 95 % of missions at p = 0.3 (slip sweep; Wilson 95 % CI).
* **H2** ``andor`` never reaches an unplanned state (Σ over every ``andor`` run; bumps too).
* **H3** ``andor`` wastes fewer moves than ``astar_replan`` (paired, both successful, p > 0).
* **H4** ``astar_fixed`` fails the first time an action misbehaves.
* **H5** ``andor`` replans less than ``astar_replan`` (paired, p > 0).
* **H6** ``vi_expected`` (expected-steps optimal) vs ``andor`` steps (paired, informational).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rescue.agents import (
    ContingencyAgent,
    FixedAStarAgent,
    ReplanAStarAgent,
    ValueIterationAgent,
)
from rescue.config import DEFAULT_SENSOR_RADIUS, DEFAULT_TERRAIN_HIGH_P
from rescue.experiments.run import META_FILE
from rescue.experiments.spec import SLIP_SWEEP_EXPERIMENT
from rescue.experiments.stats import (
    DEFAULT_CONFIDENCE,
    MAP_KEYS,
    TASK_KEYS,
    PairedDifference,
    paired_difference,
    sign_test,
    wilson_interval,
)
from rescue.mapgen import SCENARIOS

# --- verdicts and claim thresholds (from the deck, not tuned) ---------------------------------
VERDICT_PASS: str = "✅"
VERDICT_FAIL: str = "❌"
VERDICT_WARN: str = "⚠️"
NOT_MEASURED: str = "not measured"

#: H1: slip probability and success-rate target of the deck's claim.
H1_SLIP_P: float = 0.3
H1_TARGET: float = 0.95
#: H4: "no misbehaviour" slip probability and the lowest p of the "misbehaving" pool.
H4_BASE_P: float = 0.0
H4_MIN_P: float = 0.1
#: H4: significance level of the map-paired sign test (two-sided, matches the 95 % intervals).
H4_ALPHA: float = 1.0 - DEFAULT_CONFIDENCE
#: Fewest distinct slip probabilities an unnamed experiment needs to count as a slip sweep.
MIN_SWEEP_P_VALUES: int = 2
#: Tolerance when matching slip probabilities read back from CSV.
SLIP_P_TOL: float = 1e-9

#: Agent ids of the claims.
ANDOR: str = ContingencyAgent.name
ASTAR_FIXED: str = FixedAStarAgent.name
ASTAR_REPLAN: str = ReplanAStarAgent.name
VI_EXPECTED: str = ValueIterationAgent.name

#: Default experimental condition of the slip sweep (other conditions have their own blocks).
DEFAULT_REPLAN_MODE: str = "eager"
DEFAULT_AFTERSHOCK_Q: float = 0.0

#: Metadata file written by ``run.run_experiments`` next to ``raw.csv``.
RUN_META_FILE: str = META_FILE

_PCT_DIGITS: int = 1
_NUM_DIGITS: int = 2
_P_DIGITS: int = 2


@dataclass(frozen=True)
class Hypothesis:
    """One claim with its verdict, a one-line measurement and Markdown details."""

    claim: str
    verdict: str
    measured: str
    detail: str
    label: str = ""


# --- formatting -------------------------------------------------------------------------------
def _pct(x: float) -> str:
    """Fraction as a percentage string ('n/a' for NaN)."""
    return "n/a" if math.isnan(x) else f"{100.0 * x:.{_PCT_DIGITS}f}%"


def _num(x: float, unit: str = "") -> str:
    """Signed number with fixed decimals ('n/a' for NaN)."""
    if math.isnan(x):
        return "n/a"
    text = f"{x:+.{_NUM_DIGITS}f}"
    return f"{text} {unit}".rstrip()


def _pval(p: float) -> str:
    """p-value with two significant digits ('n/a' for NaN)."""
    return "n/a" if math.isnan(p) else f"{p:.{_P_DIGITS}g}"


def _rate(successes: int, n: int) -> str:
    """'97.3% [95.1%, 98.6%] (778/800)' with the Wilson 95 % interval."""
    if n == 0:
        return "n/a (0 runs)"
    lo, hi = wilson_interval(successes, n, DEFAULT_CONFIDENCE)
    return f"{_pct(successes / n)} [{_pct(lo)}, {_pct(hi)}] ({successes}/{n})"


def _diff(pd: PairedDifference, unit: str) -> str:
    """Mean paired difference with CI, number of maps / pairs and both p-values."""
    if pd.n_pairs == 0:
        return "no paired runs"
    return (
        f"{_num(pd.mean_diff, unit)} [{_num(pd.lo)}, {_num(pd.hi)}] over {pd.n_units} maps "
        f"({pd.n_pairs} pairs), paired t p = {_pval(pd.p_value)}, Wilcoxon p = "
        f"{_pval(pd.p_wilcoxon)} ({pd.n_nonzero}/{pd.n_units} maps with a non-zero difference)"
    )


def _cell(text: Any) -> str:
    """Markdown table cell text (pipes escaped, newlines flattened)."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], body: Sequence[Sequence[Any]]) -> str:
    """Markdown table."""
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def _p_label(p: float) -> str:
    """Slip probability label."""
    return f"{p:g}"


def _values(rows: Sequence[dict[str, Any]], column: str) -> list[Any]:
    """Distinct values of ``column`` in first-appearance order (sorted for numbers)."""
    seen: list[Any] = []
    for row in rows:
        if row[column] not in seen:
            seen.append(row[column])
    if all(isinstance(v, int | float) for v in seen):
        seen.sort()
    elif column == "scenario":
        order = list(SCENARIOS)
        seen.sort(key=lambda s: (order.index(s) if s in order else len(order), s))
    return seen


# --- row selection ----------------------------------------------------------------------------
def _is_p(row: dict[str, Any], p: float) -> bool:
    """Row's slip probability equals ``p`` (within :data:`SLIP_P_TOL`)."""
    return abs(float(row["slip_p"]) - p) <= SLIP_P_TOL


def _of(rows: Sequence[dict[str, Any]], agent: str) -> list[dict[str, Any]]:
    """Rows of one agent."""
    return [r for r in rows if r["agent"] == agent]


def _is_default_condition(row: dict[str, Any]) -> bool:
    """Default radius, eager replanning and no aftershocks."""
    return (
        row["sensor_radius"] == DEFAULT_SENSOR_RADIUS
        and row["replan_mode"] == DEFAULT_REPLAN_MODE
        and abs(float(row["aftershock_q"]) - DEFAULT_AFTERSHOCK_Q) <= SLIP_P_TOL
    )


def _pick_slip_experiment(rows: Sequence[dict[str, Any]], preferred: str | None) -> str | None:
    """Name of the slip-sweep experiment.

    ``preferred`` (if given and present), else :data:`SLIP_SWEEP_EXPERIMENT`, else the
    experiment whose default-condition rows span the most (at least
    :data:`MIN_SWEEP_P_VALUES`) slip probabilities (ties: more rows, then first appearance).
    Only one experiment is used: blocks share task seeds, so pooling blocks would count
    identical missions twice.
    """
    names = _values(rows, "experiment") if rows else []
    for name in (preferred, SLIP_SWEEP_EXPERIMENT):
        if name is not None and name in names:
            return name
    best: tuple[int, int, int] | None = None
    best_name: str | None = None
    for i, name in enumerate(names):
        sub = [r for r in rows if r["experiment"] == name and _is_default_condition(r)]
        score = (len({round(float(r["slip_p"]), 9) for r in sub}), len(sub), -i)
        if score[0] >= MIN_SWEEP_P_VALUES and (best is None or score > best):
            best, best_name = score, name
    return best_name


def _sweep(
    rows: Sequence[dict[str, Any]], preferred: str | None
) -> tuple[str | None, list[dict[str, Any]], str]:
    """(experiment name, its default-condition rows, a scope sentence for the details)."""
    name = _pick_slip_experiment(rows, preferred)
    if name is None:
        return None, [], "no slip-sweep experiment in the data"
    sub = [r for r in rows if r["experiment"] == name]
    default = [r for r in sub if _is_default_condition(r)]
    note = ""
    if default and len(default) < len(sub):
        note = (
            f" (restricted to r = {DEFAULT_SENSOR_RADIUS}, {DEFAULT_REPLAN_MODE} replanning, "
            "no aftershocks)"
        )
    elif not default:
        default = sub
        note = " (no default-condition rows; all its conditions pooled)"
    sizes = ", ".join(f"{s}×{s}" for s in _values(default, "rows"))
    scenarios = ", ".join(_values(default, "scenario"))
    scope = (
        f"Experiment `{name}`{note}; sizes pooled: {sizes}; scenarios: {scenarios}; "
        f"slip p ∈ {{{', '.join(_p_label(p) for p in _values(default, 'slip_p'))}}}."
    )
    return name, default, scope


def _mission_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """What determines a mission's outcome (the experiment name does not).

    ``aftershock_k`` matters only when aftershocks can happen (q > 0).
    """
    q = round(float(row["aftershock_q"]), 9)
    return (
        row["agent"], row["scenario"], row["rows"], row["cols"], row["seed"],
        round(float(row["slip_p"]), 9), row["sensor_radius"], row["replan_mode"], q,
        row["aftershock_k"] if q > 0.0 else None,
    )


def _not_measured(label: str, claim: str, why: str) -> Hypothesis:
    """A ⚠️ hypothesis for a claim the data cannot test."""
    return Hypothesis(claim, VERDICT_WARN, f"{NOT_MEASURED}: {why}", f"{NOT_MEASURED}: {why}.",
                      label)


def _diff_verdict(pd: PairedDifference) -> str:
    """✅ CI entirely below 0, ❌ entirely above 0, ⚠️ contains 0 (or undefined)."""
    if pd.n_pairs >= 2 and not math.isnan(pd.hi) and pd.hi < 0.0:
        return VERDICT_PASS
    if pd.n_pairs >= 2 and not math.isnan(pd.lo) and pd.lo > 0.0:
        return VERDICT_FAIL
    return VERDICT_WARN


# --- claims -----------------------------------------------------------------------------------
def _h1(sweep: list[dict[str, Any]], scope: str) -> Hypothesis:
    claim = (
        f"`{ANDOR}` rescues the survivor in ≥ {_pct(H1_TARGET)} of missions at "
        f"p = {H1_SLIP_P:g}"
    )
    runs = [r for r in _of(sweep, ANDOR) if _is_p(r, H1_SLIP_P)]
    if not runs:
        return _not_measured("H1", claim, f"no `{ANDOR}` runs at p = {H1_SLIP_P:g}")
    k, n = sum(1 for r in runs if r["success"]), len(runs)
    rate = k / n
    lo, hi = wilson_interval(k, n, DEFAULT_CONFIDENCE)
    if rate >= H1_TARGET:
        verdict = VERDICT_PASS
    elif hi >= H1_TARGET:
        verdict = VERDICT_WARN
    else:
        verdict = VERDICT_FAIL
    body = []
    below: list[str] = []
    for scenario in _values(runs, "scenario"):
        sub = [r for r in runs if r["scenario"] == scenario]
        ks = sum(1 for r in sub if r["success"])
        if ks / len(sub) < H1_TARGET:
            below.append(f"{scenario} {_pct(ks / len(sub))}")
        reasons = Counter(r["failure_reason"] for r in sub if not r["success"])
        body.append([scenario, _rate(ks, len(sub)),
                     ", ".join(f"{c} {why}" for why, c in sorted(reasons.items())) or "—"])
    # The rule passes on the point estimate; say whether the interval supports it too, and flag
    # any scenario below the target that the pooled verdict would otherwise hide.
    measured = (
        f"{_rate(k, n)}; Wilson lower bound {_pct(lo)} "
        + (f"≥ {_pct(H1_TARGET)}" if lo >= H1_TARGET else
           f"< {_pct(H1_TARGET)} (target not statistically established)")
    )
    if below:
        measured += f"; {VERDICT_WARN} below {_pct(H1_TARGET)} in: {', '.join(below)}"
    sizes = []
    for size in _values(runs, "rows"):
        sub = [r for r in runs if r["rows"] == size]
        sizes.append([f"{size}×{size}", _rate(sum(1 for r in sub if r["success"]), len(sub))])
    detail = "\n\n".join([
        scope,
        f"Overall at p = {H1_SLIP_P:g}: {_rate(k, n)} (Wilson 95 % CI). Rule: ✅ if the point "
        f"estimate ≥ {_pct(H1_TARGET)}; ⚠️ if it is below but the CI upper bound reaches "
        f"{_pct(H1_TARGET)}; ❌ if the upper bound is below {_pct(H1_TARGET)}. The ✅ rule "
        "rests on the point estimate, so the claims table also states whether the Wilson "
        f"lower bound clears {_pct(H1_TARGET)} and names any scenario below it.",
        _table(["scenario", "success (Wilson 95 % CI)", "failures"], body),
        _table(["size", "success (Wilson 95 % CI)"], sizes),
    ])
    return Hypothesis(claim, verdict, measured, detail, "H1")


def _h2(rows: Sequence[dict[str, Any]]) -> Hypothesis:
    claim = f"`{ANDOR}` never reaches an unplanned state (and never bumps into debris)"
    runs = _of(rows, ANDOR)
    if not runs:
        return _not_measured("H2", claim, f"no `{ANDOR}` runs")
    unplanned = sum(int(r["unplanned_states"]) for r in runs)
    bumps = sum(int(r["bumps"]) for r in runs)
    verdict = VERDICT_PASS if unplanned == 0 else VERDICT_FAIL
    unique = len({_mission_key(r) for r in runs})
    measured = (
        f"Σ unplanned states = {unplanned}, Σ bumps = {bumps} over {len(runs)} `{ANDOR}` runs "
        f"({unique} unique missions)"
    )
    body = []
    for name in _values(runs, "experiment"):
        for radius in _values([r for r in runs if r["experiment"] == name], "sensor_radius"):
            sub = [r for r in runs if r["experiment"] == name and r["sensor_radius"] == radius]
            body.append([name, radius, len(sub), sum(int(r["unplanned_states"]) for r in sub),
                         sum(int(r["bumps"]) for r in sub)])
    parts = [
        "All experiments (every condition, every size). Rule: ✅ iff the sum of unplanned "
        "states over all runs is 0. Bumps (D6) are reported alongside: with r ≥ 1 both are "
        "invariants of the design.",
        _table(["experiment", "sensor r", f"`{ANDOR}` runs", "Σ unplanned", "Σ bumps"], body),
        f"Blocks share task seeds (the seed depends only on scenario, size and replicate), so a "
        "block's default-condition runs replay missions of other blocks exactly (same map, "
        "outcome stream and parameters). Deduplicated on (scenario, size, seed, p, r, replan "
        f"mode, q, k), the {len(runs)} runs are {unique} unique missions.",
    ]
    vi = _of(rows, VI_EXPECTED)
    if vi:
        parts.append(
            f"For reference, `{VI_EXPECTED}` (same anticipation, D19): Σ unplanned = "
            f"{sum(int(r['unplanned_states']) for r in vi)}, Σ bumps = "
            f"{sum(int(r['bumps']) for r in vi)} over {len(vi)} runs."
        )
    if bumps:
        parts.append(f"⚠️ {bumps} bumps were recorded for `{ANDOR}` (see the table).")
    return Hypothesis(claim, verdict, measured, "\n\n".join(parts), "H2")


def _paired_claim(
    label: str,
    claim: str,
    sweep: list[dict[str, Any]],
    scope: str,
    metric: str,
    agent_a: str,
    agent_b: str,
    unit: str,
    rule_scope: str,
) -> Hypothesis:
    """Paired ``agent_a − agent_b`` difference of ``metric`` over p > 0, map as the unit.

    Common random numbers reuse every map and outcome stream at each slip p, so one map's
    differences at p = 0.1 … 0.4 are strongly correlated: they are averaged per map first and
    the CI and tests run over maps (``stats.MAP_KEYS``), not over pairs. Per-p, per-scenario
    and per-size rows follow, each with its own verdict.
    """
    present = {r["agent"] for r in sweep}
    missing = [a for a in (agent_a, agent_b) if a not in present]
    if missing:
        return _not_measured(label, claim, f"no {', '.join(f'`{a}`' for a in missing)} runs "
                                           "in the slip sweep")

    def paired(sub: Sequence[dict[str, Any]]) -> PairedDifference:
        return paired_difference(sub, metric, agent_a, agent_b, TASK_KEYS,
                                 cluster_keys=MAP_KEYS)

    def row(name: str, pd: PairedDifference) -> list[Any]:
        return [name, pd.n_units, pd.n_pairs, _num(pd.mean_diff),
                f"[{_num(pd.lo)}, {_num(pd.hi)}]", _pval(pd.p_value), _pval(pd.p_wilcoxon),
                _diff_verdict(pd) if pd.n_pairs else "—"]

    header = ["maps", "pairs", "mean diff", "95 % CI", "paired t p", "Wilcoxon p", "verdict"]
    positive = [r for r in sweep if float(r["slip_p"]) > SLIP_P_TOL]
    overall = paired(positive)
    if overall.n_pairs == 0:
        return _not_measured(label, claim, f"no paired {rule_scope} at p > 0")
    verdict = _diff_verdict(overall)
    by_p = []
    for p in _values(sweep, "slip_p"):
        note = "" if p > SLIP_P_TOL else " (p = 0, not in the verdict)"
        by_p.append(row(_p_label(p) + note, paired([r for r in sweep if _is_p(r, p)])))
    scenarios = [(s, paired([r for r in positive if r["scenario"] == s]))
                 for s in _values(positive, "scenario")]
    sizes = [(f"{s}×{s}", paired([r for r in positive if r["rows"] == s]))
             for s in _values(positive, "rows")]
    measured = _diff(overall, unit)
    if len(scenarios) > 1:
        counts = Counter(_diff_verdict(pd) for _, pd in scenarios if pd.n_pairs)
        measured += (f"; per scenario {VERDICT_PASS} {counts[VERDICT_PASS]} / {VERDICT_WARN} "
                     f"{counts[VERDICT_WARN]} / {VERDICT_FAIL} {counts[VERDICT_FAIL]}")
        refuted = [s for s, pd in scenarios if pd.n_pairs and _diff_verdict(pd) == VERDICT_FAIL]
        against = [f"{s} ({_num(pd.mean_diff)})" for s, pd in scenarios
                   if pd.n_pairs and pd.mean_diff > 0.0]
        if refuted:
            measured += f"; CI above 0 (claim refuted) in {', '.join(refuted)}"
        if against:
            measured += f"; mean above 0 (against the claim) in {', '.join(against)}"
    detail = "\n\n".join([
        scope,
        f"Difference = `{agent_a}` − `{agent_b}` in {metric} on identical tasks (same map, same "
        f"outcome stream), {rule_scope}, pooled over p > 0: {_diff(overall, unit)}.",
        "Unit of analysis: the map. Every map and its outcome stream are reused at each slip p "
        "(common random numbers), so a map's differences at the p > 0 values are averaged "
        "first; the Student-t CI, the paired t-test and the Wilcoxon signed-rank test (Pratt: "
        "zero differences are ranked, not discarded) run over those per-map means. Rule: ✅ if "
        "the 95 % CI lies below 0, ❌ if it lies above 0, ⚠️ if it contains 0 — applied to the "
        "pooled verdict and to every row below.",
        "By slip p (one pair per map):",
        _table(["slip p", *header], by_p),
        "By scenario (p > 0):",
        _table(["scenario", *header], [row(s, pd) for s, pd in scenarios]),
        "By grid size (p > 0):",
        _table(["size", *header], [row(s, pd) for s, pd in sizes]),
    ])
    return Hypothesis(claim, verdict, measured, detail, label)


def _h4(sweep: list[dict[str, Any]], scope: str) -> Hypothesis:
    claim = f"`{ASTAR_FIXED}` fails the first time an action misbehaves (slip or skid)"
    runs = _of(sweep, ASTAR_FIXED)
    base = [r for r in runs if _is_p(r, H4_BASE_P)]
    slippy = [r for r in runs if float(r["slip_p"]) >= H4_MIN_P - SLIP_P_TOL]
    if not base or not slippy:
        return _not_measured("H4", claim, f"`{ASTAR_FIXED}` not run at both p = {H4_BASE_P:g} "
                                          f"and p ≥ {H4_MIN_P:g}")
    k0, n0 = sum(1 for r in base if r["success"]), len(base)
    k1, n1 = sum(1 for r in slippy if r["success"]), len(slippy)
    # The p ≥ 0.1 runs reuse the p = 0 maps (common random numbers): compare within each map
    # (success at p = 0 minus mean success at p ≥ 0.1) with an exact sign test over maps,
    # instead of treating the pooled runs as independent.
    base_by_map = {tuple(r[k] for k in MAP_KEYS): float(r["success"]) for r in base}
    slippy_by_map: dict[tuple[Any, ...], list[float]] = {}
    for r in slippy:
        slippy_by_map.setdefault(tuple(r[k] for k in MAP_KEYS), []).append(float(r["success"]))
    map_diffs = [base_by_map[m] - sum(v) / len(v) for m, v in slippy_by_map.items()
                 if m in base_by_map]
    worse, better, p_sign = sign_test(map_diffs)
    misbehaved = [r for r in runs if int(r["slips"]) + int(r["skids"]) > 0]
    survived = [r for r in misbehaved if r["success"]]
    if k1 / n1 >= k0 / n0:
        verdict = VERDICT_FAIL
    elif p_sign >= H4_ALPHA or worse <= better or survived:
        verdict = VERDICT_WARN
    else:
        verdict = VERDICT_PASS
    measured = (
        f"success {_rate(k0, n0)} at p = {H4_BASE_P:g} vs {_pct(k1 / n1)} ({k1}/{n1} runs on "
        f"{len(slippy_by_map)} maps) at p ≥ {H4_MIN_P:g}; map-paired sign test p = "
        f"{_pval(p_sign)} ({worse} maps worse, {better} better at p ≥ {H4_MIN_P:g}); "
        f"{len(misbehaved) - len(survived)}/{len(misbehaved)} runs with a slip or skid failed"
    )

    def reasons(sub: list[dict[str, Any]]) -> list[int]:
        deviation = sum(1 for r in sub if not r["success"] and r["failure_reason"] == "aborted"
                        and int(r["slips"]) + int(r["skids"]) > 0)
        debris = sum(1 for r in sub if not r["success"] and r["failure_reason"] == "aborted"
                     and int(r["slips"]) + int(r["skids"]) == 0)
        other = Counter(r["failure_reason"] for r in sub
                        if not r["success"] and r["failure_reason"] != "aborted")
        return [deviation, debris, other.get("budget", 0), other.get("unreachable", 0)]

    body = [[_p_label(p), _rate(sum(1 for r in sub if r["success"]), len(sub)), *reasons(sub)]
            for p in _values(runs, "slip_p")
            for sub in [[r for r in runs if _is_p(r, p)]]]
    base_misbehaved = sum(1 for r in base if int(r["slips"]) + int(r["skids"]) > 0)
    base_debris = reasons(base)[1]
    caveat = (
        f"Caveats: (1) p = {H4_BASE_P:g} is not free of misbehaviour: high-slip terrain ('~' "
        f"cells) keeps its own slip parameter ({DEFAULT_TERRAIN_HIGH_P:g}) at every p, so "
        f"{base_misbehaved}/{n0} runs at p = {H4_BASE_P:g} had a slip or skid. "
        f"(2) `{ASTAR_FIXED}` also aborts when it senses debris on its next path cell, so its "
        f"success at p = {H4_BASE_P:g} is already only {_rate(k0, n0)} ({base_debris} aborts at "
        "debris ahead): most of its failures come from debris it did not know about, not from "
        "a misbehaving action. The verdict covers the slip/skid part of the claim only."
    )
    detail = "\n\n".join([
        scope,
        f"Rule: ✅ if the success rate at p ≥ {H4_MIN_P:g} is below the one at p = "
        f"{H4_BASE_P:g}, the drop is significant in an exact two-sided sign test over maps "
        f"(α = {H4_ALPHA:g}; per map: success at p = {H4_BASE_P:g} minus mean success at "
        f"p ≥ {H4_MIN_P:g}; with one p this is the exact McNemar test) **and** every run with "
        "at least one slip or skid failed; ⚠️ if the drop is not significant or some runs "
        "survived a slip or skid; ❌ if the success rate does not drop. The map is the unit "
        "because every p reuses the same maps (common random numbers); the per-p Wilson "
        "intervals below are each over distinct maps.",
        f"Runs with ≥ 1 non-arrived outcome: {len(misbehaved)}; survived: {len(survived)} "
        "(a run can survive only if the deviation itself lands on the survivor).",
        "Abort reasons: *after a slip/skid* = `aborted` with slips + skids > 0 (deviation from "
        "the path); *debris ahead* = `aborted` with none (the next path cell turned out to be "
        "debris; no replanning).",
        _table(["slip p", "success (Wilson 95 % CI)", "aborted after slip/skid",
                "aborted at debris ahead", "budget", "unreachable"], body),
        caveat,
    ])
    return Hypothesis(claim, verdict, measured, detail, "H4")


def _h6(rows: Sequence[dict[str, Any]], sweep_name: str | None) -> Hypothesis:
    claim = (f"Reference: `{VI_EXPECTED}` (expected-steps optimal policy) needs fewer steps "
             f"than `{ANDOR}`")
    names = [n for n in _values(rows, "experiment")
             if {VI_EXPECTED, ANDOR} <= {r["agent"] for r in rows if r["experiment"] == n}]
    if not names:
        return _not_measured("H6", claim, f"`{VI_EXPECTED}` and `{ANDOR}` never ran together")
    if sweep_name in names:
        _, sub, scope = _sweep(rows, sweep_name)
    else:
        _, sub, scope = _sweep(rows, names[0])
    hyp = _paired_claim("H6", claim, sub, scope, "steps", VI_EXPECTED, ANDOR, "steps",
                        "both successful")
    if hyp.verdict == VERDICT_WARN and hyp.measured.startswith(NOT_MEASURED):
        return hyp
    vi_runs, andor_runs = _of(sub, VI_EXPECTED), _of(sub, ANDOR)
    extra = (
        f"Informational (no deck claim). Success: `{VI_EXPECTED}` "
        f"{_rate(sum(1 for r in vi_runs if r['success']), len(vi_runs))}, `{ANDOR}` "
        f"{_rate(sum(1 for r in andor_runs if r['success']), len(andor_runs))} (all p)."
    )
    return Hypothesis(hyp.claim, hyp.verdict, hyp.measured, f"{hyp.detail}\n\n{extra}", "H6")


def evaluate(
    rows: Sequence[dict[str, Any]], slip_experiment: str | None = None
) -> list[Hypothesis]:
    """Verdicts H1–H6 on typed raw rows; missing blocks / agents give ⚠️ 'not measured'.

    ``slip_experiment`` overrides the slip-sweep experiment (default
    :data:`SLIP_SWEEP_EXPERIMENT`, else the experiment sweeping the most p values).
    """
    name, sweep, scope = _sweep(rows, slip_experiment)
    results = [
        _h1(sweep, scope),
        _h2(rows),
        _paired_claim(
            "H3", f"`{ANDOR}` wastes fewer moves than `{ASTAR_REPLAN}`", sweep, scope,
            "wasted_moves", ANDOR, ASTAR_REPLAN, "moves", "both successful",
        ),
        _h4(sweep, scope),
        _paired_claim(
            "H5", f"`{ANDOR}` replans less often than `{ASTAR_REPLAN}`", sweep, scope,
            "replans", ANDOR, ASTAR_REPLAN, "replans", "all runs",
        ),
        _h6(rows, name),
    ]
    return results


# --- report -----------------------------------------------------------------------------------
def _meta_get(meta: dict[str, Any], *names: str) -> Any:
    """First present, non-empty value among ``names`` (None if none)."""
    for name in names:
        value = meta.get(name)
        if value not in (None, ""):
            return value
    return None


def _read_meta(path: Path) -> dict[str, Any]:
    """``run_meta.json`` as a dict ({} if missing or unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _header(rows: Sequence[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    """Title and run metadata lines."""
    name = _meta_get(meta, "config_name", "name", "experiment") or "unknown configuration"
    fields: list[tuple[str, Callable[[Any], str], tuple[str, ...]]] = [
        ("Config file", lambda v: f"`{v}`", ("config_path", "config")),
        ("Base seed", str, ("base_seed",)),
        ("Tasks (maps × conditions)", str, ("n_tasks", "tasks")),
        ("Workers", str, ("workers",)),
        ("Wall time", lambda v: f"{float(v):.1f} s" if isinstance(v, int | float) else str(v),
         ("wall_time_s", "wall_time", "elapsed_s", "wall_s")),
        ("Git commit", lambda v: f"`{v}`", ("git_commit", "commit")),
    ]
    date = _meta_get(meta, "date", "finished", "finished_at", "started", "started_at",
                     "timestamp")
    agents = _values(rows, "agent") if rows else []
    lines = [
        f"# Hypotheses — `{name}`",
        "",
        f"- Date of run: {date if date is not None else 'unknown (no run metadata)'}",
        f"- Missions (rows of raw.csv): {len(rows)}; agents: "
        f"{', '.join(f'`{a}`' for a in agents) or 'none'}",
    ]
    for title, fmt, keys in fields:
        value = _meta_get(meta, *keys)
        if value is not None:
            lines.append(f"- {title}: {fmt(value)}")
    lines += [
        "- Methods: success rates with Wilson 95 % intervals; means with Student-t 95 % "
        "intervals; agent comparisons paired on identical tasks (common random numbers: same "
        "map, same outcome stream) with a paired t-test and a Wilcoxon signed-rank test "
        "(Pratt zero handling). Because each map is reused at every slip p, pooled "
        "comparisons use the map as the unit (per-map mean difference; n = maps, not pairs).",
        f"- Verdicts: {VERDICT_PASS} supported, {VERDICT_FAIL} refuted, {VERDICT_WARN} "
        "inconclusive or not measured. Thresholds come from the proposal deck; no experiment "
        "was tuned toward a verdict.",
    ]
    return lines


def render_markdown(
    rows: Sequence[dict[str, Any]],
    hypotheses: Sequence[Hypothesis],
    meta: dict[str, Any] | None = None,
) -> str:
    """HYPOTHESES.md text: header, claims table, per-claim details."""
    lines = _header(rows, meta or {})
    lines += ["", "## Claims", ""]
    lines.append(_table(
        ["#", "Claim", "Verdict", "Measured"],
        [[h.label or str(i + 1), h.claim, h.verdict, h.measured]
         for i, h in enumerate(hypotheses)],
    ))
    lines += ["", "## Details"]
    for i, h in enumerate(hypotheses):
        lines += ["", f"### {h.label or i + 1}. {h.verdict} {h.claim}", "", h.detail]
    return "\n".join(lines) + "\n"


def write_hypotheses(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    meta_path: str | Path | None = None,
    slip_experiment: str | None = None,
) -> Path:
    """Evaluate the claims on ``rows`` and write the Markdown report to ``path``.

    Run metadata is read from ``meta_path`` (default: ``run_meta.json`` next to ``path``).
    """
    out = Path(path)
    meta = _read_meta(Path(meta_path) if meta_path is not None else out.parent / RUN_META_FILE)
    text = render_markdown(rows, evaluate(rows, slip_experiment), meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


__all__ = [
    "H1_SLIP_P",
    "H1_TARGET",
    "NOT_MEASURED",
    "RUN_META_FILE",
    "VERDICT_FAIL",
    "VERDICT_PASS",
    "VERDICT_WARN",
    "Hypothesis",
    "evaluate",
    "render_markdown",
    "write_hypotheses",
]
