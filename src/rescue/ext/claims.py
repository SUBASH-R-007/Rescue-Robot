"""Verdicts on the extension claims X1–X5 → ``results/ext/EXTENSIONS.md`` (plan §3.14).

Each claim is checked against typed raw rows (:func:`rescue.ext.experiments.read_raw`) with the
core harness's statistics (:mod:`rescue.experiments.stats`: Wilson intervals, Student-t
intervals, paired differences with the map as the unit) and gets a verdict (✅ supported,
❌ refuted, ⚠️ inconclusive or not measured). Thresholds are named constants in
:mod:`rescue.config` fixed before the full run; the rule of each claim is printed next to it.

* **X1** frontier search finds and rescues an unknown survivor in ≥ 95 % of missions
  (``search`` block, ``search_andor``, location unknown);
* **X2** not knowing the survivor's location costs extra steps (paired unknown − known);
* **X3** the greedy nearest-first order is within 10 % of the optimal tour (per k);
* **X4** with noisy sensing collisions become possible, but success stays ≥ 95 % at ε = 0.1;
* **X5** under noise ``search_andor`` has at most ``EXT_CLAIM_BUMP_MARGIN`` more collisions
  per mission than ``search_astar`` (paired non-inferiority, pooled and at every ε).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rescue.config import (
    EXT_CLAIM_BUMP_MARGIN,
    EXT_CLAIM_NOISE,
    EXT_CLAIM_SUCCESS_TARGET,
    EXT_CLAIM_TOUR_RATIO_MAX,
)
from rescue.experiments.hypotheses import VERDICT_FAIL, VERDICT_PASS, VERDICT_WARN, Hypothesis
from rescue.experiments.run import META_FILE
from rescue.experiments.stats import PairedDifference, mean_ci, paired_difference, wilson_interval
from rescue.ext.experiments import (
    EXT_MAP_KEYS,
    EXT_TASK_KEYS,
    MULTI_BLOCK,
    NOISY_BLOCK,
    SEARCH_BLOCK,
    steps_per_rescue,
    tour_ratio,
    unique_layouts,
)

Row = dict[str, Any]

#: Agents of the claims.
SEARCH_ANDOR: str = "search_andor"
SEARCH_ASTAR: str = "search_astar"
#: Pseudo-agent labels used to pair a mission with its known-location twin (X2).
UNKNOWN_LABEL: str = "unknown"
KNOWN_LABEL: str = "known"
#: X2: columns that pair an unknown-location run with its known-location twin.
LOCATION_PAIR_KEYS: tuple[str, ...] = tuple(k for k in EXT_TASK_KEYS if k != "survivors_known")
#: Tolerance when matching float factors read back from CSV.
FLOAT_TOL: float = 1e-9
_PCT_DIGITS: int = 1
_NUM_DIGITS: int = 2
_P_DIGITS: int = 2

__all__ = [
    "evaluate_claims",
    "render_markdown",
    "write_extensions",
]


# --- formatting ------------------------------------------------------------------------------
def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100.0 * x:.{_PCT_DIGITS}f}%"


def _num(x: float, signed: bool = False) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{x:+.{_NUM_DIGITS}f}" if signed else f"{x:.{_NUM_DIGITS}f}"


def _pval(p: float) -> str:
    return "n/a" if math.isnan(p) else f"{p:.{_P_DIGITS}g}"


def _rate(ok: int, n: int) -> str:
    """'97.3% [95.1%, 98.6%] (778/800)' (Wilson 95 % CI)."""
    if n == 0:
        return "n/a (0 runs)"
    lo, hi = wilson_interval(ok, n)
    return f"{_pct(ok / n)} [{_pct(lo)}, {_pct(hi)}] ({ok}/{n})"


def _mean(values: Sequence[float]) -> str:
    """'12.34 [11.0, 13.6] (n = 40)' (Student-t 95 % CI; NaNs dropped)."""
    finite = [v for v in values if not math.isnan(v)]
    mean, lo, hi = mean_ci(finite)
    if not finite:
        return "n/a"
    return f"{_num(mean)} [{_num(lo)}, {_num(hi)}] (n = {len(finite)})"


def _diff(pd: PairedDifference, unit: str) -> str:
    if pd.n_pairs == 0:
        return "no paired runs"
    return (f"{_num(pd.mean_diff, True)} {unit} [{_num(pd.lo, True)}, {_num(pd.hi, True)}] over "
            f"{pd.n_units} maps ({pd.n_pairs} pairs), paired t p = {_pval(pd.p_value)}, "
            f"Wilcoxon p = {_pval(pd.p_wilcoxon)}")


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], body: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


# --- row helpers -----------------------------------------------------------------------------
def _block(rows: Sequence[Row], name: str) -> list[Row]:
    return [r for r in rows if r["experiment"] == name]


def _where(rows: Sequence[Row], **conditions: Any) -> list[Row]:
    def match(row: Row) -> bool:
        for key, want in conditions.items():
            have = row[key]
            if isinstance(want, float) and not isinstance(want, bool):
                if abs(float(have) - want) > FLOAT_TOL:
                    return False
            elif have != want:
                return False
        return True

    return [r for r in rows if match(r)]


def _distinct(rows: Sequence[Row], key: str) -> list[Any]:
    return sorted({r[key] for r in rows})


def _scenarios(rows: Sequence[Row]) -> list[str]:
    return list(dict.fromkeys(r["scenario"] for r in rows))


def _successes(rows: Sequence[Row]) -> int:
    return sum(bool(r["success"]) for r in rows)


def _rate_verdict(ok: int, n: int, target: float) -> str:
    """H1's rule: ✅ point estimate ≥ target; ⚠️ below but CI upper bound reaches it; else ❌."""
    if n == 0:
        return VERDICT_WARN
    _, hi = wilson_interval(ok, n)
    if ok / n >= target:
        return VERDICT_PASS
    return VERDICT_WARN if hi >= target else VERDICT_FAIL


def _not_measured(label: str, claim: str, why: str) -> Hypothesis:
    return Hypothesis(claim, VERDICT_WARN, f"not measured ({why})", f"Not measured: {why}.",
                      label)


def _rate_rule(target: float) -> str:
    return (f"Rule: ✅ if the success rate is ≥ {_pct(target)}; ⚠️ if it is below but the "
            f"Wilson 95 % upper bound reaches {_pct(target)}; ❌ if the upper bound is below it.")


# --- X1 --------------------------------------------------------------------------------------
X1_CLAIM = (f"frontier search finds and rescues an unknown survivor in ≥ "
            f"{_pct(EXT_CLAIM_SUCCESS_TARGET)} of missions (`{SEARCH_ANDOR}`)")


def claim_x1(rows: Sequence[Row]) -> Hypothesis:
    """X1: success of ``search_andor`` with the survivor location unknown (search block)."""
    block = _block(rows, SEARCH_BLOCK)
    runs = _where(block, agent=SEARCH_ANDOR, survivors_known=False)
    if not runs:
        return _not_measured("X1", X1_CLAIM, f"no unknown-location `{SEARCH_ANDOR}` runs in "
                                             f"block `{SEARCH_BLOCK}`")
    ok, n = _successes(runs), len(runs)
    verdict = _rate_verdict(ok, n, EXT_CLAIM_SUCCESS_TARGET)
    lo, _ = wilson_interval(ok, n)
    bumps = sum(r["bumps"] for r in runs)
    measured = (f"{_rate(ok, n)}; Wilson lower bound {_pct(lo)}; Σ bumps = {bumps}, "
                f"Σ unplanned states = {sum(r['unplanned_states'] for r in runs)}")
    body = []
    for agent in _distinct(block, "agent"):
        for p in _distinct(block, "slip_p"):
            for scenario in _scenarios(block):
                own = _where(block, agent=agent, slip_p=p, scenario=scenario,
                             survivors_known=False)
                if not own:
                    continue
                body.append([f"`{agent}`", f"{p:g}", scenario, _rate(_successes(own), len(own)),
                             _mean([r["first_detection_step"] for r in own]),
                             _mean([float(r["steps"]) for r in own if r["success"]]),
                             sum(r["bumps"] for r in own)])
    detail = "\n".join([
        f"Block `{SEARCH_BLOCK}`, survivor location unknown (the belief starts all UNKNOWN; the "
        "agent knows only how many survivors there are). Pooled over scenarios and p: "
        f"{_rate(ok, n)}.",
        "",
        _rate_rule(EXT_CLAIM_SUCCESS_TARGET),
        "",
        _table(["agent", "p", "scenario", "success (Wilson 95 % CI)",
                "first detection step", "steps (successful)", "Σ bumps"], body),
    ])
    return Hypothesis(X1_CLAIM, verdict, measured, detail, "X1")


# --- X2 --------------------------------------------------------------------------------------
X2_CLAIM = "not knowing the survivor's location costs extra steps (paired unknown − known)"


def _location_rows(rows: Sequence[Row], agent: str) -> list[Row]:
    """``agent``'s search-block rows relabelled ``unknown`` / ``known`` for pairing."""
    return [dict(r, agent=KNOWN_LABEL if r["survivors_known"] else UNKNOWN_LABEL)
            for r in rows if r["agent"] == agent]


def _location_diff(rows: Sequence[Row], agent: str) -> tuple[PairedDifference, float]:
    """Paired unknown − known steps of ``agent`` (both successful, map as the unit) and the
    ratio of mean steps over the same pairs."""
    labelled = _location_rows(rows, agent)
    pd = paired_difference(labelled, "steps", UNKNOWN_LABEL, KNOWN_LABEL,
                           keys=LOCATION_PAIR_KEYS, cluster_keys=EXT_MAP_KEYS)
    known = {tuple(r[k] for k in LOCATION_PAIR_KEYS): r for r in labelled
             if r["agent"] == KNOWN_LABEL and r["success"]}
    num = den = 0.0
    for r in labelled:
        twin = known.get(tuple(r[k] for k in LOCATION_PAIR_KEYS))
        if r["agent"] == UNKNOWN_LABEL and r["success"] and twin is not None:
            num += r["steps"]
            den += twin["steps"]
    return pd, (num / den if den > 0 else math.nan)


def claim_x2(rows: Sequence[Row]) -> Hypothesis:
    """X2: paired unknown − known steps (``search_andor`` decides; ``search_astar`` reported)."""
    block = _block(rows, SEARCH_BLOCK)
    if len({r["survivors_known"] for r in block}) < 2:
        return _not_measured("X2", X2_CLAIM, f"block `{SEARCH_BLOCK}` lacks known and unknown "
                                             "runs")
    agents = [a for a in (SEARCH_ANDOR, SEARCH_ASTAR) if any(r["agent"] == a for r in block)]
    results = {a: _location_diff(block, a) for a in agents}
    if SEARCH_ANDOR not in results or results[SEARCH_ANDOR][0].n_pairs == 0:
        return _not_measured("X2", X2_CLAIM, f"no paired `{SEARCH_ANDOR}` runs")
    pd = results[SEARCH_ANDOR][0]
    if pd.lo > 0.0:
        verdict = VERDICT_PASS
    elif pd.hi < 0.0:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_WARN
    measured = "; ".join(f"`{a}` {_diff(d, 'steps')}, ratio of mean steps ×{_num(r)}"
                         for a, (d, r) in results.items())
    body = []
    for agent in agents:
        for scenario in _scenarios(block):
            own = _where(block, scenario=scenario)
            d, r = _location_diff(own, agent)
            body.append([f"`{agent}`", scenario, _diff(d, "steps"), f"×{_num(r)}"])
        for p in _distinct(block, "slip_p"):
            d, r = _location_diff(_where(block, slip_p=p), agent)
            body.append([f"`{agent}`", f"all, p = {p:g}", _diff(d, "steps"), f"×{_num(r)}"])
    detail = "\n".join([
        f"Block `{SEARCH_BLOCK}`: every map is run with the survivor location known (pre-marked, "
        "the core setting) and unknown, with the same outcome stream. Difference = steps(unknown) "
        "− steps(known) on the same map, slip p and agent, both missions successful; the map is "
        "the unit (per-map mean over p). Ratio = mean steps unknown / mean steps known over the "
        "same pairs.",
        "",
        f"Rule (on `{SEARCH_ANDOR}`): ✅ if the 95 % CI of the mean difference lies above 0; "
        "❌ if it lies below 0; ⚠️ otherwise.",
        "",
        _table(["agent", "scenario", "unknown − known steps (95 % CI)", "ratio"], body),
    ])
    return Hypothesis(X2_CLAIM, verdict, measured, detail, "X2")


# --- X3 --------------------------------------------------------------------------------------
X3_CLAIM = (f"the greedy nearest-first order is within "
            f"{_pct(EXT_CLAIM_TOUR_RATIO_MAX - 1.0)} of the optimal tour")


def claim_x3(rows: Sequence[Row]) -> Hypothesis:
    """X3: mean greedy / optimal true-map tour ratio per k (CI upper bound ≤ the maximum)."""
    block = _block(rows, MULTI_BLOCK)
    if not block:
        return _not_measured("X3", X3_CLAIM, f"no block `{MULTI_BLOCK}`")
    maps = unique_layouts(block)
    per_k: dict[int, tuple[float, float, float, list[float]]] = {}
    for k in _distinct(maps, "k"):
        ratios = [tour_ratio(r) for r in maps if r["k"] == k]
        ratios = [x for x in ratios if not math.isnan(x)]
        if ratios:
            per_k[k] = (*mean_ci(ratios), ratios)
    if not per_k:
        return _not_measured("X3", X3_CLAIM, "no finite tour ratios")
    limit = EXT_CLAIM_TOUR_RATIO_MAX
    # A k with fewer than 2 maps has no CI (lo = hi = NaN): it can neither pass nor fail.
    if any(lo > limit for _, lo, _, _ in per_k.values()):
        verdict = VERDICT_FAIL
    elif all(hi <= limit for _, _, hi, _ in per_k.values()):
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_WARN
    measured = "; ".join(f"k = {k}: ×{mean:.3f} [{lo:.3f}, {hi:.3f}]"
                         for k, (mean, lo, hi, _) in per_k.items())
    body = []
    for k, (mean, lo, hi, ratios) in per_k.items():
        runs = _where(block, k=k, agent=SEARCH_ANDOR)
        body.append([
            k, len(ratios), f"{mean:.3f} [{lo:.3f}, {hi:.3f}]", f"{max(ratios):.3f}",
            _pct(sum(x == 1.0 for x in ratios) / len(ratios)),
            _rate(_successes(runs), len(runs)) if runs else "n/a",
            _mean([steps_per_rescue(r) for r in runs]) if runs else "n/a",
        ])
    detail = "\n".join([
        f"Block `{MULTI_BLOCK}`: k survivors at known locations (the core map plus k − 1 extra "
        "survivors). Greedy tour = true-map BFS length of the nearest-first visiting order from "
        "the start; optimal tour = the shortest over all k! orders; both on the initial true map "
        "(the oracle the agent's greedy order approximates on its belief). One value per map; "
        "Student-t 95 % CI of the mean ratio. For k ≤ 2 greedy is optimal by construction "
        "(both orders share the survivor-to-survivor leg), so only k ≥ 3 can test the claim.",
        "",
        f"Rule: ✅ if the CI upper bound of the mean ratio is ≤ {limit:g} for every k; ❌ if the "
        f"CI lower bound exceeds {limit:g} for some k; ⚠️ otherwise (also when some k has fewer "
        "than 2 maps, so no CI).",
        "",
        _table(["k", "maps", "greedy / optimal (95 % CI)", "max ratio", "greedy optimal",
                f"`{SEARCH_ANDOR}` success", f"`{SEARCH_ANDOR}` steps per rescue"], body),
    ])
    return Hypothesis(X3_CLAIM, verdict, measured, detail, "X3")


# --- X4 --------------------------------------------------------------------------------------
X4_CLAIM = (f"with noisy sensing collisions become possible (bumps > 0) but success stays ≥ "
            f"{_pct(EXT_CLAIM_SUCCESS_TARGET)} at ε = {EXT_CLAIM_NOISE:g} (`{SEARCH_ANDOR}`)")


def _noise_table(block: Sequence[Row]) -> str:
    body = []
    for agent in _distinct(block, "agent"):
        for eps in _distinct(block, "false_negative"):
            own = _where(block, agent=agent, false_negative=eps)
            body.append([f"`{agent}`", f"{eps:g}", _rate(_successes(own), len(own)),
                         sum(r["bumps"] for r in own), sum(r["bumps"] > 0 for r in own),
                         _mean([float(r["bumps"]) for r in own]),
                         sum(r["unplanned_states"] for r in own),
                         _mean([float(r["steps"]) for r in own if r["success"]])])
    return _table(["agent", "ε", "success (Wilson 95 % CI)", "Σ bumps", "missions bumped",
                   "bumps per mission", "Σ unplanned", "steps (successful)"], body)


def claim_x4(rows: Sequence[Row]) -> Hypothesis:
    """X4: bumps appear only with ε > 0, and success at ε = EXT_CLAIM_NOISE stays high."""
    block = _block(rows, NOISY_BLOCK)
    own = [r for r in block if r["agent"] == SEARCH_ANDOR]
    at_eps = _where(own, false_negative=EXT_CLAIM_NOISE)
    if not at_eps:
        return _not_measured("X4", X4_CLAIM, f"no `{SEARCH_ANDOR}` runs at ε = "
                                             f"{EXT_CLAIM_NOISE:g} in block `{NOISY_BLOCK}`")
    noisy = [r for r in own if r["false_negative"] > 0.0]
    clean = [r for r in own if r["false_negative"] == 0.0]
    noisy_bumps, clean_bumps = sum(r["bumps"] for r in noisy), sum(r["bumps"] for r in clean)
    if noisy_bumps == 0 or clean_bumps > 0:
        part_a = VERDICT_FAIL  # a bump at eps = 0 breaks the core invariant: a bug, not noise
    elif not clean:
        part_a = VERDICT_WARN
    else:
        part_a = VERDICT_PASS
    ok, n = _successes(at_eps), len(at_eps)
    part_b = _rate_verdict(ok, n, EXT_CLAIM_SUCCESS_TARGET)
    if VERDICT_FAIL in (part_a, part_b):
        verdict = VERDICT_FAIL
    elif part_a == part_b == VERDICT_PASS:
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_WARN
    clean_text = f"{clean_bumps} in {len(clean)} runs at ε = 0" if clean else "no ε = 0 runs"
    measured = (f"collisions {part_a}: Σ bumps = {noisy_bumps} in {len(noisy)} runs at ε > 0 "
                f"({sum(r['bumps'] > 0 for r in noisy)} missions bumped) vs {clean_text}; "
                f"success at ε = {EXT_CLAIM_NOISE:g} {part_b}: {_rate(ok, n)}")
    detail = "\n".join([
        f"Block `{NOISY_BLOCK}`: the debris sensor reports a true DEBRIS cell as OPEN with "
        "probability ε (false negatives only); the belief is a Bayes map (`ProbBelief`) for ε > 0. "
        "A bump = entering debris the agent did not believe was there (D6); with ε = 0 and "
        "r ≥ 1 it is impossible (core invariant).",
        "",
        f"Rule: part (a) ✅ if `{SEARCH_ANDOR}` bumps at least once at ε > 0 and never at ε = 0 "
        "(❌ if it never bumps at ε > 0, or if it bumps at ε = 0, which would break the core "
        "invariant and point to a harness or agent bug; ⚠️ if there are no ε = 0 runs); part (b) "
        + _rate_rule(EXT_CLAIM_SUCCESS_TARGET)[len("Rule: "):]
        + " Overall ✅ only if both parts are ✅, ❌ if either is ❌, else ⚠️.",
        "",
        _noise_table(block),
    ])
    return Hypothesis(X4_CLAIM, verdict, measured, detail, "X4")


# --- X5 --------------------------------------------------------------------------------------
X5_CLAIM = (f"under noise `{SEARCH_ANDOR}` has at most {EXT_CLAIM_BUMP_MARGIN:g} more "
            f"collisions per mission than `{SEARCH_ASTAR}` (non-inferiority, every ε)")


def _bump_diff(rows: Sequence[Row]) -> PairedDifference:
    """Paired ``search_andor − search_astar`` bumps per mission, all missions, map as the unit."""
    return paired_difference(rows, "bumps", SEARCH_ANDOR, SEARCH_ASTAR, keys=EXT_TASK_KEYS,
                             require_success=False, cluster_keys=EXT_MAP_KEYS)


def claim_x5(rows: Sequence[Row]) -> Hypothesis:
    """X5: paired ``search_andor − search_astar`` bumps at ε > 0 (map as the unit), pooled over
    ε and at every ε: a pooled mean must not hide a disadvantage at one noise level."""
    block = [r for r in _block(rows, NOISY_BLOCK) if r["false_negative"] > 0.0]
    pd = _bump_diff(block)
    if pd.n_pairs == 0:
        return _not_measured("X5", X5_CLAIM, f"no paired runs at ε > 0 in block `{NOISY_BLOCK}`")
    margin = EXT_CLAIM_BUMP_MARGIN
    per_eps = {eps: _bump_diff(_where(block, false_negative=eps))
               for eps in _distinct(block, "false_negative")}
    measured_eps = [d for d in per_eps.values() if d.n_pairs > 0]
    if pd.lo > 0.0 or any(d.lo > 0.0 for d in measured_eps):
        verdict = VERDICT_FAIL
    elif pd.hi <= margin and all(d.hi <= margin for d in measured_eps):
        verdict = VERDICT_PASS
    else:
        verdict = VERDICT_WARN
    totals = {a: sum(r["bumps"] for r in block if r["agent"] == a)
              for a in (SEARCH_ANDOR, SEARCH_ASTAR)}
    worst = max(measured_eps, key=lambda d: (not math.isnan(d.hi), d.hi))
    worst_eps = next(e for e, d in per_eps.items() if d is worst)
    measured = (f"pooled {_diff(pd, 'bumps per mission')}; largest per-ε upper bound "
                f"{_num(worst.hi, True)} at ε = {worst_eps:g}; Σ bumps {totals[SEARCH_ANDOR]} vs "
                f"{totals[SEARCH_ASTAR]}")
    body = []
    for eps, d in per_eps.items():
        sub = _where(block, false_negative=eps)
        body.append([f"{eps:g}", _diff(d, "bumps"),
                     sum(r["bumps"] for r in sub if r["agent"] == SEARCH_ANDOR),
                     sum(r["bumps"] for r in sub if r["agent"] == SEARCH_ASTAR)])
    detail = "\n".join([
        f"Block `{NOISY_BLOCK}`, ε > 0: both agents run every map with the same outcome and noise "
        "streams. Difference = bumps(search_andor) − bumps(search_astar) per mission, all "
        "missions; pooled over ε with the map as the unit (per ε each map appears once), and "
        "separately at every ε.",
        "",
        f"Rule (non-inferiority, margin {margin:g} bumps per mission, an absolute margin fixed "
        "before the full run): ❌ if the 95 % CI lies above 0 (significantly more collisions), "
        f"pooled or at any ε; ✅ if the CI upper bound is ≤ {margin:g}, pooled and at every ε; "
        "⚠️ otherwise. ✅ means \"not worse by more than the margin\", not \"no more "
        "collisions\".",
        "",
        _table(["ε", "search_andor − search_astar (95 % CI)", f"Σ bumps `{SEARCH_ANDOR}`",
                f"Σ bumps `{SEARCH_ASTAR}`"], body),
    ])
    return Hypothesis(X5_CLAIM, verdict, measured, detail, "X5")


# --- report ----------------------------------------------------------------------------------
CLAIMS: tuple[Callable[[Sequence[Row]], Hypothesis], ...] = (
    claim_x1, claim_x2, claim_x3, claim_x4, claim_x5,
)


def evaluate_claims(rows: Sequence[Row]) -> list[Hypothesis]:
    """X1–X5 on typed raw rows, in order."""
    return [claim(rows) for claim in CLAIMS]


def _header(rows: Sequence[Row], meta: dict[str, Any]) -> list[str]:
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    lines = [
        f"# Extension claims — `{meta.get('config_name', 'unknown configuration')}`",
        "",
        f"- Date of run: {meta.get('date', 'unknown (no run metadata)')}",
        f"- Missions (rows of raw.csv): {len(rows)}; agents: "
        f"{', '.join(f'`{a}`' for a in agents) or 'none'}",
    ]
    fields: list[tuple[str, str, Callable[[Any], str]]] = [
        ("Config file", "config_path", lambda v: f"`{v}`"),
        ("Base seed", "base_seed", str),
        ("Tasks (maps × conditions)", "n_tasks", str),
        ("Sample (--max-tasks, whole maps)", "max_tasks", str),
        ("Workers", "workers", str),
        ("Wall time", "wall_time_s", lambda v: f"{float(v):.1f} s"),
        ("Git commit", "git_commit", lambda v: f"`{v}`"),
    ]
    for title, key, fmt in fields:
        if meta.get(key) is not None:
            lines.append(f"- {title}: {fmt(meta[key])}")
    lines += [
        "- Methods: success rates with Wilson 95 % intervals; means with Student-t 95 % "
        "intervals; paired comparisons on identical maps and outcome / noise streams (a task "
        "seed depends only on base seed, scenario, size and replicate) with a paired t-test and "
        "a Wilcoxon signed-rank test; pooled paired comparisons use the map as the unit "
        "(per-map mean difference; n = maps, not pairs). The noise draws are paired per (step, "
        "cell): one uniform per grid cell per step, the same in every mission on that map.",
        f"- Verdicts: {VERDICT_PASS} supported, {VERDICT_FAIL} refuted, {VERDICT_WARN} "
        "inconclusive or not measured. Thresholds are constants in `rescue/config.py` fixed "
        "before the full run; no experiment was tuned toward a verdict.",
    ]
    return lines


def render_markdown(rows: Sequence[Row], claims: Sequence[Hypothesis],
                    meta: dict[str, Any] | None = None) -> str:
    """EXTENSIONS.md text: header, claims table, per-claim details."""
    lines = _header(rows, meta or {})
    lines += ["", "## Claims", "",
              _table(["#", "Claim", "Verdict", "Measured"],
                     [[c.label, c.claim, c.verdict, c.measured] for c in claims]),
              "", "## Details"]
    for c in claims:
        lines += ["", f"### {c.label}. {c.verdict} {c.claim}", "", c.detail]
    return "\n".join(lines) + "\n"


def write_extensions(rows: Sequence[Row], path: str | Path,
                     meta_path: str | Path | None = None) -> Path:
    """Evaluate X1–X5 and write the Markdown report (metadata from ``run_meta.json`` next to
    ``path`` unless ``meta_path`` is given)."""
    out = Path(path)
    source = Path(meta_path) if meta_path is not None else out.parent / META_FILE
    try:
        meta = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows, evaluate_claims(rows), meta), encoding="utf-8",
                   newline="\n")
    return out
