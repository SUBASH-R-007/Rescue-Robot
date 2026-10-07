"""Figures of the extension harness → ``results/ext/plots/`` (plan §3.14).

* ``search_steps.png`` — mean steps with the survivor location known vs unknown, one panel
  per scenario, ``search_andor`` vs ``search_astar`` (successful missions averaged per map
  first, so the map is the unit of the Student-t 95 % CI);
* ``multi_tours.png`` — greedy / optimal true-map tour ratio vs k (one value per map) and steps
  per rescue vs k (successful missions; the optimal tour per survivor as a reference);
* ``noisy.png`` — bumps per mission and success % vs the false-negative rate ε.

Drawn with the core harness's style and helpers (:mod:`rescue.experiments.plots`: header band,
legend, dodged CI bars, colours). The extension agents borrow the colour and marker of their
core counterpart (``search_andor`` ← ``andor``, ``search_astar`` ← ``astar_replan``); they are
registered in the core style table only while an extension figure is drawn.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rescue.config import EXT_CLAIM_TOUR_RATIO_MAX
from rescue.experiments import plots as core
from rescue.experiments.stats import mean_ci
from rescue.ext.experiments import (
    EXT_MAP_KEYS,
    MULTI_BLOCK,
    NOISY_BLOCK,
    SEARCH_BLOCK,
    steps_per_rescue,
    tour_ratio,
    unique_layouts,
)

Row = dict[str, Any]

#: Extension agent → the core agent whose colour and marker it uses, and its legend label.
EXT_STYLE_SOURCE: dict[str, tuple[str, str]] = {
    "search_andor": ("andor", "AND-OR search (search_andor)"),
    "search_astar": ("astar_replan", "A* + replan search (search_astar)"),
}
#: Agent order in legends and dodge slots.
EXT_AGENT_ORDER: tuple[str, ...] = tuple(EXT_STYLE_SOURCE)
#: File names.
SEARCH_FILE: str = "search_steps.png"
MULTI_FILE: str = "multi_tours.png"
NOISY_FILE: str = "noisy.png"
#: Location labels of the search figure (x axis order).
LOCATION_LABELS: tuple[str, str] = ("known", "unknown")
#: Dash pattern of the reference lines (tour-ratio limit, optimal tour per survivor).
REFERENCE_DASH: tuple[int, tuple[float, float]] = (0, (3.0, 2.0))

__all__ = ["make_ext_plots", "plot_multi_tours", "plot_noisy", "plot_search_steps"]


@contextmanager
def _ext_styles() -> Iterator[None]:
    """Temporarily add the extension agents to the core style table."""
    added = []
    for agent, (source, label) in EXT_STYLE_SOURCE.items():
        if agent not in core.AGENT_STYLE:
            core.AGENT_STYLE[agent] = core.AGENT_STYLE[source]._replace(label=label)
            added.append(agent)
    try:
        yield
    finally:
        for agent in added:
            core.AGENT_STYLE.pop(agent, None)


def _agents(rows: Sequence[Row]) -> list[str]:
    """Agents present, in legend order."""
    return core._ordered((r["agent"] for r in rows), EXT_AGENT_ORDER)


def _draw_order(agents: Sequence[str]) -> list[str]:
    """Drawing (and dodge) order: the proposed ``search_andor`` last, so it stays visible
    where the series coincide (the core draws only its own ``andor`` on top)."""
    return sorted(agents, key=lambda a: a == EXT_AGENT_ORDER[0])


def _block(rows: Sequence[Row], name: str) -> list[Row]:
    return [r for r in rows if r["experiment"] == name]


def _fixed_text(rows: Sequence[Row]) -> str:
    """Sizes and slip p of a figure's rows (pooled or fixed)."""
    sizes = core._sizes(rows)
    ps = core._distinct(rows, "slip_p")
    size_text = f"sizes {', '.join(sizes)} pooled" if len(sizes) > 1 else f"{sizes[0]} maps"
    p_text = (f"p ∈ {{{core._fmt_set(ps)}}} pooled" if len(ps) > 1
              else f"p{core.EQ}{core._fmt_num(ps[0])}")
    return f"{size_text}; {p_text}"


def _reference(ax: Any, y: float, label: str) -> None:
    """A dashed neutral horizontal reference line with a small right-aligned label."""
    ax.axhline(y, color=core.ZERO_LINE_COLOUR, linewidth=core.ZERO_LINE_WIDTH,
               linestyle=REFERENCE_DASH, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom", color=core.MUTED_TEXT,
                fontsize=core.TICK_SIZE)


def _per_map_steps(rows: Sequence[Row]) -> list[Row]:
    """One row per (agent, location, map): mean steps of its successful missions (pooled over
    slip p), so a map run at several p counts once in a CI. Maps without a success drop out."""
    groups: dict[tuple[Any, ...], list[Row]] = {}
    for r in rows:
        key = (r["agent"], r["location"], *(r[k] for k in EXT_MAP_KEYS))
        groups.setdefault(key, []).append(r)
    out: list[Row] = []
    for group in groups.values():
        steps = [float(r["steps"]) for r in group if r["success"]]
        if steps:
            out.append(dict(group[0], steps=sum(steps) / len(steps), success=True))
    return out


def _label_series(ax: Any, points: Sequence[Any], pos: dict[Any, float], label: str) -> None:
    """A small grey label right of a neutral series' last point."""
    if not points:
        return
    last = points[-1]
    ax.annotate(label, xy=(pos[last.x], last.y), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", color=core.MUTED_TEXT, fontsize=core.TICK_SIZE)


# --- figures ---------------------------------------------------------------------------------
def plot_search_steps(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Mean steps (successful missions) with the location known vs unknown, per scenario."""
    block = _block(rows, SEARCH_BLOCK)
    if not block:
        return None
    labelled = [dict(r, location=LOCATION_LABELS[0] if r["survivors_known"]
                     else LOCATION_LABELS[1]) for r in block]
    with _ext_styles():
        agents = _agents(block)
        scenarios = core._scenarios(block)
        fig, axes = core._new_figure(len(scenarios), sharey=False)
        counts: list[int] = []
        drawn = []
        for ax, scenario in zip(axes, scenarios, strict=True):
            subset = [r for r in labelled if r["scenario"] == scenario]
            series = core._mean_points(_per_map_steps(subset), "location", "steps",
                                       successful_only=True)
            counts.extend(p.n for points in series.values() for p in points)
            x_values = [v for v in LOCATION_LABELS if any(r["location"] == v for r in subset)]
            drawn.append(core._draw_agents(ax, series, _draw_order(agents), x_values,
                                           dodge=core.DODGE_FRACTION * 2))
            core._zero_floor(ax)
            ax.set_title(core._scenario_title(scenario), color=core.TEXT_COLOUR,
                         fontsize=core.PANEL_TITLE_SIZE, loc="left")
        merged = core._merge(drawn)
        subtitle = (f"{core._n_text(counts, 'maps per point')} (steps of successful "
                    f"missions averaged per map first); {core.MEAN_CI_TEXT}; unpaired view, the "
                    "paired unknown − known test is X2; unknown = the belief starts all UNKNOWN "
                    f"and the agent explores frontiers until it senses the survivor; "
                    f"{_fixed_text(block)}")
        return core._finish(fig, axes, "Cost of an unknown survivor location", subtitle,
                            [a for a in agents if a in merged.agents], "survivor location",
                            "steps per mission", out_dir / SEARCH_FILE)


def _ratio_points(block: Sequence[Row]) -> list[Any]:
    """Mean greedy / optimal tour ratio per k, one value per map (survivor layout)."""
    layouts = unique_layouts(block)
    points = []
    for k in core._distinct(block, "k"):
        values = [tour_ratio(r) for r in layouts if r["k"] == k]
        values = [v for v in values if not math.isnan(v)]
        if values:
            mean, lo, hi = mean_ci(values)
            points.append(core._Point(k, mean, lo, hi, len(values)))
    return points


def _optimal_per_survivor(block: Sequence[Row]) -> list[Any]:
    """Mean optimal true-map tour per survivor (``optimal_tour / k``) per k, one per map."""
    layouts = unique_layouts(block)
    points = []
    for k in core._distinct(block, "k"):
        values = [float(r["optimal_tour"]) / k for r in layouts if r["k"] == k]
        values = [v for v in values if not math.isnan(v)]
        if values:
            mean, lo, hi = mean_ci(values)
            points.append(core._Point(k, mean, lo, hi, len(values)))
    return points


def plot_multi_tours(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Greedy / optimal tour ratio vs k, and steps per rescue vs k."""
    block = _block(rows, MULTI_BLOCK)
    if not block:
        return None
    ks = core._distinct(block, "k")
    with _ext_styles():
        agents = _agents(block)
        fig, (ax_ratio, ax_steps) = core._new_figure(2, sharey=False)
        pos = core._positions(ks)
        ratio = _ratio_points(block)
        ax_ratio.axhline(1.0, color=core.ZERO_LINE_COLOUR, linewidth=core.ZERO_LINE_WIDTH,
                         zorder=1)
        _reference(ax_ratio, EXT_CLAIM_TOUR_RATIO_MAX,
                   f"claim limit ×{EXT_CLAIM_TOUR_RATIO_MAX:g}")
        core._draw_series(ax_ratio, ratio, pos, 0.0, core.NEUTRAL_SERIES, "o", "-")
        _label_series(ax_ratio, ratio, pos, "mean ratio")
        ax_ratio.set_xticks([pos[k] for k in ks])
        ax_ratio.set_xticklabels([core._fmt_num(k) for k in ks])
        top = max([EXT_CLAIM_TOUR_RATIO_MAX, *(p.hi for p in ratio if math.isfinite(p.hi))])
        ax_ratio.set_ylim(1.0 - (top - 1.0) * 0.15, 1.0 + (top - 1.0) * 1.25)
        ax_ratio.set_title("greedy / optimal tour (true map)", color=core.TEXT_COLOUR,
                           fontsize=core.PANEL_TITLE_SIZE, loc="left")
        ax_ratio.set_ylabel("tour length ratio", color=core.TEXT_COLOUR,
                            fontsize=core.LABEL_SIZE)

        labelled = [dict(r, steps_per_rescue=steps_per_rescue(r)) for r in block]
        series = core._mean_points(labelled, "k", "steps_per_rescue", successful_only=True)
        drawn = core._draw_agents(ax_steps, series, _draw_order(agents), ks)
        optimal = _optimal_per_survivor(block)
        core._draw_series(ax_steps, optimal, pos, 0.0, core.NEUTRAL_SERIES, "_", REFERENCE_DASH)
        _label_series(ax_steps, optimal, pos, "optimal tour / k\n(oracle, no slips)")
        core._zero_floor(ax_steps)
        ax_steps.set_title("steps per rescue", color=core.TEXT_COLOUR,
                           fontsize=core.PANEL_TITLE_SIZE, loc="left")
        ax_steps.set_ylabel("steps / k (successful missions)", color=core.TEXT_COLOUR,
                            fontsize=core.LABEL_SIZE)
        counts = [p.n for p in ratio]
        subtitle = (f"left: greedy nearest-first vs best-order tour on the true map, one "
                    f"value per map ({core._n_text(counts, 'maps per k')}); grey = reference "
                    f"series (labelled); {core.MEAN_CI_TEXT}; survivors known; "
                    f"{core._scope_text(block)}; {_fixed_text(block)}")
        return core._finish(fig, [ax_ratio, ax_steps], "Several survivors: visiting order",
                            subtitle, [a for a in agents if a in drawn.agents],
                            "survivors k", None, out_dir / MULTI_FILE)


def plot_noisy(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Bumps per mission and success % vs the false-negative rate ε."""
    block = _block(rows, NOISY_BLOCK)
    if not block:
        return None
    eps = core._distinct(block, "false_negative")
    with _ext_styles():
        agents = _agents(block)
        fig, (ax_bumps, ax_success) = core._new_figure(2, sharey=False)
        bumps = core._mean_points(block, "false_negative", "bumps", successful_only=False)
        success = core._success_points(block, "false_negative")
        drawn = core._merge([core._draw_agents(ax_bumps, bumps, _draw_order(agents), eps),
                             core._draw_agents(ax_success, success, _draw_order(agents), eps)])
        core._zero_floor(ax_bumps)
        core._percent_axis(ax_success)
        ax_bumps.set_ylabel("bumps per mission", color=core.TEXT_COLOUR,
                            fontsize=core.LABEL_SIZE)
        ax_success.set_ylabel(core.SUCCESS_LABEL, color=core.TEXT_COLOUR,
                              fontsize=core.LABEL_SIZE)
        ax_bumps.set_title("collisions (unexpected debris)", color=core.TEXT_COLOUR,
                           fontsize=core.PANEL_TITLE_SIZE, loc="left")
        ax_success.set_title("rescue success", color=core.TEXT_COLOUR,
                             fontsize=core.PANEL_TITLE_SIZE, loc="left")
        counts = [p.n for points in success.values() for p in points]
        subtitle = (f"{core._n_text(counts, 'missions per agent and point')}; left: all "
                    f"missions, {core.MEAN_CI_TEXT}; right: {core.SUCCESS_CI_TEXT}; the sensor "
                    "misses a debris cell with probability ε (no false alarms); survivor "
                    f"location known; {core._scope_text(block)}; {_fixed_text(block)}")
        return core._finish(fig, [ax_bumps, ax_success], "Noisy sensing", subtitle,
                            [a for a in agents if a in drawn.agents],
                            "false-negative rate ε", None, out_dir / NOISY_FILE)


def make_ext_plots(rows: Sequence[Row], out_dir: str | Path) -> list[Path]:
    """Write every extension figure the rows support; return the paths written.

    ``out_dir`` is the plots directory itself (or a results directory, whose ``plots/`` is
    used). Figures whose block is missing are skipped.
    """
    rows = list(rows)
    if not rows:
        return []
    target = core.plots_dir(out_dir)
    paths = [plot_search_steps(rows, target), plot_multi_tours(rows, target),
             plot_noisy(rows, target)]
    return [p for p in paths if p is not None]
