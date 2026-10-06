"""Report figures of the Monte Carlo harness (brief §6.6, docs/plan.md Phase 6).

:func:`make_all_plots` turns the typed rows of ``results/raw.csv`` (:func:`stats.read_raw`)
into PNGs under ``results/plots/``:

* ``success_vs_p.png`` - success % vs slip p, one panel per scenario (shared 0-100 % axis),
  Wilson 95 % error bars;
* ``steps_vs_p.png`` and ``wasted_moves_vs_p.png`` - two separate charts (never a dual axis),
  successful missions only, Student-t 95 % error bars;
* ``replans_vs_p.png``, ``cells_explored_vs_p.png`` - all missions (hollow markers where
  fewer than half of the missions succeeded); the four mean charts show one grid size when the
  slip block has several (the middle one, stated in the subtitle);
* ``planning_time_vs_size.png`` - planning ms per mission vs grid size (scaling block, log y,
  geometric mean with a log-scale t-interval; the worker count of the run is quoted);
* ``success_under_aftershocks.png`` - success % vs aftershock probability q;
* ``paired_wasted_moves_diff.png`` / ``paired_replans_diff.png`` - paired
  ``andor − astar_replan`` differences (H3 / H5) with their 95 % CI vs p (common random
  numbers: identical maps and outcome streams);
* ``sensor_radius.png`` / ``replan_mode.png`` - only when those blocks exist; one row of
  panels per scenario;
* ``tiebreak_ablation.png`` - H7: paired wasted-move differences vs p across the slip sweep and
  the ``tiebreak_ablation`` block (matched tie-breaks and each tie-break's own effect), map as
  the unit; only when the ablation block exists.

Blocks are found by the factor they sweep (a block name containing a keyword such as
``slip`` / ``scal`` / ``aftershock`` wins ties), and the factors a block does not sweep are held
at their defaults. Every plot degrades gracefully: when its block or its agents are missing it
is skipped and no path is returned. Each agent keeps the same colour **and** marker in every
figure (:data:`AGENT_STYLE`); text is dark grey, never a series colour. Rendering uses
matplotlib's Agg canvas directly (no pyplot state).
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from rescue.agents import ALL_AGENT_NAMES
from rescue.config import DEFAULT_SENSOR_RADIUS
from rescue.experiments.run import read_run_meta
from rescue.experiments.spec import TIEBREAK_ABLATION_EXPERIMENT
from rescue.experiments.stats import TASK_KEYS, mean_ci, paired_difference, wilson_interval
from rescue.mapgen import SCENARIOS

Row = dict[str, Any]


# --- style -------------------------------------------------------------------------------------
class AgentStyle(NamedTuple):
    """How one agent is drawn: colour, marker shape, legend label and line style."""

    colour: str
    marker: str
    label: str
    linestyle: Any = "-"


#: Dash pattern of the stretch agents (a second channel besides colour and marker).
STRETCH_DASH: tuple[int, tuple[float, float]] = (0, (4.0, 1.6))

#: Fixed colour + marker per agent - identical in every plot and in the UI (``viz.theme``);
#: the colour follows the agent, never its rank.
AGENT_STYLE: dict[str, AgentStyle] = {
    "andor": AgentStyle("#2a78d6", "o", "AND-OR (andor)"),
    "astar_fixed": AgentStyle("#eb6834", "s", "A* no replan (astar_fixed)"),
    "astar_replan": AgentStyle("#1baf7a", "^", "A* + replan (astar_replan)"),
    "dstar_lite": AgentStyle("#eda100", "D", "D* Lite (dstar_lite)", STRETCH_DASH),
    "vi_expected": AgentStyle("#e87ba4", "v", "VI expected steps (vi_expected)", STRETCH_DASH),
}
#: Unknown agents: neutral grey (no generated hues) with spare marker shapes.
FALLBACK_COLOUR: str = "#7b8494"
FALLBACK_MARKERS: tuple[str, ...] = ("P", "X", "*", "h", "p")

#: Output sub-directory and resolution.
PLOTS_DIR_NAME: str = "plots"
DPI: int = 150

#: Ink and chrome (light mode).
BACKGROUND: str = "#ffffff"
TEXT_COLOUR: str = "#2f3640"
MUTED_TEXT: str = "#5c6470"
GRID_COLOUR: str = "#e3e6ea"
SPINE_COLOUR: str = "#b9bfc8"
ZERO_LINE_COLOUR: str = "#5c6470"
#: Colour of a single non-agent series (a paired difference belongs to no agent).
NEUTRAL_SERIES: str = "#3f4754"

#: Mark specs (points): ~2 px lines, >= 6 pt markers with a white ring, thin CI bars.
LINE_WIDTH: float = 1.6
MARKER_SIZE: float = 6.5
MARKER_RING: float = 0.9
ERR_LINE_WIDTH: float = 1.0
ERR_CAP_SIZE: float = 2.5
ERR_ALPHA: float = 0.75
GRID_WIDTH: float = 0.6
ZERO_LINE_WIDTH: float = 1.0

#: Font sizes (points).
TITLE_SIZE: float = 11.5
SUBTITLE_SIZE: float = 9.0
PANEL_TITLE_SIZE: float = 10.0
LABEL_SIZE: float = 9.5
TICK_SIZE: float = 8.5
LEGEND_SIZE: float = 9.0

#: Small multiples: panel size (inches) and the most panels per row.
PANEL_SIZE_IN: tuple[float, float] = (3.7, 2.8)
SINGLE_PANEL_SIZE_IN: tuple[float, float] = (6.4, 4.0)
MAX_PANEL_COLUMNS: int = 3
#: Header band above the panels (inches): outer padding, title line, subtitle line, legend
#: side margin of the wrapped subtitle; subtitle characters per inch and line spacing.
HEADER_PAD_IN: float = 0.12
TITLE_LINE_IN: float = 0.30
SUBTITLE_LINE_IN: float = 0.19
HEADER_SIDE_IN: float = 0.3
SUBTITLE_CHARS_PER_INCH: float = 14.5
SUBTITLE_LINE_SPACING: float = 1.35
#: Means over fewer missions than this are not drawn (their t-interval would dominate the
#: axis); the subtitle says so.
MIN_MEAN_N: int = 5
#: Largest share of the figure width the legend may take before it wraps to more rows.
LEGEND_MAX_WIDTH_FRACTION: float = 0.96
#: Horizontal dodge between agents, as a fraction of the smallest gap between x values.
DODGE_FRACTION: float = 0.07
#: Smaller dodge for the grid-size axis: points must stay on their size (log-y CIs rarely
#: overlap, and the marker shapes already tell the agents apart).
SIZE_DODGE_FRACTION: float = DODGE_FRACTION / 3.0
#: Means over all missions: a point whose missions succeeded less often than this is drawn with
#: a hollow marker (its mean mostly reflects missions abandoned early, not frugality).
LOW_SUCCESS_FRACTION: float = 0.5
#: The proposed agent is drawn last, so its line is never hidden under coinciding series.
PROPOSED_AGENT: str = "andor"
#: Success axis: limits (with headroom so 0 % / 100 % markers are not clipped) and ticks.
PERCENT: float = 100.0
SUCCESS_YLIM: tuple[float, float] = (-4.0, 104.0)
SUCCESS_TICKS: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
#: x padding: share of the x span (numeric axes) or category units (categorical axes).
X_PAD_FRACTION: float = 0.06
CATEGORY_PAD: float = 0.5
#: Zero-based axes start this share of the axis height below 0, so markers at 0 are whole.
ZERO_PAD_FRACTION: float = 0.035
#: Non-breaking "=" for subtitles (``n = 20`` never wraps between its parts).
EQ: str = "\u00a0=\u00a0"
#: Labelled ticks per decade on the log-scale planning-time axis (1, 2, 5, 10, 20, ...).
LOG_TICK_SUBS: tuple[float, ...] = (1.0, 2.0, 5.0)

#: Factors held at these values when a block does not sweep them.
BASELINE: dict[str, Any] = {
    "sensor_radius": DEFAULT_SENSOR_RADIUS,
    "replan_mode": "eager",
    "aftershock_q": 0.0,
    "planner": "",
}
#: The paired comparison of the H3 claim.
PAIRED_AGENTS: tuple[str, str] = ("andor", "astar_replan")
ALL_SCENARIOS_LABEL: str = "all scenarios (pooled)"


class _Block(NamedTuple):
    """How to find a block: the factor it sweeps and name keywords preferred on ties."""

    factor: str
    keywords: tuple[str, ...]


SLIP_BLOCK = _Block("slip_p", ("slip", "main", "sweep"))
SIZE_BLOCK = _Block("rows", ("scal", "size"))
AFTERSHOCK_BLOCK = _Block("aftershock_q", ("aftershock", "stress"))
SENSOR_BLOCK = _Block("sensor_radius", ("sensor", "radius"))
MODE_BLOCK = _Block("replan_mode", ("mode", "lazy", "eager"))


class _Point(NamedTuple):
    x: Any
    y: float
    lo: float
    hi: float
    n: int
    #: Share of the point's missions that succeeded (NaN when not tracked).
    success: float = math.nan


class _Drawn(NamedTuple):
    """What a panel (or a whole figure) actually drew."""

    agents: frozenset[str]
    hollow: bool


# --- small helpers -----------------------------------------------------------------------------
def agent_style(agent: str, agents: Sequence[str] = ()) -> AgentStyle:
    """Style of ``agent``; unknown agents get the neutral grey and a spare marker."""
    style = AGENT_STYLE.get(agent)
    if style is not None:
        return style
    unknown = [a for a in agents if a not in AGENT_STYLE]
    index = unknown.index(agent) if agent in unknown else 0
    return AgentStyle(FALLBACK_COLOUR, FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)], agent)


def _num(value: Any) -> float:
    """``value`` as a float (NaN for None / empty / unparsable)."""
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _ok(row: Row) -> bool:
    """Whether the row's mission succeeded (bools or their text forms)."""
    value = row.get("success")
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _ordered(values: Iterable[str], order: Sequence[str]) -> list[str]:
    """Distinct ``values`` in ``order`` first, then the rest sorted."""
    present = set(values)
    return [v for v in order if v in present] + sorted(present - set(order))


def _agents(rows: Sequence[Row]) -> list[str]:
    return _ordered((r["agent"] for r in rows), ALL_AGENT_NAMES)


def _scenarios(rows: Sequence[Row]) -> list[str]:
    return _ordered((r["scenario"] for r in rows), tuple(SCENARIOS))


def _distinct(rows: Sequence[Row], field: str) -> list[Any]:
    return sorted({r[field] for r in rows})


def _fmt_num(value: Any) -> str:
    """Compact number for titles and ticks (0.1, 0.05, 2)."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_set(values: Sequence[Any]) -> str:
    return ", ".join(_fmt_num(v) for v in values)


def _size_label(rows: int, cols: int) -> str:
    return f"{rows}×{cols}"


def _sizes(rows: Sequence[Row]) -> list[str]:
    return [_size_label(r, c) for r, c in sorted({(r["rows"], r["cols"]) for r in rows})]


def _scenario_title(name: str) -> str:
    return name.replace("_", " ")


def _n_text(counts: Iterable[int], unit: str) -> str:
    """``n = 60 <unit>`` or ``n = 12–60 <unit>`` (the range over the plotted points)."""
    values = sorted(set(counts))
    if not values:
        return f"n{EQ}0 {unit}"
    if len(values) == 1:
        return f"n{EQ}{values[0]} {unit}"
    return f"n{EQ}{values[0]}–{values[-1]} {unit}"


def _pooled_text(rows: Sequence[Row], swept: str) -> str:
    """Which conditions a figure pools (or fixes), e.g. ``sizes 10×10, 15×15 pooled``."""
    parts: list[str] = []
    sizes = _sizes(rows)
    if swept != "rows":
        parts.append(f"sizes {', '.join(sizes)} pooled" if len(sizes) > 1 else f"{sizes[0]} maps")
    if swept != "slip_p":
        ps = _distinct(rows, "slip_p")
        parts.append(f"p ∈ {{{_fmt_set(ps)}}} pooled" if len(ps) > 1
                     else f"p{EQ}{_fmt_num(ps[0])}")
    for field, label in (("sensor_radius", "r"), ("aftershock_q", "q")):
        if swept != field:
            values = _distinct(rows, field)
            if len(values) == 1 and values[0] != BASELINE[field]:
                parts.append(f"{label}{EQ}{_fmt_num(values[0])}")
    if swept != "replan_mode":
        modes = _distinct(rows, "replan_mode")
        if len(modes) == 1 and modes[0] != BASELINE["replan_mode"]:
            parts.append(f"{modes[0]} replanning")
    return "; ".join(parts)


# --- block selection ---------------------------------------------------------------------------
def _select_block(rows: Sequence[Row], block: _Block) -> list[Row]:
    """Rows of the experiment that sweeps ``block.factor`` (empty if none does).

    Candidates are experiments with >= 2 values of the factor; a name containing one of the
    keywords wins, then the most values, then the most rows. Other factors the chosen block
    happens to vary are held at :data:`BASELINE` (or their most common value).
    """
    by_experiment: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_experiment[str(row.get("experiment", ""))].append(row)
    candidates = [
        (name, block_rows)
        for name, block_rows in by_experiment.items()
        if len({r[block.factor] for r in block_rows}) >= 2
    ]
    if not candidates:
        return []

    def rank(item: tuple[str, list[Row]]) -> tuple[Any, ...]:
        name, block_rows = item
        named = any(k in name.lower() for k in block.keywords)
        return (not named, -len({r[block.factor] for r in block_rows}), -len(block_rows), name)

    chosen = sorted(candidates, key=rank)[0][1]
    for field, default in BASELINE.items():
        if field == block.factor or len({r[field] for r in chosen}) < 2:
            continue
        keep = default if any(r[field] == default for r in chosen) else (
            Counter(r[field] for r in chosen).most_common(1)[0][0]
        )
        chosen = [r for r in chosen if r[field] == keep]
    return chosen


# --- aggregation -------------------------------------------------------------------------------
def _group(rows: Sequence[Row], x_field: str) -> dict[str, dict[Any, list[Row]]]:
    """agent → x value → rows."""
    out: dict[str, dict[Any, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        out[row["agent"]][row[x_field]].append(row)
    return out


def _success_points(rows: Sequence[Row], x_field: str) -> dict[str, list[_Point]]:
    """Success % with its Wilson 95 % interval per agent and x value."""
    out: dict[str, list[_Point]] = {}
    for agent, by_x in _group(rows, x_field).items():
        points = []
        for x in sorted(by_x):
            group = by_x[x]
            k = sum(_ok(r) for r in group)
            lo, hi = wilson_interval(k, len(group))
            points.append(_Point(x, PERCENT * k / len(group), PERCENT * lo, PERCENT * hi,
                                 len(group)))
        out[agent] = points
    return out


def _mean_points(
    rows: Sequence[Row],
    x_field: str,
    metric: str,
    successful_only: bool,
    geometric: bool = False,
) -> dict[str, list[_Point]]:
    """Mean of ``metric`` with its Student-t 95 % interval per agent and x value.

    x values with fewer than :data:`MIN_MEAN_N` (successful) missions are left out. With
    ``geometric`` the mean and the t-interval are computed on log values and back-transformed
    (geometric mean; always positive, so it suits a log axis); values <= 0 are dropped.
    Each point records the share of its missions that succeeded.
    """
    out: dict[str, list[_Point]] = {}
    for agent, by_x in _group(rows, x_field).items():
        points = []
        for x in sorted(by_x):
            group = by_x[x]
            values = [_num(r.get(metric)) for r in group if not successful_only or _ok(r)]
            values = [v for v in values if not math.isnan(v) and (v > 0 or not geometric)]
            if len(values) < MIN_MEAN_N:
                continue
            if geometric:
                mean, lo, hi = (math.exp(v) if math.isfinite(v) else v
                                for v in mean_ci([math.log(v) for v in values]))
            else:
                mean, lo, hi = mean_ci(values)
            success = sum(_ok(r) for r in group) / len(group)
            points.append(_Point(x, mean, lo, hi, len(values), success))
        out[agent] = points
    return out


def _paired_points(rows: Sequence[Row], x_field: str, metric: str) -> list[_Point]:
    """Paired ``andor − astar_replan`` difference of ``metric`` per x value.

    Each map appears once per x value, so the pairs of one point are independent.
    """
    a, b = PAIRED_AGENTS
    out = []
    by_x: dict[Any, list[Row]] = defaultdict(list)
    for row in rows:
        by_x[row[x_field]].append(row)
    for x in sorted(by_x):
        diff = paired_difference(by_x[x], metric, a, b, TASK_KEYS)
        mean, lo, hi, n_pairs = diff[0], diff[1], diff[2], int(diff[3])
        if n_pairs == 0 or math.isnan(mean):
            continue
        out.append(_Point(x, mean, lo, hi, n_pairs))
    return out


# --- drawing -----------------------------------------------------------------------------------
def _style_axes(ax: Axes) -> None:
    """Recessive chrome: hairline solid light-grey grid, no top/right spines, grey ink."""
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID_COLOUR, linewidth=GRID_WIDTH, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOUR)
        ax.spines[side].set_linewidth(GRID_WIDTH * 1.5)
    ax.tick_params(colors=MUTED_TEXT, labelsize=TICK_SIZE, length=3, width=GRID_WIDTH,
                   labelcolor=TEXT_COLOUR)


def _numeric(values: Iterable[Any]) -> bool:
    """Whether every value is a real number (bools excluded)."""
    return all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)


def _positions(values: Sequence[Any]) -> dict[Any, float]:
    """x position per value: numbers stay themselves, anything else becomes its index."""
    if _numeric(values):
        return {v: float(v) for v in values}
    return {v: float(i) for i, v in enumerate(values)}


def _dodge_unit(positions: Iterable[float]) -> float:
    """Smallest gap between x positions (1 if there is a single one)."""
    xs = sorted(set(positions))
    gaps = [b - a for a, b in pairwise(xs) if b > a]
    return min(gaps) if gaps else 1.0


def _draw_series(
    ax: Axes,
    points: Sequence[_Point],
    pos: dict[Any, float],
    offset: float,
    colour: str,
    marker: str,
    linestyle: Any,
    connect: bool = True,
    hollow_below: float | None = None,
) -> bool:
    """One series: thin CI bars underneath, then the 2 px line with ringed markers.

    With ``hollow_below``, points whose success share is below it get a hollow marker.
    Returns whether any hollow marker was drawn.
    """
    if not points:
        return False
    xs = [pos[p.x] + offset for p in points]
    ys = [p.y for p in points]
    bars = [(x, p) for x, p in zip(xs, points, strict=True)
            if math.isfinite(p.lo) and math.isfinite(p.hi)]
    if bars:
        lows = [max(0.0, p.y - p.lo) for _, p in bars]
        highs = [max(0.0, p.hi - p.y) for _, p in bars]
        ax.errorbar([x for x, _ in bars], [p.y for _, p in bars], yerr=[lows, highs],
                    fmt="none", ecolor=colour, elinewidth=ERR_LINE_WIDTH, capsize=ERR_CAP_SIZE,
                    capthick=ERR_LINE_WIDTH, alpha=ERR_ALPHA, zorder=2)
    ax.plot(xs, ys, color=colour, linewidth=LINE_WIDTH, linestyle=linestyle if connect else "none",
            marker=marker, markersize=MARKER_SIZE, markeredgecolor=BACKGROUND,
            markeredgewidth=MARKER_RING, solid_capstyle="round", solid_joinstyle="round",
            dash_capstyle="round", zorder=3)
    hollow = [(x, p.y) for x, p in zip(xs, points, strict=True)
              if hollow_below is not None and p.success < hollow_below]
    if hollow:
        ax.plot([x for x, _ in hollow], [y for _, y in hollow], linestyle="none", marker=marker,
                markersize=MARKER_SIZE, markerfacecolor=BACKGROUND, markeredgecolor=colour,
                markeredgewidth=LINE_WIDTH * 0.8, zorder=3)
    return bool(hollow)


def _draw_agents(
    ax: Axes,
    series: dict[str, list[_Point]],
    agents: Sequence[str],
    x_values: Sequence[Any],
    connect: bool = True,
    dodge: float = DODGE_FRACTION,
    hollow_below: float | None = None,
) -> _Drawn:
    """All agents' series, dodged horizontally so their CI bars do not overlap.

    The dodge slot follows the agent's position in ``agents``; the drawing order puts
    :data:`PROPOSED_AGENT` last, so its line stays visible where series coincide (e.g. at
    100 % success). Returns the agents with at least one drawn point.
    """
    pos = _positions(x_values)
    unit = _dodge_unit(pos.values()) * dodge
    middle = (len(agents) - 1) / 2.0
    order = sorted(range(len(agents)), key=lambda i: agents[i] == PROPOSED_AGENT)
    hollow = False
    for i in order:
        agent = agents[i]
        style = agent_style(agent, agents)
        hollow |= _draw_series(ax, series.get(agent, []), pos, (i - middle) * unit, style.colour,
                               style.marker, style.linestyle, connect, hollow_below)
    ax.set_xticks([pos[v] for v in x_values])
    ax.set_xticklabels([_fmt_num(v) for v in x_values])
    if _numeric(x_values):
        span = max(pos.values()) - min(pos.values()) if len(pos) > 1 else 1.0
        pad = max(span * X_PAD_FRACTION, (middle + 1.0) * unit)
    else:
        pad = CATEGORY_PAD
    ax.set_xlim(min(pos.values()) - pad, max(pos.values()) + pad)
    return _Drawn(frozenset(a for a in agents if series.get(a)), hollow)


def _merge(drawn: Iterable[_Drawn]) -> _Drawn:
    """Union of what several panels drew."""
    items = list(drawn)
    return _Drawn(frozenset().union(*(d.agents for d in items)), any(d.hollow for d in items))


def _zero_floor(ax: Axes) -> None:
    """Start a non-negative y axis just below 0 so markers sitting at 0 are not clipped."""
    top = ax.get_ylim()[1]
    ax.set_ylim(-ZERO_PAD_FRACTION * top, top)


def _legend_handles(agents: Sequence[str], connect: bool = True,
                    hollow: bool = False) -> list[Line2D]:
    """One handle per agent, drawn like its series (no line when the series are unconnected);
    plus a neutral hollow-marker key when hollow markers were drawn."""
    handles = []
    for agent in agents:
        s = agent_style(agent, agents)
        handles.append(Line2D([], [], color=s.colour, linewidth=LINE_WIDTH,
                              linestyle=s.linestyle if connect else "none",
                              marker=s.marker, markersize=MARKER_SIZE,
                              markeredgecolor=BACKGROUND, markeredgewidth=MARKER_RING,
                              label=s.label))
    if hollow:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE,
                              markerfacecolor=BACKGROUND, markeredgecolor=MUTED_TEXT,
                              markeredgewidth=LINE_WIDTH * 0.8,
                              label=f"hollow: < {LOW_SUCCESS_FRACTION:.0%} of missions succeeded"))
    return handles


def _new_figure(n_panels: int, sharey: bool) -> tuple[Figure, list[Axes]]:
    """A figure with ``n_panels`` small multiples (constrained layout); unused cells removed.

    The size covers the panels only; :func:`_finish` adds the header band above them.
    """
    ncols = 1 if n_panels == 1 else (2 if n_panels == 4 else min(MAX_PANEL_COLUMNS, n_panels))
    nrows = math.ceil(n_panels / ncols)
    if n_panels == 1:
        size = SINGLE_PANEL_SIZE_IN
    else:
        size = (PANEL_SIZE_IN[0] * ncols, PANEL_SIZE_IN[1] * nrows)
    fig = Figure(figsize=size, dpi=DPI, facecolor=BACKGROUND, layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.subplots(nrows, ncols, sharey=sharey, squeeze=False)
    axes = [ax for row in grid for ax in row]
    for ax in axes[n_panels:]:
        fig.delaxes(ax)
    axes = axes[:n_panels]
    for ax in axes:
        _style_axes(ax)
    return fig, axes


def _wrap(text: str, width_in: float) -> list[str]:
    """``text`` wrapped to ``width_in`` inches of subtitle type (explicit newlines kept)."""
    chars = max(20, int(width_in * SUBTITLE_CHARS_PER_INCH))
    return [line for para in text.split("\n") for line in textwrap.wrap(para, chars)]


def _label_axes(axes: Sequence[Axes], xlabel: str, ylabel: str | None) -> None:
    """x label on every panel with no panel below it, y label on the first column."""
    for i, ax in enumerate(axes):
        spec = ax.get_subplotspec()
        ncols = spec.get_gridspec().ncols if spec is not None else 1
        if i + ncols >= len(axes):
            ax.set_xlabel(xlabel, color=TEXT_COLOUR, fontsize=LABEL_SIZE)
        if ylabel and i % ncols == 0:
            ax.set_ylabel(ylabel, color=TEXT_COLOUR, fontsize=LABEL_SIZE)


def _row_major(items: Sequence[Any], ncols: int) -> list[Any]:
    """Reorder ``items`` so matplotlib's column-major legend reads row by row."""
    nrows = math.ceil(len(items) / ncols)
    return [items[r * ncols + c] for c in range(ncols) for r in range(nrows)
            if r * ncols + c < len(items)]


def _fit_legend(fig: Figure, agents: Sequence[str], connect: bool = True,
                hollow: bool = False) -> Any:
    """A figure legend with as many columns as fit the figure width (measured, not guessed)."""
    handles = _legend_handles(agents, connect, hollow)
    renderer = fig.canvas.get_renderer()
    limit = fig.get_figwidth() * fig.dpi * LEGEND_MAX_WIDTH_FRACTION
    legend = None
    for ncols in range(len(handles), 0, -1):
        if legend is not None:
            legend.remove()
        legend = fig.legend(handles=_row_major(handles, ncols), loc="upper center",
                            bbox_to_anchor=(0.5, 1.0), ncols=ncols, frameon=False,
                            fontsize=LEGEND_SIZE, handlelength=2.6, columnspacing=1.6,
                            borderaxespad=0.0, borderpad=0.2)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOUR)
        if legend.get_window_extent(renderer).width <= limit:
            break
    return legend


def _finish(
    fig: Figure,
    axes: Sequence[Axes],
    title: str,
    subtitle: str,
    agents: Sequence[str],
    xlabel: str,
    ylabel: str | None,
    path: Path,
    connect: bool = True,
    hollow: bool = False,
) -> Path:
    """Header band (title, wrapped subtitle, legend) above the panels, axis labels; save.

    The header is laid out in inches (the legend is measured) and the constrained layout is
    confined below it, so the legend never covers data and never collides with the title. Every
    figure with agent series has a legend (even a single agent is named) listing the agents
    passed in (callers pass only agents with drawn points); the paired-difference figures have
    none (their titles name the one series).
    """
    _label_axes(axes, xlabel, ylabel)
    width, height = fig.get_size_inches()
    sub_lines = _wrap(subtitle, width - 2 * HEADER_SIDE_IN)
    legend = _fit_legend(fig, agents, connect, hollow) if agents else None
    legend_in = 0.0
    if legend is not None:
        legend_in = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    header = (2 * HEADER_PAD_IN + TITLE_LINE_IN + len(sub_lines) * SUBTITLE_LINE_IN
              + legend_in)
    total = height + header
    fig.set_size_inches(width, total)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 1.0 - header / total))
    y = 1.0 - HEADER_PAD_IN / total
    fig.text(0.5, y, title, ha="center", va="top", color=TEXT_COLOUR, fontsize=TITLE_SIZE,
             fontweight="bold")
    y -= TITLE_LINE_IN / total
    if sub_lines:
        fig.text(0.5, y, "\n".join(sub_lines), ha="center", va="top", color=MUTED_TEXT,
                 fontsize=SUBTITLE_SIZE, linespacing=SUBTITLE_LINE_SPACING)
        y -= len(sub_lines) * SUBTITLE_LINE_IN / total
    if legend is not None:
        legend.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BACKGROUND, metadata={"Title": title})
    return path


# --- figures -----------------------------------------------------------------------------------
class _Metric(NamedTuple):
    """A per-mission metric plotted against slip p."""

    file: str
    column: str
    title: str
    ylabel: str
    successful_only: bool
    note: str = ""


#: The (b1), (b2), (c), (d) charts: one metric per file - never a dual axis.
P_METRICS: tuple[_Metric, ...] = (
    _Metric("steps_vs_p.png", "steps", "Steps to the survivor vs slip probability",
            "steps per successful mission", True),
    _Metric("wasted_moves_vs_p.png", "wasted_moves", "Wasted moves vs slip probability",
            "wasted moves per successful mission\n(steps − true shortest path)", True),
    _Metric("replans_vs_p.png", "replans", "Replanning calls vs slip probability",
            "replans per mission", False, "; replans counted after the initial plan"),
    _Metric("cells_explored_vs_p.png", "cells_explored", "Cells explored vs slip probability",
            "cells explored per mission (cells)", False),
)


class _Paired(NamedTuple):
    """A paired ``andor − astar_replan`` difference chart vs slip p."""

    file: str
    column: str
    title: str
    ylabel: str
    below: str
    scope: str


#: Paired-difference charts: wasted moves (H3) and replans (H5).
PAIRED_METRICS: tuple[_Paired, ...] = (
    _Paired("paired_wasted_moves_diff.png", "wasted_moves", "Paired wasted-move difference",
            "Δ wasted moves per mission", "wastes fewer moves",
            "where both agents succeeded"),
    _Paired("paired_replans_diff.png", "replans", "Paired replanning difference",
            "Δ replans per mission", "replans less", "all missions"),
)

SUCCESS_CI_TEXT: str = "error bars: Wilson 95% CI"
MEAN_CI_TEXT: str = "error bars: Student-t 95% CI of the mean"
MARGINAL_CI_TEXT: str = ("CIs are per-agent marginal (unpaired); paired tests on identical tasks "
                         "are in HYPOTHESES.md")
P_LABEL: str = "slip probability p"
Q_LABEL: str = "aftershock probability q (per step)"
SUCCESS_LABEL: str = "missions rescued (%)"


def _percent_axis(ax: Axes) -> None:
    ax.set_ylim(*SUCCESS_YLIM)
    ax.set_yticks(SUCCESS_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))


def _scope_text(rows: Sequence[Row]) -> str:
    scenarios = _scenarios(rows)
    if len(scenarios) == 1:
        return _scenario_title(scenarios[0])
    return f"{len(scenarios)} scenarios pooled"


def _not_drawn_text(agents: Sequence[str], drawn: Iterable[str], what: str) -> str:
    """``; astar_fixed not drawn (...)`` for agents with no point at all (else '')."""
    shown = set(drawn)
    missing = [a for a in agents if a not in shown]
    if not missing:
        return ""
    return f"; {', '.join(missing)} not drawn ({what})"


def _low_success_text(rows: Sequence[Row], agents: Sequence[str]) -> str:
    """Caveat for means over all missions: which agents fail often, and how often they succeed."""
    notes = []
    for agent in agents:
        own = [r for r in rows if r["agent"] == agent]
        rates = [p.y for p in _success_points(own, "slip_p").get(agent, [])]
        if rates and min(rates) < PERCENT * LOW_SUCCESS_FRACTION:
            lo, hi = min(rates), max(rates)
            span = f"{lo:.0f}%" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}%"
            notes.append(f"{agent} succeeds in {span}")
    text = ("; means over all missions incl. failed ones; hollow marker = fewer than "
            f"{LOW_SUCCESS_FRACTION:.0%} of the point's missions succeeded, so its value mostly "
            "reflects missions abandoned early, not frugality")
    return text + (f" ({'; '.join(notes)})" if notes else "")


def _metric_rows(block: Sequence[Row]) -> tuple[list[Row], str]:
    """Rows of one grid size for the mean charts, and a note naming the rule.

    Pooling several sizes would make each mean (and its CI) mostly reflect the mix of path
    lengths, so with several sizes the middle one (lower middle for an even count) is used.
    """
    sizes = sorted({(r["rows"], r["cols"]) for r in block})
    if len(sizes) < 2:
        return list(block), ""
    chosen = sizes[(len(sizes) - 1) // 2]
    others = ", ".join(_size_label(*s) for s in sizes if s != chosen)
    rows = [r for r in block if (r["rows"], r["cols"]) == chosen]
    return rows, (f"; only {_size_label(*chosen)} maps shown (middle size; {others} in "
                  "summary.csv)")


def _per_scenario(
    rows: Sequence[Row],
    x_field: str,
    make_series: Callable[[Sequence[Row]], dict[str, list[_Point]]],
    sharey: bool,
    connect: bool = True,
    hollow_below: float | None = None,
) -> tuple[Figure, list[Axes], list[int], _Drawn]:
    """Small multiples, one panel per scenario; returns the figure, axes, point counts and
    what was drawn."""
    scenarios = _scenarios(rows)
    agents = _agents(rows)
    x_values = _distinct(rows, x_field)
    fig, axes = _new_figure(len(scenarios), sharey)
    counts: list[int] = []
    drawn: list[_Drawn] = []
    for ax, scenario in zip(axes, scenarios, strict=True):
        subset = [r for r in rows if r["scenario"] == scenario]
        series = make_series(subset)
        counts.extend(p.n for pts in series.values() for p in pts)
        drawn.append(_draw_agents(ax, series, agents, x_values, connect,
                                  hollow_below=hollow_below))
        ax.set_title(_scenario_title(scenario), color=TEXT_COLOUR, fontsize=PANEL_TITLE_SIZE,
                     loc="left")
    return fig, axes, counts, _merge(drawn)


def plot_success_vs_p(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """(a) Success % vs slip p, one panel per scenario on a shared 0-100 % axis."""
    block = _select_block(rows, SLIP_BLOCK)
    if not block:
        return None
    fig, axes, counts, drawn = _per_scenario(block, "slip_p",
                                             lambda r: _success_points(r, "slip_p"), sharey=True)
    for ax in axes:
        _percent_axis(ax)
    subtitle = (f"{_n_text(counts, 'missions per agent and point')}; {SUCCESS_CI_TEXT}; "
                f"{_pooled_text(block, 'slip_p')}; agents dodged sideways, {PROPOSED_AGENT} "
                "drawn on top where lines coincide")
    agents = [a for a in _agents(block) if a in drawn.agents]
    return _finish(fig, axes, "Rescue success vs slip probability", subtitle, agents,
                   P_LABEL, SUCCESS_LABEL, out_dir / "success_vs_p.png")


def plot_metric_vs_p(rows: Sequence[Row], out_dir: Path, metric: _Metric) -> Path | None:
    """(b1) / (b2) / (c) / (d): mean of one metric vs slip p, one panel per scenario.

    Each panel has its own y scale (scenarios differ in path length); agents are compared
    within a panel. With several grid sizes in the block, one size is shown
    (:func:`_metric_rows`). Means over all missions mark low-success points hollow.
    """
    block = _select_block(rows, SLIP_BLOCK)
    if not block or metric.column not in block[0]:
        return None
    sub, size_note = _metric_rows(block)
    hollow_below = None if metric.successful_only else LOW_SUCCESS_FRACTION
    fig, axes, counts, drawn = _per_scenario(
        sub, "slip_p",
        lambda r: _mean_points(r, "slip_p", metric.column, metric.successful_only),
        sharey=False, hollow_below=hollow_below)
    if not counts:
        return None
    for ax in axes:
        _zero_floor(ax)
    all_agents = _agents(sub)
    agents = [a for a in all_agents if a in drawn.agents]
    if metric.successful_only:
        unit = "successful missions per point"
        caveat = (f"; means of < {MIN_MEAN_N} successful missions not drawn; an agent that "
                  "rarely succeeds averages over a selected subset"
                  + _not_drawn_text(all_agents, agents,
                                    f"< {MIN_MEAN_N} successful missions at every point"))
    else:
        unit = "missions per point"
        caveat = _low_success_text(sub, agents)
    subtitle = (f"{_n_text(counts, unit)}; {MEAN_CI_TEXT}; {_pooled_text(sub, 'slip_p')}"
                f"{size_note}; y scales differ per panel{metric.note}{caveat}")
    return _finish(fig, axes, metric.title, subtitle, agents, P_LABEL, metric.ylabel,
                   out_dir / metric.file, hollow=drawn.hollow)


def plot_planning_time_vs_size(
    rows: Sequence[Row], out_dir: Path, workers: int | None = None
) -> Path | None:
    """(e) Planning ms per mission vs grid size (scaling block; log y, labelled).

    Geometric mean with a Student-t 95 % interval on log(ms), back-transformed: positive by
    construction, so the error bars are meaningful on the log axis. ``workers`` (from
    ``run_meta.json``) is stated, because timings are taken under that much concurrent load.
    """
    block = _select_block(rows, SIZE_BLOCK)
    if not block:
        return None
    all_agents = _agents(block)
    series = _mean_points(block, "rows", "planning_ms", successful_only=False, geometric=True)
    counts = [p.n for pts in series.values() for p in pts]
    if not counts:
        return None
    sizes = _distinct(block, "rows")
    cols_by_rows = {r["rows"]: r["cols"] for r in block}
    fig, axes = _new_figure(1, sharey=False)
    ax = axes[0]
    ax.set_yscale("log")
    drawn = _draw_agents(ax, series, all_agents, sizes, dodge=SIZE_DODGE_FRACTION)
    ax.set_xticklabels([_size_label(s, cols_by_rows[s]) for s in sizes])
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=LOG_TICK_SUBS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    load = (f"wall-clock timings measured with {workers} worker processes running concurrently"
            if workers is not None else
            "wall-clock timings measured under the run's parallel load (see run_meta.json)")
    subtitle = (f"{_n_text(counts, 'missions per point')}; geometric mean, error bars: "
                f"Student-t 95% CI on log(ms); {_scope_text(block)}; "
                f"{_pooled_text(block, 'rows')}; {load}")
    agents = [a for a in all_agents if a in drawn.agents]
    return _finish(fig, axes, "Planning time vs grid size", subtitle, agents,
                   "grid size (rows × cols, cells)",
                   "total planning time per mission\n(ms, geometric mean, log scale)",
                   out_dir / "planning_time_vs_size.png")


def plot_success_under_aftershocks(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """(f) Success % vs aftershock probability q (one panel per scenario of the block)."""
    block = _select_block(rows, AFTERSHOCK_BLOCK)
    if not block:
        return None
    fig, axes, counts, drawn = _per_scenario(
        block, "aftershock_q", lambda r: _success_points(r, "aftershock_q"), sharey=True)
    for ax in axes:
        _percent_axis(ax)
    ks = _distinct(block, "aftershock_k")
    subtitle = (f"{_n_text(counts, 'missions per agent and point')}; {SUCCESS_CI_TEXT}; "
                f"k{EQ}{_fmt_set(ks)} debris cells per aftershock; "
                f"{_pooled_text(block, 'aftershock_q')}; agents dodged sideways, "
                f"{PROPOSED_AGENT} drawn on top where lines coincide")
    agents = [a for a in _agents(block) if a in drawn.agents]
    return _finish(fig, axes, "Rescue success under aftershocks", subtitle, agents, Q_LABEL,
                   SUCCESS_LABEL, out_dir / "success_under_aftershocks.png")


def plot_paired(rows: Sequence[Row], out_dir: Path, metric: _Paired) -> Path | None:
    """Paired ``andor − astar_replan`` difference of one metric vs p (pooled + per scenario)."""
    block = _select_block(rows, SLIP_BLOCK)
    if not block or not set(PAIRED_AGENTS) <= {r["agent"] for r in block}:
        return None
    if metric.column not in block[0]:
        return None
    scenarios = _scenarios(block)
    groups: list[tuple[str, list[Row]]] = []
    if len(scenarios) > 1:
        groups.append((ALL_SCENARIOS_LABEL, list(block)))
    groups += [(_scenario_title(s), [r for r in block if r["scenario"] == s]) for s in scenarios]
    x_values = _distinct(block, "slip_p")
    fig, axes = _new_figure(len(groups), sharey=True)
    counts: list[int] = []
    for ax, (label, subset) in zip(axes, groups, strict=True):
        points = _paired_points(subset, "slip_p", metric.column)
        counts.extend(p.n for p in points)
        ax.axhline(0.0, color=ZERO_LINE_COLOUR, linewidth=ZERO_LINE_WIDTH, zorder=1)
        pos = _positions(x_values)
        _draw_series(ax, points, pos, 0.0, NEUTRAL_SERIES, "o", "-")
        ax.set_xticks([pos[v] for v in x_values])
        ax.set_xticklabels([_fmt_num(v) for v in x_values])
        ax.set_title(label, color=TEXT_COLOUR, fontsize=PANEL_TITLE_SIZE, loc="left")
    if not counts:
        return None
    a, b = PAIRED_AGENTS
    n_text = _n_text(counts, "pairs (one per map) per point")
    subtitle = (f"below 0 = {a} {metric.below}; per-task differences on identical maps and "
                f"outcome streams, {metric.scope}; {n_text}; {MEAN_CI_TEXT}; "
                f"{_pooled_text(block, 'slip_p')}")
    return _finish(fig, axes, f"{metric.title} {a} − {b}", subtitle, [], P_LABEL,
                   metric.ylabel, out_dir / metric.file)


def plot_paired_wasted_moves(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Paired ``andor − astar_replan`` wasted-move difference vs p (H3)."""
    return plot_paired(rows, out_dir, PAIRED_METRICS[0])


def plot_paired_replans(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Paired ``andor − astar_replan`` replanning difference vs p (H5)."""
    return plot_paired(rows, out_dir, PAIRED_METRICS[1])


def _plot_factor(
    rows: Sequence[Row],
    out_dir: Path,
    block_rule: _Block,
    file: str,
    title: str,
    xlabel: str,
    tick: Callable[[Any], str],
    connect: bool,
) -> Path | None:
    """Per scenario, three panels (success %, steps of successful missions, replans) vs one
    factor.

    One row of panels per scenario, so the intervals do not mix scenarios with very different
    path lengths. Separate panels with their own labelled y axes - never a dual axis.
    """
    block = _select_block(rows, block_rule)
    if not block:
        return None
    factor = block_rule.factor
    agents = _agents(block)
    x_values = _distinct(block, factor)
    scenarios = _scenarios(block)
    metrics = ("rescue success", "steps (successful missions)", "replanning calls")
    fig, axes = _new_figure(len(metrics) * len(scenarios), sharey=False)
    counts: list[int] = []
    drawn: list[_Drawn] = []
    steps_drawn: set[str] = set()
    for s_index, scenario in enumerate(scenarios):
        subset = [r for r in block if r["scenario"] == scenario]
        panels: list[tuple[str, dict[str, list[_Point]], float | None]] = [
            (SUCCESS_LABEL, _success_points(subset, factor), None),
            ("steps per successful mission",
             _mean_points(subset, factor, "steps", successful_only=True), None),
            ("replans per mission",
             _mean_points(subset, factor, "replans", successful_only=False),
             LOW_SUCCESS_FRACTION),
        ]
        for m_index, (ylabel, series, hollow_below) in enumerate(panels):
            ax = axes[s_index * len(metrics) + m_index]
            panel = _draw_agents(ax, series, agents, x_values, connect, hollow_below=hollow_below)
            drawn.append(panel)
            if m_index == 1:
                steps_drawn |= panel.agents
            ax.set_xticklabels([tick(v) for v in x_values])
            ax.set_title(f"{_scenario_title(scenario)}: {metrics[m_index]}", color=TEXT_COLOUR,
                         fontsize=PANEL_TITLE_SIZE, loc="left")
            ax.set_ylabel(ylabel, color=TEXT_COLOUR, fontsize=LABEL_SIZE)
            if m_index == 0:
                _percent_axis(ax)
                counts.extend(p.n for pts in series.values() for p in pts)
            else:
                _zero_floor(ax)
    merged = _merge(drawn)
    shown = [a for a in agents if a in merged.agents]
    steps_missing = [a for a in shown if a not in steps_drawn]
    steps_note = (f"; {', '.join(steps_missing)}: steps not drawn (< {MIN_MEAN_N} successful "
                  "missions)" if steps_missing else "")
    hollow_note = (f"; hollow markers: < {LOW_SUCCESS_FRACTION:.0%} of the missions succeeded "
                   "(replans then mostly reflect missions abandoned early)"
                   if merged.hollow else "")
    subtitle = (f"{_n_text(counts, 'missions per agent and point')}; error bars: Wilson 95% CI "
                f"(success), Student-t 95% CI (means); {MARGINAL_CI_TEXT}; "
                f"{_pooled_text(block, factor)}{steps_note}{hollow_note}")
    return _finish(fig, axes, title, subtitle, shown, xlabel, None, out_dir / file,
                   connect=connect, hollow=merged.hollow)


def plot_sensor_radius(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Sensor-radius block: success, steps and replans for each radius (if present)."""
    return _plot_factor(rows, out_dir, SENSOR_BLOCK, "sensor_radius.png",
                        "Effect of the sensor radius", "sensor radius r (Manhattan, cells)",
                        lambda v: f"r = {v}", connect=True)


def plot_replan_mode(rows: Sequence[Row], out_dir: Path) -> Path | None:
    """Replan-mode block: success, steps and replans for eager vs lazy (if present)."""
    return _plot_factor(rows, out_dir, MODE_BLOCK, "replan_mode.png",
                        "Eager vs lazy replanning", "replan mode", str, connect=False)


#: File of the H7 tie-break ablation figure, and its panel titles keyed by comparison name.
TIEBREAK_FILE: str = "tiebreak_ablation.png"
TIEBREAK_PANEL_TITLES: dict[str, str] = {
    "(a) matched, diagonal": "(a) andor − astar_replan, both diagonal",
    "(b) matched, plain": "(b) andor fixed − astar_replan larger-g",
    "(c) andor diagonal − fixed": "(c) andor: diagonal − fixed",
    "(d) A* diagonal − g": "(d) astar_replan: diagonal − larger-g",
}


def plot_tiebreak_ablation(
    rows: Sequence[Row], out_dir: Path, slip_experiment: str | None = None
) -> Path | None:
    """H7: paired wasted-move differences vs p for the four tie-break comparisons.

    Each panel pairs runs of the slip sweep with runs of the ``tiebreak_ablation`` block on the
    same map and outcome stream (``hypotheses.h7_paired``; map as the unit). The slip sweep is
    chosen exactly as H7 chooses it (``hypotheses.pick_slip_experiment``, honouring
    ``slip_experiment``), so the figure and the H7 table always use the same pair of blocks.
    Skipped when either block or one of the four agent / tie-break combinations is missing.
    """
    from rescue.experiments import hypotheses as hyp

    sweep_name = hyp.pick_slip_experiment(rows, slip_experiment)
    if sweep_name is None:
        return None
    blocks = (sweep_name, TIEBREAK_ABLATION_EXPERIMENT)
    present = {str(r.get("experiment", "")) for r in rows}
    if not set(blocks) <= present:
        return None
    comparisons = [c for c in hyp.H7_COMPARISONS if c.name in TIEBREAK_PANEL_TITLES]
    sides = {side: hyp.h7_side_rows(rows, side, blocks)
             for c in comparisons for side in (c.a, c.b)}
    if any(not sub or len({r["experiment"] for r in sub}) > 1 for sub in sides.values()):
        return None
    x_values = _distinct([r for sub in sides.values() for r in sub], "slip_p")
    pos = _positions(x_values)
    fig, axes = _new_figure(len(comparisons), sharey=True)
    counts: list[int] = []
    for ax, comp in zip(axes, comparisons, strict=True):
        points = []
        for x in x_values:
            diff = hyp.h7_paired([r for r in sides[comp.a] if r["slip_p"] == x],
                                 [r for r in sides[comp.b] if r["slip_p"] == x], "wasted_moves")
            if diff.n_pairs and not math.isnan(diff.mean_diff):
                points.append(_Point(x, diff.mean_diff, diff.lo, diff.hi, diff.n_units))
        counts.extend(p.n for p in points)
        ax.axhline(0.0, color=ZERO_LINE_COLOUR, linewidth=ZERO_LINE_WIDTH, zorder=1)
        _draw_series(ax, points, pos, 0.0, NEUTRAL_SERIES, "o", "-")
        ax.set_xticks([pos[v] for v in x_values])
        ax.set_xticklabels([_fmt_num(v) for v in x_values])
        ax.set_title(TIEBREAK_PANEL_TITLES[comp.name], color=TEXT_COLOUR,
                     fontsize=PANEL_TITLE_SIZE, loc="left")
    if not counts:
        return None
    pooled_rows = sides[comparisons[0].a]
    subtitle = (f"below 0 = the first term wastes fewer moves; {sweep_name} runs "
                f"paired with {TIEBREAK_ABLATION_EXPERIMENT} runs on identical maps and outcome "
                f"streams, where both succeeded; (a)+(b) compare the agents with matched "
                f"tie-breaks, (c)+(d) each tie-break's own effect; "
                f"{_n_text(counts, 'maps per point')}; {MEAN_CI_TEXT} (map as the unit); "
                f"{_pooled_text(pooled_rows, 'slip_p')}")
    return _finish(fig, axes, "Tie-break ablation: paired wasted-move differences", subtitle,
                   [], P_LABEL, "Δ wasted moves per mission", out_dir / TIEBREAK_FILE)


def plots_dir(out_dir: str | Path) -> Path:
    """The PNG directory: ``out_dir`` itself if it is named ``plots``, else ``out_dir/plots``."""
    out = Path(out_dir)
    return out if out.name == PLOTS_DIR_NAME else out / PLOTS_DIR_NAME


def make_all_plots(rows: Sequence[Row], out_dir: str | Path) -> list[Path]:
    """Write every figure the rows support into ``results/plots/``; return the paths written.

    ``out_dir`` is the results directory (PNGs go to ``out_dir/plots``) or the plots
    directory itself. Figures whose block or agents are missing are skipped silently. The
    worker count in the results directory's ``run_meta.json`` (if any) is quoted on the
    planning-time chart.
    """
    rows = list(rows)
    if not rows:
        return []
    target = plots_dir(out_dir)
    meta = read_run_meta(target.parent) or {}
    workers = meta.get("workers") if isinstance(meta.get("workers"), int) else None
    paths: list[Path | None] = [plot_success_vs_p(rows, target)]
    paths += [plot_metric_vs_p(rows, target, metric) for metric in P_METRICS]
    paths += [
        plot_planning_time_vs_size(rows, target, workers),
        plot_success_under_aftershocks(rows, target),
        *(plot_paired(rows, target, metric) for metric in PAIRED_METRICS),
        plot_sensor_radius(rows, target),
        plot_replan_mode(rows, target),
        plot_tiebreak_ablation(rows, target),
    ]
    return [path for path in paths if path is not None]


__all__ = [
    "AGENT_STYLE",
    "PAIRED_METRICS",
    "P_METRICS",
    "AgentStyle",
    "agent_style",
    "make_all_plots",
    "plot_metric_vs_p",
    "plot_paired",
    "plot_paired_replans",
    "plot_paired_wasted_moves",
    "plot_planning_time_vs_size",
    "plot_replan_mode",
    "plot_sensor_radius",
    "plot_success_under_aftershocks",
    "plot_success_vs_p",
    "plot_tiebreak_ablation",
    "plots_dir",
]
