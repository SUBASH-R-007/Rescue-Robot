"""Head-to-head comparison app (``python -m rescue.compare``, brief §6.2, docs/plan.md §3.11).

:class:`CompareApp` shows 2-6 agents (default ``andor | astar_fixed | astar_replan``) side by
side. Every pane owns its own :class:`~rescue.runner.Mission` on the **same** true map with the
**same** world seed, so every :class:`~rescue.world.World` draws its one outcome uniform per
step from an identically seeded stream (common random numbers, D3): at step ``t`` all robots
that act from cells with the same slip probability get the same outcome kind. Aftershock draws
are shared too, but their debris avoids each robot's own sensor footprint, so with aftershocks
on the true maps can drift apart once the robots stand in different places (D20). The missions
advance in lock-step: one tick steps every still-RUNNING mission exactly once, so all running
panes always share the same world time ``t``.

Each pane shows the agent name in its colour, a status badge (RUNNING / RESCUED / ABORTED /
BUDGET / UNREACHABLE), the robot's map (fog of war, executed trace, replan flash, optional
policy arrows and true-map overlay), compact metrics and the pane's latest notable event. A
footer shows the scenario, seed, p, map size, speed, the tick and the key hints.

Like the simulator it contains no mission logic: every step is ``Mission.step()``; trace marks,
event lines and flashes come from the resulting :class:`~rescue.runner.StepEvent` (through the
same helpers as :mod:`rescue.viz.sim_app`). It runs headless (``SDL_VIDEODRIVER=dummy``) for
tests, figures and recordings.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pygame

from rescue import mapgen
from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES, make_agent
from rescue.agents.base import PlanSnapshot
from rescue.config import (
    COMPARE_AGENT_COLOURS,
    COMPARE_DEFAULT_SCENARIO,
    COMPARE_DEFAULT_SEED,
    COMPARE_DEFAULT_SIZE,
    COMPARE_DEFAULT_SPEED_INDEX,
    COMPARE_EVENT_LINES,
    COMPARE_MAX_AGENTS,
    COMPARE_MAX_PANE_ROWS,
    COMPARE_METRIC_COLUMNS,
    COMPARE_MIN_AGENTS,
    COMPARE_PANE_GAP,
    COMPARE_PANE_PADDING,
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
    MAX_SLIP_P,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.grid import Pos, TrueGrid, fmt_pos
from rescue.model import Action
from rescue.runner import Mission, MissionStatus, StepEvent, make_transition
from rescue.viz import theme
from rescue.viz.grid_view import GridScene, GridView, TraceMark
from rescue.viz.panel import (
    INFO_SEPARATOR,
    PAUSED_TEXT,
    PLAYING_TEXT,
    REC_TEXT,
    PanelInfo,
    badge_size,
    draw_badge,
    draw_help_overlay,
    log_kind_colour,
)
from rescue.viz.sim_app import (
    BaseApp,
    ScenarioEntry,
    event_messages,
    flash_cells,
    resolve_path,
)
from rescue.viz.tree_view import text_cache

log = logging.getLogger(__name__)

#: Window title.
COMPARE_CAPTION = "Rescue Robot - head-to-head comparison"
#: Footer key hints.
COMPARE_HINT = (
    "Space play · N step · R reset · S new seed · +/- speed · [ ] slip p · M scenario · "
    "T true map · P arrows · C contrast · G record · H help · Esc quit"
)
#: Footer / pane texts.
CRN_TEXT = (
    "Same map and world seed in every pane (common random numbers); "
    "each tick steps every running mission once"
)
#: With aftershocks the debris lands outside each robot's own sensor footprint (D9), so the
#: true maps stay equal only while the robots stand in the same places (D20).
CRN_AFTERSHOCK_TEXT = (
    "Same start map, seed and random draws in every pane; aftershock debris avoids each "
    "robot's own surroundings, so the true maps can drift apart"
)
ALL_DONE_TEXT = "All missions over: R resets, S tries a new seed"
READY_TEXT = "Planned at {pos}"
EVENT_PREFIX = "t={t}: "
#: Lock-step tick counter of the footer (a pane's own world time is its Steps metric; an
#: agent's final Abort takes a tick but no world step).
TICK_TEXT = "tick {t}"
#: Which event line a pane shows when one step produced several (highest first; among equal
#: priorities the later line wins).
EVENT_PRIORITY: dict[str, int] = {
    "failure": 5, "success": 5, "warn": 4, "replan": 3, "slip": 2, "skid": 2, "info": 1,
}
#: Labels of the per-pane metrics, in display order (column-major within the block).
STEPS_LABEL = "Steps"
METRIC_LABELS: tuple[str, ...] = (
    STEPS_LABEL, "Replans", "Slips handled", "Explored", "Plan ms", "Unplanned",
)
#: Footer lines: settings, latest app message, key hints.
FOOTER_LINES = 3
#: Rounding of the slip probability after ``[`` / ``]`` (avoids 0.30000000000000004).
SLIP_P_DECIMALS = 2

_FASTER_KEYS = frozenset({pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS})
_SLOWER_KEYS = frozenset({pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS})


def compare_help_lines() -> list[tuple[str, str] | str]:
    """Help-overlay items of the comparison app."""
    return [
        "Missions (all panes)",
        ("Space", "play / pause"),
        ("N", "one lock-step tick"),
        ("R", "reset (same seed)"),
        ("S", "new seed (seed + 1)"),
        ("+ / -", "speed"),
        ("[ / ]", f"slip p −/+ {theme.SLIP_P_STEP:g}"),
        ("M", "next scenario"),
        "View",
        ("T", "true-map overlay (35 %)"),
        ("P", "policy arrows"),
        ("C", "high-contrast palette"),
        ("G", "record GIF / MP4"),
        ("H", "this help"),
        ("Esc", "quit"),
    ]


def pane_event_line(event: StepEvent) -> tuple[str, str] | None:
    """The one ``(text, kind)`` line a pane shows for ``event`` (None if nothing notable).

    Picks the most important of ``sim_app.event_messages`` (:data:`EVENT_PRIORITY`); a final
    Abort's note (``"reason: detail"``) is folded into the failure line, which keeps the
    failure colour.
    """
    lines = event_messages(event)
    if not lines:
        return None
    text, kind = max(reversed(lines), key=lambda line: EVENT_PRIORITY.get(line[1], 0))
    if kind == "failure" and event.action is None and event.note:
        _, sep, detail = event.note.partition(": ")
        text = f"{text}: {detail if sep else event.note}"
    return EVENT_PREFIX.format(t=event.t) + text, kind


def agent_colour(palette: theme.Palette, agent: str) -> theme.RGB:
    """Colour of ``agent`` in ``palette``, falling back to :data:`COMPARE_AGENT_COLOURS`."""
    if agent in palette.agent:
        return palette.agent[agent]
    return COMPARE_AGENT_COLOURS.get(palette.name, {}).get(agent, palette.highlight)


def parse_agents(text: str | Sequence[str] | None) -> tuple[str, ...]:
    """Validate an agent list (``"a,b,c"`` or a sequence); ``None`` → :data:`AGENT_NAMES`.

    ``ValueError`` for unknown or repeated names, or fewer than ``COMPARE_MIN_AGENTS`` / more
    than ``COMPARE_MAX_AGENTS`` agents.
    """
    if text is None:
        return AGENT_NAMES
    raw = text.split(",") if isinstance(text, str) else list(text)
    names = tuple(n.strip() for n in raw if n.strip())
    unknown = [n for n in names if n not in ALL_AGENT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown agent(s) {', '.join(unknown)}; choose from {', '.join(ALL_AGENT_NAMES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"agents must be distinct, got {', '.join(names)}")
    if not COMPARE_MIN_AGENTS <= len(names) <= COMPARE_MAX_AGENTS:
        raise ValueError(
            f"compare {COMPARE_MIN_AGENTS}-{COMPARE_MAX_AGENTS} agents, got {len(names)}"
        )
    return names


@dataclass(frozen=True)
class CompareOptions:
    """Comparison settings (the CLI flags of ``python -m rescue.compare``).

    ``map_path`` (an ASCII map) takes precedence over ``scenario`` (a ``rescue.mapgen``
    scenario generated at ``rows`` x ``cols`` with ``seed``). ``agents`` are 2-6 distinct ids
    of ``ALL_AGENT_NAMES``. ``autoplay`` None means "play when headless".
    """

    map_path: str | None = None
    scenario: str = COMPARE_DEFAULT_SCENARIO
    rows: int = COMPARE_DEFAULT_SIZE[0]
    cols: int = COMPARE_DEFAULT_SIZE[1]
    seed: int = COMPARE_DEFAULT_SEED
    agents: tuple[str, ...] = AGENT_NAMES
    slip_p: float = DEFAULT_SLIP_P
    sensor_radius: int = DEFAULT_SENSOR_RADIUS
    replan_mode: str = "eager"
    aftershock_q: float = 0.0
    aftershock_k: int = 1
    high_contrast: bool = False
    headless: bool = False
    frames: int = theme.HEADLESS_FRAMES
    screenshot: str | None = None
    record: str | None = None
    autoplay: bool | None = None
    mp4: bool = False
    speed_index: int = COMPARE_DEFAULT_SPEED_INDEX
    recordings_dir: str = theme.RECORDINGS_DIR


@dataclass
class ComparePane:
    """UI state of one pane: its mission plus render-side bookkeeping (no logic)."""

    agent: str
    mission: Mission
    view: GridView
    trace_id: int
    trace: list[TraceMark] = field(default_factory=list)
    #: Latest notable event line ``(text, kind)`` (see ``sim_app.event_messages``).
    last_event: tuple[str, str] = ("", "info")
    flash: tuple[Pos, ...] = ()
    flash_start: float = -math.inf
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 1, 1))
    _metrics_key: int = -1
    _metrics: dict[str, str] = field(default_factory=dict)
    _arrows_key: tuple[int, int] | None = None
    _arrows: dict[Pos, Action] = field(default_factory=dict)

    @property
    def status(self) -> MissionStatus:
        """The mission's status."""
        return self.mission.status

    @property
    def t(self) -> int:
        """World time of the pane's mission."""
        return self.mission.world.t

    def record(self, event: StepEvent, now_ms: float) -> None:
        """Bookkeeping for a new event: trace mark, last event line, replan flash."""
        mark = TraceMark.from_event(event)
        if mark is not None:
            self.trace.append(mark)
        line = pane_event_line(event)
        if line is not None:
            self.last_event = line
        elif event.action is None and event.t == 0:
            self.last_event = (READY_TEXT.format(pos=fmt_pos(event.new_pos)), "info")
        cells = flash_cells(event)
        if cells:
            self.flash, self.flash_start = cells, now_ms

    def snapshot(self) -> PlanSnapshot | None:
        """The agent's current plan snapshot (None before the first plan)."""
        agent = self.mission.agent
        pid = agent.current_plan_id
        return agent.snapshots[pid] if 0 <= pid < len(agent.snapshots) else None

    def policy(self) -> dict[Pos, Action]:
        """The agent's policy arrows (cached per event count and plan id)."""
        key = (len(self.mission.events), self.mission.agent.current_plan_id)
        if key != self._arrows_key:
            self._arrows_key, self._arrows = key, self.mission.agent.policy_arrows()
        return self._arrows

    def metric_values(self) -> dict[str, str]:
        """Display values of :data:`METRIC_LABELS` (cached per event count)."""
        key = len(self.mission.events)
        if key != self._metrics_key:
            m = self.mission.metrics()
            values = (
                f"{m.steps}/{self.mission.budget}", str(m.replans), str(m.slips_handled),
                str(m.cells_explored), f"{m.planning_ms:.1f}", str(m.unplanned_states),
            )
            self._metrics_key = key
            self._metrics = dict(zip(METRIC_LABELS, values, strict=True))
        return self._metrics


def scenario_entries(map_path: str | None) -> list[ScenarioEntry]:
    """The ``M`` cycle: every ``mapgen.SCENARIOS`` generator, preceded by ``map_path``."""
    entries = [ScenarioEntry(name, generator=name) for name in mapgen.SCENARIOS]
    if map_path is not None:
        path = resolve_path(map_path)
        if not path.is_file():
            raise FileNotFoundError(f"map file not found: {map_path}")
        entries.insert(0, ScenarioEntry(path.stem, path=path))
    return entries


def _line_h(font: pygame.font.Font) -> int:
    return font.get_linesize() + theme.LINE_SPACING


class CompareApp(BaseApp):
    """Side-by-side panes of 2-6 agents on one map and one seed, stepped in lock-step."""

    caption = COMPARE_CAPTION
    help_items = True

    def __init__(self, options: CompareOptions | None = None) -> None:
        opts = options or CompareOptions()
        self.agents = parse_agents(opts.agents)
        if opts.map_path is None and opts.scenario not in mapgen.SCENARIOS:
            raise ValueError(
                f"unknown scenario {opts.scenario!r}; choose from {', '.join(mapgen.SCENARIOS)}"
            )
        self.options = opts
        self.scenarios = scenario_entries(opts.map_path)
        self.scenario_index = 0 if opts.map_path is not None else next(
            i for i, e in enumerate(self.scenarios) if e.generator == opts.scenario
        )
        self.seed = int(opts.seed)
        self.slip_p = float(opts.slip_p)
        self.base_grid = self._load_grid(self.scenarios[self.scenario_index])
        WorldConfig(slip_p=self.slip_p, sensor_radius=opts.sensor_radius)  # validate early
        super().__init__(
            headless=opts.headless,
            high_contrast=opts.high_contrast,
            frames=opts.frames,
            screenshot=opts.screenshot,
            record=opts.record,
            mp4=opts.mp4,
            recordings_dir=opts.recordings_dir,
            speed_index=opts.speed_index,
            autoplay=opts.autoplay,
        )
        self.panes: list[ComparePane] = []
        #: Rows of panes and the shared agent-name font chosen by :meth:`layout`.
        self.pane_rows = 1
        self.name_font: pygame.font.Font = self.fonts.heading
        #: Lock-step ticks taken since the last reset.
        self.ticks = 0
        self._serial = 0
        self._layout_key: tuple[int, ...] | None = None
        self._footer = pygame.Rect(0, 0, 1, 1)
        self.reset_missions()

    # --- scenarios -------------------------------------------------------------------------
    def _load_grid(self, entry: ScenarioEntry) -> TrueGrid:
        """The true map of ``entry`` (generated with the current seed for generators)."""
        if entry.path is not None:
            return TrueGrid.load(entry.path)
        assert entry.generator is not None
        return mapgen.generate(entry.generator, self.options.rows, self.options.cols, self.seed)

    @property
    def scenario_name(self) -> str:
        """Name shown for the current map."""
        return self.base_grid.name or self.scenarios[self.scenario_index].name

    @property
    def missions(self) -> list[Mission]:
        """The panes' missions, in pane order."""
        return [pane.mission for pane in self.panes]

    def next_scenario(self) -> None:
        """``M``: switch to the next scenario of the cycle and reset every mission."""
        n = len(self.scenarios)
        for k in range(1, n + 1):
            index = (self.scenario_index + k) % n
            entry = self.scenarios[index]
            try:
                grid = self._load_grid(entry)
            except (OSError, ValueError, RuntimeError) as exc:
                self.log.add(f"Cannot load {entry.name}: {exc}", "warn")
                continue
            self.scenario_index, self.base_grid = index, grid
            self.reset_missions()
            return

    # --- missions --------------------------------------------------------------------------
    def world_config(self) -> WorldConfig:
        """World settings shared by every pane's mission."""
        o = self.options
        return WorldConfig(
            slip_p=self.slip_p,
            sensor_radius=o.sensor_radius,
            aftershock=AftershockConfig(q=o.aftershock_q, k=o.aftershock_k),
        )

    def reset_missions(self) -> None:
        """Start one fresh mission per agent on :attr:`base_grid` with the same seed and p."""
        cfg = self.world_config()
        grid = self.base_grid
        agent_cfg = AgentConfig(replan_mode=self.options.replan_mode)  # type: ignore[arg-type]
        old_views = [pane.view for pane in self.panes]
        self.panes = []
        for i, name in enumerate(self.agents):
            agent = make_agent(name, make_transition(grid, cfg), PlannerConfig(), agent_cfg)
            mission = Mission(grid, agent, cfg, seed=self.seed, scenario=self.scenario_name)
            view = old_views[i] if i < len(old_views) else GridView(
                pygame.Rect(0, 0, 1, 1), self.palette, self.fonts)
            self._serial += 1
            self.panes.append(ComparePane(agent=name, mission=mission, view=view,
                                          trace_id=self._serial))
        self.ticks = 0
        self._accumulator = 0.0
        self._layout_key = None
        self.log.clear()
        for pane in self.panes:
            pane.record(pane.mission.start(), self.clock_ms)
        if self.all_done:
            self.paused = True

    @property
    def all_done(self) -> bool:
        """True once every pane's mission reached a terminal status."""
        return all(pane.mission.done for pane in self.panes)

    def _can_advance(self) -> bool:
        return not self.all_done

    def _advance(self) -> bool:
        """One lock-step tick: step every still-RUNNING mission once."""
        if self.all_done:
            return False
        for pane in self.panes:
            if not pane.mission.done:
                pane.record(pane.mission.step(), self.clock_ms)
        self.ticks += 1
        if self.all_done:
            self.paused = True
            self.log.add(ALL_DONE_TEXT, "info")
        return True

    def step_once(self) -> bool:
        """``N``: pause and take exactly one lock-step tick; False if every mission is over."""
        self.paused = True
        if self.all_done:
            self.log.add(ALL_DONE_TEXT, "info")
            return False
        return self._advance()

    def toggle_play(self) -> None:
        """``Space``: play / pause."""
        if self.all_done:
            self.paused = True
            self.log.add(ALL_DONE_TEXT, "info")
            return
        self.paused = not self.paused

    def new_seed(self) -> None:
        """``S``: seed + 1; a generated scenario is regenerated with it, a map file stays."""
        self.seed += 1
        entry = self.scenarios[self.scenario_index]
        if entry.generator is not None:
            try:
                self.base_grid = self._load_grid(entry)
            except (OSError, ValueError, RuntimeError) as exc:
                self.log.add(f"Cannot regenerate {entry.name}: {exc}", "warn")
        self.reset_missions()

    def change_slip(self, delta: float) -> None:
        """``[`` / ``]``: change p by ``delta`` within [0, MAX_SLIP_P] and reset."""
        p = round(min(MAX_SLIP_P, max(0.0, self.slip_p + delta)), SLIP_P_DECIMALS)
        if p == self.slip_p:
            self.log.add(f"Slip p stays at {p:.2f} (range 0 – {MAX_SLIP_P:g})", "info")
            return
        self.slip_p = p
        self.reset_missions()

    # --- input -----------------------------------------------------------------------------
    def handle_key(self, key: int, mods: int = 0) -> bool:
        """Every comparison control (see :func:`compare_help_lines`); True if handled."""
        if super().handle_key(key, mods):
            return True
        if key == pygame.K_SPACE:
            self.toggle_play()
        elif key == pygame.K_n:
            self.step_once()
        elif key == pygame.K_r:
            self.reset_missions()
        elif key == pygame.K_s:
            self.new_seed()
        elif key == pygame.K_LEFTBRACKET:
            self.change_slip(-theme.SLIP_P_STEP)
        elif key == pygame.K_RIGHTBRACKET:
            self.change_slip(theme.SLIP_P_STEP)
        elif key == pygame.K_m:
            self.next_scenario()
        else:
            return False
        return True

    def _handle_mouse(self, event: pygame.event.Event) -> bool:
        return True  # nothing is clickable; keep events away from the (hidden) panel

    def _cycle_palette(self) -> None:
        super()._cycle_palette()
        for pane in self.panes:
            pane.view.set_palette(self.palette)

    # --- layout ----------------------------------------------------------------------------
    def _metric_rows(self) -> int:
        return math.ceil(len(METRIC_LABELS) / COMPARE_METRIC_COLUMNS)

    def _info_height(self) -> int:
        """Height of a pane's metrics block plus its event line(s)."""
        small = _line_h(self.fonts.small)
        return (self._metric_rows() * small + theme.LINE_SPACING
                + COMPARE_EVENT_LINES * small)

    def _footer_height(self) -> int:
        return FOOTER_LINES * _line_h(self.fonts.small) + 2 * theme.PANEL_PADDING

    def _cell_size(self, pane_w: int, row_h: int) -> int:
        """Map cell size (pixels) in a pane ``pane_w`` wide inside a row ``row_h`` tall."""
        pad = COMPARE_PANE_PADDING
        grid_h_max = (row_h - 2 * pad - theme.PANE_HEADER_HEIGHT - self._info_height()
                      - 2 * theme.SECTION_GAP)
        return max(theme.MIN_CELL_SIZE,
                   min(theme.MAX_CELL_SIZE, (pane_w - 2 * pad) // self.base_grid.cols,
                       max(1, grid_h_max) // self.base_grid.rows))

    def _name_font(self, pane_w: int) -> pygame.font.Font:
        """Largest header font in which every agent name fits a pane ``pane_w`` wide (shared by
        all panes, so no name looks ranked by its size)."""
        fonts = self.fonts
        r = theme.SWATCH_RADIUS
        name_w = (pane_w - 2 * COMPARE_PANE_PADDING - badge_size(fonts)[0]
                  - 2 * r - theme.SECTION_GAP)
        return next((f for f in (fonts.heading, fonts.body)
                     if all(text_cache.width(f, pane.agent) <= name_w for pane in self.panes)),
                    fonts.small)

    def layout(self) -> None:
        """Place the panes and the footer.

        The panes fill up to ``COMPARE_MAX_PANE_ROWS`` rows (``ceil(n / rows)`` per row, the
        last row centred); the row count giving the largest map cells wins, fewer rows on a
        tie. The block of panes is centred vertically.
        """
        rows, cols = self.base_grid.rows, self.base_grid.cols
        width, height = self.screen.get_size()
        key = (width, height, rows, cols, len(self.panes), id(self.fonts))
        if key == self._layout_key:
            return
        self._layout_key = key
        footer_h = self._footer_height()
        self._footer = pygame.Rect(0, height - footer_h, width, footer_h)
        n = len(self.panes)
        m, gap, pad = theme.MARGIN, COMPARE_PANE_GAP, COMPARE_PANE_PADDING
        area_w = width - 2 * m
        area_h = height - footer_h - 2 * m
        best: tuple[int, int, int] | None = None  # (cell, pane rows, panes per row)
        for pane_rows in range(1, min(n, COMPARE_MAX_PANE_ROWS) + 1):
            per_row = math.ceil(n / pane_rows)
            pane_w = (area_w - (per_row - 1) * gap) // per_row
            row_h = (area_h - (pane_rows - 1) * gap) // pane_rows
            cell = self._cell_size(pane_w, row_h)
            if best is None or cell > best[0]:
                best = (cell, pane_rows, per_row)
        assert best is not None
        cell, pane_rows, per_row = best
        self.pane_rows = pane_rows
        pane_w = (area_w - (per_row - 1) * gap) // per_row
        self.name_font = self._name_font(pane_w)
        header_h = theme.PANE_HEADER_HEIGHT
        inner_w = pane_w - 2 * pad
        board_w, board_h = cell * cols, cell * rows
        pane_h = (2 * pad + header_h + theme.SECTION_GAP + board_h + theme.SECTION_GAP
                  + self._info_height())
        block_h = pane_rows * pane_h + (pane_rows - 1) * gap
        top0 = m + max(0, (area_h - block_h) // 2)
        for i, pane in enumerate(self.panes):
            row, col = divmod(i, per_row)
            in_row = min(per_row, n - row * per_row)
            left0 = m + (area_w - (in_row * pane_w + (in_row - 1) * gap)) // 2
            left = left0 + col * (pane_w + gap)
            top = top0 + row * (pane_h + gap)
            pane.rect = pygame.Rect(left, top, pane_w, pane_h)
            board = pygame.Rect(left + pad + (inner_w - board_w) // 2,
                                top + pad + header_h + theme.SECTION_GAP, board_w, board_h)
            pane.view.set_rect(board)
            pane.view.layout(rows, cols)

    # --- rendering -------------------------------------------------------------------------
    def _text(self, surface: pygame.Surface, font: pygame.font.Font, text: str,
              colour: theme.RGB, x: int, y: int, max_w: int | None = None) -> int:
        """Blit (fitted) text at (x, y); returns its width."""
        if max_w is not None:
            text = text_cache.fit(font, text, max(0, max_w))
        if not text:
            return 0
        surf = text_cache.render(font, text, colour)
        surface.blit(surf, (x, y))
        return surf.get_width()

    def _scene(self) -> GridScene:  # pragma: no cover - CompareApp draws per pane
        raise NotImplementedError("CompareApp renders one scene per pane")

    def pane_scene(self, pane: ComparePane) -> GridScene:
        """The grid scene of one pane (its agent's belief, its world, its trace)."""
        mission = pane.mission
        world, agent = mission.world, mission.agent
        grid = world.grid
        snap = pane.snapshot()
        alpha = theme.flash_alpha(self.clock_ms - pane.flash_start) if pane.flash else 0.0
        return GridScene(
            rows=grid.rows,
            cols=grid.cols,
            belief=agent.belief.grid,
            true_cells=grid.cells,
            high_slip=grid.high_slip,
            robot=world.pos,
            start=grid.start,
            survivor=grid.survivor,
            show_true=self.show_true,
            show_policy=self.show_policy,
            policy=pane.policy() if self.show_policy else {},
            envelope=snap.envelope if snap is not None else frozenset(),
            trace=pane.trace,
            flash_cells=pane.flash if alpha > 0 else (),
            flash_alpha=alpha,
            trace_id=pane.trace_id,
        )

    def _draw_pane(self, surface: pygame.Surface, pane: ComparePane) -> None:
        pal, fonts = self.palette, self.fonts
        rect, pad = pane.rect, COMPARE_PANE_PADDING
        pygame.draw.rect(surface, pal.panel_bg, rect, border_radius=theme.CORNER_RADIUS)
        colour = agent_colour(pal, pane.agent)
        pygame.draw.rect(surface, colour, rect, 1, border_radius=theme.CORNER_RADIUS)
        left, width = rect.x + pad, rect.width - 2 * pad
        # header: swatch + agent name (agent colour) ... badge
        header = pygame.Rect(left, rect.y + pad, width, theme.PANE_HEADER_HEIGHT)
        bw, bh = badge_size(fonts)
        badge = pygame.Rect(header.right - bw, header.centery - bh // 2, bw, bh)
        draw_badge(surface, badge, pane.status, pal, fonts)
        r = theme.SWATCH_RADIUS
        pygame.draw.circle(surface, colour, (left + r, header.centery), r)
        name_x = left + 2 * r + theme.SECTION_GAP // 2
        name_w = badge.x - theme.SECTION_GAP // 2 - name_x
        # one font for every pane's name (narrow panes step it down for all, see layout())
        font = self.name_font
        self._text(surface, font, pane.agent, colour, name_x,
                   header.centery - font.get_height() // 2, name_w)
        # board
        pane.view.draw(surface, self.pane_scene(pane))
        # metrics: COMPARE_METRIC_COLUMNS columns of "label  value"
        small = fonts.small
        line_h = _line_h(small)
        y = pane.view.board_rect.bottom + theme.SECTION_GAP
        per_col = self._metric_rows()
        cols = COMPARE_METRIC_COLUMNS
        col_w = (width - (cols - 1) * theme.METRIC_COLUMN_GAP) // cols
        for i, (label, value) in enumerate(pane.metric_values().items()):
            col, row = divmod(i, per_col)
            x = left + col * (col_w + theme.METRIC_COLUMN_GAP)
            ry = y + row * line_h
            value_text = text_cache.fit(small, value, col_w // 2)
            vw = text_cache.width(small, value_text)
            self._text(surface, small, value_text, pal.text, x + col_w - vw, ry)
            self._text(surface, small, label, pal.text_dim, x, ry,
                       col_w - vw - 2 * theme.LINE_SPACING)
        # last event line(s)
        y += per_col * line_h + theme.LINE_SPACING
        text, kind = pane.last_event
        lines = list(text_cache.wrap(small, text, width)) if text else []
        if len(lines) > COMPARE_EVENT_LINES:  # the last shown line ends with an ellipsis
            tail = " ".join(lines[COMPARE_EVENT_LINES - 1:])
            lines = [*lines[:COMPARE_EVENT_LINES - 1], text_cache.fit(small, tail, width)]
        colour = log_kind_colour(kind, pal)
        for k, line in enumerate(lines):
            self._text(surface, small, line, colour, left, y + k * line_h, width)

    def footer_text(self) -> str:
        """Settings line of the footer (scenario, size, seed, p, r, speed, tick, state)."""
        g = self.base_grid
        o = self.options
        parts = [
            self.scenario_name or "custom map",
            f"{g.rows}x{g.cols}",
            f"seed {self.seed}",
            f"p = {self.slip_p:.2f}",
            f"r = {o.sensor_radius}",
        ]
        if o.replan_mode != "eager":
            parts.append(o.replan_mode)
        if o.aftershock_q > 0:
            parts.append(f"aftershocks q = {o.aftershock_q:g}, k = {o.aftershock_k}")
        parts += [
            f"{self.speed:g} ticks/s",
            TICK_TEXT.format(t=self.ticks),
            PAUSED_TEXT if self.paused or self.all_done else PLAYING_TEXT,
        ]
        if self.recorder.active:
            parts.append(REC_TEXT)
        return INFO_SEPARATOR.join(parts)

    def crn_text(self) -> str:
        """Footer explanation of what the panes share (aftershocks weaken "same map", D20)."""
        return CRN_AFTERSHOCK_TEXT if self.world_config().aftershock.enabled else CRN_TEXT

    def _draw_footer(self, surface: pygame.Surface) -> None:
        pal, small = self.palette, self.fonts.small
        rect = self._footer
        surface.fill(pal.panel_bg, rect)
        pygame.draw.line(surface, pal.grid_line, rect.topleft, (rect.right - 1, rect.top))
        left = rect.x + theme.MARGIN
        width = rect.width - 2 * theme.MARGIN
        y = rect.y + theme.PANEL_PADDING
        line_h = _line_h(small)
        settings = self.footer_text()
        self._text(surface, small, settings, pal.text, left, y, width)
        lines = self.log.lines
        text, kind = lines[-1] if lines else (self.crn_text(), "info")
        colour = log_kind_colour(kind, pal) if lines else pal.text_dim
        self._text(surface, small, text, colour, left, y + line_h, width)
        self._text(surface, small, COMPARE_HINT, pal.text_dim, left, y + 2 * line_h, width)

    def render(self) -> pygame.Surface:
        """Draw every pane, the footer and (if open) the help overlay; returns the surface."""
        surface = self.screen
        pal = self.palette
        surface.fill(pal.background)
        self.layout()
        for pane in self.panes:
            self._draw_pane(surface, pane)
        self._draw_footer(surface)
        if self.show_help:
            draw_help_overlay(surface, pal, self.fonts, compare_help_lines())
        return surface

    def _panel_info(self) -> PanelInfo:  # pragma: no cover - unused
        raise NotImplementedError("CompareApp has no info panel")

    def _update_tree(self) -> None:  # pragma: no cover - unused
        return None


def outcome_rows(app: CompareApp) -> list[tuple[int, tuple[str | None, ...]]]:
    """Per world time ``t`` the outcome kind each pane's step to ``t`` drew (None when the pane
    did not move to ``t``); handy to check common random numbers in tests and figures."""
    by_t: dict[int, list[str | None]] = {}
    for i, pane in enumerate(app.panes):
        for event in pane.mission.events:
            if event.outcome is None:
                continue
            row = by_t.setdefault(event.t, [None] * len(app.panes))
            row[i] = event.outcome.value
    return [(t, tuple(by_t[t])) for t in sorted(by_t)]


__all__ = [
    "CompareApp",
    "CompareOptions",
    "ComparePane",
    "agent_colour",
    "compare_help_lines",
    "outcome_rows",
    "pane_event_line",
    "parse_agents",
    "scenario_entries",
]
