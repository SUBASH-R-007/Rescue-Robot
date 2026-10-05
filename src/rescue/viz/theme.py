"""Every UI constant: palettes, fonts, layout, speeds, durations, alphas (docs/plan.md §3.11).

UI modules never hard-code a size, colour, speed or alpha: they read it from here. Two palettes
are provided, :data:`DEFAULT_PALETTE` (dark slate) and :data:`HIGH_CONTRAST_PALETTE`
(black / white / yellow, WCAG-friendly). Colour is never the only signal: every cell state and
trace mark also has its own glyph or shape (see ``grid_view.py``).

Fonts come from :func:`load_fonts`, which picks the first installed font from
:data:`FONT_FALLBACKS` that actually contains the arrow glyphs used by plan labels
(``→ ↺ … ×``), falling back to ``SysFont`` and finally pygame's built-in font.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pygame

from rescue.runner import MissionStatus

log = logging.getLogger(__name__)

RGB = tuple[int, int, int]

# --- window and layout ---------------------------------------------------------------------
#: Default window size (brief §6.1: at least 1280x800).
WINDOW_SIZE: tuple[int, int] = (1280, 800)
#: Width of the right-hand info panel.
PANEL_WIDTH: int = 470
#: Outer margin around the grid and the panel.
MARGIN: int = 16
#: Inner padding of the panel.
PANEL_PADDING: int = 14
#: Vertical gap between panel sections.
SECTION_GAP: int = 10
#: Extra pixels between text lines.
LINE_SPACING: int = 3
#: Padding inside a status badge (x, y).
BADGE_PADDING: tuple[int, int] = (10, 4)
#: Corner radius of badges and boxes.
CORNER_RADIUS: int = 6
#: Height of a plan-tree row and the indent per tree level.
TREE_ROW_HEIGHT: int = 22
TREE_INDENT: int = 18
#: Rows scrolled per mouse-wheel notch in scrollable lists.
SCROLL_STEP: int = 3
#: Alpha of the help overlay backdrop.
HELP_OVERLAY_ALPHA: int = 220
#: Height reserved for a pane title / badge row (comparison app).
PANE_HEADER_HEIGHT: int = 34

# --- info panel (panel.py) ---------------------------------------------------------------------
#: Event-log lines shown at once.
LOG_VISIBLE_LINES: int = 8
#: Smallest height given to the plan tree when the panel is short.
MIN_TREE_HEIGHT: int = 48
#: Horizontal gap between the two metric columns.
METRIC_COLUMN_GAP: int = 18
#: Radius of the agent-colour swatch and of the recording dot.
SWATCH_RADIUS: int = 6
REC_DOT_RADIUS: int = 5
#: Width of the left accent bar of an event-log line.
LOG_ACCENT_W: int = 3
#: Gap between a log line's accent bar / glyph and its text.
LOG_GLYPH_GAP: int = 6
#: Badge icon size relative to the badge font height, stroke width, gap to the label.
BADGE_ICON_FRAC: float = 0.72
BADGE_ICON_STROKE: int = 2
BADGE_ICON_GAP: int = 6
#: RUNNING icon: three dots (an ellipsis, not a play triangle - the mission may be paused):
#: dot radius and dot centres, in units of the icon size.
BADGE_DOT_FRAC: float = 0.13
BADGE_DOT_XS: tuple[float, ...] = (0.14, 0.5, 0.86)
#: Check-mark vertices of the RESCUED icon, in units of the icon size.
BADGE_CHECK_POINTS: tuple[tuple[float, float], ...] = ((0.05, 0.55), (0.38, 0.88), (0.95, 0.12))
#: Help overlay: outer margin, inner padding, column gap, maximum box width, keycap padding,
#: gap between rows.
HELP_MARGIN: int = 40
HELP_PADDING: int = 22
HELP_COLUMN_GAP: int = 36
HELP_MAX_WIDTH: int = 920
HELP_KEY_PADDING: tuple[int, int] = (6, 2)
HELP_ROW_GAP: int = 6
#: Largest share of the help items one column may take when splitting at a section heading.
HELP_MAX_COLUMN_SHARE: float = 0.7

# --- plan tree (tree_view.py) ------------------------------------------------------------------
#: Default number of Act levels shown before a subtree is truncated.
TREE_MAX_DEPTH: int = 6
#: Width of the collapse-triangle column, in pixels.
TREE_TRIANGLE_W: int = 12
#: Half-height of the drawn collapse triangle, in pixels.
TREE_TRIANGLE_SIZE: int = 4
#: Extra horizontal slack around the triangle that still counts as a click on it.
TREE_TRIANGLE_HIT_SLACK: int = 4
#: Inner padding of the tree area (x, y), in pixels.
TREE_PAD_X: int = 6
TREE_PAD_Y: int = 4
#: Width of the left accent bar of a highlighted row and of the scrollbar, in pixels.
TREE_ACCENT_W: int = 3
TREE_SCROLLBAR_W: int = 4
#: Minimum scrollbar thumb length, in pixels.
TREE_SCROLLBAR_MIN: int = 12
#: Fraction of the visible rows kept above an auto-scrolled highlighted row.
TREE_AUTOSCROLL_FRACTION: float = 0.3
#: Most rows a plan-tree message (e.g. "survivor unreachable") is wrapped over.
TREE_MESSAGE_MAX_LINES: int = 8
#: Maximum number of cached text / row surfaces (LRU) per cache.
TEXT_CACHE_CAPACITY: int = 4096

# --- fonts ---------------------------------------------------------------------------------
#: Font families tried in order (pygame ``SysFont`` names). Segoe UI lacks ``↺``, so on
#: Windows Segoe UI Symbol is normally chosen; DejaVu Sans covers everything on Linux.
FONT_FALLBACKS: tuple[str, ...] = ("segoeui", "segoeuisymbol", "dejavusans", "arial")
#: Monospace families for the event log.
MONO_FONT_FALLBACKS: tuple[str, ...] = ("consolas", "dejavusansmono", "couriernew")
#: System fonts tried (in order) for single glyphs the primary font lacks (tree_view.TextCache).
SYMBOL_FONT_NAMES: tuple[str, ...] = (
    "segoeuisymbol", "dejavusans", "notosanssymbols2", "symbola", "arialunicodems",
)
#: Glyphs the main UI font must contain (plan labels and traces).
REQUIRED_GLYPHS: str = "→↺…×"
#: Glyphs the monospace font must contain (trace lines in the log).
MONO_REQUIRED_GLYPHS: str = "→…"
#: A code point no font defines: its metrics are those of the "missing glyph" box.
_NOTDEF_PROBE: str = "￿"
FONT_SIZE_SMALL: int = 13
FONT_SIZE_BODY: int = 16
FONT_SIZE_HEADING: int = 19
FONT_SIZE_TITLE: int = 24
FONT_SIZE_MONO: int = 14

# --- timing --------------------------------------------------------------------------------
#: Frame-rate cap of the interactive apps (the grid must sustain >= 30 FPS on 30x30).
TARGET_FPS: int = 60
#: Mission steps per second selectable with +/-.
SPEEDS: tuple[float, ...] = (0.5, 1, 2, 4, 8, 16, 32)
#: Index into :data:`SPEEDS` at start-up.
DEFAULT_SPEED_INDEX: int = 2
#: Duration of the replan flash and the number of pulses within it.
REPLAN_FLASH_MS: int = 900
REPLAN_FLASH_PULSES: int = 2
#: Lines kept in the event log.
LOG_CAPACITY: int = 200
#: Slip-probability change per ``[`` / ``]`` key press.
SLIP_P_STEP: float = 0.05

# --- recording -----------------------------------------------------------------------------
#: Frames per second of recorded GIF / MP4 files.
RECORD_FPS: int = 12
#: Scale factor applied to captured frames. 1.0 keeps the panel text legible (at 0.6 the
#: ~8 px text blurred into GIF quantisation noise); use MP4 (``--mp4``) for smaller files.
RECORD_SCALE: float = 1.0
#: Default directory for recordings and screenshots.
RECORDINGS_DIR: str = "recordings"

# --- grid view -----------------------------------------------------------------------------
#: Alpha of the true-map overlay (``T``): 35 % opacity.
TRUE_MAP_ALPHA: int = round(0.35 * 255)
#: Width of the lines between cells.
GRID_LINE_WIDTH: int = 1
#: Cell size limits in pixels.
MIN_CELL_SIZE: int = 4
MAX_CELL_SIZE: int = 190
#: Below this cell size text glyphs ('?', 'S') are omitted (shapes remain).
GLYPH_MIN_CELL: int = 10
#: Font size of the unknown-cell '?' as a fraction of the cell size.
UNKNOWN_GLYPH_FRAC: float = 0.55
#: On unknown cells that carry a policy arrow the '?' moves to the top-right corner (two
#: centred glyphs never share a cell): its font size and corner inset (fractions of the cell
#: size), and the smallest cell that still gets the corner '?' (smaller cells show the arrow).
UNKNOWN_CORNER_GLYPH_FRAC: float = 0.38
UNKNOWN_CORNER_INSET_FRAC: float = 0.03
UNKNOWN_CORNER_MIN_CELL: int = 16
#: Side of the start-cell 'S' tag and its font size, as fractions of the cell size.
START_TAG_FRAC: float = 0.42
START_FONT_FRAC: float = 0.36
#: Debris cross-hatch spacing and stroke width, border width (fractions of the cell size).
HATCH_SPACING_FRAC: float = 0.25
HATCH_WIDTH_FRAC: float = 0.06
DEBRIS_BORDER_FRAC: float = 0.08
#: High-slip ('~') wave strokes.
WAVE_COUNT: int = 3
WAVE_AMPLITUDE_FRAC: float = 0.07
WAVE_PERIODS: float = 1.5
WAVE_MARGIN_FRAC: float = 0.14
WAVE_WIDTH_FRAC: float = 0.06
WAVE_SAMPLES: int = 16
#: Survivor cross: arm length and thickness (fractions of the cell size).
SURVIVOR_LEN_FRAC: float = 0.62
SURVIVOR_THICK_FRAC: float = 0.24
#: Robot disc radius and outline width.
ROBOT_RADIUS_FRAC: float = 0.33
ROBOT_OUTLINE_FRAC: float = 0.08
#: Robot on the survivor (rescue): a smaller disc, so the arms of the survivor cross stay
#: visible around it, inside a ring in the survivor colour (radius, width).
ROBOT_ON_SURVIVOR_RADIUS_FRAC: float = 0.19
RESCUE_RING_RADIUS_FRAC: float = 0.45
RESCUE_RING_WIDTH_FRAC: float = 0.06
#: Tint of cells in the current plan envelope.
ENVELOPE_ALPHA: int = 38
#: Policy arrows: total length, head length / half-width, shaft width, alpha.
POLICY_ARROW_LEN_FRAC: float = 0.56
POLICY_ARROW_HEAD_FRAC: float = 0.24
POLICY_ARROW_HALF_WIDTH_FRAC: float = 0.17
POLICY_ARROW_SHAFT_FRAC: float = 0.07
POLICY_ARROW_ALPHA: int = 190
#: Width of the contrasting halo (``Palette.arrow_outline``) around a policy arrow, so arrows
#: stay visible on dark unknown cells as well as on light open ones.
POLICY_ARROW_OUTLINE_FRAC: float = 0.04
#: Executed-trace strokes.
TRACE_LINE_FRAC: float = 0.09
#: Smallest trace stroke / mark size in pixels on cells of at least GLYPH_MIN_CELL (thin
#: 1-2 px strokes were unreadable on 25 px cells).
TRACE_MIN_STROKE: int = 3
#: Direction chevron in the middle of an arrived segment: size (fraction of the cell, at least
#: TRACE_MIN_CHEVRON px).
TRACE_CHEVRON_FRAC: float = 0.13
TRACE_MIN_CHEVRON: int = 4
TRACE_RING_RADIUS_FRAC: float = 0.43
TRACE_RING_WIDTH_FRAC: float = 0.07
SKID_ARROW_HEAD_FRAC: float = 0.22
SKID_ARROW_END_FRAC: float = 0.18
#: Half-width of the skid arrow head relative to its length.
SKID_ARROW_HALF_WIDTH: float = 0.6
BLOCKED_X_FRAC: float = 0.13
BLOCKED_OFFSET_FRAC: float = 0.36
#: Minimum stroke width of any trace / glyph line.
MIN_STROKE: int = 1
#: Lower bounds (pixels) of glyph parts on tiny cells: debris hatch spacing, survivor-cross
#: thickness, start-tag side, radius of the robot disc / slip ring, half-size of the blocked
#: '×', wave amplitude and policy-arrow shaft width.
MIN_HATCH_SPACING: int = 3
MIN_SURVIVOR_THICK: int = 2
MIN_START_TAG: int = 3
MIN_MARK_RADIUS: int = 2
MIN_BLOCKED_X: float = 3.0
#: Smallest skid arrow head (pixels).
MIN_SKID_HEAD: float = 6.0
MIN_WAVE_AMPLITUDE: float = 1.0
MIN_ARROW_SHAFT: float = 1.0
#: Replan flash: fill alpha at full strength and outline width.
FLASH_FILL_ALPHA: int = 120
FLASH_OUTLINE_FRAC: float = 0.1
#: Editor-mode dashed border.
DASH_LENGTH: int = 10
DASH_GAP: int = 6
EDITOR_BORDER_WIDTH: int = 3
EDITOR_BORDER_PAD: int = 4

# --- plan-tree export (matplotlib) -------------------------------------------------------------
EXPORT_DPI: int = 150
#: Layout direction of exported trees: "lr" (root on the left, levels as columns, leaves as
#: rows - suits the deep, narrow AND-OR plans) or "tb" (root on top).
EXPORT_LAYOUT: str = "lr"
#: "tb" layout: inches per leaf (horizontal) and per level (vertical).
EXPORT_LEAF_WIDTH_IN: float = 1.55
EXPORT_LEVEL_HEIGHT_IN: float = 1.05
#: "lr" layout: inches per level (horizontal) and per leaf row (vertical).
EXPORT_LR_LEVEL_WIDTH_IN: float = 2.35
EXPORT_LR_LEAF_HEIGHT_IN: float = 0.42
#: Position of an edge label along its edge (0 = parent, 1 = child): near the child, so the
#: label is next to the node it leads to.
EXPORT_EDGE_LABEL_POS: float = 0.62
EXPORT_MIN_SIZE_IN: tuple[float, float] = (6.0, 3.0)
EXPORT_MAX_WIDTH_IN: float = 80.0
#: Largest width:height (or height:width) ratio of an exported figure; the short side grows.
EXPORT_MAX_ASPECT: float = 2.5
EXPORT_MARGIN_IN: float = 0.6
EXPORT_FONT_SIZE: float = 9.0
EXPORT_EDGE_FONT_SIZE: float = 7.5
EXPORT_TITLE_FONT_SIZE: float = 12.0
EXPORT_BOX_PAD: float = 0.35
EXPORT_EDGE_WIDTH: float = 1.1
#: Half the height of a node box in points (edges attach to the box top / bottom).
EXPORT_BOX_HALF_HEIGHT_PT: float = 9.5
#: Padding of the white box behind edge labels.
EXPORT_EDGE_LABEL_PAD: float = 0.15
#: Vertical space reserved for the title, in title-font lines.
EXPORT_TITLE_LINES: float = 2.2
#: Typographic points per inch.
POINTS_PER_INCH: float = 72.0
EXPORT_BACKGROUND: str = "#ffffff"
EXPORT_TEXT: str = "#1d232e"
EXPORT_EDGE_COLOUR: str = "#7a8394"
#: Node box (face, edge) colours and line style per ``to_tree`` kind.
EXPORT_NODE_STYLE: dict[str, tuple[str, str, str]] = {
    "act": ("#dbe8ff", "#2f5fb3", "solid"),
    "goal": ("#d8f3dc", "#2b8a3e", "solid"),
    "loop": ("#fff1d6", "#c77c02", "solid"),
    "ref": ("#ece4ff", "#6741d9", "solid"),
    "truncated": ("#eef0f3", "#868e96", "dashed"),
}
#: Edge-label colours by label prefix.
EXPORT_EDGE_LABEL_COLOURS: dict[str, str] = {
    "arrived": "#2b8a3e",
    "slipped": "#b86e00",
    "skid": "#c92a2a",
    "blocked": "#495057",
}
#: Indent per level of :func:`rescue.viz.export.export_plan_text`.
EXPORT_TEXT_INDENT: str = "  "


# --- palettes ----------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Palette:
    """All UI colours. ``badge`` maps mission statuses, ``agent`` maps agent ids to colours.

    Compared and hashed by identity (it holds dicts), which makes it a cheap cache key.
    """

    name: str
    background: RGB
    panel_bg: RGB
    text: RGB
    text_dim: RGB
    grid_line: RGB
    unknown: RGB
    unknown_glyph: RGB
    open: RGB
    debris: RGB
    debris_glyph: RGB
    debris_border: RGB
    high_slip: RGB
    high_slip_glyph: RGB
    survivor: RGB
    robot: RGB
    robot_outline: RGB
    start: RGB
    start_text: RGB
    arrow: RGB
    arrow_outline: RGB
    envelope: RGB
    trace_ok: RGB
    trace_slip: RGB
    trace_skid: RGB
    replan_flash: RGB
    highlight: RGB
    editor_border: RGB
    badge_text: RGB
    #: Text colours (>= 4.5:1 on ``background``, the tree / log background), kept apart from
    #: the grid-mark colours above: branch labels and log lines of each kind.
    edge_ok_text: RGB
    edge_slip_text: RGB
    edge_skid_text: RGB
    log_replan_text: RGB
    log_success_text: RGB
    log_failure_text: RGB
    log_warn_text: RGB
    #: Solid fill of the highlighted (current-state) tree row and the text drawn on it.
    highlight_row: RGB
    highlight_text: RGB
    badge: dict[MissionStatus, RGB] = field(default_factory=dict)
    agent: dict[str, RGB] = field(default_factory=dict)

    def badge_colour(self, status: MissionStatus) -> RGB:
        """Badge colour of ``status`` (``text_dim`` if the palette has none)."""
        return self.badge.get(status, self.text_dim)

    def agent_colour(self, agent: str) -> RGB:
        """Colour of an agent id (``highlight`` for unknown ids)."""
        return self.agent.get(agent, self.highlight)


DEFAULT_PALETTE = Palette(
    name="default",
    background=(24, 28, 36),
    panel_bg=(33, 39, 50),
    text=(230, 234, 240),
    text_dim=(148, 158, 174),
    grid_line=(70, 78, 92),
    unknown=(42, 48, 60),
    unknown_glyph=(150, 160, 178),
    open=(222, 226, 232),
    debris=(164, 112, 72),
    debris_glyph=(104, 66, 38),
    debris_border=(74, 46, 26),
    high_slip=(112, 196, 200),
    high_slip_glyph=(22, 104, 120),
    survivor=(218, 44, 56),
    robot=(46, 116, 232),
    robot_outline=(245, 248, 255),
    start=(52, 64, 86),
    start_text=(240, 244, 250),
    arrow=(44, 54, 74),
    arrow_outline=(230, 234, 240),
    envelope=(84, 160, 255),
    trace_ok=(36, 168, 84),
    trace_slip=(240, 160, 24),
    trace_skid=(214, 40, 48),
    replan_flash=(230, 64, 230),
    highlight=(255, 208, 80),
    editor_border=(255, 208, 80),
    badge_text=(255, 255, 255),
    edge_ok_text=(90, 210, 130),
    edge_slip_text=(245, 170, 50),
    edge_skid_text=(255, 110, 110),
    log_replan_text=(240, 120, 240),
    log_success_text=(90, 210, 130),
    log_failure_text=(255, 110, 110),
    log_warn_text=(245, 170, 50),
    highlight_row=(72, 62, 26),
    highlight_text=(240, 242, 246),
    # White badge text needs >= 4.5:1, hence these fairly dark badge colours.
    badge={
        MissionStatus.RUNNING: (40, 96, 190),
        MissionStatus.RESCUED: (25, 120, 60),
        MissionStatus.ABORTED: (190, 40, 40),
        MissionStatus.BUDGET: (150, 88, 0),
        MissionStatus.UNREACHABLE: (112, 70, 170),
    },
    agent={
        "andor": (72, 200, 160),
        "astar_fixed": (240, 138, 60),
        "astar_replan": (176, 122, 240),
    },
)

HIGH_CONTRAST_PALETTE = Palette(
    name="high-contrast",
    background=(0, 0, 0),
    panel_bg=(0, 0, 0),
    text=(255, 255, 255),
    text_dim=(235, 235, 235),
    grid_line=(128, 128, 128),
    unknown=(0, 0, 0),
    unknown_glyph=(255, 255, 255),
    open=(255, 255, 255),
    debris=(0, 0, 0),
    debris_glyph=(255, 255, 0),
    debris_border=(255, 255, 0),
    high_slip=(150, 230, 255),
    high_slip_glyph=(0, 0, 0),
    survivor=(200, 0, 0),
    robot=(0, 70, 255),
    robot_outline=(0, 0, 0),
    start=(0, 0, 0),
    start_text=(255, 255, 0),
    arrow=(0, 0, 0),
    arrow_outline=(255, 255, 255),
    envelope=(255, 255, 0),
    trace_ok=(0, 135, 0),
    trace_slip=(190, 90, 0),
    trace_skid=(220, 0, 0),
    replan_flash=(255, 0, 255),
    highlight=(255, 255, 0),
    editor_border=(255, 255, 0),
    badge_text=(0, 0, 0),
    edge_ok_text=(0, 255, 0),
    edge_slip_text=(255, 170, 0),
    edge_skid_text=(255, 128, 128),
    log_replan_text=(255, 128, 255),
    log_success_text=(0, 255, 0),
    log_failure_text=(255, 128, 128),
    log_warn_text=(255, 255, 0),
    highlight_row=(255, 255, 0),
    highlight_text=(0, 0, 0),
    badge={
        MissionStatus.RUNNING: (255, 255, 255),
        MissionStatus.RESCUED: (0, 255, 0),
        MissionStatus.ABORTED: (255, 80, 80),
        MissionStatus.BUDGET: (255, 255, 0),
        MissionStatus.UNREACHABLE: (255, 150, 255),
    },
    agent={
        "andor": (0, 255, 0),
        "astar_fixed": (255, 255, 0),
        "astar_replan": (0, 255, 255),
    },
)

#: Palettes in the order the ``C`` key cycles them.
PALETTES: tuple[Palette, ...] = (DEFAULT_PALETTE, HIGH_CONTRAST_PALETTE)


# --- fonts -------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Fonts:
    """The UI fonts. :meth:`sized` makes (and caches) the same family at any size.

    ``path`` is the resolved font file (None = pygame's built-in font); ``mono_path`` the
    monospace one.
    """

    small: pygame.font.Font
    body: pygame.font.Font
    heading: pygame.font.Font
    title: pygame.font.Font
    mono: pygame.font.Font
    path: str | None = None
    mono_path: str | None = None
    _cache: dict[tuple[int, bool], pygame.font.Font] = field(
        default_factory=dict, repr=False
    )

    def sized(self, size: int, bold: bool = False) -> pygame.font.Font:
        """The main UI font at ``size`` points (cached per size and weight)."""
        key = (max(1, int(size)), bool(bold))
        font = self._cache.get(key)
        if font is None:
            font = _make_font(self.path, key[0], bold=key[1])
            self._cache[key] = font
        return font


def _make_font(path: str | None, size: int, bold: bool = False) -> pygame.font.Font:
    """``pygame.font.Font(path, size)`` with synthetic bold."""
    font = pygame.font.Font(path, size)
    font.bold = bold
    return font


def _covers(path: str, glyphs: str) -> bool:
    """True if the font file at ``path`` has a real glyph for every char of ``glyphs``."""
    try:
        font = pygame.font.Font(path, FONT_SIZE_BODY)
    except (OSError, pygame.error):
        return False
    missing = font.metrics(_NOTDEF_PROBE)[0]
    metrics = font.metrics(glyphs)
    return all(m is not None and m != missing for m in metrics)


def resolve_font_path(names: tuple[str, ...], glyphs: str) -> str | None:
    """Path of the first installed family in ``names`` covering ``glyphs``, else of the first
    installed one at all, else None (pygame's built-in font)."""
    first: str | None = None
    for name in names:
        path = pygame.font.match_font(name)
        if not path:
            continue
        first = first or path
        if _covers(path, glyphs):
            return path
    if first is not None:
        log.info("no font in %s covers %r; using %s", names, glyphs, first)
    return first


def load_fonts() -> Fonts:
    """Load the UI fonts (initialises ``pygame.font`` if needed).

    Families are tried in :data:`FONT_FALLBACKS` order, keeping the first that has the arrow
    glyphs; if none is installed, ``pygame.font.SysFont`` with the comma-separated list and
    finally ``pygame.font.Font(None, size)`` are used.
    """
    if not pygame.font.get_init():
        pygame.font.init()
    path = resolve_font_path(FONT_FALLBACKS, REQUIRED_GLYPHS)
    mono_path = resolve_font_path(MONO_FONT_FALLBACKS, MONO_REQUIRED_GLYPHS)
    if mono_path is None or not _covers(mono_path, MONO_REQUIRED_GLYPHS):
        mono_path = path

    def make(p: str | None, size: int, bold: bool = False) -> pygame.font.Font:
        if p is not None:
            try:
                return _make_font(p, size, bold)
            except (OSError, pygame.error):
                log.warning("could not load font %s; falling back", p)
        try:
            font = pygame.font.SysFont(",".join(FONT_FALLBACKS), size, bold=bold)
        except (OSError, pygame.error):
            font = _make_font(None, size, bold)
        return font

    return Fonts(
        small=make(path, FONT_SIZE_SMALL),
        body=make(path, FONT_SIZE_BODY),
        heading=make(path, FONT_SIZE_HEADING, bold=True),
        title=make(path, FONT_SIZE_TITLE, bold=True),
        mono=make(mono_path, FONT_SIZE_MONO),
        path=path,
        mono_path=mono_path,
    )


# --- small helpers -------------------------------------------------------------------------
def grid_area(window_size: tuple[int, int] = WINDOW_SIZE) -> pygame.Rect:
    """The rectangle left of the panel available to the grid view."""
    width, height = window_size
    return pygame.Rect(MARGIN, MARGIN, width - PANEL_WIDTH - 2 * MARGIN, height - 2 * MARGIN)


def panel_area(window_size: tuple[int, int] = WINDOW_SIZE) -> pygame.Rect:
    """The rectangle of the right-hand info panel."""
    width, height = window_size
    return pygame.Rect(width - PANEL_WIDTH, 0, PANEL_WIDTH, height)


def relative_luminance(rgb: RGB) -> float:
    """WCAG 2 relative luminance of an sRGB colour (0 = black, 1 = white)."""

    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: RGB, b: RGB) -> float:
    """WCAG 2 contrast ratio of two colours (1 .. 21; text needs >= 4.5 for AA)."""
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def stroke(cell: int, frac: float) -> int:
    """A line width of ``frac`` x ``cell`` pixels, at least :data:`MIN_STROKE`."""
    return max(MIN_STROKE, round(cell * frac))


def flash_alpha(elapsed_ms: float, duration_ms: float = REPLAN_FLASH_MS) -> float:
    """Strength (0..1) of the replan flash ``elapsed_ms`` after it started.

    Pulses :data:`REPLAN_FLASH_PULSES` times while fading out linearly; 0 once
    ``duration_ms`` has passed (or for negative times).
    """
    if duration_ms <= 0 or not 0 <= elapsed_ms < duration_ms:
        return 0.0
    phase = elapsed_ms / duration_ms
    pulse = 0.5 + 0.5 * math.cos(2.0 * math.pi * REPLAN_FLASH_PULSES * phase)
    return max(0.0, min(1.0, (1.0 - phase) * (0.35 + 0.65 * pulse)))


# --- apps (sim_app.py / replay_app.py) ---------------------------------------------------------
#: Frames rendered by a headless run when no frame count is given.
HEADLESS_FRAMES: int = 100
#: Simulated frame time of a headless run (the apps advance their clocks by this per frame).
HEADLESS_FRAME_MS: float = 1000.0 / TARGET_FPS
#: Longest frame time fed to the step accumulator (a stalled frame does not burst steps).
MAX_FRAME_MS: float = 250.0
#: Most mission steps taken in a single frame at high speeds.
MAX_STEPS_PER_FRAME: int = 8
#: Act depth of the plan tree shown in the panel.
PANEL_TREE_DEPTH: int = 6
#: Replay: progress bar height, gap above it, and the Shift+arrow scrub stride.
PROGRESS_BAR_HEIGHT: int = 24
PROGRESS_BAR_GAP: int = 8
SCRUB_BIG_STEP: int = 10
#: Alpha of the played part of the replay progress bar, and width of its position marker.
PROGRESS_FILL_ALPHA: int = 76
PROGRESS_MARKER_WIDTH: int = 2
#: Gap between the progress bar and its "step k / n" label (drawn right of the bar, so it
#: never covers the replan ticks or the position marker), and the width of a replan tick.
PROGRESS_LABEL_GAP: int = 10
PROGRESS_TICK_WIDTH: int = 2
#: Window titles.
SIM_CAPTION: str = "Rescue Robot - simulator"
REPLAY_CAPTION: str = "Rescue Robot - replay"
#: Default size of maps produced by ``rescue.mapgen`` scenarios in the simulator (rows, cols).
GENERATED_MAP_SIZE: tuple[int, int] = (15, 15)
