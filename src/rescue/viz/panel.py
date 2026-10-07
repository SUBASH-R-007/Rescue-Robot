"""Right-hand info panel, event log, status badges and the help overlay (docs/plan.md §3.11).

The panel is a pure renderer: the app fills a :class:`PanelInfo` (title, agent, status, run
settings, pre-formatted metrics) from the :class:`~rescue.runner.Mission` and its
:class:`~rescue.runner.StepEvent`s, appends human-readable lines to an :class:`EventLog`, and
calls :meth:`Panel.draw` once per frame. Layout, top to bottom::

    title                                   [badge]
    ● agent name
    seed · p · scenario · speed · paused · ● REC
    ───────────────────────────────────────────────
    metrics (two columns)
    PLAN TREE   (TreeView)
    EVENT LOG   (newest line at the bottom)
    H help · …  (one-line key hint)

Status is never shown by colour alone: :func:`draw_badge` pairs every colour with a text label
and an icon shape, and log lines carry a per-kind glyph. Long log lines are word-wrapped (with
a hanging indent under the glyph column); the log header counts the lines scrolled out of view.
The RUNNING badge uses a neutral "…" icon: play / pause is shown only in the settings line.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygame

from rescue.runner import MissionStatus
from rescue.viz.theme import (
    BADGE_CHECK_POINTS,
    BADGE_DOT_FRAC,
    BADGE_DOT_XS,
    BADGE_ICON_FRAC,
    BADGE_ICON_GAP,
    BADGE_ICON_STROKE,
    BADGE_PADDING,
    CORNER_RADIUS,
    HELP_COLUMN_GAP,
    HELP_KEY_PADDING,
    HELP_MARGIN,
    HELP_MAX_COLUMN_SHARE,
    HELP_MAX_WIDTH,
    HELP_OVERLAY_ALPHA,
    HELP_PADDING,
    HELP_ROW_GAP,
    LEGEND_SWATCH_GAP,
    LINE_SPACING,
    LOG_ACCENT_W,
    LOG_CAPACITY,
    LOG_GLYPH_GAP,
    LOG_VISIBLE_LINES,
    METRIC_COLUMN_GAP,
    MIN_TREE_HEIGHT,
    PANEL_PADDING,
    REC_DOT_RADIUS,
    RGB,
    SCROLL_STEP,
    SECTION_GAP,
    SWATCH_RADIUS,
)
from rescue.viz.tree_view import TreeView, text_cache

if TYPE_CHECKING:
    from rescue.viz.theme import Fonts, Palette

# Sizes, gaps and alphas used here live in viz/theme.py (docs/plan.md §3.11).
#: Fixed texts.
SECTION_PLAN = "PLAN TREE"
SECTION_LOG = "EVENT LOG"
INFO_SEPARATOR = " · "
DEFAULT_HINT = "H help · Space play/pause · N step · Esc quit"
HELP_TITLE = "Controls"
#: Glyph whose height is the font's cap height (optical centring of keycap labels).
CAP_HEIGHT_PROBE = "H"
HELP_FOOTER = "Press H or Esc to close"
LOG_OLDER_TEXT = "↑ {n} older"
LOG_NEWER_TEXT = "↓ {n} newer"
LOG_SCROLL_SEPARATOR = " · "
#: Label of the badge shown while the map editor is open (no mission status applies).
EDITING_LABEL = "EDITING"
REC_TEXT = "REC"
PLAYING_TEXT = "▶ playing"
PAUSED_TEXT = "❚❚ paused"

#: Event-log line kinds.
LOG_KINDS: tuple[str, ...] = ("info", "slip", "skid", "replan", "success", "failure", "warn")
#: Glyph shown before a log line of each kind (shape besides colour).
LOG_KIND_GLYPHS: dict[str, str] = {
    "info": "•",
    "slip": "~",
    "skid": "»",
    "replan": "↻",
    "success": "✓",
    "failure": "✗",
    "warn": "!",
}
#: Order in which statuses are measured for the (fixed) badge width.
_BADGE_STATUSES: tuple[MissionStatus, ...] = tuple(MissionStatus)


def log_kind_colour(kind: str, palette: Palette) -> RGB:
    """Text colour of an event-log line of ``kind`` (the palette's text colours, readable on
    ``palette.background``)."""
    if kind == "slip":
        return palette.edge_slip_text
    if kind == "skid":
        return palette.edge_skid_text
    if kind == "replan":
        return palette.log_replan_text
    if kind == "success":
        return palette.log_success_text
    if kind == "failure":
        return palette.log_failure_text
    if kind == "warn":
        return palette.log_warn_text
    return palette.text


# --- event log ---------------------------------------------------------------------------------
class EventLog:
    """Bounded log of ``(text, kind)`` lines, newest last, with a scroll offset.

    ``scroll`` counts lines scrolled up from the newest one (0 = following the tail). While
    scrolled up, new lines do not move the view.
    """

    def __init__(self, capacity: int = LOG_CAPACITY) -> None:
        self._lines: deque[tuple[str, str]] = deque(maxlen=capacity)
        self.scroll = 0
        #: Incremented on every change (lets renderers cache).
        self.version = 0

    def add(self, text: str, kind: str = "info") -> None:
        """Append a line; ``kind`` is one of :data:`LOG_KINDS` (ValueError otherwise)."""
        if kind not in LOG_KINDS:
            raise ValueError(f"unknown log kind {kind!r}; choose from {', '.join(LOG_KINDS)}")
        full = len(self._lines) == self._lines.maxlen
        self._lines.append((text, kind))
        if self.scroll and not full:  # keep the viewed lines in place
            self.scroll += 1
        self.version += 1

    def clear(self) -> None:
        """Remove every line and reset the scroll offset."""
        self._lines.clear()
        self.scroll = 0
        self.version += 1

    @property
    def lines(self) -> list[tuple[str, str]]:
        """All lines, oldest first."""
        return list(self._lines)

    def __len__(self) -> int:
        return len(self._lines)

    def scroll_by(self, delta: int, visible: int = LOG_VISIBLE_LINES) -> None:
        """Scroll ``delta`` lines towards older lines (negative = newer), clamped."""
        self.scroll = min(max(self.scroll + delta, 0), max(0, len(self._lines) - visible))

    def bounds(self, visible: int) -> tuple[int, int]:
        """``(start, end)`` indices of the ``visible`` lines currently in view."""
        n = len(self._lines)
        end = n - min(self.scroll, max(0, n - visible))
        return max(0, end - visible), end

    def window(self, visible: int) -> list[tuple[str, str]]:
        """The ``visible`` lines currently in view, oldest first."""
        start, end = self.bounds(visible)
        return [self._lines[i] for i in range(start, end)]


# --- panel -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Legend:
    """A one-row legend drawn under the metrics (extension simulator, plan §3.15.3):
    ``title``, then ``low`` · the ``swatches`` left to right · ``high``."""

    title: str
    swatches: tuple[pygame.Surface, ...]
    low: str = ""
    high: str = ""


@dataclass
class PanelInfo:
    """Everything the panel shows besides the plan tree and the log (all pre-formatted).

    ``metrics`` maps display labels to display values, shown in two columns in mapping order
    (left column first). ``speed`` is in mission steps per second. ``editing`` replaces the
    status badge by an EDITING badge (the map editor has no mission status).
    """

    title: str
    agent: str
    status: MissionStatus
    seed: int
    slip_p: float
    scenario: str
    metrics: Mapping[str, str] = field(default_factory=dict)
    speed: float = 1.0
    paused: bool = True
    recording: bool = False
    hint: str = DEFAULT_HINT
    editing: bool = False
    #: Optional legend under the metrics (None draws nothing: the core panel is unchanged).
    legend: Legend | None = None


def _line_h(font: pygame.font.Font) -> int:
    return font.get_linesize() + LINE_SPACING


def badge_size(fonts: Fonts) -> tuple[int, int]:
    """Size of a status badge: wide enough for every status, so it never jitters."""
    font = fonts.small
    icon = round(font.get_height() * BADGE_ICON_FRAC)
    labels = [status.name for status in _BADGE_STATUSES] + [EDITING_LABEL]
    text_w = max(text_cache.width(font, label) for label in labels)
    pad_x, pad_y = BADGE_PADDING
    return 2 * pad_x + icon + BADGE_ICON_GAP + text_w, font.get_height() + 2 * pad_y


def _draw_badge_icon(
    surface: pygame.Surface, box: pygame.Rect, status: MissionStatus, colour: RGB
) -> None:
    """Shape that identifies ``status`` without colour (… / check / X / hourglass / ⦸).

    RUNNING is a neutral ellipsis, not a play triangle: the mission may be paused.
    """
    s, w = box.width, BADGE_ICON_STROKE
    left, top, right, bottom = box.left, box.top, box.right - 1, box.bottom - 1
    if status is MissionStatus.RUNNING:
        radius = max(1, round(s * BADGE_DOT_FRAC))
        for fx in BADGE_DOT_XS:
            pygame.draw.circle(surface, colour, (round(left + fx * s), box.centery), radius)
    elif status is MissionStatus.RESCUED:
        pts = [(left + fx * s, top + fy * s) for fx, fy in BADGE_CHECK_POINTS]
        pygame.draw.lines(surface, colour, False, pts, w + 1)
    elif status is MissionStatus.ABORTED:
        pygame.draw.line(surface, colour, (left, top), (right, bottom), w + 1)
        pygame.draw.line(surface, colour, (left, bottom), (right, top), w + 1)
    elif status is MissionStatus.BUDGET:  # hourglass
        pygame.draw.polygon(surface, colour, [(left, top), (right, top), box.center])
        pygame.draw.polygon(surface, colour, [box.center, (left, bottom), (right, bottom)])
    else:  # UNREACHABLE: circle with a slash
        pygame.draw.circle(surface, colour, box.center, s // 2, w)
        off = round(s / 2 * math.sqrt(0.5))
        cx, cy = box.center
        pygame.draw.line(surface, colour, (cx - off, cy + off), (cx + off, cy - off), w)


def draw_badge(
    surface: pygame.Surface,
    rect: pygame.Rect,
    status: MissionStatus,
    palette: Palette,
    fonts: Fonts,
) -> None:
    """Rounded status badge in ``rect``: icon shape + status name on the status colour."""
    rect = pygame.Rect(rect)
    font = fonts.small
    pygame.draw.rect(surface, palette.badge_colour(status), rect, border_radius=CORNER_RADIUS)
    fg = palette.badge_text
    icon = round(font.get_height() * BADGE_ICON_FRAC)
    label = text_cache.render(font, status.name, fg)
    content_w = icon + BADGE_ICON_GAP + label.get_width()
    x = rect.x + max(BADGE_PADDING[0], (rect.width - content_w) // 2)
    box = pygame.Rect(x, rect.centery - icon // 2, icon, icon)
    _draw_badge_icon(surface, box, status, fg)
    surface.blit(label, (box.right + BADGE_ICON_GAP, rect.centery - label.get_height() // 2))


def draw_editing_badge(surface: pygame.Surface, rect: pygame.Rect, palette: Palette,
                       fonts: Fonts) -> None:
    """Badge shown while the map editor is open: a pencil + EDITING, dark on the highlight."""
    rect = pygame.Rect(rect)
    font = fonts.small
    pygame.draw.rect(surface, palette.highlight, rect, border_radius=CORNER_RADIUS)
    fg = palette.background
    icon = round(font.get_height() * BADGE_ICON_FRAC)
    label = text_cache.render(font, EDITING_LABEL, fg)
    content_w = icon + BADGE_ICON_GAP + label.get_width()
    x = rect.x + max(BADGE_PADDING[0], (rect.width - content_w) // 2)
    box = pygame.Rect(x, rect.centery - icon // 2, icon, icon)
    tip = (box.left, box.bottom - 1)
    tail = (box.right - 1, box.top)
    pygame.draw.line(surface, fg, tip, tail, BADGE_ICON_STROKE + 2)  # pencil body
    pygame.draw.circle(surface, fg, tip, BADGE_ICON_STROKE)  # its point
    surface.blit(label, (box.right + BADGE_ICON_GAP, rect.centery - label.get_height() // 2))


def _baseline_offsets(fonts: Sequence[pygame.font.Font], line_h: int) -> list[int]:
    """Top offsets that put every font of ``fonts`` on one shared baseline, with the tallest
    ascent + descent centred in ``line_h``."""
    ascent = max(f.get_ascent() for f in fonts)
    descent = max(-f.get_descent() for f in fonts)
    baseline = (line_h - ascent - descent) // 2 + ascent
    return [baseline - f.get_ascent() for f in fonts]


class Panel:
    """The simulator's info panel; see the module docstring for the layout."""

    def __init__(self, rect: pygame.Rect, palette: Palette, fonts: Fonts) -> None:
        self.rect = pygame.Rect(rect)
        self.palette = palette
        self.fonts = fonts
        #: Screen rects of the last drawn tree and log areas (for event routing).
        self.tree_rect = pygame.Rect(0, 0, 0, 0)
        self.log_rect = pygame.Rect(0, 0, 0, 0)

    def set_rect(self, rect: pygame.Rect) -> None:
        """Move/resize the panel."""
        self.rect = pygame.Rect(rect)

    def set_palette(self, palette: Palette, fonts: Fonts | None = None) -> None:
        """Switch palette (and optionally fonts)."""
        self.palette = palette
        if fonts is not None:
            self.fonts = fonts

    # --- events ----------------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event, log: EventLog, tree: TreeView) -> bool:
        """Route mouse events: tree clicks/wheel to ``tree``, wheel over the log scrolls it."""
        if tree.handle_event(event):
            return True
        if event.type == pygame.MOUSEWHEEL and self.log_rect.collidepoint(pygame.mouse.get_pos()):
            log.scroll_by(event.y * SCROLL_STEP, LOG_VISIBLE_LINES)
            return True
        return False

    # --- drawing ---------------------------------------------------------------------------
    def _text(self, surface: pygame.Surface, font: pygame.font.Font, text: str, colour: RGB,
              x: int, y: int, max_w: int | None = None) -> int:
        """Blit (fitted) text at (x, y); returns its width."""
        if max_w is not None:
            text = text_cache.fit(font, text, max_w)
        if not text:
            return 0
        surf = text_cache.render(font, text, colour)
        surface.blit(surf, (x, y))
        return surf.get_width()

    def draw(self, surface: pygame.Surface, info: PanelInfo, log: EventLog, tree: TreeView) -> None:
        """Render the whole panel (and ``tree`` inside it) into :attr:`rect`."""
        pal, fonts, rect = self.palette, self.fonts, self.rect
        old_clip = surface.get_clip()
        surface.set_clip(rect.clip(old_clip))
        try:
            surface.fill(pal.panel_bg, rect)
            pygame.draw.line(surface, pal.grid_line, rect.topleft, rect.bottomleft)
            left = rect.x + PANEL_PADDING
            width = rect.width - 2 * PANEL_PADDING
            y = self._draw_header(surface, info, left, rect.y + PANEL_PADDING, width)
            y = self._draw_settings(surface, info, left, y, width)
            y += SECTION_GAP // 2
            pygame.draw.line(surface, pal.grid_line, (left, y), (left + width - 1, y))
            y += SECTION_GAP
            y = self._draw_metrics(surface, info.metrics, left, y, width)
            if info.legend is not None:
                y = self._draw_legend(surface, info.legend, left, y + SECTION_GAP // 2, width)

            footer_h = _line_h(fonts.small)
            footer_y = rect.bottom - PANEL_PADDING - footer_h
            log_line_h = _line_h(fonts.mono)
            log_h = _line_h(fonts.small) + LOG_VISIBLE_LINES * log_line_h
            log_y = footer_y - SECTION_GAP - log_h
            tree_title_y = y + SECTION_GAP
            tree_y = tree_title_y + _line_h(fonts.small)
            tree_h = max(MIN_TREE_HEIGHT, log_y - SECTION_GAP - tree_y)
            log_y = max(log_y, tree_y + tree_h + SECTION_GAP)

            self._text(surface, fonts.small, SECTION_PLAN, pal.text_dim, left, tree_title_y)
            self.tree_rect = pygame.Rect(left, tree_y, width, tree_h)
            tree.set_rect(self.tree_rect)
            tree.draw(surface)

            self._draw_log(surface, log, left, log_y, width, log_line_h)
            pygame.draw.line(surface, pal.grid_line, (left, footer_y - LINE_SPACING),
                             (left + width - 1, footer_y - LINE_SPACING))
            self._text(surface, fonts.small, info.hint, pal.text_dim, left, footer_y, width)
        finally:
            surface.set_clip(old_clip)

    def _draw_header(self, surface: pygame.Surface, info: PanelInfo, left: int, y: int,
                     width: int) -> int:
        """Title + badge, then the agent name with its colour swatch; returns the next y."""
        pal, fonts = self.palette, self.fonts
        bw, bh = badge_size(fonts)
        title_h = fonts.title.get_linesize()
        badge = pygame.Rect(left + width - bw, y + (title_h - bh) // 2, bw, bh)
        if info.editing:
            draw_editing_badge(surface, badge, pal, fonts)
        else:
            draw_badge(surface, badge, info.status, pal, fonts)
        self._text(surface, fonts.title, info.title, pal.text, left, y,
                   width - bw - SECTION_GAP)
        y += title_h + LINE_SPACING
        colour = pal.agent_colour(info.agent)
        body_h = fonts.body.get_height()
        pygame.draw.circle(surface, colour, (left + SWATCH_RADIUS, y + body_h // 2),
                           SWATCH_RADIUS)
        name_x = left + 2 * SWATCH_RADIUS + SECTION_GAP // 2
        self._text(surface, fonts.body, info.agent, colour, name_x, y, left + width - name_x)
        return y + _line_h(fonts.body)

    def _draw_settings(self, surface: pygame.Surface, info: PanelInfo, left: int, y: int,
                       width: int) -> int:
        """Seed / p / scenario / speed / play state / REC, wrapped to ``width``."""
        pal, font = self.palette, self.fonts.small
        rec_colour = pal.log_failure_text
        items: list[tuple[str, RGB, bool]] = [
            (f"seed {info.seed}", pal.text_dim, False),
            (f"p = {info.slip_p:.2f}", pal.text_dim, False),
            (info.scenario or "custom map", pal.text_dim, False),
            (f"{info.speed:g} steps/s", pal.text_dim, False),
            (PAUSED_TEXT if info.paused else PLAYING_TEXT, pal.text, False),
        ]
        if info.recording:
            items.append((REC_TEXT, rec_colour, True))
        sep_w = text_cache.width(font, INFO_SEPARATOR)
        line_h = _line_h(font)
        x = left
        for i, (text, colour, dot) in enumerate(items):
            w = text_cache.width(font, text) + (2 * REC_DOT_RADIUS + LINE_SPACING if dot else 0)
            lead = sep_w if i else 0
            if x + lead + w > left + width and x > left:
                x, y, lead = left, y + line_h, 0
            if lead:
                x += self._text(surface, font, INFO_SEPARATOR, pal.text_dim, x, y)
            if dot:
                cy = y + font.get_height() // 2
                pygame.draw.circle(surface, colour, (x + REC_DOT_RADIUS, cy), REC_DOT_RADIUS)
                x += 2 * REC_DOT_RADIUS + LINE_SPACING
            x += self._text(surface, font, text, colour, x, y, left + width - x)
        return y + line_h

    def _draw_metrics(self, surface: pygame.Surface, metrics: Mapping[str, str], left: int,
                      y: int, width: int) -> int:
        """Two columns of ``label  value`` (left column first); returns the next y."""
        if not metrics:
            return y
        pal, label_font, value_font = self.palette, self.fonts.small, self.fonts.mono
        items = list(metrics.items())
        per_col = math.ceil(len(items) / 2)
        col_w = (width - METRIC_COLUMN_GAP) // 2
        line_h = max(_line_h(label_font), _line_h(value_font))
        label_dy, value_dy = _baseline_offsets((label_font, value_font), line_h)
        for i, (label, value) in enumerate(items):
            col, row = divmod(i, per_col)
            x = left + col * (col_w + METRIC_COLUMN_GAP)
            ry = y + row * line_h
            value_text = text_cache.fit(value_font, str(value), col_w // 2)
            vw = text_cache.width(value_font, value_text)
            self._text(surface, value_font, value_text, pal.text, x + col_w - vw, ry + value_dy)
            self._text(surface, label_font, label, pal.text_dim, x, ry + label_dy,
                       col_w - vw - LINE_SPACING * 2)
        return y + per_col * line_h

    def _draw_legend(self, surface: pygame.Surface, legend: Legend, left: int, y: int,
                     width: int) -> int:
        """``title  low [swatches] high`` on one row; returns the next y."""
        pal, font = self.palette, self.fonts.small
        size = max((s.get_height() for s in legend.swatches), default=0)
        line_h = max(_line_h(font), size + LINE_SPACING)
        text_dy = (line_h - font.get_height()) // 2
        x = left
        x += self._text(surface, font, legend.title, pal.text_dim, x, y + text_dy, width)
        x += SECTION_GAP
        if legend.low:
            x += self._text(surface, font, legend.low, pal.text_dim, x, y + text_dy)
            x += LEGEND_SWATCH_GAP
        for swatch in legend.swatches:
            surface.blit(swatch, (x, y + (line_h - swatch.get_height()) // 2))
            x += swatch.get_width() + LEGEND_SWATCH_GAP
        if legend.high:
            self._text(surface, font, legend.high, pal.text_dim, x, y + text_dy,
                       max(0, left + width - x))
        return y + line_h

    def _draw_log(self, surface: pygame.Surface, log: EventLog, left: int, y: int, width: int,
                  line_h: int) -> None:
        """Section title (with the counts of lines out of view) and the visible log entries,
        word-wrapped, newest at the bottom.

        :data:`LOG_VISIBLE_LINES` counts *visual* lines: the newest entries that fit entirely
        are shown (only the newest one may be cut when it alone is taller than the box).
        """
        pal, fonts = self.palette, self.fonts
        font = fonts.mono
        glyph_w = max(text_cache.width(font, g) for g in LOG_KIND_GLYPHS.values())
        text_x = left + LOG_ACCENT_W + LOG_GLYPH_GAP + glyph_w + LOG_GLYPH_GAP
        max_w = left + width - LOG_GLYPH_GAP - text_x
        start, end = log.bounds(LOG_VISIBLE_LINES)
        entries = log.lines[start:end]
        blocks: list[tuple[tuple[str, ...], str]] = []
        used = 0
        for text, kind in reversed(entries):  # newest first
            wrapped = text_cache.wrap(font, text, max_w)
            if used + len(wrapped) > LOG_VISIBLE_LINES:
                if not blocks:  # the newest entry alone overflows: keep its first lines
                    blocks.append((wrapped[:LOG_VISIBLE_LINES], kind))
                break
            blocks.append((wrapped, kind))
            used += len(wrapped)
        blocks.reverse()
        older = end - len(blocks)
        newer = len(log) - end

        self._text(surface, fonts.small, SECTION_LOG, pal.text_dim, left, y)
        notes = []
        if older:
            notes.append(LOG_OLDER_TEXT.format(n=older))
        if newer:
            notes.append(LOG_NEWER_TEXT.format(n=newer))
        if notes:
            note = LOG_SCROLL_SEPARATOR.join(notes)
            nw = text_cache.width(fonts.small, note)
            self._text(surface, fonts.small, note, pal.text_dim, left + width - nw, y)
        y += _line_h(fonts.small)
        self.log_rect = pygame.Rect(left, y, width, LOG_VISIBLE_LINES * line_h)
        surface.fill(pal.background, self.log_rect)
        pygame.draw.rect(surface, pal.grid_line, self.log_rect, 1)
        total = sum(len(wrapped) for wrapped, _ in blocks)
        ly = self.log_rect.bottom - total * line_h  # newest line at the bottom
        dy = (line_h - font.get_height()) // 2
        for wrapped, kind in blocks:
            colour = log_kind_colour(kind, pal)
            if kind != "info":
                surface.fill(colour, (left + 1, ly + 1, LOG_ACCENT_W, len(wrapped) * line_h - 2))
            glyph = LOG_KIND_GLYPHS[kind]
            self._text(surface, font, glyph, colour, left + LOG_ACCENT_W + LOG_GLYPH_GAP, ly + dy)
            for part in wrapped:  # continuation lines hang under the text, not the glyph
                self._text(surface, font, part, colour, text_x, ly + dy, max_w)
                ly += line_h


# --- help overlay ------------------------------------------------------------------------------
_dim_cache: dict[tuple[tuple[int, int], int], pygame.Surface] = {}


def _dim_surface(size: tuple[int, int], palette: Palette) -> pygame.Surface:
    key = (size, id(palette))
    surf = _dim_cache.get(key)
    if surf is None:
        _dim_cache.clear()
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill((*palette.background, HELP_OVERLAY_ALPHA))
        _dim_cache[key] = surf
    return surf


def _help_split(items: Sequence[tuple[str, str] | str]) -> int:
    """Index where the right help column starts: the section heading nearest the middle (so
    no section is split across columns), else the middle itself."""
    half = math.ceil(len(items) / 2)
    headings = [i for i, item in enumerate(items) if isinstance(item, str) and i > 0]
    if not headings:
        return half
    best = min(headings, key=lambda i: (max(i, len(items) - i), i))
    return best if max(best, len(items) - best) <= HELP_MAX_COLUMN_SHARE * len(items) else half


def draw_help_overlay(
    surface: pygame.Surface,
    palette: Palette,
    fonts: Fonts,
    lines: Sequence[tuple[str, str] | str],
) -> None:
    """Darken ``surface`` and list the controls in a centred two-column box.

    ``lines`` are ``(key, description)`` pairs; a plain string is a section heading. Items fill
    the left column first.
    """
    surface.blit(_dim_surface(surface.get_size(), palette), (0, 0))
    key_font, desc_font, head_font = fonts.mono, fonts.body, fonts.small
    row_h = max(_line_h(key_font), _line_h(desc_font)) + HELP_ROW_GAP
    items = list(lines)
    split = _help_split(items)
    columns = [items[:split], items[split:]]
    per_col = max(len(columns[0]), len(columns[1]))
    sw, sh = surface.get_size()
    box_w = min(HELP_MAX_WIDTH, sw - 2 * HELP_MARGIN)
    title_h = _line_h(fonts.title)
    footer_h = _line_h(head_font)
    box_h = 2 * HELP_PADDING + title_h + SECTION_GAP + per_col * row_h + footer_h
    box_h = min(box_h, sh - 2 * HELP_MARGIN)
    box = pygame.Rect((sw - box_w) // 2, (sh - box_h) // 2, box_w, box_h)
    pygame.draw.rect(surface, palette.panel_bg, box, border_radius=CORNER_RADIUS)
    pygame.draw.rect(surface, palette.grid_line, box, 1, border_radius=CORNER_RADIUS)

    old_clip = surface.get_clip()
    surface.set_clip(box.clip(old_clip))
    try:
        x0, y0 = box.x + HELP_PADDING, box.y + HELP_PADDING
        surface.blit(text_cache.render(fonts.title, HELP_TITLE, palette.text), (x0, y0))
        top = y0 + title_h + SECTION_GAP
        col_w = (box_w - 2 * HELP_PADDING - HELP_COLUMN_GAP) // 2
        kpx, kpy = HELP_KEY_PADDING
        cap_h = key_font.metrics(CAP_HEIGHT_PROBE)[0][3]  # maxy of 'H' = cap height
        for c, column in enumerate(columns):
            x = x0 + c * (col_w + HELP_COLUMN_GAP)
            keys = [item[0] for item in column if not isinstance(item, str)]
            key_w = max((text_cache.width(key_font, k) for k in keys), default=0) + 2 * kpx
            for r, item in enumerate(column):
                y = top + r * row_h
                if isinstance(item, str):
                    text = text_cache.fit(head_font, item.upper(), col_w)
                    surface.blit(text_cache.render(head_font, text, palette.text_dim),
                                 (x, y + row_h - HELP_ROW_GAP - head_font.get_height()))
                    continue
                key, desc = item
                cap = pygame.Rect(x, y, key_w, key_font.get_height() + 2 * kpy)
                pygame.draw.rect(surface, palette.grid_line, cap, 1, border_radius=CORNER_RADIUS)
                # Optical centring: the cap-height band (baseline .. top of 'H') sits in the
                # middle of the keycap, and the description shares the key's baseline.
                baseline = cap.centery + cap_h // 2
                ks = text_cache.render(key_font, key, palette.text)
                surface.blit(ks, (cap.centerx - ks.get_width() // 2,
                                  baseline - key_font.get_ascent()))
                desc_fit = text_cache.fit(desc_font, desc, col_w - key_w - HELP_COLUMN_GAP // 2)
                ds = text_cache.render(desc_font, desc_fit, palette.text)
                surface.blit(ds, (cap.right + HELP_COLUMN_GAP // 3,
                                  baseline - desc_font.get_ascent()))
        footer = text_cache.render(head_font, HELP_FOOTER, palette.text_dim)
        surface.blit(footer, (box.centerx - footer.get_width() // 2,
                              box.bottom - HELP_PADDING - footer.get_height()))
    finally:
        surface.set_clip(old_clip)


__all__ = [
    "LOG_KINDS",
    "EventLog",
    "Legend",
    "Panel",
    "PanelInfo",
    "badge_size",
    "draw_badge",
    "draw_editing_badge",
    "draw_help_overlay",
    "log_kind_colour",
]
