"""Grid rendering: fog of war, terrain, plan envelope, policy arrows and the executed trace.

The view only *renders* a :class:`GridScene` (built by an app from a ``Mission`` / replay);
it contains no mission logic. Rendering is layered for speed (>= 30 FPS on 30x30):

* **tiles** - one pre-drawn surface per cell state and cell size (unknown '?', open, debris
  cross-hatch, '~' waves, survivor cross, start tag, envelope tint, policy arrows, translucent
  true-map tiles, flash tile), rebuilt only when the cell size or palette changes;
* **static layer** - the whole board (cells, true-map overlay, envelope, arrows, grid
  lines), rebuilt only when a cheap version key of the scene changes;
* **trace layer** - executed moves, drawn incrementally while the trace only grows (a new
  ``trace_id``, e.g. after a reset, forces a rebuild);
* per frame: replan flash, robot disc (a smaller disc inside a survivor-coloured ring once
  it stands on the survivor, so the cross stays visible), the start tag (above the trace and
  the disc, so slip rings never hide it) and the editor border.

Colour is never the only signal: unknown = '?', debris = cross-hatch + border, '~' = waves,
survivor = cross, start = 'S' tag, robot = disc with outline, arrived = segment with a
direction chevron, slipped = ring, skid = arrow, blocked = '×', policy = arrows (with a
contrasting halo, so they read on dark unknown cells and light open cells alike; on an unknown
cell the '?' then moves to the top-right corner instead of sitting under the arrow).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame

from rescue.grid import Cell, Pos
from rescue.model import ACTIONS, Action, OutcomeKind, kind_direction
from rescue.viz import theme
from rescue.viz.theme import Fonts, Palette, stroke

if TYPE_CHECKING:
    from rescue.runner import Mission, StepEvent

_UNKNOWN = int(Cell.UNKNOWN)
_OPEN = int(Cell.OPEN)
_DEBRIS = int(Cell.DEBRIS)
_SURVIVOR = int(Cell.SURVIVOR)


@dataclass(frozen=True)
class TraceMark:
    """One executed move as drawn on the grid (from a :class:`~rescue.runner.StepEvent`)."""

    pos: Pos
    new_pos: Pos
    action: Action
    outcome: OutcomeKind
    blocked: bool

    @classmethod
    def from_event(cls, event: StepEvent) -> TraceMark | None:
        """The mark of a move event; None for the t = 0 event and abort events."""
        if event.action is None or event.outcome is None:
            return None
        return cls(event.pos, event.new_pos, event.action, event.outcome, event.blocked)


def trace_from_events(events: Iterable[StepEvent]) -> list[TraceMark]:
    """Trace marks of every move event, in order."""
    marks = (TraceMark.from_event(e) for e in events)
    return [m for m in marks if m is not None]


@dataclass
class GridScene:
    """Everything :meth:`GridView.draw` needs for one frame.

    ``belief`` is the agent's int8 :class:`~rescue.grid.Cell` array, or None to show the true
    map fully (map editor). ``true_cells`` / ``high_slip`` describe the current true world.
    ``flash_cells`` pulse with ``flash_alpha`` (0..1) after a replan. ``editor`` draws a
    dashed border around the board. ``trace_id`` names the trace (see the field).
    """

    rows: int
    cols: int
    belief: np.ndarray | None
    true_cells: np.ndarray
    high_slip: np.ndarray
    robot: Pos | None
    start: Pos
    survivor: Pos
    show_true: bool = False
    show_policy: bool = False
    policy: Mapping[Pos, Action] = field(default_factory=dict)
    envelope: frozenset[Pos] = frozenset()
    trace: Sequence[TraceMark] = ()
    flash_cells: Sequence[Pos] = ()
    flash_alpha: float = 0.0
    editor: bool = False
    #: Identity of the trace (e.g. a mission serial): a different id redraws the trace layer
    #: from scratch even if the new trace happens to extend the old one's first / last marks.
    trace_id: int = 0

    @classmethod
    def from_mission(
        cls,
        mission: Mission,
        *,
        show_true: bool = False,
        show_policy: bool = False,
        trace: Sequence[TraceMark] | None = None,
        flash_cells: Sequence[Pos] = (),
        flash_alpha: float = 0.0,
    ) -> GridScene:
        """Scene of a started mission: agent belief, current true world, latest plan.

        Pass ``trace`` when the caller keeps the marks incrementally (cheaper than rebuilding
        them from ``mission.events`` every frame).
        """
        world = mission.world
        agent = mission.agent
        snapshots = agent.snapshots
        envelope = snapshots[-1].envelope if snapshots else frozenset()
        grid = world.grid
        return cls(
            rows=grid.rows,
            cols=grid.cols,
            belief=agent.belief.grid,
            true_cells=grid.cells,
            high_slip=grid.high_slip,
            robot=world.pos,
            start=grid.start,
            survivor=grid.survivor,
            show_true=show_true,
            show_policy=show_policy,
            policy=agent.policy_arrows() if show_policy else {},
            envelope=envelope,
            trace=trace if trace is not None else trace_from_events(mission.events),
            flash_cells=flash_cells,
            flash_alpha=flash_alpha,
        )


def _arrow_polygon(
    centre: tuple[float, float], direction: tuple[int, int], length: float, head: float,
    half_width: float, shaft: float,
) -> list[tuple[float, float]]:
    """Polygon of an arrow centred on ``centre`` pointing along ``direction`` (d_row, d_col)."""
    dr, dc = direction
    ux, uy = float(dc), float(dr)  # screen x follows columns, y follows rows
    px, py = -uy, ux  # perpendicular
    cx, cy = centre
    tail = (cx - ux * length / 2, cy - uy * length / 2)
    tip = (cx + ux * length / 2, cy + uy * length / 2)
    neck = (tip[0] - ux * head, tip[1] - uy * head)
    s = shaft / 2
    return [
        (tail[0] + px * s, tail[1] + py * s),
        (neck[0] + px * s, neck[1] + py * s),
        (neck[0] + px * half_width, neck[1] + py * half_width),
        tip,
        (neck[0] - px * half_width, neck[1] - py * half_width),
        (neck[0] - px * s, neck[1] - py * s),
        (tail[0] - px * s, tail[1] - py * s),
    ]


def _display_ready() -> bool:
    """True if a display surface exists (``convert`` / ``convert_alpha`` are allowed)."""
    return pygame.display.get_init() and pygame.display.get_surface() is not None


def _finish(surface: pygame.Surface, alpha: bool) -> pygame.Surface:
    """Convert ``surface`` to the display format when possible (faster blits)."""
    if not _display_ready():
        return surface
    return surface.convert_alpha() if alpha else surface.convert()


def draw_dashed_rect(
    surface: pygame.Surface, colour: tuple[int, int, int], rect: pygame.Rect, width: int,
    dash: int = theme.DASH_LENGTH, gap: int = theme.DASH_GAP,
) -> None:
    """Draw a dashed rectangle outline."""
    period = dash + gap
    x0, y0, x1, y1 = rect.left, rect.top, rect.right - 1, rect.bottom - 1
    for x in range(x0, x1 + 1, period):
        end = min(x + dash, x1)
        pygame.draw.line(surface, colour, (x, y0), (end, y0), width)
        pygame.draw.line(surface, colour, (x, y1), (end, y1), width)
    for y in range(y0, y1 + 1, period):
        end = min(y + dash, y1)
        pygame.draw.line(surface, colour, (x0, y), (x0, end), width)
        pygame.draw.line(surface, colour, (x1, y), (x1, end), width)


class GridView:
    """Renders a :class:`GridScene` into ``rect`` with square cells, centred.

    The cell size adapts to the rect and the scene's rows/cols on every :meth:`draw` (or
    :meth:`layout`); :meth:`cell_rect` / :meth:`pos_at` use the latest layout.
    """

    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int], palette: Palette,
                 fonts: Fonts) -> None:
        self.rect = pygame.Rect(rect)
        self.palette = palette
        self.fonts = fonts
        self.rows = 0
        self.cols = 0
        self.cell = 0
        self.origin = (self.rect.x, self.rect.y)
        self._tiles: dict[Any, pygame.Surface] = {}
        self._tiles_key: tuple[Any, ...] | None = None
        self._static: pygame.Surface | None = None
        self._static_key: tuple[Any, ...] | None = None
        self._trace_layer: pygame.Surface | None = None
        self._trace_key: tuple[Any, ...] | None = None
        self._trace_n = 0
        self._trace_ends: tuple[TraceMark, TraceMark] | None = None

    # --- configuration ---------------------------------------------------------------------
    def set_rect(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        """Move / resize the view (layers are rebuilt lazily)."""
        self.rect = pygame.Rect(rect)
        if self.rows and self.cols:
            self.layout(self.rows, self.cols)

    def set_palette(self, palette: Palette) -> None:
        """Switch palettes (e.g. the high-contrast one); caches rebuild on the next draw."""
        self.palette = palette

    def invalidate(self) -> None:
        """Drop every cached layer (forces a full redraw on the next :meth:`draw`)."""
        self._tiles_key = self._static_key = self._trace_key = None
        self._tiles.clear()

    def layout(self, rows: int, cols: int) -> None:
        """Compute the cell size and origin for a ``rows`` x ``cols`` board."""
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got {rows}x{cols}")
        self.rows, self.cols = rows, cols
        fit = min(self.rect.width // cols, self.rect.height // rows)
        self.cell = max(theme.MIN_CELL_SIZE, min(theme.MAX_CELL_SIZE, fit))
        width, height = self.cell * cols, self.cell * rows
        self.origin = (
            self.rect.x + (self.rect.width - width) // 2,
            self.rect.y + (self.rect.height - height) // 2,
        )

    @property
    def board_rect(self) -> pygame.Rect:
        """Screen rectangle covered by the cells (latest layout)."""
        return pygame.Rect(self.origin, (self.cell * self.cols, self.cell * self.rows))

    # --- geometry --------------------------------------------------------------------------
    def cell_rect(self, p: Pos) -> pygame.Rect:
        """Screen rectangle of cell ``p`` (latest layout)."""
        r, c = p
        return pygame.Rect(self.origin[0] + c * self.cell, self.origin[1] + r * self.cell,
                           self.cell, self.cell)

    def cell_centre(self, p: Pos) -> tuple[int, int]:
        """Screen coordinates of the centre of cell ``p``."""
        return self.cell_rect(p).center

    def pos_at(self, xy: tuple[int, int]) -> Pos | None:
        """The cell under screen point ``xy``, or None outside the board / before layout."""
        if not self.cell or not self.rows:
            return None
        x, y = xy
        c = (x - self.origin[0]) // self.cell
        r = (y - self.origin[1]) // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (int(r), int(c))
        return None

    # --- tiles -----------------------------------------------------------------------------
    def _ensure_tiles(self) -> None:
        """(Re)build the per-cell-size glyph tiles if the cell size or palette changed."""
        key = (self.cell, id(self.palette), id(self.fonts))
        if key == self._tiles_key:
            return
        self._tiles.clear()
        self._tiles_key = key
        s, pal = self.cell, self.palette
        tiles = self._tiles

        def opaque(colour: tuple[int, int, int]) -> pygame.Surface:
            surf = pygame.Surface((s, s))
            surf.fill(colour)
            return surf

        def text_onto(surf: pygame.Surface, text: str, colour: tuple[int, int, int],
                      size: int, centre: tuple[float, float]) -> None:
            glyph = self.fonts.sized(size).render(text, True, colour)
            surf.blit(glyph, glyph.get_rect(center=(round(centre[0]), round(centre[1]))))

        # open
        tiles["open"] = _finish(opaque(pal.open), alpha=False)
        # unknown: dark + '?'
        unknown = opaque(pal.unknown)
        if s >= theme.GLYPH_MIN_CELL:
            text_onto(unknown, "?", pal.unknown_glyph, round(s * theme.UNKNOWN_GLYPH_FRAC),
                      (s / 2, s / 2))
        tiles["unknown"] = _finish(unknown, alpha=False)
        # unknown under a policy arrow: the '?' moves to the top-right corner
        corner = opaque(pal.unknown)
        if s >= theme.UNKNOWN_CORNER_MIN_CELL:
            size = round(s * theme.UNKNOWN_CORNER_GLYPH_FRAC)
            glyph = self.fonts.sized(size).render("?", True, pal.unknown_glyph)
            inset = max(1, round(s * theme.UNKNOWN_CORNER_INSET_FRAC))
            corner.blit(glyph, glyph.get_rect(topright=(s - inset, inset)))
        tiles["unknown_corner"] = _finish(corner, alpha=False)
        # debris: fill, cross-hatch, darker border
        debris = opaque(pal.debris)
        spacing = max(theme.MIN_HATCH_SPACING, round(s * theme.HATCH_SPACING_FRAC))
        hatch_w = stroke(s, theme.HATCH_WIDTH_FRAC)
        for k in range(-s, 2 * s, spacing):
            pygame.draw.line(debris, pal.debris_glyph, (k, 0), (k + s, s), hatch_w)
            pygame.draw.line(debris, pal.debris_glyph, (k + s, 0), (k, s), hatch_w)
        pygame.draw.rect(debris, pal.debris_border, debris.get_rect(),
                         stroke(s, theme.DEBRIS_BORDER_FRAC))
        tiles["debris"] = _finish(debris, alpha=False)
        # high slip '~': teal + wave strokes
        water = opaque(pal.high_slip)
        margin = s * theme.WAVE_MARGIN_FRAC
        amp = max(theme.MIN_WAVE_AMPLITUDE, s * theme.WAVE_AMPLITUDE_FRAC)
        wave_w = stroke(s, theme.WAVE_WIDTH_FRAC)
        span = s - 2 * margin
        for i in range(theme.WAVE_COUNT):
            y0 = margin + (i + 0.5) * span / theme.WAVE_COUNT
            pts = [
                (margin + span * j / theme.WAVE_SAMPLES,
                 y0 + amp * math.sin(2 * math.pi * theme.WAVE_PERIODS * j / theme.WAVE_SAMPLES))
                for j in range(theme.WAVE_SAMPLES + 1)
            ]
            pygame.draw.lines(water, pal.high_slip_glyph, False, pts, wave_w)
        tiles["water"] = _finish(water, alpha=False)
        # survivor: open + red cross
        survivor = opaque(pal.open)
        arm = round(s * theme.SURVIVOR_LEN_FRAC)
        thick = max(theme.MIN_SURVIVOR_THICK, round(s * theme.SURVIVOR_THICK_FRAC))
        cross_h = pygame.Rect(0, 0, arm, thick)
        cross_v = pygame.Rect(0, 0, thick, arm)
        cross_h.center = cross_v.center = (s // 2, s // 2)
        pygame.draw.rect(survivor, pal.survivor, cross_h)
        pygame.draw.rect(survivor, pal.survivor, cross_v)
        tiles["survivor"] = _finish(survivor, alpha=False)
        # translucent copies for the true-map overlay
        for name in ("open", "debris", "water", "survivor"):
            ghost = tiles[name].copy()
            ghost.set_alpha(theme.TRUE_MAP_ALPHA)
            tiles[f"true_{name}"] = ghost
        # start tag: small box with 'S' in the top-left corner
        tag = pygame.Surface((s, s), pygame.SRCALPHA)
        side = max(theme.MIN_START_TAG, round(s * theme.START_TAG_FRAC))
        pygame.draw.rect(tag, pal.start, pygame.Rect(0, 0, side, side))
        if s >= theme.GLYPH_MIN_CELL:
            text_onto(tag, "S", pal.start_text, round(s * theme.START_FONT_FRAC),
                      (side / 2, side / 2))
        tiles["start"] = _finish(tag, alpha=True)
        # envelope tint
        tint = pygame.Surface((s, s), pygame.SRCALPHA)
        tint.fill((*pal.envelope, theme.ENVELOPE_ALPHA))
        tiles["envelope"] = _finish(tint, alpha=True)
        # policy arrows: arrow colour on a contrasting halo (visible on dark and light cells)
        halo = stroke(s, theme.POLICY_ARROW_OUTLINE_FRAC)
        offsets = [(dx, dy) for dx in range(-halo, halo + 1) for dy in range(-halo, halo + 1)
                   if dx * dx + dy * dy <= halo * halo]
        fill_rgba = (*pal.arrow, theme.POLICY_ARROW_ALPHA)
        halo_rgba = (*pal.arrow_outline, theme.POLICY_ARROW_ALPHA)
        for action in ACTIONS:
            arrow = pygame.Surface((s, s), pygame.SRCALPHA)
            poly = _arrow_polygon(
                (s / 2, s / 2), action.delta, s * theme.POLICY_ARROW_LEN_FRAC,
                s * theme.POLICY_ARROW_HEAD_FRAC, s * theme.POLICY_ARROW_HALF_WIDTH_FRAC,
                max(theme.MIN_ARROW_SHAFT, s * theme.POLICY_ARROW_SHAFT_FRAC),
            )
            # Halo = the polygon dilated by `halo` px (stamped at every offset in a disc).
            # Drawing on a per-pixel-alpha surface replaces pixels, so the fill drawn last
            # keeps the halo's alpha from adding up.
            for dx, dy in offsets:
                pygame.draw.polygon(arrow, halo_rgba, [(x + dx, y + dy) for x, y in poly])
            pygame.draw.polygon(arrow, fill_rgba, poly)
            tiles[("arrow", action)] = _finish(arrow, alpha=True)
        # replan flash (global alpha set per frame)
        flash = pygame.Surface((s, s), pygame.SRCALPHA)
        flash.fill((*pal.replan_flash, theme.FLASH_FILL_ALPHA))
        pygame.draw.rect(flash, (*pal.replan_flash, 255), flash.get_rect(),
                         stroke(s, theme.FLASH_OUTLINE_FRAC))
        tiles["flash"] = _finish(flash, alpha=True)

    # --- static layer ----------------------------------------------------------------------
    @staticmethod
    def _cell_tile(value: int, slip: bool) -> str:
        if value == _UNKNOWN:
            return "unknown"
        if value == _DEBRIS:
            return "debris"
        if value == _SURVIVOR:
            return "survivor"
        return "water" if slip else "open"

    def _static_version(self, scene: GridScene) -> tuple[Any, ...]:
        """Cheap key that changes whenever the static layer would look different."""
        belief = None if scene.belief is None else np.ascontiguousarray(scene.belief).tobytes()
        policy = tuple(scene.policy.items()) if scene.show_policy else None
        envelope = scene.envelope
        if not isinstance(envelope, frozenset):  # a mutable set could change behind the key
            envelope = frozenset(envelope)
        return (
            self._tiles_key, scene.rows, scene.cols, belief,
            np.ascontiguousarray(scene.true_cells).tobytes(),
            np.ascontiguousarray(scene.high_slip).tobytes(),
            scene.start, scene.survivor, scene.show_true and scene.belief is not None,
            envelope, policy,
        )

    @staticmethod
    def _arrow_cells(scene: GridScene) -> dict[Pos, Action]:
        """The policy arrows drawn: on board cells of the envelope (if any), when shown."""
        if not scene.show_policy:
            return {}
        envelope = scene.envelope
        return {
            (r, c): action
            for (r, c), action in scene.policy.items()
            if (not envelope or (r, c) in envelope)
            and 0 <= r < scene.rows and 0 <= c < scene.cols
        }

    def _build_static(self, scene: GridScene) -> pygame.Surface:
        s, tiles = self.cell, self._tiles
        width, height = s * scene.cols, s * scene.rows
        layer = pygame.Surface((width, height))
        layer.fill(self.palette.background)
        truth = np.asarray(scene.true_cells)
        slip = np.asarray(scene.high_slip, dtype=bool)
        shown = truth if scene.belief is None else np.asarray(scene.belief)
        blits: list[tuple[pygame.Surface, tuple[int, int]]] = []
        overlay: list[tuple[pygame.Surface, tuple[int, int]]] = []
        show_true = scene.show_true and scene.belief is not None
        arrows = self._arrow_cells(scene)
        for r in range(scene.rows):
            y = r * s
            for c in range(scene.cols):
                value = int(shown[r, c])
                high = bool(slip[r, c])
                name = self._cell_tile(value, high)
                if name == "unknown" and (r, c) in arrows:
                    name = "unknown_corner"
                blits.append((tiles[name], (c * s, y)))
                if show_true:
                    true_value = int(truth[r, c])
                    if true_value != value:
                        overlay.append((tiles["true_" + self._cell_tile(true_value, high)],
                                        (c * s, y)))
        layer.blits(blits, doreturn=False)
        if overlay:
            layer.blits(overlay, doreturn=False)
        env = tiles["envelope"]
        layer.blits([(env, (c * s, r * s)) for r, c in scene.envelope
                     if 0 <= r < scene.rows and 0 <= c < scene.cols], doreturn=False)
        layer.blits([(tiles[("arrow", action)], (c * s, r * s))
                      for (r, c), action in arrows.items()], doreturn=False)
        line = self.palette.grid_line
        for r in range(scene.rows + 1):
            y = min(r * s, height - 1)
            pygame.draw.line(layer, line, (0, y), (width - 1, y), theme.GRID_LINE_WIDTH)
        for c in range(scene.cols + 1):
            x = min(c * s, width - 1)
            pygame.draw.line(layer, line, (x, 0), (x, height - 1), theme.GRID_LINE_WIDTH)
        return _finish(layer, alpha=False)

    # --- trace layer -----------------------------------------------------------------------
    def _local_centre(self, p: Pos) -> tuple[int, int]:
        """Integer centre of cell ``p`` on the board layers (joins of adjacent segments meet
        exactly, so the round caps leave no notches)."""
        s = self.cell
        return (p[1] * s + s // 2, p[0] * s + s // 2)

    def _trace_stroke(self, frac: float) -> int:
        """A trace stroke width: ``frac`` of the cell, at least :data:`theme.TRACE_MIN_STROKE`
        px on cells big enough for glyphs, and never more than a quarter cell."""
        s = self.cell
        floor = theme.TRACE_MIN_STROKE if s >= theme.GLYPH_MIN_CELL else theme.MIN_STROKE
        return max(1, min(max(floor, round(s * frac)), s // 4))

    def _draw_mark(self, layer: pygame.Surface, mark: TraceMark) -> None:
        """Draw one executed move: segment + chevron / ring / arrow / ×."""
        s, pal = self.cell, self.palette
        width = self._trace_stroke(theme.TRACE_LINE_FRAC)
        a = self._local_centre(mark.pos)
        if mark.outcome is OutcomeKind.SLIPPED:
            radius = max(theme.MIN_MARK_RADIUS, round(s * theme.TRACE_RING_RADIUS_FRAC))
            pygame.draw.circle(layer, pal.trace_slip, a, radius,
                               self._trace_stroke(theme.TRACE_RING_WIDTH_FRAC))
            return
        if mark.blocked or mark.pos == mark.new_pos:
            direction = kind_direction(mark.outcome, mark.action) or mark.action
            dr, dc = direction.delta
            off = s * theme.BLOCKED_OFFSET_FRAC
            cx, cy = a[0] + dc * off, a[1] + dr * off
            h = max(theme.MIN_BLOCKED_X, s * theme.BLOCKED_X_FRAC)
            pygame.draw.line(layer, pal.trace_skid, (cx - h, cy - h), (cx + h, cy + h), width)
            pygame.draw.line(layer, pal.trace_skid, (cx - h, cy + h), (cx + h, cy - h), width)
            return
        b = self._local_centre(mark.new_pos)
        if mark.outcome is OutcomeKind.ARRIVED:
            pygame.draw.line(layer, pal.trace_ok, a, b, width)
            if width > 1:  # round joins: no notches where consecutive segments meet
                for end in (a, b):
                    pygame.draw.circle(layer, pal.trace_ok, end, width / 2)
            self._draw_chevron(layer, a, b, width)
            return
        # skid: red arrow from pos toward new_pos, stopping short of the centre
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        tip = (b[0] - ux * s * theme.SKID_ARROW_END_FRAC, b[1] - uy * s * theme.SKID_ARROW_END_FRAC)
        head = max(theme.MIN_SKID_HEAD, s * theme.SKID_ARROW_HEAD_FRAC)
        base = (tip[0] - ux * head, tip[1] - uy * head)
        pygame.draw.line(layer, pal.trace_skid, a, base, width)
        half = head * theme.SKID_ARROW_HALF_WIDTH
        px, py = -uy * half, ux * half
        pygame.draw.polygon(layer, pal.trace_skid,
                            [tip, (base[0] + px, base[1] + py), (base[0] - px, base[1] - py)])

    def _draw_chevron(self, layer: pygame.Surface, a: tuple[int, int], b: tuple[int, int],
                      width: int) -> None:
        """Small filled arrow head at the middle of the arrived segment a -> b (direction)."""
        if self.cell < theme.GLYPH_MIN_CELL:
            return
        size = max(theme.TRACE_MIN_CHEVRON, self.cell * theme.TRACE_CHEVRON_FRAC, 1.5 * width)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tip = (mx + ux * size / 2, my + uy * size / 2)
        back = (mx - ux * size / 2, my - uy * size / 2)
        px, py = -uy * size * theme.SKID_ARROW_HALF_WIDTH, ux * size * theme.SKID_ARROW_HALF_WIDTH
        pygame.draw.polygon(layer, self.palette.trace_ok,
                            [tip, (back[0] + px, back[1] + py), (back[0] - px, back[1] - py)])

    def _update_trace(self, scene: GridScene) -> pygame.Surface:
        """Trace layer for ``scene.trace``; appends new marks when the trace only grew."""
        trace = scene.trace
        n = len(trace)
        key = (self._tiles_key, scene.rows, scene.cols, scene.trace_id)
        layer = self._trace_layer
        grew = (
            layer is not None and key == self._trace_key and self._trace_ends is not None
            and n >= self._trace_n and trace[0] == self._trace_ends[0]
            and trace[self._trace_n - 1] == self._trace_ends[1]
        )
        if layer is not None and key == self._trace_key and n == 0 and self._trace_n == 0:
            return layer
        if not grew or layer is None:
            layer = pygame.Surface((self.cell * scene.cols, self.cell * scene.rows),
                                   pygame.SRCALPHA)
            layer = _finish(layer, alpha=True)
            layer.fill((0, 0, 0, 0))
            first = 0
        else:
            first = self._trace_n
        for i in range(first, n):
            self._draw_mark(layer, trace[i])
        self._trace_layer, self._trace_key, self._trace_n = layer, key, n
        self._trace_ends = (trace[0], trace[n - 1]) if n else None
        return layer

    # --- drawing ---------------------------------------------------------------------------
    def _draw_robot(self, surface: pygame.Surface, scene: GridScene) -> None:
        """Robot disc; on the survivor a smaller disc inside a survivor-coloured ring, so the
        cross arms (and the rescue) stay visible."""
        assert scene.robot is not None
        s, pal = self.cell, self.palette
        centre = self.cell_centre(scene.robot)
        frac = theme.ROBOT_RADIUS_FRAC
        if scene.robot == scene.survivor:
            frac = theme.ROBOT_ON_SURVIVOR_RADIUS_FRAC
            ring = max(theme.MIN_MARK_RADIUS, round(s * theme.RESCUE_RING_RADIUS_FRAC))
            pygame.draw.circle(surface, pal.survivor, centre, ring,
                               stroke(s, theme.RESCUE_RING_WIDTH_FRAC))
        radius = max(theme.MIN_MARK_RADIUS, round(s * frac))
        pygame.draw.circle(surface, pal.robot, centre, radius)
        pygame.draw.circle(surface, pal.robot_outline, centre, radius,
                           stroke(s, theme.ROBOT_OUTLINE_FRAC))

    def draw(self, surface: pygame.Surface, scene: GridScene) -> None:
        """Render ``scene`` into this view's rect on ``surface``."""
        if (scene.rows, scene.cols) != (self.rows, self.cols) or not self.cell:
            self.layout(scene.rows, scene.cols)
        self._ensure_tiles()
        version = self._static_version(scene)
        if self._static is None or version != self._static_key:
            self._static = self._build_static(scene)
            self._static_key = version
        surface.fill(self.palette.background, self.rect)
        surface.blit(self._static, self.origin)
        surface.blit(self._update_trace(scene), self.origin)
        if scene.flash_cells and scene.flash_alpha > 0:
            flash = self._tiles["flash"]
            flash.set_alpha(round(255 * max(0.0, min(1.0, scene.flash_alpha))))
            surface.blits([(flash, self.cell_rect(p).topleft) for p in scene.flash_cells
                           if 0 <= p[0] < scene.rows and 0 <= p[1] < scene.cols],
                          doreturn=False)
        if scene.robot is not None:
            self._draw_robot(surface, scene)
        # The corner start tag goes on top of the trace (slip rings) and of the robot disc
        # (which may overlap the corner on large cells), so the 'S' always stays readable.
        surface.blit(self._tiles["start"], self.cell_rect(scene.start).topleft)
        if scene.editor:
            border = self.board_rect.inflate(2 * theme.EDITOR_BORDER_PAD,
                                             2 * theme.EDITOR_BORDER_PAD).clip(self.rect)
            draw_dashed_rect(surface, self.palette.editor_border, border,
                             theme.EDITOR_BORDER_WIDTH)
