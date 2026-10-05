"""Collapsible, depth-limited plan-tree widget for the simulator side panel (docs/plan.md §3.11).

:class:`TreeView` renders :meth:`rescue.plan.Plan.to_tree` as indented rows::

    ▾ E @ (1,1)
        arrived → ▾ S @ (1,2)
        slipped → ↺ retry E
        skid→(2,1) → ▸ E @ (2,1)

Act rows carry a collapse triangle (click to fold/unfold), every child row is prefixed by the
label of the branch leading to it, and the row(s) of the robot's current state are highlighted
and kept scrolled into view. When the robot's state lies below the depth limit of the tree
drawn from the plan root, the view re-roots itself at that state's Act node (with a header
row naming the plan root), so a highlighted row always exists. The widget has no mission
logic: the app hands it a plan (or an A* path, or a message) and the robot position, and it
only renders them.

The module also provides :class:`TextCache`, the cached text renderer shared with
:mod:`rescue.viz.panel`. It renders glyphs the primary font lacks (``↺ ★ ✓ …``) with a symbol
fallback font, or as ASCII substitutes when no installed font has them.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pygame

from rescue.grid import Pos, fmt_pos
from rescue.plan import Act, Plan
from rescue.viz.theme import (
    RGB,
    SCROLL_STEP,
    SYMBOL_FONT_NAMES,
    TEXT_CACHE_CAPACITY,
    TREE_ACCENT_W,
    TREE_AUTOSCROLL_FRACTION,
    TREE_INDENT,
    TREE_MAX_DEPTH,
    TREE_MESSAGE_MAX_LINES,
    TREE_PAD_X,
    TREE_PAD_Y,
    TREE_ROW_HEIGHT,
    TREE_SCROLLBAR_MIN,
    TREE_SCROLLBAR_W,
    TREE_TRIANGLE_HIT_SLACK,
    TREE_TRIANGLE_SIZE,
    TREE_TRIANGLE_W,
)

if TYPE_CHECKING:
    from rescue.viz.theme import Fonts, Palette

# Sizes, alphas, cache capacity and symbol fonts live in viz/theme.py (docs/plan.md §3.11).
#: Message rows shown when there is no contingency plan.
TREE_NO_PLAN_TEXT = "No plan yet"
TREE_PATH_AGENT_TEXT = "No contingency plan (A* path agent)"
#: Header row of a tree re-rooted at the robot's state (the plan root is deeper than shown).
TREE_REROOT_TEXT = "⤷ subtree at {pos} · plan root {root}"
#: Highlighted row shown when the robot's state is in no visible row (e.g. below the depth of
#: a stored replay tree).
TREE_BELOW_TEXT = "robot at {pos}: below the depth shown"
#: Separator between a branch label and its target node.
TREE_EDGE_SEPARATOR = " → "
#: Prefix of goal rows.
TREE_GOAL_PREFIX = "★ "
#: Leading marker of a Ref row's ``to_tree`` text, and what replaces it below a branch label
#: (avoids ``arrived → → E-plan``).
TREE_REF_MARK = "→ "
TREE_REF_JUMP = "↪ "

# --- text rendering ----------------------------------------------------------------------------
#: ASCII substitutes for glyphs no available font can render.
ASCII_FALLBACK: dict[str, str] = {
    "▾": "v",
    "▸": ">",
    "▶": ">",
    "❚": "|",
    "↺": "@",
    "↻": "R",
    "★": "*",
    "→": "->",
    "↪": "->",
    "←": "<-",
    "…": "...",
    "●": "*",
    "•": "-",
    "·": "-",
    "✓": "+",
    "✗": "x",
    "×": "x",
    "»": ">>",
    "≤": "<=",
    "↑": "^",
    "↓": "v",
    "⤷": "->",
    "∓": "-/+",
}
#: A code point no font is expected to have; its rendering is the "missing glyph" box.
_MISSING_SENTINEL = "\U0010fffd"
#: Colour used for the off-screen glyph-support probe renders.
_PROBE_COLOUR: RGB = (255, 255, 255)
#: Ellipsis appended to text cut by :meth:`TextCache.fit`.
ELLIPSIS = "…"


class TextCache:
    """Cached text rendering with per-glyph font fallback.

    ``render(font, text, colour)`` returns an antialiased, per-pixel-alpha surface. Characters
    the font cannot draw are drawn with the first :data:`SYMBOL_FONT_NAMES` font that can (at
    the same point size), else replaced by their :data:`ASCII_FALLBACK`. Surfaces are kept in a
    bounded LRU cache, so steady-state frames do no text rendering at all.
    """

    def __init__(self, capacity: int = TEXT_CACHE_CAPACITY) -> None:
        self._capacity = capacity
        self._surfaces: OrderedDict[tuple[int, str, RGB], pygame.Surface] = OrderedDict()
        self._fits: OrderedDict[tuple[int, str, int], str] = OrderedDict()
        self._wraps: OrderedDict[tuple[int, str, int], tuple[str, ...]] = OrderedDict()
        self._support: dict[tuple[int, str], bool] = {}
        self._sentinels: dict[int, bytes] = {}
        self._fallback: dict[int, pygame.font.Font | None] = {}
        # Fonts are keyed by id(); holding a reference keeps the ids from being reused.
        self._fonts: dict[int, pygame.font.Font] = {}

    def clear(self) -> None:
        """Drop every cached surface (glyph-support results are kept)."""
        self._surfaces.clear()
        self._fits.clear()
        self._wraps.clear()

    # --- glyph support ---------------------------------------------------------------------
    @staticmethod
    def _probe(font: pygame.font.Font, text: str) -> bytes:
        surf = font.render(text, True, _PROBE_COLOUR)
        w, h = surf.get_size()
        return w.to_bytes(4, "little") + h.to_bytes(4, "little") + pygame.image.tobytes(
            surf, "RGBA"
        )

    def has_glyph(self, font: pygame.font.Font, ch: str) -> bool:
        """Whether ``font`` has a real glyph for the single character ``ch``."""
        if ord(ch) < 128:
            return True
        key = (id(font), ch)
        cached = self._support.get(key)
        if cached is not None:
            return cached
        self._fonts[id(font)] = font
        sentinel = self._sentinels.get(id(font))
        if sentinel is None:
            sentinel = self._sentinels[id(font)] = self._probe(font, _MISSING_SENTINEL)
        ok = self._probe(font, ch) != sentinel
        self._support[key] = ok
        return ok

    def _fallback_font(self, font: pygame.font.Font) -> pygame.font.Font | None:
        key = id(font)
        if key not in self._fallback:
            size = getattr(font, "point_size", None) or font.get_height()
            try:
                self._fallback[key] = pygame.font.SysFont(list(SYMBOL_FONT_NAMES), size)
            except (OSError, pygame.error):  # pragma: no cover - broken font setups
                self._fallback[key] = None
        return self._fallback[key]

    def runs(self, font: pygame.font.Font, text: str) -> list[tuple[pygame.font.Font, str]]:
        """Split ``text`` into ``(font, substring)`` runs that each font can render."""
        out: list[tuple[pygame.font.Font, str]] = []
        for ch in text:
            use, piece = font, ch
            if not self.has_glyph(font, ch):
                fb = self._fallback_font(font)
                if fb is not None and self.has_glyph(fb, ch):
                    use = fb
                else:
                    piece = ASCII_FALLBACK.get(ch, "?")
            if out and out[-1][0] is use:
                out[-1] = (use, out[-1][1] + piece)
            else:
                out.append((use, piece))
        return out

    # --- rendering -------------------------------------------------------------------------
    def render(self, font: pygame.font.Font, text: str, colour: RGB) -> pygame.Surface:
        """Antialiased surface of ``text`` in ``colour`` (cached; do not modify it)."""
        key = (id(font), text, tuple(colour))
        surf = self._surfaces.get(key)
        if surf is not None:
            self._surfaces.move_to_end(key)
            return surf
        self._fonts[id(font)] = font
        runs = self.runs(font, text)
        if len(runs) == 1 and runs[0][0] is font:
            surf = font.render(runs[0][1], True, colour)
        else:
            parts = [(f, f.render(piece, True, colour)) for f, piece in runs]
            width = sum(p.get_width() for _, p in parts)
            height = font.get_height()
            surf = pygame.Surface((max(width, 1), height), pygame.SRCALPHA)
            x = 0
            for f, part in parts:
                y = font.get_ascent() - f.get_ascent()
                surf.blit(part, (x, y))
                x += part.get_width()
        self._surfaces[key] = surf
        if len(self._surfaces) > self._capacity:
            self._surfaces.popitem(last=False)
        return surf

    def width(self, font: pygame.font.Font, text: str) -> int:
        """Rendered width of ``text`` in pixels (as drawn by :meth:`render`)."""
        return sum(f.size(piece)[0] for f, piece in self.runs(font, text)) if text else 0

    def fit(self, font: pygame.font.Font, text: str, max_width: int) -> str:
        """``text``, cut and suffixed with an ellipsis if wider than ``max_width`` pixels."""
        key = (id(font), text, max_width)
        cached = self._fits.get(key)
        if cached is not None:
            return cached
        result = text
        if self.width(font, text) > max_width:
            lo, hi = 0, len(text)
            while lo < hi:  # longest prefix that fits together with the ellipsis
                mid = (lo + hi + 1) // 2
                if self.width(font, text[:mid].rstrip() + ELLIPSIS) <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            result = text[:lo].rstrip() + ELLIPSIS if lo > 0 else ""
        self._fits[key] = result
        if len(self._fits) > self._capacity:
            self._fits.popitem(last=False)
        return result

    def wrap(self, font: pygame.font.Font, text: str, max_width: int) -> tuple[str, ...]:
        """``text`` word-wrapped into lines no wider than ``max_width`` pixels (cached).

        Breaks at spaces; a single word wider than the line is split between characters.
        Always returns at least one line.
        """
        key = (id(font), text, -max_width - 1)  # negative widths: no clash with fit()
        cached = self._wraps.get(key)
        if cached is not None:
            return cached
        lines: list[str] = []
        current = ""
        for word in text.split(" "):
            candidate = f"{current} {word}" if current else word
            if self.width(font, candidate) <= max_width or not candidate:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            while current and self.width(font, current) > max_width:
                cut = len(current) - 1
                while cut > 1 and self.width(font, current[:cut]) > max_width:
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
        lines.append(current)
        result = tuple(lines)
        self._wraps[key] = result
        if len(self._wraps) > self._capacity:
            self._wraps.popitem(last=False)
        return result


#: Shared text cache for every viz widget.
text_cache = TextCache()


@dataclass(frozen=True)
class TreeRow:
    """One visible row of the tree.

    ``path`` is the sequence of child indices from the root (the collapse key), ``kind`` is the
    ``to_tree`` kind (``act``/``goal``/``loop``/``ref``/``truncated``) or ``info``/``path``
    for the message and A*-path rows, ``edge`` the branch label from the parent (None at the
    root), ``text`` the node's display text without the edge prefix.
    """

    path: tuple[int, ...]
    depth: int
    kind: str
    state: Pos | None
    edge: str | None
    text: str
    has_children: bool
    collapsed: bool


def _edge_colour(edge: str, palette: Palette) -> RGB:
    """Text colour of a branch label (arrived / slipped|blocked / skid)."""
    if edge.startswith("arrived"):
        return palette.edge_ok_text
    if edge.startswith("skid"):
        return palette.edge_skid_text
    return palette.edge_slip_text


def find_act(plan: Plan, state: Pos) -> Act | None:
    """The (unique) Act node of ``state`` in ``plan``, shallowest first; None if absent."""
    queue: deque[Any] = deque([plan.root])
    while queue:
        node = queue.popleft()
        if isinstance(node, Act):
            if node.state == state:
                return node
            queue.extend(br.node for br in node.branches)
    return None


def plan_subtree(plan: Any, state: Pos, depth: int) -> dict[str, Any] | None:
    """``to_tree(depth)`` of the subplan rooted at ``state``'s Act node, or None.

    Works for a :class:`~rescue.plan.Plan` and for any object with a
    ``subtree(state, depth)`` method (the replay's stored trees). Ref rows keep the action of
    their target from the *whole* plan (a subplan alone may not contain it).
    """
    if not isinstance(plan, Plan):
        subtree = getattr(plan, "subtree", None)
        return subtree(state, depth) if callable(subtree) else None
    act = find_act(plan, state)
    if act is None:
        return None
    tree = Plan(root=act, start=state, goal=plan.goal).to_tree(depth)
    policy = plan.to_policy()
    stack = [tree]
    while stack:
        node = stack.pop()
        if node["kind"] == "ref":
            action = policy.get(node["state"])
            name = "?" if action is None else str(action)
            node["text"] = f"{TREE_REF_MARK}{name}-plan @ {fmt_pos(node['state'])}"
        stack.extend(node["children"])
    return tree


class TreeView:
    """Scrollable, collapsible view of a contingency plan (or of an A* path).

    Call :meth:`set_plan` (and :meth:`set_path` for path agents) whenever the agent's plan or
    the robot's position may have changed: rows are rebuilt only when the plan, the current
    state, the depth or the collapse state actually changed. :meth:`draw` blits cached row
    surfaces clipped to :attr:`rect`; :meth:`handle_event` folds/unfolds Act rows on a click on
    their triangle and scrolls on the mouse wheel.
    """

    def __init__(self, rect: pygame.Rect, palette: Palette, fonts: Fonts) -> None:
        self.rect = pygame.Rect(rect)
        self.palette = palette
        self.fonts = fonts
        self._plan: Plan | None = None
        self._plan_key: object = None
        self._tree: dict[str, Any] | None = None
        self._max_depth = TREE_MAX_DEPTH
        self._current: Pos | None = None
        self._path: tuple[Pos, ...] = ()
        self._collapsed: set[tuple[int, ...]] = set()
        self._rows: list[TreeRow] = []
        self._highlight: list[int] = []
        self._message: str | None = None
        #: State of the Act node the shown tree is rooted at (None = the plan root).
        self._display_root: Pos | None = None
        self._subtrees: dict[Pos, dict[str, Any] | None] = {}
        self._dirty = True
        self._autoscroll = True
        self.scroll = 0
        self._row_surfaces: OrderedDict[tuple[Any, ...], pygame.Surface] = OrderedDict()

    # --- configuration ---------------------------------------------------------------------
    @property
    def font(self) -> pygame.font.Font:
        """Font of the tree rows."""
        return self.fonts.small

    @property
    def row_height(self) -> int:
        """Height of one row in pixels."""
        return max(TREE_ROW_HEIGHT, self.font.get_linesize())

    @property
    def visible_rows(self) -> int:
        """Number of rows that fit in :attr:`rect`."""
        return max(1, (self.rect.height - 2 * TREE_PAD_Y) // self.row_height)

    def set_rect(self, rect: pygame.Rect) -> None:
        """Move/resize the widget (keeps scroll position and collapse state)."""
        rect = pygame.Rect(rect)
        if rect != self.rect:
            if rect.size != self.rect.size:
                self._autoscroll = True
                if rect.width != self.rect.width and self._plan is None:
                    self._dirty = True  # message rows are wrapped to the width
            self.rect = rect
            self._clamp_scroll()

    def set_palette(self, palette: Palette, fonts: Fonts | None = None) -> None:
        """Switch palette (e.g. high contrast) and optionally fonts; drops cached rows."""
        self.palette = palette
        if fonts is not None:
            self.fonts = fonts
            self._dirty = True
        self._row_surfaces.clear()

    def set_message(self, text: str | None) -> None:
        """Message shown (word-wrapped) when there is neither a plan nor a path; None restores
        the default "No plan yet"."""
        if text != self._message:
            self._message = text
            if self._plan is None:
                self.scroll = 0
                self._dirty = True

    def set_plan(
        self,
        plan: Plan | None,
        current: Pos | None,
        max_depth: int = TREE_MAX_DEPTH,
        plan_id: int | None = None,
    ) -> None:
        """Show ``plan`` with ``current`` (the robot's state) highlighted.

        ``plan_id`` (e.g. ``StepEvent.plan_id``) identifies the plan; without it plans are
        compared structurally. While the plan is unchanged, collapse state and scroll position
        are kept; a new plan resets both. Cheap to call every frame.
        """
        key: object = plan_id if plan_id is not None else plan
        if plan is None or self._plan is None:
            same = plan is None and self._plan is None
        elif plan_id is not None:
            same = key == self._plan_key
        else:
            same = plan is self._plan or key == self._plan_key
        if not same:
            self._plan, self._plan_key = plan, key
            self._collapsed.clear()
            self._tree = None
            self._subtrees.clear()
            self._display_root = None
            self.scroll = 0
            self._dirty = self._autoscroll = True
        if max_depth != self._max_depth:
            self._max_depth = max_depth
            self._tree = None
            self._subtrees.clear()
            self._dirty = True
        if current != self._current:
            self._current = current
            self._dirty = self._autoscroll = True

    def set_path(self, path: Sequence[Pos] | None) -> None:
        """Path cells shown when there is no contingency plan (A* agents); None clears."""
        new = tuple(path) if path is not None else ()
        if new != self._path:
            self._path = new
            if self._plan is None:
                self.scroll = 0
                self._dirty = self._autoscroll = True

    @property
    def plan(self) -> Plan | None:
        """The plan currently shown."""
        return self._plan

    @property
    def rows(self) -> list[TreeRow]:
        """The visible (not folded away) rows, rebuilt if needed."""
        self._ensure_rows()
        return self._rows

    @property
    def highlighted(self) -> list[int]:
        """Indices into :attr:`rows` of the highlighted (current-state) rows."""
        self._ensure_rows()
        return self._highlight

    @property
    def display_root(self) -> Pos | None:
        """State the shown tree is re-rooted at (None while it starts at the plan root)."""
        self._ensure_rows()
        return self._display_root

    def is_collapsed(self, path: tuple[int, ...]) -> bool:
        """Whether the Act row at ``path`` is folded."""
        return path in self._collapsed

    def toggle(self, path: tuple[int, ...]) -> None:
        """Fold or unfold the Act row at ``path`` (a tuple of child indices from the root)."""
        if path in self._collapsed:
            self._collapsed.remove(path)
        else:
            self._collapsed.add(path)
        self._dirty = True
        self._autoscroll = False

    # --- row model -------------------------------------------------------------------------
    def _ensure_rows(self) -> None:
        if not self._dirty:
            return
        if self._plan is not None:
            self._rows, self._highlight = self._plan_rows()
        else:
            self._rows, self._highlight = self._path_rows()
        self._dirty = False
        self._clamp_scroll()

    def _subtree(self, state: Pos) -> dict[str, Any] | None:
        """Cached :func:`plan_subtree` of the current plan at ``state``."""
        if state not in self._subtrees:
            self._subtrees[state] = plan_subtree(self._plan, state, self._max_depth)
        return self._subtrees[state]

    def _rooted_rows(self, root: Pos | None) -> tuple[list[TreeRow], list[int]] | None:
        """Rows and highlight of the tree rooted at ``root`` (None = the plan root), with a
        header row when re-rooted; None if ``root`` has no Act node."""
        if root is None:
            if self._tree is None:
                assert self._plan is not None
                self._tree = self._plan.to_tree(self._max_depth)
            rows = self._tree_rows(self._tree)
            return rows, self._find_highlight(rows)
        subtree = self._subtree(root)
        if subtree is None:
            return None
        if self._tree is None:
            assert self._plan is not None
            self._tree = self._plan.to_tree(self._max_depth)
        header = TREE_REROOT_TEXT.format(pos=fmt_pos(root), root=self._tree["text"])
        rows = [TreeRow((), 0, "info", None, None, header, False, False)]
        rows += self._tree_rows(subtree)
        return rows, self._find_highlight(rows)

    def _plan_rows(self) -> tuple[list[TreeRow], list[int]]:
        """Rows of the plan, re-rooted at the robot's state when it is in no visible row.

        Tried in order: the current display root (stable while the robot stays in view), the
        plan root, the robot's own Act node. If none shows the robot, a highlighted message
        row says so.
        """
        cur = self._current
        candidates: list[Pos | None] = [None]
        if cur is not None:
            candidates = [self._display_root, None, cur]
        for root in dict.fromkeys(candidates):  # unique, in order
            built = self._rooted_rows(root)
            if built is not None and (built[1] or cur is None):
                self._set_display_root(root)
                return built
        self._set_display_root(None)
        built = self._rooted_rows(None)
        assert built is not None and cur is not None
        if cur == getattr(self._plan, "goal", None):
            text = f"{TREE_GOAL_PREFIX}robot at the goal {fmt_pos(cur)}"
        else:
            text = TREE_BELOW_TEXT.format(pos=fmt_pos(cur))
        note = TreeRow((), 0, "info", cur, None, text, False, False)
        return [note, *built[0]], [0]

    def _set_display_root(self, root: Pos | None) -> None:
        if root != self._display_root:
            self._display_root = root
            self._collapsed.clear()  # collapse paths are relative to the shown root
            self.scroll = 0
            self._autoscroll = True

    def _tree_rows(self, tree: dict[str, Any]) -> list[TreeRow]:
        rows: list[TreeRow] = []
        stack: list[tuple[dict[str, Any], tuple[int, ...]]] = [(tree, ())]
        while stack:
            node, path = stack.pop()
            children = node["children"]
            collapsed = bool(children) and path in self._collapsed
            text = node["text"]
            if node["kind"] == "goal":
                text = TREE_GOAL_PREFIX + text
            elif node["kind"] == "ref" and node["edge"] and text.startswith(TREE_REF_MARK):
                text = TREE_REF_JUMP + text[len(TREE_REF_MARK) :]
            rows.append(
                TreeRow(
                    path=path,
                    depth=len(path),
                    kind=node["kind"],
                    state=node["state"],
                    edge=node["edge"],
                    text=text,
                    has_children=bool(children),
                    collapsed=collapsed,
                )
            )
            if children and not collapsed:
                for i in range(len(children) - 1, -1, -1):
                    stack.append((children[i], (*path, i)))
        return rows

    def _find_highlight(self, rows: Sequence[TreeRow]) -> list[int]:
        cur = self._current
        if cur is None:
            return []
        acts = [i for i, r in enumerate(rows) if r.kind == "act" and r.state == cur]
        if acts:
            return acts
        return [
            i
            for i, r in enumerate(rows)
            if r.kind in ("ref", "truncated", "goal") and r.state == cur
        ]

    def _message_rows(self, text: str) -> list[TreeRow]:
        """``text`` word-wrapped to the tree width as info rows (at most
        :data:`TREE_MESSAGE_MAX_LINES`; explicit newlines start new paragraphs)."""
        width = max(1, self.rect.width - 2 * TREE_PAD_X - TREE_TRIANGLE_W - TREE_PAD_X // 2)
        lines: list[str] = []
        for paragraph in text.split("\n"):
            lines.extend(text_cache.wrap(self.font, paragraph, width))
        if len(lines) > TREE_MESSAGE_MAX_LINES:
            lines = lines[:TREE_MESSAGE_MAX_LINES]
            lines[-1] = text_cache.fit(self.font, lines[-1] + ELLIPSIS, width)
        return [TreeRow((), 0, "info", None, None, line, False, False) for line in lines]

    def _path_rows(self) -> tuple[list[TreeRow], list[int]]:
        if not self._path:
            return self._message_rows(self._message or TREE_NO_PLAN_TEXT), []
        rows = [TreeRow((), 0, "info", None, None, TREE_PATH_AGENT_TEXT, False, False)]
        rows.append(
            TreeRow((), 0, "info", None, None, f"A* path, {len(self._path)} cells:", False, False)
        )
        highlight: list[int] = []
        width = len(str(len(self._path)))
        for i, p in enumerate(self._path):
            if p == self._current:
                highlight.append(len(rows))
            label = f"{i + 1:>{width}}. {fmt_pos(p)}"
            rows.append(TreeRow((i,), 1, "path", p, None, label, False, False))
        return rows, highlight

    # --- scrolling -------------------------------------------------------------------------
    def _max_scroll(self) -> int:
        return max(0, len(self._rows) - self.visible_rows)

    def _clamp_scroll(self) -> None:
        self.scroll = min(max(self.scroll, 0), self._max_scroll())

    def scroll_by(self, rows: int) -> None:
        """Scroll by ``rows`` (positive = down), clamped to the content."""
        self._ensure_rows()
        self.scroll += rows
        self._clamp_scroll()

    def _apply_autoscroll(self) -> None:
        if not self._autoscroll:
            return
        self._autoscroll = False
        if not self._highlight:
            return
        idx, vis = self._highlight[0], self.visible_rows
        if idx < self.scroll or idx >= self.scroll + vis:
            self.scroll = idx - int(vis * TREE_AUTOSCROLL_FRACTION)
            self._clamp_scroll()

    # --- geometry --------------------------------------------------------------------------
    def _row_x(self, row: TreeRow) -> int:
        return self.rect.x + TREE_PAD_X + row.depth * TREE_INDENT

    def row_at(self, xy: tuple[int, int]) -> int | None:
        """Index into :attr:`rows` of the row under the screen point ``xy`` (None if none)."""
        if not self.rect.collidepoint(xy):
            return None
        self._ensure_rows()
        offset = xy[1] - self.rect.y - TREE_PAD_Y
        if offset < 0:
            return None
        idx = self.scroll + offset // self.row_height
        if idx >= min(len(self._rows), self.scroll + self.visible_rows):
            return None
        return idx

    def triangle_rect(self, index: int) -> pygame.Rect:
        """Screen rect of the collapse triangle of row ``index`` (assumed visible)."""
        row = self._rows[index]
        y = self.rect.y + TREE_PAD_Y + (index - self.scroll) * self.row_height
        return pygame.Rect(self._row_x(row), y, TREE_TRIANGLE_W, self.row_height)

    # --- events ----------------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Fold/unfold on a left click on a triangle; scroll on the wheel. True if consumed."""
        if event.type == pygame.MOUSEWHEEL:
            if not self.rect.collidepoint(pygame.mouse.get_pos()):
                return False
            self.scroll_by(-event.y * SCROLL_STEP)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if event.button != pygame.BUTTON_LEFT:
                return True  # wheel buttons (4/5) arrive as MOUSEWHEEL too
            idx = self.row_at(event.pos)
            if idx is not None and self._rows[idx].has_children:
                hit = self.triangle_rect(idx).inflate(2 * TREE_TRIANGLE_HIT_SLACK, 0)
                if hit.collidepoint(event.pos):
                    self.toggle(self._rows[idx].path)
            return True
        return False

    # --- drawing ---------------------------------------------------------------------------
    def _row_surface(self, row: TreeRow, highlighted: bool = False) -> pygame.Surface:
        """Cached surface of a row's text (edge prefix + node text), without the triangle.

        A highlighted row is drawn entirely in ``palette.highlight_text`` (it sits on the solid
        ``highlight_row`` fill, where the per-kind colours would lose contrast).
        """
        key = (row.kind, row.edge, row.text, highlighted)
        surf = self._row_surfaces.get(key)
        if surf is not None:
            self._row_surfaces.move_to_end(key)
            return surf
        pal, font = self.palette, self.font
        parts: list[pygame.Surface] = []
        if row.edge is not None:
            edge_colour = pal.highlight_text if highlighted else _edge_colour(row.edge, pal)
            sep_colour = pal.highlight_text if highlighted else pal.text_dim
            parts.append(text_cache.render(font, row.edge, edge_colour))
            parts.append(text_cache.render(font, TREE_EDGE_SEPARATOR, sep_colour))
        colour = pal.text_dim if row.kind in ("loop", "ref", "truncated", "info") else pal.text
        if highlighted:
            colour = pal.highlight_text
        parts.append(text_cache.render(font, row.text, colour))
        width = sum(p.get_width() for p in parts)
        surf = pygame.Surface((max(width, 1), font.get_height()), pygame.SRCALPHA)
        x = 0
        for p in parts:
            surf.blit(p, (x, 0))
            x += p.get_width()
        self._row_surfaces[key] = surf
        if len(self._row_surfaces) > TEXT_CACHE_CAPACITY:
            self._row_surfaces.popitem(last=False)
        return surf

    def _draw_triangle(self, surface: pygame.Surface, rect: pygame.Rect, collapsed: bool,
                       colour: RGB) -> None:
        cx, cy, s = rect.centerx, rect.centery, TREE_TRIANGLE_SIZE
        if collapsed:  # ▸
            pts = [(cx - s // 2, cy - s), (cx - s // 2, cy + s), (cx + s, cy)]
        else:  # ▾
            pts = [(cx - s, cy - s // 2), (cx + s, cy - s // 2), (cx, cy + s)]
        pygame.draw.polygon(surface, colour, pts)

    def draw(self, surface: pygame.Surface) -> None:
        """Render the visible rows into :attr:`rect` (clipped to it)."""
        self._ensure_rows()
        self._apply_autoscroll()
        pal, rect, row_h = self.palette, self.rect, self.row_height
        old_clip = surface.get_clip()
        surface.set_clip(rect.clip(old_clip))
        try:
            surface.fill(pal.background, rect)
            pygame.draw.rect(surface, pal.grid_line, rect, 1)
            highlight = set(self._highlight)
            text_dy = (row_h - self.font.get_height()) // 2
            first, last = self.scroll, min(len(self._rows), self.scroll + self.visible_rows + 1)
            for idx in range(first, last):
                row = self._rows[idx]
                y = rect.y + TREE_PAD_Y + (idx - self.scroll) * row_h
                lit = idx in highlight
                if lit:
                    self._draw_highlight(surface, y)
                x = self._row_x(row)
                guide = pal.highlight_text if lit else pal.grid_line
                for level in range(1, row.depth + 1):  # indentation guides
                    gx = rect.x + TREE_PAD_X + (level - 1) * TREE_INDENT + TREE_TRIANGLE_W // 2
                    pygame.draw.line(surface, guide, (gx, y), (gx, y + row_h - 1))
                tri = pygame.Rect(x, y, TREE_TRIANGLE_W, row_h)
                if row.has_children:
                    self._draw_triangle(surface, tri, row.collapsed,
                                        pal.highlight_text if lit else pal.text)
                surface.blit(self._row_surface(row, lit),
                             (tri.right + TREE_PAD_X // 2, y + text_dy))
            self._draw_scrollbar(surface)
        finally:
            surface.set_clip(old_clip)

    def _draw_highlight(self, surface: pygame.Surface, y: int) -> None:
        """Solid ``highlight_row`` bar with a ``highlight`` outline and left accent."""
        rect, row_h, pal = self.rect, self.row_height, self.palette
        bar = pygame.Rect(rect.x + 1, y, rect.width - 2, row_h)
        surface.fill(pal.highlight_row, bar)
        pygame.draw.rect(surface, pal.highlight, bar, 1)
        surface.fill(pal.highlight, (bar.x, y, TREE_ACCENT_W, row_h))

    def _draw_scrollbar(self, surface: pygame.Surface) -> None:
        total, vis = len(self._rows), self.visible_rows
        if total <= vis:
            return
        track = self.rect.inflate(0, -2 * TREE_PAD_Y)
        length = max(TREE_SCROLLBAR_MIN, track.height * vis // total)
        top = track.y + (track.height - length) * self.scroll // max(1, total - vis)
        bar = pygame.Rect(track.right - TREE_SCROLLBAR_W - 1, top, TREE_SCROLLBAR_W, length)
        pygame.draw.rect(surface, self.palette.text_dim, bar, border_radius=TREE_SCROLLBAR_W // 2)


__all__ = [
    "ASCII_FALLBACK",
    "TREE_MAX_DEPTH",
    "TextCache",
    "TreeRow",
    "TreeView",
    "find_act",
    "plan_subtree",
    "text_cache",
]
