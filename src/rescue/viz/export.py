"""Plan-tree export: a PNG via matplotlib (no Graphviz) and an indented text view.

Both work from ``Plan.to_tree(max_depth)`` (docs/plan.md §3.10), or from an already built tree
dict such as the ``tree`` of a run-file snapshot. The PNG uses a tidy layout: leaves get evenly
spaced slots and every parent is centred on its children, laid out left-to-right by default
(levels as columns: AND-OR plans are deep and narrow) or top-to-bottom; boxes are styled by
node kind (Act ``E @ (1,1)``, goal, loop ``↺ retry E``, ref ``→ E-plan @ (r,c)``, truncated
``…``) and edges carry the branch labels (arrived / slipped / blocked / skid→(r,c)).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rescue.plan import Plan
from rescue.viz import theme

#: Default Act depth of exported trees.
DEFAULT_EXPORT_DEPTH = 5
#: Arrow between an edge label and the node text in the text export.
_ARROW_PREFIX = "→ "


@dataclass
class _Node:
    """A tree node with its layout position (x in leaf slots, y = level)."""

    text: str
    kind: str
    edge: str | None
    children: list[_Node] = field(default_factory=list)
    x: float = 0.0
    y: int = 0


def _tree_dict(plan: Plan | Mapping[str, Any], max_depth: int) -> Mapping[str, Any]:
    """``plan.to_tree(max_depth)``, or ``plan`` itself when it is already a tree dict."""
    if isinstance(plan, Plan):
        return plan.to_tree(max_depth)
    if isinstance(plan, Mapping) and "children" in plan and "text" in plan:
        return plan
    raise TypeError(f"expected a Plan or a to_tree() dict, got {type(plan).__name__}")


def _build(tree: Mapping[str, Any]) -> _Node:
    """Convert the nested dicts into :class:`_Node` objects (iteratively)."""
    root = _Node(str(tree["text"]), str(tree["kind"]), tree.get("edge"))
    stack: list[tuple[Mapping[str, Any], _Node]] = [(tree, root)]
    while stack:
        data, node = stack.pop()
        for child in data.get("children") or ():
            sub = _Node(str(child["text"]), str(child["kind"]), child.get("edge"),
                        y=node.y + 1)
            node.children.append(sub)
            stack.append((child, sub))
    return root


def _layout(root: _Node) -> tuple[int, int]:
    """Tidy layout in place; returns ``(leaf count, level count)``.

    Leaves get consecutive x slots in pre-order; a parent sits at the midpoint of its first
    and last child.
    """
    leaves = 0
    levels = 0
    # Post-order without recursion: (node, children_done).
    stack: list[tuple[_Node, bool]] = [(root, False)]
    while stack:
        node, done = stack.pop()
        levels = max(levels, node.y + 1)
        if not node.children:
            node.x = float(leaves)
            leaves += 1
            continue
        if done:
            node.x = (node.children[0].x + node.children[-1].x) / 2.0
            continue
        stack.append((node, True))
        for child in reversed(node.children):
            stack.append((child, False))
    return leaves, levels


def _edge_colour(label: str) -> str:
    """Colour of an edge label by its prefix (arrived / slipped / skid / blocked)."""
    for prefix, colour in theme.EXPORT_EDGE_LABEL_COLOURS.items():
        if label.startswith(prefix):
            return colour
    return theme.EXPORT_TEXT


def export_plan_tree(plan: Plan | Mapping[str, Any], path: str | Path,
                     max_depth: int = DEFAULT_EXPORT_DEPTH, title: str | None = None,
                     layout: str = theme.EXPORT_LAYOUT) -> Path:
    """Draw the plan tree (Act depth ≤ ``max_depth``) to a PNG at ``path`` and return it.

    ``layout`` is ``"lr"`` (root on the left, one column per level - AND-OR plans are deep and
    narrow, so this keeps the figure close to the page's aspect ratio) or ``"tb"`` (root on
    top). The figure's aspect ratio is capped at :data:`theme.EXPORT_MAX_ASPECT` and edge labels
    sit near the child end of their edge. ``plan`` may also be a ``to_tree`` dict (e.g. a
    run-file snapshot's ``tree``). Parent directories are created. Uses matplotlib's Agg
    canvas only (no pyplot state, no Graphviz).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if layout not in ("lr", "tb"):
        raise ValueError(f"layout must be 'lr' or 'tb', got {layout!r}")
    lr = layout == "lr"
    root = _build(_tree_dict(plan, max_depth))
    leaves, levels = _layout(root)
    # Screen axes: X grows to the right, Y grows downwards (row 0 at the top).
    n_x, n_y = (levels, leaves) if lr else (leaves, levels)
    unit_x = theme.EXPORT_LR_LEVEL_WIDTH_IN if lr else theme.EXPORT_LEAF_WIDTH_IN
    unit_y = theme.EXPORT_LR_LEAF_HEIGHT_IN if lr else theme.EXPORT_LEVEL_HEIGHT_IN

    def screen(node: _Node) -> tuple[float, float]:
        return (float(node.y), node.x) if lr else (node.x, float(node.y))

    margin = theme.EXPORT_MARGIN_IN
    min_w, min_h = theme.EXPORT_MIN_SIZE_IN
    title_in = (theme.EXPORT_TITLE_FONT_SIZE * theme.EXPORT_TITLE_LINES / theme.POINTS_PER_INCH
                if title else 0.0)
    width = min(theme.EXPORT_MAX_WIDTH_IN, max(min_w, n_x * unit_x + 2 * margin))
    height = max(min_h, n_y * unit_y + 2 * margin) + title_in
    if width > theme.EXPORT_MAX_ASPECT * height:
        height = width / theme.EXPORT_MAX_ASPECT
    elif height > theme.EXPORT_MAX_ASPECT * width:
        width = height / theme.EXPORT_MAX_ASPECT
    fig = Figure(figsize=(width, height), facecolor=theme.EXPORT_BACKGROUND)
    canvas = FigureCanvasAgg(fig)
    axes_h = height - title_in
    ax = fig.add_axes((0.0, 0.0, 1.0, axes_h / height))
    ax.set_axis_off()
    ax.set_facecolor(theme.EXPORT_BACKGROUND)
    # Constant spacing (inches per slot) whatever the figure size: the content is centred.
    half_x, half_y = width / unit_x / 2, axes_h / unit_y / 2
    cx, cy = (n_x - 1) / 2, (n_y - 1) / 2
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy + half_y, cy - half_y)  # row 0 at the top
    if title:
        fig.suptitle(title, fontsize=theme.EXPORT_TITLE_FONT_SIZE, color=theme.EXPORT_TEXT,
                     wrap=True)

    nodes: list[_Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    boxes: dict[int, Any] = {}
    for node in nodes:
        face, edge, style = theme.EXPORT_NODE_STYLE.get(
            node.kind, theme.EXPORT_NODE_STYLE["truncated"]
        )
        x, y = screen(node)
        boxes[id(node)] = ax.text(
            x, y, node.text, ha="center", va="center",
            fontsize=theme.EXPORT_FONT_SIZE, color=theme.EXPORT_TEXT, zorder=4,
            bbox={"boxstyle": f"round,pad={theme.EXPORT_BOX_PAD}", "fc": face, "ec": edge,
                  "ls": style, "lw": theme.EXPORT_EDGE_WIDTH},
        )
    canvas.draw()  # lays out the node boxes, so edges can end exactly at their borders
    to_data = ax.transData.inverted()
    extents: dict[int, tuple[float, float, float, float]] = {}
    for node in nodes:
        patch = boxes[id(node)].get_bbox_patch()
        artist = patch if patch is not None else boxes[id(node)]
        bb = artist.get_window_extent()
        (x0, y0), (x1, y1) = to_data.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        extents[id(node)] = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    for node in nodes:
        right, bottom = extents[id(node)][1], extents[id(node)][3]
        nx, ny = screen(node)
        for child in node.children:
            left, top = extents[id(child)][0], extents[id(child)][2]
            kx, ky = screen(child)
            # parent's right side -> child's left side (lr); parent's bottom -> child's top
            start, end = ((right, ny), (left, ky)) if lr else ((nx, bottom), (kx, top))
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": theme.EXPORT_EDGE_COLOUR,
                            "lw": theme.EXPORT_EDGE_WIDTH, "shrinkA": 0, "shrinkB": 0},
                zorder=1,
            )
            if child.edge:
                t = theme.EXPORT_EDGE_LABEL_POS
                ax.text(
                    start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]),
                    str(child.edge), ha="center", va="center",
                    fontsize=theme.EXPORT_EDGE_FONT_SIZE,
                    color=_edge_colour(str(child.edge)), zorder=3,
                    bbox={"boxstyle": f"round,pad={theme.EXPORT_EDGE_LABEL_PAD}",
                          "fc": theme.EXPORT_BACKGROUND, "ec": "none"},
                )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=theme.EXPORT_DPI, facecolor=theme.EXPORT_BACKGROUND)
    return out


def export_plan_text(plan: Plan | Mapping[str, Any],
                     max_depth: int = DEFAULT_EXPORT_DEPTH) -> str:
    """Indented text view of the plan tree, one node per line.

    The root is ``E @ (1,1)``; every other line is ``"{edge} → {text}"``, e.g.
    ``"  slipped → ↺ retry E"`` or ``"  skid→(2,1) → E @ (2,1)"``.
    """
    tree = _tree_dict(plan, max_depth)
    lines: list[str] = []
    stack: list[tuple[Mapping[str, Any], int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        text = str(node["text"])
        edge = node.get("edge")
        prefix = theme.EXPORT_TEXT_INDENT * depth
        if edge:
            # A ref's text already starts with "→ "; do not print two arrows in a row.
            body = text.removeprefix(_ARROW_PREFIX)
            lines.append(f"{prefix}{edge} {_ARROW_PREFIX}{body}")
        else:
            lines.append(f"{prefix}{text}")
        for child in reversed(node.get("children") or ()):
            stack.append((child, depth + 1))
    return "\n".join(lines)
