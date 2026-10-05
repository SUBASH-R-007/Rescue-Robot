"""Contingency plans: the tree of Act / Goal / Loop / Ref nodes produced by AND-OR search.

A plan is a tree rooted at the start state. An :class:`Act` node chooses an action and has one
:class:`Branch` per possible outcome (``ProblemModel.results``); leaves are

* :class:`Goal` - the survivor cell was reached,
* :class:`Loop` - a back-edge to a state on the current path ("retry" when it is the parent),
* :class:`Ref`  - a state whose subplan appears elsewhere in this plan (graph search).

Every state has at most one :class:`Act` node, so ``to_policy()`` is a well-defined
``state -> action`` map. Plans can be several hundred Act levels deep on large maps, so every
traversal here is iterative (no Python recursion). That includes the identity methods of the
inner nodes: :class:`Act` and :class:`Branch` define their own structural ``==``/``hash``,
a short ``repr`` and a flat pickle/deepcopy encoding instead of the recursive methods a
dataclass would generate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from rescue.grid import Pos, fmt_pos
from rescue.model import Action, Outcome, OutcomeKind, ProblemModel

#: Separator between branches in the deck one-liner.
PRETTY_SEPARATOR = ", "
#: Marker appended to the display text of a truncated subtree in ``to_tree``.
TRUNCATED_MARK = "…"


@dataclass(frozen=True)
class Goal:
    """Leaf: the goal (survivor) state."""

    state: Pos


@dataclass(frozen=True)
class Loop:
    """Leaf: back-edge to an ancestor on the current path (the parent itself = "retry")."""

    state: Pos


@dataclass(frozen=True)
class Ref:
    """Leaf: the subplan for ``state`` is the :class:`Act` node elsewhere in this plan."""

    state: Pos


@dataclass(frozen=True, eq=False, repr=False)
class Branch:
    """One outcome of an action: the next state, every label leading there, and its subplan.

    Equality and hashing are structural (iterative through :class:`Act`).
    """

    state: Pos
    labels: tuple[OutcomeKind, ...]
    node: PlanNode

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Branch):
            return NotImplemented
        return self is other or (
            self.state == other.state and self.labels == other.labels and self.node == other.node
        )

    def __hash__(self) -> int:
        return hash(("Branch", self.state, self.labels, self.node))

    def __repr__(self) -> str:
        labels = tuple(str(kind) for kind in self.labels)
        return f"Branch(state={self.state!r}, labels={labels!r}, node={self.node!r})"


@dataclass(frozen=True, eq=False, repr=False)
class Act:
    """OR choice: do ``action`` at ``state``; one branch per possible outcome (AND node).

    Equality and hashing are structural over the whole subtree but computed iteratively;
    ``repr`` is one level deep; pickling and deep copies use the flat pre-order encoding
    (see :func:`_signature`), so all of them work on plans of any depth.
    """

    state: Pos
    action: Action
    branches: tuple[Branch, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Act):
            return NotImplemented
        return self is other or _signature(self) == _signature(other)

    def __hash__(self) -> int:
        return hash(("Act", _signature(self)))

    def __repr__(self) -> str:
        return (
            f"Act(state={self.state!r}, action={self.action}, "
            f"branches=<{len(self.branches)} branches>)"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (_from_signature, (_signature(self),))


PlanNode = Goal | Loop | Ref | Act
#: Leaf node classes by name, as used in :func:`_signature`.
_LEAF_TYPES: dict[str, type[Goal | Loop | Ref]] = {"Goal": Goal, "Loop": Loop, "Ref": Ref}


@dataclass(frozen=True)
class PlanStats:
    """Size measures of a plan."""

    nodes: int
    act_nodes: int
    depth: int
    loops: int
    refs: int
    envelope_size: int


def branch_label(parent: Pos, br: Branch) -> str:
    """Deck label of a branch from ``parent``: ``arrived``/``slipped``/``blocked``/``skid→(r,c)``.

    A branch that stays at ``parent`` is ``"slipped"`` if it carries the SLIPPED label, else
    ``"blocked"`` (a move or skid into a wall/known debris); otherwise ``"arrived"`` if the
    intended move lands there, else a skid shown with its 1-indexed target.
    """
    if br.state == parent:
        return "slipped" if OutcomeKind.SLIPPED in br.labels else "blocked"
    if OutcomeKind.ARRIVED in br.labels:
        return "arrived"
    return f"skid→{fmt_pos(br.state)}"


@dataclass(frozen=True)
class _Visit:
    """One node visited by :func:`_walk` (pre-order)."""

    node: PlanNode
    parent: Act | None
    branch: Branch | None
    #: Number of Act nodes on the root path *above* this node.
    act_level: int


def _walk(root: PlanNode) -> Iterator[_Visit]:
    """Iterative pre-order traversal; branches are visited in their stored order."""
    stack: list[_Visit] = [_Visit(root, None, None, 0)]
    while stack:
        visit = stack.pop()
        yield visit
        node = visit.node
        if isinstance(node, Act):
            for br in reversed(node.branches):
                stack.append(_Visit(br.node, node, br, visit.act_level + 1))


def _signature(root: PlanNode) -> tuple[tuple[Any, ...], ...]:
    """Flat pre-order encoding of the tree (structural equality without recursion).

    One entry ``(kind, state, action, n_branches, edge)`` per node, where ``kind`` is
    ``"act"`` or the leaf class name and ``edge`` is ``(branch.state, branch.labels)`` of the
    branch leading to the node (``None`` at the root). :func:`_from_signature` inverts it.
    """
    out: list[tuple[Any, ...]] = []
    for visit in _walk(root):
        node = visit.node
        edge = None if visit.branch is None else (visit.branch.state, visit.branch.labels)
        if isinstance(node, Act):
            out.append(("act", node.state, node.action, len(node.branches), edge))
        else:
            out.append((type(node).__name__, node.state, None, 0, edge))
    return tuple(out)


def _from_signature(signature: tuple[tuple[Any, ...], ...]) -> PlanNode:
    """Rebuild the tree encoded by :func:`_signature` (iteratively)."""
    root: PlanNode | None = None
    # Open Act nodes: (state, action, n_branches, edge, branches built so far).
    stack: list[tuple[Pos, Action, int, Any, list[Branch]]] = []
    for kind, state, action, n_branches, edge in signature:
        if kind == "act" and n_branches > 0:
            stack.append((state, action, n_branches, edge, []))
            continue
        node: PlanNode = Act(state, action, ()) if kind == "act" else _LEAF_TYPES[kind](state)
        while True:  # attach the finished node; close every parent it completes
            if not stack:
                root = node
                break
            p_state, p_action, p_n, p_edge, built = stack[-1]
            built.append(Branch(edge[0], edge[1], node))
            if len(built) < p_n:
                break
            stack.pop()
            node, edge = Act(p_state, p_action, tuple(built)), p_edge
    if root is None or stack:
        raise ValueError("truncated plan signature")
    return root


@dataclass(frozen=True, eq=False)
class Plan:
    """A contingency plan rooted at ``start`` for reaching ``goal``.

    Equality and hashing are structural but computed iteratively (deep plans would overflow
    the recursion limit with the generated dataclass methods).
    """

    root: PlanNode
    start: Pos
    goal: Pos

    # --- identity --------------------------------------------------------------------------
    @cached_property
    def _sig(self) -> tuple[Any, ...]:
        return (self.start, self.goal, _signature(self.root))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return self is other or self._sig == other._sig

    def __hash__(self) -> int:
        return hash(self._sig)

    def __repr__(self) -> str:
        return (
            f"Plan(start={fmt_pos(self.start)}, goal={fmt_pos(self.goal)}, "
            f"root={self.pretty()!r})"
        )

    # --- queries ---------------------------------------------------------------------------
    @cached_property
    def _policy(self) -> dict[Pos, Action]:
        return {
            v.node.state: v.node.action for v in _walk(self.root) if isinstance(v.node, Act)
        }

    def to_policy(self) -> dict[Pos, Action]:
        """Map every Act node's state to its action (a fresh dict)."""
        return dict(self._policy)

    def envelope(self) -> frozenset[Pos]:
        """Every state appearing anywhere in the plan (including the goal leaves)."""
        return frozenset(v.node.state for v in _walk(self.root))

    def stats(self) -> PlanStats:
        """Node counts, Act depth, loop/ref counts and envelope size."""
        nodes = act_nodes = depth = loops = refs = 0
        states: set[Pos] = set()
        for visit in _walk(self.root):
            node = visit.node
            nodes += 1
            states.add(node.state)
            if isinstance(node, Act):
                act_nodes += 1
                depth = max(depth, visit.act_level + 1)
            elif isinstance(node, Loop):
                loops += 1
            elif isinstance(node, Ref):
                refs += 1
        return PlanStats(
            nodes=nodes,
            act_nodes=act_nodes,
            depth=depth,
            loops=loops,
            refs=refs,
            envelope_size=len(states),
        )

    # --- rendering -------------------------------------------------------------------------
    def pretty_node(self, node: PlanNode, parent_state: Pos | None) -> str:
        """Deck-style description of ``node`` as the target of a branch from ``parent_state``.

        ``goal`` | ``retry {a}`` (loop to the parent) | ``↺(r,c)`` (loop to another ancestor)
        | ``{b}-plan`` (an Act choosing ``b``, or a Ref to a state whose action is ``b``).
        """
        if isinstance(node, Goal):
            return "goal"
        if isinstance(node, Loop):
            if parent_state is not None and node.state == parent_state:
                action = self._policy.get(parent_state)
                if action is not None:
                    return f"retry {action}"
            return f"↺{fmt_pos(node.state)}"
        if isinstance(node, Act):
            return f"{node.action}-plan"
        action = self._policy.get(node.state)
        return f"{action}-plan" if action is not None else f"→{fmt_pos(node.state)}"

    def pretty(self) -> str:
        """Deck one-liner for the root, e.g. ``E → {arrived: S-plan, slipped: retry E}``."""
        root = self.root
        if not isinstance(root, Act):
            return self.pretty_node(root, None)
        parts = [
            f"{branch_label(root.state, br)}: {self.pretty_node(br.node, root.state)}"
            for br in root.branches
        ]
        return f"{root.action} → {{{PRETTY_SEPARATOR.join(parts)}}}"

    def _node_text(self, node: PlanNode, parent: Act | None) -> tuple[str, str, str | None]:
        """``(kind, text, action)`` of a node for :meth:`to_tree`."""
        if isinstance(node, Act):
            return "act", f"{node.action} @ {fmt_pos(node.state)}", str(node.action)
        if isinstance(node, Goal):
            return "goal", f"goal {fmt_pos(node.state)}", None
        if isinstance(node, Loop):
            if parent is not None and node.state == parent.state:
                return "loop", f"↺ retry {parent.action}", None
            return "loop", f"↺ {fmt_pos(node.state)}", None
        action = self._policy.get(node.state)
        name = "?" if action is None else str(action)
        return "ref", f"→ {name}-plan @ {fmt_pos(node.state)}", None

    def to_tree(self, depth: int) -> dict[str, Any]:
        """Nested dict view for the UI, with Act nodes below ``depth`` Act levels truncated.

        Each dict has ``state``, ``kind`` (``act``/``goal``/``loop``/``ref``/``truncated``),
        ``action`` (str or None), ``edge`` (branch label from the parent, None at the root;
        also available as ``label``), ``text`` (1-indexed display string) and ``children``.
        An Act node is truncated when the number of Act nodes on its root path, itself
        included, exceeds ``depth`` (so ``to_tree(stats().depth)`` is the full plan).
        """
        result: dict[str, Any] | None = None
        stack: list[tuple[_Visit, list[dict[str, Any]] | None]] = [
            (_Visit(self.root, None, None, 0), None)
        ]
        while stack:
            visit, siblings = stack.pop()
            node = visit.node
            kind, text, action = self._node_text(node, visit.parent)
            truncated = isinstance(node, Act) and visit.act_level + 1 > depth
            if truncated:
                kind, text = "truncated", f"{text} {TRUNCATED_MARK}"
            edge = (
                None
                if visit.parent is None or visit.branch is None
                else branch_label(visit.parent.state, visit.branch)
            )
            entry: dict[str, Any] = {
                "state": node.state,
                "kind": kind,
                "action": action,
                "edge": edge,
                "label": edge,
                "text": text,
                "children": [],
            }
            if siblings is None:
                result = entry
            else:
                siblings.append(entry)
            if isinstance(node, Act) and not truncated:
                for br in reversed(node.branches):
                    child = _Visit(br.node, node, br, visit.act_level + 1)
                    stack.append((child, entry["children"]))
        assert result is not None
        return result


@dataclass
class _Frame:
    """An Act node under construction in :func:`plan_from_policy`."""

    state: Pos
    action: Action
    outcomes: tuple[Outcome, ...]
    branches: list[Branch] = field(default_factory=list)

    def done(self) -> bool:
        return len(self.branches) == len(self.outcomes)


def plan_from_policy(policy: Mapping[Pos, Action], model: ProblemModel, start: Pos) -> Plan:
    """Unfold ``policy`` into a plan tree by depth-first search from ``start``.

    The first visit of a state becomes an :class:`Act` with one branch per outcome of
    ``model.results``; a state on the current DFS path becomes a :class:`Loop`; a state
    expanded elsewhere becomes a :class:`Ref`; the goal becomes :class:`Goal`. Raises
    ``ValueError`` if a reachable non-goal state has no action in ``policy``.
    """
    goal = model.goal
    if start == goal:
        return Plan(root=Goal(start), start=start, goal=goal)
    expanded: set[Pos] = set()
    on_path: set[Pos] = set()
    stack: list[_Frame] = []

    def open_frame(s: Pos) -> None:
        action = policy.get(s)
        if action is None:
            raise ValueError(f"policy has no action for reachable state {fmt_pos(s)}")
        expanded.add(s)
        on_path.add(s)
        stack.append(_Frame(s, action, model.results(s, action)))

    open_frame(start)
    root: Act | None = None
    while stack:
        frame = stack[-1]
        if frame.done():
            stack.pop()
            on_path.discard(frame.state)
            node = Act(frame.state, frame.action, tuple(frame.branches))
            if stack:
                parent = stack[-1]
                outcome = parent.outcomes[len(parent.branches)]
                parent.branches.append(Branch(outcome.state, outcome.labels, node))
            else:
                root = node
            continue
        outcome = frame.outcomes[len(frame.branches)]
        s = outcome.state
        leaf: PlanNode
        if s == goal:
            leaf = Goal(s)
        elif s in on_path:
            leaf = Loop(s)
        elif s in expanded:
            leaf = Ref(s)
        else:
            open_frame(s)
            continue
        frame.branches.append(Branch(s, outcome.labels, leaf))
    assert root is not None
    return Plan(root=root, start=start, goal=goal)
