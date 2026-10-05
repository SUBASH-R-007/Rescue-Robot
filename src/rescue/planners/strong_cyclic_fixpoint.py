"""Reference strong-cyclic planner: the fixpoint algorithm of Cimatti, Pistore, Roveri &
Traverso (2003), "Weak, strong, and strong cyclic planning via symbolic model checking".

It works on explicit state-action pairs instead of BDDs::

    SA := {(s, a) : s free non-goal state, a in ACTIONS}
    repeat
        old := SA
        PruneOutgoing:    repeatedly drop (s, a) with an outcome not in states(SA) ∪ {goal}
        PruneUnconnected: Conn := ∅
                          repeat Conn := {(s,a) ∈ SA : Results(s,a) ∩ ({goal} ∪ states(Conn)) ≠ ∅}
                          until stable;  SA := Conn
    until SA = old

``states(SA) ∪ {goal}`` is the strong-cyclic winning region. ``RemoveNonProgress`` then
extracts a policy by distance layers from the goal so every chosen action has an outcome
strictly closer (in layers) to the goal. Cimatti et al. keep *every* such action; a single
policy has to pick one, and it prefers an action whose intended (``arrived``) outcome makes
the progress, so the policy is not just strong cyclic but also reasonably fast to execute
(a wall bump whose only progress is a rare skid is a valid choice, but a terrible one). Both
prunes are computed with worklists (same fixpoints, linear time), so a 30x30 map takes well
under a second.

The planner exists to cross-check AND-OR search in tests and as its last-resort fallback.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable

from rescue.grid import Pos
from rescue.model import ACTIONS, Action, OutcomeKind, ProblemModel

Pair = tuple[Pos, Action]
#: Sort key of a progress action: (0, layer of the arrived outcome, ACTIONS index) when the
#: intended move itself enters a lower layer, else (1, 0, ACTIONS index).
ProgressKey = tuple[int, int, int]


def _outcomes(model: ProblemModel, pair: Pair) -> list[Pos]:
    """Distinct next states of a state-action pair."""
    return [o.state for o in model.results(*pair)]


def _users(model: ProblemModel, sa: Iterable[Pair]) -> dict[Pos, list[Pair]]:
    """Index: state ``t`` -> pairs of ``sa`` having ``t`` among their outcomes."""
    users: dict[Pos, list[Pair]] = {}
    for pair in sa:
        for t in _outcomes(model, pair):
            users.setdefault(t, []).append(pair)
    return users


def _prune_outgoing(model: ProblemModel, sa: set[Pair]) -> set[Pair]:
    """Greatest subset of ``sa`` whose outcomes all lie in ``states(subset) ∪ {goal}``."""
    goal = model.goal
    kept = set(sa)
    pairs_at: Counter[Pos] = Counter(s for s, _ in kept)
    users = _users(model, kept)
    worklist = [
        pair
        for pair in kept
        if any(t != goal and pairs_at[t] == 0 for t in _outcomes(model, pair))
    ]
    while worklist:
        pair = worklist.pop()
        if pair not in kept:
            continue
        kept.discard(pair)
        s = pair[0]
        pairs_at[s] -= 1
        if pairs_at[s] == 0:  # s left states(SA): every pair leading to s must go
            worklist.extend(p for p in users.get(s, ()) if p in kept)
    return kept


def _prune_unconnected(model: ProblemModel, sa: set[Pair]) -> set[Pair]:
    """Least fixpoint ``Conn`` of pairs in ``sa`` with an outcome in ``{goal} ∪ states(Conn)``."""
    users = _users(model, sa)
    conn: set[Pair] = set()
    reached: set[Pos] = {model.goal}
    queue: deque[Pos] = deque([model.goal])
    while queue:
        t = queue.popleft()
        for pair in users.get(t, ()):
            if pair not in conn:
                conn.add(pair)
                s = pair[0]
                if s not in reached:
                    reached.add(s)
                    queue.append(s)
    return conn


def strong_cyclic_pairs(model: ProblemModel) -> frozenset[Pair]:
    """The fixpoint set SA of state-action pairs from which the goal is strong-cyclic."""
    goal = model.goal
    sa: set[Pair] = {(s, a) for s in model.states() if s != goal for a in ACTIONS}
    while True:
        old = sa
        sa = _prune_outgoing(model, sa)
        sa = _prune_unconnected(model, sa)
        if sa == old:
            return frozenset(sa)


def winning_region(model: ProblemModel) -> frozenset[Pos]:
    """All states with a strong-cyclic solution: ``states(SA) ∪ {goal}``."""
    return frozenset({s for s, _ in strong_cyclic_pairs(model)} | {model.goal})


def _progress_key(
    model: ProblemModel, pair: Pair, layer_of: dict[Pos, int]
) -> ProgressKey | None:
    """Rank ``pair`` as a RemoveNonProgress choice; ``None`` if no outcome is layered yet.

    Prefers an action whose ``arrived`` outcome lands in a lower layer (lowest layer first),
    then any action with some outcome in a lower layer; ties go to the ``ACTIONS`` order.
    Only the outcome *support* and labels are used, never probabilities (D5).
    """
    outcomes = model.results(*pair)
    if not any(o.state in layer_of for o in outcomes):
        return None
    index = ACTIONS.index(pair[1])
    for o in outcomes:
        if OutcomeKind.ARRIVED in o.labels and o.state in layer_of:
            return (0, layer_of[o.state], index)
    return (1, 0, index)


def strong_cyclic_policy(model: ProblemModel, start: Pos) -> dict[Pos, Action] | None:
    """A strong-cyclic policy from ``start`` (``{}`` at the goal), or ``None`` if none exists.

    RemoveNonProgress: layer ``L0 = {goal}``; layer ``i+1`` takes every state not yet layered
    that has a pair in SA with an outcome in ``L0 ∪ … ∪ Li``. Among those pairs it chooses
    the one whose intended (``arrived``) outcome is in the lowest layer, falling back to any
    progressing pair, ties broken by ``ACTIONS`` order (see :func:`_progress_key`). The
    result is restricted to states reachable from ``start``.
    """
    goal = model.goal
    if start == goal:
        return {}
    sa = strong_cyclic_pairs(model)
    if start not in {s for s, _ in sa}:
        return None
    users = _users(model, sa)
    layer_of: dict[Pos, int] = {goal: 0}
    frontier: list[Pos] = [goal]
    full: dict[Pos, Action] = {}
    layer = 0
    while frontier:
        layer += 1
        candidates = sorted(
            {s for t in frontier for s, _ in users.get(t, ()) if s not in layer_of}
        )
        new_layer: list[Pos] = []
        for s in candidates:
            keyed = [
                (key, a)
                for a in ACTIONS
                if (s, a) in sa and (key := _progress_key(model, (s, a), layer_of)) is not None
            ]
            if keyed:
                full[s] = min(keyed)[1]
                new_layer.append(s)
        for s in new_layer:  # after the loop: layers are synchronous
            layer_of[s] = layer
        frontier = new_layer

    policy: dict[Pos, Action] = {}
    queue: deque[Pos] = deque([start])
    seen: set[Pos] = {start}
    while queue:
        s = queue.popleft()
        if s == goal:
            continue
        action = full[s]  # every outcome of a pair in SA lies in states(SA) ∪ {goal}
        policy[s] = action
        for t in _outcomes(model, (s, action)):
            if t not in seen:
                seen.add(t)
                queue.append(t)
    return policy
