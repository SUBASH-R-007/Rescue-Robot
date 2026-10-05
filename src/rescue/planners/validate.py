"""Strong-cyclic policy validation.

A policy is *strong cyclic* from ``start`` (Cimatti et al. 2003) when, in the graph of states
reachable from ``start`` under the policy (every possible outcome followed, the goal
absorbing),

* every reachable non-goal state has an action (no ``missing`` states), and
* every reachable state can still reach the goal (no ``dead`` states).

Under fair nondeterminism such a policy reaches the goal with probability 1.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from rescue.grid import Pos
from rescue.model import Action, ProblemModel


@dataclass(frozen=True)
class ValidationReport:
    """Result of :func:`validate_policy`.

    ``reachable``: states reachable from the start under the policy (incl. start and goal if
    reached). ``missing``: reachable non-goal states without an action. ``dead``: reachable
    non-goal states that cannot reach the goal in the outcome graph (a missing state has no
    successors, so it is also dead). ``ok`` iff both are empty.
    """

    ok: bool
    reachable: frozenset[Pos]
    missing: frozenset[Pos]
    dead: frozenset[Pos]


def validate_policy(
    policy: Mapping[Pos, Action], model: ProblemModel, start: Pos
) -> ValidationReport:
    """Check that ``policy`` is strong cyclic from ``start`` on ``model``."""
    goal = model.goal
    reachable: set[Pos] = {start}
    missing: set[Pos] = set()
    predecessors: dict[Pos, list[Pos]] = {}
    queue: deque[Pos] = deque([start])
    while queue:
        s = queue.popleft()
        if s == goal:
            continue  # absorbing
        action = policy.get(s)
        if action is None:
            missing.add(s)
            continue
        for outcome in model.results(s, action):
            t = outcome.state
            predecessors.setdefault(t, []).append(s)
            if t not in reachable:
                reachable.add(t)
                queue.append(t)

    # Reverse BFS from the goal over the reachable outcome graph.
    can_reach: set[Pos] = set()
    if goal in reachable:
        can_reach.add(goal)
        back: deque[Pos] = deque([goal])
        while back:
            t = back.popleft()
            for s in predecessors.get(t, ()):
                if s not in can_reach:
                    can_reach.add(s)
                    back.append(s)
    dead = {s for s in reachable if s != goal and s not in can_reach}
    return ValidationReport(
        ok=not missing and not dead,
        reachable=frozenset(reachable),
        missing=frozenset(missing),
        dead=frozenset(dead),
    )


def is_strong_cyclic(policy: Mapping[Pos, Action], model: ProblemModel, start: Pos) -> bool:
    """True if ``policy`` is a strong-cyclic solution from ``start``."""
    return validate_policy(policy, model, start).ok
