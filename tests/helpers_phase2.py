"""Shared helpers for the Phase 2 planner tests (plan structures, AND-OR, validators).

Besides map and model builders this holds the AND-OR search subclasses with forced (hostile)
action orderings and the plan-structure check, shared by ``test_andor`` and the property
tests. Every tree walk here is iterative, so the helpers work on plans of any depth.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rescue.belief import BeliefMap
from rescue.config import PlannerConfig
from rescue.grid import Pos, TrueGrid
from rescue.model import ACTIONS, Action, ProblemModel, TransitionModel, UniformSlip, step_pos
from rescue.plan import Act, Goal, Loop, Plan, Ref
from rescue.planners.andor import AndOrSearch
from rescue.planners.bfs import bfs_distances, reachable
from rescue.world import World

#: Directory holding the bundled ASCII maps.
MAPS_DIR: Path = Path(__file__).resolve().parent.parent / "maps"
#: Slip parameter used for the deck reproduction (deck: 0.80 / 0.15 / 0.05).
DECK_P: float = 0.2
#: Regeneration attempts when drawing a connected random map.
MAX_MAP_ATTEMPTS: int = 500


def deck_model(p: float = DECK_P, radius: int = 1) -> ProblemModel:
    """Planning model for ``maps/deck_demo.txt`` after the t=0 sensing from the start."""
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    transition = UniformSlip(p)
    world = World(grid, transition, sensor_radius=radius, seed=0)
    belief = BeliefMap(grid.rows, grid.cols, grid.survivor)
    belief.update(world.sense().cells)
    return ProblemModel.from_belief(belief, transition)


def deck_start() -> Pos:
    """Start cell of ``maps/deck_demo.txt``."""
    return TrueGrid.load(MAPS_DIR / "deck_demo.txt").start


def ascii_model(text: str, transition: TransitionModel) -> tuple[ProblemModel, Pos]:
    """Model on the TRUE map of an ASCII grid (fully known) plus its start cell."""
    grid = TrueGrid.from_ascii(text)
    return ProblemModel(grid.passable_mask(), grid.survivor, transition), grid.start


def random_passable(rng: np.random.Generator, rows: int, cols: int, density: float) -> np.ndarray:
    """Bool mask with each cell free with probability ``1 - density``."""
    return rng.random((rows, cols)) >= density


def free_cells(passable: np.ndarray) -> list[Pos]:
    """All free cells, row-major."""
    rows, cols = np.nonzero(passable)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]


def random_instance(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    density: float,
    *,
    connected: bool = False,
    distinct: bool = True,
) -> tuple[np.ndarray, Pos, Pos]:
    """Random ``(passable, start, goal)``; start and goal are free cells.

    With ``connected`` the map is redrawn until start reaches goal; with ``distinct`` start
    and goal differ (the grid must then have at least two cells).
    """
    for _ in range(MAX_MAP_ATTEMPTS):
        passable = random_passable(rng, rows, cols, density)
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        i = int(rng.integers(len(cells)))
        j = int(rng.integers(len(cells)))
        if distinct and i == j:
            continue
        start, goal = cells[i], cells[j]
        passable[start] = True
        passable[goal] = True
        if connected and not reachable(passable, start, goal):
            continue
        return passable, start, goal
    raise RuntimeError("could not draw a suitable random map")


def walk_nodes(node: Any) -> Iterator[Any]:
    """Every plan node in the tree rooted at ``node`` (pre-order, leaves included)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, Act):
            stack.extend(br.node for br in reversed(current.branches))


def walk_branches(node: Any) -> Iterator[tuple[Any, Any]]:
    """Every ``(act_node, branch)`` pair in the tree rooted at ``node`` (pre-order)."""
    stack: list[tuple[Any, Any]] = []
    if isinstance(node, Act):
        stack.extend((node, br) for br in reversed(node.branches))
    while stack:
        act, br = stack.pop()
        yield act, br
        if isinstance(br.node, Act):
            stack.extend((br.node, child) for child in reversed(br.node.branches))


def bump_actions(model: ProblemModel, s: Pos) -> list[Action]:
    """Actions whose intended target from ``s`` is a wall or debris (in ACTIONS order)."""
    return [a for a in ACTIONS if not model.is_free(step_pos(s, a))]


def all_stay_actions(model: ProblemModel, s: Pos) -> list[Action]:
    """Actions whose every outcome from ``s`` is ``s`` itself (in ACTIONS order)."""
    return [a for a in ACTIONS if [o.state for o in model.results(s, a)] == [s]]


def run_deterministic(
    policy: dict[Pos, Action], model: ProblemModel, start: Pos, max_steps: int
) -> list[Pos]:
    """Execute ``policy`` on a deterministic (p = 0) model; returns the visited states."""
    s = start
    visited = [s]
    for _ in range(max_steps):
        if model.is_goal(s):
            break
        outcomes = model.results(s, policy[s])
        assert len(outcomes) == 1
        s = outcomes[0].state
        visited.append(s)
    return visited


def serpentine_mask(n: int) -> np.ndarray:
    """``n`` x ``n`` corridor: even rows open, odd rows open only at alternating ends."""
    mask = np.zeros((n, n), dtype=bool)
    mask[0::2, :] = True
    for k, r in enumerate(range(1, n, 2)):
        mask[r, n - 1 if k % 2 == 0 else 0] = True
    return mask


def farthest_cell(passable: np.ndarray, start: Pos) -> Pos:
    """A free cell at maximum BFS distance from ``start`` (first in row-major order)."""
    dist = bfs_distances(passable, start)
    r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(r), int(c)


def expected_steps(policy: Mapping[Pos, Action], model: ProblemModel, start: Pos) -> float:
    """Exact expected number of steps to the goal under ``policy`` (true probabilities).

    Solves ``E[s] = 1 + sum over outcomes t of P(s -> t) * E[t]`` with ``E[goal] = 0`` over
    the states reachable under the policy. The policy must be strong cyclic from ``start``.
    """
    states: list[Pos] = [start]
    index = {start: 0}
    for s in states:  # grows while iterating: BFS over the policy graph
        if s == model.goal:
            continue
        for o in model.results(s, policy[s]):
            if o.state not in index:
                index[o.state] = len(states)
                states.append(o.state)
    system = np.eye(len(states))
    rhs = np.zeros(len(states))
    for s in states:
        if s == model.goal:
            continue
        i = index[s]
        rhs[i] = 1.0
        probs = dict(model.transition.distribution(s))
        for o in model.results(s, policy[s]):
            system[i, index[o.state]] -= sum(probs[kind] for kind in o.labels)
    return float(np.linalg.solve(system, rhs)[index[start]])


# --- AND-OR search with forced action orderings ------------------------------------------------
class BumpFirst(AndOrSearch):
    """Tries every wall/debris-bump action before the default order."""

    def ordered_actions(self, s: Pos) -> list[Action]:
        base = super().ordered_actions(s)
        bumps = bump_actions(self.model, s)
        return bumps + [a for a in base if a not in bumps]


class BumpOnly(AndOrSearch):
    """Only offers actions whose intended move is blocked."""

    def ordered_actions(self, s: Pos) -> list[Action]:
        return bump_actions(self.model, s)


class AllStayFirst(AndOrSearch):
    """Tries every action whose only outcome is "stay" (even with slip) first."""

    def ordered_actions(self, s: Pos) -> list[Action]:
        base = super().ordered_actions(s)
        stays = all_stay_actions(self.model, s)
        return stays + [a for a in base if a not in stays]


class Reversed(AndOrSearch):
    """Worst-first: the default order reversed (farthest intended target first)."""

    def ordered_actions(self, s: Pos) -> list[Action]:
        return list(reversed(super().ordered_actions(s)))


class Textbook(AndOrSearch):
    """The textbook order N, E, S, W with no distance key; records every HAS_PROGRESS verdict."""

    def __init__(self, model: ProblemModel, config: PlannerConfig | None = None) -> None:
        super().__init__(model, config)
        self.verdicts: list[tuple[Pos, Action, bool]] = []

    def ordered_actions(self, s: Pos) -> list[Action]:
        return list(ACTIONS)

    def has_progress(self, s: Pos) -> bool:
        accepted = super().has_progress(s)
        self.verdicts.append((s, self.solved[s], accepted))
        return accepted


def seeded_hostile(seed: int) -> type[AndOrSearch]:
    """Search class whose order at ``s`` is a permutation drawn from a generator keyed by s."""

    class Permuted(AndOrSearch):
        def ordered_actions(self, s: Pos) -> list[Action]:
            base = super().ordered_actions(s)
            rng = np.random.default_rng([seed, s[0], s[1]])
            return [base[i] for i in rng.permutation(len(base))]

    Permuted.__name__ = Permuted.__qualname__ = f"Permuted{seed}"
    return Permuted


#: Adversarial orderings: plans must stay sound under each (completeness is not required).
HOSTILE_CLASSES: tuple[type[AndOrSearch], ...] = (
    Reversed,
    BumpFirst,
    AllStayFirst,
    Textbook,
    *(seeded_hostile(seed) for seed in range(4)),
)


def run_search(
    cls: type[AndOrSearch], model: ProblemModel, start: Pos
) -> tuple[Plan | None, AndOrSearch]:
    """Run a (possibly subclassed) search; returns the plan or ``None`` and the search."""
    search = cls(model, PlannerConfig())
    root = search.run(start)
    plan = None if root is None else Plan(root=root, start=start, goal=model.goal)
    return plan, search


def check_structure(plan: Plan, model: ProblemModel) -> None:
    """Every Act's branches are exactly Results(s, a); leaves point at solved states/goal."""
    policy = plan.to_policy()
    act_states = set()
    for node in walk_nodes(plan.root):
        if isinstance(node, Act):
            act_states.add(node.state)
            want = [(o.state, o.labels) for o in model.results(node.state, node.action)]
            assert [(br.state, br.labels) for br in node.branches] == want
            for br in node.branches:
                assert br.node.state == br.state
        elif isinstance(node, Goal):
            assert node.state == model.goal
        else:
            assert isinstance(node, Loop | Ref)
            assert node.state in policy
    assert set(policy) == act_states
    assert model.goal not in policy
