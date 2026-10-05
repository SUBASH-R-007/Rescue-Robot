"""Actions, the nondeterministic transition model and the planner's ``Results(s, a)``.

* :class:`TransitionModel` gives the outcome distribution at a position (slip depends on the
  cell the robot stands on, D4) and samples outcomes by inverse CDF from one uniform (D3).
* :class:`ProblemModel` is the planner's view: a frozen passability mask (unknown assumed
  open) on which ``results(s, a)`` returns the merged set of possible next positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from rescue.config import MAX_SLIP_P, PROB_EPS, SKID_FRACTION, SLIP_STAY_FRACTION
from rescue.grid import Pos

if TYPE_CHECKING:
    from rescue.belief import BeliefMap


class Action(Enum):
    """Compass moves; the value is the ``(d_row, d_col)`` delta."""

    N = (-1, 0)
    E = (0, 1)
    S = (1, 0)
    W = (0, -1)

    @property
    def delta(self) -> tuple[int, int]:
        """``(d_row, d_col)`` of the move."""
        return self.value

    @property
    def left(self) -> Action:
        """Direction to the left of this heading (N→W, E→N, S→E, W→S)."""
        return _LEFT[self]

    @property
    def right(self) -> Action:
        """Direction to the right of this heading (N→E, E→S, S→W, W→N)."""
        return _RIGHT[self]

    def __str__(self) -> str:
        return self.name


_LEFT: dict[Action, Action] = {
    Action.N: Action.W,
    Action.E: Action.N,
    Action.S: Action.E,
    Action.W: Action.S,
}
_RIGHT: dict[Action, Action] = {
    Action.N: Action.E,
    Action.E: Action.S,
    Action.S: Action.W,
    Action.W: Action.N,
}

#: Fixed action iteration order (D13).
ACTIONS: tuple[Action, ...] = (Action.N, Action.E, Action.S, Action.W)


def step_pos(p: Pos, a: Action) -> Pos:
    """Position one step from ``p`` in direction ``a`` (may be off-grid)."""
    dr, dc = a.value
    return (p[0] + dr, p[1] + dc)


class OutcomeKind(Enum):
    """What happened to a move."""

    ARRIVED = "arrived"
    SLIPPED = "slipped"
    SKID_L = "skid-L"
    SKID_R = "skid-R"

    def __str__(self) -> str:
        return self.value


#: Fixed outcome order used for inverse-CDF sampling and label ordering (D3).
OUTCOME_ORDER: tuple[OutcomeKind, ...] = (
    OutcomeKind.ARRIVED,
    OutcomeKind.SLIPPED,
    OutcomeKind.SKID_L,
    OutcomeKind.SKID_R,
)


def kind_direction(kind: OutcomeKind, action: Action) -> Action | None:
    """Direction the robot actually moves for ``kind`` (``None`` = stays in place)."""
    if kind is OutcomeKind.ARRIVED:
        return action
    if kind is OutcomeKind.SKID_L:
        return action.left
    if kind is OutcomeKind.SKID_R:
        return action.right
    return None


Distribution = tuple[tuple[OutcomeKind, float], ...]


@lru_cache(maxsize=256)
def outcome_distribution(p: float) -> Distribution:
    """Outcome probabilities for slip parameter ``p``, in ``OUTCOME_ORDER``.

    Intended ``1 - p``, stay ``SLIP_STAY_FRACTION * p``, skid ``SKID_FRACTION * p`` split
    equally left/right.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"slip probability must be in [0, 1], got {p}")
    skid = SKID_FRACTION * p / 2.0
    return (
        (OutcomeKind.ARRIVED, 1.0 - p),
        (OutcomeKind.SLIPPED, SLIP_STAY_FRACTION * p),
        (OutcomeKind.SKID_L, skid),
        (OutcomeKind.SKID_R, skid),
    )


@lru_cache(maxsize=256)
def _support_for_p(p: float) -> tuple[OutcomeKind, ...]:
    return tuple(kind for kind, prob in outcome_distribution(p) if prob > PROB_EPS)


def _check_p(name: str, p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= MAX_SLIP_P:
        raise ValueError(f"{name} must be in [0, {MAX_SLIP_P}], got {p}")
    return p


class TransitionModel(ABC):
    """Pluggable nondeterministic dynamics; slip depends on the robot's current cell (D4)."""

    @abstractmethod
    def slip_p(self, pos: Pos) -> float:
        """Slip parameter ``p`` when moving from ``pos``."""

    @abstractmethod
    def max_p(self) -> float:
        """Largest slip parameter used anywhere on the map."""

    def distribution(self, pos: Pos) -> Distribution:
        """Outcome probabilities at ``pos`` in ``OUTCOME_ORDER``."""
        return outcome_distribution(self.slip_p(pos))

    def support(self, pos: Pos) -> tuple[OutcomeKind, ...]:
        """Outcome kinds with probability above ``PROB_EPS``, in ``OUTCOME_ORDER``."""
        return _support_for_p(self.slip_p(pos))

    def sample_kind(self, pos: Pos, u: float) -> OutcomeKind:
        """Map one uniform ``u`` in ``[0, 1)`` to an outcome by inverse CDF (D3).

        Uses strict ``u < cumulative`` and never returns a kind outside :meth:`support`; if
        float rounding leaves ``u`` above the total, the last supported kind is returned.
        """
        cumulative = 0.0
        last: OutcomeKind | None = None
        for kind, prob in self.distribution(pos):
            cumulative += prob
            if prob > PROB_EPS:
                last = kind
                if u < cumulative:
                    return kind
        if last is None:  # pragma: no cover - ARRIVED always has prob >= 1 - MAX_SLIP_P
            raise RuntimeError("outcome distribution has no positive-probability outcome")
        return last


class UniformSlip(TransitionModel):
    """The same slip parameter ``p`` on every cell."""

    def __init__(self, p: float) -> None:
        self.p = _check_p("p", p)

    def slip_p(self, pos: Pos) -> float:
        """Slip parameter (independent of ``pos``)."""
        return self.p

    def max_p(self) -> float:
        """The uniform slip parameter."""
        return self.p

    def __repr__(self) -> str:
        return f"UniformSlip(p={self.p})"


class TerrainSlip(TransitionModel):
    """Slip ``high_p`` on high-slip cells (``high_mask``), ``base_p`` elsewhere."""

    def __init__(self, base_p: float, high_p: float, high_mask: np.ndarray) -> None:
        self.base_p = _check_p("base_p", base_p)
        self.high_p = _check_p("high_p", high_p)
        mask = np.array(high_mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"high_mask must be 2-D, got shape {mask.shape}")
        mask.flags.writeable = False
        self.high_mask = mask
        rows, cols = np.nonzero(mask)
        self._high: frozenset[Pos] = frozenset(
            (int(r), int(c)) for r, c in zip(rows, cols, strict=True)
        )

    def slip_p(self, pos: Pos) -> float:
        """``high_p`` if ``pos`` is high-slip terrain, else ``base_p``."""
        return self.high_p if pos in self._high else self.base_p

    def max_p(self) -> float:
        """Largest slip parameter actually present on the map."""
        return max(self.base_p, self.high_p) if self._high else self.base_p

    def __repr__(self) -> str:
        return (
            f"TerrainSlip(base_p={self.base_p}, high_p={self.high_p}, "
            f"high_cells={len(self._high)})"
        )


@dataclass(frozen=True)
class Outcome:
    """One possible next position with every outcome label that leads there."""

    state: Pos
    labels: tuple[OutcomeKind, ...]


class ProblemModel:
    """Planner's model on a frozen passability mask (unknown assumed open)."""

    def __init__(self, passable: np.ndarray, goal: Pos, transition: TransitionModel) -> None:
        mask = np.array(passable, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"passable must be 2-D, got shape {mask.shape}")
        mask.flags.writeable = False
        self.passable: np.ndarray = mask
        self.rows: int = int(mask.shape[0])
        self.cols: int = int(mask.shape[1])
        self.goal: Pos = (int(goal[0]), int(goal[1]))
        self.transition = transition
        rows, cols = np.nonzero(mask)
        # Python set of free cells: membership also covers the bounds check (hot path).
        self._free: frozenset[Pos] = frozenset(
            (int(r), int(c)) for r, c in zip(rows, cols, strict=True)
        )
        self._cache: dict[tuple[Pos, Action], tuple[Outcome, ...]] = {}

    def is_goal(self, s: Pos) -> bool:
        """True if ``s`` is the goal (survivor) cell."""
        return s == self.goal

    def is_free(self, p: Pos) -> bool:
        """True if ``p`` is in bounds and passable."""
        return p in self._free

    def actions(self, s: Pos) -> tuple[Action, ...]:
        """Actions available at ``s`` (always ``ACTIONS``; blocked moves just stay)."""
        return ACTIONS

    def results(self, s: Pos, a: Action) -> tuple[Outcome, ...]:
        """Possible next positions of ``a`` from ``s`` with merged labels (cached).

        Blocked or off-grid targets collapse to ``s``; outcomes landing on the same cell merge
        (labels in ``OUTCOME_ORDER``); outcomes are ordered by their first label.
        """
        key = (s, a)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        free = self._free
        r, c = s
        merged: dict[Pos, list[OutcomeKind]] = {}
        for kind in self.transition.support(s):
            direction = kind_direction(kind, a)
            if direction is None:
                target = s
            else:
                dr, dc = direction.value
                target = (r + dr, c + dc)
                if target not in free:
                    target = s
            labels = merged.get(target)
            if labels is None:
                merged[target] = [kind]
            else:
                labels.append(kind)
        result = tuple(Outcome(t, tuple(labels)) for t, labels in merged.items())
        self._cache[key] = result
        return result

    def states(self) -> list[Pos]:
        """All free cells in row-major order."""
        return sorted(self._free)

    @classmethod
    def from_belief(cls, belief: BeliefMap, transition: TransitionModel) -> ProblemModel:
        """Build the model from a belief map (optimistic: UNKNOWN counts as passable)."""
        return cls(belief.passable_mask(), belief.survivor, transition)
