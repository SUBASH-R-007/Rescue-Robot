"""The simulated true world: robot dynamics, sensing and aftershocks.

Determinism and common random numbers (D3, D9): every step consumes exactly one uniform from
the outcome stream and, when aftershocks are enabled, exactly one ``(u, permutation)`` pair
from the aftershock stream, regardless of the agent or the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rescue.config import DEFAULT_SENSOR_RADIUS, AftershockConfig
from rescue.grid import Cell, Pos, TrueGrid, footprint
from rescue.model import Action, OutcomeKind, TransitionModel, kind_direction, step_pos
from rescue.planners.bfs import reachable

#: Number of independent RNG streams spawned from the seed (outcome, aftershock, misc).
_N_STREAMS = 3


@dataclass(frozen=True)
class Percept:
    """Observed ``(position, true cell)`` pairs."""

    cells: tuple[tuple[Pos, Cell], ...]

    def as_dict(self) -> dict[Pos, Cell]:
        """Observations as a ``{pos: cell}`` dict."""
        return dict(self.cells)


@dataclass(frozen=True)
class StepResult:
    """Everything that happened in one :meth:`World.step`."""

    prev: Pos
    action: Action
    kind: OutcomeKind
    #: Cell the outcome tried to enter (``prev`` for SLIPPED; may be off-grid).
    target: Pos
    new_pos: Pos
    #: The outcome tried to move but the robot stayed (wall or debris).
    blocked: bool
    #: The target was a true DEBRIS cell.
    hit_debris: bool
    percept: Percept
    aftershock_cells: tuple[Pos, ...]


class RngStreams:
    """Independent seeded generators: ``outcome``, ``aftershock`` and ``misc``."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        outcome, aftershock, misc = np.random.SeedSequence(self.seed).spawn(_N_STREAMS)
        self.outcome: np.random.Generator = np.random.default_rng(outcome)
        self.aftershock: np.random.Generator = np.random.default_rng(aftershock)
        self.misc: np.random.Generator = np.random.default_rng(misc)


class World:
    """True-world simulator holding a private mutable copy of the map."""

    def __init__(
        self,
        grid: TrueGrid,
        transition: TransitionModel,
        *,
        sensor_radius: int = DEFAULT_SENSOR_RADIUS,
        seed: int = 0,
        aftershock: AftershockConfig | None = None,
    ) -> None:
        if sensor_radius < 0:
            raise ValueError(f"sensor_radius must be >= 0, got {sensor_radius}")
        self._grid = grid.copy()
        self.transition = transition
        self.sensor_radius = sensor_radius
        self.aftershock = aftershock if aftershock is not None else AftershockConfig()
        self.rng = RngStreams(seed)
        self.pos: Pos = self._grid.start
        self.t: int = 0
        self.aftershock_events: int = 0

    @property
    def grid(self) -> TrueGrid:
        """The current true map (do not mutate; aftershocks change it)."""
        return self._grid

    @property
    def rescued(self) -> bool:
        """True when the robot stands on the survivor cell."""
        return self.pos == self._grid.survivor

    def true_cell(self, p: Pos) -> Cell:
        """True contents of in-grid cell ``p``."""
        if not self._grid.in_bounds(p):
            raise ValueError(f"position {p} is off-grid")
        return Cell(int(self._grid.cells[p]))

    def _footprint(self, p: Pos) -> list[Pos]:
        return footprint(p, self.sensor_radius, self._grid.rows, self._grid.cols)

    def sense(self) -> Percept:
        """True values of every cell within the sensor radius of the robot (incl. own cell)."""
        cells = self._grid.cells
        return Percept(tuple((p, Cell(int(cells[p]))) for p in self._footprint(self.pos)))

    def step(self, action: Action) -> StepResult:
        """Execute ``action``: draw outcome → move → aftershock → sense (D3, D9)."""
        prev = self.pos
        u = float(self.rng.outcome.random())
        kind = self.transition.sample_kind(prev, u)
        direction = kind_direction(kind, action)
        target = prev if direction is None else step_pos(prev, direction)
        moved = target != prev
        in_grid = self._grid.in_bounds(target)
        hit_debris = moved and in_grid and self._grid.is_debris(target)
        blocked = moved and (not in_grid or hit_debris)
        new_pos = prev if blocked else target

        aftershock_cells = self._aftershock(new_pos)
        self.pos = new_pos
        percept = self.sense()
        if hit_debris and target not in self._footprint(new_pos):
            # Touch sensing: a bump reveals the bumped cell (D6).
            percept = Percept((*percept.cells, (target, Cell.DEBRIS)))
        self.t += 1
        return StepResult(
            prev=prev,
            action=action,
            kind=kind,
            target=target,
            new_pos=new_pos,
            blocked=blocked,
            hit_debris=hit_debris,
            percept=percept,
            aftershock_cells=aftershock_cells,
        )

    def _aftershock(self, robot: Pos) -> tuple[Pos, ...]:
        """Maybe drop debris on unsensed open cells, keeping the survivor reachable (D9).

        When enabled, ``(u, permutation)`` is drawn every step whether or not an event fires.
        """
        config = self.aftershock
        if not config.enabled:
            return ()
        rows, cols = self._grid.rows, self._grid.cols
        u_event = float(self.rng.aftershock.random())
        order = self.rng.aftershock.permutation(rows * cols)
        if u_event >= config.q:
            return ()
        cells = self._grid.cells
        excluded = set(self._footprint(robot))
        excluded.update((robot, self._grid.start, self._grid.survivor))
        passable = self._grid.passable_mask()
        added: list[Pos] = []
        for index in order.tolist():
            if len(added) >= config.k:
                break
            p = divmod(index, cols)
            if p in excluded or cells[p] != Cell.OPEN:
                continue
            passable[p] = False
            if reachable(passable, robot, self._grid.survivor):
                added.append(p)
            else:
                passable[p] = True
        for p in added:
            self._grid.add_debris(p)
        if added:
            self.aftershock_events += 1
        return tuple(added)
