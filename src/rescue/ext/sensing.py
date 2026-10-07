"""Noisy sensing (brief §8.3, docs/plan.md §3.14 and §3.15.2): a two-sided debris sensor and a
probabilistic belief map.

* :class:`NoisySensor` corrupts a true :class:`~rescue.world.Percept`: each true ``DEBRIS`` cell
  is reported ``OPEN`` with probability ``false_negative`` and each true ``OPEN`` cell is
  reported ``DEBRIS`` with probability ``false_positive`` (round 2; default 0, the round-1
  one-sided sensor). ``SURVIVOR`` cells are never changed, and cells in ``exempt`` (the cell a
  bump touched, the robot's own cell) are always reported truthfully. Every call draws one
  uniform per *grid cell* (a fixed ``rows × cols`` block, whatever the percept holds) and a
  cell's verdict uses the uniform at its own index, so the noise at a given (call, cell) is the
  same for every agent and every noise level, whatever path led there (common random numbers
  per (t, cell)); a cell is either debris or free, so one uniform serves both directions.
* :class:`ProbBelief` keeps a debris probability per cell and updates it by Bayes' rule under
  that sensor model (:func:`debris_posterior`). Its categorical view (:meth:`ProbBelief.get`)
  and optimistic passability (:meth:`ProbBelief.passable_mask`) mirror
  :class:`~rescue.belief.BeliefMap`, so agents can use either. With ``false_negative = 0`` and
  ``false_positive = 0`` it makes exactly the same passability decisions; with
  ``false_positive = 0`` its probabilities are bit-identical to the round-1 one-sided filter.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import numpy as np

from rescue.belief import BeliefUpdate
from rescue.config import EXT_MAX_FALSE_POSITIVE, NOISE_STREAM_KEY
from rescue.grid import Cell, Pos, in_bounds
from rescue.world import Percept

__all__ = ["NOISE_STREAM_KEY", "NoisySensor", "ProbBelief", "debris_posterior"]

#: Debris probability of a cell known to be debris / known to be free.
_CERTAIN_DEBRIS = 1.0
_CERTAIN_FREE = 0.0

Observations = Percept | Mapping[Pos, Cell] | Iterable[tuple[Pos, Cell]]


def _check_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _check_false_positive(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < EXT_MAX_FALSE_POSITIVE:
        raise ValueError(f"false_positive must be in [0, {EXT_MAX_FALSE_POSITIVE:g}), got {value}")
    return value


def debris_posterior(prior: float, reads_debris: bool, false_negative: float,
                     false_positive: float) -> float:
    """Debris probability after one reading of a cell (Bayes' rule, two-sided sensor).

    With ``fn`` = ``false_negative``, ``fp`` = ``false_positive`` and ``P`` = ``prior``:

    * reading ``OPEN``:   ``P' = fn·P / (fn·P + (1 − fp)·(1 − P))``;
    * reading ``DEBRIS``: ``P' = (1 − fn)·P / ((1 − fn)·P + fp·(1 − P))``.

    With ``fp = 0`` these are the round-1 one-sided updates bit for bit: the OPEN formula's
    ``(1 − fp)`` factor is exactly ``1.0`` (multiplying by it is exact), and the DEBRIS formula
    is ``x / (x + 0) = 1`` whenever it is defined, so a DEBRIS reading returns ``1.0`` as in
    round 1 (also where the formula is ``0 / 0``: ``P = 0`` or ``fn = 1``, a reading impossible
    under the model, e.g. after the world changed). A zero denominator with ``fp > 0`` or on an
    OPEN reading (again an impossible reading) keeps the prior.
    """
    fn, fp = false_negative, false_positive
    if reads_debris:
        if fp <= 0.0:
            return _CERTAIN_DEBRIS
        numerator = (1.0 - fn) * prior
        denominator = numerator + fp * (1.0 - prior)
    else:
        numerator = fn * prior
        denominator = numerator + (1.0 - fp) * (1.0 - prior)
    # denominator == 0 only for readings impossible under the sensor model (e.g. an OPEN
    # reading of a cell known to be debris with fn = 0); the prior is kept.
    return numerator / denominator if denominator > 0.0 else prior


class NoisySensor:
    """Two-sided debris sensor with its own seeded stream.

    The stream is ``np.random.default_rng(SeedSequence([seed, NOISE_STREAM_KEY]))``,
    independent of the World's outcome and aftershock streams. Each :meth:`observe` call
    consumes exactly ``rows * cols`` uniforms (one per grid cell, row-major), so call ``n``
    gives cell ``(r, c)`` the same uniform in every mission with this seed and grid size: the
    draws are paired per (call, cell), not per position in the percept (whose cells and order
    depend on where the robot stands).
    """

    def __init__(self, false_negative: float, seed: int, rows: int, cols: int,
                 false_positive: float = 0.0) -> None:
        self.false_negative = _check_probability("false_negative", false_negative)
        self.false_positive = _check_false_positive(false_positive)
        self.seed = int(seed)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got {rows}x{cols}")
        self.rows: int = int(rows)
        self.cols: int = int(cols)
        self.rng: np.random.Generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, NOISE_STREAM_KEY])
        )

    def observe(self, percept: Percept, exempt: Collection[Pos] = ()) -> Percept:
        """The percept as the noisy sensor reports it.

        One uniform per grid cell is drawn (always, for common random numbers); for a cell
        ``(r, c)`` not in ``exempt`` with uniform ``u`` at index ``r * cols + c``, a true
        ``DEBRIS`` cell is reported ``OPEN`` when ``u < false_negative`` and a true ``OPEN``
        cell is reported ``DEBRIS`` when ``u < false_positive``. ``SURVIVOR`` cells and exempt
        cells are reported as they are. A percept cell outside the grid raises ``ValueError``.
        """
        rows, cols = self.rows, self.cols
        for p, _ in percept.cells:
            if not in_bounds(p, rows, cols):
                raise ValueError(f"percept cell {p} is outside the {rows}x{cols} grid")
        draws = self.rng.random(rows * cols)
        eps = self.false_negative
        fp = self.false_positive
        if eps <= 0.0 and fp <= 0.0:
            return percept
        exempt_set = {(int(p[0]), int(p[1])) for p in exempt}
        cells: list[tuple[Pos, Cell]] = []
        changed = False
        for p, cell in percept.cells:
            if p in exempt_set:
                cells.append((p, cell))
                continue
            u = draws[int(p[0]) * cols + int(p[1])]
            if cell is Cell.DEBRIS and u < eps:
                cells.append((p, Cell.OPEN))
                changed = True
            elif cell is Cell.OPEN and u < fp:
                cells.append((p, Cell.DEBRIS))
                changed = True
            else:
                cells.append((p, cell))
        return Percept(tuple(cells)) if changed else percept

    def __repr__(self) -> str:
        return (f"NoisySensor(false_negative={self.false_negative}, "
                f"false_positive={self.false_positive}, seed={self.seed}, "
                f"grid={self.rows}x{self.cols})")


class ProbBelief:
    """Per-cell debris probabilities under a two-sided noisy sensor (Bayes filter).

    ``p_debris`` (``float64``) starts at ``prior_debris`` everywhere; ``observed`` (bool) marks
    cells that were ever sensed or touched. Updates per observed cell (``fn`` =
    ``false_negative``, ``fp`` = ``false_positive``; :func:`debris_posterior`):

    * reading ``OPEN``: ``P ← fn·P / (fn·P + (1 − fp)·(1 − P))``; with ``P = 1`` this stays 1
      (a cell known to be debris is not un-seen by a false negative);
    * reading ``DEBRIS``: ``P ← (1 − fn)·P / ((1 − fn)·P + fp·(1 − P))``, which is ``1`` when
      ``fp = 0`` (one DEBRIS reading is final without false positives, as in round 1);
    * touched by a bump: ``P ← 1`` (contact is certain);
    * reading ``SURVIVOR``: ``P ← 0`` and the cell is a known survivor;
    * the robot's own cell (``here``), when ``fp > 0``: ``P ← 0`` (it stands there). With
      ``fp = 0`` ``here`` is ignored so the probabilities stay bit-identical to round 1; the
      OPEN reading of the robot's cell already keeps it passable there (an OPEN reading never
      raises ``P``), so passability is the same either way.

    With ``fp > 0`` debris can be *un-seen*: a phantom DEBRIS reading of a free cell is
    corrected by later OPEN readings (and, as in round 1, a cell read as debris only by noisy
    readings is never certain).

    The filter assumes a static map. When the map can change (aftershocks bury free cells),
    ``debris_floor > 0`` applies every DEBRIS reading to ``max(P, debris_floor)``: otherwise a
    cell at ``P = 0`` (the robot's own cell, or a cell read OPEN with ``fn = 0``) keeps
    ``P = 0`` under every later DEBRIS reading (``(1 − fn)·0 / (0 + fp)``), so debris an
    aftershock drops there is never seen, only bumped into (and a cell read OPEN many times is
    nearly as deaf). With ``fp = 0`` a DEBRIS reading gives ``P = 1`` whatever the prior, so
    the floor has no effect. ``ExtMission`` sets ``debris_floor = prior_debris`` when
    aftershocks are enabled and ``fp > 0`` (a disturbed cell is no better known than an unseen
    one), else 0; ``debris_floor`` must be in ``[0, threshold)``.

    A *contradiction* is a cell believed passable (``P < threshold``) before the update and
    impassable after it (only DEBRIS readings and touches can do that: an OPEN reading never
    raises ``P``). ``survivor`` (if given) is pre-marked like :class:`~rescue.belief.BeliefMap`
    does: observed, ``P = 0``, a known survivor.

    A ``prior_debris`` at or above ``threshold`` is accepted here but makes every never-observed
    cell impassable (no optimism about unknown cells); :class:`~rescue.ext.mission.ExtMission`
    rejects it because frontier search is then incomplete.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        survivor: Pos | None,
        prior_debris: float,
        false_negative: float,
        threshold: float,
        false_positive: float = 0.0,
        debris_floor: float = 0.0,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"grid must be non-empty, got {rows}x{cols}")
        self.rows: int = rows
        self.cols: int = cols
        self.prior_debris = _check_probability("prior_debris", prior_debris)
        self.false_negative = _check_probability("false_negative", false_negative)
        self.false_positive = _check_false_positive(false_positive)
        self.threshold = float(threshold)
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.debris_floor = float(debris_floor)
        if not 0.0 <= self.debris_floor < self.threshold:
            raise ValueError(f"debris_floor must be in [0, threshold = {self.threshold:g}), "
                             f"got {debris_floor}")
        self.p_debris: np.ndarray = np.full((rows, cols), self.prior_debris, dtype=np.float64)
        self.observed: np.ndarray = np.zeros((rows, cols), dtype=bool)
        self._survivor_mask: np.ndarray = np.zeros((rows, cols), dtype=bool)
        self.survivor: Pos | None = None
        if survivor is not None:
            self._check(survivor)
            self.survivor = (int(survivor[0]), int(survivor[1]))
            self.mark_survivor(self.survivor)

    # --- queries ---------------------------------------------------------------------------
    def _check(self, p: Pos) -> None:
        if not in_bounds(p, self.rows, self.cols):
            raise ValueError(f"position {p} is outside the {self.rows}x{self.cols} grid")

    def get(self, p: Pos) -> Cell:
        """Categorical view: UNKNOWN if never observed, SURVIVOR, DEBRIS if ``P ≥ threshold``,
        else OPEN."""
        self._check(p)
        if self._survivor_mask[p]:
            return Cell.SURVIVOR
        if not self.observed[p]:
            return Cell.UNKNOWN
        return Cell.DEBRIS if self.p_debris[p] >= self.threshold else Cell.OPEN

    def is_known(self, p: Pos) -> bool:
        """True if ``p`` has been observed (or is a pre-marked survivor)."""
        self._check(p)
        return bool(self.observed[p])

    def passable_mask(self) -> np.ndarray:
        """Bool array, True where ``P < threshold`` (unobserved cells keep the prior)."""
        return self.p_debris < self.threshold

    def known_mask(self) -> np.ndarray:
        """Bool array of observed cells (a copy)."""
        return self.observed.copy()

    @property
    def grid(self) -> np.ndarray:
        """The categorical view as an ``int8`` array of :class:`Cell` values (a fresh array)."""
        out = np.full((self.rows, self.cols), int(Cell.UNKNOWN), dtype=np.int8)
        out[self.observed] = int(Cell.OPEN)
        out[self.observed & (self.p_debris >= self.threshold)] = int(Cell.DEBRIS)
        out[self._survivor_mask] = int(Cell.SURVIVOR)
        return out

    def explored_count(self) -> int:
        """Number of observed cells (a pre-marked survivor counts from the start)."""
        return int(np.count_nonzero(self.observed))

    def survivors_seen(self) -> tuple[Pos, ...]:
        """Cells known to hold a survivor, in row-major order."""
        found = np.argwhere(self._survivor_mask)
        return tuple((int(r), int(c)) for r, c in found)

    # --- updates ---------------------------------------------------------------------------
    def mark_survivor(self, p: Pos) -> None:
        """Record a survivor known in advance at ``p`` (observed, ``P = 0``)."""
        self._check(p)
        q = (int(p[0]), int(p[1]))
        self.observed[q] = True
        self.p_debris[q] = _CERTAIN_FREE
        self._survivor_mask[q] = True

    def update(self, percept: Observations, touched: Collection[Pos] = (),
               here: Pos | None = None) -> BeliefUpdate:
        """Apply one percept (and the cells a bump touched, and the robot's cell); report what
        changed.

        ``newly_known`` lists cells observed for the first time (observation order, touched
        cells, then ``here``); ``contradictions`` the cells that went from passable to
        impassable. ``here`` (the robot's cell) is set to ``P = 0`` only when
        ``false_positive > 0`` (see the class docstring). Observing ``UNKNOWN`` or an off-grid
        cell raises ``ValueError``, before anything is written.
        """
        if isinstance(percept, Percept):
            items: Iterable[tuple[Pos, Cell]] = percept.cells
        elif isinstance(percept, Mapping):
            items = percept.items()
        else:
            items = percept
        checked: list[tuple[Pos, Cell]] = []
        for raw_pos, raw_cell in items:
            p = (int(raw_pos[0]), int(raw_pos[1]))
            self._check(p)
            cell = Cell(raw_cell)
            if cell is Cell.UNKNOWN:
                raise ValueError(f"cannot observe UNKNOWN at {p}")
            checked.append((p, cell))
        touched_cells: list[Pos] = []
        for raw in touched:
            p = (int(raw[0]), int(raw[1]))
            self._check(p)
            touched_cells.append(p)
        here_cell: Pos | None = None
        if here is not None and self.false_positive > 0.0:
            here_cell = (int(here[0]), int(here[1]))
            self._check(here_cell)

        eps = self.false_negative
        fp = self.false_positive
        floor = self.debris_floor
        threshold = self.threshold
        newly_known: list[Pos] = []
        contradictions: list[Pos] = []
        seen_new: set[Pos] = set()
        seen_contra: set[Pos] = set()

        def write(p: Pos, value: float, survivor: bool) -> None:
            before = float(self.p_debris[p])
            if not self.observed[p] and p not in seen_new:
                newly_known.append(p)
                seen_new.add(p)
            self.observed[p] = True
            self.p_debris[p] = value
            if survivor:
                self._survivor_mask[p] = True
            if before < threshold <= value and p not in seen_contra:
                contradictions.append(p)
                seen_contra.add(p)

        for p, cell in checked:
            if cell is Cell.SURVIVOR:
                write(p, _CERTAIN_FREE, True)
            else:
                reads_debris = cell is Cell.DEBRIS
                prior = float(self.p_debris[p])
                if reads_debris and prior < floor:
                    prior = floor  # the map may have changed (class docstring)
                write(p, debris_posterior(prior, reads_debris, eps, fp), False)
        for p in touched_cells:
            write(p, _CERTAIN_DEBRIS, False)
        if here_cell is not None:
            write(here_cell, _CERTAIN_FREE, False)
        return BeliefUpdate(tuple(newly_known), tuple(contradictions))

    def copy(self) -> ProbBelief:
        """Independent deep copy."""
        other = ProbBelief(self.rows, self.cols, None, self.prior_debris, self.false_negative,
                           self.threshold, self.false_positive, self.debris_floor)
        other.survivor = self.survivor
        other.p_debris = self.p_debris.copy()
        other.observed = self.observed.copy()
        other._survivor_mask = self._survivor_mask.copy()
        return other

    def __repr__(self) -> str:
        return (
            f"ProbBelief({self.rows}x{self.cols}, false_negative={self.false_negative}, "
            f"false_positive={self.false_positive}, explored={self.explored_count()})"
        )
