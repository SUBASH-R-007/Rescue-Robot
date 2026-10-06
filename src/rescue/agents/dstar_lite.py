"""D* Lite baseline (stretch, registry id ``"dstar_lite"``).

Same trigger set and replan counting as :class:`~rescue.agents.astar_replan.ReplanAStarAgent`
(docs/plan.md §3.10): an initial plan (not a replan), then a replan whenever the outcome
deviates from the expected path cell (an unplanned state, D10), the next path cell is known
debris, or the sensor contradicted the belief (``eager``: any contradiction; ``lazy``: a
contradicted cell on the remaining path). The difference is *how* it replans: instead of a
fresh A* from scratch, one :class:`~rescue.planners.dstar_lite.DStarLite` search lives for the
whole mission. Each replan moves its start to the robot, applies every passability change of
the belief since the previous call (cells that became debris, including those lazy mode did
not act on) and runs ``ComputeShortestPath`` incrementally; ``nodes_expanded`` counts the
vertices that call expanded. The path is then followed cell by cell like the A* agents.
"""

from __future__ import annotations

import logging
import time

from rescue.agents.astar_replan import ReplanAStarAgent
from rescue.agents.base import Abort
from rescue.grid import fmt_pos
from rescue.planners.andor import MS_PER_S
from rescue.planners.dstar_lite import DStarLite

log = logging.getLogger(__name__)


class DStarLiteAgent(ReplanAStarAgent):
    """Incremental-replanning path agent (registry id ``"dstar_lite"``)."""

    name = "dstar_lite"

    def _reset_plan(self) -> None:
        super()._reset_plan()
        self._dstar: DStarLite | None = None

    @property
    def dstar(self) -> DStarLite | None:
        """The mission's D* Lite search (None before the first plan)."""
        return self._dstar

    def _plan_path(self) -> bool:
        """(Re)plan from ``self.pos`` incrementally; True if it was a replan (D12)."""
        t0 = time.perf_counter()
        passable = self.belief.passable_mask()
        if self._dstar is None:
            self._dstar = DStarLite(passable, self.pos, self.goal)
        result = self._dstar.plan(passable, self.pos)
        elapsed_ms = (time.perf_counter() - t0) * MS_PER_S
        path = tuple(result.path) if result.path is not None else None
        self._path = path
        self._index = {p: i for i, p in enumerate(path)} if path is not None else {}
        self._cursor = 0
        if path is None:
            log.info("%s: no path from %s (t=%d)", self.name, fmt_pos(self.pos), self.t)
        return self._record_plan(
            plan=None,
            path=path,
            envelope=frozenset(self._index),
            elapsed_ms=elapsed_ms,
            nodes_expanded=result.nodes_expanded,
            plan_size=len(path) - 1 if path is not None else 0,
        )

    def _unreachable(self) -> Abort:
        return Abort("unreachable", f"D* Lite found no path from {fmt_pos(self.pos)}")
