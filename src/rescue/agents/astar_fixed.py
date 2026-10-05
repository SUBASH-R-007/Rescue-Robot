"""Fixed A* baseline: plan once, execute open-loop, abort on the first surprise.

docs/plan.md §3.10. One A* path on the optimistic belief at t = 0. ``choose_action`` aborts
when the robot is not on the expected path cell (the first deviation — slip or skid — which is
also counted as one unplanned state, D10) or when the next path cell is known debris.
"""

from __future__ import annotations

from rescue.agents.base import (
    NO_REPLAN,
    Abort,
    Decision,
    Observation,
    PathAgent,
    ReplanInfo,
    direction,
)
from rescue.belief import BeliefUpdate
from rescue.grid import fmt_pos


class FixedAStarAgent(PathAgent):
    """Open-loop A* agent (registry id ``"astar_fixed"``)."""

    name = "astar_fixed"

    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        if not self.snapshots:
            if self.pos != self.goal:
                self._plan_path()  # the one and only plan (not a replan)
            return NO_REPLAN
        self._advance()
        return NO_REPLAN

    def choose_action(self, obs: Observation) -> Decision:
        """Next path step, or ``Abort`` on deviation / known debris ahead / no path."""
        pos = self._check_obs(obs)
        if self._path is None:
            return self._unreachable()
        expected = self._expected_cell()
        if expected != pos:
            where = "the end of the path" if expected is None else fmt_pos(expected)
            return Abort("aborted", f"deviated from the path: at {fmt_pos(pos)}, expected {where}")
        nxt = self._next_cell()
        if nxt is None:  # pragma: no cover - the last path cell is the goal (_check_obs raises)
            return Abort("aborted", f"path ended at {fmt_pos(pos)} without reaching the goal")
        if self._next_is_known_debris():
            return Abort("aborted", f"next path cell {fmt_pos(nxt)} is debris")
        return self._begin_action(pos, direction(pos, nxt))
