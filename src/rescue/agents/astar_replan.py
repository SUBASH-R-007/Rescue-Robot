"""A* + replan baseline: re-run A* whenever reality departs from the path.

docs/plan.md §3.10. Replans from the current position when the outcome deviates from the
expected cell (counted as an unplanned state, D10), the next path cell is known debris, or the
sensor contradicted the belief (``eager``: any contradiction; ``lazy``: a contradicted cell on
the remaining path — those are the update's ``triggering_contradictions``). Replans after the
initial plan are counted (D12). All replanning happens in ``on_outcome``.
"""

from __future__ import annotations

from rescue.agents.base import NO_REPLAN, Decision, Observation, PathAgent, ReplanInfo, direction
from rescue.belief import BeliefUpdate
from rescue.grid import Cell, fmt_pos


class ReplanAStarAgent(PathAgent):
    """A* agent that replans on deviation or contradiction (registry id ``"astar_replan"``)."""

    name = "astar_replan"

    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        if not self.snapshots:
            if self.pos != self.goal:
                self._plan_path()  # initial plan: not a replan (D12)
            return NO_REPLAN
        self._advance()
        if self.pos == self.goal:
            return NO_REPLAN
        # Judged against the remaining path of the plan in force, before any replan.
        triggers = self._relevant_contradictions(update.contradictions, self._remaining_path())
        needs_replan = (
            unplanned
            or self._path is None
            or self._expected_cell() != self.pos
            or self._next_is_known_debris()
            or bool(triggers)
        )
        if not needs_replan:
            return NO_REPLAN
        return ReplanInfo(replanned=self._plan_path(), triggers=triggers)

    def choose_action(self, obs: Observation) -> Decision:
        """Next path step; ``Abort("unreachable")`` if A* found no path.

        Never plans: ``on_outcome`` replans whenever the path is stale (off the expected cell,
        at its end or blocked by known debris), so a stale path here is a logic error
        (``RuntimeError``).
        """
        pos = self._check_obs(obs)
        if self._path is None:
            return self._unreachable()
        nxt = self._next_cell()
        if self._expected_cell() != pos or nxt is None or self.belief.get(nxt) is Cell.DEBRIS:
            raise RuntimeError(
                f"astar_replan: stale path at {fmt_pos(pos)} (t={self.t}); on_outcome() should "
                "have replanned"
            )
        return self._begin_action(pos, direction(pos, nxt))
