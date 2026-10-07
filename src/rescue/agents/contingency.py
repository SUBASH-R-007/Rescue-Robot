"""The contingency agent: AND-OR cyclic plans, replanning only when the map surprises it.

docs/plan.md §3.10. The agent plans with :class:`AndOrPlanner` on
``ProblemModel.from_belief(belief, transition)`` snapshotted at planning time, then follows the
branch that matches each observed outcome (a policy lookup). It replans from the current
position when there is no plan yet (the initial plan, not counted as a replan), the outcome was
not anticipated by the snapshot model (D10 — a bug with sensor radius ≥ 1), the position has no
policy entry, or the sensor contradicted the belief (``eager``: any contradiction; ``lazy``:
only a contradicted cell in the plan's envelope). Those contradictions are reported as the
update's ``triggering_contradictions``.
"""

from __future__ import annotations

import logging

from rescue.agents.base import NO_REPLAN, Abort, Agent, Decision, Observation, ReplanInfo
from rescue.belief import BeliefUpdate
from rescue.config import AgentConfig, PlannerConfig
from rescue.grid import Pos, fmt_pos
from rescue.model import Action, ProblemModel, TransitionModel
from rescue.plan import Plan
from rescue.planners.andor import AndOrPlanner, PlanResult
from rescue.planners.lao_star import LaoStarPlanner
from rescue.planners.value_iteration import ValueIterationPlanner

#: Planners a contingency agent can use: all map ``plan(model, start)`` to a ``PlanResult``.
Planner = AndOrPlanner | ValueIterationPlanner | LaoStarPlanner

log = logging.getLogger(__name__)


class ContingencyAgent(Agent):
    """AND-OR contingency agent (registry id ``"andor"``)."""

    name = "andor"

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        super().__init__(transition, planner_config, agent_config)
        self.planner = self._make_planner()

    def _make_planner(self) -> Planner:
        """The planner of this agent (subclasses swap in another ``plan(model, start)``)."""
        return AndOrPlanner(self.planner_config)

    # --- plan state ------------------------------------------------------------------------
    def _reset_plan(self) -> None:
        self._model: ProblemModel | None = None
        self._plan: Plan | None = None
        self._policy: dict[Pos, Action] = {}
        self._envelope: frozenset[Pos] = frozenset()

    @property
    def plan(self) -> Plan | None:
        """The plan in force (None before the first plan or after a failed one)."""
        return self._plan

    @property
    def model(self) -> ProblemModel | None:
        """The planning model the current plan was made on."""
        return self._model

    def _plan_made(self, result: PlanResult) -> None:
        """Hook called after every planning call, once the plan is in force (no-op here)."""

    def _replan(self) -> bool:
        """Plan from ``self.pos`` on a fresh belief snapshot; True if it was a replan."""
        model = ProblemModel.from_belief(self.belief, self.transition)
        result = self.planner.plan(model, self.pos)
        plan = result.plan
        self._model = model
        self._plan = plan
        if plan is None:
            self._policy = {}
            self._envelope = frozenset()
            log.info("%s: no strong-cyclic plan from %s (t=%d)", self.name, fmt_pos(self.pos),
                     self.t)
        else:
            # One shallow copy and one tree walk per plan; per-step lookups use these caches.
            self._policy = plan.to_policy()
            self._envelope = plan.envelope()
        self._plan_made(result)
        stats = result.stats
        return self._record_plan(
            plan=plan,
            path=None,
            envelope=self._envelope,
            elapsed_ms=stats.time_ms,
            nodes_expanded=stats.nodes_expanded,
            plan_size=stats.plan_size,
            fallback_used=stats.fallback_used,
        )

    # --- agent interface -------------------------------------------------------------------
    def _after_outcome(self, update: BeliefUpdate, unplanned: bool) -> ReplanInfo:
        pos = self.pos
        if pos == self.goal:
            return NO_REPLAN
        if not self.snapshots:
            self._replan()  # initial plan: not a replan (D12)
            return NO_REPLAN
        # Judged against the envelope of the plan in force, before any replan.
        triggers = self._relevant_contradictions(update.contradictions, self._envelope)
        if not (unplanned or pos not in self._policy or triggers):
            return NO_REPLAN
        return ReplanInfo(replanned=self._replan(), triggers=triggers)

    def choose_action(self, obs: Observation) -> Decision:
        """Follow the plan's branch for ``obs.pos``; ``Abort("unreachable")`` without a plan.

        Never plans: ``on_outcome`` replans whenever the position has no policy entry, so a
        missing entry here is a logic error (``RuntimeError``).
        """
        pos = self._check_obs(obs)
        if self._plan is None:
            return Abort("unreachable", f"no strong-cyclic plan from {fmt_pos(pos)}")
        action = self._policy.get(pos)
        if action is None:
            raise RuntimeError(
                f"andor: no policy entry at {fmt_pos(pos)} (t={self.t}); on_outcome() should "
                "have replanned"
            )
        return self._begin_action(pos, action)

    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]:
        """``Results(pos, action)`` on the snapshot model of the current plan (D10)."""
        if self._model is None:
            return frozenset()
        return frozenset(o.state for o in self._model.results(pos, action))

    def policy_arrows(self) -> dict[Pos, Action]:
        """The current plan's policy (a fresh dict)."""
        return dict(self._policy)
