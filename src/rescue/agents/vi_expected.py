"""Expected-cost optimal agent (stretch, registry id ``"vi_expected"``).

Same plan → act → observe → replan loop as :class:`~rescue.agents.contingency.ContingencyAgent`
(docs/plan.md §3.10), but each plan is the expected-steps optimal policy found by
:func:`~rescue.planners.value_iteration.value_iteration` on the belief snapshot with the true
outcome probabilities, instead of the first strong-cyclic plan AND-OR search finds. Replanning
is identical: an initial plan (not a replan), then a replan when the outcome was not
anticipated (D10: ``anticipated`` = the support of ``Results`` on the planning snapshot — a
safety net that never fires with sensor radius ≥ 1), the position has no policy entry, or the
sensor contradicted the belief (``eager``: any contradiction; ``lazy``: a contradicted cell in
the plan envelope). The policy is unfolded from the robot's position with
:func:`~rescue.plan.plan_from_policy`, so the UI, snapshots and run files see an ordinary
:class:`~rescue.plan.Plan`; ``plan_size`` counts the states reachable under the policy.

The agent only swaps the planner (``planner`` is a
:class:`~rescue.planners.value_iteration.ValueIterationPlanner`; ``vi_planner`` is the same
object) and records the expected steps after each plan; the replan logic is inherited.
"""

from __future__ import annotations

import math

from rescue.agents.contingency import ContingencyAgent
from rescue.config import AgentConfig, PlannerConfig
from rescue.model import TransitionModel
from rescue.planners.andor import PlanResult
from rescue.planners.value_iteration import ValueIterationPlanner


class ValueIterationAgent(ContingencyAgent):
    """Value-iteration (expected-cost optimal) contingency agent (id ``"vi_expected"``)."""

    name = "vi_expected"

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self._expected_steps = math.inf
        super().__init__(transition, planner_config, agent_config)

    def _make_planner(self) -> ValueIterationPlanner:
        return ValueIterationPlanner(self.planner_config)

    @property
    def vi_planner(self) -> ValueIterationPlanner:
        """The agent's :class:`ValueIterationPlanner` (alias of ``planner``)."""
        assert isinstance(self.planner, ValueIterationPlanner)
        return self.planner

    def _reset_plan(self) -> None:
        super()._reset_plan()
        self._expected_steps = math.inf

    @property
    def expected_steps(self) -> float:
        """Expected steps to the goal under the plan in force, from where it was made.

        ``inf`` before the first plan or when no proper policy exists; ``nan`` when the plan
        in force is the fixpoint reference policy (``fallback_used``, only possible with
        ``PlannerConfig.validate``), whose expected cost value iteration did not compute.
        """
        return self._expected_steps

    def _plan_made(self, result: PlanResult) -> None:
        if result.plan is None:
            self._expected_steps = math.inf
        elif result.stats.fallback_used:
            self._expected_steps = math.nan
        else:
            self._expected_steps = self.vi_planner.last_values.get(self.pos, math.inf)
