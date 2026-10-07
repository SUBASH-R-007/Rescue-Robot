"""Expected-cost optimal agent planned by LAO* (stretch, registry id ``"lao_expected"``).

Brief §8.1, docs/plan.md §3.15.1. Identical to
:class:`~rescue.agents.vi_expected.ValueIterationAgent` (the plan → act → observe → replan
loop of :class:`~rescue.agents.contingency.ContingencyAgent`, the same replan triggers and D10
anticipation, so 0 unplanned states with sensor radius ≥ 1) except for the planner:
:class:`~rescue.planners.lao_star.LaoStarPlanner` computes the same expected-steps optimal
policy by heuristic search from the robot's position, expanding only the states the best
policy can reach instead of sweeping every free cell. ``nodes_expanded`` counts the states
LAO* expanded; ``expected_steps`` is the LAO* value at the planning position.
"""

from __future__ import annotations

import math

from rescue.agents.contingency import ContingencyAgent
from rescue.config import AgentConfig, PlannerConfig
from rescue.model import TransitionModel
from rescue.planners.andor import PlanResult
from rescue.planners.lao_star import LaoStarPlanner


class LaoExpectedAgent(ContingencyAgent):
    """LAO* (expected-cost optimal) contingency agent (id ``"lao_expected"``)."""

    name = "lao_expected"

    def __init__(
        self,
        transition: TransitionModel,
        planner_config: PlannerConfig | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self._expected_steps = math.inf
        super().__init__(transition, planner_config, agent_config)

    def _make_planner(self) -> LaoStarPlanner:
        return LaoStarPlanner(self.planner_config)

    @property
    def lao_planner(self) -> LaoStarPlanner:
        """The agent's :class:`LaoStarPlanner` (alias of ``planner``)."""
        assert isinstance(self.planner, LaoStarPlanner)
        return self.planner

    def _reset_plan(self) -> None:
        super()._reset_plan()
        self._expected_steps = math.inf

    @property
    def expected_steps(self) -> float:
        """Expected steps to the goal under the plan in force, from where it was made.

        ``inf`` before the first plan or when no proper policy exists; ``nan`` when the plan
        in force is the fixpoint reference policy (``fallback_used``, only possible with
        ``PlannerConfig.validate``), whose expected cost LAO* did not compute.
        """
        return self._expected_steps

    def _plan_made(self, result: PlanResult) -> None:
        if result.plan is None:
            self._expected_steps = math.inf
        elif result.stats.fallback_used:
            self._expected_steps = math.nan
        else:
            self._expected_steps = self.lao_planner.last_values.get(self.pos, math.inf)
