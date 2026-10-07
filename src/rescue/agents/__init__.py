"""Agents: the AND-OR contingency agent, the two A* baselines and the stretch agents.

docs/plan.md §3.8. ``AGENT_NAMES`` are the three agents the UI and the head-to-head comparison
show; ``STRETCH_AGENT_NAMES`` are the stretch agents (brief §5: ``dstar_lite``,
``vi_expected``; brief §8.1: ``lao_expected``); ``make_agent`` builds any of
``ALL_AGENT_NAMES``.
"""

from __future__ import annotations

from rescue.agents.astar_fixed import FixedAStarAgent
from rescue.agents.astar_replan import ReplanAStarAgent
from rescue.agents.base import (
    Abort,
    Agent,
    AgentStats,
    AgentUpdate,
    Decision,
    Observation,
    PathAgent,
    PlanSnapshot,
)
from rescue.agents.contingency import ContingencyAgent
from rescue.agents.dstar_lite import DStarLiteAgent
from rescue.agents.lao_expected import LaoExpectedAgent
from rescue.agents.vi_expected import ValueIterationAgent
from rescue.config import AgentConfig, PlannerConfig
from rescue.model import TransitionModel

#: Agent ids in display order (comparison panes, tables, plots).
AGENT_NAMES: tuple[str, ...] = (
    ContingencyAgent.name,
    FixedAStarAgent.name,
    ReplanAStarAgent.name,
)

#: Stretch agents (brief §5 and §8.1), not part of the UI's agent cycle.
STRETCH_AGENT_NAMES: tuple[str, ...] = (
    DStarLiteAgent.name,
    ValueIterationAgent.name,
    LaoExpectedAgent.name,
)

#: Every agent ``make_agent`` can build, in display order.
ALL_AGENT_NAMES: tuple[str, ...] = AGENT_NAMES + STRETCH_AGENT_NAMES

#: Registry: agent id → agent class.
AGENTS: dict[str, type[Agent]] = {
    cls.name: cls
    for cls in (
        ContingencyAgent,
        FixedAStarAgent,
        ReplanAStarAgent,
        DStarLiteAgent,
        ValueIterationAgent,
        LaoExpectedAgent,
    )
}


def make_agent(
    name: str,
    transition: TransitionModel,
    planner_config: PlannerConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> Agent:
    """Instantiate the agent registered as ``name`` (ValueError for unknown names)."""
    try:
        cls = AGENTS[name]
    except KeyError:
        raise ValueError(
            f"unknown agent {name!r}; choose from {', '.join(ALL_AGENT_NAMES)}"
        ) from None
    return cls(transition, planner_config, agent_config)


__all__ = [
    "AGENTS",
    "AGENT_NAMES",
    "ALL_AGENT_NAMES",
    "STRETCH_AGENT_NAMES",
    "Abort",
    "Agent",
    "AgentStats",
    "AgentUpdate",
    "ContingencyAgent",
    "DStarLiteAgent",
    "Decision",
    "FixedAStarAgent",
    "LaoExpectedAgent",
    "Observation",
    "PathAgent",
    "PlanSnapshot",
    "ReplanAStarAgent",
    "ValueIterationAgent",
    "make_agent",
]
