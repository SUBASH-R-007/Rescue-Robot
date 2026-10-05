"""Agents: the AND-OR contingency agent and the two A* baselines (docs/plan.md §3.8)."""

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
from rescue.config import AgentConfig, PlannerConfig
from rescue.model import TransitionModel

#: Agent ids in display order (comparison panes, tables, plots).
AGENT_NAMES: tuple[str, ...] = (
    ContingencyAgent.name,
    FixedAStarAgent.name,
    ReplanAStarAgent.name,
)

#: Registry: agent id → agent class.
AGENTS: dict[str, type[Agent]] = {
    cls.name: cls for cls in (ContingencyAgent, FixedAStarAgent, ReplanAStarAgent)
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
        raise ValueError(f"unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}") from None
    return cls(transition, planner_config, agent_config)


__all__ = [
    "AGENTS",
    "AGENT_NAMES",
    "Abort",
    "Agent",
    "AgentStats",
    "AgentUpdate",
    "ContingencyAgent",
    "Decision",
    "FixedAStarAgent",
    "Observation",
    "PathAgent",
    "PlanSnapshot",
    "ReplanAStarAgent",
    "make_agent",
]
