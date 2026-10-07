"""Optional extensions (brief §8.3–§8.5, docs/plan.md §3.14).

* :mod:`rescue.ext.sensing` — noisy (false-negative) debris sensing and a Bayes belief map;
* :mod:`rescue.ext.scenarios` — core scenario maps with several survivors;
* :mod:`rescue.ext.agents` — search-and-rescue agents (frontier exploration, greedy rescue
  order) planning with AND-OR or A*;
* :mod:`rescue.ext.mission` — the extension mission loop, events, metrics and trace;
* :mod:`rescue.ext.demo` — ``python -m rescue.ext.demo --mode search|multi|noisy``.

The core modules use none of this; with default arguments they behave exactly as without the
extensions (plan §3.14, rule zero).
"""

from __future__ import annotations

from rescue.ext.agents import (
    EXT_AGENT_NAMES,
    EXT_AGENTS,
    SearchAgent,
    SearchAndOrAgent,
    SearchAStarAgent,
    make_ext_agent,
)
from rescue.ext.mission import (
    ExtMetrics,
    ExtMission,
    ExtStepEvent,
    format_ext_trace,
    oracle_tours,
    run_ext_mission,
)
from rescue.ext.scenarios import add_survivors, multi_survivor_grid
from rescue.ext.sensing import NoisySensor, ProbBelief

__all__ = [
    "EXT_AGENTS",
    "EXT_AGENT_NAMES",
    "ExtMetrics",
    "ExtMission",
    "ExtStepEvent",
    "NoisySensor",
    "ProbBelief",
    "SearchAStarAgent",
    "SearchAgent",
    "SearchAndOrAgent",
    "add_survivors",
    "format_ext_trace",
    "make_ext_agent",
    "multi_survivor_grid",
    "oracle_tours",
    "run_ext_mission",
]
