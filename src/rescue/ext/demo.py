"""``python -m rescue.ext.demo``: one extension mission, printed (docs/plan.md §3.14).

Modes (each sets defaults that ``--k`` / ``--noise`` override):

* ``search`` — one survivor at an unknown location (frontier exploration, brief §8.4);
* ``multi`` — ``EXT_DEMO_MULTI_K`` survivors at known locations, greedy order (brief §8.5);
* ``noisy`` — one known survivor, debris missed with probability ``EXT_DEMO_NOISE`` (§8.3).

Prints a header, the map (every survivor ``P``, 1-indexed rulers), the execution trace (at most
``EXT_DEMO_TRACE_SEGMENTS`` segments) and the metrics table; exits 0 when every survivor was
rescued.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import (
    DEFAULT_SLIP_P,
    EXT_DEMO_MULTI_K,
    EXT_DEMO_NOISE,
    EXT_DEMO_SCENARIO,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
    EXT_DEMO_TRACE_SEGMENTS,
    WorldConfig,
)
from rescue.demo import format_map
from rescue.ext.agents import EXT_AGENT_NAMES, make_ext_agent
from rescue.ext.mission import ExtMetrics, format_ext_trace, run_ext_mission
from rescue.ext.scenarios import multi_survivor_grid
from rescue.grid import fmt_pos
from rescue.mapgen import SCENARIOS
from rescue.runner import make_transition


@dataclass(frozen=True)
class DemoMode:
    """Defaults of one demo mode."""

    description: str
    survivors_known: bool
    k: int
    noise: float


#: The demo's modes.
MODES: dict[str, DemoMode] = {
    "search": DemoMode("survivor location unknown (frontier exploration)", False, 1, 0.0),
    "multi": DemoMode("several known survivors (greedy nearest-first order)", True,
                      EXT_DEMO_MULTI_K, 0.0),
    "noisy": DemoMode("noisy sensing (debris missed with probability eps)", True, 1,
                      EXT_DEMO_NOISE),
}
#: Agent of the demo.
DEFAULT_EXT_DEMO_AGENT = "search_andor"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the extension demo."""
    parser = argparse.ArgumentParser(
        prog="python -m rescue.ext.demo",
        description="Run one extension mission (search / multi / noisy) and print it.",
    )
    parser.add_argument("--mode", choices=sorted(MODES), default="search", help="demo mode")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=EXT_DEMO_SCENARIO,
                        help="map scenario (rescue.mapgen)")
    parser.add_argument("--rows", type=int, default=EXT_DEMO_SIZE[0], help="map rows")
    parser.add_argument("--cols", type=int, default=EXT_DEMO_SIZE[1], help="map columns")
    parser.add_argument("--seed", type=int, default=EXT_DEMO_SEED, help="map and world seed")
    parser.add_argument("--p", type=float, default=DEFAULT_SLIP_P, help="slip probability")
    parser.add_argument("--k", type=int, default=None,
                        help="number of survivors (default: the mode's)")
    parser.add_argument("--noise", type=float, default=None,
                        help="false-negative rate of the debris sensor (default: the mode's)")
    parser.add_argument("--agent", choices=EXT_AGENT_NAMES, default=DEFAULT_EXT_DEMO_AGENT,
                        help="extension agent")
    return parser


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:g}" if value.is_integer() else f"{value:.2f}"
    return str(value)


def format_ext_metrics(metrics: ExtMetrics) -> str:
    """All metrics as aligned ``key: value`` lines."""
    items = metrics.as_dict()
    width = max(len(key) for key in items)
    return "\n".join(f"{key + ':':<{width + 1}} {_format_value(v)}" for key, v in items.items())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo and print it; returns the exit code (0 when every survivor was rescued)."""
    ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = MODES[args.mode]
    k = mode.k if args.k is None else args.k
    noise = mode.noise if args.noise is None else args.noise
    if k < 1:
        parser.error(f"--k must be >= 1, got {k}")
    if not 0.0 <= noise <= 1.0:
        parser.error(f"--noise must be in [0, 1], got {noise}")
    try:
        grid = multi_survivor_grid(args.scenario, args.rows, args.cols, args.seed, k)
        world_config = WorldConfig(slip_p=args.p)
    except ValueError as exc:
        parser.error(str(exc))
    agent = make_ext_agent(args.agent, make_transition(grid, world_config))
    mission = run_ext_mission(
        grid, agent, world_config, args.seed,
        survivors_known=mode.survivors_known, false_negative=noise,
        scenario=args.scenario, mode=args.mode,
    )
    metrics = mission.metrics()

    known = "known" if mode.survivors_known else "unknown"
    print(f"Rescue Robot — extension demo: {args.mode} ({mode.description})")
    print(f"map {args.scenario} {grid.rows}x{grid.cols}, seed {args.seed}, p = {args.p:g}, "
          f"k = {k} ({known} locations), noise eps = {noise:g}, agent {agent.name}")
    print()
    print(format_map(grid))
    print()
    survivors = ", ".join(fmt_pos(s) for s in grid.survivors)
    print(f"Start {fmt_pos(grid.start)}; survivors at {survivors}")
    print()
    print("Execution trace:")
    print(format_ext_trace(mission.events, max_segments=EXT_DEMO_TRACE_SEGMENTS))
    print()
    print(format_ext_metrics(metrics))
    return 0 if metrics.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
