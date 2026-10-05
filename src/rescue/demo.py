"""``python -m rescue.demo``: the deck's slide 9 sample output.

Runs one mission on ``maps/deck_demo.txt`` with the recorded ``config.DEMO_SEED`` and prints a
header, the map (1-indexed rulers), the initial root plan, the execution trace and the summary.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rescue.agents import AGENTS, make_agent
from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import DEFAULT_SLIP_P, DEMO_SEED, AgentConfig, PlannerConfig, WorldConfig
from rescue.grid import TrueGrid, fmt_pos
from rescue.runner import Mission, format_summary, format_trace, make_transition, run_mission
from rescue.runs import DEFAULT_RUNS_DIR, save_run

log = logging.getLogger(__name__)

#: Map used by the demo (relative to the working directory or the repository root).
DEFAULT_DEMO_MAP = "maps/deck_demo.txt"
#: Agent used by the demo.
DEFAULT_DEMO_AGENT = "andor"
#: Repository root (src/rescue/demo.py → three levels up), for resolving relative map paths.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_map_path(path: str | Path) -> Path:
    """``path`` if it exists, else the same relative path under the repository root."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    candidate = _REPO_ROOT / p
    return candidate if candidate.exists() else p


def format_map(grid: TrueGrid) -> str:
    """The map in ASCII with 1-indexed row and column rulers."""
    width = len(str(grid.cols))
    row_width = len(str(grid.rows))
    pad = " " * (row_width + 2)
    header = pad + " ".join(f"{c:>{width}}" for c in range(1, grid.cols + 1))
    lines = [header]
    for r, row in enumerate(grid.to_ascii().splitlines(), start=1):
        cells = " ".join(f"{ch:>{width}}" for ch in row)
        lines.append(f"{r:>{row_width}}  {cells}")
    return "\n".join(lines)


def initial_plan_text(mission: Mission) -> str:
    """Root plan of the first snapshot (``Plan.pretty()``), or the A* path for A* agents."""
    snapshots = mission.agent.snapshots
    if not snapshots:
        return "(no plan)"
    first = snapshots[0]
    if first.plan is not None:
        return first.plan.pretty()
    if first.path is not None:
        return " → ".join(fmt_pos(p) for p in first.path)
    return "(no plan)"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the demo."""
    parser = argparse.ArgumentParser(
        prog="python -m rescue.demo",
        description="Print the AND-OR contingency plan demo (deck slide 9).",
    )
    parser.add_argument("--map", default=DEFAULT_DEMO_MAP, help="ASCII map file")
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help="world seed")
    parser.add_argument("--p", type=float, default=DEFAULT_SLIP_P, help="slip probability")
    parser.add_argument(
        "--agent", default=DEFAULT_DEMO_AGENT, choices=sorted(AGENTS), help="agent to run"
    )
    parser.add_argument(
        "--lazy", action="store_true", help="lazy replanning (only on plan-envelope contradictions)"
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=f"write a run file (default name under {DEFAULT_RUNS_DIR}/)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo and print it; returns the process exit code (0 on rescue)."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    grid = TrueGrid.load(resolve_map_path(args.map))
    world_config = WorldConfig(slip_p=args.p)
    agent_config = AgentConfig(replan_mode="lazy" if args.lazy else "eager")
    agent = make_agent(args.agent, make_transition(grid, world_config), PlannerConfig(),
                       agent_config)
    mission = run_mission(grid, agent, world_config, seed=args.seed, scenario=grid.name)
    metrics = mission.metrics()

    print(f"Rescue Robot — AND-OR contingency plan demo "
          f"(map {grid.name}, seed {args.seed}, p = {args.p:g})")
    print()
    print(format_map(grid))
    print()
    print(f"Initial plan from {fmt_pos(grid.start)}: {initial_plan_text(mission)}")
    print()
    print("Execution trace:")
    print(format_trace(mission.events))
    print()
    print(format_summary(metrics))
    if args.save is not None:
        path = save_run(mission, args.save or None)
        print()
        print(f"Run saved to {path}")
    return 0 if metrics.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
