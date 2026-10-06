"""``python -m rescue.compare``: head-to-head comparison of agents (brief §6.2).

Examples::

    python -m rescue.compare --scenario collapsed_building --seed 7
    python -m rescue.compare --map maps/building_01.txt --agents andor,astar_replan
    python -m rescue.compare --headless --frames 150 --screenshot out.png

Every pane runs its own mission on the same true map with the same world seed (common random
numbers) and all missions advance in lock-step. Press ``H`` in the window for the controls.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES
from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import (
    COMPARE_DEFAULT_SCENARIO,
    COMPARE_DEFAULT_SEED,
    COMPARE_DEFAULT_SIZE,
    COMPARE_DEFAULT_SPEED_INDEX,
    COMPARE_MAX_AGENTS,
    COMPARE_MIN_AGENTS,
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
)

log = logging.getLogger(__name__)

# pygame prints a banner on import unless this is set (rescue.viz imports pygame).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the comparison app."""
    from rescue import mapgen
    from rescue.viz import theme

    rows, cols = COMPARE_DEFAULT_SIZE
    parser = argparse.ArgumentParser(
        prog="python -m rescue.compare",
        description="Head-to-head comparison: same map, same seed, lock-step panes.",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--scenario", metavar="NAME", default=COMPARE_DEFAULT_SCENARIO,
                       choices=tuple(mapgen.SCENARIOS),
                       help=f"rescue.mapgen scenario (default {COMPARE_DEFAULT_SCENARIO}; "
                            f"one of {', '.join(mapgen.SCENARIOS)})")
    where.add_argument("--map", metavar="PATH", help="ASCII map file instead of a scenario")
    parser.add_argument("--rows", type=int, default=rows, help="rows of generated scenarios")
    parser.add_argument("--cols", type=int, default=cols, help="columns of generated scenarios")
    parser.add_argument("--seed", type=int, default=COMPARE_DEFAULT_SEED,
                        help="world (and map) seed shared by every pane")
    parser.add_argument("--agents", metavar="A,B,...", default=",".join(AGENT_NAMES),
                        help=f"{COMPARE_MIN_AGENTS}-{COMPARE_MAX_AGENTS} comma-separated agents "
                             f"of {', '.join(ALL_AGENT_NAMES)} (default "
                             f"{','.join(AGENT_NAMES)})")
    parser.add_argument("--p", type=float, default=DEFAULT_SLIP_P, help="slip probability")
    parser.add_argument("--radius", type=int, default=DEFAULT_SENSOR_RADIUS,
                        help="sensor radius (Manhattan)")
    parser.add_argument("--lazy", action="store_true",
                        help="lazy replanning (only on plan-envelope contradictions)")
    parser.add_argument("--aftershock-q", type=float, default=0.0,
                        help="aftershock probability per step")
    parser.add_argument("--aftershock-k", type=int, default=1,
                        help="debris cells per aftershock")
    parser.add_argument("--high-contrast", action="store_true", help="high-contrast palette")
    parser.add_argument("--headless", action="store_true",
                        help="no window (SDL dummy driver); renders --frames frames")
    parser.add_argument("--frames", type=int, default=theme.HEADLESS_FRAMES,
                        help="frames to render in headless mode")
    parser.add_argument("--screenshot", metavar="PATH", help="save the last frame (headless)")
    parser.add_argument("--record", metavar="PATH", help="record a GIF / MP4 from the start")
    parser.add_argument("--mp4", action="store_true", help="G records MP4 instead of GIF")
    parser.add_argument("--autoplay", action="store_true",
                        help="start playing immediately (always on in headless mode)")
    parser.add_argument("--speed", type=int, default=COMPARE_DEFAULT_SPEED_INDEX,
                        metavar="INDEX",
                        help=f"initial speed index into {list(theme.SPEEDS)} ticks/s")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the comparison app and return the exit code."""
    ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    from rescue.viz.compare_app import CompareApp, CompareOptions, parse_agents
    from rescue.viz.sim_app import prepare_environment

    try:
        agents = parse_agents(args.agents)
    except ValueError as exc:
        parser.error(str(exc))
    prepare_environment(args.headless)
    options = CompareOptions(
        map_path=args.map,
        scenario=args.scenario,
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
        agents=agents,
        slip_p=args.p,
        sensor_radius=args.radius,
        replan_mode="lazy" if args.lazy else "eager",
        aftershock_q=args.aftershock_q,
        aftershock_k=args.aftershock_k,
        high_contrast=args.high_contrast,
        headless=args.headless,
        frames=args.frames,
        screenshot=args.screenshot,
        record=args.record,
        autoplay=True if args.autoplay else None,
        mp4=args.mp4,
        speed_index=args.speed,
    )
    try:
        app = CompareApp(options)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    code = app.run()
    if args.headless:
        for pane in app.panes:
            m = pane.mission.metrics()
            print(f"{pane.agent:>13}: {pane.status.name:<11} t = {m.steps:<4} "
                  f"replans {m.replans:<3} slips handled {m.slips_handled:<3} "
                  f"unplanned {m.unplanned_states}")
    for path in app.recordings:
        print(f"Recording saved to {path}")
    if args.screenshot and args.headless:
        print(f"Screenshot saved to {args.screenshot}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
