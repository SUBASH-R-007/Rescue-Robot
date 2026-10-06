"""``python -m rescue.sim``: the interactive Pygame simulator (brief §6.1).

Examples::

    python -m rescue.sim --map maps/deck_demo.txt
    python -m rescue.sim --scenario collapsed_building --rows 20 --cols 20 --seed 7
    python -m rescue.sim --headless --frames 100 --screenshot recordings/out.png

Without ``--map`` / ``--scenario`` the simulator opens ``maps/deck_demo.txt`` with
``config.DEMO_SEED``. Press ``H`` in the window for the controls.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

from rescue.agents import AGENT_NAMES
from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import (
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
    DEMO_SEED,
)

log = logging.getLogger(__name__)

# pygame prints a banner on import unless this is set (rescue.viz imports pygame).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the simulator."""
    from rescue.viz import theme

    rows, cols = theme.GENERATED_MAP_SIZE
    parser = argparse.ArgumentParser(
        prog="python -m rescue.sim",
        description="Interactive rescue-robot simulator (AND-OR contingency planning).",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--map", metavar="PATH", help="ASCII map file (default maps/deck_demo.txt)")
    where.add_argument("--scenario", metavar="NAME",
                       help="map name in maps/ or a rescue.mapgen scenario")
    parser.add_argument("--rows", type=int, default=rows, help="rows of generated scenarios")
    parser.add_argument("--cols", type=int, default=cols, help="columns of generated scenarios")
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help="world (and map) seed")
    parser.add_argument("--agent", default=AGENT_NAMES[0], choices=AGENT_NAMES, help="agent")
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
    parser.add_argument("--speed", type=int, default=theme.DEFAULT_SPEED_INDEX,
                        metavar="INDEX",
                        help=f"initial speed index into {list(theme.SPEEDS)} steps/s")
    parser.add_argument("--save-run", action="store_true",
                        help="write runs/<timestamp>_<agent>_<seed>.json when a mission ends")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the simulator and return the exit code."""
    ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    from rescue.viz.sim_app import SimApp, SimOptions, prepare_environment

    prepare_environment(args.headless)
    options = SimOptions(
        map_path=args.map,
        scenario=args.scenario,
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
        agent=args.agent,
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
        save_run=args.save_run,
    )
    try:
        app = SimApp(options)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    code = app.run()
    for path in app.saved_runs:
        print(f"Run saved to {path}")
    for path in app.recordings:
        print(f"Recording saved to {path}")
    if args.screenshot and args.headless:
        print(f"Screenshot saved to {args.screenshot}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
