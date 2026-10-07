"""``python -m rescue.sim``: the interactive Pygame simulator (brief §6.1).

Examples::

    python -m rescue.sim --map maps/deck_demo.txt
    python -m rescue.sim --scenario collapsed_building --rows 20 --cols 20 --seed 7
    python -m rescue.sim --headless --frames 100 --screenshot recordings/out.png
    python -m rescue.sim --ext search            # extensions (plan §3.15.3): search | multi | noisy
    python -m rescue.sim --ext multi --k 4 --agent search_astar

Without ``--map`` / ``--scenario`` the simulator opens ``maps/deck_demo.txt`` with
``config.DEMO_SEED``. With ``--ext`` it runs the extension missions of ``rescue.ext`` instead
(:class:`~rescue.viz.ext_sim_app.ExtSimApp`); its defaults are those of
``python -m rescue.ext.demo`` (``EXT_DEMO_SCENARIO``, ``EXT_DEMO_SIZE``, ``EXT_DEMO_SEED``,
``search_andor``). Without ``--ext`` nothing changes. Press ``H`` in the window for the controls.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rescue.agents import AGENT_NAMES
from rescue.cli_utils import ensure_utf8_stdout
from rescue.config import (
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
    DEMO_SEED,
    EXT_DEMO_MULTI_K,
    EXT_DEMO_NOISE,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
)

if TYPE_CHECKING:
    from rescue.viz.ext_sim_app import ExtSimOptions
    from rescue.viz.sim_app import SimApp

#: Agents of ``--ext`` (= ``rescue.ext.agents.EXT_AGENT_NAMES``, tested). Spelled out so that
#: the core simulator never imports ``rescue.ext`` (the browser build ships without it).
EXT_AGENT_CHOICES: tuple[str, ...] = ("search_andor", "search_astar")
#: Modes of ``--ext`` (``rescue.ext.demo.MODES``; the ``X`` key cycles them in this order).
EXT_MODE_CHOICES: tuple[str, ...] = ("search", "multi", "noisy")
#: Flags that only make sense with ``--ext``.
EXT_ONLY_FLAGS: tuple[str, ...] = ("k", "noise", "fp")

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
    parser.add_argument("--rows", type=int, default=None,
                        help=f"rows of generated scenarios (default {rows}; "
                             f"--ext: {EXT_DEMO_SIZE[0]})")
    parser.add_argument("--cols", type=int, default=None,
                        help=f"columns of generated scenarios (default {cols}; "
                             f"--ext: {EXT_DEMO_SIZE[1]})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"world (and map) seed (default {DEMO_SEED}; --ext: {EXT_DEMO_SEED})")
    parser.add_argument("--agent", default=None, choices=(*AGENT_NAMES, *EXT_AGENT_CHOICES),
                        help=f"agent (default {AGENT_NAMES[0]}; --ext: {EXT_AGENT_CHOICES[0]}, "
                             f"or {' / '.join(EXT_AGENT_CHOICES[1:])})")
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
    ext = parser.add_argument_group("extensions (brief §8.3–§8.5, docs/plan.md §3.15.3)")
    ext.add_argument("--ext", choices=EXT_MODE_CHOICES, default=None,
                     help="run the extension missions: survivor location unknown (search), "
                          "k known survivors (multi) or noisy sensing (noisy); X cycles them")
    ext.add_argument("--k", type=int, default=None,
                     help=f"number of survivors (default: the mode's; multi: {EXT_DEMO_MULTI_K})")
    ext.add_argument("--noise", type=float, default=None, metavar="EPS",
                     help=f"false-negative rate of the debris sensor (default: noisy "
                          f"{EXT_DEMO_NOISE:g}, else 0)")
    ext.add_argument("--fp", type=float, default=None, metavar="EPS",
                     help="false-positive rate (only if this build's sensor supports it)")
    return parser


def _ext_options(args: argparse.Namespace) -> ExtSimOptions:
    """The :class:`~rescue.viz.ext_sim_app.ExtSimOptions` of ``--ext`` arguments."""
    from rescue.viz.ext_sim_app import ExtSimOptions

    defaults = ExtSimOptions()
    return ExtSimOptions(
        map_path=args.map,
        scenario=args.scenario if args.scenario is not None else defaults.scenario,
        rows=defaults.rows if args.rows is None else args.rows,
        cols=defaults.cols if args.cols is None else args.cols,
        seed=defaults.seed if args.seed is None else args.seed,
        agent=defaults.agent if args.agent is None else args.agent,
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
        mode=args.ext,
        k=args.k,
        noise=args.noise,
        false_positive=args.fp,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the simulator and return the exit code."""
    ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    from rescue.viz import theme
    from rescue.viz.sim_app import SimApp, SimOptions, prepare_environment

    if args.ext is None:
        given = [f"--{name}" for name in EXT_ONLY_FLAGS if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join(given)} need(s) --ext")
    elif args.save_run:
        parser.error("--save-run is not available with --ext")
    prepare_environment(args.headless)
    if args.ext is not None:
        from rescue.viz.ext_sim_app import ExtSimApp

        try:
            ext_app = ExtSimApp(_ext_options(args))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return _finish(ext_app, args)
    rows, cols = theme.GENERATED_MAP_SIZE
    options = SimOptions(
        map_path=args.map,
        scenario=args.scenario,
        rows=rows if args.rows is None else args.rows,
        cols=cols if args.cols is None else args.cols,
        seed=DEMO_SEED if args.seed is None else args.seed,
        agent=AGENT_NAMES[0] if args.agent is None else args.agent,
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
    return _finish(app, args)


def _finish(app: SimApp, args: argparse.Namespace) -> int:
    """Run ``app`` and print the files it wrote; the exit code."""
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
