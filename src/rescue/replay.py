"""``python -m rescue.replay <run.json>``: scrub through a saved mission (brief §6.5).

Run files come from ``python -m rescue.sim --save-run``, ``python -m rescue.demo --save`` or
:func:`rescue.runs.save_run`. Controls: ``←/→`` step, ``Shift+←/→`` ten steps, ``Home/End``,
``Space`` play/pause, ``+/-`` speed, ``T`` true map, ``P`` policy arrows, ``C`` contrast,
``G`` record, ``H`` help, ``Esc`` quit.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from rescue.cli_utils import ensure_utf8_stdout

# pygame prints a banner on import unless this is set (rescue.viz imports pygame).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the replay viewer."""
    from rescue.viz import theme

    parser = argparse.ArgumentParser(
        prog="python -m rescue.replay",
        description="Replay a saved rescue mission (run file) with scrubbing.",
    )
    parser.add_argument("run_file", help="run JSON written by save_run / --save-run")
    parser.add_argument("--headless", action="store_true",
                        help="no window (SDL dummy driver); renders --frames frames")
    parser.add_argument("--frames", type=int, default=theme.HEADLESS_FRAMES,
                        help="frames to render in headless mode")
    parser.add_argument("--screenshot", metavar="PATH", help="save the last frame (headless)")
    parser.add_argument("--step", type=int, default=0,
                        help="step shown first (negative counts from the end)")
    parser.add_argument("--autoplay", action="store_true",
                        help="start playing immediately (always on in headless mode)")
    parser.add_argument("--speed", type=int, default=theme.DEFAULT_SPEED_INDEX, metavar="INDEX",
                        help=f"initial speed index into {list(theme.SPEEDS)} steps/s")
    parser.add_argument("--high-contrast", action="store_true", help="high-contrast palette")
    parser.add_argument("--record", metavar="PATH", help="record a GIF / MP4 from the start")
    parser.add_argument("--mp4", action="store_true", help="G records MP4 instead of GIF")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the run file, run the replay viewer and return the exit code."""
    ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    from rescue.runs import load_run
    from rescue.viz.sim_app import prepare_environment

    prepare_environment(args.headless)
    from rescue.viz.replay_app import ReplayApp, ReplayOptions

    try:
        record = load_run(args.run_file)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"cannot read run file {args.run_file}: {exc}")
    options = ReplayOptions(
        high_contrast=args.high_contrast,
        headless=args.headless,
        frames=args.frames,
        screenshot=args.screenshot,
        record=args.record,
        mp4=args.mp4,
        autoplay=True if args.autoplay else None,
        speed_index=args.speed,
        step=args.step,
    )
    app = ReplayApp(record, options)
    code = app.run()
    for path in app.recordings:
        print(f"Recording saved to {path}")
    if args.screenshot and args.headless:
        print(f"Screenshot saved to {args.screenshot}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
