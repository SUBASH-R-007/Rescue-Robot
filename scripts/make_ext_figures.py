"""Regenerate the extension-simulator figures in ``docs/figures/`` (headless, deterministic).

Usage (from the repository root, with the dev venv)::

    python scripts/make_ext_figures.py                 # all three into docs/figures/
    python scripts/make_ext_figures.py --out some/dir  # somewhere else
    python scripts/make_ext_figures.py --only search noisy

Figures (:class:`~rescue.viz.ext_sim_app.ExtSimApp` with its defaults: the ext demo's map
``EXT_DEMO_SCENARIO`` ``EXT_DEMO_SIZE``, seed ``EXT_DEMO_SEED``, p = ``DEFAULT_SLIP_P``,
``search_andor``; docs/plan.md §3.15.3):

* ``ext_search.png`` - ``search`` mode (survivor location unknown) mid-exploration: the first
  step at or after :data:`SEARCH_STEP` whose frontier target is at least
  :data:`SEARCH_TARGET_DISTANCE` cells away, policy arrows on, so the dashed frontier target
  and the plan leading to it are visible;
* ``ext_multi.png`` - ``multi`` mode (3 known survivors) right after the first rescue: the
  rescued survivor's ring and tick, the next survivor as the target;
* ``ext_noisy.png`` - ``noisy`` mode (ε = ``EXT_DEMO_NOISE``) right after the
  :data:`NOISY_BUMPS`-th bump into debris the sensor missed, with the P(debris) shading of the
  observed open cells and its legend.

Like ``scripts/make_figures.py`` (which this script leaves alone) it only drives the public app
API, advances the simulated clock by fixed ticks and shows the planner's node count in place of
the wall-clock "Plan ms", so every run writes the same files.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rescue.viz.ext_sim_app import ExtSimApp

#: Repository root (scripts/make_ext_figures.py -> one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Default output directory.
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
#: Output file names.
SEARCH_PNG_NAME = "ext_search.png"
MULTI_PNG_NAME = "ext_multi.png"
NOISY_PNG_NAME = "ext_noisy.png"
#: ``ext_search.png``: earliest step captured, then the first one whose explore target is at
#: least SEARCH_TARGET_DISTANCE cells (Manhattan) from the robot (so the plan has some extent).
SEARCH_STEP = 30
SEARCH_TARGET_DISTANCE = 3
#: ``ext_noisy.png``: captured right after this many bumps (or at the end of the mission).
NOISY_BUMPS = 3
#: Simulated time between the captured step and the screenshot (ms): half of the replan flash,
#: as in ``make_figures.py``, so a contradicted cell is highlighted but its hatch shows through.
SCREENSHOT_DELAY_MS = 450.0
#: Safety cap on mission steps while looking for a capture point.
MAX_STEPS = 5000
#: Metric label replaced in the figures' panel, and its deterministic substitute.
PLAN_MS_LABEL = "Plan ms"
NODES_LABEL = "Nodes"
#: SDL drivers for running without a window or a sound device.
HEADLESS_ENV: dict[str, str] = {
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
}


def set_headless_environment() -> None:
    """Select SDL's dummy video / audio drivers; must run before ``pygame.init()``."""
    os.environ.update(HEADLESS_ENV)


def make_app(mode: str) -> ExtSimApp:
    """A headless, paused :class:`ExtSimApp` in ``mode`` whose panel shows the planner's node
    count instead of the wall-clock planning time."""
    set_headless_environment()
    from rescue.viz.ext_sim_app import ExtSimApp, ExtSimOptions

    class FigureExtSimApp(ExtSimApp):
        def _metric_values(self) -> dict[str, str]:
            values = super()._metric_values()
            nodes = str(self.mission.metrics().nodes_expanded)
            return {
                (NODES_LABEL if label == PLAN_MS_LABEL else label):
                    (nodes if label == PLAN_MS_LABEL else value)
                for label, value in values.items()
            }

    return FigureExtSimApp(ExtSimOptions(mode=mode, headless=True, autoplay=False))


def _step_until(app: ExtSimApp, done: Callable[[ExtSimApp], bool]) -> None:
    """Single-step ``app`` until ``done(app)`` or the end of the mission."""
    for _ in range(MAX_STEPS):
        if done(app) or not app.step_once():
            return
    raise RuntimeError(f"no capture point within {MAX_STEPS} steps")


def _capture(app: ExtSimApp, out: Path) -> Path:
    """Advance the clock by :data:`SCREENSHOT_DELAY_MS` in frame-sized ticks, render, save."""
    from rescue.viz import theme

    frame_ms = theme.HEADLESS_FRAME_MS
    for _ in range(round(SCREENSHOT_DELAY_MS / frame_ms)):
        app.tick(frame_ms)
    app.render()
    return app.save_screenshot(out)


def _describe(app: ExtSimApp) -> str:
    from rescue.grid import fmt_pos

    m = app.mission.metrics()
    target = app.mission.agent.target
    where = fmt_pos(target) if target is not None else "none"
    return (f"t = {m.steps}, mode {app.mission.agent.mode}, target {where}, rescued "
            f"{m.rescued}/{m.k}, bumps {m.bumps}")


def make_search(out: Path) -> Path:
    """``search`` mode mid-exploration, policy arrows on."""
    import pygame

    from rescue.grid import manhattan

    app = make_app("search")
    try:
        app.handle_key(pygame.K_p)  # policy arrows on
        def ready(a: ExtSimApp) -> bool:
            agent, t = a.mission.agent, a.mission.world.t
            return (t >= SEARCH_STEP and agent.mode == "explore" and agent.target is not None
                    and manhattan(agent.target, a.mission.world.pos) >= SEARCH_TARGET_DISTANCE)

        _step_until(app, ready)
        print(f"  {_describe(app)}")
        return _capture(app, out)
    finally:
        app.close()


def make_multi(out: Path) -> Path:
    """``multi`` mode right after the first rescue."""
    app = make_app("multi")
    try:
        _step_until(app, lambda a: bool(a.mission.rescued))
        print(f"  {_describe(app)}")
        return _capture(app, out)
    finally:
        app.close()


def make_noisy(out: Path) -> Path:
    """``noisy`` mode right after the :data:`NOISY_BUMPS`-th bump, with the P(debris) shading."""
    app = make_app("noisy")
    try:
        _step_until(app, lambda a: a.mission.metrics().bumps >= NOISY_BUMPS)
        print(f"  {_describe(app)}")
        return _capture(app, out)
    finally:
        app.close()


#: Figure key -> (file name, generator).
FIGURES: dict[str, tuple[str, Callable[[Path], Path]]] = {
    "search": (SEARCH_PNG_NAME, make_search),
    "multi": (MULTI_PNG_NAME, make_multi),
    "noisy": (NOISY_PNG_NAME, make_noisy),
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the figure script."""
    parser = argparse.ArgumentParser(
        prog="python scripts/make_ext_figures.py",
        description="Regenerate the extension-simulator figures in docs/figures/ (headless).",
    )
    parser.add_argument("--out", type=Path, default=FIGURES_DIR,
                        help=f"output directory (default {FIGURES_DIR.relative_to(REPO_ROOT)})")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), metavar="FIGURE",
                        help=f"generate only these figures ({', '.join(FIGURES)})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the requested figures and print their paths and sizes; exit code 0."""
    from rescue.cli_utils import ensure_utf8_stdout

    set_headless_environment()
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.only or FIGURES:
        name, make = FIGURES[key]
        print(f"{name}:")
        path = make(out_dir / name)
        print(f"  -> {path} ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
