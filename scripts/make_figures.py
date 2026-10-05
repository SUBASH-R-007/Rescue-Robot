"""Regenerate the Phase 4 figures in ``docs/figures/`` (headless and deterministic).

Usage (from the repository root, with the dev venv)::

    python scripts/make_figures.py                 # all figures into docs/figures/
    python scripts/make_figures.py --out some/dir  # somewhere else
    python scripts/make_figures.py --only gif tree

Figures (``config.DEMO_SEED``, ``config.DEFAULT_SLIP_P``, the ``andor`` agent):

* ``deck_demo.gif`` - the simulator on ``maps/deck_demo.txt`` from the start until the rescue,
  at :data:`GIF_STEPS_PER_SECOND`, captured at ``theme.RECORD_FPS`` / ``theme.RECORD_SCALE``
  with policy arrows on, a short lead-in on the initial plan and :data:`GIF_HOLD_SECONDS` of
  hold frames on the rescue;
* ``sim_deck_demo.png`` - the simulator right after the replan at (2,2) (debris sensed at (3,2)),
  policy arrows on;
* ``sim_deck_figure.png`` - ``maps/deck_figure.txt`` at t = 0 with the true-map overlay on (the
  title-slide figure);
* ``plan_tree_deck_demo.png`` - ``export_plan_tree`` of the initial deck_demo plan
  (Act depth :data:`TREE_DEPTH`, laid out left to right so the deep, narrow plan keeps a
  page-friendly aspect ratio).

The script only drives the public app API (:class:`~rescue.viz.sim_app.SimApp` ``tick`` /
``render`` / ``handle_key`` / ``step_once``, :class:`~rescue.viz.record.Recorder`) and the
:class:`~rescue.runner.Mission` runner; it contains no mission logic. The simulated clock is
advanced by exactly one recording interval per frame, so every run produces the same frames.
The panel's planning wall-clock time ("Plan ms") would differ between runs, so the figures show
the (deterministic) number of nodes expanded in its place - see :class:`FigureSimApp`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rescue.viz.sim_app import SimApp, SimOptions

#: Repository root (scripts/make_figures.py -> one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Default output directory.
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
#: Maps used by the figures.
DEMO_MAP = REPO_ROOT / "maps" / "deck_demo.txt"
FIGURE_MAP = REPO_ROOT / "maps" / "deck_figure.txt"
#: Output file names.
GIF_NAME = "deck_demo.gif"
REPLAN_PNG_NAME = "sim_deck_demo.png"
FIGURE_PNG_NAME = "sim_deck_figure.png"
TREE_PNG_NAME = "plan_tree_deck_demo.png"

#: Agent shown in every figure.
FIGURE_AGENT = "andor"
#: Mission steps per second in the GIF (must be one of ``theme.SPEEDS``).
GIF_STEPS_PER_SECOND = 2
#: Seconds the GIF shows the initial plan before the first step is due.
GIF_LEAD_SECONDS = 1.0
#: Seconds of hold frames after the rescue.
GIF_HOLD_SECONDS = 1.5
#: Policy arrows in the GIF (the current contingency plan, redrawn after each replan).
GIF_POLICY_ARROWS = True
#: Safety cap on GIF frames (the deck_demo mission needs about 80).
GIF_MAX_FRAMES = 2000
#: Simulated time between the replan and the ``sim_deck_demo.png`` capture (ms). Half of the
#: replan flash: the contradicted cell is still highlighted but its debris hatch shows through.
REPLAN_SCREENSHOT_DELAY_MS = 450.0
#: Act depth of the exported plan tree.
TREE_DEPTH = 4
#: Milliseconds per second.
MS_PER_S = 1000.0
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


def _figure_app_class() -> type[SimApp]:
    """:class:`FigureSimApp` (defined lazily: importing the viz package imports pygame)."""
    from rescue.viz.sim_app import SimApp

    class FigureSimApp(SimApp):
        """The simulator with a reproducible panel: "Plan ms" (wall-clock planning time,
        different on every run) is replaced by the number of nodes the planner expanded."""

        def _metric_values(self) -> dict[str, str]:
            values = super()._metric_values()
            if PLAN_MS_LABEL not in values:
                return values
            nodes = str(self.mission.metrics().nodes_expanded)
            return {
                (NODES_LABEL if label == PLAN_MS_LABEL else label):
                    (nodes if label == PLAN_MS_LABEL else value)
                for label, value in values.items()
            }

    return FigureSimApp


def _options(map_path: Path, *, autoplay: bool, speed: float | None = None) -> SimOptions:
    """Headless simulator options for ``map_path`` with the demo seed, p and agent."""
    from rescue.config import DEFAULT_SLIP_P, DEMO_SEED
    from rescue.viz import theme
    from rescue.viz.sim_app import SimOptions

    speed_index = theme.DEFAULT_SPEED_INDEX if speed is None else theme.SPEEDS.index(speed)
    return SimOptions(
        map_path=str(map_path),
        seed=DEMO_SEED,
        agent=FIGURE_AGENT,
        slip_p=DEFAULT_SLIP_P,
        headless=True,
        autoplay=autoplay,
        speed_index=speed_index,
        maps_dir=str(map_path.parent),
    )


def _make_app(options: SimOptions) -> SimApp:
    set_headless_environment()
    return _figure_app_class()(options)


def make_gif(out: Path) -> Path:
    """Record the deck_demo mission (start to rescue plus hold frames) to ``out``."""
    import pygame

    from rescue.viz import theme
    from rescue.viz.record import Recorder

    app = _make_app(_options(DEMO_MAP, autoplay=False, speed=GIF_STEPS_PER_SECOND))
    # A recorder of our own (not app.recorder): the panel would otherwise show "REC" in every
    # frame, and every frame is captured instead of those Recorder.due() picks at 60 FPS.
    recorder = Recorder()
    try:
        if GIF_POLICY_ARROWS:
            app.handle_key(pygame.K_p)
        frame_ms = MS_PER_S / theme.RECORD_FPS
        recorder.start(out, fps=theme.RECORD_FPS, scale=theme.RECORD_SCALE)

        def capture() -> None:
            app.tick(frame_ms)
            app.render()
            recorder.add_frame(app.screen)

        for _ in range(round(GIF_LEAD_SECONDS * theme.RECORD_FPS)):
            capture()
        app.handle_key(pygame.K_SPACE)  # play
        frames = 0
        while not app.mission.done:
            if frames >= GIF_MAX_FRAMES:
                raise RuntimeError(f"mission still running after {GIF_MAX_FRAMES} frames")
            capture()
            frames += 1
        for _ in range(round(GIF_HOLD_SECONDS * theme.RECORD_FPS)):
            capture()
        n_frames = recorder.frame_count
        written = recorder.stop()
        status = app.mission.status.name
        steps = app.mission.events[-1].t
    finally:
        if recorder.active:
            recorder.stop()
        app.close()
    if written is None:
        raise RuntimeError("no GIF frames were captured")
    print(f"  {n_frames} frames, {steps} steps, final status {status}")
    return written


def make_replan_screenshot(out: Path) -> Path:
    """The deck_demo mission just after its first contradiction-triggered replan."""
    import pygame

    from rescue.grid import fmt_pos
    from rescue.viz import theme

    app = _make_app(_options(DEMO_MAP, autoplay=False))
    try:
        app.handle_key(pygame.K_p)  # policy arrows on
        while True:
            event = app.mission.events[-1]
            if event.replanned and event.triggering_contradictions:
                break
            if not app.step_once():
                raise RuntimeError("the deck_demo mission ended without a contradiction replan")
        # The app clamps a single tick to theme.MAX_FRAME_MS: advance in frame-sized ticks.
        frame_ms = theme.HEADLESS_FRAME_MS
        for _ in range(round(REPLAN_SCREENSHOT_DELAY_MS / frame_ms)):
            app.tick(frame_ms)
        app.render()
        path = app.save_screenshot(out)
    finally:
        app.close()
    cells = ", ".join(fmt_pos(p) for p in event.triggering_contradictions)
    print(f"  replanned at {fmt_pos(event.new_pos)} (t = {event.t}) on debris at {cells}")
    return path


def make_figure_screenshot(out: Path) -> Path:
    """deck_figure at t = 0 with the true-map overlay (title-slide figure)."""
    import pygame

    app = _make_app(_options(FIGURE_MAP, autoplay=False))
    try:
        app.handle_key(pygame.K_t)  # true-map overlay on
        app.render()
        path = app.save_screenshot(out)
    finally:
        app.close()
    return path


def make_plan_tree(out: Path) -> Path:
    """``export_plan_tree`` of the first plan the agent makes on deck_demo."""
    from rescue.agents import make_agent
    from rescue.config import DEFAULT_SLIP_P, DEMO_SEED, AgentConfig, PlannerConfig, WorldConfig
    from rescue.grid import TrueGrid, fmt_pos
    from rescue.runner import Mission, make_transition
    from rescue.viz.export import export_plan_tree

    grid = TrueGrid.load(DEMO_MAP)
    cfg = WorldConfig(slip_p=DEFAULT_SLIP_P)
    agent = make_agent(FIGURE_AGENT, make_transition(grid, cfg), PlannerConfig(), AgentConfig())
    mission = Mission(grid, agent, cfg, seed=DEMO_SEED, scenario=grid.name)
    mission.start()
    plan = agent.snapshots[0].plan
    if plan is None:
        raise RuntimeError("the agent made no initial plan on deck_demo")
    title = (f"Initial AND-OR plan on {grid.name} from {fmt_pos(grid.start)} "
             f"(p = {DEFAULT_SLIP_P:g}):  {plan.pretty()}")
    print(f"  {plan.pretty()}")
    return export_plan_tree(plan, out, max_depth=TREE_DEPTH, title=title)


#: Figure key -> (file name, generator).
FIGURES: dict[str, tuple[str, Callable[[Path], Path]]] = {
    "gif": (GIF_NAME, make_gif),
    "replan": (REPLAN_PNG_NAME, make_replan_screenshot),
    "figure": (FIGURE_PNG_NAME, make_figure_screenshot),
    "tree": (TREE_PNG_NAME, make_plan_tree),
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the figure script."""
    parser = argparse.ArgumentParser(
        prog="python scripts/make_figures.py",
        description="Regenerate the simulator figures in docs/figures/ (headless).",
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
