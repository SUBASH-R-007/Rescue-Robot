"""Interactive simulator (``python -m rescue.sim``, brief §6.1, docs/plan.md §3.11).

:class:`SimApp` drives a :class:`~rescue.runner.Mission` and renders it: the grid (fog of war,
trace, policy arrows, true-map overlay, replan flash), the info panel (status badge, metrics,
plan tree, event log) and the help overlay. It contains no mission logic: every step is
``Mission.step()``, every log line and trace mark comes from the resulting
:class:`~rescue.runner.StepEvent`, and the plan view is the agent's current
:class:`~rescue.agents.base.PlanSnapshot`.

:class:`BaseApp` holds what the simulator shares with the replay app (window, fonts, views,
palette switching, speed, play/pause, help, recording and the frame loop). Both apps run
headless (``SDL_VIDEODRIVER=dummy``) for tests and recordings: ``render()``, ``tick(dt_ms)`` and
``handle_key(key, mods)`` can be driven without an event loop.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pygame

from rescue.agents import AGENT_NAMES, make_agent
from rescue.config import (
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
    DEMO_SEED,
    MAX_SLIP_P,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.grid import Cell, Pos, TrueGrid, fmt_pos
from rescue.model import Action, OutcomeKind, kind_direction, step_pos
from rescue.runner import FAILURE_REASONS, Mission, MissionStatus, StepEvent, make_transition
from rescue.runs import DEFAULT_RUNS_DIR, save_run
from rescue.viz import theme
from rescue.viz.grid_view import GridScene, GridView, TraceMark
from rescue.viz.panel import (
    DEFAULT_HINT,
    EventLog,
    Panel,
    PanelInfo,
    draw_help_overlay,
)
from rescue.viz.record import Recorder
from rescue.viz.tree_view import TreeView

if TYPE_CHECKING:
    from rescue.agents.base import PlanSnapshot

log = logging.getLogger(__name__)

#: Environment that makes SDL run without a window or sound device.
HEADLESS_ENV: dict[str, str] = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
#: Repository root (src/rescue/viz/sim_app.py → four levels up).
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Map directory name (relative to the working directory, else the repository root).
MAPS_DIRNAME = "maps"
#: Glob of map files cycled by ``M``.
MAP_GLOB = "*.txt"
#: File-name prefix and timestamp format of maps saved with Ctrl+S, and of recordings.
EDITED_MAP_PREFIX = "edited_"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
#: Grid name given to a map changed in the editor.
EDITED_GRID_NAME = "edited"
#: Recording suffixes.
GIF_SUFFIX = ".gif"
MP4_SUFFIX = ".mp4"
#: Panel titles and editor hint.
SIM_TITLE = "Rescue simulator"
EDITOR_TITLE = "Map editor"
EDITOR_HINT = "Click debris · Shift+click start · Ctrl+click survivor · E done · Ctrl+S save"
#: Plan-tree text while the editor is open (the plan of the previous map no longer applies).
EDITOR_TREE_TEXT = (
    "Editing the true map: the plan is built when you press E again.\n"
    "• Click: toggle debris\n"
    "• Shift+click: move the start\n"
    "• Ctrl+click: move the survivor\n"
    "• Ctrl+S: save the map to maps/"
)
#: Plan-tree text when the latest planning call found no plan (A* or AND-OR).
NO_PLAN_TEXT = "No plan: the survivor is unreachable from {pos} on the robot's map"
#: Label of the step counter (value "steps/budget"; the replay shows "step/run length").
STEPS_LABEL = "Steps / budget"
#: Rounding of the slip probability after ``[`` / ``]`` (avoids 0.30000000000000004).
SLIP_P_DECIMALS = 2

#: Number keys (top row and keypad) selecting AGENT_NAMES[i].
_AGENT_KEYS: dict[int, int] = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
    pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2,
}
_FASTER_KEYS = frozenset({pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS})
_SLOWER_KEYS = frozenset({pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS})

#: Short labels of the outcome kinds in log lines.
_SKID_LABELS: dict[OutcomeKind, str] = {
    OutcomeKind.SKID_L: "Skid-L",
    OutcomeKind.SKID_R: "Skid-R",
}


def prepare_environment(headless: bool) -> None:
    """Hide pygame's banner and, when ``headless``, select the dummy SDL drivers.

    Must run before ``pygame.init()`` (the drivers are read at display / mixer init).
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if headless:
        os.environ.update(HEADLESS_ENV)


def timestamp() -> str:
    """Current local time as ``YYYYmmdd-HHMMSS`` (file names)."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def unique_path(path: Path) -> Path:
    """``path``, or ``stem-1``, ``stem-2`` … if it already exists."""
    candidate, n = path, 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


def default_maps_dir() -> Path:
    """``./maps`` if it exists, else the repository's ``maps`` directory."""
    local = Path(MAPS_DIRNAME)
    return local if local.is_dir() else _REPO_ROOT / MAPS_DIRNAME


def resolve_path(path: str | Path) -> Path:
    """``path`` if it exists (or is absolute), else the same relative path under the repo."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    candidate = _REPO_ROOT / p
    return candidate if candidate.exists() else p


# --- event-log text ----------------------------------------------------------------------------
def _cells(cells: Sequence[Pos]) -> str:
    return ", ".join(fmt_pos(p) for p in cells)


def _attempted_target(event: StepEvent) -> Pos | None:
    """The cell a move event tried to enter (None for SLIPPED / non-move events)."""
    if event.action is None or event.outcome is None:
        return None
    direction = kind_direction(event.outcome, event.action)
    return None if direction is None else step_pos(event.pos, direction)


def event_messages(event: StepEvent) -> list[tuple[str, str]]:
    """Event-log lines ``(text, kind)`` describing one :class:`StepEvent`.

    Kinds are those of :class:`~rescue.viz.panel.EventLog`. Ordinary ``arrived`` moves produce
    no line; slips, skids, blocked moves, bumps, aftershocks, contradictions / replans,
    unplanned states and the mission's end do.
    """
    lines: list[tuple[str, str]] = []
    target = _attempted_target(event)
    if event.outcome is OutcomeKind.SLIPPED:
        lines.append((f"Slip at {fmt_pos(event.pos)}", "slip"))
    elif event.outcome in _SKID_LABELS:
        label = _SKID_LABELS[event.outcome]
        if event.blocked and target is not None:
            lines.append((f"{label} to {fmt_pos(target)} (blocked)", "skid"))
        else:
            lines.append((f"{label} to {fmt_pos(event.new_pos)}", "skid"))
    elif event.outcome is OutcomeKind.ARRIVED and event.blocked and target is not None:
        lines.append((f"Move {event.action} into {fmt_pos(target)} blocked", "info"))
    if event.bump and target is not None:
        lines.append((f"Bumped into debris at {fmt_pos(target)}", "warn"))
    if event.aftershock_cells:
        lines.append((f"Aftershock: debris at {_cells(event.aftershock_cells)}", "warn"))
    triggering = tuple(event.triggering_contradictions)
    if event.replanned and triggering:
        lines.append((f"Debris sensed at {_cells(triggering)} → replanning", "replan"))
    ignored = tuple(p for p in event.contradictions if p not in triggering)
    if ignored and event.action is None:  # t = 0: sensed before the initial plan
        lines.append((f"Debris sensed at {_cells(ignored)}", "info"))
    elif ignored:
        lines.append((f"Debris sensed at {_cells(ignored)} (no replan needed)", "info"))
    if event.replanned and not triggering:
        lines.append((f"Replanned at {fmt_pos(event.new_pos)}", "replan"))
    if event.unplanned:
        lines.append((f"Unplanned state at {fmt_pos(event.new_pos)}!", "warn"))
    if event.status is MissionStatus.RESCUED:
        lines.append(("Rescue successful", "success"))
    elif event.status is not MissionStatus.RUNNING:
        reason = FAILURE_REASONS.get(event.status, event.status.name.lower())
        lines.append((f"Mission failed ({reason})", "failure"))
        if event.action is None and event.note:
            lines.append((event.note, "info"))
    return lines


def flash_cells(event: StepEvent) -> tuple[Pos, ...]:
    """Cells the replan flash highlights after ``event``: the contradictions (debris sensed
    where the belief had unknown / open) that triggered a replan - the same cells
    :func:`rescue.runner.format_trace` annotates. Empty when nothing was replanned, when the
    replan had other causes only (a deviation), and at t = 0 (the initial plan is no replan);
    contradictions lazy mode ignored are logged but do not flash."""
    if not event.replanned:
        return ()
    return tuple(event.triggering_contradictions)


def is_flash_event(event: StepEvent) -> bool:
    """True if ``event`` triggers the replan flash (see :func:`flash_cells`)."""
    return bool(flash_cells(event))


def no_plan_message(snapshot_pos: Pos | None, has_plan: bool) -> str | None:
    """Plan-tree message for a snapshot without plan / path: None when there is a plan, "no
    plan yet" before the first snapshot, else the "survivor unreachable from (r,c)" text."""
    if has_plan:
        return None
    if snapshot_pos is None:
        return None  # TreeView's default "No plan yet"
    return NO_PLAN_TEXT.format(pos=fmt_pos(snapshot_pos))


def help_lines(mission_keys: bool = True) -> list[tuple[str, str] | str]:
    """Help-overlay items for the simulator (``mission_keys``) or the replay app."""
    mouse: list[tuple[str, str] | str] = [
        "Mouse",
        ("Tree click", "fold / unfold a row (on its ▸ / ▾)"),
        ("Wheel", "scroll the plan tree / event log"),
    ]
    view: list[tuple[str, str] | str] = [
        "View",
        ("T", "true-map overlay (35 %)"),
        ("P", "policy arrows"),
        ("C", "high-contrast palette"),
        ("G", "record GIF / MP4"),
        ("H", "this help"),
        ("Esc", "quit"),
    ]
    if not mission_keys:
        return [
            "Replay",
            ("Space", "play / pause"),
            ("← / →", "one step back / forward"),
            ("Shift+← / →", f"{theme.SCRUB_BIG_STEP} steps"),
            ("Home / End", "first / last step"),
            ("+ / -", "speed"),
            *view,
            *mouse,
            ("Click bar", "jump to that step"),
        ]
    agents = " / ".join(AGENT_NAMES)
    return [
        "Mission",
        ("Space", "play / pause"),
        ("N", "single step"),
        ("R", "reset (same seed)"),
        ("S", "new seed (seed + 1)"),
        ("+ / -", "speed"),
        ("[ / ]", f"slip p −/+ {theme.SLIP_P_STEP:g}"),
        ("1 2 3", agents),
        ("M", "next scenario"),
        "Map editor",
        ("E", "edit the true map (pauses)"),
        ("Click", "toggle debris"),
        ("Shift+click", "set start"),
        ("Ctrl+click", "set survivor"),
        ("Ctrl+S", "save map to maps/"),
        *view,
        *mouse,
    ]


# --- shared app shell --------------------------------------------------------------------------
class BaseApp:
    """Window, views, palette, speed, play/pause, help, recording and the frame loop.

    Subclasses implement :meth:`_can_advance`, :meth:`_advance` (one mission/replay step),
    :meth:`_scene`, :meth:`_panel_info`, :meth:`_update_tree` and may extend
    :meth:`handle_key` / :meth:`_handle_mouse`.
    """

    caption: ClassVar[str] = theme.SIM_CAPTION
    help_items: ClassVar[bool] = True

    def __init__(
        self,
        *,
        headless: bool,
        high_contrast: bool,
        frames: int,
        screenshot: str | Path | None,
        record: str | Path | None,
        mp4: bool,
        recordings_dir: str | Path,
        speed_index: int,
        autoplay: bool | None,
        grid_rect: pygame.Rect | None = None,
    ) -> None:
        prepare_environment(headless)
        pygame.init()
        self.headless = headless
        self.frames = int(frames)
        self.screenshot = Path(screenshot) if screenshot else None
        self.mp4 = mp4
        self.recordings_dir = Path(recordings_dir)
        self.screen = pygame.display.set_mode(theme.WINDOW_SIZE)
        pygame.display.set_caption(self.caption)
        self.fonts = theme.load_fonts()
        self.palette_index = 1 if high_contrast else 0
        palette = self.palette
        self.grid_view = GridView(grid_rect or theme.grid_area(theme.WINDOW_SIZE), palette,
                                  self.fonts)
        self.tree = TreeView(pygame.Rect(0, 0, 1, 1), palette, self.fonts)
        self.panel = Panel(theme.panel_area(theme.WINDOW_SIZE), palette, self.fonts)
        self.log = EventLog()
        self.recorder = Recorder()
        self.clock_ms = 0.0
        self.running = True
        self.paused = not (headless if autoplay is None else autoplay)
        self.speed_index = max(0, min(len(theme.SPEEDS) - 1, speed_index))
        self.show_true = False
        self.show_policy = False
        self.show_help = False
        self._accumulator = 0.0
        self._flash_cells: tuple[Pos, ...] = ()
        self._flash_start = -math.inf
        self._closed = False
        #: Files written by recordings that were stopped.
        self.recordings: list[Path] = []
        if record:
            self._start_recording(Path(record))

    # --- properties ------------------------------------------------------------------------
    @property
    def palette(self) -> theme.Palette:
        """The palette in use (``C`` cycles :data:`theme.PALETTES`)."""
        return theme.PALETTES[self.palette_index % len(theme.PALETTES)]

    @property
    def speed(self) -> float:
        """Steps per second while playing."""
        return theme.SPEEDS[self.speed_index]

    # --- hooks -----------------------------------------------------------------------------
    def _can_advance(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _advance(self) -> bool:  # pragma: no cover - abstract
        """Take one step; False if nothing could be stepped."""
        raise NotImplementedError

    def _scene(self) -> GridScene:  # pragma: no cover - abstract
        raise NotImplementedError

    def _panel_info(self) -> PanelInfo:  # pragma: no cover - abstract
        raise NotImplementedError

    def _update_tree(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _draw_extra(self, surface: pygame.Surface) -> None:
        """Extra drawing after grid and panel (replay progress bar)."""

    def _handle_mouse(self, event: pygame.event.Event) -> bool:
        """App-specific mouse handling; True if consumed."""
        return False

    # --- flash -----------------------------------------------------------------------------
    def _flash(self, cells: Sequence[Pos]) -> None:
        """Start the replan flash on ``cells``."""
        if cells:
            self._flash_cells = tuple(cells)
            self._flash_start = self.clock_ms

    def _flash_alpha(self) -> float:
        return theme.flash_alpha(self.clock_ms - self._flash_start) if self._flash_cells else 0.0

    # --- loop ------------------------------------------------------------------------------
    def tick(self, dt_ms: float) -> int:
        """Advance the clock by ``dt_ms``; while playing, take the steps that are due.

        Steps happen at :attr:`speed` steps per second of accumulated frame time (at most
        :data:`theme.MAX_STEPS_PER_FRAME` per call). Returns the number of steps taken.
        """
        dt = max(0.0, min(float(dt_ms), theme.MAX_FRAME_MS))
        self.clock_ms += dt
        if self.paused or not self._can_advance():
            self._accumulator = 0.0
            return 0
        self._accumulator += dt * self.speed / 1000.0
        due = int(self._accumulator)
        self._accumulator -= due
        taken = 0
        for _ in range(min(due, theme.MAX_STEPS_PER_FRAME)):
            if not self._advance():
                break
            taken += 1
        return taken

    def render(self) -> pygame.Surface:
        """Draw the whole frame into the window surface and return it."""
        surface = self.screen
        pal = self.palette
        surface.fill(pal.background)
        scene = self._scene()
        scene.flash_alpha = self._flash_alpha()
        if scene.flash_alpha <= 0:
            scene.flash_cells = ()
        self.grid_view.draw(surface, scene)
        self._update_tree()
        self.panel.draw(surface, self._panel_info(), self.log, self.tree)
        self._draw_extra(surface)
        if self.show_help:
            draw_help_overlay(surface, pal, self.fonts, help_lines(self.help_items))
        return surface

    def frame(self, dt_ms: float) -> None:
        """One frame: :meth:`tick`, :meth:`render` and capture for an active recording."""
        self.tick(dt_ms)
        self.render()
        if self.recorder.active and self.recorder.due(self.clock_ms):
            self.recorder.add_frame(self.screen)

    def run(self) -> int:
        """Run the app until Esc / window close (or ``frames`` frames headless); exit code."""
        try:
            if self.headless:
                for _ in range(self.frames):
                    self.frame(theme.HEADLESS_FRAME_MS)
                    if not self.running:
                        break
                if self.screenshot is not None:
                    self.save_screenshot(self.screenshot)
                return 0
            clock = pygame.time.Clock()
            while self.running:
                dt = clock.tick(theme.TARGET_FPS)
                for event in pygame.event.get():
                    self.handle_event(event)
                if not self.running:
                    break
                self.frame(dt)
                pygame.display.flip()
            return 0
        finally:
            self.close()

    def close(self) -> None:
        """Finish any recording and shut pygame down (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self.recorder.active:
            self._stop_recording()
        pygame.quit()

    def save_screenshot(self, path: str | Path) -> Path:
        """Save the current frame as an image (parent directories are created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(target))
        log.info("screenshot saved to %s", target)
        return target

    # --- input -----------------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        """Dispatch one pygame event (quit, keys, mouse)."""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key, event.mod)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL):
            if self.show_help:
                return
            if not self._handle_mouse(event):
                self.panel.handle_event(event, self.log, self.tree)

    def handle_key(self, key: int, mods: int = 0) -> bool:
        """Keys shared by every app; True if ``key`` was handled."""
        if self.show_help and key in (pygame.K_h, pygame.K_ESCAPE):
            self.show_help = False
            return True
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_t:
            self.show_true = not self.show_true
        elif key == pygame.K_p:
            self.show_policy = not self.show_policy
        elif key == pygame.K_c:
            self._cycle_palette()
        elif key == pygame.K_g:
            self.toggle_recording()
        elif key in _FASTER_KEYS:
            self.speed_index = min(len(theme.SPEEDS) - 1, self.speed_index + 1)
        elif key in _SLOWER_KEYS:
            self.speed_index = max(0, self.speed_index - 1)
        else:
            return False
        return True

    def _cycle_palette(self) -> None:
        self.palette_index = (self.palette_index + 1) % len(theme.PALETTES)
        pal = self.palette
        self.grid_view.set_palette(pal)
        self.tree.set_palette(pal)
        self.panel.set_palette(pal)
        self.log.add(f"Palette: {pal.name}")

    # --- recording -------------------------------------------------------------------------
    def toggle_recording(self) -> Path | None:
        """Start a recording to ``recordings/<timestamp>.gif|.mp4``, or stop the current one.

        Returns the path written when stopping (None otherwise).
        """
        if self.recorder.active:
            return self._stop_recording()
        suffix = MP4_SUFFIX if self.mp4 else GIF_SUFFIX
        self._start_recording(unique_path(self.recordings_dir / f"{timestamp()}{suffix}"))
        return None

    def _start_recording(self, path: Path) -> None:
        actual = self.recorder.start(path)
        self.log.add(f"Recording to {actual}", "info")

    def _stop_recording(self) -> Path | None:
        path = self.recorder.stop()
        if path is None:
            self.log.add("Recording stopped (no frames captured)", "warn")
        else:
            self.recordings.append(path)
            self.log.add(f"Recording saved to {path}", "success")
            log.info("recording saved to %s", path)
        return path


# --- simulator ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimOptions:
    """Simulator settings (the CLI flags of ``python -m rescue.sim``).

    ``map_path`` and ``scenario`` choose the first map (default ``maps/deck_demo.txt``);
    ``rows``/``cols`` size generated scenarios. ``autoplay`` None means "play when headless".
    ``maps_dir`` (None = ``./maps`` or the repository's) is where ``M`` looks for maps and
    Ctrl+S saves them.
    """

    map_path: str | None = None
    scenario: str | None = None
    rows: int = theme.GENERATED_MAP_SIZE[0]
    cols: int = theme.GENERATED_MAP_SIZE[1]
    seed: int = DEMO_SEED
    agent: str = AGENT_NAMES[0]
    slip_p: float = DEFAULT_SLIP_P
    sensor_radius: int = DEFAULT_SENSOR_RADIUS
    replan_mode: str = "eager"
    aftershock_q: float = 0.0
    aftershock_k: int = 1
    high_contrast: bool = False
    headless: bool = False
    frames: int = theme.HEADLESS_FRAMES
    screenshot: str | None = None
    record: str | None = None
    autoplay: bool | None = None
    mp4: bool = False
    speed_index: int = theme.DEFAULT_SPEED_INDEX
    save_run: bool = False
    maps_dir: str | None = None
    recordings_dir: str = theme.RECORDINGS_DIR
    runs_dir: str = DEFAULT_RUNS_DIR


@dataclass(frozen=True)
class ScenarioEntry:
    """One entry of the ``M`` cycle: a map file or a ``rescue.mapgen`` generator."""

    name: str
    path: Path | None = None
    generator: str | None = None


def _mapgen() -> Any | None:
    """The ``rescue.mapgen`` module, or None before Phase 5 adds it."""
    try:
        from rescue import mapgen
    except ImportError:
        return None
    return mapgen


def generator_names() -> tuple[str, ...]:
    """Names of the ``rescue.mapgen`` generated scenarios (empty if mapgen is missing)."""
    mapgen = _mapgen()
    if mapgen is None:
        return ()
    names = getattr(mapgen, "GENERATED_SCENARIOS", None)
    if names is None:
        names = tuple(getattr(mapgen, "SCENARIOS", {}))
    return tuple(str(n) for n in names)


def discover_scenarios(maps_dir: Path) -> list[ScenarioEntry]:
    """Sorted ``maps_dir/*.txt`` files, then the mapgen generators not already listed."""
    entries = [ScenarioEntry(p.stem, path=p) for p in sorted(maps_dir.glob(MAP_GLOB))]
    known = {e.name for e in entries}
    entries += [ScenarioEntry(n, generator=n) for n in generator_names() if n not in known]
    return entries


class SimApp(BaseApp):
    """The interactive simulator; see the module docstring and :func:`help_lines`."""

    caption = theme.SIM_CAPTION
    help_items = True

    def __init__(self, options: SimOptions | None = None) -> None:
        opts = options or SimOptions()
        if opts.agent not in AGENT_NAMES:
            raise ValueError(f"unknown agent {opts.agent!r}; choose from {', '.join(AGENT_NAMES)}")
        self.options = opts
        self.maps_dir = Path(opts.maps_dir) if opts.maps_dir else default_maps_dir()
        self.scenarios = discover_scenarios(self.maps_dir)
        self.scenario_index = self._initial_scenario(opts)
        self.seed = int(opts.seed)
        self.slip_p = float(opts.slip_p)
        self.agent_name = opts.agent
        self.base_grid = self._load_grid(self.scenarios[self.scenario_index])
        WorldConfig(slip_p=self.slip_p, sensor_radius=opts.sensor_radius)  # validate early
        super().__init__(
            headless=opts.headless,
            high_contrast=opts.high_contrast,
            frames=opts.frames,
            screenshot=opts.screenshot,
            record=opts.record,
            mp4=opts.mp4,
            recordings_dir=opts.recordings_dir,
            speed_index=opts.speed_index,
            autoplay=opts.autoplay,
        )
        self.editor = False
        self.edit_grid: TrueGrid | None = None
        #: '~' mask of the map the editor started from (restored on cells that become open).
        self._edit_slip: np.ndarray | None = None
        self.trace: list[TraceMark] = []
        self.saved_runs: list[Path] = []
        self._serial = 0
        self._metrics_key: tuple[int, int] | None = None
        self._metrics: dict[str, str] = {}
        self._arrows_key: tuple[int, int, int] | None = None
        self._arrows: dict[Pos, Action] = {}
        self.mission: Mission
        self.reset_mission()

    # --- scenarios -------------------------------------------------------------------------
    def _initial_scenario(self, opts: SimOptions) -> int:
        """Index of the starting entry, inserting it when it is not in the cycle."""
        if opts.map_path is not None:
            path = resolve_path(opts.map_path)
            if not path.is_file():
                raise FileNotFoundError(f"map file not found: {opts.map_path}")
            for i, entry in enumerate(self.scenarios):
                if entry.path is not None and entry.path.resolve() == path.resolve():
                    return i
            self.scenarios.insert(0, ScenarioEntry(path.stem, path=path))
            return 0
        if opts.scenario is not None:
            for i, entry in enumerate(self.scenarios):
                if entry.name == opts.scenario:
                    return i
            mapgen = _mapgen()
            if mapgen is not None and opts.scenario in getattr(mapgen, "SCENARIOS", {}):
                self.scenarios.append(ScenarioEntry(opts.scenario, generator=opts.scenario))
                return len(self.scenarios) - 1
            available = ", ".join(e.name for e in self.scenarios) or "none"
            raise ValueError(f"unknown scenario {opts.scenario!r} (available: {available})")
        default = resolve_path(Path(MAPS_DIRNAME) / "deck_demo.txt")
        for i, entry in enumerate(self.scenarios):
            if entry.path is not None and entry.path.resolve() == default.resolve():
                return i
        if default.is_file():
            self.scenarios.insert(0, ScenarioEntry(default.stem, path=default))
            return 0
        if not self.scenarios:
            raise FileNotFoundError(f"no maps found in {self.maps_dir}")
        return 0

    def _load_grid(self, entry: ScenarioEntry) -> TrueGrid:
        """The true map of ``entry`` (generated with the current seed for mapgen entries)."""
        if entry.path is not None:
            return TrueGrid.load(entry.path)
        mapgen = _mapgen()
        if mapgen is None or entry.generator is None:
            raise ValueError(f"scenario {entry.name!r} needs rescue.mapgen")
        return mapgen.generate(entry.generator, self.options.rows, self.options.cols, self.seed)

    @property
    def scenario_name(self) -> str:
        """Name shown for the current map."""
        return self.base_grid.name or self.scenarios[self.scenario_index].name

    def next_scenario(self) -> None:
        """Switch to the next map / generator of the cycle (``M``) and reset the mission."""
        n = len(self.scenarios)
        for k in range(1, n + 1):
            index = (self.scenario_index + k) % n
            entry = self.scenarios[index]
            try:
                grid = self._load_grid(entry)
            except (OSError, ValueError, RuntimeError) as exc:
                self.log.add(f"Cannot load {entry.name}: {exc}", "warn")
                continue
            self.scenario_index, self.base_grid = index, grid
            self.reset_mission()
            return

    # --- mission ---------------------------------------------------------------------------
    def world_config(self) -> WorldConfig:
        """World settings of the next mission."""
        o = self.options
        return WorldConfig(
            slip_p=self.slip_p,
            sensor_radius=o.sensor_radius,
            aftershock=AftershockConfig(q=o.aftershock_q, k=o.aftershock_k),
        )

    def reset_mission(self) -> None:
        """Start a fresh mission on :attr:`base_grid` with the current seed / p / agent."""
        cfg = self.world_config()
        grid = self.base_grid
        agent = make_agent(
            self.agent_name,
            make_transition(grid, cfg),
            PlannerConfig(),
            AgentConfig(replan_mode=self.options.replan_mode),  # type: ignore[arg-type]
        )
        self.mission = Mission(grid, agent, cfg, seed=self.seed, scenario=self.scenario_name)
        self._serial += 1
        self.trace = []
        self.log.clear()
        self._accumulator = 0.0
        self._flash_cells = ()
        self._run_saved = False
        self.tree.set_plan(None, None)
        self.log.add(
            f"{self.agent_name} on {self.scenario_name} {grid.rows}x{grid.cols}, "
            f"seed {self.seed}, p = {self.slip_p:.2f}",
            "info",
        )
        self._on_event(self.mission.start())

    def _on_event(self, event: StepEvent) -> None:
        """Render-side bookkeeping for a new event: trace, log, flash, end of mission."""
        mark = TraceMark.from_event(event)
        if mark is not None:
            self.trace.append(mark)
        for text, kind in event_messages(event):
            self.log.add(text, kind)
        self._flash(flash_cells(event))
        if self.mission.done:
            self.paused = True
            if self.options.save_run and not self._run_saved:
                self._run_saved = True
                path = save_run(self.mission, runs_dir=self.options.runs_dir)
                self.saved_runs.append(path)
                self.log.add(f"Run saved to {path}", "info")

    def _can_advance(self) -> bool:
        return not self.editor and not self.mission.done

    def _advance(self) -> bool:
        if not self._can_advance():
            return False
        self._on_event(self.mission.step())
        return True

    def step_once(self) -> bool:
        """``N``: pause and take exactly one step; False if the mission is over."""
        self.paused = True
        if self.mission.done:
            self.log.add("Mission over: R resets, S tries a new seed", "info")
            return False
        return self._advance()

    def toggle_play(self) -> None:
        """``Space``: play / pause."""
        if self.mission.done:
            self.paused = True
            self.log.add("Mission over: R resets, S tries a new seed", "info")
            return
        self.paused = not self.paused

    def new_seed(self) -> None:
        """``S``: seed + 1 and a fresh mission. A generated (mapgen) scenario is regenerated
        with the new seed, like ``--seed`` does; a map file (or an edited map) stays."""
        self.seed += 1
        entry = self.scenarios[self.scenario_index]
        if entry.generator is not None and self.base_grid.name != EDITED_GRID_NAME:
            try:
                self.base_grid = self._load_grid(entry)
            except (OSError, ValueError, RuntimeError) as exc:
                self.log.add(f"Cannot regenerate {entry.name}: {exc}", "warn")
        self.reset_mission()

    def change_slip(self, delta: float) -> None:
        """``[`` / ``]``: change p by ``delta`` within [0, MAX_SLIP_P] and reset the mission."""
        p = round(min(MAX_SLIP_P, max(0.0, self.slip_p + delta)), SLIP_P_DECIMALS)
        if p == self.slip_p:
            self.log.add(f"Slip p stays at {p:.2f} (range 0 – {MAX_SLIP_P:g})", "info")
            return
        self.slip_p = p
        self.reset_mission()

    def select_agent(self, index: int) -> None:
        """``1``/``2``/``3``: switch agent and reset the mission."""
        if 0 <= index < len(AGENT_NAMES):
            self.agent_name = AGENT_NAMES[index]
            self.reset_mission()

    # --- map editor ------------------------------------------------------------------------
    def toggle_editor(self) -> None:
        """``E``: enter the editor (pauses, shows the mission's initial true map - without the
        debris aftershocks added since) or leave it (resets the mission on the edited map; an
        unchanged map keeps its name and the same mission is replayed)."""
        if not self.editor:
            self.editor = True
            self.paused = True
            self.edit_grid = self.base_grid.copy()
            self._edit_slip = self.base_grid.high_slip.copy()
            self.log.add("Map editor: click debris, Shift+click start, Ctrl+click survivor; "
                         "E to finish", "info")
            return
        self.editor = False
        grid = self.edit_grid if self.edit_grid is not None else self.base_grid
        self.edit_grid = None
        self._edit_slip = None
        if grid != self.base_grid:
            grid = grid.copy()
            grid.name = EDITED_GRID_NAME
            self.base_grid = grid
        self.reset_mission()
        if self.mission.shortest_path is None:
            self.log.add("Edited map is unsolvable: the survivor cannot be reached", "warn")

    def editor_click(self, pos: Pos, mods: int = 0) -> bool:
        """Apply an editor click on cell ``pos`` (Shift = start, Ctrl = survivor, plain =
        toggle debris); True if the map changed."""
        grid = self.edit_grid
        if not self.editor or grid is None or not grid.in_bounds(pos):
            return False
        cells, slip = grid.cells.copy(), grid.high_slip.copy()
        start, survivor = grid.start, grid.survivor
        if mods & pygame.KMOD_CTRL:
            if pos in (start, survivor):
                return False
            cells[survivor] = Cell.OPEN
            survivor = pos
            cells[pos] = Cell.SURVIVOR
        elif mods & pygame.KMOD_SHIFT:
            if pos in (start, survivor):
                return False
            start = pos
            cells[pos] = Cell.OPEN
        else:
            if pos in (start, survivor):
                self.log.add(f"No debris on the start / survivor cell {fmt_pos(pos)}", "warn")
                return False
            cells[pos] = Cell.OPEN if cells[pos] == Cell.DEBRIS else Cell.DEBRIS
        # '~' lives only on open cells other than start / survivor (TrueGrid invariants);
        # an open cell gets its original flag back (toggling debris twice keeps a '~' cell).
        original = self._edit_slip if self._edit_slip is not None else slip
        slip = np.asarray(original, dtype=bool) & (cells == Cell.OPEN)
        slip[start] = slip[survivor] = False
        self.edit_grid = TrueGrid(cells=cells, start=start, survivor=survivor,
                                  high_slip=slip, name=grid.name)
        return True

    def save_map(self) -> Path:
        """Ctrl+S: write the map to ``maps/edited_<timestamp>.txt``: the map being edited, else
        the mission's initial true map (aftershock debris is part of the run, not the map)."""
        grid = self.edit_grid if self.editor and self.edit_grid else self.mission.grid
        path = unique_path(self.maps_dir / f"{EDITED_MAP_PREFIX}{timestamp()}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(path)
        if not any(e.path == path for e in self.scenarios):
            self.scenarios.append(ScenarioEntry(path.stem, path=path))
        self.log.add(f"Map saved to {path}", "success")
        return path

    # --- input -----------------------------------------------------------------------------
    def handle_key(self, key: int, mods: int = 0) -> bool:
        """Every simulator control (brief §6.1 plus ``C`` and ``Esc``); True if handled."""
        if key == pygame.K_s and mods & pygame.KMOD_CTRL:
            self.save_map()
            return True
        if super().handle_key(key, mods):
            return True
        if key == pygame.K_e:
            self.toggle_editor()
            return True
        mission_keys = {pygame.K_SPACE, pygame.K_n, pygame.K_r, pygame.K_s, pygame.K_m,
                        pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, *_AGENT_KEYS}
        if self.editor:
            if key in mission_keys:
                self.log.add("Leave the map editor (E) first", "info")
                return True
            return False
        if key == pygame.K_SPACE:
            self.toggle_play()
        elif key == pygame.K_n:
            self.step_once()
        elif key == pygame.K_r:
            self.reset_mission()
        elif key == pygame.K_s:
            self.new_seed()
        elif key == pygame.K_LEFTBRACKET:
            self.change_slip(-theme.SLIP_P_STEP)
        elif key == pygame.K_RIGHTBRACKET:
            self.change_slip(theme.SLIP_P_STEP)
        elif key in _AGENT_KEYS:
            self.select_agent(_AGENT_KEYS[key])
        elif key == pygame.K_m:
            self.next_scenario()
        else:
            return False
        return True

    def _handle_mouse(self, event: pygame.event.Event) -> bool:
        if not self.editor or event.type != pygame.MOUSEBUTTONDOWN:
            return False
        if event.button != pygame.BUTTON_LEFT:
            return False
        pos = self.grid_view.pos_at(event.pos)
        if pos is None:
            return False
        self.editor_click(pos, pygame.key.get_mods())
        return True

    # --- rendering -------------------------------------------------------------------------
    def _current_snapshot(self) -> Any | None:
        agent = self.mission.agent
        pid = agent.current_plan_id
        return agent.snapshots[pid] if 0 <= pid < len(agent.snapshots) else None

    def _policy(self) -> dict[Pos, Action]:
        agent = self.mission.agent
        key = (self._serial, len(self.mission.events), agent.current_plan_id)
        if key != self._arrows_key:
            self._arrows_key, self._arrows = key, agent.policy_arrows()
        return self._arrows

    def _scene(self) -> GridScene:
        if self.editor and self.edit_grid is not None:
            g = self.edit_grid
            return GridScene(rows=g.rows, cols=g.cols, belief=None, true_cells=g.cells,
                             high_slip=g.high_slip, robot=None, start=g.start,
                             survivor=g.survivor, editor=True)
        mission = self.mission
        world, agent = mission.world, mission.agent
        grid = world.grid
        snap = self._current_snapshot()
        return GridScene(
            rows=grid.rows,
            cols=grid.cols,
            belief=agent.belief.grid,
            true_cells=grid.cells,
            high_slip=grid.high_slip,
            robot=world.pos,
            start=grid.start,
            survivor=grid.survivor,
            show_true=self.show_true,
            show_policy=self.show_policy,
            policy=self._policy() if self.show_policy else {},
            envelope=snap.envelope if snap is not None else frozenset(),
            trace=self.trace,
            flash_cells=self._flash_cells,
            trace_id=self._serial,
        )

    def _update_tree(self) -> None:
        if self.editor:
            self.tree.set_plan(None, None, theme.PANEL_TREE_DEPTH)
            self.tree.set_path(None)
            self.tree.set_message(EDITOR_TREE_TEXT)
            return
        snap: PlanSnapshot | None = self._current_snapshot()
        pos = self.mission.world.pos
        pid = self.mission.agent.current_plan_id
        if snap is not None and snap.plan is not None:
            self.tree.set_plan(snap.plan, pos, theme.PANEL_TREE_DEPTH, plan_id=pid)
        else:
            self.tree.set_plan(None, pos, theme.PANEL_TREE_DEPTH)
            self.tree.set_path(snap.path if snap is not None else None)
            self.tree.set_message(no_plan_message(
                snap.pos if snap is not None else None,
                snap is not None and snap.path is not None,
            ))

    def _metric_values(self) -> dict[str, str]:
        key = (self._serial, len(self.mission.events))
        if key != self._metrics_key:
            m = self.mission.metrics()
            self._metrics_key = key
            self._metrics = {
                STEPS_LABEL: f"{m.steps}/{self.mission.budget}",
                "Replans": str(m.replans),
                "Slips": str(m.slips),
                "Skids": str(m.skids),
                "Explored": str(m.cells_explored),
                "Aftershocks": str(m.aftershock_events),
                "Plan ms": f"{m.planning_ms:.1f}",
                "Plan size": str(m.plan_size),
                "Unplanned": str(m.unplanned_states),
                "Bumps": str(m.bumps),
                "Plan #": str(self.mission.agent.current_plan_id),
            }
        return self._metrics

    def _panel_info(self) -> PanelInfo:
        return PanelInfo(
            title=EDITOR_TITLE if self.editor else SIM_TITLE,
            agent=self.agent_name,
            status=self.mission.status,
            seed=self.seed,
            slip_p=self.slip_p,
            scenario=self.scenario_name,
            # the metrics describe the mission on the previous map: hidden while editing
            metrics={} if self.editor else self._metric_values(),
            speed=self.speed,
            paused=self.paused or self.mission.done or self.editor,
            recording=self.recorder.active,
            hint=EDITOR_HINT if self.editor else DEFAULT_HINT,
            editing=self.editor,
        )


__all__ = [
    "BaseApp",
    "ScenarioEntry",
    "SimApp",
    "SimOptions",
    "discover_scenarios",
    "event_messages",
    "flash_cells",
    "help_lines",
    "is_flash_event",
    "no_plan_message",
    "prepare_environment",
]
