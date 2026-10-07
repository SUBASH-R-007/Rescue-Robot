"""The extensions in the interactive simulator (``python -m rescue.sim --ext …``, plan §3.15.3).

:class:`ExtSimApp` is the simulator (:class:`~rescue.viz.sim_app.SimApp`) driving a
:class:`~rescue.ext.mission.ExtMission` with a :class:`~rescue.ext.agents.SearchAgent`
(``search_andor`` / ``search_astar``) instead of a core ``Mission``. Three modes, the same as
``python -m rescue.ext.demo`` (brief §8.3–§8.5), cycled with ``X``:

* ``search`` — one survivor at an unknown location: the robot explores frontier cells; the
  survivor stays under the fog until it is sensed (or ``T`` shows the true map);
* ``multi`` — ``k`` known survivors (default ``EXT_DEMO_MULTI_K``), rescued in greedy
  nearest-first order (:func:`rescue.ext.scenarios.multi_survivor_grid`);
* ``noisy`` — one known survivor, debris missed with probability ε (default
  ``EXT_DEMO_NOISE``); a false-positive rate is passed only when this build's
  :class:`~rescue.ext.sensing.NoisySensor` and :class:`~rescue.ext.mission.ExtMission` accept
  one (checked at run time, :func:`false_positive_supported`). Observed open cells are shaded by
  their P(debris); with false positives, believed debris below P = 1 (a possible phantom) is
  drawn lighter with a '?' badge.

``--k`` / ``--noise`` / ``--fp`` override the mode's defaults in every mode, as in the demo. With
the same scenario, size, seed, p and agent a mission is exactly the demo's (same map, outcome
stream and noise draws). The map editor (``E``, Ctrl+S) is disabled; ``M`` cycles the generated
scenarios; ``1`` / ``2`` select the agent. Like the core simulator the app holds no mission
logic: every step is ``ExtMission.step()`` and every log line comes from the
:class:`~rescue.ext.mission.ExtStepEvent` (:func:`ext_event_messages`).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pygame

from rescue.config import (
    EXT_DEMO_SCENARIO,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.ext.agents import EXT_AGENT_NAMES, make_ext_agent
from rescue.ext.demo import MODES
from rescue.ext.mission import ExtMission, ExtStepEvent
from rescue.ext.scenarios import add_survivors, multi_survivor_grid
from rescue.ext.sensing import NoisySensor, ProbBelief
from rescue.grid import Pos, TrueGrid, fmt_pos
from rescue.model import OutcomeKind, kind_direction, step_pos
from rescue.runner import FAILURE_REASONS, MissionStatus, make_transition
from rescue.viz import theme
from rescue.viz.grid_view import GridScene, TraceMark, prob_legend_swatches
from rescue.viz.panel import Legend, PanelInfo
from rescue.viz.record import in_browser
from rescue.viz.sim_app import (
    BROWSER_HINT,
    STEPS_LABEL,
    BaseApp,
    ScenarioEntry,
    SimApp,
    SimOptions,
    default_maps_dir,
    event_messages,
    flash_cells,
    generator_names,
    help_lines,
    resolve_path,
)

#: Mode cycle of ``X`` (the demo's modes, in this order).
EXT_MODES: tuple[str, ...] = ("search", "multi", "noisy")
#: Agent at start-up.
DEFAULT_EXT_AGENT: str = EXT_AGENT_NAMES[0]
#: Panel title (``{mode}`` = the current mode) and footer hint.
EXT_TITLE = "Extensions: {mode}"
EXT_HINT = "H help · X mode · Space play/pause · N step · Esc quit"
#: Log line for the disabled map editor (``E``, Ctrl+S).
EDITOR_DISABLED_TEXT = "Map editor is disabled in extension mode"
#: Log line when a false-positive rate is asked for but this build's sensor has none.
FP_UNSUPPORTED_TEXT = "False positives are not supported by this build's sensor: --fp ignored"
#: Log line with a false-positive rate: how possible phantom debris is drawn.
PHANTOM_HINT_TEXT = "Debris drawn lighter with a '?' badge may be a phantom (P(debris) < 1)"
#: Plan-tree texts without a plan.
NO_TARGET_TEXT = "No target: no reachable survivor or frontier cell from {pos}"
NO_EXT_PLAN_TEXT = "No plan: the target {target} is unreachable from {pos} on the robot's map"
#: Panel metric labels.
MODE_LABEL = "Mode"
RESCUED_LABEL = "Rescued"
FIRST_FOUND_LABEL = "First found"
FALLBACKS_LABEL = "Fallbacks"
AFTERSHOCKS_LABEL = "Aftershocks"
NOISE_LABEL = "Noise ε"
#: Value shown for a metric that has none yet (e.g. no survivor found).
NO_VALUE = "–"
#: Help-overlay items replaced / added in extension mode.
EXT_HELP_AGENTS = ("1 2", " / ".join(EXT_AGENT_NAMES))
EXT_HELP_MODE = ("X", "mode: " + " → ".join(EXT_MODES))
EXT_HELP_SCENARIO = ("M", "next generated scenario")
#: Suffix of map files (a ``--scenario`` that is no generator may name one in ``maps/``).
MAP_SUFFIX = ".txt"
#: Name of the ``NoisySensor`` / ``ExtMission`` keyword of a false-positive rate.
FALSE_POSITIVE_ARG = "false_positive"


def false_positive_supported() -> bool:
    """True if this build's :class:`NoisySensor` and :class:`ExtMission` both take a
    ``false_positive`` rate (the two-sided noise track); the UI degrades gracefully otherwise."""
    return all(
        FALSE_POSITIVE_ARG in inspect.signature(fn).parameters
        for fn in (NoisySensor.__init__, ExtMission.__init__)
    )


@dataclass(frozen=True)
class ExtModeSettings:
    """What one mode runs: the demo mode's defaults with the CLI overrides applied."""

    mode: str
    survivors_known: bool
    k: int
    false_negative: float
    false_positive: float


def mode_settings(mode: str, k: int | None = None, noise: float | None = None,
                  false_positive: float | None = None) -> ExtModeSettings:
    """Settings of ``mode`` (:data:`rescue.ext.demo.MODES`) with ``k`` / ``noise`` /
    ``false_positive`` overriding its defaults (None keeps them; the default false-positive
    rate is 0). ``ValueError`` for an unknown mode or an out-of-range value."""
    if mode not in MODES:
        raise ValueError(f"unknown extension mode {mode!r}; choose from {', '.join(EXT_MODES)}")
    spec = MODES[mode]
    kk = spec.k if k is None else int(k)
    eps = spec.noise if noise is None else float(noise)
    fp = 0.0 if false_positive is None else float(false_positive)
    if kk < 1:
        raise ValueError(f"k must be >= 1, got {kk}")
    for name, value in (("noise", eps), ("false positive rate", fp)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    return ExtModeSettings(mode, spec.survivors_known, kk, eps, fp)


def ext_event_messages(
    event: ExtStepEvent, *, previous_target: Pos | None, rescued_count: int, k: int,
) -> list[tuple[str, str]]:
    """Event-log lines ``(text, kind)`` of one :class:`ExtStepEvent`.

    The core lines (slip, skid, blocked move, aftershock, contradiction / replan, unplanned
    state; :func:`rescue.viz.sim_app.event_messages`) with the bump worded for the extension
    ("Bump into unseen debris at (r,c)"), then "Survivor found at (r,c)" per sighting,
    "Rescued (r,c) — i/k" (``rescued_count`` = rescues so far, this one included), the new
    target when it changed from ``previous_target`` ("Exploring frontier (r,c)" / "Heading to
    survivor (r,c)"), and the end of the mission last.
    """
    running = dataclasses.replace(event, bump=False, status=MissionStatus.RUNNING)
    lines = event_messages(running)  # type: ignore[arg-type]  # duck-typed StepEvent
    if event.bump and event.action is not None and event.outcome is not None:
        direction = kind_direction(event.outcome, event.action) or event.action
        bump = (f"Bump into unseen debris at {fmt_pos(step_pos(event.pos, direction))}", "warn")
        if event.outcome is OutcomeKind.ARRIVED and lines:
            lines[0] = bump  # replaces the core's "Move … blocked" line
        else:
            lines.insert(min(1, len(lines)), bump)  # after the skid line
    lines.extend((f"Survivor found at {fmt_pos(p)}", "success") for p in event.found_now)
    if event.rescued_now is not None:
        lines.append((f"Rescued {fmt_pos(event.rescued_now)} — {rescued_count}/{k}", "success"))
    if (event.status is MissionStatus.RUNNING and event.target is not None
            and event.target != previous_target):
        if event.mode == "explore":
            lines.append((f"Exploring frontier {fmt_pos(event.target)}", "info"))
        else:
            lines.append((f"Heading to survivor {fmt_pos(event.target)}", "info"))
    if event.status is MissionStatus.RESCUED:
        text = f"All {k} survivors rescued" if k > 1 else "Rescue successful"
        lines.append((text, "success"))
    elif event.status is not MissionStatus.RUNNING:
        reason = FAILURE_REASONS.get(event.status, event.status.name.lower())
        lines.append((f"Mission failed ({reason})", "failure"))
        if event.action is None and event.note:
            lines.append((event.note, "info"))
    return lines


@dataclass(frozen=True)
class ExtSimOptions(SimOptions):
    """:class:`SimOptions` of the extension simulator: the ext demo's map, size, seed and agent
    by default, plus the mode and its overrides (None = the mode's default, see
    :func:`mode_settings`). ``save_run`` is not supported (runs files hold core missions)."""

    scenario: str | None = EXT_DEMO_SCENARIO
    rows: int = EXT_DEMO_SIZE[0]
    cols: int = EXT_DEMO_SIZE[1]
    seed: int = EXT_DEMO_SEED
    agent: str = DEFAULT_EXT_AGENT
    mode: str = EXT_MODES[0]
    k: int | None = None
    noise: float | None = None
    false_positive: float | None = None


class ExtSimApp(SimApp):
    """The simulator for the extension missions; see the module docstring."""

    caption = theme.EXT_SIM_CAPTION

    def __init__(self, options: ExtSimOptions | None = None) -> None:
        opts = options or ExtSimOptions()
        if opts.agent not in EXT_AGENT_NAMES:
            raise ValueError(f"unknown extension agent {opts.agent!r}; choose from "
                             f"{', '.join(EXT_AGENT_NAMES)}")
        if opts.save_run:
            raise ValueError("saving runs is not supported in extension mode")
        self.options = opts
        self.fp_supported = false_positive_supported()
        self.settings = mode_settings(opts.mode, opts.k, opts.noise, opts.false_positive)
        self.maps_dir = Path(opts.maps_dir) if opts.maps_dir else default_maps_dir()
        self.scenarios = [ScenarioEntry(n, generator=n) for n in generator_names()]
        self.scenario_index = self._ext_initial_scenario(opts)
        self.seed = int(opts.seed)
        self.slip_p = float(opts.slip_p)
        self.agent_name = opts.agent
        self.base_grid = self._load_grid(self.scenarios[self.scenario_index])
        WorldConfig(slip_p=self.slip_p, sensor_radius=opts.sensor_radius)  # validate early
        BaseApp.__init__(
            self,
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
        # SimApp state the inherited methods use (the editor stays off).
        self.editor = False
        self.edit_grid = None
        self._edit_slip = None
        self.trace = []
        self.saved_runs = []
        self._serial = 0
        self._metrics_key = None
        self._metrics = {}
        self._arrows_key = None
        self._arrows = {}
        self._legend: tuple[int, Legend] | None = None
        self._last_target: Pos | None = None
        self._rescued_count = 0
        self.mission: ExtMission  # type: ignore[assignment]
        self.reset_mission()

    # --- scenarios -------------------------------------------------------------------------
    def _ext_initial_scenario(self, opts: ExtSimOptions) -> int:
        """Index of the starting entry: ``map_path``, else ``scenario`` (a generator or a map
        in ``maps_dir``), inserted when it is not a generator."""
        if opts.map_path is not None:
            path = resolve_path(opts.map_path)
            if not path.is_file():
                raise FileNotFoundError(f"map file not found: {opts.map_path}")
            self.scenarios.insert(0, ScenarioEntry(path.stem, path=path))
            return 0
        name = opts.scenario or EXT_DEMO_SCENARIO
        for i, entry in enumerate(self.scenarios):
            if entry.name == name:
                return i
        path = self.maps_dir / f"{name}{MAP_SUFFIX}"
        if path.is_file():
            self.scenarios.insert(0, ScenarioEntry(name, path=path))
            return 0
        available = ", ".join(e.name for e in self.scenarios) or "none"
        raise ValueError(f"unknown scenario {name!r} (generated: {available})")

    def _load_grid(self, entry: ScenarioEntry) -> TrueGrid:
        """The map of ``entry`` with the mode's ``k`` survivors (extra survivors are placed by
        :func:`~rescue.ext.scenarios.add_survivors`, seeded by the current seed)."""
        k = self.settings.k
        if entry.generator is not None:
            return multi_survivor_grid(entry.generator, self.options.rows, self.options.cols,
                                       self.seed, k)
        if entry.path is None:  # pragma: no cover - entries have a path or a generator
            raise ValueError(f"scenario {entry.name!r} has no map")
        grid = TrueGrid.load(entry.path)
        return add_survivors(grid, self.seed, k) if k > len(grid.survivors) else grid

    def _reload_grid(self) -> None:
        entry = self.scenarios[self.scenario_index]
        try:
            self.base_grid = self._load_grid(entry)
        except (OSError, ValueError, RuntimeError) as exc:
            self.log.add(f"Cannot load {entry.name}: {exc}", "warn")

    # --- mission ---------------------------------------------------------------------------
    def mission_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of the next :class:`ExtMission` (mode settings)."""
        s = self.settings
        kwargs: dict[str, Any] = {
            "survivors_known": s.survivors_known,
            "false_negative": s.false_negative,
            "scenario": self.scenario_name,
            "mode": s.mode,
        }
        if s.false_positive > 0.0 and self.fp_supported:
            kwargs[FALSE_POSITIVE_ARG] = s.false_positive
        return kwargs

    def reset_mission(self) -> None:
        """Start a fresh extension mission on :attr:`base_grid` (current seed / p / agent /
        mode)."""
        cfg = self.world_config()
        grid = self.base_grid
        agent = make_ext_agent(
            self.agent_name,
            make_transition(grid, cfg),
            PlannerConfig(),
            AgentConfig(replan_mode=self.options.replan_mode),  # type: ignore[arg-type]
        )
        self.mission = ExtMission(grid, agent, cfg, self.seed, **self.mission_kwargs())
        self._serial += 1
        self.trace = []
        self.log.clear()
        self._accumulator = 0.0
        self._flash_cells = ()
        self._run_saved = True  # runs files hold core missions only
        self._last_target = None
        self._rescued_count = 0
        self.tree.set_plan(None, None)
        s = self.settings
        where = "known" if s.survivors_known else "unknown"
        noise = f", ε = {s.false_negative:g}" if s.false_negative > 0 else ""
        if s.false_positive > 0 and self.fp_supported:
            noise += f", fp = {s.false_positive:g}"
        self.log.add(
            f"{self.agent_name} · {s.mode} on {self.scenario_name} {grid.rows}x{grid.cols}, "
            f"seed {self.seed}, p = {self.slip_p:.2f}, k = {s.k} ({where}){noise}",
            "info",
        )
        if s.false_positive > 0 and not self.fp_supported:
            self.log.add(FP_UNSUPPORTED_TEXT, "warn")
        elif s.false_positive > 0:
            self.log.add(PHANTOM_HINT_TEXT, "info")
        self._on_event(self.mission.start())

    def _on_event(self, event: ExtStepEvent) -> None:  # type: ignore[override]
        """Trace, log lines, flash and end of mission for a new extension event."""
        mark = TraceMark.from_event(event)  # type: ignore[arg-type]  # duck-typed StepEvent
        if mark is not None:
            self.trace.append(mark)
        if event.rescued_now is not None:
            self._rescued_count += 1
        for text, kind in ext_event_messages(event, previous_target=self._last_target,
                                             rescued_count=self._rescued_count,
                                             k=self.mission.k):
            self.log.add(text, kind)
        self._last_target = event.target
        self._flash(flash_cells(event))  # type: ignore[arg-type]
        if self.mission.done:
            self.paused = True

    def new_seed(self) -> None:
        """``S``: seed + 1; the map (generated, or the extra survivors of a map file) is
        rebuilt with the new seed."""
        self.seed += 1
        self._reload_grid()
        self.reset_mission()

    def select_agent(self, index: int) -> None:
        """``1`` / ``2``: ``search_andor`` / ``search_astar`` (a higher number is ignored)."""
        if 0 <= index < len(EXT_AGENT_NAMES):
            self.agent_name = EXT_AGENT_NAMES[index]
            self.reset_mission()
        else:
            self.log.add(f"Extension agents: {EXT_HELP_AGENTS[1]} (keys {EXT_HELP_AGENTS[0]})",
                         "info")

    def set_mode(self, mode: str) -> None:
        """Switch to ``mode`` (same CLI overrides), rebuild the map and reset the mission."""
        o = self.options
        self.settings = mode_settings(mode, o.k, o.noise, o.false_positive)
        self._reload_grid()
        self.reset_mission()

    def cycle_mode(self) -> None:
        """``X``: the next mode of :data:`EXT_MODES`."""
        i = EXT_MODES.index(self.settings.mode)
        self.set_mode(EXT_MODES[(i + 1) % len(EXT_MODES)])

    # --- input -----------------------------------------------------------------------------
    def toggle_editor(self) -> None:
        """``E``: disabled in extension mode (logs a line)."""
        self.log.add(EDITOR_DISABLED_TEXT, "info")

    def save_map(self) -> Path:  # type: ignore[override]
        """Ctrl+S: disabled in extension mode (logs a line)."""
        self.log.add(EDITOR_DISABLED_TEXT, "info")
        return Path()

    def handle_key(self, key: int, mods: int = 0) -> bool:
        """The simulator's keys, plus ``X`` (mode); ``1``/``2`` pick the extension agent (the
        inherited dispatch calls the overridden :meth:`select_agent`). Like every mission key of
        the simulator, ``X`` also works while the help overlay is open."""
        if key == pygame.K_x:
            self.cycle_mode()
            return True
        return super().handle_key(key, mods)

    def _handle_mouse(self, event: pygame.event.Event) -> bool:
        return False

    def _help_lines(self) -> list[tuple[str, str] | str]:
        """The simulator's help without the map editor, with ``X`` and the extension agents."""
        items: list[tuple[str, str] | str] = []
        in_editor = False
        for item in help_lines(True):
            if isinstance(item, str):
                in_editor = item == "Map editor"
                if in_editor:
                    continue
            elif in_editor:
                continue
            elif item[0] == "1 2 3":
                items.extend((EXT_HELP_AGENTS, EXT_HELP_MODE))
                continue
            elif item[0] == "M":
                item = EXT_HELP_SCENARIO
            items.append(item)
        return items

    # --- rendering -------------------------------------------------------------------------
    def debris_prob(self) -> Any | None:
        """The belief's P(debris) array with noisy sensing (a ``ProbBelief``), else None."""
        belief = self.mission.agent.belief
        return belief.p_debris if isinstance(belief, ProbBelief) else None

    def _scene(self) -> GridScene:
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
            survivors=tuple(grid.extra_survivors),
            rescued=frozenset(mission.rescued),
            target=None if mission.done else agent.target,
            debris_prob=self.debris_prob(),
        )

    def _update_tree(self) -> None:
        agent = self.mission.agent
        pos = self.mission.world.pos
        snap = self._current_snapshot()
        if snap is not None and snap.plan is not None and agent.target is not None:
            self.tree.set_plan(snap.plan, pos, theme.PANEL_TREE_DEPTH,
                               plan_id=agent.current_plan_id)
            return
        self.tree.set_plan(None, pos, theme.PANEL_TREE_DEPTH)
        if agent.target is None:
            self.tree.set_path(None)
            self.tree.set_message(NO_TARGET_TEXT.format(pos=fmt_pos(pos)))
            return
        self.tree.set_path(snap.path if snap is not None else None)
        if snap is None or snap.path is not None:
            self.tree.set_message(None)
        else:
            self.tree.set_message(NO_EXT_PLAN_TEXT.format(target=fmt_pos(agent.target),
                                                          pos=fmt_pos(snap.pos)))

    def _metric_values(self) -> dict[str, str]:
        key = (self._serial, len(self.mission.events))
        if key != self._metrics_key:
            mission, agent = self.mission, self.mission.agent
            m = mission.metrics()
            first = m.first_detection_step
            noise = f"{m.false_negative:g}"
            if self.settings.false_positive > 0 and self.fp_supported:
                noise += f" / fp {self.settings.false_positive:g}"
            self._metrics_key = key
            self._metrics = {
                STEPS_LABEL: f"{m.steps}/{mission.budget}",
                MODE_LABEL: "done" if mission.done else agent.mode,
                RESCUED_LABEL: f"{m.rescued}/{m.k}",
                FIRST_FOUND_LABEL: NO_VALUE if math.isnan(first) else f"t = {int(first)}",
                "Replans": str(m.replans),
                "Slips": str(m.slips),
                "Skids": str(m.skids),
                "Bumps": str(m.bumps),
                "Explored": str(m.cells_explored),
                "Unplanned": str(m.unplanned_states),
                # Relaxed-passability fallbacks (SearchAgent.fallback_replans, plan §3.15.2):
                # the fallback --fp triggers, and the raw.csv column of the same name.
                FALLBACKS_LABEL: str(m.fallback_replans),
                "Plan ms": f"{m.planning_ms:.1f}",
                "Plan #": str(agent.current_plan_id),
                NOISE_LABEL: noise,
            }
            if self.options.aftershock_q > 0:
                self._metrics[AFTERSHOCKS_LABEL] = str(m.aftershock_events)
        return self._metrics

    def _prob_legend(self) -> Legend | None:
        """The P(debris) legend while the belief is probabilistic (cached per palette)."""
        if self.debris_prob() is None:
            return None
        pal = self.palette
        if self._legend is None or self._legend[0] != id(pal):
            legend = Legend(
                title=theme.PROB_LEGEND_TITLE,
                swatches=prob_legend_swatches(pal),
                low=f"{theme.PROB_SHADE_MIN:g}",
                high=f"{theme.PROB_SHADE_MAX:g}",
            )
            self._legend = (id(pal), legend)
        return self._legend[1]

    def _panel_info(self) -> PanelInfo:
        return PanelInfo(
            title=EXT_TITLE.format(mode=self.settings.mode),
            agent=self.agent_name,
            status=self.mission.status,
            seed=self.seed,
            slip_p=self.slip_p,
            scenario=self.scenario_name,
            metrics=self._metric_values(),
            speed=self.speed,
            paused=self.paused or self.mission.done,
            recording=self.recorder.active,
            hint=BROWSER_HINT if in_browser() else EXT_HINT,
            legend=self._prob_legend(),
        )


__all__ = [
    "DEFAULT_EXT_AGENT",
    "EXT_MODES",
    "ExtModeSettings",
    "ExtSimApp",
    "ExtSimOptions",
    "ext_event_messages",
    "false_positive_supported",
    "mode_settings",
]
