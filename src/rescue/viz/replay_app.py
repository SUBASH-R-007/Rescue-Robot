"""Replay a saved run file with scrubbing (``python -m rescue.replay``, brief §6.5).

:class:`ReplayApp` never re-simulates: the state shown at step ``k`` is rebuilt from the
recorded :class:`~rescue.runner.StepEvent`s ``events[0..k]`` of a
:class:`~rescue.runs.RunRecord`:

* belief = all unknown + the survivor, then each event's ``new_cells`` and ``contradictions``
  (sensed debris on cells believed open, e.g. after an aftershock);
* true map = the initial map plus every ``aftershock_cells`` so far;
* robot = ``events[k].new_pos``; trace marks of the moves so far;
* plan = the snapshot whose ``plan_id`` is ``events[k].plan_id`` (the stored plan tree for
  contingency agents, the stored path for A* agents), policy arrows from its ``policy`` / the
  part of the path still ahead (the path cursor advances once per move since the snapshot,
  exactly as :class:`~rescue.agents.base.PathAgent` does).

The metrics block uses the simulator's labels in the simulator's order; the run-total planning
time ("Plan ms (all)") is the only value not reconstructed per step.

Controls: ``←/→`` one step, ``Shift+←/→`` ten steps, ``Home/End``, ``Space`` play/pause,
``+/-`` speed, a click on the progress bar jumps there, plus ``T P C G H Esc`` as in the
simulator.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from rescue.agents.base import direction
from rescue.grid import Cell, Pos
from rescue.model import Action, OutcomeKind
from rescue.plan import TRUNCATED_MARK
from rescue.runner import StepEvent
from rescue.runs import RunRecord
from rescue.viz import theme
from rescue.viz.grid_view import GridScene, TraceMark, trace_from_events
from rescue.viz.panel import PanelInfo
from rescue.viz.sim_app import BaseApp, event_messages, flash_cells, no_plan_message
from rescue.viz.tree_view import text_cache

log = logging.getLogger(__name__)

#: Panel title, footer hint and progress-bar label.
REPLAY_TITLE = "Replay"
REPLAY_HINT = "←/→ step · Shift ×10 · Home/End · Space play · H help · Esc quit"
PROGRESS_TEXT = "step {k} / {n}   t = {t}"
#: Label of the run-total planning time (per-step planning times are not stored).
PLAN_MS_ALL_LABEL = "Plan ms (all)"
#: Label of the step counter: the shown step / the run length (the simulator shows the budget).
REPLAY_STEPS_LABEL = "Steps / run"


def _pos(raw: Any) -> Pos:
    return (int(raw[0]), int(raw[1]))


def _snapshot_tree(node: Mapping[str, Any], depth: int, act_level: int = 0) -> dict[str, Any]:
    """A run-file plan-tree dict as :meth:`rescue.plan.Plan.to_tree` would return it at
    ``depth``: positions as tuples, Act nodes deeper than ``depth`` truncated."""
    state = node.get("state")
    out: dict[str, Any] = dict(node)
    out["state"] = None if state is None else _pos(state)
    out["children"] = []
    if node.get("kind") == "act":
        if act_level + 1 > depth:
            out["kind"] = "truncated"
            out["text"] = f"{node.get('text', '')} {TRUNCATED_MARK}"
            return out
        out["children"] = [
            _snapshot_tree(child, depth, act_level + 1) for child in node.get("children", ())
        ]
    return out


class SnapshotPlan:
    """Adapter giving a stored plan-tree dict the ``to_tree(depth)`` / ``subtree(state,
    depth)`` methods :class:`~rescue.viz.tree_view.TreeView` uses (no
    :class:`~rescue.plan.Plan` needed). ``goal`` is the survivor cell."""

    def __init__(self, tree: Mapping[str, Any], goal: Pos | None = None) -> None:
        self._tree = tree
        self.goal = goal
        self._cache: dict[int, dict[str, Any]] = {}

    def to_tree(self, depth: int) -> dict[str, Any]:
        """The stored tree limited to ``depth`` Act levels (cached per depth)."""
        tree = self._cache.get(depth)
        if tree is None:
            tree = self._cache[depth] = _snapshot_tree(self._tree, depth)
        return tree

    def subtree(self, state: Pos, depth: int) -> dict[str, Any] | None:
        """The stored subtree of ``state``'s Act node (shallowest first) limited to ``depth``
        Act levels, or None when ``state`` has no Act node within the stored depth."""
        queue: list[Mapping[str, Any]] = [self._tree]
        for node in queue:  # breadth-first: the list grows while it is walked
            if node.get("kind") != "act":
                continue
            raw = node.get("state")
            if raw is not None and _pos(raw) == state:
                return _snapshot_tree(node, depth)
            queue.extend(node.get("children", ()))
        return None


@dataclass(frozen=True)
class ReplayOptions:
    """Replay settings (``python -m rescue.replay`` flags). ``autoplay`` None = play when
    headless; ``step`` is the step shown first (negative counts from the end)."""

    high_contrast: bool = False
    headless: bool = False
    frames: int = theme.HEADLESS_FRAMES
    screenshot: str | None = None
    record: str | None = None
    mp4: bool = False
    autoplay: bool | None = None
    speed_index: int = theme.DEFAULT_SPEED_INDEX
    step: int = 0
    recordings_dir: str = theme.RECORDINGS_DIR


class ReplayApp(BaseApp):
    """Scrubbable replay of a :class:`~rescue.runs.RunRecord`."""

    caption = theme.REPLAY_CAPTION
    help_items = False

    def __init__(self, record: RunRecord, options: ReplayOptions | None = None) -> None:
        opts = options or ReplayOptions()
        if not record.events:
            raise ValueError("run file has no events")
        self.record = record
        self.options = opts
        self.events = list(record.events)
        self.n = len(self.events) - 1
        area = theme.grid_area(theme.WINDOW_SIZE)
        bar_h = theme.PROGRESS_BAR_HEIGHT + theme.PROGRESS_BAR_GAP
        #: The whole progress strip (bar + label); :attr:`bar_rect` is set once fonts exist.
        self.strip_rect = pygame.Rect(area.x, area.bottom - theme.PROGRESS_BAR_HEIGHT,
                                      area.width, theme.PROGRESS_BAR_HEIGHT)
        self.bar_rect = pygame.Rect(self.strip_rect)
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
            grid_rect=pygame.Rect(area.x, area.y, area.width, area.height - bar_h),
        )
        # The "step k / n  t = t" label goes right of the bar, in room for its widest value.
        widest = PROGRESS_TEXT.format(k=self.n, n=self.n, t=self.events[-1].t)
        label_w = text_cache.width(self.fonts.small, widest) + theme.PROGRESS_LABEL_GAP
        self.bar_rect = pygame.Rect(self.strip_rect.x, self.strip_rect.y,
                                    max(1, self.strip_rect.width - label_w),
                                    self.strip_rect.height)
        self._build_states()
        self._snapshots: dict[int, dict[str, Any]] = {
            int(s["plan_id"]): s for s in record.snapshots
        }
        self._plans: dict[int, SnapshotPlan] = {}
        self._arrows: dict[tuple[int, int], dict[Pos, Action]] = {}
        self._envelopes: dict[int, frozenset[Pos]] = {}
        self._bar_cache: tuple[Any, pygame.Surface] | None = None
        self.k = -1
        self.log.add(self._header(), "info")
        start = opts.step if opts.step >= 0 else self.n + 1 + opts.step
        self.set_step(max(0, min(self.n, start)))

    # --- reconstruction --------------------------------------------------------------------
    def _build_states(self) -> None:
        """Per-step belief / truth / high-slip arrays (shared while unchanged) and counters."""
        grid = self.record.grid
        belief = np.zeros(grid.shape, dtype=np.int8)
        belief[grid.survivor] = int(Cell.SURVIVOR)
        truth = grid.cells.copy()
        slip = grid.high_slip.copy()
        self.beliefs: list[np.ndarray] = []
        self.truths: list[np.ndarray] = []
        self.slips: list[np.ndarray] = []
        self.marks: list[TraceMark] = trace_from_events(self.events)
        self.mark_counts: list[int] = []
        #: Per step: moves made since the step whose event introduced the shown plan (the
        #: A* path cursor of :class:`~rescue.agents.base.PathAgent`).
        self.path_cursors: list[int] = []
        first_step: dict[int, int] = {}
        self.counters: list[dict[str, int]] = []
        counts = {"replans": 0, "slips": 0, "skids": 0, "bumps": 0, "unplanned": 0,
                  "aftershocks": 0}
        n_marks = 0
        for event in self.events:
            if event.new_cells or event.contradictions:
                belief = belief.copy()
                for p, cell in event.new_cells:
                    belief[p] = int(cell)
                for p in event.contradictions:
                    belief[p] = int(Cell.DEBRIS)
            if event.aftershock_cells:
                truth, slip = truth.copy(), slip.copy()
                for p in event.aftershock_cells:
                    truth[p] = int(Cell.DEBRIS)
                    slip[p] = False
                counts["aftershocks"] += 1
            if event.action is not None and event.outcome is not None:
                n_marks += 1
                if event.outcome is OutcomeKind.SLIPPED:
                    counts["slips"] += 1
                elif event.outcome in (OutcomeKind.SKID_L, OutcomeKind.SKID_R):
                    counts["skids"] += 1
            counts["replans"] += int(event.replanned and event.t > 0)
            counts["bumps"] += int(event.bump)
            counts["unplanned"] += int(event.unplanned)
            self.beliefs.append(belief)
            self.truths.append(truth)
            self.slips.append(slip)
            self.mark_counts.append(n_marks)
            first = first_step.setdefault(int(event.plan_id), len(self.mark_counts) - 1)
            self.path_cursors.append(n_marks - self.mark_counts[first])
            self.counters.append(dict(counts))
        self._trace_k = -1
        self._trace: list[TraceMark] = []

    # --- stepping --------------------------------------------------------------------------
    def _header(self) -> str:
        """First event-log line, as the simulator writes it when a mission starts."""
        rec, grid = self.record, self.record.grid
        return (f"{rec.agent} on {rec.scenario or grid.name} {grid.rows}x{grid.cols}, "
                f"seed {rec.seed}, p = {rec.world_config.slip_p:.2f}")

    @property
    def event(self) -> StepEvent:
        """The event of the step shown."""
        return self.events[self.k]

    def set_step(self, k: int) -> None:
        """Show the state after ``events[0..k]`` (clamped to ``0..n``)."""
        k = max(0, min(self.n, int(k)))
        if k == self.k:
            return
        if k == self.k + 1:
            for text, kind in event_messages(self.events[k]):
                self.log.add(text, kind)
        else:
            self.log.clear()
            self.log.add(self._header(), "info")
            first = max(0, k - theme.LOG_CAPACITY + 1)
            for event in self.events[first : k + 1]:
                for text, kind in event_messages(event):
                    self.log.add(text, kind)
        self.k = k
        self._flash_cells = ()
        self._flash(flash_cells(self.events[k]))
        if k >= self.n:
            self.paused = True

    def _can_advance(self) -> bool:
        return self.k < self.n

    def _advance(self) -> bool:
        if self.k >= self.n:
            return False
        self.set_step(self.k + 1)
        return True

    def handle_key(self, key: int, mods: int = 0) -> bool:
        """Replay controls plus the shared ones; True if handled."""
        if super().handle_key(key, mods):
            return True
        stride = theme.SCRUB_BIG_STEP if mods & pygame.KMOD_SHIFT else 1
        if key == pygame.K_RIGHT:
            self.paused = True
            self.set_step(self.k + stride)
        elif key == pygame.K_LEFT:
            self.paused = True
            self.set_step(self.k - stride)
        elif key == pygame.K_HOME:
            self.set_step(0)
        elif key == pygame.K_END:
            self.set_step(self.n)
        elif key == pygame.K_SPACE:
            if self.paused and self.k >= self.n:
                self.set_step(0)
            self.paused = not self.paused
        else:
            return False
        return True

    def _handle_mouse(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != pygame.BUTTON_LEFT:
            return False
        if not self.bar_rect.collidepoint(event.pos) or self.n <= 0:
            return False
        frac = (event.pos[0] - self.bar_rect.x) / max(1, self.bar_rect.width - 1)
        self.paused = True
        self.set_step(round(frac * self.n))
        return True

    # --- plan data -------------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any] | None:
        return self._snapshots.get(int(self.event.plan_id))

    def _policy(self, snap: dict[str, Any]) -> dict[Pos, Action]:
        """Policy arrows of ``snap`` at the shown step: the whole policy of a contingency plan,
        or the A* path from the path cursor on (as ``PathAgent.policy_arrows``)."""
        pid = int(snap["plan_id"])
        policy = snap.get("policy") or {}
        cursor = 0 if policy else self.path_cursors[self.k]
        arrows = self._arrows.get((pid, cursor))
        if arrows is None:
            arrows = {}
            for key, name in policy.items():
                r, c = (int(v) for v in str(key).split(","))
                arrows[(r, c)] = Action[name]
            path = snap.get("path")
            if not arrows and path:
                cells = [_pos(p) for p in path]
                for i in range(cursor, len(cells) - 1):
                    arrows[cells[i]] = direction(cells[i], cells[i + 1])
            self._arrows[(pid, cursor)] = arrows
        return arrows

    @staticmethod
    def _plan_size(snap: dict[str, Any] | None) -> int:
        """Size of the snapshot's plan as ``AgentStats.plan_size`` counts it (states with an
        action; path length - 1 for A*; 0 for a failed planning call)."""
        if snap is None:
            return 0
        policy = snap.get("policy") or {}
        if policy:
            return len(policy)
        path = snap.get("path")
        return len(path) - 1 if path else 0

    def _envelope(self, snap: dict[str, Any] | None) -> frozenset[Pos]:
        if snap is None:
            return frozenset()
        pid = int(snap["plan_id"])
        env = self._envelopes.get(pid)
        if env is None:
            env = self._envelopes[pid] = frozenset(_pos(p) for p in snap.get("envelope", ()))
        return env

    def _trace_upto(self) -> list[TraceMark]:
        if self._trace_k != self.k:
            self._trace = self.marks[: self.mark_counts[self.k]]
            self._trace_k = self.k
        return self._trace

    # --- rendering -------------------------------------------------------------------------
    def _scene(self) -> GridScene:
        grid, k = self.record.grid, self.k
        snap = self._snapshot()
        show_policy = self.show_policy and snap is not None
        return GridScene(
            rows=grid.rows,
            cols=grid.cols,
            belief=self.beliefs[k],
            true_cells=self.truths[k],
            high_slip=self.slips[k],
            robot=self.event.new_pos,
            start=grid.start,
            survivor=grid.survivor,
            show_true=self.show_true,
            show_policy=show_policy,
            policy=self._policy(snap) if show_policy and snap is not None else {},
            envelope=self._envelope(snap),
            trace=self._trace_upto(),
            flash_cells=self._flash_cells,
        )

    def _update_tree(self) -> None:
        snap = self._snapshot()
        pos = self.event.new_pos
        if snap is not None and snap.get("tree"):
            pid = int(snap["plan_id"])
            plan = self._plans.get(pid)
            if plan is None:
                plan = self._plans[pid] = SnapshotPlan(snap["tree"], self.record.grid.survivor)
            self.tree.set_plan(plan, pos, theme.PANEL_TREE_DEPTH, plan_id=pid)  # type: ignore[arg-type]
        else:
            self.tree.set_plan(None, pos, theme.PANEL_TREE_DEPTH)
            path = snap.get("path") if snap is not None else None
            self.tree.set_path([_pos(p) for p in path] if path else None)
            self.tree.set_message(no_plan_message(
                _pos(snap["pos"]) if snap is not None else None, bool(path)
            ))

    def _panel_info(self) -> PanelInfo:
        c = self.counters[self.k]
        explored = int(np.count_nonzero(self.beliefs[self.k]))
        final = self.record.metrics
        # The simulator's labels, in its order ("Plan ms" is the run total here).
        metrics = {
            REPLAY_STEPS_LABEL: f"{self.event.t}/{self.events[-1].t}",
            "Replans": str(c["replans"]),
            "Slips": str(c["slips"]),
            "Skids": str(c["skids"]),
            "Explored": str(explored),
            "Aftershocks": str(c["aftershocks"]),
            PLAN_MS_ALL_LABEL: f"{final.planning_ms:.1f}",
            "Plan size": str(self._plan_size(self._snapshot())),
            "Unplanned": str(c["unplanned"]),
            "Bumps": str(c["bumps"]),
            "Plan #": str(self.event.plan_id),
        }
        return PanelInfo(
            title=REPLAY_TITLE,
            agent=self.record.agent,
            status=self.event.status,
            seed=self.record.seed,
            slip_p=self.record.world_config.slip_p,
            scenario=self.record.scenario or self.record.grid.name,
            metrics=metrics,
            speed=self.speed,
            paused=self.paused or self.k >= self.n,
            recording=self.recorder.active,
            hint=REPLAY_HINT,
        )

    def _bar_background(self) -> pygame.Surface:
        """Progress-bar track with ticks at replan steps (cached per palette)."""
        key = (id(self.palette), self.bar_rect.size)
        if self._bar_cache is not None and self._bar_cache[0] == key:
            return self._bar_cache[1]
        pal, rect = self.palette, self.bar_rect
        surf = pygame.Surface(rect.size)
        surf.fill(pal.panel_bg)
        if self.n > 0:
            tick = theme.PROGRESS_TICK_WIDTH
            for i, event in enumerate(self.events):
                if event.replanned and event.t > 0:
                    x = round(i / self.n * (rect.width - 1))
                    x = min(max(0, x - tick // 2), rect.width - tick)
                    surf.fill(pal.replan_flash, (x, 0, tick, rect.height))
        pygame.draw.rect(surf, pal.grid_line, surf.get_rect(), 1)
        self._bar_cache = (key, surf)
        return surf

    def _draw_extra(self, surface: pygame.Surface) -> None:
        pal, rect = self.palette, self.bar_rect
        surface.blit(self._bar_background(), rect.topleft)
        frac = self.k / self.n if self.n > 0 else 1.0
        fill = pygame.Rect(rect.x, rect.y, max(1, round(frac * rect.width)), rect.height)
        tint = pygame.Surface(fill.size, pygame.SRCALPHA)
        tint.fill((*pal.highlight, theme.PROGRESS_FILL_ALPHA))
        surface.blit(tint, fill.topleft)
        pygame.draw.line(surface, pal.highlight, (fill.right - 1, rect.y),
                         (fill.right - 1, rect.bottom - 1), theme.PROGRESS_MARKER_WIDTH)
        label = PROGRESS_TEXT.format(k=self.k, n=self.n, t=self.event.t)
        text = text_cache.render(self.fonts.small, label, pal.text)
        strip = self.strip_rect
        surface.fill(pal.background, (rect.right, strip.y, strip.right - rect.right, strip.height))
        where = text.get_rect(midright=(strip.right, strip.centery))
        surface.blit(text, where)


def load_replay(path: str | Path, options: ReplayOptions | None = None) -> ReplayApp:
    """Load a run file and build its :class:`ReplayApp`."""
    from rescue.runs import load_run

    return ReplayApp(load_run(path), options)


__all__ = ["ReplayApp", "ReplayOptions", "SnapshotPlan", "load_replay"]
