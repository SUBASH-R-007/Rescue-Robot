"""Phase 5 tests of the head-to-head comparison app (brief §6.2, §9; docs/plan.md §3.11).

Everything runs headless (SDL dummy drivers, selected before pygame is imported). The app is
driven through its public loop API (``handle_key``, ``step_once``, ``tick``, ``render``,
``run``); every file it writes goes to ``tmp_path``.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import itertools
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pygame = pytest.importorskip("pygame")

from rescue import compare, mapgen  # noqa: E402
from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES  # noqa: E402
from rescue.config import (  # noqa: E402
    COMPARE_DEFAULT_SCENARIO,
    COMPARE_DEFAULT_SEED,
    COMPARE_MAX_AGENTS,
    COMPARE_PANE_GAP,
    DEFAULT_SLIP_P,
    MAX_SLIP_P,
)
from rescue.model import Action, OutcomeKind  # noqa: E402
from rescue.runner import MissionStatus, StepEvent  # noqa: E402
from rescue.viz import theme  # noqa: E402
from rescue.viz.compare_app import (  # noqa: E402
    ALL_DONE_TEXT,
    CRN_AFTERSHOCK_TEXT,
    CRN_TEXT,
    TICK_TEXT,
    CompareApp,
    CompareOptions,
    agent_colour,
    outcome_rows,
    pane_event_line,
    parse_agents,
)
from rescue.viz.sim_app import SimApp, SimOptions  # noqa: E402

pytestmark = pytest.mark.viz

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Headless frames of the smoke tests.
SMOKE_FRAMES = 150
#: Lock-step ticks after which every mission of the default comparison must have ended.
MAX_TICKS = 4 * 15 * 15
#: Seeds of the common-random-numbers check, per scenario.
CRN_SEEDS = (7, 11)
#: Frames rendered while a ``G`` recording is on.
RECORD_FRAMES = 6
#: A map whose survivor is walled in: the optimistic agents find out while exploring.
WALLED_MAP = "S....\n.....\n...##\n...#P\n...##\n"


# --- helpers -----------------------------------------------------------------------------------
def png_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's IHDR chunk (AssertionError if it is not a PNG)."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE, f"{path} is not a PNG"
    return struct.unpack(">II", data[16:24])


def options(tmp_path: Path, **overrides: Any) -> CompareOptions:
    """Headless, paused comparison options writing recordings under ``tmp_path``."""
    values: dict[str, Any] = {
        "headless": True,
        "autoplay": False,
        "recordings_dir": str(tmp_path / "recordings"),
    }
    values.update(overrides)
    return CompareOptions(**values)


def run_to_end(app: CompareApp) -> None:
    """Step every mission to its end (bounded), checking lock-step after every tick."""
    for _ in range(MAX_TICKS + 1):
        if app.all_done:
            return
        app.step_once()
        running = [pane for pane in app.panes if not pane.mission.done]
        assert all(pane.t == app.ticks for pane in running)
    raise AssertionError("missions did not finish within the step budget")


def outcome_by_t(app: CompareApp) -> dict[int, list[tuple[float, str]]]:
    """Per world time t: ``(slip p of the cell acted from, outcome kind)`` of each pane that
    moved to t."""
    rows: dict[int, list[tuple[float, str]]] = {}
    for pane in app.panes:
        transition = pane.mission.transition
        for event in pane.mission.events:
            if event.outcome is not None:
                rows.setdefault(event.t, []).append(
                    (transition.slip_p(event.pos), event.outcome.value))
    return rows


@pytest.fixture
def app(tmp_path: Path) -> Iterator[CompareApp]:
    """The default comparison (collapsed_building 15x15, seed 7), paused and headless."""
    application = CompareApp(options(tmp_path))
    application.render()
    yield application
    application.close()


# --- smoke -------------------------------------------------------------------------------------
def test_headless_run_renders_frames_and_saves_screenshot(tmp_path: Path) -> None:
    shot = tmp_path / "shots" / "compare.png"
    application = CompareApp(CompareOptions(headless=True, frames=SMOKE_FRAMES,
                                            screenshot=str(shot)))
    assert not application.paused  # headless plays by default
    assert application.run() == 0
    assert png_size(shot) == theme.WINDOW_SIZE
    assert application.ticks > 0
    assert [pane.agent for pane in application.panes] == list(AGENT_NAMES)


def test_cli_headless_screenshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shot = tmp_path / "cli.png"
    code = compare.main(["--scenario", "collapsed_building", "--seed", "7", "--headless",
                         "--frames", "30", "--screenshot", str(shot)])
    assert code == 0
    assert png_size(shot) == theme.WINDOW_SIZE
    out = capsys.readouterr().out
    assert all(name in out for name in AGENT_NAMES)
    assert "Screenshot saved" in out


@pytest.mark.parametrize("argv", [
    ["--agents", "andor"],
    ["--agents", "andor,andor"],
    ["--agents", "andor,nobody"],
    ["--agents", ",".join([*ALL_AGENT_NAMES, AGENT_NAMES[0]])],
    ["--scenario", "no_such_scenario"],
    ["--map", "no/such/map.txt", "--headless", "--frames", "1"],
    ["--p", str(MAX_SLIP_P + 0.1), "--headless", "--frames", "1"],
])
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        compare.main(argv)
    assert info.value.code == 2


def test_defaults_match_the_brief() -> None:
    opts = CompareOptions()
    assert opts.scenario == COMPARE_DEFAULT_SCENARIO == "collapsed_building"
    assert opts.seed == COMPARE_DEFAULT_SEED == 7
    assert (opts.rows, opts.cols) == (15, 15)
    assert opts.agents == AGENT_NAMES == ("andor", "astar_fixed", "astar_replan")
    args = compare.build_parser().parse_args([])
    assert parse_agents(args.agents) == AGENT_NAMES
    assert args.p == DEFAULT_SLIP_P


def test_parse_agents_validation() -> None:
    assert parse_agents(None) == AGENT_NAMES
    assert parse_agents(" andor , astar_replan ") == ("andor", "astar_replan")
    assert parse_agents(list(ALL_AGENT_NAMES)) == ALL_AGENT_NAMES
    assert len(ALL_AGENT_NAMES) == COMPARE_MAX_AGENTS
    for bad in ("andor", "andor,andor", "andor,bogus", ""):
        with pytest.raises(ValueError):
            parse_agents(bad)


# --- lock-step, CRN, terminal statuses ---------------------------------------------------------
def test_panes_share_map_and_seed(app: CompareApp) -> None:
    grids = [pane.mission.grid for pane in app.panes]
    assert all(g == grids[0] for g in grids)
    assert grids[0] == app.base_grid
    assert {pane.mission.seed for pane in app.panes} == {app.seed}
    assert {pane.mission.world_config for pane in app.panes} == {app.world_config()}


def test_lock_step_and_terminal_statuses(app: CompareApp) -> None:
    assert all(pane.t == 0 and pane.status is MissionStatus.RUNNING for pane in app.panes)
    run_to_end(app)
    statuses = {pane.agent: pane.status for pane in app.panes}
    assert all(s is not MissionStatus.RUNNING for s in statuses.values())
    assert statuses["andor"] is MissionStatus.RESCUED
    assert statuses["astar_fixed"] is MissionStatus.ABORTED  # first deviation (seed 7)
    # the last tick is the longest mission's last step
    assert app.ticks == max(pane.t for pane in app.panes)
    assert app.paused
    assert app.log.lines[-1] == (ALL_DONE_TEXT, "info")
    assert not app.step_once()
    assert app.tick(1000.0) == 0
    app.render()


@pytest.mark.parametrize("scenario", ["collapsed_building", "flooded_zone", "maze_corridors"])
@pytest.mark.parametrize("seed", CRN_SEEDS)
def test_common_random_numbers(tmp_path: Path, scenario: str, seed: int) -> None:
    """Robots acting from cells with the same slip p at step t draw the same outcome kind."""
    application = CompareApp(options(tmp_path, scenario=scenario, seed=seed))
    try:
        run_to_end(application)
        rows = outcome_by_t(application)
        assert rows
        non_arrived = 0
        for t, row in rows.items():
            by_p: dict[float, set[str]] = {}
            for p, kind in row:
                by_p.setdefault(p, set()).add(kind)
            assert all(len(kinds) == 1 for kinds in by_p.values()), (t, row)
            non_arrived += any(kind != "arrived" for _, kind in row)
        assert non_arrived > 0  # the check saw slips / skids, not only "arrived"
    finally:
        application.close()


def test_crn_identical_kinds_on_uniform_terrain(app: CompareApp) -> None:
    run_to_end(app)
    assert not app.base_grid.high_slip.any()
    for _t, row in outcome_rows(app):
        assert len({kind for kind in row if kind is not None}) == 1


def test_walled_in_survivor_ends_unreachable(tmp_path: Path) -> None:
    path = tmp_path / "walled.txt"
    path.write_text(WALLED_MAP, encoding="utf-8")
    application = CompareApp(options(tmp_path, map_path=str(path)))
    try:
        run_to_end(application)
        statuses = {pane.agent: pane.status for pane in application.panes}
        assert statuses["andor"] is MissionStatus.UNREACHABLE
        assert statuses["astar_replan"] is MissionStatus.UNREACHABLE
        assert statuses["astar_fixed"] in (MissionStatus.ABORTED, MissionStatus.UNREACHABLE)
        assert application.paused
        application.render()
    finally:
        application.close()


def test_five_agents_including_stretch_agents(tmp_path: Path) -> None:
    application = CompareApp(options(tmp_path, agents=ALL_AGENT_NAMES, scenario="random_rubble",
                                     rows=10, cols=10))
    try:
        run_to_end(application)
        assert len(application.panes) == len(ALL_AGENT_NAMES)
        application.render()
        rects = [pane.rect for pane in application.panes]
        assert all(a.right < b.left for a, b in itertools.pairwise(rects))
        assert all(r.right <= theme.WINDOW_SIZE[0] for r in rects)
        for pane in application.panes:
            assert pane.rect.contains(pane.view.board_rect)
    finally:
        application.close()


# --- keys --------------------------------------------------------------------------------------
def test_space_toggles_play_and_n_steps_one_tick(app: CompareApp) -> None:
    assert app.paused
    assert app.handle_key(pygame.K_SPACE)
    assert not app.paused
    assert app.handle_key(pygame.K_SPACE)
    assert app.paused
    assert app.handle_key(pygame.K_n)
    assert app.paused and app.ticks == 1
    assert all(pane.t == 1 for pane in app.panes if not pane.mission.done)
    app.handle_key(pygame.K_SPACE)
    steps = app.tick(theme.MAX_FRAME_MS)  # the longest frame the accumulator accepts
    assert steps == int(theme.MAX_FRAME_MS * app.speed / 1000.0) > 0
    assert app.ticks == 1 + steps


def test_r_resets_with_the_same_seed(app: CompareApp) -> None:
    run_to_end(app)
    before = [list(pane.mission.events) for pane in app.panes]
    assert app.handle_key(pygame.K_r)
    assert app.ticks == 0 and all(pane.t == 0 for pane in app.panes)
    assert app.seed == COMPARE_DEFAULT_SEED
    run_to_end(app)
    assert [pane.mission.events for pane in app.panes] == before


def test_s_new_seed_regenerates_the_scenario(app: CompareApp) -> None:
    grid = app.base_grid
    app.handle_key(pygame.K_n)
    assert app.handle_key(pygame.K_s)
    assert app.seed == COMPARE_DEFAULT_SEED + 1
    assert app.ticks == 0
    assert app.base_grid == mapgen.generate("collapsed_building", 15, 15, app.seed)
    assert app.base_grid != grid
    assert {pane.mission.seed for pane in app.panes} == {app.seed}


def test_speed_and_slip_keys(app: CompareApp) -> None:
    index = app.speed_index
    assert app.handle_key(pygame.K_PLUS)
    assert app.speed_index == index + 1
    assert app.handle_key(pygame.K_MINUS)
    assert app.handle_key(pygame.K_MINUS)
    assert app.speed_index == index - 1
    assert app.handle_key(pygame.K_RIGHTBRACKET)
    assert app.slip_p == pytest.approx(DEFAULT_SLIP_P + theme.SLIP_P_STEP)
    assert {pane.mission.world_config.slip_p for pane in app.panes} == {app.slip_p}
    for _ in range(20):
        app.handle_key(pygame.K_RIGHTBRACKET)
    assert app.slip_p == MAX_SLIP_P
    for _ in range(20):
        app.handle_key(pygame.K_LEFTBRACKET)
    assert app.slip_p == 0.0
    run_to_end(app)  # p = 0: deterministic moves, no slips at all
    assert all(pane.mission.metrics().slips == 0 for pane in app.panes)


def test_view_keys_help_and_quit(app: CompareApp) -> None:
    for key, attr in ((pygame.K_t, "show_true"), (pygame.K_p, "show_policy"),
                      (pygame.K_h, "show_help")):
        assert app.handle_key(key)
        assert getattr(app, attr)
        app.render()
        assert app.handle_key(key)
        assert not getattr(app, attr)
    palette = app.palette
    assert app.handle_key(pygame.K_c)
    assert app.palette is not palette
    assert all(pane.view.palette is app.palette for pane in app.panes)
    app.render()
    assert not app.handle_key(pygame.K_q)
    assert app.handle_key(pygame.K_ESCAPE)
    assert not app.running


def test_policy_arrows_come_from_the_agents(app: CompareApp) -> None:
    app.handle_key(pygame.K_p)
    for pane in app.panes:
        scene = app.pane_scene(pane)
        assert scene.show_policy
        assert dict(scene.policy) == pane.mission.agent.policy_arrows()
        assert scene.belief is pane.mission.agent.belief.grid


def test_g_records_a_gif(app: CompareApp, tmp_path: Path) -> None:
    app.handle_key(pygame.K_SPACE)
    assert app.handle_key(pygame.K_g)
    assert app.recorder.active
    for _ in range(RECORD_FRAMES):
        app.frame(1000.0 / theme.RECORD_FPS)
    app.handle_key(pygame.K_g)
    assert not app.recorder.active
    assert len(app.recordings) == 1
    path = app.recordings[0]
    assert path.parent == tmp_path / "recordings" and path.suffix == ".gif"
    assert path.stat().st_size > 0


def test_m_cycles_the_mapgen_scenarios(app: CompareApp) -> None:
    names = list(mapgen.SCENARIOS)
    start = names.index(COMPARE_DEFAULT_SCENARIO)
    seen = [app.scenario_name]
    for _ in range(len(names)):
        app.handle_key(pygame.K_n)
        assert app.handle_key(pygame.K_m)
        assert app.ticks == 0
        assert all(pane.mission.grid == app.base_grid for pane in app.panes)
        seen.append(app.scenario_name)
    assert seen == [names[(start + k) % len(names)] for k in range(len(names) + 1)]
    app.render()


def test_map_file_is_first_in_the_cycle(tmp_path: Path) -> None:
    application = CompareApp(options(tmp_path, map_path=str(MAPS_DIR / "deck_demo.txt")))
    try:
        assert application.scenario_index == 0
        assert (application.base_grid.rows, application.base_grid.cols) == (4, 4)
        application.handle_key(pygame.K_m)
        assert application.scenario_name == next(iter(mapgen.SCENARIOS))
        application.handle_key(pygame.K_s)  # a generated scenario follows the new seed
        assert application.base_grid == mapgen.generate(
            application.scenario_name, 15, 15, COMPARE_DEFAULT_SEED + 1)
    finally:
        application.close()


@pytest.mark.parametrize("palette", theme.PALETTES, ids=lambda p: p.name)
def test_every_agent_has_a_distinct_readable_colour(palette: theme.Palette) -> None:
    colours = [agent_colour(palette, name) for name in ALL_AGENT_NAMES]
    assert len(set(colours)) == len(colours)
    assert all(theme.contrast_ratio(c, palette.panel_bg) >= 3.0 for c in colours)


# --- simulator hookup --------------------------------------------------------------------------
def test_sim_m_key_cycles_mapgen_generators(tmp_path: Path) -> None:
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "deck_demo.txt").write_text((MAPS_DIR / "deck_demo.txt").read_text())
    sim = SimApp(SimOptions(headless=True, autoplay=False, maps_dir=str(maps),
                            map_path=str(maps / "deck_demo.txt"),
                            recordings_dir=str(tmp_path / "rec")))
    try:
        names = [sim.scenario_name]
        for _ in range(len(sim.scenarios)):
            sim.handle_key(pygame.K_m)
            names.append(sim.scenario_name)
        assert names[0] == names[-1] == "deck_demo"
        assert names[1:-1] == list(mapgen.GENERATED_SCENARIOS)
        sim.render()
    finally:
        sim.close()


# --- fix pass: raw CRN draws, aftershock text, tick label, event lines, fonts, two rows ---------
class _RecordingGenerator:
    """Delegates to a numpy Generator and logs every ``random`` / ``permutation`` draw."""

    def __init__(self, gen: Any, log: list[tuple[str, Any]]) -> None:
        self.gen, self.log = gen, log

    def random(self, *args: Any, **kwargs: Any) -> Any:
        value = self.gen.random(*args, **kwargs)
        self.log.append(("random", value.tolist() if hasattr(value, "tolist") else value))
        return value

    def permutation(self, *args: Any, **kwargs: Any) -> Any:
        value = self.gen.permutation(*args, **kwargs)
        self.log.append(("permutation", value.tolist()))
        return value


@pytest.mark.parametrize(("scenario", "seed", "q"), [
    ("collapsed_building", 7, 0.0),
    ("flooded_zone", 3, 0.0),
    ("random_rubble", 4, 0.5),
    ("collapsed_building", 9, 0.3),
])
def test_crn_raw_draws_are_identical_in_every_pane(
    tmp_path: Path, scenario: str, seed: int, q: float
) -> None:
    """The outcome uniforms u_t and the aftershock (u, permutation) draws themselves, not only
    the outcome kinds, are the same sequence in every pane (each pane's log is a prefix of the
    longest one), with and without aftershocks (D3, D20)."""
    application = CompareApp(options(tmp_path, scenario=scenario, seed=seed,
                                     agents=ALL_AGENT_NAMES, aftershock_q=q, aftershock_k=2))
    try:
        logs: list[tuple[list[tuple[str, Any]], list[tuple[str, Any]]]] = []
        for pane in application.panes:
            rng = pane.mission.world.rng
            outcome_log: list[tuple[str, Any]] = []
            shock_log: list[tuple[str, Any]] = []
            rng.outcome = _RecordingGenerator(rng.outcome, outcome_log)
            rng.aftershock = _RecordingGenerator(rng.aftershock, shock_log)
            logs.append((outcome_log, shock_log))
        run_to_end(application)
        for which in (0, 1):
            seqs = [log[which] for log in logs]
            longest = max(seqs, key=len)
            assert all(seq == longest[:len(seq)] for seq in seqs)
        steps = [len(log[0]) for log in logs]
        assert steps == [pane.t for pane in application.panes]
        assert max(steps) > 0
        if q > 0:  # one (u, permutation) aftershock draw per step, whatever happens
            assert [len(log[1]) for log in logs] == [2 * n for n in steps]
    finally:
        application.close()


def test_crn_footer_text_depends_on_aftershocks(tmp_path: Path) -> None:
    for q, expected in ((0.0, CRN_TEXT), (0.3, CRN_AFTERSHOCK_TEXT)):
        application = CompareApp(options(tmp_path, aftershock_q=q, aftershock_k=2))
        try:
            assert application.crn_text() == expected
            application.render()
        finally:
            application.close()
    assert "same map" not in CRN_AFTERSHOCK_TEXT.lower()


def test_footer_counts_ticks_not_world_time(tmp_path: Path) -> None:
    """A final Abort takes a lock-step tick but no world step, so the tick counter can exceed
    every pane's t; the footer labels it as a tick (each pane shows its own Steps)."""
    path = tmp_path / "walled.txt"
    path.write_text(WALLED_MAP, encoding="utf-8")
    application = CompareApp(options(tmp_path, map_path=str(path)))
    try:
        run_to_end(application)
        assert application.ticks == max(pane.t for pane in application.panes) + 1
        footer = application.footer_text()
        assert TICK_TEXT.format(t=application.ticks) in footer
        assert f"t = {application.ticks}" not in footer
        # the failure line keeps the failure colour and carries the agent's reason
        for pane in application.panes:
            text, kind = pane.last_event
            assert kind == "failure" and "Mission failed" in text, pane.last_event
        assert "no strong-cyclic plan" in application.panes[0].last_event[0]
    finally:
        application.close()


def _event(**overrides: Any) -> StepEvent:
    values: dict[str, Any] = {
        "t": 5, "pos": (0, 0), "action": Action.E, "outcome": OutcomeKind.ARRIVED,
        "new_pos": (0, 1), "blocked": False, "bump": False, "new_cells": (),
        "contradictions": (), "triggering_contradictions": (), "replanned": False,
        "unplanned": False, "plan_id": 0, "aftershock_cells": (),
        "status": MissionStatus.RUNNING, "note": "",
    }
    values.update(overrides)
    return StepEvent(**values)


def test_pane_event_line_picks_the_most_important_line() -> None:
    assert pane_event_line(_event()) is None
    slip_shock_replan = _event(outcome=OutcomeKind.SLIPPED, new_pos=(0, 0),
                               contradictions=((1, 1),), triggering_contradictions=((1, 1),),
                               replanned=True, aftershock_cells=((2, 2),))
    text, kind = pane_event_line(slip_shock_replan)  # type: ignore[misc]
    assert kind == "warn" and text.startswith("t=5: Aftershock")
    slip_replan = _event(outcome=OutcomeKind.SLIPPED, new_pos=(0, 0), replanned=True)
    assert pane_event_line(slip_replan)[1] == "replan"  # type: ignore[index]
    rescue_after_skid = _event(outcome=OutcomeKind.SKID_L, new_pos=(1, 0),
                               status=MissionStatus.RESCUED)
    assert pane_event_line(rescue_after_skid) == ("t=5: Rescue successful", "success")
    abort = _event(action=None, outcome=None, new_pos=(0, 0), status=MissionStatus.ABORTED,
                   note="aborted: deviated from the path: at (1,4), expected (1,5)")
    assert pane_event_line(abort) == (
        "t=5: Mission failed (aborted): deviated from the path: at (1,4), expected (1,5)",
        "failure",
    )


def test_agent_names_share_one_font(tmp_path: Path, app: CompareApp) -> None:
    assert app.name_font is app.fonts.heading  # three panes: room for the big font
    application = CompareApp(options(tmp_path, agents=ALL_AGENT_NAMES))
    try:
        application.render()
        font = application.name_font
        assert font in (application.fonts.heading, application.fonts.body,
                        application.fonts.small)
        names = [pane.agent for pane in application.panes]
        heights = {font.size(name)[1] for name in names}
        assert len(heights) == 1
    finally:
        application.close()


def _one_row_cell(app: CompareApp) -> int:
    """Cell size the single-row layout would give (same formula as ``layout``)."""
    width, height = app.screen.get_size()
    n = len(app.panes)
    area_w = width - 2 * theme.MARGIN
    area_h = height - app._footer_height() - 2 * theme.MARGIN
    pane_w = (area_w - (n - 1) * COMPARE_PANE_GAP) // n
    return app._cell_size(pane_w, area_h)


def test_wide_maps_use_two_rows_of_panes(tmp_path: Path) -> None:
    """Five agents on a wide map: two rows of panes give clearly larger cells than one row."""
    application = CompareApp(options(tmp_path, agents=ALL_AGENT_NAMES, scenario="random_rubble",
                                     rows=6, cols=40))
    try:
        application.render()
        assert application.pane_rows == 2
        cell = application.panes[0].view.board_rect.width // 40
        assert cell > _one_row_cell(application)
        rects = [pane.rect for pane in application.panes]
        window = pygame.Rect((0, 0), theme.WINDOW_SIZE)
        assert all(window.contains(r) for r in rects)
        assert not any(a.colliderect(b) for a, b in itertools.combinations(rects, 2))
        assert len({r.y for r in rects}) == 2
        for pane in application.panes:
            assert pane.rect.contains(pane.view.board_rect)
    finally:
        application.close()


def test_square_maps_keep_one_row_when_two_rows_would_not_help(tmp_path: Path) -> None:
    application = CompareApp(options(tmp_path, agents=ALL_AGENT_NAMES, scenario="maze_corridors",
                                     rows=30, cols=30))
    try:
        application.render()
        assert application.pane_rows == 1
        assert application.panes[0].view.board_rect.width // 30 == _one_row_cell(application)
    finally:
        application.close()
