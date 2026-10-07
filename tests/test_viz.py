"""Phase 4 acceptance tests for the Pygame UI (docs/brief.md §6.1, §6.5, §9; plan.md §3.11).

Everything runs headless: the SDL dummy video / audio drivers are selected below, before any
pygame-dependent module is imported. The apps are driven through their public loop API
(``handle_key``, ``handle_event``, ``tick``, ``render``, ``frame``, ``run``), never through a
real event queue, and every file they write goes to ``tmp_path``.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
import shutil
import struct
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
imageio = pytest.importorskip("imageio.v2")

from rescue.agents import AGENT_NAMES, make_agent  # noqa: E402
from rescue.config import DEFAULT_SLIP_P, DEMO_SEED, MAX_SLIP_P, WorldConfig  # noqa: E402
from rescue.grid import Cell, Pos, TrueGrid  # noqa: E402
from rescue.planners.bfs import reachable  # noqa: E402
from rescue.runner import Mission, make_transition, run_mission  # noqa: E402
from rescue.runs import load_run, save_run  # noqa: E402
from rescue.viz import theme  # noqa: E402
from rescue.viz.export import export_plan_text, export_plan_tree  # noqa: E402
from rescue.viz.grid_view import GridScene, GridView  # noqa: E402
from rescue.viz.record import Recorder  # noqa: E402
from rescue.viz.replay_app import REPLAY_STEPS_LABEL, ReplayApp, ReplayOptions  # noqa: E402
from rescue.viz.sim_app import SimApp, SimOptions  # noqa: E402

pytestmark = pytest.mark.viz

#: Repository root and bundled maps (tests/test_viz.py → two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
#: Map files copied into the temporary maps directory used by ``M`` and Ctrl+S.
TMP_MAP_NAMES = ("deck_demo.txt", "deck_figure.txt")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Frames of the headless smoke test (brief §9, Phase 4).
SMOKE_FRAMES = 100
#: FPS acceptance: 30 FPS on a 30x30 map ⇒ < 33 ms per frame (render + step) while playing.
FPS_GRID_SIZE = 30
FPS_FRAMES = 60
FPS_WARMUP_FRAMES = 1
#: Measurement windows tried before failing (best one counts): wall-clock timing is noisy
#: when other processes (e.g. pytest-xdist workers) compete for the CPU.
FPS_ATTEMPTS = 3
#: Under pytest-xdist every CPU runs another test process, so a 60-frame window can land on a
#: burst of contention that says nothing about the simulator; allow more windows there (the
#: budget itself is unchanged, and the first window under budget ends the measurement).
FPS_ATTEMPTS_PARALLEL = 10
FPS_BUDGET_MS = 33.0
FPS_DEBRIS_DENSITY = 0.2
FPS_SLIP_FRACTION = 0.1
FPS_MAP_SEED = 2024
#: Frames rendered while the ``G`` recording is on.
RECORD_FRAMES = 12
#: Frames written by the stand-alone Recorder test.
RECORDER_FRAMES = 5


# --- helpers -----------------------------------------------------------------------------------
def png_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's IHDR chunk (AssertionError if it is not a PNG)."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE, f"{path} is not a PNG"
    assert data[12:16] == b"IHDR", f"{path} has no IHDR chunk"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def key(app: Any, k: int, mods: int = 0) -> None:
    """Press ``k`` (with ``mods``), assert the app handled it, and render a frame."""
    assert app.handle_key(k, mods), f"key {pygame.key.name(k)!r} was not handled"
    app.render()


def play_one_step(app: Any) -> int:
    """Tick headless frames for one step period plus one frame; returns the steps taken.

    Uses :data:`theme.HEADLESS_FRAME_MS` frames (a single long ``tick`` is clamped to
    :data:`theme.MAX_FRAME_MS`, so it cannot be used to force a step at slow speeds).
    """
    frames = math.ceil(1000.0 / app.speed / theme.HEADLESS_FRAME_MS) + 1
    return sum(app.tick(theme.HEADLESS_FRAME_MS) for _ in range(frames))


def click(app: SimApp, p: Pos, mods: int = 0) -> None:
    """Left-click the centre of cell ``p`` through the app's event handler."""
    pygame.key.set_mods(mods)
    try:
        xy = app.grid_view.cell_rect(p).center
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=pygame.BUTTON_LEFT,
                                            pos=xy))
    finally:
        pygame.key.set_mods(pygame.KMOD_NONE)
    app.render()


def centre_pixel(view: GridView, surface: Any, p: Pos) -> tuple[int, int, int]:
    """Colour of the rendered pixel at the centre of cell ``p`` (where the robot disc is)."""
    colour = surface.get_at(view.cell_centre(p))
    return (colour.r, colour.g, colour.b)


def corner_grid(n: int, seed: int) -> TrueGrid:
    """Seeded solvable ``n`` x ``n`` rubble map from corner (0,0) to corner (n-1,n-1).

    Opposite corners guarantee at least ``2(n-1)`` moves, so a mission keeps playing for the
    whole FPS measurement. Some open cells are high-slip terrain ('~').
    """
    rng = np.random.default_rng(seed)
    start, survivor = (0, 0), (n - 1, n - 1)
    for _ in range(100):
        cells = np.where(rng.random((n, n)) < FPS_DEBRIS_DENSITY, int(Cell.DEBRIS),
                         int(Cell.OPEN)).astype(np.int8)
        cells[start] = Cell.OPEN
        cells[survivor] = Cell.SURVIVOR
        slip = (rng.random((n, n)) < FPS_SLIP_FRACTION) & (cells == Cell.OPEN)
        slip[start] = False
        if reachable(cells != Cell.DEBRIS, start, survivor):
            return TrueGrid(cells=cells, start=start, survivor=survivor, high_slip=slip,
                            name=f"fps{n}x{n}")
    raise RuntimeError("no solvable corner map")  # pragma: no cover


def sim_options(tmp_path: Path, maps_dir: Path, map_name: str = "deck_demo.txt",
                **overrides: Any) -> SimOptions:
    """Headless, paused simulator options writing only below ``tmp_path``."""
    settings: dict[str, Any] = {
        "map_path": str(maps_dir / map_name),
        "headless": True,
        "autoplay": False,
        "maps_dir": str(maps_dir),
        "recordings_dir": str(tmp_path / "recordings"),
        "runs_dir": str(tmp_path / "runs"),
    }
    settings.update(overrides)
    return SimOptions(**settings)


# --- fixtures ----------------------------------------------------------------------------------
@pytest.fixture
def maps_tmp(tmp_path: Path) -> Path:
    """A temporary maps directory holding copies of the bundled maps."""
    directory = tmp_path / "maps"
    directory.mkdir()
    for name in TMP_MAP_NAMES:
        shutil.copy(MAPS_DIR / name, directory / name)
    return directory


@pytest.fixture
def sim(tmp_path: Path, maps_tmp: Path) -> Iterator[SimApp]:
    """A paused headless simulator on deck_demo (closed afterwards)."""
    app = SimApp(sim_options(tmp_path, maps_tmp))
    app.render()  # lays out the grid view (needed by clicks)
    yield app
    app.close()


@pytest.fixture
def pg() -> Iterator[None]:
    """pygame initialised for tests that use views directly."""
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def deck_run(tmp_path: Path) -> Path:
    """A finished andor mission on deck_demo (DEMO_SEED) saved as a run file."""
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig()
    agent = make_agent(AGENT_NAMES[0], make_transition(grid, cfg))
    mission = run_mission(grid, agent, cfg, DEMO_SEED, scenario="deck_demo")
    assert mission.done
    return save_run(mission, runs_dir=tmp_path / "runs")


# --- (1) headless smoke test -------------------------------------------------------------------
def test_headless_smoke_renders_100_frames_and_saves_screenshot(
    tmp_path: Path, maps_tmp: Path
) -> None:
    shot = tmp_path / "shots" / "deck_demo.png"
    app = SimApp(sim_options(tmp_path, maps_tmp, autoplay=None, frames=SMOKE_FRAMES,
                             screenshot=str(shot)))
    assert not app.paused  # headless plays by default
    renders = 0
    original = app.render

    def counting_render() -> Any:
        nonlocal renders
        renders += 1
        return original()

    app.render = counting_render  # type: ignore[method-assign]
    assert app.run() == 0
    assert renders == SMOKE_FRAMES
    assert app.mission.world.t >= 1  # the mission was actually played
    assert shot.is_file()
    assert png_size(shot) == theme.WINDOW_SIZE
    image = imageio.imread(shot)
    assert image.shape[:2] == (theme.WINDOW_SIZE[1], theme.WINDOW_SIZE[0])
    assert len(np.unique(image.reshape(-1, image.shape[2]), axis=0)) > 10  # not a blank frame


def test_sim_cli_headless_screenshot(tmp_path: Path) -> None:
    from rescue.sim import main

    shot = tmp_path / "cli.png"
    code = main(["--map", str(MAPS_DIR / "deck_demo.txt"), "--headless", "--frames", "10",
                 "--screenshot", str(shot)])
    assert code == 0
    assert png_size(shot) == theme.WINDOW_SIZE


# --- (2) frame rate ----------------------------------------------------------------------------
def test_render_and_step_sustain_30_fps_on_30x30(
    tmp_path: Path, maps_tmp: Path, record_property: Callable[[str, object], None]
) -> None:
    grid = corner_grid(FPS_GRID_SIZE, FPS_MAP_SEED)
    grid.save(maps_tmp / "fps30.txt")
    fastest = len(theme.SPEEDS) - 1
    app = SimApp(sim_options(tmp_path, maps_tmp, map_name="fps30.txt", autoplay=True,
                             speed_index=fastest))
    try:
        key(app, pygame.K_t)  # true-map overlay and policy arrows: the heaviest scene
        key(app, pygame.K_p)
        dt_ms = 1000.0 / app.speed  # exactly one mission step per frame
        for _ in range(FPS_WARMUP_FRAMES):
            app.tick(dt_ms)
            app.render()
        windows: list[tuple[float, float, int]] = []
        parallel = bool(os.environ.get("PYTEST_XDIST_WORKER"))
        for _ in range(FPS_ATTEMPTS_PARALLEL if parallel else FPS_ATTEMPTS):
            steps = 0
            times: list[float] = []
            for _ in range(FPS_FRAMES):
                begin = time.perf_counter()
                if app.mission.done:  # keep playing: a fresh mission (cost included)
                    app.handle_key(pygame.K_r)
                    app.handle_key(pygame.K_SPACE)
                steps += app.tick(dt_ms)
                app.render()
                times.append(time.perf_counter() - begin)
            windows.append((1000.0 * sum(times) / len(times), 1000.0 * max(times), steps))
            if windows[-1][0] < FPS_BUDGET_MS:
                break
        mean_ms, max_ms, steps = min(windows)
        record_property("frame_ms_mean", round(mean_ms, 2))  # visible in --junitxml reports
        record_property("frame_ms_max", round(max_ms, 2))
        record_property("fps_windows", len(windows))
        assert steps >= FPS_FRAMES - 2, f"only {steps} steps in {FPS_FRAMES} frames"
        assert mean_ms < FPS_BUDGET_MS, (
            f"best of {len(windows)} windows: {mean_ms:.1f} ms per frame (max {max_ms:.1f} ms)"
            f" = {1000 / mean_ms:.0f} FPS < 30 FPS; all windows (mean, max, steps): {windows}"
        )
    finally:
        app.close()


# --- (3) controls (brief §6.1 + C, Esc) ----------------------------------------------------------
def test_space_toggles_play_and_n_steps(sim: SimApp) -> None:
    assert sim.paused
    key(sim, pygame.K_SPACE)
    assert not sim.paused
    key(sim, pygame.K_SPACE)
    assert sim.paused
    t0, n0, marks0 = sim.mission.world.t, len(sim.mission.events), len(sim.trace)
    key(sim, pygame.K_n)
    assert sim.paused
    assert sim.mission.world.t == t0 + 1
    assert len(sim.mission.events) == n0 + 1
    assert len(sim.trace) == marks0 + 1
    assert sim.mission.events[-1].action is not None


def test_playing_advances_with_tick(sim: SimApp) -> None:
    key(sim, pygame.K_SPACE)
    assert play_one_step(sim) == 1
    assert sim.mission.world.t == 1
    key(sim, pygame.K_SPACE)
    assert sim.tick(theme.MAX_FRAME_MS) == 0  # paused: nothing happens
    assert sim.mission.world.t == 1


def test_r_resets_with_the_same_seed(sim: SimApp) -> None:
    for _ in range(4):
        key(sim, pygame.K_n)
    first = list(sim.mission.events)
    seed = sim.seed
    key(sim, pygame.K_r)
    assert sim.seed == seed
    assert sim.mission.seed == seed
    assert len(sim.mission.events) == 1 and sim.mission.world.t == 0
    assert sim.trace == []
    for _ in range(4):
        key(sim, pygame.K_n)
    assert sim.mission.events == first  # same seed ⇒ identical events


def test_s_starts_a_new_seed(sim: SimApp) -> None:
    key(sim, pygame.K_n)
    seed, grid = sim.seed, sim.base_grid
    key(sim, pygame.K_s)
    assert sim.seed == seed + 1
    assert sim.mission.seed == seed + 1
    assert len(sim.mission.events) == 1
    assert sim.base_grid == grid  # the map stays


def test_speed_keys_change_and_clamp_the_speed_index(sim: SimApp) -> None:
    start = sim.speed_index
    assert start == theme.DEFAULT_SPEED_INDEX
    key(sim, pygame.K_PLUS)
    assert sim.speed_index == start + 1
    key(sim, pygame.K_EQUALS)  # unshifted '+' key
    assert sim.speed_index == start + 2
    key(sim, pygame.K_MINUS)
    assert sim.speed_index == start + 1
    key(sim, pygame.K_KP_MINUS)
    assert sim.speed_index == start
    for _ in range(len(theme.SPEEDS) + 2):
        sim.handle_key(pygame.K_PLUS)
    assert sim.speed_index == len(theme.SPEEDS) - 1
    assert sim.speed == theme.SPEEDS[-1]
    for _ in range(len(theme.SPEEDS) + 2):
        sim.handle_key(pygame.K_MINUS)
    assert sim.speed_index == 0
    assert sim.speed == theme.SPEEDS[0]


def test_bracket_keys_change_slip_p_by_step_and_clamp(sim: SimApp) -> None:
    assert sim.slip_p == pytest.approx(DEFAULT_SLIP_P)
    key(sim, pygame.K_n)
    key(sim, pygame.K_RIGHTBRACKET)
    assert sim.slip_p == pytest.approx(DEFAULT_SLIP_P + theme.SLIP_P_STEP)
    assert sim.mission.world_config.slip_p == pytest.approx(sim.slip_p)
    assert len(sim.mission.events) == 1  # a new mission with the new p
    key(sim, pygame.K_LEFTBRACKET)
    assert sim.slip_p == pytest.approx(DEFAULT_SLIP_P)
    presses = round(MAX_SLIP_P / theme.SLIP_P_STEP) + 3
    for _ in range(presses):
        sim.handle_key(pygame.K_RIGHTBRACKET)
    assert sim.slip_p == pytest.approx(MAX_SLIP_P)
    assert sim.mission.world_config.slip_p == pytest.approx(MAX_SLIP_P)
    for _ in range(presses):
        sim.handle_key(pygame.K_LEFTBRACKET)
    assert sim.slip_p == pytest.approx(0.0)
    assert sim.mission.world_config.slip_p == pytest.approx(0.0)
    sim.render()


@pytest.mark.parametrize(
    ("k", "index"),
    [(pygame.K_2, 1), (pygame.K_3, 2), (pygame.K_1, 0), (pygame.K_KP2, 1)],
)
def test_number_keys_switch_agent(sim: SimApp, k: int, index: int) -> None:
    if index == 0:
        key(sim, pygame.K_3)
    key(sim, pygame.K_n)
    key(sim, k)
    assert sim.agent_name == AGENT_NAMES[index]
    assert sim.mission.agent.name == AGENT_NAMES[index]
    assert len(sim.mission.events) == 1


def test_m_cycles_scenarios(sim: SimApp, maps_tmp: Path) -> None:
    assert sim.scenario_name == "deck_demo"
    key(sim, pygame.K_m)
    assert sim.scenario_name == "deck_figure"
    figure = TrueGrid.load(maps_tmp / "deck_figure.txt")
    assert np.array_equal(sim.mission.world.grid.cells, figure.cells)
    assert sim.mission.world.pos == figure.start
    for _ in range(len(sim.scenarios) - 1):  # full cycle back to the first map
        sim.handle_key(pygame.K_m)
    assert sim.scenario_name == "deck_demo"


def test_editor_clicks_and_ctrl_s_save(sim: SimApp, maps_tmp: Path) -> None:
    demo = TrueGrid.load(maps_tmp / "deck_demo.txt")
    key(sim, pygame.K_SPACE)  # playing; E must pause
    key(sim, pygame.K_e)
    assert sim.editor and sim.paused
    assert sim.edit_grid is not None
    events = len(sim.mission.events)
    key(sim, pygame.K_n)  # mission keys are ignored while editing
    assert len(sim.mission.events) == events
    # plain click toggles debris on an open cell, and back
    click(sim, (0, 2))
    assert sim.edit_grid.cells[0, 2] == Cell.DEBRIS
    click(sim, (0, 2))
    assert sim.edit_grid.cells[0, 2] == Cell.OPEN
    click(sim, (2, 1))  # known debris of deck_demo → open
    assert demo.cells[2, 1] == Cell.DEBRIS
    assert sim.edit_grid.cells[2, 1] == Cell.OPEN
    # shift-click moves the start, ctrl-click the survivor
    click(sim, (1, 0), pygame.KMOD_LSHIFT)
    assert sim.edit_grid.start == (1, 0)
    click(sim, (0, 3), pygame.KMOD_LCTRL)
    assert sim.edit_grid.survivor == (0, 3)
    assert sim.edit_grid.cells[0, 3] == Cell.SURVIVOR
    assert sim.edit_grid.cells[demo.survivor] == Cell.OPEN
    # Ctrl+S writes the edited map into the (temporary) maps directory
    before = set(maps_tmp.iterdir())
    key(sim, pygame.K_s, pygame.KMOD_LCTRL)
    new = set(maps_tmp.iterdir()) - before
    assert len(new) == 1
    saved_path = new.pop()
    assert saved_path.suffix == ".txt"
    saved = TrueGrid.load(saved_path)
    assert np.array_equal(saved.cells, sim.edit_grid.cells)
    assert (saved.start, saved.survivor) == ((1, 0), (0, 3))
    assert any(e.path == saved_path for e in sim.scenarios)
    # E again leaves the editor and restarts the mission on the edited map
    key(sim, pygame.K_e)
    assert not sim.editor
    assert sim.mission.world.pos == (1, 0)
    assert sim.mission.world.grid.survivor == (0, 3)
    assert len(sim.mission.events) == 1


def test_ctrl_s_outside_editor_saves_current_map(sim: SimApp, maps_tmp: Path) -> None:
    before = set(maps_tmp.iterdir())
    key(sim, pygame.K_s, pygame.KMOD_RCTRL)
    new = set(maps_tmp.iterdir()) - before
    assert len(new) == 1
    saved = TrueGrid.load(new.pop())
    assert np.array_equal(saved.cells, sim.mission.grid.cells)  # the initial true map
    assert sim.seed == DEMO_SEED  # Ctrl+S is not S (new seed)


def test_g_records_a_gif(sim: SimApp, tmp_path: Path) -> None:
    key(sim, pygame.K_SPACE)
    key(sim, pygame.K_g)
    assert sim.recorder.active
    for _ in range(RECORD_FRAMES):
        sim.frame(theme.HEADLESS_FRAME_MS)
    captured = sim.recorder.frame_count
    assert captured >= 2
    key(sim, pygame.K_g)
    assert not sim.recorder.active
    assert len(sim.recordings) == 1
    gif = sim.recordings[0]
    assert gif.is_file() and gif.suffix == ".gif"
    assert gif.parent == tmp_path / "recordings"
    frames = imageio.mimread(gif)
    assert 1 <= len(frames) <= captured  # identical consecutive frames may be merged
    width, height = theme.WINDOW_SIZE
    assert frames[0].shape[:2] == (round(height * theme.RECORD_SCALE),
                                   round(width * theme.RECORD_SCALE))


def test_view_toggles_h_t_p_c_and_esc(sim: SimApp) -> None:
    for k, attr in ((pygame.K_h, "show_help"), (pygame.K_t, "show_true"),
                    (pygame.K_p, "show_policy")):
        assert getattr(sim, attr) is False
        key(sim, k)
        assert getattr(sim, attr) is True
        if k != pygame.K_h:
            key(sim, k)
            assert getattr(sim, attr) is False
    # Esc with the help open only closes the help
    key(sim, pygame.K_ESCAPE)
    assert not sim.show_help and sim.running
    key(sim, pygame.K_h)
    key(sim, pygame.K_h)
    assert not sim.show_help
    assert sim.palette is theme.DEFAULT_PALETTE
    key(sim, pygame.K_c)
    assert sim.palette is theme.HIGH_CONTRAST_PALETTE
    assert sim.grid_view.palette is theme.HIGH_CONTRAST_PALETTE
    key(sim, pygame.K_c)
    assert sim.palette is theme.DEFAULT_PALETTE
    key(sim, pygame.K_ESCAPE)
    assert not sim.running


def test_high_contrast_option_and_unknown_key(tmp_path: Path, maps_tmp: Path) -> None:
    app = SimApp(sim_options(tmp_path, maps_tmp, high_contrast=True))
    try:
        assert app.palette is theme.HIGH_CONTRAST_PALETTE
        assert app.handle_key(pygame.K_F12) is False
        app.render()
    finally:
        app.close()


# --- (4) replay --------------------------------------------------------------------------------
def test_replay_scrubbing_matches_recorded_events(deck_run: Path) -> None:
    record = load_run(deck_run)
    events = record.events
    n = len(events) - 1
    assert n >= 3
    app = ReplayApp(record, ReplayOptions(headless=True, autoplay=False))

    def check(k: int) -> None:
        assert app.k == k
        assert app.event == events[k]
        surface = app.render()
        pos = events[k].new_pos
        assert app._scene().robot == pos
        assert centre_pixel(app.grid_view, surface, pos) == app.palette.robot
        assert app._panel_info().metrics[REPLAY_STEPS_LABEL] == f"{events[k].t}/{events[-1].t}"

    try:
        check(0)
        for k in range(1, n + 1):
            key(app, pygame.K_RIGHT)
            check(k)
        key(app, pygame.K_RIGHT)  # clamped at the end
        check(n)
        for k in range(n - 1, -1, -1):
            key(app, pygame.K_LEFT)
            check(k)
        key(app, pygame.K_LEFT)  # clamped at the start
        check(0)
        key(app, pygame.K_END)
        check(n)
        key(app, pygame.K_HOME)
        check(0)
        key(app, pygame.K_RIGHT, pygame.KMOD_LSHIFT)
        check(min(n, theme.SCRUB_BIG_STEP))
        key(app, pygame.K_LEFT, pygame.KMOD_LSHIFT)
        check(0)
        # Space at the end restarts from step 0 and plays
        key(app, pygame.K_END)
        assert app.paused
        key(app, pygame.K_SPACE)
        assert not app.paused
        check(0)
        assert play_one_step(app) == 1
        check(1)
    finally:
        app.close()


def test_replay_cli_headless(deck_run: Path, tmp_path: Path) -> None:
    from rescue.replay import main

    shot = tmp_path / "replay.png"
    assert main([str(deck_run), "--headless", "--frames", "5", "--step", "-1",
                 "--screenshot", str(shot)]) == 0
    assert png_size(shot) == theme.WINDOW_SIZE


# --- (5) recorder ------------------------------------------------------------------------------
def test_recorder_writes_a_readable_gif(tmp_path: Path, pg: None) -> None:
    surface = pygame.Surface((64, 48))
    recorder = Recorder()
    assert not recorder.active
    target = recorder.start(tmp_path / "out" / "clip.gif", fps=10, scale=1.0)
    assert recorder.active and target.suffix == ".gif"
    for i in range(RECORDER_FRAMES):
        surface.fill((40 * i, 255 - 40 * i, 90))  # distinct frames are never merged
        recorder.add_frame(surface)
    assert recorder.frame_count == RECORDER_FRAMES
    path = recorder.stop()
    assert path == target and path.is_file()
    assert not recorder.active
    frames = imageio.mimread(path)
    assert len(frames) == RECORDER_FRAMES
    assert frames[0].shape[:2] == (48, 64)


def test_recorder_without_frames_writes_nothing(tmp_path: Path, pg: None) -> None:
    recorder = Recorder()
    recorder.start(tmp_path / "empty.gif")
    assert recorder.stop() is None
    assert not (tmp_path / "empty.gif").exists()


# --- (6) plan export ---------------------------------------------------------------------------
def test_export_plan_tree_png_and_text(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig()
    mission = Mission(grid, make_agent(AGENT_NAMES[0], make_transition(grid, cfg)), cfg,
                      seed=DEMO_SEED)
    mission.start()
    plan = mission.agent.snapshots[0].plan
    assert plan is not None
    out = export_plan_tree(plan, tmp_path / "figs" / "deck_plan.png", title="deck_demo")
    assert out.is_file()
    width, height = png_size(out)
    assert width > 0 and height > 0
    text = export_plan_text(plan)
    assert "E @ (1,1)" in text
    assert text.splitlines()[0] == "E @ (1,1)"
    assert "slipped → ↺ retry E" in text


# --- (7) grid geometry -------------------------------------------------------------------------
@pytest.mark.parametrize(("rows", "cols"), [(4, 4), (7, 11), (30, 30), (3, 40)])
def test_grid_view_pos_at_cell_rect_round_trip(pg: None, rows: int, cols: int) -> None:
    rect = pygame.Rect(10, 20, 600, 400)
    view = GridView(rect, theme.DEFAULT_PALETTE, theme.load_fonts())
    assert view.pos_at(rect.center) is None  # before any layout
    view.layout(rows, cols)
    board = view.board_rect
    assert rect.contains(board)
    for r in range(rows):
        for c in range(cols):
            cell = view.cell_rect((r, c))
            assert cell.size == (view.cell, view.cell)
            assert board.contains(cell)
            assert view.pos_at(cell.center) == (r, c)
            assert view.pos_at(cell.topleft) == (r, c)
            assert view.pos_at((cell.right - 1, cell.bottom - 1)) == (r, c)
    for outside in ((board.left - 1, board.top), (board.left, board.top - 1),
                    (board.right, board.top), (board.left, board.bottom)):
        assert view.pos_at(outside) is None


def test_grid_view_draws_robot_at_cell_rect(pg: None) -> None:
    surface = pygame.display.set_mode(theme.WINDOW_SIZE)
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    view = GridView(theme.grid_area(), theme.DEFAULT_PALETTE, theme.load_fonts())
    belief = np.zeros(grid.shape, dtype=np.int8)
    scene = GridScene(rows=grid.rows, cols=grid.cols, belief=belief, true_cells=grid.cells,
                      high_slip=grid.high_slip, robot=(1, 2), start=grid.start,
                      survivor=grid.survivor)
    view.draw(surface, scene)
    assert view.pos_at(view.cell_centre((1, 2))) == (1, 2)
    assert centre_pixel(view, surface, (1, 2)) == theme.DEFAULT_PALETTE.robot


# --- (8) Phase 4 fix pass: visual QA and controls regressions ------------------------------------
#: WCAG AA minimum contrast of text (and the target of the fog-of-war '?').
MIN_TEXT_CONTRAST = 4.5
#: Map size / seed, world seed and slip p of the plan-tree highlight test: a mission in which
#: the robot walks below the panel's tree depth several times (14 re-rooted frames).
HIGHLIGHT_GRID_SIZE = 16
HIGHLIGHT_MAP_SEED = 1
HIGHLIGHT_SEED = 1
HIGHLIGHT_P = 0.2
#: A long event-log line (wider than the panel).
LONG_LOG_TEXT = ("aborted: deviated from the path: at (1,2), expected (2,2) after the move; "
                 "the survivor cannot be reached from here")


def text_pairs(pal: theme.Palette) -> list[tuple[str, tuple[int, int, int], tuple[int, int, int]]]:
    """Every (name, text colour, background) pair the UI draws text with."""
    pairs = [
        ("text/panel", pal.text, pal.panel_bg),
        ("text/background", pal.text, pal.background),
        ("text_dim/panel", pal.text_dim, pal.panel_bg),
        ("text_dim/background", pal.text_dim, pal.background),
        ("highlight_text/highlight_row", pal.highlight_text, pal.highlight_row),
        ("editing badge", pal.background, pal.highlight),
        ("REC/panel", pal.log_failure_text, pal.panel_bg),
        ("start S/start", pal.start_text, pal.start),
        ("? / unknown", pal.unknown_glyph, pal.unknown),
    ]
    for name in ("edge_ok_text", "edge_slip_text", "edge_skid_text", "log_replan_text",
                 "log_success_text", "log_failure_text", "log_warn_text"):
        pairs.append((f"{name}/background", getattr(pal, name), pal.background))
    for status, colour in pal.badge.items():
        pairs.append((f"badge {status.name}", pal.badge_text, colour))
    return pairs


@pytest.mark.parametrize("palette", theme.PALETTES, ids=lambda p: p.name)
def test_every_text_colour_meets_wcag_aa(palette: theme.Palette) -> None:
    low = [(name, round(theme.contrast_ratio(fg, bg), 2)) for name, fg, bg in text_pairs(palette)
           if theme.contrast_ratio(fg, bg) < MIN_TEXT_CONTRAST]
    assert not low, f"{palette.name}: contrast below {MIN_TEXT_CONTRAST}:1 for {low}"
    assert theme.relative_luminance(palette.unknown_glyph) <= theme.relative_luminance(
        palette.text)  # the '?' stays dimmer than body text


def test_plan_tree_always_highlights_the_robot_state(tmp_path: Path, maps_tmp: Path) -> None:
    grid = corner_grid(HIGHLIGHT_GRID_SIZE, HIGHLIGHT_MAP_SEED)
    grid.save(maps_tmp / "highlight.txt")
    app = SimApp(sim_options(tmp_path, maps_tmp, map_name="highlight.txt", slip_p=HIGHLIGHT_P,
                             seed=HIGHLIGHT_SEED))
    rerooted = 0
    try:
        while not app.mission.done:
            app.step_once()
            app.render()
            tree, pos = app.tree, app.mission.world.pos
            if tree.plan is None or pos == app.mission.grid.survivor:
                continue
            rows, lit = tree.rows, tree.highlighted
            assert lit, f"no highlighted row at {pos} (t = {app.mission.world.t})"
            assert all(rows[i].state == pos and rows[i].kind != "info" for i in lit)
            assert tree.scroll <= lit[0] < tree.scroll + tree.visible_rows  # in view
            rerooted += tree.display_root is not None
    finally:
        app.close()
    assert rerooted > 0, "the mission never went below the tree depth: test map too easy"


def test_text_cache_wrap_keeps_every_word_within_width(pg: None) -> None:
    from rescue.viz.tree_view import text_cache

    font = theme.load_fonts().mono
    width = 220
    lines = text_cache.wrap(font, LONG_LOG_TEXT, width)
    assert len(lines) > 1
    assert all(text_cache.width(font, line) <= width for line in lines)
    assert " ".join(lines).split() == LONG_LOG_TEXT.split()
    assert text_cache.wrap(font, "short", width) == ("short",)


def test_long_log_lines_wrap_instead_of_being_cut(sim: SimApp) -> None:
    from rescue.viz.panel import LOG_VISIBLE_LINES

    sim.log.add(LONG_LOG_TEXT, "failure")
    surface = sim.render()
    assert surface.get_size() == theme.WINDOW_SIZE
    _, end = sim.log.bounds(LOG_VISIBLE_LINES)
    assert sim.log.lines[end - 1] == (LONG_LOG_TEXT, "failure")


def test_editor_panel_shows_no_stale_mission_state(sim: SimApp) -> None:
    from rescue.viz.sim_app import EDITOR_TREE_TEXT

    key(sim, pygame.K_n)
    key(sim, pygame.K_e)
    info = sim._panel_info()
    assert info.editing and not info.metrics
    assert sim.tree.plan is None
    assert [r.kind for r in sim.tree.rows] == ["info"] * len(sim.tree.rows)
    assert sim.tree.rows[0].text.split()[0] == EDITOR_TREE_TEXT.split()[0]
    key(sim, pygame.K_e)
    assert not sim._panel_info().editing
    assert sim.tree.plan is not None


def test_editor_starts_from_the_initial_map_not_the_aftershock_world(
    tmp_path: Path, maps_tmp: Path
) -> None:
    (maps_tmp / "open.txt").write_text("S.......\n........\n........\n........\n.......P\n",
                                       encoding="utf-8")
    app = SimApp(sim_options(tmp_path, maps_tmp, map_name="open.txt", aftershock_q=1.0,
                             aftershock_k=3))
    try:
        original = app.base_grid.copy()
        for _ in range(4):
            app.handle_key(pygame.K_n)
        assert app.mission.world.grid != original  # aftershocks added debris
        app.handle_key(pygame.K_e)
        assert app.edit_grid == original
        app.handle_key(pygame.K_e)  # leave without edits
        assert app.base_grid == original and app.base_grid.name == original.name
        app.handle_key(pygame.K_r)
        assert app.mission.grid == original
        path = app.save_map()  # outside the editor: the initial map, not the aftershock one
        assert np.array_equal(TrueGrid.load(path).cells, original.cells)
    finally:
        app.close()


def test_editor_debris_toggle_keeps_high_slip_terrain(tmp_path: Path, maps_tmp: Path) -> None:
    (maps_tmp / "tilde.txt").write_text("S.~~.\n.....\n....P\n", encoding="utf-8")
    app = SimApp(sim_options(tmp_path, maps_tmp, map_name="tilde.txt"))
    try:
        app.render()
        key(app, pygame.K_e)
        assert app.edit_grid is not None and app.edit_grid.high_slip[0, 2]
        click(app, (0, 2))
        assert app.edit_grid.cells[0, 2] == Cell.DEBRIS and not app.edit_grid.high_slip[0, 2]
        click(app, (0, 2))
        assert app.edit_grid.cells[0, 2] == Cell.OPEN and app.edit_grid.high_slip[0, 2]
        click(app, (0, 3), pygame.KMOD_LSHIFT)  # the start may not be '~' ...
        assert app.edit_grid.start == (0, 3) and not app.edit_grid.high_slip[0, 3]
        click(app, (0, 0), pygame.KMOD_LSHIFT)  # ... and gets its '~' back when it leaves
        assert app.edit_grid.high_slip[0, 3]
        key(app, pygame.K_e)
        assert app.base_grid.name != "edited"  # nothing changed in the end
    finally:
        app.close()


def test_unsolvable_edit_warns_and_tree_says_unreachable(sim: SimApp) -> None:
    from rescue.runner import MissionStatus

    key(sim, pygame.K_e)
    survivor = sim.edit_grid.survivor if sim.edit_grid is not None else (3, 3)
    for p in ((survivor[0] - 1, survivor[1]), (survivor[0], survivor[1] - 1)):
        if sim.edit_grid is not None and sim.edit_grid.cells[p] != Cell.DEBRIS:
            click(sim, p)
    key(sim, pygame.K_e)
    assert sim.mission.shortest_path is None
    assert any("unsolvable" in text for text, _ in sim.log.lines)
    for _ in range(sim.mission.budget):
        if sim.mission.done:
            break
        key(sim, pygame.K_n)
    assert sim.mission.status is MissionStatus.UNREACHABLE
    texts = " ".join(r.text for r in sim.tree.rows)
    assert "unreachable" in texts and "No plan yet" not in texts


def test_s_regenerates_a_generated_scenario(
    tmp_path: Path, maps_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    from rescue.viz import sim_app

    def generate(name: str, rows: int, cols: int, seed: int) -> TrueGrid:
        cells = ["." * 5 for _ in range(3)]
        col = 1 + seed % 3
        cells[1] = cells[1][:col] + "#" + cells[1][col + 1:]
        text = "\n".join(cells)
        text = "S" + text[1:-1] + "P"
        return TrueGrid.from_ascii(text, name=name)

    fake = types.SimpleNamespace(GENERATED_SCENARIOS=("fake",), SCENARIOS={"fake": None},
                                 generate=generate)
    monkeypatch.setattr(sim_app, "_mapgen", lambda: fake)
    app = SimApp(sim_options(tmp_path, maps_tmp, map_path=None, scenario="fake", seed=0))
    try:
        first = app.base_grid
        key(app, pygame.K_s)
        assert app.seed == 1 and app.base_grid != first
        assert app.base_grid == generate("fake", 0, 0, 1)
    finally:
        app.close()


def test_trace_layer_redraws_after_a_reset_without_a_render(sim: SimApp) -> None:
    for _ in range(4):
        key(sim, pygame.K_n)
    for k in (pygame.K_s, pygame.K_1):
        sim.handle_key(k)
        for _ in range(6):  # several steps between two renders
            if not sim.mission.done:
                sim.handle_key(pygame.K_n)
        sim.render()
        fresh = GridView(sim.grid_view.rect, sim.palette, sim.fonts)
        fresh.draw(pygame.Surface(theme.WINDOW_SIZE), sim._scene())
        assert sim.grid_view._trace_layer is not None and fresh._trace_layer is not None
        got = pygame.surfarray.array_alpha(sim.grid_view._trace_layer)
        want = pygame.surfarray.array_alpha(fresh._trace_layer)
        assert np.array_equal(got, want)


def test_replay_path_arrows_match_the_live_agent(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig(slip_p=0.3)
    agent = make_agent("astar_replan", make_transition(grid, cfg))
    mission = Mission(grid, agent, cfg, seed=3, scenario="deck_demo")
    mission.start()
    live = [dict(agent.policy_arrows())]
    while not mission.done:
        mission.step()
        live.append(dict(agent.policy_arrows()))
    app = ReplayApp(load_run(save_run(mission, runs_dir=tmp_path)),
                    ReplayOptions(headless=True, autoplay=False))
    try:
        app.show_policy = True
        for k in [*range(app.n + 1), *range(app.n, -1, -1)]:
            app.set_step(k)
            assert dict(app._scene().policy) == live[k], f"step {k}"
    finally:
        app.close()


def test_replay_metrics_follow_the_simulator_order(sim: SimApp, deck_run: Path) -> None:
    from rescue.viz.replay_app import PLAN_MS_ALL_LABEL
    from rescue.viz.sim_app import STEPS_LABEL

    app = ReplayApp(load_run(deck_run), ReplayOptions(headless=True, autoplay=False))
    try:
        replay = list(app._panel_info().metrics)
        rename = {REPLAY_STEPS_LABEL: STEPS_LABEL, PLAN_MS_ALL_LABEL: "Plan ms"}
        assert [rename.get(label, label) for label in replay] == list(sim._metric_values())
        assert app.bar_rect.right < app.strip_rect.right  # the label is beside the bar
    finally:
        app.close()


def test_rescue_keeps_the_survivor_cross_visible(deck_run: Path) -> None:
    app = ReplayApp(load_run(deck_run), ReplayOptions(headless=True, autoplay=False, step=-1))
    try:
        surface = app.render()
        view, survivor = app.grid_view, app.record.grid.survivor
        assert app._scene().robot == survivor
        cx, cy = view.cell_centre(survivor)
        arm = round(view.cell * theme.SURVIVOR_LEN_FRAC / 2) - 2  # just inside the arm's end
        colour = surface.get_at((cx + arm, cy))  # survivor red (under the envelope tint)
        assert colour.r > 2 * colour.g and colour.r > 2 * colour.b
        assert centre_pixel(view, surface, survivor) == app.palette.robot
    finally:
        app.close()


def test_unknown_cells_with_an_arrow_move_the_question_mark(pg: None) -> None:
    from rescue.model import Action

    surface = pygame.display.set_mode(theme.WINDOW_SIZE)
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    view = GridView(theme.grid_area(), theme.DEFAULT_PALETTE, theme.load_fonts())
    belief = np.zeros(grid.shape, dtype=np.int8)
    scene = GridScene(rows=grid.rows, cols=grid.cols, belief=belief, true_cells=grid.cells,
                      high_slip=grid.high_slip, robot=None, start=grid.start,
                      survivor=grid.survivor, show_policy=True, policy={(2, 2): Action.E})
    view.draw(surface, scene)
    rect = view.cell_rect((2, 2))
    corner = pygame.Rect(rect.x + rect.w * 3 // 4, rect.y + 2, rect.w // 4 - 2, rect.h // 3)
    pal = theme.DEFAULT_PALETTE
    colours = {tuple(surface.get_at((x, y)))[:3] for x in range(corner.left, corner.right)
               for y in range(corner.top, corner.bottom)}
    assert pal.unknown_glyph in colours  # the corner '?' is drawn
    plain = view.cell_rect((3, 0))  # unknown, no arrow: the '?' is centred, the corner empty
    plain_corner = corner.move(plain.x - rect.x, plain.y - rect.y)
    assert {tuple(surface.get_at((x, y)))[:3] for x in range(plain_corner.left, plain_corner.right)
            for y in range(plain_corner.top, plain_corner.bottom)} == {pal.unknown}


def test_export_lr_layout_keeps_a_readable_aspect_ratio(tmp_path: Path) -> None:
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig()
    mission = Mission(grid, make_agent(AGENT_NAMES[0], make_transition(grid, cfg)), cfg,
                      seed=DEMO_SEED)
    mission.start()
    plan = mission.agent.snapshots[0].plan
    assert plan is not None
    for layout in ("lr", "tb"):
        width, height = png_size(export_plan_tree(plan, tmp_path / f"{layout}.png",
                                                  max_depth=4, layout=layout))
        ratio = max(width, height) / min(width, height)
        assert ratio <= theme.EXPORT_MAX_ASPECT + 0.01, (layout, width, height)
    with pytest.raises(ValueError):
        export_plan_tree(plan, tmp_path / "bad.png", layout="diagonal")
