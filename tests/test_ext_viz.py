"""The extensions in the interactive simulator (docs/plan.md §3.15.3).

Headless (SDL dummy drivers, selected before pygame is imported). Covers the three modes of
:class:`~rescue.viz.ext_sim_app.ExtSimApp` (100 headless frames + screenshot each), its keys,
its log lines, the CLI (``python -m rescue.sim --ext …``), the new GridScene marks (pixel
probes), and rule zero for rendering: a core scene / panel renders exactly as with the
pre-round-2 commit's ``grid_view.py`` / ``panel.py`` (:data:`RULE_ZERO_BASE`), and
``scripts/make_figures.py`` reproduces the committed figures byte for byte.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import ast
import importlib.util
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from rescue import sim  # noqa: E402
from rescue.agents import AGENT_NAMES, make_agent  # noqa: E402
from rescue.config import (  # noqa: E402
    DEFAULT_SLIP_P,
    DEMO_SEED,
    EXT_DEMO_MULTI_K,
    EXT_DEMO_NOISE,
    EXT_DEMO_SCENARIO,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
    WorldConfig,
)
from rescue.ext.agents import EXT_AGENT_NAMES, make_ext_agent  # noqa: E402
from rescue.ext.mission import format_ext_trace, run_ext_mission  # noqa: E402
from rescue.ext.scenarios import multi_survivor_grid  # noqa: E402
from rescue.grid import Cell, TrueGrid  # noqa: E402
from rescue.runner import Mission, MissionStatus, make_transition  # noqa: E402
from rescue.viz import theme  # noqa: E402
from rescue.viz.ext_sim_app import (  # noqa: E402
    AFTERSHOCKS_LABEL,
    EDITOR_DISABLED_TEXT,
    EXT_MODES,
    FALLBACKS_LABEL,
    FALSE_POSITIVE_ARG,
    FP_UNSUPPORTED_TEXT,
    PHANTOM_HINT_TEXT,
    ExtSimApp,
    ExtSimOptions,
    false_positive_supported,
    mode_settings,
)
from rescue.viz.grid_view import (  # noqa: E402
    GridScene,
    GridView,
    prob_shade_levels,
    trace_from_events,
)
from rescue.viz.panel import EventLog, Panel, PanelInfo  # noqa: E402
from rescue.viz.tree_view import TreeView  # noqa: E402

pytestmark = pytest.mark.viz

#: Repository root, maps, committed figures and the figure scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Frames of the per-mode headless smoke test (brief §9, as for the core simulator).
SMOKE_FRAMES = 100
#: Steps taken before comparing a reset (R) replay.
REPLAY_STEPS = 12
#: Size of the boards drawn directly through GridView in the pixel probes.
PROBE_SIZE = 5
#: Steps of the core deck_demo mission drawn in the rule-zero comparison.
CORE_STEPS = 6
#: The commit before round 2 (plan §3.15): the reference ``grid_view.py`` / ``panel.py`` of the
#: rule-zero rendering tests. Pinned (not HEAD), so the tests keep their meaning once the round
#: is committed.
RULE_ZERO_BASE = "a948f27"
#: A noisy mission with false positives whose relaxed-passability fallback fires (seed 8).
FP_RATE = 0.2
FP_SEED = 8
#: Aftershock probability of the panel test.
AFTERSHOCK_Q = 0.2
#: Board size whose cells are below theme.TARGET_COMPACT_MAX_CELL (25 px cells).
LARGE_BOARD = 30
#: P(debris) of a possible phantom in the pixel probe.
PHANTOM_P = 0.9
#: Core figures that must stay byte-identical (rule zero, plan §3.15), and the extension ones.
CORE_PNG_FIGURES = ("sim_deck_demo.png", "sim_deck_figure.png")
EXT_FIGURES = ("ext_search.png", "ext_multi.png", "ext_noisy.png")


# --- helpers -----------------------------------------------------------------------------------
def png_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's IHDR chunk."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE, f"{path} is not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def ext_options(tmp_path: Path, **overrides: Any) -> ExtSimOptions:
    """Headless, paused extension-simulator options writing only below ``tmp_path``."""
    settings: dict[str, Any] = {
        "headless": True,
        "autoplay": False,
        "maps_dir": str(MAPS_DIR),
        "recordings_dir": str(tmp_path / "recordings"),
        "runs_dir": str(tmp_path / "runs"),
    }
    settings.update(overrides)
    return ExtSimOptions(**settings)


def run_to_end(app: ExtSimApp) -> None:
    """Single-step the app's mission until it is over."""
    while not app.mission.done:
        assert app.step_once()


def log_texts(app: ExtSimApp) -> list[str]:
    return [text for text, _kind in app.log.lines]


def rgb(surface: Any, xy: tuple[int, int]) -> tuple[int, int, int]:
    c = surface.get_at(xy)
    return (c.r, c.g, c.b)


def load_base_module(name: str, relpath: str, tmp_path: Path) -> ModuleType:
    """:data:`RULE_ZERO_BASE`'s version of ``relpath`` imported as module ``name`` (skips
    without git or without that commit, e.g. in a shallow clone; the committed-figure byte
    checks below stay the permanent guard)."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    done = subprocess.run([git, "-C", str(REPO_ROOT), "show", f"{RULE_ZERO_BASE}:{relpath}"],
                          capture_output=True, check=False)
    if done.returncode != 0:
        pytest.skip(f"git show {RULE_ZERO_BASE}:{relpath} failed")
    path = tmp_path / f"{name}.py"
    path.write_bytes(done.stdout)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while being created
    spec.loader.exec_module(module)
    return module


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- fixtures ----------------------------------------------------------------------------------
@pytest.fixture
def app(tmp_path: Path) -> Iterator[ExtSimApp]:
    """A paused headless extension simulator in ``search`` mode (closed afterwards)."""
    a = ExtSimApp(ext_options(tmp_path))
    a.render()
    yield a
    a.close()


@pytest.fixture
def pg() -> Iterator[Any]:
    """pygame with a display (for direct GridView / Panel drawing); yields the fonts."""
    pygame.init()
    pygame.display.set_mode(theme.WINDOW_SIZE)
    yield theme.load_fonts()
    pygame.quit()


# --- (1) every mode: 100 headless frames and a screenshot --------------------------------------
@pytest.mark.parametrize("mode", EXT_MODES)
def test_each_mode_renders_headless_frames_and_saves_screenshot(tmp_path: Path,
                                                                mode: str) -> None:
    shot = tmp_path / "shots" / f"{mode}.png"
    a = ExtSimApp(ext_options(tmp_path, mode=mode, autoplay=None, frames=SMOKE_FRAMES,
                              screenshot=str(shot), speed_index=len(theme.SPEEDS) - 1))
    assert not a.paused  # headless plays by default
    renders = 0
    original = a.render

    def counting_render() -> Any:
        nonlocal renders
        renders += 1
        return original()

    a.render = counting_render  # type: ignore[method-assign]
    assert a.run() == 0
    assert renders == SMOKE_FRAMES
    assert png_size(shot) == theme.WINDOW_SIZE
    assert a.mission.events[-1].t > 0  # it played


def test_defaults_are_the_ext_demo_mission(tmp_path: Path) -> None:
    """Same scenario, size, seed, p and agent as ``python -m rescue.ext.demo``: same trace."""
    for mode in EXT_MODES:
        a = ExtSimApp(ext_options(tmp_path, mode=mode))
        try:
            run_to_end(a)
            s = mode_settings(mode)
            grid = multi_survivor_grid(EXT_DEMO_SCENARIO, *EXT_DEMO_SIZE, EXT_DEMO_SEED, s.k)
            cfg = WorldConfig(slip_p=DEFAULT_SLIP_P)
            ref = run_ext_mission(grid, make_ext_agent(EXT_AGENT_NAMES[0],
                                                       make_transition(grid, cfg)),
                                  cfg, EXT_DEMO_SEED, survivors_known=s.survivors_known,
                                  false_negative=s.false_negative)
            assert format_ext_trace(a.mission.events) == format_ext_trace(ref.events)
            assert a.mission.status is ref.status
        finally:
            a.close()


def test_mode_settings_defaults_and_overrides() -> None:
    assert mode_settings("search") == mode_settings("search", None, None, None)
    assert not mode_settings("search").survivors_known
    assert mode_settings("multi").k == EXT_DEMO_MULTI_K
    assert mode_settings("noisy").false_negative == EXT_DEMO_NOISE
    assert mode_settings("search", k=2, noise=0.2).k == 2
    with pytest.raises(ValueError):
        mode_settings("nope")
    with pytest.raises(ValueError):
        mode_settings("multi", k=0)
    with pytest.raises(ValueError):
        mode_settings("noisy", noise=1.5)


# --- (2) keys ----------------------------------------------------------------------------------
def test_x_cycles_modes_and_rebuilds_the_map(app: ExtSimApp) -> None:
    seen = []
    for _ in range(len(EXT_MODES) + 1):
        seen.append(app.settings.mode)
        assert app.mission.k == app.settings.k == len(app.base_grid.survivors)
        assert app.handle_key(pygame.K_x)
        app.render()
    assert seen == [*EXT_MODES, EXT_MODES[0]]


def test_number_keys_switch_extension_agents(app: ExtSimApp) -> None:
    assert app.agent_name == EXT_AGENT_NAMES[0]
    assert app.handle_key(pygame.K_2)
    assert app.agent_name == app.mission.agent.name == EXT_AGENT_NAMES[1]
    app.render()
    assert app.handle_key(pygame.K_3)  # no third extension agent: logged, nothing changes
    assert app.agent_name == EXT_AGENT_NAMES[1]
    assert app.handle_key(pygame.K_1)
    assert app.mission.agent.name == EXT_AGENT_NAMES[0]


def test_mission_keys_work_alike_while_the_help_is_open(app: ExtSimApp) -> None:
    """X and 1 / 2 behave like the simulator's other mission keys (N, R, ...) under the help
    overlay: they act, and the overlay stays open."""
    assert app.handle_key(pygame.K_h) and app.show_help
    assert app.handle_key(pygame.K_2)
    assert app.agent_name == app.mission.agent.name == EXT_AGENT_NAMES[1]
    mode = app.settings.mode
    assert app.handle_key(pygame.K_x)
    assert app.settings.mode != mode
    t0 = app.mission.world.t
    assert app.handle_key(pygame.K_n) and app.mission.world.t == t0 + 1
    assert app.show_help
    app.render()


def test_r_replays_the_same_seed_and_s_changes_it(app: ExtSimApp) -> None:
    for _ in range(REPLAY_STEPS):
        app.step_once()
    first = [(e.pos, e.new_pos, e.outcome) for e in app.mission.events]
    seed = app.seed
    assert app.handle_key(pygame.K_r)
    for _ in range(REPLAY_STEPS):
        app.step_once()
    assert app.seed == seed
    assert [(e.pos, e.new_pos, e.outcome) for e in app.mission.events] == first
    assert app.handle_key(pygame.K_s)
    assert app.seed == seed + 1 and app.mission.seed == seed + 1


def test_play_step_speed_slip_and_scenario_keys(app: ExtSimApp) -> None:
    assert app.paused
    assert app.handle_key(pygame.K_SPACE) and not app.paused
    assert app.handle_key(pygame.K_SPACE) and app.paused
    t0 = app.mission.world.t
    assert app.handle_key(pygame.K_n) and app.mission.world.t == t0 + 1
    speed = app.speed
    assert app.handle_key(pygame.K_PLUS) and app.speed > speed
    p = app.slip_p
    assert app.handle_key(pygame.K_RIGHTBRACKET) and app.slip_p == pytest.approx(
        p + theme.SLIP_P_STEP)
    name = app.scenario_name
    assert app.handle_key(pygame.K_m) and app.scenario_name != name
    assert all(e.generator is not None for e in app.scenarios)  # M cycles generated maps
    for k in (pygame.K_t, pygame.K_p, pygame.K_c, pygame.K_h):
        assert app.handle_key(k)
        app.render()


def test_map_editor_is_disabled(app: ExtSimApp, tmp_path: Path) -> None:
    assert app.handle_key(pygame.K_e)
    assert not app.editor
    assert app.handle_key(pygame.K_s, pygame.KMOD_CTRL)
    assert log_texts(app).count(EDITOR_DISABLED_TEXT) == 2
    assert not list(MAPS_DIR.glob("edited_*.txt"))
    help_items = app._help_lines()
    assert "Map editor" not in help_items
    assert any(isinstance(i, tuple) and i[0] == "X" for i in help_items)


# --- (3) log lines and panel -------------------------------------------------------------------
def test_search_log_lines(app: ExtSimApp) -> None:
    run_to_end(app)
    texts = log_texts(app)
    assert any(t.startswith("Exploring frontier (") for t in texts)
    assert any(t.startswith("Survivor found at (") for t in texts)
    assert any(t.startswith("Rescued (") and t.endswith("— 1/1") for t in texts)
    assert texts[-1] == "Rescue successful"
    assert app.mission.status is MissionStatus.RESCUED


def test_multi_log_lines_and_panel(tmp_path: Path) -> None:
    a = ExtSimApp(ext_options(tmp_path, mode="multi"))
    try:
        info = a._panel_info()
        assert info.metrics["Rescued"] == f"0/{EXT_DEMO_MULTI_K}"
        assert info.metrics["Mode"] == "rescue"
        assert info.legend is None  # no probabilistic belief
        run_to_end(a)
        texts = log_texts(a)
        for i in range(1, EXT_DEMO_MULTI_K + 1):
            assert any(t.startswith("Rescued (") and t.endswith(f"— {i}/{EXT_DEMO_MULTI_K}")
                       for t in texts)
        assert texts[-1] == f"All {EXT_DEMO_MULTI_K} survivors rescued"
        assert a._panel_info().metrics["Rescued"] == f"{EXT_DEMO_MULTI_K}/{EXT_DEMO_MULTI_K}"
        a.render()
    finally:
        a.close()


def test_noisy_bump_lines_and_legend(tmp_path: Path) -> None:
    a = ExtSimApp(ext_options(tmp_path, mode="noisy"))
    try:
        assert a._panel_info().legend is not None
        run_to_end(a)
        bumps = [t for t in log_texts(a) if t.startswith("Bump into unseen debris at (")]
        assert len(bumps) == a.mission.metrics().bumps > 0
        assert not any(t.startswith("Bumped into debris") for t in log_texts(a))
        a.render()
    finally:
        a.close()


def test_false_positive_degrades_gracefully(tmp_path: Path) -> None:
    a = ExtSimApp(ext_options(tmp_path, mode="noisy", false_positive=0.05))
    try:
        supported = false_positive_supported()
        assert supported == a.fp_supported
        assert (FALSE_POSITIVE_ARG in a.mission_kwargs()) == supported
        assert (FP_UNSUPPORTED_TEXT in log_texts(a)) == (not supported)
        a.fp_supported = True  # a build with two-sided noise passes the rate on
        assert a.mission_kwargs()[FALSE_POSITIVE_ARG] == 0.05
    finally:
        a.close()


def test_fallbacks_metric_counts_relaxed_replans(tmp_path: Path) -> None:
    """"Fallbacks" is the relaxed-passability fallback of two-sided noise (ExtMetrics /
    raw.csv ``fallback_replans``), not the core planner-budget fallback."""
    if not false_positive_supported():
        pytest.skip("this build's sensor has no false-positive rate")
    a = ExtSimApp(ext_options(tmp_path, mode="noisy", false_positive=FP_RATE, seed=FP_SEED))
    try:
        assert PHANTOM_HINT_TEXT in log_texts(a)
        run_to_end(a)
        m = a.mission.metrics()
        assert m.fallback_replans == a.mission.agent.fallback_replans > 0
        assert a._panel_info().metrics[FALLBACKS_LABEL] == str(m.fallback_replans)
        a.render()
    finally:
        a.close()


def test_aftershocks_metric_shown_with_aftershocks(tmp_path: Path) -> None:
    a = ExtSimApp(ext_options(tmp_path, mode="multi", aftershock_q=AFTERSHOCK_Q))
    try:
        run_to_end(a)
        m = a.mission.metrics()
        assert m.aftershock_events > 0
        assert a._panel_info().metrics[AFTERSHOCKS_LABEL] == str(m.aftershock_events)
        a.render()
    finally:
        a.close()
    b = ExtSimApp(ext_options(tmp_path, mode="multi"))
    try:
        assert AFTERSHOCKS_LABEL not in b._panel_info().metrics  # no aftershocks, no row
        assert PHANTOM_HINT_TEXT not in log_texts(b)
    finally:
        b.close()


# --- (4) CLI -----------------------------------------------------------------------------------
def test_cli_ext_headless_screenshot(tmp_path: Path, capsys: Any) -> None:
    shot = tmp_path / "cli.png"
    code = sim.main(["--ext", "multi", "--k", "2", "--agent", "search_astar", "--headless",
                     "--frames", "5", "--screenshot", str(shot)])
    assert code == 0
    assert png_size(shot) == theme.WINDOW_SIZE
    assert str(shot) in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--k", "3"],                       # ext-only flag without --ext
    ["--noise", "0.1"],
    ["--ext", "search", "--save-run"],  # runs files hold core missions
    ["--ext", "search", "--agent", "andor"],
    ["--agent", "search_andor"],        # extension agent without --ext
    ["--ext", "noisy", "--noise", "2"],
])
def test_cli_rejects_bad_combinations(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        sim.main([*argv, "--headless", "--frames", "1"])
    assert info.value.code == 2


def test_core_sim_does_not_import_the_extensions() -> None:
    """``rescue.sim`` without ``--ext`` loads no ``rescue.ext`` module, and its agent choices
    are the extension's agents."""
    assert sim.EXT_AGENT_CHOICES == EXT_AGENT_NAMES
    code = ("import sys, rescue.sim; "
            "print(sorted(m for m in sys.modules if m.startswith('rescue.ext')))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True)
    assert done.stdout.strip() == "[]"


def test_web_stage_ships_no_module_that_imports_the_extensions(tmp_path: Path) -> None:
    """The browser bundle (``scripts/build_web.py``) has no ``ext/`` and no module importing
    ``rescue.ext`` (the extension simulator is left out)."""
    build_web = load_script("build_web")
    fonts = tmp_path / "fonts_src"
    fonts.mkdir()
    for name in (*build_web.FONT_FILES, build_web.FONT_LICENCE_FILE):
        (fonts / name).write_bytes(b"")
    package = build_web.stage(tmp_path / "web_app", fonts_dir=fonts) / build_web.APP_PACKAGE
    assert (package / "sim.py").is_file() and (package / "viz" / "sim_app.py").is_file()
    assert not (package / "viz" / "ext_sim_app.py").exists()
    offenders = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}: {n}" for n in names
                          if n == "rescue.ext" or n.startswith("rescue.ext.")]
    assert offenders == []


# --- (5) GridScene marks: pixel probes ---------------------------------------------------------
def probe_scene(**fields: Any) -> GridScene:
    """An all-open, fully known PROBE_SIZE² board, robot top-left, survivor bottom-right."""
    n = PROBE_SIZE
    cells = np.full((n, n), int(Cell.OPEN), dtype=np.int8)
    cells[n - 1, n - 1] = Cell.SURVIVOR
    settings: dict[str, Any] = {
        "rows": n, "cols": n, "belief": cells.copy(), "true_cells": cells,
        "high_slip": np.zeros((n, n), dtype=bool), "robot": (0, 0), "start": (0, 0),
        "survivor": (n - 1, n - 1),
    }
    settings.update(fields)
    return GridScene(**settings)


def draw(fonts: Any, scene: GridScene, palette: theme.Palette = theme.DEFAULT_PALETTE,
         ) -> tuple[GridView, Any]:
    surface = pygame.Surface(theme.WINDOW_SIZE)
    view = GridView(theme.grid_area(), palette, fonts)
    view.draw(surface, scene)
    return view, surface


def diff_mask(a: Any, b: Any) -> np.ndarray:
    return np.any(pygame.surfarray.array3d(a) != pygame.surfarray.array3d(b), axis=2)


def test_rescued_mark_drawn(pg: Any) -> None:
    p = (2, 2)
    view, plain = draw(pg, probe_scene())
    _, marked = draw(pg, probe_scene(rescued=frozenset({p})))
    colours = theme.ext_colours(theme.DEFAULT_PALETTE)
    cx, cy = view.cell_centre(p)
    ring = round(view.cell * theme.RESCUED_RING_RADIUS_FRAC)
    assert rgb(marked, (cx, cy - ring)) == colours.rescued
    assert rgb(plain, (cx, cy - ring)) != colours.rescued
    rect = view.cell_rect(p)
    changed = diff_mask(plain, marked)
    assert changed.any()
    assert not changed[: rect.left - 1, :].any()  # only that cell changes
    assert not changed[rect.right + 1:, :].any()


def test_target_mark_drawn(pg: Any) -> None:
    p = (1, 3)
    view, plain = draw(pg, probe_scene())
    _, marked = draw(pg, probe_scene(target=p))
    colours = theme.ext_colours(theme.DEFAULT_PALETTE)
    rect = view.cell_rect(p)
    inset = max(1, round(view.cell * theme.TARGET_INSET_FRAC))
    corner = (rect.left + inset, rect.top + inset)  # a dash starts at each corner
    assert rgb(marked, corner) == colours.target
    assert rgb(plain, corner) != colours.target
    ys, xs = np.nonzero(diff_mask(plain, marked).T)
    assert rect.collidepoint(int(xs.min()), int(ys.min()))
    assert rect.collidepoint(int(xs.max()), int(ys.max()))


def test_target_mark_is_compact_on_small_cells(pg: Any) -> None:
    """On a large map (cells < TARGET_COMPACT_MAX_CELL) the target is a thin square on the cell
    edge without a flag: a survivor's cross under it stays untouched."""
    n = LARGE_BOARD
    p = (n // 2, n // 2)
    cells = np.full((n, n), int(Cell.OPEN), dtype=np.int8)
    cells[p] = Cell.SURVIVOR
    settings: dict[str, Any] = {
        "rows": n, "cols": n, "belief": cells.copy(), "true_cells": cells,
        "high_slip": np.zeros((n, n), dtype=bool), "robot": (0, 0), "start": (0, 0),
        "survivor": p,
    }
    view, plain = draw(pg, GridScene(**settings))
    _, marked = draw(pg, GridScene(**settings, target=p))
    assert view.cell < theme.TARGET_COMPACT_MAX_CELL
    band = (theme.stroke(view.cell, theme.TARGET_STROKE_FRAC)
            + theme.TARGET_COMPACT_HALO_EXTRA)
    rect = view.cell_rect(p)
    changed = diff_mask(plain, marked)
    inner = rect.inflate(-2 * band, -2 * band)
    assert changed[rect.left:rect.right, rect.top:rect.bottom].any()
    assert not changed[inner.left:inner.right, inner.top:inner.bottom].any()  # cross intact


def test_possible_phantom_debris_marked(pg: Any) -> None:
    """Believed debris with P(debris) < 1 (false positives) is lightened with a '?' badge;
    certain debris (P = 1) and scenes without ``debris_prob`` look as before."""
    n = PROBE_SIZE
    phantom, certain = (1, 1), (2, 3)
    belief = probe_scene().belief.copy()
    belief[phantom] = belief[certain] = Cell.DEBRIS
    prob = np.full((n, n), 1e-5)
    prob[phantom] = PHANTOM_P
    prob[certain] = 1.0
    view, plain = draw(pg, probe_scene(belief=belief))
    _, marked = draw(pg, probe_scene(belief=belief, debris_prob=prob))
    changed = diff_mask(plain, marked)
    for p, expected in ((phantom, True), (certain, False)):
        rect = view.cell_rect(p)
        assert changed[rect.left:rect.right, rect.top:rect.bottom].any() == expected, p
    prob[phantom] = 1.0
    _, none = draw(pg, probe_scene(belief=belief, debris_prob=prob))
    assert not diff_mask(plain, none).any()


def test_debris_prob_shading_drawn_on_seen_open_cells_only(pg: Any) -> None:
    n = PROBE_SIZE
    prob = np.full((n, n), 1e-5)
    shaded, faint, unknown = (1, 1), (2, 2), (3, 3)
    prob[shaded] = 0.3
    prob[unknown] = 0.3
    belief = probe_scene().belief.copy()
    belief[unknown] = Cell.UNKNOWN
    view, plain = draw(pg, probe_scene(belief=belief))
    _, marked = draw(pg, probe_scene(belief=belief, debris_prob=prob))
    changed = diff_mask(plain, marked)
    for p, expected in ((shaded, True), (faint, False), (unknown, False)):
        rect = view.cell_rect(p)
        assert changed[rect.left:rect.right, rect.top:rect.bottom].any() == expected, p
    levels = prob_shade_levels(np.array([theme.PROB_SHADE_MIN, 0.002, 0.3, 1.0]))
    assert levels[0] == 0 and 1 <= levels[1] < levels[2] <= levels[3] == theme.PROB_SHADE_LEVELS


def test_robot_on_extra_survivor_gets_the_rescue_ring(pg: Any) -> None:
    extra = (0, 0)  # the robot's cell
    view, plain = draw(pg, probe_scene())
    _, ringed = draw(pg, probe_scene(survivors=(extra,)))
    cx, cy = view.cell_centre(extra)
    ring = round(view.cell * theme.RESCUE_RING_RADIUS_FRAC)
    assert rgb(ringed, (cx, cy - ring)) == theme.DEFAULT_PALETTE.survivor
    assert rgb(plain, (cx, cy - ring)) != theme.DEFAULT_PALETTE.survivor


def test_unseen_survivor_stays_under_the_fog(app: ExtSimApp) -> None:
    survivor = app.mission.grid.survivor
    scene = app._scene()
    assert scene.belief is not None and scene.belief[survivor] == Cell.UNKNOWN
    app.render()
    fogged = rgb(app.screen, app.grid_view.cell_centre(survivor))
    assert fogged != app.palette.survivor
    assert app.handle_key(pygame.K_t)
    app.render()
    assert rgb(app.screen, app.grid_view.cell_centre(survivor)) != fogged  # true-map ghost


# --- (6) rule zero: core rendering unchanged ---------------------------------------------------
def core_scenes() -> list[GridScene]:
    """Core scenes of a deck_demo mission: plain, true map + policy + flash, and the editor."""
    grid = TrueGrid.load(MAPS_DIR / "deck_demo.txt")
    cfg = WorldConfig()
    mission = Mission(grid, make_agent(AGENT_NAMES[0], make_transition(grid, cfg)), cfg,
                      seed=DEMO_SEED)
    mission.start()
    for _ in range(CORE_STEPS):
        mission.step()
    trace = trace_from_events(mission.events)
    rich = GridScene.from_mission(mission, show_true=True, show_policy=True, trace=trace,
                                  flash_cells=(grid.survivor,), flash_alpha=0.6)
    editor = GridScene(rows=grid.rows, cols=grid.cols, belief=None, true_cells=grid.cells,
                       high_slip=grid.high_slip, robot=None, start=grid.start,
                       survivor=grid.survivor, editor=True)
    return [GridScene.from_mission(mission), rich, editor]


def test_core_grid_rendering_identical_to_base(pg: Any, tmp_path: Path) -> None:
    head = load_base_module("base_grid_view", "src/rescue/viz/grid_view.py", tmp_path)
    for palette in theme.PALETTES:
        for scene in core_scenes():
            now = pygame.Surface(theme.WINDOW_SIZE)
            before = pygame.Surface(theme.WINDOW_SIZE)
            GridView(theme.grid_area(), palette, pg).draw(now, scene)
            head.GridView(theme.grid_area(), palette, pg).draw(before, scene)
            assert pygame.image.tobytes(now, "RGB") == pygame.image.tobytes(before, "RGB")


def test_core_panel_rendering_identical_to_base(pg: Any, tmp_path: Path) -> None:
    head = load_base_module("base_panel", "src/rescue/viz/panel.py", tmp_path)
    log = EventLog()
    log.add("Slip at (1,2)", "slip")
    log.add("Debris sensed at (3,2) → replanning", "replan")
    metrics = {"Steps / budget": "4/320", "Replans": "1", "Bumps": "0"}
    for palette in theme.PALETTES:
        surfaces = []
        for panel_cls, info_cls in ((Panel, PanelInfo), (head.Panel, head.PanelInfo)):
            surface = pygame.Surface(theme.WINDOW_SIZE)
            info = info_cls(title="Rescue simulator", agent="andor",
                            status=MissionStatus.RUNNING, seed=DEMO_SEED, slip_p=0.2,
                            scenario="deck_demo", metrics=metrics)
            tree = TreeView(pygame.Rect(0, 0, 1, 1), palette, pg)
            panel_cls(theme.panel_area(), palette, pg).draw(surface, info, log, tree)
            surfaces.append(pygame.image.tobytes(surface, "RGB"))
        assert surfaces[0] == surfaces[1]


@pytest.mark.parametrize("name", CORE_PNG_FIGURES)
def test_make_figures_reproduces_committed_core_figures(tmp_path: Path, name: str) -> None:
    """Rule zero (plan §3.15): the core simulator figures are byte-identical."""
    figures = load_script("make_figures")
    key = next(k for k, (file, _make) in figures.FIGURES.items() if file == name)
    _file, make = figures.FIGURES[key]
    out = make(tmp_path / name)
    assert out.read_bytes() == (FIGURES_DIR / name).read_bytes()


@pytest.mark.slow
def test_make_ext_figures_reproduces_committed_figures(tmp_path: Path) -> None:
    """``scripts/make_ext_figures.py`` is deterministic and matches docs/figures/ext_*.png."""
    script = load_script("make_ext_figures")
    assert script.main(["--out", str(tmp_path)]) == 0
    for name in EXT_FIGURES:
        assert png_size(tmp_path / name) == theme.WINDOW_SIZE
        assert (tmp_path / name).read_bytes() == (FIGURES_DIR / name).read_bytes(), name
