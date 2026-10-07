"""Browser build (brief §6.7, docs/web.md): the asyncio frame loop, the browser fallbacks, the
bundled fonts and the staging step of ``scripts/build_web.py`` (pygbag itself is never run).

Everything runs headless with the SDL dummy drivers, selected before pygame is imported.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

pygame = pytest.importorskip("pygame")

from rescue.viz import sim_app, theme  # noqa: E402
from rescue.viz.record import BROWSER_PLATFORM, in_browser, recording_available  # noqa: E402
from rescue.viz.sim_app import RECORDING_UNAVAILABLE_TEXT, SimApp, SimOptions  # noqa: E402

pytestmark = pytest.mark.viz

#: Repository root (tests/test_web.py -> two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_MAP = REPO_ROOT / "maps" / "deck_demo.txt"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_web.py"
WEB_MAIN = REPO_ROOT / "web" / "main.py"
#: Headless frames driven through the loops.
LOOP_FRAMES = 30


# --- helpers -----------------------------------------------------------------------------------
def make_app(tmp_path: Path, **overrides: object) -> SimApp:
    """A deck_demo simulator writing only under ``tmp_path``."""
    options = {
        "map_path": str(DEMO_MAP),
        "headless": True,
        "frames": LOOP_FRAMES,
        "autoplay": True,
        "recordings_dir": str(tmp_path / "recordings"),
        "runs_dir": str(tmp_path / "runs"),
        "maps_dir": str(DEMO_MAP.parent),
    }
    options.update(overrides)
    return SimApp(SimOptions(**options))  # type: ignore[arg-type]


def load_build_script() -> ModuleType:
    """``scripts/build_web.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("build_web", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matplotlib_fonts() -> Path:
    """matplotlib's TrueType directory (the source of the bundled fonts)."""
    matplotlib = pytest.importorskip("matplotlib")
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


# --- asyncio frame loop -------------------------------------------------------------------------
def test_run_async_drives_frames_headless_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_async`` runs exactly ``frames`` headless frames, yields once per frame, saves the
    screenshot and closes the app."""
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(sim_app.asyncio, "sleep", counting_sleep)
    shot = tmp_path / "shot.png"
    app = make_app(tmp_path, screenshot=str(shot), speed_index=len(theme.SPEEDS) - 1)
    assert asyncio.run(app.run_async()) == 0
    assert sleeps == [0] * LOOP_FRAMES
    assert app.clock_ms == pytest.approx(LOOP_FRAMES * theme.HEADLESS_FRAME_MS)
    assert len(app.mission.events) > 1  # autoplay took steps
    assert shot.is_file()
    assert app._closed


def test_run_and_run_async_take_the_same_frames(tmp_path: Path) -> None:
    """Desktop ``run`` and browser ``run_async`` share the per-frame work: same clock, same
    mission after the same number of headless frames."""
    fastest = len(theme.SPEEDS) - 1
    blocking = make_app(tmp_path, speed_index=fastest)
    assert blocking.run() == 0
    looped = make_app(tmp_path, speed_index=fastest)
    assert asyncio.run(looped.run_async()) == 0
    assert looped.clock_ms == pytest.approx(blocking.clock_ms)
    assert len(looped.mission.events) > 1
    assert [e.new_pos for e in looped.mission.events] == [
        e.new_pos for e in blocking.mission.events
    ]


def test_run_async_interactive_stops_on_escape(tmp_path: Path) -> None:
    """Interactive (non-headless) ``run_async`` dispatches queued events and ends on Esc."""
    app = make_app(tmp_path, headless=False, autoplay=False)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert asyncio.run(app.run_async()) == 0
    assert not app.running
    assert app._closed


def test_run_frame_reports_quit(tmp_path: Path) -> None:
    """``run_frame`` is False once a quit event has been handled, True otherwise."""
    app = make_app(tmp_path, headless=False, autoplay=False)
    try:
        clock = pygame.time.Clock()
        assert app.run_frame(clock)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert not app.run_frame(clock)
    finally:
        app.close()


# --- browser fallbacks --------------------------------------------------------------------------
def test_browser_detection_follows_sys_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not in_browser() and recording_available()
    monkeypatch.setattr(sys, "platform", BROWSER_PLATFORM)
    assert in_browser() and not recording_available()


def test_recording_unavailable_in_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``G`` logs a notice instead of starting a recorder where nothing can be recorded."""
    monkeypatch.setattr(sim_app, "recording_available", lambda: False)
    app = make_app(tmp_path)
    try:
        assert app.handle_key(pygame.K_g)
        assert not app.recorder.active
        assert any(RECORDING_UNAVAILABLE_TEXT in text for text, _kind in app.log.lines)
        app.frame(theme.HEADLESS_FRAME_MS)  # rendering still works
    finally:
        app.close()
    assert app.recordings == []
    assert not (tmp_path / "recordings").exists()


def test_escape_does_not_quit_in_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the browser Esc must not end the loop (the canvas would freeze until a reload)."""
    monkeypatch.setattr(sim_app, "in_browser", lambda: True)
    app = make_app(tmp_path, headless=False, autoplay=False)
    try:
        app.handle_key(pygame.K_ESCAPE)
        assert app.running
        app.handle_key(pygame.K_h)  # Esc still closes the help overlay
        assert app.show_help
        app.handle_key(pygame.K_ESCAPE)
        assert not app.show_help and app.running
    finally:
        app.close()


def test_help_hides_record_and_quit_in_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    def keys() -> set[str]:
        return {item[0] for item in sim_app.help_lines() if isinstance(item, tuple)}

    assert {"G", "Esc"} <= keys()
    monkeypatch.setattr(sim_app, "in_browser", lambda: True)
    assert not ({"G", "Esc"} & keys())
    assert {"T", "P", "H", "Space"} <= keys()
    assert "Esc" not in sim_app.BROWSER_HINT


def test_sim_imports_no_heavy_optional_modules() -> None:
    """The simulator's import closure needs none of matplotlib / scipy / imageio / PyYAML,
    which the pygbag runtime does not provide (checked in a fresh interpreter)."""
    import subprocess

    code = (
        "import sys; import rescue.viz.sim_app; "
        "bad = sorted({m.split('.')[0] for m in sys.modules} & "
        "{'matplotlib', 'scipy', 'imageio', 'yaml'}); print(','.join(bad))"
    )
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert out.stdout.strip() == ""


# --- bundled fonts ------------------------------------------------------------------------------
def test_no_bundled_fonts_in_a_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``$RESCUE_FONTS_DIR`` and without ``src/fonts`` the desktop font choice is
    unchanged (no bundled directory)."""
    monkeypatch.delenv(theme.FONTS_DIR_ENV, raising=False)
    assert theme.bundled_fonts_dir() is None
    assert theme.bundled_font_path(theme.BUNDLED_FONT_FILE, theme.REQUIRED_GLYPHS) is None


def test_load_fonts_prefers_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    for name in (theme.BUNDLED_FONT_FILE, theme.BUNDLED_MONO_FONT_FILE):
        shutil.copy2(matplotlib_fonts() / name, fonts_dir / name)
    monkeypatch.setenv(theme.FONTS_DIR_ENV, str(fonts_dir))
    pygame.font.init()
    assert theme.bundled_fonts_dir() == fonts_dir
    fonts = theme.load_fonts()
    assert fonts.path == str(fonts_dir / theme.BUNDLED_FONT_FILE)
    assert fonts.mono_path == str(fonts_dir / theme.BUNDLED_MONO_FONT_FILE)


def test_load_fonts_ignores_a_fonts_dir_without_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(theme.FONTS_DIR_ENV, str(tmp_path))
    assert theme.bundled_fonts_dir() is None
    fonts = theme.load_fonts()
    assert fonts.path is None or not fonts.path.startswith(str(tmp_path))


# --- build script -------------------------------------------------------------------------------
def test_stage_copies_main_package_maps_and_fonts(tmp_path: Path) -> None:
    build_web = load_build_script()
    app_dir = build_web.stage(tmp_path / "web_app")
    assert (app_dir / "main.py").read_bytes() == WEB_MAIN.read_bytes()
    package = app_dir / "rescue"
    for module in ("__init__.py", "mapgen.py", "runner.py", "viz/sim_app.py", "viz/theme.py"):
        assert (package / module).is_file(), module
    assert not (package / "experiments").exists()
    assert not (package / "ext").exists()
    assert not list(package.rglob("__pycache__"))
    staged_maps = sorted(p.name for p in (app_dir / "maps").glob("*.txt"))
    assert staged_maps == sorted(p.name for p in DEMO_MAP.parent.glob("*.txt"))
    assert "deck_demo.txt" in staged_maps
    for name in (*build_web.FONT_FILES, build_web.FONT_LICENCE_FILE):
        assert (app_dir / "fonts" / name).is_file(), name
    # the staged package finds the staged fonts next to itself
    assert theme.BUNDLED_FONTS_DIRNAME == "fonts"


def test_stage_replaces_an_earlier_copy(tmp_path: Path) -> None:
    build_web = load_build_script()
    app_dir = tmp_path / "web_app"
    build_web.stage(app_dir)
    stale = app_dir / "rescue" / "stale.py"
    stale.write_text("x = 1\n", encoding="utf-8")
    keep = app_dir / "build" / "web-cache" / "cached.tmpl"
    keep.parent.mkdir(parents=True)
    keep.write_text("cache", encoding="utf-8")
    build_web.stage(app_dir)
    assert not stale.exists()
    assert keep.is_file()  # pygbag's cache survives a re-stage


def test_stage_reports_missing_fonts(tmp_path: Path) -> None:
    build_web = load_build_script()
    with pytest.raises(FileNotFoundError):
        build_web.stage(tmp_path / "web_app", fonts_dir=tmp_path / "nofonts")


def test_pygbag_command_builds_the_app_folder(tmp_path: Path) -> None:
    build_web = load_build_script()
    cmd = build_web.pygbag_command(tmp_path)
    assert cmd[:4] == [sys.executable, "-m", "pygbag", "--build"]
    assert cmd[-1] == str(tmp_path)
    width, height = theme.WINDOW_SIZE
    assert cmd[cmd.index("--width") + 1] == str(width)
    assert cmd[cmd.index("--height") + 1] == str(height)


def test_main_stages_and_runs_pygbag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` stages, calls pygbag (stubbed here) and prints the output path and size."""
    build_web = load_build_script()
    calls: list[Path] = []

    def fake_pygbag(app_dir: Path) -> Path:
        calls.append(app_dir)
        out = app_dir / build_web.PYGBAG_OUTPUT
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text("<html></html>", encoding="utf-8")
        (out / "web_app.apk").write_bytes(b"0" * 10)
        return out

    monkeypatch.setattr(build_web, "run_pygbag", fake_pygbag)
    app_dir = tmp_path / "web_app"
    assert build_web.main(["--out", str(app_dir)]) == 0
    assert calls == [app_dir.resolve()]
    printed = capsys.readouterr().out
    assert str(app_dir.resolve() / build_web.PYGBAG_OUTPUT) in printed
    assert "index.html" in printed and "web_app.apk" in printed


def test_main_stage_only_skips_pygbag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_web = load_build_script()

    def fail(app_dir: Path) -> Path:
        raise AssertionError("pygbag must not run with --stage-only")

    monkeypatch.setattr(build_web, "run_pygbag", fail)
    assert build_web.main(["--out", str(tmp_path / "app"), "--stage-only"]) == 0
    assert (tmp_path / "app" / "main.py").is_file()


def test_main_reports_a_pygbag_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_web = load_build_script()

    def broken(app_dir: Path) -> Path:
        raise RuntimeError("pygbag failed (exit code 1)")

    monkeypatch.setattr(build_web, "run_pygbag", broken)
    assert build_web.main(["--out", str(tmp_path / "app")]) == 1
    assert "pygbag failed" in capsys.readouterr().err
