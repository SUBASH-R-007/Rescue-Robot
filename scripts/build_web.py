"""Build the browser version of the interactive simulator with pygbag (brief §6.7, docs/web.md).

Usage (from the repository root, with the dev venv and ``pip install -e ".[web]"``)::

    python scripts/build_web.py                    # stage build/web_app/ and run pygbag
    python scripts/build_web.py --stage-only       # stage only (no pygbag, no network)
    python scripts/build_web.py --out some/dir     # stage somewhere else

Two steps:

1. :func:`stage` copies what the browser app needs into the app folder (default
   ``build/web_app/``): ``web/main.py``, the ``rescue`` package without the packages and
   modules the simulator never imports (:data:`EXCLUDED_PACKAGE_DIRS`,
   :data:`EXCLUDED_MODULES`) and without ``__pycache__``, the maps (``maps/*.txt``) and two
   DejaVu fonts from matplotlib's ``mpl-data/fonts/ttf`` with their licence (``fonts/``;
   ``theme.load_fonts`` prefers them, so plan labels keep their arrows).
2. :func:`run_pygbag` runs ``python -m pygbag --build <app folder>``, which packs the folder into
   ``<app folder>/build/web/`` (``index.html``, ``<name>.apk``, ``favicon.png``). pygbag
   downloads its page template from its CDN on the first build, so that step needs network
   access; the Python runtime itself is fetched by the browser when the page loads.

The script prints the output directory and the bundle size. Nothing is deployed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

#: Repository root (scripts/build_web.py -> one level up).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Default app folder that pygbag packs (git-ignored, like everything under build/).
DEFAULT_APP_DIR = REPO_ROOT / "build" / "web_app"
#: Entry point, package and maps copied into the app folder.
WEB_MAIN = REPO_ROOT / "web" / "main.py"
PACKAGE_DIR = REPO_ROOT / "src" / "rescue"
MAPS_DIR = REPO_ROOT / "maps"
MAP_GLOB = "*.txt"
#: Sub-packages of ``rescue`` the simulator never imports (experiment harness, extensions).
EXCLUDED_PACKAGE_DIRS: tuple[str, ...] = ("experiments", "ext")
#: Modules of ``rescue`` (paths relative to the package) left out because they import an
#: excluded package: the extension simulator (``sim --ext``, plan §3.15.3) needs ``rescue.ext``.
EXCLUDED_MODULES: tuple[str, ...] = ("viz/ext_sim_app.py",)
#: Files and directories never copied.
IGNORED_PATTERNS: tuple[str, ...] = ("__pycache__", "*.pyc", "*.pyo")
#: Fonts copied from matplotlib's mpl-data/fonts/ttf (DejaVu: Bitstream Vera licence).
FONT_FILES: tuple[str, ...] = ("DejaVuSans.ttf", "DejaVuSansMono.ttf")
FONT_LICENCE_FILE = "LICENSE_DEJAVU"
#: Names inside the app folder.
APP_MAIN = "main.py"
APP_PACKAGE = "rescue"
APP_MAPS = "maps"
APP_FONTS = "fonts"
#: pygbag's output directory inside the app folder.
PYGBAG_OUTPUT = Path("build") / "web"
#: Page title shown by the pygbag loader.
WEB_TITLE = "Rescue Robot simulator"
#: Bytes per MiB (size report).
BYTES_PER_MIB = 1024 * 1024


def matplotlib_fonts_dir() -> Path:
    """matplotlib's bundled TrueType directory (``mpl-data/fonts/ttf``)."""
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _ignore_for(root: Path) -> Callable[[str, list[str]], set[str]]:
    """copytree ignore callback: :data:`IGNORED_PATTERNS` everywhere, plus
    :data:`EXCLUDED_PACKAGE_DIRS` directly under ``root`` and the :data:`EXCLUDED_MODULES`."""
    patterns = shutil.ignore_patterns(*IGNORED_PATTERNS)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(patterns(directory, names))
        here = Path(directory).resolve()
        if here == root.resolve():
            ignored |= {n for n in names if n in EXCLUDED_PACKAGE_DIRS}
        ignored |= {n for n in names
                    if (here / n).relative_to(root.resolve()).as_posix() in EXCLUDED_MODULES}
        return ignored

    return ignore


def stage(app_dir: Path = DEFAULT_APP_DIR, fonts_dir: Path | None = None) -> Path:
    """Copy the browser app into ``app_dir`` (see the module docstring); returns ``app_dir``.

    Earlier staged copies are replaced; pygbag's own ``build/`` folder (download cache and
    output) is kept. Raises ``FileNotFoundError`` if a font or the licence is missing.
    """
    app_dir = Path(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    for name in (APP_PACKAGE, APP_MAPS, APP_FONTS):
        if (app_dir / name).exists():
            shutil.rmtree(app_dir / name)
    shutil.copy2(WEB_MAIN, app_dir / APP_MAIN)
    shutil.copytree(PACKAGE_DIR, app_dir / APP_PACKAGE, ignore=_ignore_for(PACKAGE_DIR))
    maps = app_dir / APP_MAPS
    maps.mkdir()
    for path in sorted(MAPS_DIR.glob(MAP_GLOB)):
        shutil.copy2(path, maps / path.name)
    source = Path(fonts_dir) if fonts_dir is not None else matplotlib_fonts_dir()
    fonts = app_dir / APP_FONTS
    fonts.mkdir()
    for name in (*FONT_FILES, FONT_LICENCE_FILE):
        if not (source / name).is_file():
            raise FileNotFoundError(f"{name} not found in {source}")
        shutil.copy2(source / name, fonts / name)
    return app_dir


def pygbag_command(app_dir: Path) -> list[str]:
    """The pygbag build command for ``app_dir`` (canvas = the simulator window size)."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from rescue.viz import theme

    width, height = theme.WINDOW_SIZE
    return [
        sys.executable, "-m", "pygbag", "--build",
        "--width", str(width), "--height", str(height),
        "--title", WEB_TITLE,
        str(app_dir),
    ]


def run_pygbag(app_dir: Path) -> Path:
    """Run pygbag on ``app_dir`` (in that folder, so no stray ``pygbag.ini`` applies); returns
    the output directory. Raises ``RuntimeError`` if pygbag fails or writes no index.html."""
    result = subprocess.run(pygbag_command(app_dir), cwd=app_dir, check=False)
    output = app_dir / PYGBAG_OUTPUT
    if result.returncode != 0 or not (output / "index.html").is_file():
        raise RuntimeError(f"pygbag failed (exit code {result.returncode}); see its output above")
    return output


def directory_size(path: Path) -> int:
    """Total size in bytes of the files directly in ``path``."""
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_APP_DIR,
                        help=f"app folder to stage and build (default {DEFAULT_APP_DIR})")
    parser.add_argument("--stage-only", action="store_true",
                        help="stage the app folder but do not run pygbag")
    args = parser.parse_args(argv)
    app_dir = stage(args.out.resolve())
    print(f"staged {app_dir}", flush=True)
    if args.stage_only:
        return 0
    try:
        output = run_pygbag(app_dir)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    size = directory_size(output)
    print(f"web build: {output} ({size / BYTES_PER_MIB:.2f} MiB)")
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(f"  {path.name}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
