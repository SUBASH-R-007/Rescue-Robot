# /// script
# dependencies = [
#     "numpy",
#     "pygame-ce",
# ]
# ///
"""Browser entry point of the interactive simulator (pygbag build, docs/web.md).

pygbag packs the folder holding this ``main.py`` (staged by ``scripts/build_web.py`` as
``build/web_app/``: this file, the ``rescue`` package, ``maps/`` and ``fonts/``) and runs it with
CPython compiled to WebAssembly. The simulator starts on ``maps/deck_demo.txt`` with the demo
seed, paused, exactly like ``python -m rescue.sim --map maps/deck_demo.txt``; the loop is
:meth:`rescue.viz.sim_app.BaseApp.run_async`, which yields to the browser once per frame.

On a desktop, ``python web/main.py`` runs the same entry point in a normal window.
The PEP 723 block above and the top-level ``numpy`` import tell pygbag which wheels to fetch.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy  # noqa: F401 - pygbag fetches the WebAssembly wheels of the imports it sees here
import pygame  # noqa: F401

#: Folder of this file: the staged app folder in the browser build, ``web/`` in a checkout.
APP_DIR: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
#: The map the browser demo opens (relative to the app folder or the repository root).
DEMO_MAP: str = "maps/deck_demo.txt"


def _setup_paths() -> None:
    """Make the bundled ``rescue`` package importable and point the UI at the bundled fonts."""
    if (APP_DIR / "rescue").is_dir() and str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from rescue.viz import theme

    fonts = APP_DIR / theme.BUNDLED_FONTS_DIRNAME
    if fonts.is_dir():
        os.environ.setdefault(theme.FONTS_DIR_ENV, str(fonts))


async def main() -> int:
    """Build the deck_demo simulator (demo seed, paused) and run its asyncio frame loop."""
    _setup_paths()
    from rescue.config import DEMO_SEED
    from rescue.viz.sim_app import SimApp, SimOptions

    bundled_map = APP_DIR / DEMO_MAP
    options = SimOptions(
        map_path=str(bundled_map) if bundled_map.is_file() else DEMO_MAP,
        maps_dir=str(bundled_map.parent) if bundled_map.is_file() else None,
        seed=DEMO_SEED,
        autoplay=False,
    )
    return await SimApp(options).run_async()


asyncio.run(main())
