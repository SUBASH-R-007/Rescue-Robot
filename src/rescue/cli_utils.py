"""Small helpers shared by the command-line entry points."""

from __future__ import annotations

import contextlib
import sys


def ensure_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so arrows in traces print on Windows consoles and pipes."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")
