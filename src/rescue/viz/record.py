"""Record the UI to an animated GIF (imageio) or an MP4 (imageio-ffmpeg) - brief §6.5.

Usage::

    rec = Recorder()
    rec.start("docs/figures/deck_demo.gif")       # parent directories are created
    for each frame worth keeping: rec.add_frame(screen)
    path = rec.stop()                             # None if no frame was captured

Frames are copied with ``pygame.surfarray.array3d`` (after an optional ``smoothscale`` by
``scale``), so the recorder works with the dummy SDL video driver. The writer opens lazily on
the first frame; ``.mp4`` falls back to ``.gif`` (with a logged warning) when imageio-ffmpeg is
not installed. :meth:`Recorder.due` helps apps running at 60 FPS capture at the record rate.
imageio is imported only when the first frame is written, so this module also imports in the
pygbag browser build, where :func:`recording_available` is False (docs/web.md).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from rescue.viz import theme

log = logging.getLogger(__name__)

#: Suffixes written through ffmpeg.
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".m4v", ".mov"})
#: Suffix used when video is unavailable.
GIF_SUFFIX = ".gif"
#: Video codec and the frame-size multiple it needs (yuv420p wants even sizes).
VIDEO_CODEC = "libx264"
VIDEO_BLOCK = 2
#: GIF loop count (0 = forever).
GIF_LOOP = 0
MS_PER_S = 1000.0
#: ``sys.platform`` of the pygbag (WebAssembly / Emscripten) browser build.
BROWSER_PLATFORM = "emscripten"


def in_browser() -> bool:
    """True when running in the browser build (pygbag, ``sys.platform == "emscripten"``)."""
    return sys.platform == BROWSER_PLATFORM


def recording_available() -> bool:
    """False in the browser build: imageio is not shipped there and files cannot be saved."""
    return not in_browser()


def ffmpeg_available() -> bool:
    """True if imageio-ffmpeg (and so MP4 output) is importable."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    return True


class Recorder:
    """Collects frames from a pygame surface and writes a GIF or MP4."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.fps: float = float(theme.RECORD_FPS)
        self.scale: float = theme.RECORD_SCALE
        #: Frames captured by :meth:`add_frame`. A GIF may store fewer: the Pillow writer merges
        #: identical consecutive frames (e.g. while paused) into one, adding their durations,
        #: so playback timing is unchanged.
        self.frame_count: int = 0
        self._writer: Any = None
        self._video = False
        self._size: tuple[int, int] | None = None
        self._last_ms: float | None = None

    @property
    def active(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self.path is not None

    def start(self, path: str | Path, fps: float = theme.RECORD_FPS,
              scale: float = theme.RECORD_SCALE) -> Path:
        """Begin a recording to ``path`` (``.gif`` or ``.mp4``); returns the actual path.

        A recording already in progress is stopped first. Raises ``ValueError`` for a
        non-positive ``fps`` or ``scale``.
        """
        if fps <= 0 or scale <= 0:
            raise ValueError(f"fps and scale must be positive, got fps={fps}, scale={scale}")
        if self.active:
            self.stop()
        target = Path(path)
        suffix = target.suffix.lower()
        self._video = suffix in VIDEO_SUFFIXES
        if self._video and not ffmpeg_available():
            log.warning("imageio-ffmpeg is not installed; recording %s as a GIF instead", target)
            target = target.with_suffix(GIF_SUFFIX)
            self._video = False
        elif not self._video and suffix != GIF_SUFFIX:
            target = target.with_suffix(GIF_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.fps, self.scale = target, float(fps), float(scale)
        self.frame_count = 0
        self._writer = None
        self._size = None
        self._last_ms = None
        log.info("recording to %s at %g fps (scale %g)", target, fps, scale)
        return target

    def due(self, now_ms: float) -> bool:
        """True if a frame should be captured at time ``now_ms`` to keep :attr:`fps`.

        Marks the frame as taken when it returns True.
        """
        if not self.active:
            return False
        interval = MS_PER_S / self.fps
        if self._last_ms is None or now_ms - self._last_ms >= interval:
            self._last_ms = now_ms if self._last_ms is None else self._last_ms + interval * (
                (now_ms - self._last_ms) // interval
            )
            return True
        return False

    def _frame(self, surface: pygame.Surface) -> np.ndarray:
        """``surface`` as an ``H x W x 3`` uint8 array, scaled and size-normalised."""
        src = surface
        if self.scale != 1.0:
            w, h = surface.get_size()
            size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
            if src.get_bitsize() not in (24, 32):
                src = src.convert(32, 0)
            src = pygame.transform.smoothscale(src, size)
        frame = np.ascontiguousarray(pygame.surfarray.array3d(src).transpose(1, 0, 2))
        if self._size is None:
            h, w = frame.shape[:2]
            if self._video:
                h, w = h - h % VIDEO_BLOCK, w - w % VIDEO_BLOCK
            self._size = (max(1, h), max(1, w))
        h, w = self._size
        if frame.shape[:2] != (h, w):  # a resized window: crop / pad to the first frame
            out = np.zeros((h, w, 3), dtype=np.uint8)
            fh, fw = min(h, frame.shape[0]), min(w, frame.shape[1])
            out[:fh, :fw] = frame[:fh, :fw]
            frame = out
        return frame

    def _open(self) -> Any:
        import imageio.v2 as imageio

        assert self.path is not None
        if self._video:
            return imageio.get_writer(
                self.path, format="FFMPEG", mode="I", fps=self.fps, codec=VIDEO_CODEC,
                macro_block_size=VIDEO_BLOCK,
            )
        return imageio.get_writer(
            self.path, format="GIF", mode="I", duration=MS_PER_S / self.fps, loop=GIF_LOOP,
        )

    def add_frame(self, surface: pygame.Surface) -> None:
        """Capture ``surface`` as the next frame (ignored when not recording)."""
        if not self.active:
            return
        frame = self._frame(surface)
        if self._writer is None:
            self._writer = self._open()
        self._writer.append_data(frame)
        self.frame_count += 1

    def stop(self) -> Path | None:
        """Finish the file; returns its path, or None if no frame was captured."""
        path, writer = self.path, self._writer
        self.path, self._writer, self._last_ms = None, None, None
        if path is None:
            return None
        if writer is None:
            log.info("recording to %s stopped without frames; nothing written", path)
            return None
        writer.close()
        log.info("captured %d frames to %s", self.frame_count, path)
        return path
