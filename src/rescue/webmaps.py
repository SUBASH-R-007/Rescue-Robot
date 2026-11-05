"""Maps for the web app (docs/plan.md §3.16): bundled map files, the pre-generated scenario
library and pure ASCII map-editing helpers.

The browser runs the exact Python core in Pyodide, where loading scipy (which
:mod:`rescue.mapgen` needs) costs tens of megabytes. So the static build, run with CPython,
writes :func:`build_map_library` — ``mapgen.generate(scenario, n, n, seed).to_ascii()`` for every
generated scenario, size in :data:`LIBRARY_SIZES` and seed in :data:`LIBRARY_SEEDS` — to
``maps_library.json``; :func:`generated_grid` parses a library entry when there is one and only
otherwise imports :mod:`rescue.mapgen` (the web worker then loads scipy on demand). A library
grid equals the generator's output exactly, name included (tested).

Nothing here imports scipy, pygame, matplotlib or PyYAML at module import.

Editing (:func:`edit_map`, :func:`validate_map`) works on the ASCII text of
:mod:`rescue.grid` (``.`` open, ``#`` debris, ``S`` start, ``P`` survivor, ``~`` high slip).
Positions in requests are 0-indexed ``[row, col]``; every message is 1-indexed
(:func:`rescue.grid.fmt_pos`).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rescue.config import EXT_DEMO_SIZE, MAPGEN_DEFAULT_SIZE, MAPGEN_MIN_SIZE
from rescue.grid import Pos, TrueGrid, fmt_pos
from rescue.planners.bfs import reachable

__all__ = [
    "EDIT_OPS",
    "GENERATED_SCENARIO_INFO",
    "LIBRARY_FILENAME",
    "LIBRARY_SEEDS",
    "LIBRARY_SIZES",
    "MapLibrary",
    "build_map_library",
    "check_map_size",
    "edit_map",
    "generated_grid",
    "get_map_library",
    "library_key",
    "list_map_files",
    "load_map_file",
    "maps_dir",
    "needs_generator",
    "problem_cells",
    "set_map_library",
    "set_maps_dir",
    "validate_map",
    "write_map_library",
]

# --- constants ------------------------------------------------------------------------------
#: Version of the ``maps_library.json`` format.
LIBRARY_FORMAT_VERSION = 1
#: File name of the pre-generated map library (next to ``index.html`` in the static site).
LIBRARY_FILENAME = "maps_library.json"
#: Square sizes in the library: the simulator / comparison default (15), the extension demo's
#: (12) and a small and a large one.
LIBRARY_SIZES: tuple[int, ...] = tuple(sorted({10, EXT_DEMO_SIZE[0], MAPGEN_DEFAULT_SIZE[0], 20}))
#: Number of seeds per scenario and size in the library (seeds 0 .. N-1).
LIBRARY_SEED_COUNT = 30
#: Seeds in the library.
LIBRARY_SEEDS: tuple[int, ...] = tuple(range(LIBRARY_SEED_COUNT))
#: Largest rows / cols the web app generates or edits (keeps a browser step fast).
WEB_MAX_MAP_SIZE = 40
#: Smallest rows / cols of an edited map (a start and a survivor need two cells).
EDITOR_MIN_SIZE = 2
#: Suffix of the bundled map files and the directory name they live in.
MAP_FILE_SUFFIX = ".txt"
MAPS_DIRNAME = "maps"
#: Map-file names accepted from the browser (no paths: letters, digits, ``_`` and ``-``).
MAP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
#: ASCII map characters (rescue.grid's format).
CH_OPEN, CH_DEBRIS, CH_START, CH_SURVIVOR, CH_HIGH_SLIP = ".", "#", "S", "P", "~"
MAP_CHARS: frozenset[str] = frozenset({CH_OPEN, CH_DEBRIS, CH_START, CH_SURVIVOR, CH_HIGH_SLIP})
#: Values ``paint`` may write.
PAINT_CHARS: frozenset[str] = frozenset({CH_OPEN, CH_DEBRIS, CH_HIGH_SLIP})
COMMENT_PREFIX = ";"
#: Repository root (src/rescue/webmaps.py → three levels up), for running from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Titles and descriptions of the generated scenarios. The descriptions are those of
#: ``rescue.mapgen.SCENARIOS`` (tested), spelled out so the catalog needs no scipy.
GENERATED_SCENARIO_INFO: dict[str, tuple[str, str]] = {
    "random_rubble": (
        "Random rubble",
        "Independent rubble on every cell (density 0.15-0.35).",
    ),
    "collapsed_building": (
        "Collapsed building",
        "Rooms and hallways behind 1-cell walls with narrow doorways, some collapsed; start in "
        "the street.",
    ),
    "flooded_zone": (
        "Flooded zone",
        "Open ground with pools of high-slip water / mud ('~') on every shortest route and "
        "sparse debris.",
    ),
    "maze_corridors": (
        "Maze corridors",
        "Winding 1-cell corridors with long dead ends and a few loops.",
    ),
}
#: Friendly titles and plain-English descriptions of the bundled map files (others are titled
#: from their file name and described by their leading comment, which is kept as ``source``).
MAP_FILE_INFO: dict[str, tuple[str, str]] = {
    "deck_demo": (
        "Quick demo (4×4)",
        "A tiny map with two pieces of debris between the robot and the survivor; the "
        "proposal's sample run (slide 9).",
    ),
    "deck_figure": (
        "Small figure map (4×4)",
        "The 4×4 map of the proposal's first figure: one piece of debris next to the start.",
    ),
    "building_01": (
        "Collapsed building (15×20)",
        "Rooms and hallways behind walls with narrow doorways; the robot starts in the street.",
    ),
    "flooded_01": (
        "Flooded zone (15×15)",
        "Open ground with pools of slippery water and mud on the short routes.",
    ),
    "maze_01": (
        "Maze (15×15)",
        "Winding one-cell corridors with dead ends and a few loops.",
    ),
    "rubble_01": (
        "Random rubble (15×15)",
        "Rubble scattered at random over the whole area.",
    ),
}
#: Longest map text accepted from the browser (characters): a full 40x40 map with comments.
WEB_MAX_MAP_TEXT = 8192
#: Edit operations of :func:`edit_map`.
EDIT_OPS: tuple[str, ...] = (
    "toggle_debris",
    "paint",
    "set_start",
    "set_survivor",
    "add_survivor",
    "remove_survivor",
    "toggle_high_slip",
    "resize",
    "clear",
)
#: Editor warnings (the first is the Pygame editor's wording).
NO_DEBRIS_ON_ENDPOINT = "No debris on the start / survivor cell {pos}"
NO_TERRAIN_ON_ENDPOINT = "No terrain on the start / survivor cell {pos}"
START_ON_SURVIVOR = "The start cannot be on a survivor {pos}"
SURVIVOR_ON_START = "A survivor cannot be on the start {pos}"
ALREADY_SURVIVOR = "There is already a survivor at {pos}"
NOT_A_SURVIVOR = "There is no survivor at {pos}"
LAST_SURVIVOR = "The map needs at least one survivor: {pos} stays"
UNREACHABLE_SURVIVOR = "The survivor at {pos} cannot be reached from the start {start}"
SEVERAL_SURVIVORS_CORE = (
    "The core mission has exactly one survivor; maps with several need the 'multi' mode"
)


# --- map files ------------------------------------------------------------------------------
_maps_dir_override: Path | None = None


def set_maps_dir(path: str | Path | None) -> None:
    """Use ``path`` as the directory of the bundled map files (None: discover again)."""
    global _maps_dir_override
    _maps_dir_override = None if path is None else Path(path)


def maps_dir() -> Path:
    """The bundled maps: the configured directory, else ``./maps``, else the repository's."""
    if _maps_dir_override is not None:
        return _maps_dir_override
    local = Path.cwd() / MAPS_DIRNAME
    return local if local.is_dir() else _REPO_ROOT / MAPS_DIRNAME


def _comment_text(text: str) -> str:
    """The leading ``;`` comment lines of a map file, joined into one sentence-like string."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith(COMMENT_PREFIX):
            break
        lines.append(line[len(COMMENT_PREFIX):].strip())
    return " ".join(lines)


def _title_from_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize()


def _map_path(name: str) -> Path:
    if not isinstance(name, str) or not MAP_NAME_PATTERN.match(name):
        raise ValueError(f"invalid map name {name!r} (letters, digits, '_' and '-' only)")
    path = maps_dir() / f"{name}{MAP_FILE_SUFFIX}"
    if not path.is_file():
        available = ", ".join(m["name"] for m in list_map_files()) or "none"
        raise ValueError(f"unknown map {name!r} (available: {available})")
    return path


def load_map_file(name: str, *, multi: bool = False) -> TrueGrid:
    """The bundled map ``name`` (file stem), named after it; ``multi`` accepts several ``P``."""
    return TrueGrid.load(_map_path(name), multi=multi)


def list_map_files() -> list[dict[str, Any]]:
    """Catalog entries of the bundled map files, sorted by name: ``name``, ``title``,
    ``description`` (plain English), ``source`` (the file's leading comment), ``rows``,
    ``cols``, ``survivors`` and
    ``high_slip`` (whether the map has '~' terrain). Unreadable files are skipped."""
    directory = maps_dir()
    if not directory.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*{MAP_FILE_SUFFIX}")):
        name = path.stem
        if not MAP_NAME_PATTERN.match(name):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            grid = TrueGrid.from_ascii(text, name=name, multi=True)
        except (OSError, ValueError):
            continue
        comment = _comment_text(text)
        title, description = MAP_FILE_INFO.get(name, (_title_from_name(name), comment))
        entries.append({
            "name": name,
            "title": title,
            "description": description,
            "source": comment,
            "rows": grid.rows,
            "cols": grid.cols,
            "survivors": len(grid.survivors),
            "high_slip": bool(grid.high_slip.any()),
            "ascii": grid.to_ascii(),
        })
    return entries


# --- the pre-generated library --------------------------------------------------------------
def library_key(scenario: str, rows: int, cols: int, seed: int) -> str:
    """Key of one library entry: ``"<scenario>/<rows>x<cols>/<seed>"``."""
    return f"{scenario}/{int(rows)}x{int(cols)}/{int(seed)}"


def build_map_library(
    scenarios: Sequence[str] | None = None,
    sizes: Iterable[int] = LIBRARY_SIZES,
    seeds: Iterable[int] = LIBRARY_SEEDS,
) -> dict[str, Any]:
    """The library document: ``mapgen.generate(scenario, n, n, seed).to_ascii()`` for every
    scenario (default: ``mapgen.GENERATED_SCENARIOS``), square size ``n`` and seed.

    Run by the static build with CPython (imports :mod:`rescue.mapgen`, hence scipy).
    """
    from rescue import mapgen

    names = tuple(scenarios) if scenarios is not None else mapgen.GENERATED_SCENARIOS
    size_list = [int(n) for n in sizes]
    seed_list = [int(s) for s in seeds]
    maps: dict[str, str] = {}
    for name in names:
        if name not in mapgen.GENERATED_SCENARIOS:
            raise ValueError(f"{name!r} is not a generated scenario")
        for n in size_list:
            for seed in seed_list:
                maps[library_key(name, n, n, seed)] = mapgen.generate(name, n, n, seed).to_ascii()
    return {
        "version": LIBRARY_FORMAT_VERSION,
        "scenarios": list(names),
        "sizes": size_list,
        "seeds": seed_list,
        "maps": maps,
    }


def write_map_library(path: str | Path, **kwargs: Any) -> Path:
    """Write :func:`build_map_library` (``kwargs`` passed on) as compact JSON to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = build_map_library(**kwargs)
    target.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8",
                      newline="\n")
    return target


@dataclass(frozen=True)
class MapLibrary:
    """A loaded ``maps_library.json``: ASCII maps by :func:`library_key`."""

    maps: Mapping[str, str]
    scenarios: tuple[str, ...]
    sizes: tuple[int, ...]
    seeds: tuple[int, ...]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> MapLibrary:
        """Parse the JSON document (``ValueError`` for another format version)."""
        version = int(document.get("version", 0))
        if version != LIBRARY_FORMAT_VERSION:
            raise ValueError(f"unsupported map library version {version}")
        return cls(
            maps=dict(document["maps"]),
            scenarios=tuple(str(s) for s in document.get("scenarios", ())),
            sizes=tuple(int(n) for n in document.get("sizes", ())),
            seeds=tuple(int(s) for s in document.get("seeds", ())),
        )

    def has(self, scenario: str, rows: int, cols: int, seed: int) -> bool:
        """True if the library holds this map."""
        return library_key(scenario, rows, cols, seed) in self.maps

    def grid(self, scenario: str, rows: int, cols: int, seed: int) -> TrueGrid | None:
        """The library map as a :class:`TrueGrid` named ``scenario`` (None if absent)."""
        text = self.maps.get(library_key(scenario, rows, cols, seed))
        if text is None:
            return None
        return TrueGrid.from_ascii(text, name=scenario)

    def summary(self) -> dict[str, Any]:
        """What the catalog reports about the library."""
        return {
            "loaded": True,
            "scenarios": list(self.scenarios),
            "sizes": list(self.sizes),
            "seeds": list(self.seeds),
            "count": len(self.maps),
        }


_library: MapLibrary | None = None
_library_searched = False


def set_map_library(source: MapLibrary | Mapping[str, Any] | str | Path | None) -> None:
    """Use ``source`` as the map library: a :class:`MapLibrary`, its JSON document, a path to
    ``maps_library.json``, or None (forget it; it is discovered again on the next use)."""
    global _library, _library_searched
    if source is None:
        _library, _library_searched = None, False
        return
    if isinstance(source, MapLibrary):
        _library = source
    elif isinstance(source, Mapping):
        _library = MapLibrary.from_document(source)
    else:
        text = Path(source).read_text(encoding="utf-8")
        _library = MapLibrary.from_document(json.loads(text))
    _library_searched = True


def get_map_library() -> MapLibrary | None:
    """The map library: the one set with :func:`set_map_library`, else the first
    ``maps_library.json`` found in the working directory or next to the maps directory."""
    global _library, _library_searched
    if _library is None and not _library_searched:
        _library_searched = True
        for candidate in (Path.cwd() / LIBRARY_FILENAME, maps_dir().parent / LIBRARY_FILENAME):
            if candidate.is_file():
                try:
                    set_map_library(candidate)
                except (OSError, ValueError, KeyError):
                    continue
                break
    return _library


def needs_generator(scenario: str, rows: int, cols: int, seed: int) -> bool:
    """True if this generated map is not in the library (so scipy must be loaded)."""
    library = get_map_library()
    return library is None or not library.has(scenario, rows, cols, seed)


def generated_grid(scenario: str, rows: int, cols: int, seed: int) -> TrueGrid:
    """``mapgen.generate(scenario, rows, cols, seed)``, from the library when it holds it."""
    if scenario not in GENERATED_SCENARIO_INFO:
        raise ValueError(
            f"unknown scenario {scenario!r}; choose from {', '.join(GENERATED_SCENARIO_INFO)}"
        )
    for label, value in (("rows", rows), ("cols", cols)):
        if not MAPGEN_MIN_SIZE <= value <= WEB_MAX_MAP_SIZE:
            raise ValueError(
                f"{label} must be between {MAPGEN_MIN_SIZE} and {WEB_MAX_MAP_SIZE}, got {value}"
            )
    library = get_map_library()
    if library is not None:
        grid = library.grid(scenario, rows, cols, seed)
        if grid is not None:
            return grid
    from rescue import mapgen  # needs scipy: loaded on demand in the browser

    return mapgen.generate(scenario, rows, cols, seed)


# --- ASCII editing --------------------------------------------------------------------------
def _parse_chars(text: str) -> list[list[str]]:
    """The map's rows as character lists (comments and blank lines dropped).

    ``ValueError`` for unknown characters, ragged rows or an empty map; the start and survivor
    counts are not checked here (an edited map may be temporarily incomplete).
    """
    if not isinstance(text, str):
        raise ValueError("the map must be text")
    if len(text) > WEB_MAX_MAP_TEXT:
        raise ValueError(f"the map text is too long ({len(text)} characters; at most "
                         f"{WEB_MAX_MAP_TEXT})")
    rows: list[list[str]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIX):
            continue
        bad = sorted(set(line) - MAP_CHARS)
        if bad:
            col = next(i for i, ch in enumerate(line) if ch not in MAP_CHARS) + 1
            raise ValueError(f"line {line_no}, column {col}: unknown map character "
                             f"{line[col - 1]!r}")
        if rows and len(line) != len(rows[0]):
            raise ValueError(f"line {line_no}: row has {len(line)} cells, expected {len(rows[0])}")
        rows.append(list(line))
    if not rows:
        raise ValueError("map contains no rows")
    check_map_size(len(rows), len(rows[0]))
    return rows


def check_map_size(rows: int, cols: int) -> None:
    """``ValueError`` unless ``rows`` x ``cols`` is a size the web app accepts (at most
    :data:`WEB_MAX_MAP_SIZE` on both axes, so a browser step stays fast)."""
    for label, value in (("rows", rows), ("columns", cols)):
        if not 1 <= value <= WEB_MAX_MAP_SIZE:
            raise ValueError(f"the map has {value} {label}; the web app accepts at most "
                             f"{WEB_MAX_MAP_SIZE}")


def problem_cells(text: str) -> list[Pos]:
    """0-indexed cells of a map text that hold a character the map format does not know
    (comment lines and blank lines are not map rows), for the editor to outline."""
    cells: list[Pos] = []
    if not isinstance(text, str) or len(text) > WEB_MAX_MAP_TEXT:
        return cells
    row = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIX):
            continue
        cells.extend((row, c) for c, ch in enumerate(line) if ch not in MAP_CHARS)
        row += 1
    return cells


def _to_text(chars: Sequence[Sequence[str]]) -> str:
    return "\n".join("".join(row) for row in chars) + "\n"


def _find(chars: Sequence[Sequence[str]], ch: str) -> list[Pos]:
    return [(r, c) for r, row in enumerate(chars) for c, value in enumerate(row) if value == ch]


def _cell(op: Mapping[str, Any], rows: int, cols: int) -> Pos:
    raw = op.get("cell")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 2:
        raise ValueError(f"edit {op.get('op')!r} needs a cell [row, col], got {raw!r}")
    r, c = int(raw[0]), int(raw[1])
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"cell {fmt_pos((r, c))} is outside the {rows}x{cols} map")
    return (r, c)


def _size(op: Mapping[str, Any], key: str) -> int:
    value = op.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value:
        raise ValueError(f"resize needs an integer {key}, got {value!r}")
    n = int(value)
    if not EDITOR_MIN_SIZE <= n <= WEB_MAX_MAP_SIZE:
        raise ValueError(f"{key} must be between {EDITOR_MIN_SIZE} and {WEB_MAX_MAP_SIZE}")
    return n


def _first_free(chars: list[list[str]], avoid: set[Pos]) -> Pos | None:
    for r, row in enumerate(chars):
        for c, value in enumerate(row):
            if value in (CH_OPEN, CH_HIGH_SLIP) and (r, c) not in avoid:
                return (r, c)
    for r, row in enumerate(chars):
        for c, _ in enumerate(row):
            if (r, c) not in avoid:
                return (r, c)
    return None


def _resize(chars: list[list[str]], rows: int, cols: int) -> list[list[str]]:
    """Crop / pad with open cells; a start or survivor cut off moves to a free cell."""
    old_rows, old_cols = len(chars), len(chars[0])
    out = [[chars[r][c] if r < old_rows and c < old_cols else CH_OPEN for c in range(cols)]
           for r in range(rows)]
    lost = [chars[r][c] for r in range(old_rows) for c in range(old_cols)
            if (r >= rows or c >= cols) and chars[r][c] in (CH_START, CH_SURVIVOR)]
    for ch in lost:
        if ch == CH_SURVIVOR and _find(out, CH_SURVIVOR):
            continue  # keep only survivors that still fit (at least one)
        if ch == CH_START and _find(out, CH_START):
            continue
        spot = _first_free(out, set(_find(out, CH_START)) | set(_find(out, CH_SURVIVOR)))
        if spot is not None:
            out[spot[0]][spot[1]] = ch
    return out


def _apply(chars: list[list[str]], op: Mapping[str, Any]) -> list[str]:
    """Apply one edit in place; returns warnings for refused edits."""
    kind = op.get("op")
    rows, cols = len(chars), len(chars[0])
    if kind == "clear":
        for row in chars:
            for c, value in enumerate(row):
                if value not in (CH_START, CH_SURVIVOR):
                    row[c] = CH_OPEN
        return []
    if kind == "resize":
        new = _resize(chars, _size(op, "rows"), _size(op, "cols"))
        chars[:] = new
        return []
    if kind not in EDIT_OPS:
        raise ValueError(f"unknown edit {kind!r}; choose from {', '.join(EDIT_OPS)}")
    p = _cell(op, rows, cols)
    r, c = p
    here = chars[r][c]
    endpoint = here in (CH_START, CH_SURVIVOR)
    if kind == "toggle_debris":
        if endpoint:
            return [NO_DEBRIS_ON_ENDPOINT.format(pos=fmt_pos(p))]
        chars[r][c] = CH_OPEN if here == CH_DEBRIS else CH_DEBRIS
    elif kind == "paint":
        value = op.get("value")
        if value not in PAINT_CHARS:
            raise ValueError(f"paint value must be one of '.', '#', '~', got {value!r}")
        if endpoint:
            if value == CH_OPEN:
                return []
            return [NO_DEBRIS_ON_ENDPOINT.format(pos=fmt_pos(p)) if value == CH_DEBRIS
                    else NO_TERRAIN_ON_ENDPOINT.format(pos=fmt_pos(p))]
        chars[r][c] = value
    elif kind == "toggle_high_slip":
        if endpoint:
            return [NO_TERRAIN_ON_ENDPOINT.format(pos=fmt_pos(p))]
        chars[r][c] = CH_OPEN if here == CH_HIGH_SLIP else CH_HIGH_SLIP
    elif kind == "set_start":
        if here == CH_SURVIVOR:
            return [START_ON_SURVIVOR.format(pos=fmt_pos(p))]
        for q in _find(chars, CH_START):
            chars[q[0]][q[1]] = CH_OPEN
        chars[r][c] = CH_START
    elif kind == "set_survivor":
        if here == CH_START:
            return [SURVIVOR_ON_START.format(pos=fmt_pos(p))]
        for q in _find(chars, CH_SURVIVOR):
            chars[q[0]][q[1]] = CH_OPEN
        chars[r][c] = CH_SURVIVOR
    elif kind == "add_survivor":
        if here == CH_START:
            return [SURVIVOR_ON_START.format(pos=fmt_pos(p))]
        if here == CH_SURVIVOR:
            return [ALREADY_SURVIVOR.format(pos=fmt_pos(p))]
        chars[r][c] = CH_SURVIVOR
    elif kind == "remove_survivor":
        if here != CH_SURVIVOR:
            return [NOT_A_SURVIVOR.format(pos=fmt_pos(p))]
        if len(_find(chars, CH_SURVIVOR)) == 1:
            return [LAST_SURVIVOR.format(pos=fmt_pos(p))]
        chars[r][c] = CH_OPEN
    return []


def _solvability(grid: TrueGrid) -> tuple[bool, list[str]]:
    """Whether every survivor is reachable from the start on the true map, with warnings."""
    passable = grid.passable_mask()
    warnings = [
        UNREACHABLE_SURVIVOR.format(pos=fmt_pos(s), start=fmt_pos(grid.start))
        for s in grid.survivors
        if not reachable(passable, grid.start, s)
    ]
    return (not warnings, warnings)


def validate_map(text: str, mode: str = "core") -> dict[str, Any]:
    """Check an ASCII map: ``{ok, errors, warnings, rows, cols, survivors, solvable}``.

    ``problems`` lists the 0-indexed cells the editor outlines (unknown characters, walled-in
    survivors). ``ok`` means a mission can be built from it in ``mode`` (``core`` needs exactly one
    survivor; the extension modes accept several). ``solvable`` (None when the map is invalid)
    says whether every survivor is reachable from the start on the true map; an unsolvable map
    is still ``ok`` (the robot then fails with ``unreachable``) but gets a warning.
    """
    try:
        _parse_chars(text)  # characters, ragged rows and the web app's size limits first
        grid = TrueGrid.from_ascii(text, multi=True)
    except ValueError as exc:
        rows = cols = 0
        with contextlib.suppress(ValueError):
            chars = _parse_chars(text)
            rows, cols = len(chars), len(chars[0])
        return {"ok": False, "errors": [str(exc)], "warnings": [], "rows": rows, "cols": cols,
                "survivors": [], "solvable": None,
                "problems": [[r, c] for r, c in problem_cells(text)]}
    errors: list[str] = []
    if mode == "core" and grid.extra_survivors:
        errors.append(SEVERAL_SURVIVORS_CORE)
    solvable, warnings = _solvability(grid)
    passable = grid.passable_mask()
    walled = [s for s in grid.survivors if not reachable(passable, grid.start, s)]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": grid.rows,
        "cols": grid.cols,
        "survivors": [[int(s[0]), int(s[1])] for s in grid.survivors],
        "solvable": solvable,
        "problems": [[int(s[0]), int(s[1])] for s in walled],
    }


def edit_map(text: str, op: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one edit ``op`` to an ASCII map: ``{ascii, changed, warnings, solvable, valid}``.

    Ops (``op["op"]``, cells 0-indexed ``[row, col]``): ``toggle_debris`` ``{cell}`` (never on
    the start / a survivor), ``paint`` ``{cell, value}`` with value ``.``/``#``/``~`` (for
    drag painting), ``set_start`` ``{cell}``, ``set_survivor`` ``{cell}`` (moves the one
    survivor: every ``P`` is replaced), ``add_survivor`` / ``remove_survivor`` ``{cell}``
    (several survivors, extension modes; the last one stays), ``toggle_high_slip`` ``{cell}``
    (``.``/``#`` ↔ ``~``), ``resize`` ``{rows, cols}`` (crop / pad with open cells; a start or
    survivor cut off moves to the first free cell) and ``clear`` (everything open but the start
    and survivors). The output is the canonical text (no comments). A refused edit leaves the
    map unchanged and explains why in ``warnings``; malformed requests raise ``ValueError``.
    ``solvable`` is None while the map is invalid (``valid`` False, e.g. no survivor).
    """
    if not isinstance(op, Mapping):
        raise ValueError(f"an edit must be an object with an 'op', got {op!r}")
    chars = _parse_chars(text)
    before = _to_text(chars)
    warnings = _apply(chars, op)
    after = _to_text(chars)
    check = validate_map(after, mode="multi")
    return {
        "ascii": after,
        "changed": after != before,
        "warnings": warnings + list(check["warnings"]),
        "errors": list(check["errors"]),
        "valid": bool(check["ok"]),
        "solvable": check["solvable"],
        "problems": check["problems"],
        "rows": len(chars),
        "cols": len(chars[0]),
    }
