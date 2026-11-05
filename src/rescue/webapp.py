"""``python -m rescue.webapp``: build and serve the browser app (docs/webapp.md, plan §3.16).

Usage (from the repository root)::

    python -m rescue.webapp                      # build build/webapp/, serve it, open a browser
    python -m rescue.webapp --build-only --out _site     # build only (GitHub Pages workflow)
    python -m rescue.webapp --port 8765 --no-open        # another port, no browser
    python -m rescue.webapp --skip-build                 # serve an existing build

The site is static. The simulation runs the exact Python core in the browser via Pyodide
(``webapp/js/worker.js``), so the build only packages files:

* the front end, copied from ``webapp/`` without ``webapp/tests/`` and the development-only
  mock backend (:data:`SITE_IGNORED`);
* ``rescue.zip``: the ``rescue`` package without the parts the browser never imports
  (:data:`ZIP_EXCLUDED_DIRS`, :data:`ZIP_EXCLUDED_MODULES`), plus ``maps/*.txt`` and
  ``maps_library.json`` (pre-generated scenario maps, :func:`rescue.webmaps.build_map_library`,
  so the common case needs no scipy in the browser; the worker reads it inside the zip only).
  Entries are sorted and time-stamped :data:`ZIP_TIMESTAMP`, so two builds of the same tree give
  identical bytes;
* ``results.json``: the claims tables of the committed reports (:data:`REPORTS`) and the
  curated plots and figures (:data:`PLOTS`, :data:`FIGURES`) with captions taken from the
  docs; the images are copied under ``assets/``;
* ``version.json``: package version, git commit, build time and the pinned Pyodide release;
* ``.nojekyll`` (GitHub Pages serves the files as they are).

The server is :mod:`http.server` bound to 127.0.0.1 (not ``localhost``, docs/web.md); any
other address needs ``--allow-remote`` (it exposes the server to the network).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from rescue import __version__
from rescue.config import (
    WEB_DEFAULT_HOST,
    WEB_DEFAULT_PORT,
    WEB_PYODIDE_CDN,
    WEB_PYODIDE_NUMPY,
    WEB_PYODIDE_PYTHON,
    WEB_PYODIDE_VERSION,
    WEB_THREE_CDN,
    WEB_THREE_VERSION,
)

# --- paths -------------------------------------------------------------------------------------
#: Repository root (src/rescue/webapp.py → three levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Front-end sources, the package and the bundled maps.
WEBAPP_SOURCE_DIR = REPO_ROOT / "webapp"
PACKAGE_DIR = Path(__file__).resolve().parent
MAPS_DIR = REPO_ROOT / "maps"
MAP_GLOB = "*.txt"
#: Default output directory (git-ignored, like everything under build/).
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "webapp"

# --- site layout -------------------------------------------------------------------------------
ZIP_FILE = "rescue.zip"
LIBRARY_FILE = "maps_library.json"
RESULTS_FILE = "results.json"
VERSION_FILE = "version.json"
NOJEKYLL_FILE = ".nojekyll"
INDEX_FILE = "index.html"
#: Images used by the Results tab (report plots) and the Learn tab (simulator figures).
RESULTS_ASSETS_DIR = "assets/results"
FIGURES_ASSETS_DIR = "assets/figures"
#: Names in ``webapp/`` never copied into the site: tests, tooling caches and the mock backend
#: (``?mock=1`` replays ``webapp/tests/fixtures``, so it only works when ``webapp/`` itself is
#: served during development).
MOCK_MODULE = "mock.js"
SITE_IGNORED: tuple[str, ...] = (
    "tests", "node_modules", "__pycache__", "*.pyc", ".DS_Store", "Thumbs.db", "package*.json",
    MOCK_MODULE,
)
#: Text files whose ``__PYODIDE_VERSION__`` / ``__APP_VERSION__`` / ``__THREE_VERSION__``
#: placeholders the build fills
#: in (optional: the worker may read ``version.json`` instead).
TEMPLATED_SUFFIXES: tuple[str, ...] = (".html", ".js", ".mjs", ".css")
PLACEHOLDER_PYODIDE = "__PYODIDE_VERSION__"
PLACEHOLDER_APP = "__APP_VERSION__"
PLACEHOLDER_THREE = "__THREE_VERSION__"

# --- rescue.zip --------------------------------------------------------------------------------
#: Top-level names inside the zip.
ZIP_PACKAGE = "rescue"
ZIP_MAPS = "maps"
#: Sub-packages of ``rescue`` the browser never imports (pygame UI, experiment harness).
ZIP_EXCLUDED_DIRS: tuple[str, ...] = ("viz", "experiments")
#: Modules (paths relative to the package) left out: the extension harness / claims / plots /
#: demo (they import the experiment harness or matplotlib) and the entry points of the Pygame
#: apps and of this build. ``demo.py`` stays: the bridge reuses its map and plan formatting.
ZIP_EXCLUDED_MODULES: tuple[str, ...] = (
    "ext/experiments.py", "ext/claims.py", "ext/plots.py", "ext/demo.py",
    "sim.py", "compare.py", "replay.py", "webapp.py",
)
#: Only Python sources are packed from the package.
ZIP_PACKAGE_SUFFIX = ".py"
#: Fixed entry timestamp (the zip format's earliest date) for reproducible archives.
ZIP_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
#: Unix permission bits stored for every entry (rw-r--r--).
ZIP_FILE_MODE = 0o644
#: Modules a browser bundle must never import at module level (no wheels / not needed).
FORBIDDEN_TOP_LEVEL_IMPORTS: tuple[str, ...] = ("pygame", "matplotlib", "imageio")

# --- results.json ------------------------------------------------------------------------------
RESULTS_FORMAT = 1
#: Verdict symbols of the reports → verdict words (``⚠️`` also without its variation selector).
VERDICT_WORDS: dict[str, str] = {
    "✅": "supported", "❌": "refuted", "⚠️": "inconclusive", "⚠": "inconclusive",
}
#: Verdict of a descriptive report's findings (no threshold, no verdict).
DESCRIPTIVE_VERDICT = "descriptive"
DESCRIPTIVE_SYMBOL = "ℹ️"
#: Heading of the claims table in HYPOTHESES.md / EXTENSIONS.md.
CLAIMS_HEADING = "Claims"
#: Columns of a claims table (lower-cased header cells).
CLAIM_COLUMNS: tuple[str, ...] = ("#", "claim", "verdict", "measured")
#: Docs scanned for image captions (``![alt](path)`` plus the paragraph that follows).
CAPTION_DOCS: tuple[str, ...] = (
    "docs/results.md", "docs/extensions.md", "README.md", "docs/web.md", "docs/algorithm.md",
)
#: Longest caption kept (characters; cut at a sentence end).
MAX_CAPTION_CHARS = 420

# --- serving -----------------------------------------------------------------------------------
#: Extra MIME types (Windows registries often lack or mis-map these; ES modules need JS).
EXTRA_MIME_TYPES: dict[str, str] = {
    ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
    ".wasm": "application/wasm", ".zip": "application/zip", ".css": "text/css",
    ".svg": "image/svg+xml", ".gif": "image/gif", ".png": "image/png",
}
#: Seconds to wait before opening the browser (the server is already listening by then).
BROWSER_OPEN_DELAY_S = 0.3
#: Prefix of the line announcing the server URL (tests parse it).
SERVING_PREFIX = "Serving "
BYTES_PER_KIB = 1024
#: Packages whose versions ``version.json`` records (they generate the map library).
BUILD_PACKAGES: tuple[str, ...] = ("numpy", "scipy")


@dataclass(frozen=True)
class ReportSource:
    """One committed report shown in the Results tab."""

    id: str
    title: str
    path: str  # relative to the repository root
    descriptive: bool = False  # no claims table: its first table's rows are reported instead
    command: str = ""


#: Reports parsed into ``results.json`` (paths relative to the repository root).
REPORTS: tuple[ReportSource, ...] = (
    ReportSource("main", "Core hypotheses (H1–H7)", "results/HYPOTHESES.md",
                 command="python -m rescue.experiments --config experiments/main.yaml"),
    ReportSource("ext", "Extension claims (X1–X6)", "results/ext/EXTENSIONS.md",
                 command="python -m rescue.ext.experiments"),
    ReportSource("lao", "LAO* vs value iteration", "results/lao/LAO.md", descriptive=True,
                 command="python -m rescue.experiments --config experiments/lao.yaml "
                         "--out results/lao && python -m rescue.experiments.lao_report "
                         "results/lao"),
)


@dataclass(frozen=True)
class ImageSource:
    """A curated image: ``group`` is the report id (plots) or ``"figures"``."""

    group: str
    path: str  # relative to the repository root
    title: str  # fallback title when no doc references the image


#: Report plots in reading order (the Results tab).
PLOTS: tuple[ImageSource, ...] = (
    ImageSource("main", "results/plots/success_vs_p.png", "Success vs slip probability"),
    ImageSource("main", "results/plots/steps_vs_p.png", "Steps vs slip probability"),
    ImageSource("main", "results/plots/wasted_moves_vs_p.png", "Wasted moves vs p"),
    ImageSource("main", "results/plots/paired_wasted_moves_diff.png",
                "Paired wasted-move difference"),
    ImageSource("main", "results/plots/replans_vs_p.png", "Replans vs p"),
    ImageSource("main", "results/plots/paired_replans_diff.png", "Paired replan difference"),
    ImageSource("main", "results/plots/cells_explored_vs_p.png", "Cells explored vs p"),
    ImageSource("main", "results/plots/tiebreak_ablation.png", "Tie-break ablation"),
    ImageSource("main", "results/plots/planning_time_vs_size.png", "Planning time vs size"),
    ImageSource("main", "results/plots/success_under_aftershocks.png",
                "Success under aftershocks"),
    ImageSource("main", "results/plots/sensor_radius.png", "Sensor radius"),
    ImageSource("main", "results/plots/replan_mode.png", "Replan mode"),
    ImageSource("ext", "results/ext/plots/search_steps.png", "Search: steps known vs unknown"),
    ImageSource("ext", "results/ext/plots/multi_tours.png", "Multi-survivor tours"),
    ImageSource("ext", "results/ext/plots/noisy.png", "Noisy sensing (false negatives)"),
    ImageSource("ext", "results/ext/plots/noisy_fp.png", "Noisy sensing (false positives)"),
    ImageSource("lao", "results/lao/plots/planning_time_vs_size.png",
                "Planning time per plan: andor, vi_expected and lao_expected"),
)

#: Simulator figures (the Learn tab).
FIGURES: tuple[ImageSource, ...] = (
    ImageSource("figures", "docs/figures/deck_demo.gif", "The AND-OR agent on deck_demo"),
    ImageSource("figures", "docs/figures/plan_tree_deck_demo.png",
                "Initial AND-OR plan tree on deck_demo"),
    ImageSource("figures", "docs/figures/sim_deck_demo.png", "Simulator after the replan"),
    ImageSource("figures", "docs/figures/sim_deck_figure.png", "deck_figure, true-map overlay"),
    ImageSource("figures", "docs/figures/compare_collapsed_building.png",
                "Head-to-head comparison"),
    ImageSource("figures", "docs/figures/ext_search.png", "Search mode"),
    ImageSource("figures", "docs/figures/ext_multi.png", "Multi mode"),
    ImageSource("figures", "docs/figures/ext_noisy.png", "Noisy mode"),
)


@dataclass
class BuildReport:
    """What :func:`build_site` wrote (sizes in bytes)."""

    out_dir: Path
    files: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Sum of the recorded file sizes."""
        return sum(self.files.values())


# === rescue.zip =================================================================================
def zip_members(package_dir: Path = PACKAGE_DIR, maps_dir: Path = MAPS_DIR) -> list[
        tuple[str, Path]]:
    """``(archive name, source path)`` for every file of ``rescue.zip``, sorted by name.

    The package's ``*.py`` files minus :data:`ZIP_EXCLUDED_DIRS` / :data:`ZIP_EXCLUDED_MODULES`
    (and ``__pycache__``), then ``maps/*.txt``.
    """
    members: list[tuple[str, Path]] = []
    for path in sorted(package_dir.rglob(f"*{ZIP_PACKAGE_SUFFIX}")):
        rel = path.relative_to(package_dir)
        parts = rel.parts
        if "__pycache__" in parts:
            continue
        if len(parts) > 1 and parts[0] in ZIP_EXCLUDED_DIRS:
            continue
        if rel.as_posix() in ZIP_EXCLUDED_MODULES:
            continue
        members.append((f"{ZIP_PACKAGE}/{rel.as_posix()}", path))
    for path in sorted(maps_dir.glob(MAP_GLOB)):
        members.append((f"{ZIP_MAPS}/{path.name}", path))
    return sorted(members)


def _zip_info(name: str) -> zipfile.ZipInfo:
    """A deterministic entry header for ``name``."""
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ZIP_FILE_MODE << 16
    return info


def write_zip(target: Path, members: Sequence[tuple[str, Path]],
              extra: dict[str, bytes] | None = None) -> Path:
    """Write ``members`` (plus ``extra`` name → bytes) to ``target`` reproducibly."""
    entries: dict[str, bytes] = {name: path.read_bytes() for name, path in members}
    entries.update(extra or {})
    with zipfile.ZipFile(target, "w") as archive:
        for name in sorted(entries):
            archive.writestr(_zip_info(name), entries[name])
    return target


_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)


def top_level_imports(source: str) -> set[str]:
    """Top-level package names imported anywhere in ``source`` (``import x`` / ``from x``)."""
    return {match.group(1).split(".")[0] for match in _IMPORT_RE.finditer(source)}


def forbidden_imports(members: Sequence[tuple[str, Path]]) -> dict[str, set[str]]:
    """Archive name → :data:`FORBIDDEN_TOP_LEVEL_IMPORTS` it imports (empty dict: clean)."""
    found: dict[str, set[str]] = {}
    for name, path in members:
        if not name.endswith(ZIP_PACKAGE_SUFFIX):
            continue
        bad = top_level_imports(path.read_text(encoding="utf-8")) & set(
            FORBIDDEN_TOP_LEVEL_IMPORTS)
        if bad:
            found[name] = bad
    return found


# === maps_library.json ==========================================================================
def build_maps_library() -> dict[str, Any]:
    """The document written to ``maps_library.json``: :func:`rescue.webmaps.build_map_library`
    with its defaults (every generated scenario, its library sizes and seeds; imports mapgen,
    hence scipy, which the build machine has). The bridge reads the same format."""
    from rescue.webmaps import build_map_library

    return build_map_library()


# === results.json ===============================================================================
def split_row(line: str) -> list[str]:
    """Cells of a Markdown table row; ``|`` inside backticks or escaped (``\\|``) is kept."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] == "|":
            current.append("|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    cells.append("".join(current).strip())
    return cells


def _is_separator(cells: Sequence[str]) -> bool:
    """True for a Markdown table separator row (``---``, ``:--:`` …)."""
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells)


def parse_tables(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Every Markdown table in ``lines``: ``{"header": [...], "rows": [[...], ...]}``."""
    tables: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines)
                and _is_separator(split_row(lines[i + 1]))):
            header = split_row(lines[i])
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            tables.append({"header": header, "rows": rows})
        else:
            i += 1
    return tables


def split_sections(text: str, level: int) -> list[tuple[str, list[str]]]:
    """``(heading, body lines)`` for each heading of exactly ``level`` ``#``; the text before
    the first heading is returned under the heading ``""``."""
    marker = "#" * level + " "
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.startswith(marker):
            sections.append((line[len(marker):].strip(), []))
        elif line.startswith("#" * (level - 1) + " ") and level > 1 and sections[-1][0]:
            sections.append(("", [line]))  # a higher-level heading closes the section
        else:
            sections[-1][1].append(line)
    return sections


def verdict_word(symbol: str) -> str:
    """``supported`` / ``refuted`` / ``inconclusive`` for a verdict cell (``ValueError`` if the
    cell is none of :data:`VERDICT_WORDS`)."""
    key = symbol.strip()
    if key in VERDICT_WORDS:
        return VERDICT_WORDS[key]
    raise ValueError(f"unknown verdict {symbol!r}")


def _meta(lines: Sequence[str]) -> dict[str, str]:
    """``- Key: value`` bullets → dict (the run metadata at the top of a report)."""
    meta: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^- ([^:]+):\s*(.+)$", line.strip())
        if match:
            meta[match.group(1).strip()] = match.group(2).strip()
    return meta


def _paragraphs(lines: Sequence[str]) -> list[str]:
    """Non-table, non-heading paragraphs of ``lines`` (joined, comments removed)."""
    text = re.sub(r"<!--.*?-->", "", "\n".join(lines), flags=re.DOTALL)
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block and not block.startswith(("|", "#", "-", "!")):
            out.append(" ".join(part.strip() for part in block.splitlines()))
    return out


def parse_claims_report(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Run metadata and claims of a HYPOTHESES.md / EXTENSIONS.md report.

    Each claim: ``{"id", "claim", "verdict", "symbol", "measured", "details"}`` where
    ``details`` is the Markdown of the claim's ``### <id>.`` section (``""`` if none).
    ``ValueError`` when the report has no claims table or a row has an unknown verdict.
    """
    top = split_sections(text, 2)
    meta = _meta(top[0][1])
    claims_body = next((body for heading, body in top if heading == CLAIMS_HEADING), None)
    if claims_body is None:
        raise ValueError(f"no '## {CLAIMS_HEADING}' section")
    tables = [t for t in parse_tables(claims_body)
              if [h.lower() for h in t["header"]] == list(CLAIM_COLUMNS)]
    if not tables:
        raise ValueError("no claims table (columns #, Claim, Verdict, Measured)")
    details = {}
    for heading, body in split_sections(text, 3):
        match = re.match(r"^(\w+)\.\s", heading)
        if match:
            details[match.group(1)] = "\n".join(body).strip()
    claims = []
    for row in tables[0]["rows"]:
        if len(row) != len(CLAIM_COLUMNS):
            raise ValueError(f"claims row with {len(row)} cells: {row!r}")
        claim_id, claim, symbol, measured = row
        claims.append({
            "id": claim_id,
            "claim": claim,
            "verdict": verdict_word(symbol),
            "symbol": symbol.strip(),
            "measured": measured,
            "details": details.get(claim_id, ""),
        })
    return meta, claims


def parse_descriptive_report(text: str, report_id: str) -> tuple[str, list[dict[str, Any]],
                                                                 list[dict[str, Any]]]:
    """Intro, findings and tables of a descriptive report (LAO.md).

    Every ``## `` section's tables become ``{"title", "header", "rows", "notes"}``; the rows of
    the first table become findings with verdict :data:`DESCRIPTIVE_VERDICT` (ids
    ``<REPORT>1``, ``<REPORT>2`` …), ``claim`` = the row label and ``measured`` = the other
    cells with their column names.
    """
    sections = split_sections(text, 2)
    intro = " ".join(_paragraphs(sections[0][1]))
    tables: list[dict[str, Any]] = []
    for heading, body in sections[1:]:
        notes = _paragraphs(body)
        for table in parse_tables(body):
            tables.append({"title": heading, **table, "notes": notes})
    if not tables:
        raise ValueError("descriptive report without tables")
    first = tables[0]
    prefix = report_id.upper()
    findings = []
    for number, row in enumerate(first["rows"], start=1):
        measured = "; ".join(f"{h}: {v}" for h, v in zip(first["header"][1:], row[1:],
                                                         strict=False))
        findings.append({
            "id": f"{prefix}{number}",
            "claim": f"{row[0]} — {first['header'][0]}" if first["header"] else row[0],
            "verdict": DESCRIPTIVE_VERDICT,
            "symbol": DESCRIPTIVE_SYMBOL,
            "measured": measured,
            "details": "\n".join(first["notes"]),
        })
    return intro, findings, tables


_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def harvest_captions(repo_root: Path = REPO_ROOT,
                     docs: Sequence[str] = CAPTION_DOCS) -> dict[str, dict[str, str]]:
    """Repo-relative image path → ``{"title", "caption", "doc", "section"}`` from the docs.

    ``title`` is the image's alt text; ``caption`` the first prose paragraph after the image
    (or after the run of images it belongs to), comments removed and cut to
    :data:`MAX_CAPTION_CHARS` at a sentence end. The first doc mentioning an image wins.
    """
    found: dict[str, dict[str, str]] = {}
    for doc in docs:
        path = repo_root / doc
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        section = ""
        for index, line in enumerate(lines):
            if line.startswith("#"):
                section = line.lstrip("#").strip()
            for match in _IMAGE_RE.finditer(line):
                target = (path.parent / match.group(2)).resolve()
                try:
                    rel = target.relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    continue
                if rel in found:
                    continue
                found[rel] = {
                    "title": match.group(1).strip(),
                    "caption": _caption_after(lines, index),
                    "doc": doc,
                    "section": section,
                }
    return found


def _caption_after(lines: Sequence[str], index: int) -> str:
    """The prose paragraph following the image line ``index`` (see :func:`harvest_captions`)."""
    i = index + 1
    while i < len(lines) and (not lines[i].strip() or _IMAGE_RE.match(lines[i].strip())
                              or lines[i].strip().startswith("<!--")):
        i += 1
    block: list[str] = []
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![")):
        block.append(lines[i].strip())
        i += 1
    if block and block[-1].endswith(":"):  # "… as follows:" + a bullet list: keep the list
        while i < len(lines) and not lines[i].strip():
            i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|")):
            block.append(re.sub(r"^[*-]\s+", "• ", lines[i].strip()))
            i += 1
    text = re.sub(r"<!--.*?-->", "", " ".join(block)).strip()
    if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
        text = text[1:-1].strip()  # an italic caption line
    return _shorten(text)


def _shorten(text: str, limit: int = MAX_CAPTION_CHARS) -> str:
    """``text`` cut after the last sentence end before ``limit`` (or at ``limit`` + "…")."""
    text = re.sub(r"\s{2,}", " ", text)
    if len(text) <= limit:
        return text
    cut = max(text.rfind(". ", 0, limit), text.rfind(".) ", 0, limit))
    return text[:cut + 1] if cut > 0 else text[:limit].rstrip() + "…"


def _image_entries(images: Sequence[ImageSource], captions: dict[str, dict[str, str]],
                   assets_dir: str, repo_root: Path) -> list[dict[str, Any]]:
    """``results.json`` entries for the existing images of ``images``."""
    entries: list[dict[str, Any]] = []
    for image in images:
        if not (repo_root / image.path).is_file():
            continue
        doc = captions.get(image.path, {})
        site_path = f"{assets_dir}/{_asset_name(image)}"
        entries.append({
            "id": Path(_asset_name(image)).stem,
            "group": image.group,
            "src": site_path,
            "source": image.path,
            "title": doc.get("title") or image.title,
            "caption": doc.get("caption") or doc.get("title") or image.title,
            "doc": doc.get("doc", ""),
            "section": doc.get("section", ""),
        })
    return entries


def _asset_name(image: ImageSource) -> str:
    """File name under the assets directory (report plots are prefixed by their report)."""
    name = Path(image.path).name
    return name if image.group == "figures" else f"{image.group}_{name}"


def build_results(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The ``results.json`` document (see the module docstring)."""
    captions = harvest_captions(repo_root)
    reports = []
    for source in REPORTS:
        text = (repo_root / source.path).read_text(encoding="utf-8")
        title_line = next((ln for ln in text.splitlines() if ln.startswith("# ")), "")
        entry: dict[str, Any] = {
            "id": source.id,
            "title": source.title,
            "heading": title_line[2:].strip(),
            "source": source.path,
            "command": source.command,
            "descriptive": source.descriptive,
        }
        if source.descriptive:
            intro, claims, tables = parse_descriptive_report(text, source.id)
            entry.update(meta={}, intro=intro, claims=claims, tables=tables)
        else:
            meta, claims = parse_claims_report(text)
            entry.update(meta=meta, intro="", claims=claims, tables=[])
        counts = {word: 0 for word in (*dict.fromkeys(VERDICT_WORDS.values()),
                                        DESCRIPTIVE_VERDICT)}
        for claim in entry["claims"]:
            counts[claim["verdict"]] += 1
        entry["counts"] = counts
        reports.append(entry)
    return {
        "format": RESULTS_FORMAT,
        "verdicts": {"supported": "✅", "refuted": "❌", "inconclusive": "⚠️",
                     DESCRIPTIVE_VERDICT: DESCRIPTIVE_SYMBOL},
        "reports": reports,
        "plots": _image_entries(PLOTS, captions, RESULTS_ASSETS_DIR, repo_root),
        "figures": _image_entries(FIGURES, captions, FIGURES_ASSETS_DIR, repo_root),
    }


# === version.json ===============================================================================
def git_commit(repo_root: Path = REPO_ROOT) -> tuple[str, bool | None]:
    """``(short commit, dirty)`` of ``repo_root``: ``$GITHUB_SHA`` in CI, else ``git``
    (read-only commands); ``("unknown", None)`` without git."""
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha[:7], False
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit or "unknown", bool(status.strip())


def build_time() -> str:
    """ISO-8601 UTC build time (``$SOURCE_DATE_EPOCH`` when set, for reproducible builds)."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = (datetime.fromtimestamp(int(epoch), tz=UTC) if epoch and epoch.isdigit()
              else datetime.now(tz=UTC))
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_version(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The ``version.json`` document."""
    commit, dirty = git_commit(repo_root)
    api = None
    with contextlib.suppress(ImportError):
        from rescue import webapi

        api = getattr(webapi, "API_VERSION", None)
    return {
        "app": "rescue-robot",
        "version": __version__,
        "api": api,
        "commit": commit,
        "dirty": dirty,
        "built_at": build_time(),
        "pyodide": WEB_PYODIDE_VERSION,
        "pyodide_url": WEB_PYODIDE_CDN.format(version=WEB_PYODIDE_VERSION),
        "python": WEB_PYODIDE_PYTHON,
        "numpy": WEB_PYODIDE_NUMPY,
        "three": WEB_THREE_VERSION,
        "three_url": WEB_THREE_CDN.format(version=WEB_THREE_VERSION),
        "build_packages": build_packages(),
        "zip": ZIP_FILE,
        "maps_library": LIBRARY_FILE,
        "results": RESULTS_FILE,
    }


def build_packages() -> dict[str, str | None]:
    """Versions of the CPython packages the build used to pre-generate the map library (the
    site's "same seed, same map" promise depends on them; None if one is missing)."""
    from importlib import metadata

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in BUILD_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# === build ======================================================================================
def _dump_json(data: Any) -> bytes:
    """Compact, stable JSON (UTF-8, keys in insertion order)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _prepare_out_dir(out_dir: Path) -> None:
    """Create ``out_dir``, emptying it first if it is an earlier build (it holds
    :data:`VERSION_FILE`) or empty. ``FileExistsError`` for any other non-empty directory, so a
    mistyped ``--out`` never deletes unrelated files."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"{out_dir} exists and is not a directory")
        if any(out_dir.iterdir()):
            if not (out_dir / VERSION_FILE).is_file():
                raise FileExistsError(
                    f"{out_dir} is not empty and is not an earlier web app build "
                    f"(no {VERSION_FILE}); choose another --out")
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _fill_placeholders(site_dir: Path) -> None:
    """Replace the version placeholders in the copied front-end text files."""
    for path in site_dir.rglob("*"):
        if path.suffix in TEMPLATED_SUFFIXES and path.is_file():
            text = path.read_text(encoding="utf-8")
            if (PLACEHOLDER_PYODIDE in text or PLACEHOLDER_APP in text
                    or PLACEHOLDER_THREE in text):
                text = text.replace(PLACEHOLDER_PYODIDE, WEB_PYODIDE_VERSION)
                text = text.replace(PLACEHOLDER_APP, __version__)
                text = text.replace(PLACEHOLDER_THREE, WEB_THREE_VERSION)
                path.write_text(text, encoding="utf-8")


def build_site(out_dir: Path = DEFAULT_OUT_DIR, *, source_dir: Path = WEBAPP_SOURCE_DIR,
               repo_root: Path = REPO_ROOT,
               library: Callable[[], dict[str, Any]] | None = None,
               log: Callable[[str], None] | None = None) -> BuildReport:
    """Build the static site into ``out_dir`` (see the module docstring).

    ``library`` overrides the map-library builder (tests pass a small one). ``FileNotFoundError``
    if ``source_dir`` has no ``index.html``; ``ValueError`` if a packed module imports one of
    :data:`FORBIDDEN_TOP_LEVEL_IMPORTS`.
    """
    say = log or (lambda _message: None)
    out_dir = Path(out_dir).resolve()
    source_dir = Path(source_dir)
    if not (source_dir / INDEX_FILE).is_file():
        raise FileNotFoundError(f"{source_dir / INDEX_FILE} not found (front-end sources)")
    _prepare_out_dir(out_dir)
    report = BuildReport(out_dir)

    say(f"copying {source_dir}")
    shutil.copytree(source_dir, out_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*SITE_IGNORED))
    _fill_placeholders(out_dir)

    say("generating the map library")
    library_data = (library or build_maps_library)()
    library_bytes = _dump_json(library_data)

    members = zip_members(repo_root / "src" / ZIP_PACKAGE, repo_root / "maps")
    bad = forbidden_imports(members)
    if bad:
        details = ", ".join(f"{name} ({', '.join(sorted(mods))})" for name, mods in bad.items())
        raise ValueError(f"browser bundle must not import {details}")
    say(f"packing {ZIP_FILE} ({len(members)} files)")
    write_zip(out_dir / ZIP_FILE, members, extra={LIBRARY_FILE: library_bytes})

    say("collecting results and figures")
    results = build_results(repo_root)
    for entry in (*results["plots"], *results["figures"]):
        target = out_dir / entry["src"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / entry["source"], target)
    for image in (*PLOTS, *FIGURES):
        if not (repo_root / image.path).is_file():
            report.warnings.append(f"missing image {image.path}")
    (out_dir / RESULTS_FILE).write_bytes(_dump_json(results))
    (out_dir / VERSION_FILE).write_bytes(_dump_json(build_version(repo_root)))
    (out_dir / NOJEKYLL_FILE).write_bytes(b"")

    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            report.files[path.relative_to(out_dir).as_posix()] = path.stat().st_size
    return report


# === serve ======================================================================================
class SiteRequestHandler(SimpleHTTPRequestHandler):
    """Static handler with the MIME types the app needs and no caching (rebuilds show at
    once). ``quiet`` silences the per-request log lines."""

    extensions_map: ClassVar[dict[str, str]] = {**SimpleHTTPRequestHandler.extensions_map,
                                                **EXTRA_MIME_TYPES}
    quiet: ClassVar[bool] = False

    def end_headers(self) -> None:
        """Add ``Cache-Control: no-cache`` to every response."""
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Request log (stderr), unless :attr:`quiet`."""
        if not self.quiet:
            super().log_message(format, *args)


def make_server(site_dir: Path, host: str = WEB_DEFAULT_HOST, port: int = WEB_DEFAULT_PORT,
                *, quiet: bool = False) -> ThreadingHTTPServer:
    """A threading HTTP server for ``site_dir`` bound to ``host:port`` (port 0 = any free)."""
    handler = type("Handler", (SiteRequestHandler,), {"quiet": quiet})
    return ThreadingHTTPServer((host, port), functools.partial(handler,
                                                               directory=str(site_dir)))


def server_url(server: ThreadingHTTPServer) -> str:
    """``http://host:port/`` of a bound server."""
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"


@contextlib.contextmanager
def serving(site_dir: Path, host: str = WEB_DEFAULT_HOST, port: int = 0,
            *, quiet: bool = True) -> Iterator[str]:
    """Serve ``site_dir`` in a background thread for the ``with`` block; yields the URL."""
    server = make_server(site_dir, host, port, quiet=quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server_url(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def serve(site_dir: Path, host: str = WEB_DEFAULT_HOST, port: int = WEB_DEFAULT_PORT,
          *, open_browser: bool = True) -> int:
    """Serve ``site_dir`` until Ctrl+C; returns the exit code (1 if the port is taken)."""
    try:
        server = make_server(site_dir, host, port)
    except OSError as exc:
        print(f"error: cannot listen on {host}:{port} ({exc}); try --port", file=sys.stderr)
        return 1
    url = server_url(server)
    print(f"{SERVING_PREFIX}{site_dir} at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(BROWSER_OPEN_DELAY_S, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0


def is_loopback(host: str) -> bool:
    """True for an address only this machine can reach (127.0.0.0/8, ::1, ``localhost``)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# === CLI ========================================================================================
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m rescue.webapp",
        description="Build the browser app (Pyodide + the exact Python core) and serve it.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--host", default=WEB_DEFAULT_HOST,
                        help=f"address to bind (default {WEB_DEFAULT_HOST}; a non-loopback "
                             "address needs --allow-remote)")
    parser.add_argument("--allow-remote", action="store_true",
                        help="allow a --host that other machines can reach (e.g. 0.0.0.0)")
    parser.add_argument("--port", type=int, default=WEB_DEFAULT_PORT,
                        help=f"port (default {WEB_DEFAULT_PORT}; 0 = any free port)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true", help="build, do not serve")
    mode.add_argument("--skip-build", action="store_true",
                      help="serve an existing build without rebuilding")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the exit code (2 for a failed build)."""
    from rescue.cli_utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if not args.build_only and not is_loopback(args.host) and not args.allow_remote:
        print(f"error: --host {args.host} would expose the server to other machines; pass "
              f"--allow-remote if you mean it (default {WEB_DEFAULT_HOST})", file=sys.stderr)
        return 2
    if args.host == "localhost":
        print("warning: 'localhost' may resolve to IPv6 first and fail in some browsers; "
              f"prefer {WEB_DEFAULT_HOST} (docs/web.md)", file=sys.stderr)
    out_dir = args.out.resolve()
    if not args.skip_build:
        try:
            report = build_site(out_dir, log=lambda message: print(message, flush=True))
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for warning in report.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"built {out_dir} ({len(report.files)} files, "
              f"{report.total_bytes / BYTES_PER_KIB:.0f} KiB)", flush=True)
    elif not (out_dir / INDEX_FILE).is_file():
        print(f"error: no build in {out_dir}; run without --skip-build", file=sys.stderr)
        return 2
    if args.build_only:
        return 0
    return serve(out_dir, args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    raise SystemExit(main())
