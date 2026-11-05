"""Browser app build and server (``python -m rescue.webapp``, plan §3.16, docs/webapp.md).

One build into a temporary directory is shared by the module. It checks the contents of
``rescue.zip`` (no pygame UI, no experiment harness, no pygame / matplotlib imports), that the
zipped package imports in a fresh interpreter with pygame, matplotlib, scipy and PyYAML blocked
and replays the CLI demo through ``rescue.webapi.WebSession``, the ``results.json`` claims
against the committed reports, ``version.json``, and the HTTP server (in-process and the real
CLI on an ephemeral 127.0.0.1 port). The front end's own tests (``node --test``) run when
Node.js is installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from rescue import webapp
from rescue.config import (
    DEFAULT_SLIP_P,
    DEMO_SEED,
    WEB_PYODIDE_VERSION,
    WEB_THREE_VERSION,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)

#: Repository root (tests/test_webapp_build.py -> two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent
WEBAPI_SOURCE = REPO_ROOT / "src" / "rescue" / "webapi.py"
NODE_TESTS_DIR = REPO_ROOT / "webapp" / "tests"
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
#: Modules the zipped package must import without (blocked in the subprocess).
BLOCKED_MODULES = ("pygame", "matplotlib", "scipy", "yaml", "imageio")
#: Seconds allowed for subprocesses and HTTP requests.
SUBPROCESS_TIMEOUT_S = 180
HTTP_TIMEOUT_S = 10
SERVER_START_TIMEOUT_S = 60
#: Steps of the CLI demo (python -m rescue.demo: rescue after 9 steps).
DEMO_STEPS = 9
#: Upper bound on WebSession.step() calls in the replay subprocess (a stuck session fails).
MAX_SESSION_STEPS = 200
#: Sizes and seed count the map library must cover (plan §3.16).
LIBRARY_CONTRACT_SIZES = (10, 15, 20)
#: Size of the generated map the bundle test loads from the library (random_rubble, seed 3).
LIBRARY_MAP_SIZE = 15
LIBRARY_CONTRACT_SEEDS = 30


# --- fixtures ----------------------------------------------------------------------------------
def _stub_front_end(directory: Path) -> Path:
    """A minimal front end (used only while ``webapp/index.html`` does not exist yet)."""
    (directory / "js").mkdir(parents=True)
    (directory / "index.html").write_text(
        "<!doctype html><title>Rescue Robot</title><script type=module src=js/main.js></script>",
        encoding="utf-8")
    (directory / "js" / "main.js").write_text(
        f"export const PYODIDE = '{webapp.PLACEHOLDER_PYODIDE}';\n", encoding="utf-8")
    (directory / "tests").mkdir()
    (directory / "tests" / "skip.test.js").write_text("// not copied\n", encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def front_end(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A snapshot of ``webapp/`` when the front end exists (so files edited while the tests
    run cannot make the build and the comparison disagree), else a stub."""
    if (webapp.WEBAPP_SOURCE_DIR / webapp.INDEX_FILE).is_file():
        snapshot = tmp_path_factory.mktemp("front_end") / "webapp"
        shutil.copytree(webapp.WEBAPP_SOURCE_DIR, snapshot,
                        ignore=shutil.ignore_patterns("node_modules"))
        return snapshot
    return _stub_front_end(tmp_path_factory.mktemp("front_end"))


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory, front_end: Path) -> Path:
    """One full build (the real map library) into a temporary directory."""
    out = tmp_path_factory.mktemp("site") / "webapp"
    webapp.build_site(out, source_dir=front_end)
    return out


@pytest.fixture(scope="module")
def zip_names(site: Path) -> list[str]:
    """Entry names of the built ``rescue.zip``."""
    with zipfile.ZipFile(site / webapp.ZIP_FILE) as archive:
        return archive.namelist()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- site layout -------------------------------------------------------------------------------
def test_site_has_every_generated_file(site: Path, front_end: Path) -> None:
    for name in (webapp.INDEX_FILE, webapp.ZIP_FILE, webapp.RESULTS_FILE,
                 webapp.VERSION_FILE, webapp.NOJEKYLL_FILE):
        assert (site / name).is_file(), name
    # The map library is read inside rescue.zip only: no second copy at the site root.
    assert not (site / webapp.LIBRARY_FILE).exists()
    assert not (site / "tests").exists(), "front-end tests must not be published"
    assert not any(site.rglob(webapp.MOCK_MODULE)), "the mock backend is development-only"
    copied = {p.relative_to(front_end).as_posix() for p in front_end.rglob("*")
              if p.is_file() and "tests" not in p.relative_to(front_end).parts
              and "node_modules" not in p.parts and p.name != webapp.MOCK_MODULE}
    built = {p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file()}
    assert copied <= built


def test_placeholders_are_filled(site: Path) -> None:
    for path in site.rglob("*"):
        if path.suffix in webapp.TEMPLATED_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            assert webapp.PLACEHOLDER_PYODIDE not in text, path
            assert webapp.PLACEHOLDER_APP not in text, path
            assert webapp.PLACEHOLDER_THREE not in text, path


def test_out_dir_guard_refuses_unrelated_directories(tmp_path: Path, front_end: Path) -> None:
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        webapp.build_site(precious, source_dir=front_end, library=dict)
    assert (precious / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert webapp.main(["--build-only", "--out", str(precious)]) == 2


def test_missing_front_end_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        webapp.build_site(tmp_path / "out", source_dir=tmp_path / "nowhere", library=dict)


# --- rescue.zip --------------------------------------------------------------------------------
def test_zip_contains_the_core_and_the_maps(zip_names: list[str]) -> None:
    for module in ("__init__", "config", "grid", "belief", "model", "world", "plan", "runner",
                   "agents/__init__", "planners/andor", "ext/mission", "ext/agents",
                   "ext/sensing", "ext/scenarios", "mapgen"):
        assert f"rescue/{module}.py" in zip_names, module
    for module in ("webapi", "webmaps", "webtext"):
        assert f"rescue/{module}.py" in zip_names, module
    for path in sorted(webapp.MAPS_DIR.glob("*.txt")):
        assert f"maps/{path.name}" in zip_names
    assert webapp.LIBRARY_FILE in zip_names
    assert zip_names == sorted(zip_names)


def test_zip_excludes_ui_harness_and_cli_modules(zip_names: list[str]) -> None:
    for name in zip_names:
        assert "__pycache__" not in name and not name.endswith((".pyc", ".pyo")), name
        assert not name.startswith(("rescue/viz/", "rescue/experiments/")), name
    for module in webapp.ZIP_EXCLUDED_MODULES:
        assert f"rescue/{module}" not in zip_names, module


def test_no_module_in_the_zip_imports_pygame_or_matplotlib(site: Path) -> None:
    """Scan every packed source (not just the module-level imports)."""
    pattern = re.compile(r"^\s*(?:import|from)\s+(pygame|matplotlib|imageio)\b", re.MULTILINE)
    with zipfile.ZipFile(site / webapp.ZIP_FILE) as archive:
        for name in archive.namelist():
            if name.endswith(".py"):
                source = archive.read(name).decode("utf-8")
                assert not pattern.search(source), name
                assert "pygame" not in webapp.top_level_imports(source), name


def test_forbidden_import_scan_flags_offenders(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    import matplotlib.pyplot as plt\n", encoding="utf-8")
    assert webapp.forbidden_imports([("rescue/bad.py", bad)]) == {"rescue/bad.py": {"matplotlib"}}


def test_zip_is_reproducible(tmp_path: Path) -> None:
    members = webapp.zip_members()
    first = webapp.write_zip(tmp_path / "a.zip", members, extra={"x.json": b"{}"})
    second = webapp.write_zip(tmp_path / "b.zip", list(reversed(members)),
                              extra={"x.json": b"{}"})
    assert first.read_bytes() == second.read_bytes()


#: Fresh-interpreter check: block the heavy modules, import the unpacked package, replay the
#: deck_demo mission through WebSession and print the trace as JSON on the last line.
ZIP_IMPORT_SCRIPT = textwrap.dedent('''
    import importlib, json, sys
    BLOCKED = set(sys.argv[2].split(","))

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"{name} is blocked in the browser bundle test")
            return None

    sys.meta_path.insert(0, Blocker())
    sys.path.insert(0, sys.argv[1])
    import rescue
    assert rescue.__file__.startswith(sys.argv[1]), rescue.__file__
    for module in sys.argv[3].split(","):
        importlib.import_module(module)
    webapi = importlib.import_module("rescue.webapi")
    session = webapi.WebSession({"mode": "core", "agent": "andor",
                                 "map": {"source": "file", "name": "deck_demo"}})
    for _ in range(int(sys.argv[4])):
        if session.done:
            break
        session.step()
    state = session.state()
    json.dumps(state)
    out = dict(status=state["status"], t=state["t"], trace=state["trace"],
               metrics=state["metrics"], robot=state["robot"])
    # A generated scenario map comes from maps_library.json (scipy stays blocked).
    generated = webapi.WebSession({"mode": "core", "agent": "andor", "map": {
        "source": "scenario", "scenario": "random_rubble", "rows": 15, "cols": 15,
        "seed": 3}}).state()
    out["library_map"] = [generated["rows"], generated["cols"]]
    out["loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    print(json.dumps(out))
''')


def _core_modules(names: list[str]) -> list[str]:
    """Dotted names of the packed modules that must import without scipy: all of them but
    :mod:`rescue.mapgen`, the only one importing scipy at module level (LAO* imports it while
    planning). ``rescue.webapi`` is imported and driven separately by the script."""
    needs_scipy = {"rescue.mapgen", "rescue.webapi"}
    modules = []
    for name in names:
        if name.startswith("rescue/") and name.endswith(".py"):
            dotted = name[:-3].replace("/", ".").removesuffix(".__init__")
            if dotted not in needs_scipy:
                modules.append(dotted)
    return modules


def test_zipped_package_runs_the_demo_without_heavy_dependencies(
    site: Path, zip_names: list[str], tmp_path: Path
) -> None:
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(site / webapp.ZIP_FILE) as archive:
        archive.extractall(unpacked)
    result = subprocess.run(
        [sys.executable, "-c", ZIP_IMPORT_SCRIPT, str(unpacked), ",".join(BLOCKED_MODULES),
         ",".join(_core_modules(zip_names)), str(MAX_SESSION_STEPS)],
        capture_output=True, text=True, encoding="utf-8", timeout=SUBPROCESS_TIMEOUT_S,
        cwd=unpacked, check=False,  # the worker chdirs into the unpacked zip too
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["loaded"] == []

    # The same mission through the CLI path (python -m rescue.demo).
    from rescue.agents import make_agent
    from rescue.grid import TrueGrid
    from rescue.runner import make_transition, run_mission

    grid = TrueGrid.load(REPO_ROOT / "maps" / "deck_demo.txt")
    world_config = WorldConfig(slip_p=DEFAULT_SLIP_P)
    agent = make_agent("andor", make_transition(grid, world_config), PlannerConfig(),
                       AgentConfig())
    mission = run_mission(grid, agent, world_config, seed=DEMO_SEED, scenario=grid.name)
    moves = [e for e in mission.events if e.action is not None]
    assert len(moves) == DEMO_STEPS

    assert out["library_map"] == [LIBRARY_MAP_SIZE, LIBRARY_MAP_SIZE]
    assert out["status"] == "rescued"
    assert out["t"] == DEMO_STEPS
    assert out["robot"] == list(grid.survivor)
    trace = out["trace"]
    assert [step["from"] for step in trace] == [list(e.pos) for e in moves]
    assert [step["to"] for step in trace] == [list(e.new_pos) for e in moves]
    assert [step["action"] for step in trace] == [e.action.name for e in moves]
    for step, event in zip(trace, moves, strict=True):
        assert event.outcome is not None
        assert str(step["outcome"]).lower() in {event.outcome.value.lower(),
                                                event.outcome.name.lower()}
    assert out["metrics"]["replans"] == mission.metrics().replans


# --- maps_library.json -------------------------------------------------------------------------
def test_published_library_is_the_generator_output(site: Path) -> None:
    """``maps_library.json`` (inside the zip) holds every generated scenario at sizes
    10, 15 and 20 (at least) and seeds 0-29, and its entries are mapgen's output."""
    from rescue.mapgen import GENERATED_SCENARIOS, generate
    from rescue.webmaps import MapLibrary, library_key

    with zipfile.ZipFile(site / webapp.ZIP_FILE) as archive:
        published = archive.read(webapp.LIBRARY_FILE)
    library = MapLibrary.from_document(json.loads(published))
    assert set(LIBRARY_CONTRACT_SIZES) <= set(library.sizes)
    assert library.seeds == tuple(range(LIBRARY_CONTRACT_SEEDS))
    assert library.scenarios == GENERATED_SCENARIOS
    for scenario in GENERATED_SCENARIOS:
        for n in library.sizes:
            for seed in library.seeds:
                assert library.has(scenario, n, n, seed)
        for n, seed in ((min(library.sizes), 0), (max(library.sizes), library.seeds[-1])):
            assert library.maps[library_key(scenario, n, n, seed)] == generate(
                scenario, n, n, seed).to_ascii()


# --- results.json ------------------------------------------------------------------------------
def _claim_rows(path: Path) -> dict[str, str]:
    """Claim id → verdict symbol, read straight from a report's claims table."""
    text = path.read_text(encoding="utf-8")
    section = text.split("## Claims", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\|\s*([A-Z]+\d+)\s*\|.*?\|\s*(✅|❌|⚠️?)\s*\|", section, re.MULTILINE)
    return dict(rows)


@pytest.mark.parametrize("report_id", ["main", "ext"])
def test_results_json_has_every_claim_with_its_verdict(site: Path, report_id: str) -> None:
    results = _read_json(site / webapp.RESULTS_FILE)
    report = next(r for r in results["reports"] if r["id"] == report_id)
    source = next(s for s in webapp.REPORTS if s.id == report_id)
    expected = _claim_rows(REPO_ROOT / source.path)
    assert len(expected) >= 6
    assert [c["id"] for c in report["claims"]] == list(expected)
    for claim in report["claims"]:
        assert claim["symbol"] == expected[claim["id"]]
        assert claim["verdict"] == webapp.verdict_word(expected[claim["id"]])
        assert claim["claim"] and claim["measured"]
        assert claim["details"], f"{claim['id']} has a details section in the report"
    assert sum(report["counts"].values()) == len(expected)
    assert report["meta"].get("Git commit")


def test_results_json_reports_the_lao_tables(site: Path) -> None:
    results = _read_json(site / webapp.RESULTS_FILE)
    report = next(r for r in results["reports"] if r["id"] == "lao")
    text = (REPO_ROOT / "results" / "lao" / "LAO.md").read_text(encoding="utf-8")
    quality = text.split("## 1.", 1)[1].split("\n## ", 1)[0]
    rows = [ln for ln in quality.splitlines() if ln.startswith("| `")]
    assert len(report["claims"]) == len(rows) >= 2
    for claim in report["claims"]:
        assert claim["verdict"] == webapp.DESCRIPTIVE_VERDICT
        assert "mean diff" in claim["measured"]
    headings = {t["title"] for t in report["tables"]}
    assert len(headings) == text.count("\n## ")
    assert "Descriptive only" in report["intro"]


def test_results_plots_are_copied_with_captions(site: Path) -> None:
    results = _read_json(site / webapp.RESULTS_FILE)
    assert {p["source"] for p in results["plots"]} == {p.path for p in webapp.PLOTS}
    assert {f["source"] for f in results["figures"]} == {f.path for f in webapp.FIGURES}
    for entry in (*results["plots"], *results["figures"]):
        assert (site / entry["src"]).read_bytes() == (REPO_ROOT / entry["source"]).read_bytes()
        assert entry["title"] and entry["caption"]
        assert len(entry["caption"]) <= webapp.MAX_CAPTION_CHARS + 1
    documented = [p for p in results["plots"] if p["doc"]]
    assert len(documented) == len(results["plots"]), "every report plot is captioned in a doc"


def test_split_row_keeps_pipes_in_code_and_escapes() -> None:
    assert webapp.split_row("| a | `x | y` | b \\| c |") == ["a", "`x | y`", "b | c"]
    assert webapp._is_separator(webapp.split_row("|---|:---:|"))


def test_claims_parser_rejects_unknown_verdicts() -> None:
    report = "# R\n\n## Claims\n\n| # | Claim | Verdict | Measured |\n|---|---|---|---|\n" \
             "| H1 | c | ?? | m |\n"
    with pytest.raises(ValueError, match="unknown verdict"):
        webapp.parse_claims_report(report)
    with pytest.raises(ValueError, match="Claims"):
        webapp.parse_claims_report("# R\n\nno claims here\n")


# --- version.json ------------------------------------------------------------------------------
def test_version_json(site: Path) -> None:
    version = _read_json(site / webapp.VERSION_FILE)
    from rescue import __version__

    assert version["version"] == __version__
    assert version["pyodide"] == WEB_PYODIDE_VERSION
    assert version["pyodide_url"] == f"https://cdn.jsdelivr.net/pyodide/v{WEB_PYODIDE_VERSION}/full/"
    assert version["three"] == WEB_THREE_VERSION
    assert version["three_url"] == f"https://cdn.jsdelivr.net/npm/three@{WEB_THREE_VERSION}/"
    assert re.fullmatch(r"[0-9a-f]{7,}|unknown", version["commit"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", version["built_at"])
    assert version["python"] and version["numpy"]
    assert version["zip"] == webapp.ZIP_FILE
    assert isinstance(version["api"], int)
    packages = version["build_packages"]
    assert packages["numpy"] and packages["scipy"] and packages["python"]


def test_build_time_honours_source_date_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert webapp.build_time() == "1970-01-01T00:00:00Z"
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    assert webapp.git_commit() == ("0123456", False)


def test_front_end_pins_the_same_pyodide_release() -> None:
    """Any Pyodide URL or ``…PYODIDE_VERSION = '…'`` fallback hard-coded in the front end
    names WEB_PYODIDE_VERSION."""
    pattern = re.compile(r"pyodide/v(\d+\.\d+\.\d+)|PYODIDE_VERSION\s*=\s*['\"]([\d.]+)['\"]")
    for path in webapp.WEBAPP_SOURCE_DIR.rglob("*"):
        if path.suffix in webapp.TEMPLATED_SUFFIXES and "tests" not in path.parts:
            for match in pattern.finditer(path.read_text("utf-8")):
                assert (match.group(1) or match.group(2)) == WEB_PYODIDE_VERSION, path


def test_front_end_pins_the_same_three_release(site: Path) -> None:
    """Every three.js URL in the built site names WEB_THREE_VERSION, and the import map the
    3D view needs survives the build."""
    pattern = re.compile(r"npm/three@([\d.]+)/")
    seen = 0
    for path in site.rglob("*"):
        if path.suffix in webapp.TEMPLATED_SUFFIXES:
            for match in pattern.finditer(path.read_text("utf-8")):
                assert match.group(1) == WEB_THREE_VERSION, path
                seen += 1
    assert seen >= 2, "the import map should name three and three/addons/"
    index = (site / "index.html").read_text("utf-8")
    assert '"three/addons/"' in index
    assert (site / "js" / "render3d.js").is_file()


def _workflow_steps(text: str) -> list[tuple[str, str]]:
    """``(job, run-or-uses)`` of every step of a workflow, in order (a small YAML reader for
    this file's layout: two-space indents, ``- name:`` steps, ``run: |`` blocks)."""
    steps: list[tuple[str, str]] = []
    job = ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^  [a-z][\w-]*:\s*$", line):
            job = line.strip()[:-1]
        match = re.match(r"^\s+(?:- )?(run|uses):\s*(.*)$", line)
        if match:
            body = match.group(2)
            if body.strip() == "|":
                indent = len(line) - len(line.lstrip()) + 2
                block = []
                for nxt in lines[i + 1:]:
                    if nxt.strip() and len(nxt) - len(nxt.lstrip()) < indent:
                        break
                    block.append(nxt.strip())
                body = "\n".join(block)
            steps.append((job, body.strip()))
    return steps


def test_pages_workflow_tests_before_it_builds_and_deploys() -> None:
    text = PAGES_WORKFLOW.read_text(encoding="utf-8")
    steps = _workflow_steps(text)
    commands = [body for _, body in steps]
    def index(needle: str) -> int:
        return next(i for i, body in enumerate(commands) if needle in body)
    test_step = index("pytest")
    assert "tests/test_webapi.py" in commands[test_step]
    assert index("ruff check") < index("python -m rescue.webapp --build-only --out _site")
    assert index("node --test webapp/tests") < index("python -m rescue.webapp --build-only")
    assert test_step < index("python -m rescue.webapp --build-only")
    assert index("actions/upload-pages-artifact") < index("actions/deploy-pages")
    jobs = {job for job, body in steps if "deploy-pages" in body}
    assert jobs == {"deploy"} and "needs: build" in text
    for needle in ("pages: write", "id-token: write", "python-version: \"3.11\"",
                   "branches: [main]"):
        assert needle in text, needle


# --- serving -----------------------------------------------------------------------------------
def _get(url: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "")


def test_server_answers_on_loopback(site: Path) -> None:
    with webapp.serving(site) as url:
        assert url.startswith("http://127.0.0.1:")
        status, body, _ = _get(url + webapp.INDEX_FILE)
        assert status == 200 and body == (site / webapp.INDEX_FILE).read_bytes()
        status, body, kind = _get(url + webapp.ZIP_FILE)
        assert status == 200 and body == (site / webapp.ZIP_FILE).read_bytes()
        assert kind == "application/zip"
        status, _, kind = _get(url + webapp.VERSION_FILE)
        assert status == 200 and kind.startswith("application/json")
        script = next(site.rglob("*.js"), None)
        if script is not None:
            _, _, kind = _get(url + script.relative_to(site).as_posix())
            assert kind.startswith("text/javascript")


def test_cli_serves_an_existing_build(site: Path) -> None:
    """``python -m rescue.webapp --skip-build --port 0 --no-open`` serves on 127.0.0.1."""
    process = subprocess.Popen(
        [sys.executable, "-m", "rescue.webapp", "--out", str(site), "--skip-build",
         "--port", "0", "--no-open"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        match = re.search(r"(http://127\.0\.0\.1:\d+/)", line)
        assert line.startswith(webapp.SERVING_PREFIX) and match, line
        for name in (webapp.INDEX_FILE, webapp.ZIP_FILE):
            status, _, _ = _get(match.group(1) + name)
            assert status == 200
    finally:
        process.terminate()
        process.wait(timeout=SERVER_START_TIMEOUT_S)


def test_cli_rejects_serving_a_missing_build(tmp_path: Path) -> None:
    assert webapp.main(["--skip-build", "--out", str(tmp_path / "none"), "--no-open"]) == 2


def test_cli_refuses_a_network_address_without_allow_remote(site: Path) -> None:
    assert webapp.is_loopback("127.0.0.1") and webapp.is_loopback("::1")
    assert webapp.is_loopback("localhost")
    assert not webapp.is_loopback("0.0.0.0") and not webapp.is_loopback("192.168.1.5")
    assert webapp.main(["--skip-build", "--out", str(site), "--host", "0.0.0.0",
                        "--no-open"]) == 2


# --- shipped front end -------------------------------------------------------------------------
#: Relative URLs in the shipped JavaScript: static / dynamic imports, ``new URL('…',
#: import.meta.url / self.location)`` and ``fetch('…')`` of a literal.
JS_URL_PATTERNS = (
    re.compile(r"""\bfrom\s+['"](\.{1,2}/[^'"]+)['"]"""),
    re.compile(r"""\bimport\(\s*['"](\.{1,2}/[^'"]+)['"]\s*\)"""),
    re.compile(r"""new URL\(\s*['"]([^'"]+)['"]\s*,\s*(?:import\.meta\.url|self\.location)"""),
    re.compile(r"""\bfetch\(\s*['"]([^'"]+)['"]"""),
)
#: Files the site references that only exist in development (``?mock=1``).
DEV_ONLY_URLS = frozenset({"./mock.js"})


def test_every_url_the_shipped_javascript_loads_exists(site: Path) -> None:
    checked = 0
    for script in site.rglob("*.js"):
        text = script.read_text(encoding="utf-8")
        for pattern in JS_URL_PATTERNS:
            for match in pattern.finditer(text):
                url = match.group(1)
                if url in DEV_ONLY_URLS or "://" in url:
                    continue
                base = script.parent
                if pattern.pattern.startswith(r"\bfetch") and not url.startswith("."):
                    base = site  # fetch('results.json') resolves against the page
                target = (base / url).resolve()
                assert target.is_file(), f"{script.relative_to(site)} loads {url}"
                checked += 1
    assert checked > 10


# --- front-end unit tests ----------------------------------------------------------------------
def test_front_end_node_tests() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    # The test files themselves (webapp/tests/index.js only re-imports them, for the folder
    # form ``node --test webapp/tests``).
    files = sorted(str(p) for p in NODE_TESTS_DIR.rglob("*.test.mjs")
                   if p.is_file()) if NODE_TESTS_DIR.is_dir() else []
    if not files:
        pytest.skip("no front-end tests in webapp/tests")
    result = subprocess.run([node, "--test", *files], capture_output=True, text=True,
                            encoding="utf-8", timeout=SUBPROCESS_TIMEOUT_S, cwd=REPO_ROOT,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
