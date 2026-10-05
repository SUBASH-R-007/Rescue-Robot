"""Tests for rescue.demo (docs/plan.md §3.10 "Demo", brief §7 / Phase 3 row of §9)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rescue import demo
from rescue.config import DEMO_SEED
from rescue.runs import load_run
from tests.conftest import REPO_ROOT

ROOT_PLAN = "E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}"
TRACE_PREFIX = "(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned]"
#: The full slide-9 output of ``python -m rescue.demo`` (DEMO_SEED = 76, config.py).
GOLDEN_TRACE = (
    "(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned] E ok → "
    "(2,3) S slip → (2,3) S ok → (3,3) [debris at (4,3), replanned] E slip → (3,3) E ok → "
    "(3,4) S ok → (4,4)"
)
GOLDEN_SUMMARY = "Cells explored: 12   Replans: 2   Slips handled: 3   Collisions: 0"


def _lines(out: str) -> list[str]:
    return [line.strip() for line in out.splitlines()]


def test_demo_reproduces_slide_9(capsys: pytest.CaptureFixture[str]) -> None:
    rc = demo.main([])
    assert rc == 0  # exit code 0 on rescue
    out = capsys.readouterr().out
    lines = _lines(out)
    assert any(ROOT_PLAN in line for line in lines)
    trace_lines = [line for line in lines if line.startswith("(1,1) ")]
    assert len(trace_lines) == 1
    trace = trace_lines[0]
    assert trace.startswith(TRACE_PREFIX)
    assert trace.endswith("(4,4)")
    assert "Collisions: 0" in out
    assert "Rescue successful" in out
    assert "Rescue failed" not in out


def test_demo_golden_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Pins the recorded seed's whole slide-9 shape (Replans: 2, Slips handled: 3)."""
    assert DEMO_SEED == 76
    assert demo.main([]) == 0
    lines = _lines(capsys.readouterr().out)
    assert GOLDEN_TRACE in lines
    assert GOLDEN_SUMMARY in lines
    assert lines[lines.index(GOLDEN_SUMMARY) + 1] == "Rescue successful"


def test_demo_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    demo.main([])
    first = capsys.readouterr().out
    demo.main([])
    assert capsys.readouterr().out == first


def _run_demo_process(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "rescue.demo"],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=120,
    )
    return proc.stdout.decode("utf-8")


def test_demo_is_deterministic_across_processes() -> None:
    """Fresh interpreters with different hash seeds print the same output (no hidden state,
    no dependence on set/dict hash order)."""
    first = _run_demo_process("1")
    second = _run_demo_process("2024")
    assert first == second
    assert GOLDEN_TRACE in _lines(first)


def test_demo_astar_fixed_fails_with_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    rc = demo.main(["--agent", "astar_fixed"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Rescue failed (aborted)" in out
    plan_line = next(line for line in _lines(out) if line.startswith("Initial plan from (1,1): "))
    path = plan_line.removeprefix("Initial plan from (1,1): ").split(" → ")
    assert path[0] == "(1,1)" and path[-1] == "(4,4)"
    assert len(path) == 7  # optimistic shortest path


def test_demo_save_writes_loadable_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "r.json"
    assert demo.main(["--save", str(target)]) == 0
    out = capsys.readouterr().out
    assert f"Run saved to {target}" in out
    rec = load_run(target)
    assert rec.agent == "andor"
    assert rec.seed == DEMO_SEED
    assert rec.metrics.success


def test_demo_runs_from_other_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default relative map path falls back to the repository root."""
    monkeypatch.chdir(tmp_path)
    assert not Path(demo.DEFAULT_DEMO_MAP).exists()
    assert demo.resolve_map_path(demo.DEFAULT_DEMO_MAP) == REPO_ROOT / demo.DEFAULT_DEMO_MAP
    assert demo.main([]) == 0
    assert GOLDEN_TRACE in _lines(capsys.readouterr().out)
