"""Phase 7 coverage pass: behaviour of error paths and CLI branches no other test runs in-process.

Each test pins a user-visible contract rather than touching lines:

* ``rescue.mapgen`` CLI in-process (the subprocess tests in ``test_mapgen.py`` are not measured):
  ``--param`` parsing, the reproducible ``--out`` header, LF endings, argparse-style errors.
* Experiment configs reject every malformed value with a message naming the field.
* ``python -m rescue.experiments``: ``--max-tasks 0`` is refused before anything runs; the plot
  step is wired to ``<out>/plots``; worker failures carry the failing task; ``git`` and CPU
  fall-backs.
* Stretch agents (``dstar_lite``, ``vi_expected``) end a walled-off mission as
  ``UNREACHABLE`` without bumping (the registry's three main agents are covered elsewhere).
* ``World.true_cell`` / ``World.rescued``; ``rescue.sim`` / ``rescue.replay`` turn bad input
  files into exit code 2 with a readable message (headless, SDL dummy drivers).
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rescue import mapgen
from rescue.agents import STRETCH_AGENT_NAMES, make_agent
from rescue.config import MAPGEN_MIN_SIZE, MAX_SLIP_P, WorldConfig
from rescue.experiments import run, spec
from rescue.experiments.spec import (
    ExperimentBlock,
    ExperimentConfig,
    block_from_dict,
    config_from_dict,
)
from rescue.grid import Cell, TrueGrid
from rescue.model import UniformSlip
from rescue.runner import MissionStatus, make_transition, run_mission
from rescue.world import World

MAIN_AGENTS: tuple[str, ...] = ("andor", "astar_fixed", "astar_replan")
#: Survivor sealed off from the start by a wall column.
WALLED_OFF = "S#P"
#: Survivor sealed off by debris the robot only discovers while exploring (sensor radius 1).
HIDDEN_SEAL = "S..\n.##\n.#P"
SMALL_SIZE = 6


def _cli_module() -> Any:
    import importlib

    return importlib.import_module("rescue.experiments.__main__")


# --- mapgen CLI (in-process) -------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [
    ("density=0.3", ("density", 0.3)),
    ("room_size=4", ("room_size", 4)),
    ("loop_fraction=1e-1", ("loop_fraction", 0.1)),
])
def test_parse_param_keeps_ints_and_floats_apart(text: str, expected: tuple[str, Any]) -> None:
    key, value = mapgen._parse_param(text)
    assert (key, value) == expected
    assert type(value) is type(expected[1])  # "4" stays an int, never 4.0


@pytest.mark.parametrize(("text", "message"), [
    ("density", "expected KEY=VALUE"),
    ("=0.3", "expected KEY=VALUE"),
    ("density=lots", "not a number"),
])
def test_parse_param_rejects_malformed_pairs(text: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        mapgen._parse_param(text)


def test_mapgen_main_out_writes_reproducible_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Parent directories are created; the stem matches the scenario (TrueGrid.load names
    # the grid after the file).
    out = tmp_path / "nested" / "random_rubble.txt"
    code = mapgen.main(["random_rubble", "--rows", "7", "--cols", "8", "--seed", "4",
                        "--param", "density=0.2", "--out", str(out)])
    assert code == 0
    expected = mapgen.generate("random_rubble", 7, 8, 4, density=0.2)
    printed = capsys.readouterr().out
    assert TrueGrid.from_ascii(printed, name="random_rubble") == expected
    raw = out.read_bytes()
    assert b"\r\n" not in raw  # LF on every OS
    text = raw.decode("utf-8")
    header = [line for line in text.splitlines() if line.startswith(";")]
    assert header == [
        "; random_rubble 7x8, seed 4, density=0.2",
        "; python -m rescue.mapgen random_rubble --rows 7 --cols 8 --seed 4"
        " --param density=0.2",
    ]
    assert TrueGrid.load(out) == expected
    assert text.endswith(printed)  # the file is the header plus exactly what was printed


def test_map_header_of_a_fixed_deck_map_names_its_file() -> None:
    assert mapgen.map_header("deck_demo", 0, 0, 0, {}) == (
        f"; deck_demo: fixed map {mapgen.SCENARIOS['deck_demo'].fixed_map}\n"
    )
    plain = mapgen.map_header("maze_corridors", 9, 9, 1, {})
    assert plain.splitlines()[0] == "; maze_corridors 9x9, seed 1"
    assert "--param" not in plain


@pytest.mark.parametrize(("argv", "message"), [
    (["random_rubble", "--param", "no_such_knob=1"], "unexpected keyword argument 'no_such_knob'"),
    (["random_rubble", "--param", "density=lots"], "not a number"),
    (["random_rubble", "--rows", str(MAPGEN_MIN_SIZE - 1)],
     f"at least {MAPGEN_MIN_SIZE}x{MAPGEN_MIN_SIZE} cells"),
    (["volcano"], "invalid choice"),
])
def test_mapgen_main_reports_bad_input_as_usage_error(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        mapgen.main(argv)
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert captured.out == ""  # no half-written map on stdout


# --- experiment config validation --------------------------------------------------------------
def _block(**overrides: Any) -> ExperimentBlock:
    values: dict[str, Any] = {
        "name": "b",
        "scenarios": ("random_rubble",),
        "sizes": (SMALL_SIZE,),
        "slip_p": (0.1,),
    }
    values.update(overrides)
    return ExperimentBlock(**values)


def _config(**overrides: Any) -> ExperimentConfig:
    values: dict[str, Any] = {
        "name": "cfg",
        "base_seed": 0,
        "n_maps": 1,
        "agents": MAIN_AGENTS,
        "blocks": (_block(),),
    }
    values.update(overrides)
    return ExperimentConfig(**values)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"name": ""}, "block name must not be empty"),
    ({"scenarios": ()}, "scenarios must not be empty"),
    ({"scenarios": ("random_rubble", "random_rubble")}, "scenarios has duplicates"),
    ({"scenarios": ("volcano",)}, "unknown scenarios"),
    ({"sizes": (MAPGEN_MIN_SIZE - 1,)}, f"size must be an int >= {MAPGEN_MIN_SIZE}"),
    ({"sizes": (True,)}, "size must be an int"),
    ({"slip_p": (MAX_SLIP_P + 0.1,)}, "slip_p must be in"),
    ({"slip_p": (-0.1,)}, "slip_p must be in"),
    ({"sensor_radius": (-1,)}, "sensor_radius must be an int >= 0"),
    ({"replan_mode": ("sometimes",)}, "unknown replan_mode"),
    ({"aftershock_q": (1.5,)}, r"aftershock_q must be in \[0, 1\]"),
    ({"aftershock_k": -1}, "aftershock_k must be an int >= 0"),
    ({"agents": ("teleporter",)}, "unknown agents"),
    ({"agents": ()}, "agents must not be empty"),
    ({"n_maps": 0}, "n_maps must be an int >= 1"),
    ({"planner": {"no_such_field": 1}}, "unknown planner keys"),
    ({"planner": {"tie_break": ["fixed"]}}, "must be a scalar"),
    ({"planner": {"tie_break": f"fixed{spec.PLANNER_ITEM_SEP}x"}}, "contains"),
    ({"planner": {"tie_break": f"a{spec.PLANNER_KV_SEP}b"}}, "contains"),
])
def test_block_rejects_bad_values_naming_the_field(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _block(**overrides)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"name": ""}, "experiment name must not be empty"),
    ({"base_seed": -1}, "base_seed must be an int >= 0"),
    ({"n_maps": 0}, "n_maps must be an int >= 1"),
    ({"agents": ()}, "agents must not be empty"),
    ({"agents": ("andor", "andor")}, "agents has duplicates"),
    ({"blocks": ()}, "at least one block"),
    ({"blocks": (_block(), _block())}, "block names has duplicates"),
    ({"workers": 0}, "workers must be an int >= 1"),
])
def test_config_rejects_bad_values_naming_the_field(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def _block_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "b", "scenarios": ["random_rubble"], "sizes": [SMALL_SIZE], "slip_p": [0.1],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(("data", "message"), [
    (["not", "a", "mapping"], "must be a mapping"),
    ({"name": "x", "base_seed": 0, "n_maps": 1, "agents": ["andor"]}, r"missing keys \['blocks'\]"),
    ({"name": "x", "base_seed": 0, "n_maps": 1, "agents": ["andor"], "blocks": {"b": {}}},
     "blocks must be a list"),
    ({"name": "x", "base_seed": 0, "n_maps": 1, "agents": [True], "blocks": [_block_dict()]},
     "agents: expected str"),
])
def test_config_from_dict_rejects_malformed_yaml(data: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict(data)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"sizes": [True]}, "sizes: expected int"),
    ({"slip_p": ["high"]}, "slip_p: expected float"),
    ({"scenarios": [3]}, "scenarios: expected str"),
    ({"sensor_radius": [1.5]}, "sensor_radius: expected int"),
])
def test_block_from_dict_type_checks_every_list(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        block_from_dict(_block_dict(**overrides))


def test_block_from_dict_accepts_scalars_and_int_probabilities() -> None:
    """A YAML scalar is a one-element axis, and ``slip_p: 0`` is the float 0.0."""
    block = block_from_dict(_block_dict(sizes=SMALL_SIZE, slip_p=0, aftershock_q=[0, 1],
                                        replan_mode="lazy", agents=None, n_maps=None))
    assert block.sizes == (SMALL_SIZE,)
    assert block.slip_p == (0.0,) and isinstance(block.slip_p[0], float)
    assert block.aftershock_q == (0.0, 1.0)
    assert block.replan_mode == ("lazy",)
    assert block.agents is None and block.n_maps is None  # inherit the config's values


def test_subsample_rejects_a_non_positive_budget() -> None:
    tasks = spec.expand_tasks(_config(blocks=(_block(slip_p=(0.0, 0.1)),)))
    assert spec.subsample_tasks(tasks, None) == tasks
    assert spec.subsample_tasks(tasks, len(tasks) + 5) == tasks
    with pytest.raises(ValueError, match="max_tasks must be >= 1"):
        spec.subsample_tasks(tasks, 0)


# --- experiments CLI and harness -----------------------------------------------------------------
def test_cli_refuses_max_tasks_zero_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _cli_module()

    def must_not_load(path: str | Path) -> ExperimentConfig:
        raise AssertionError("config loaded although the arguments were invalid")

    monkeypatch.setattr(cli, "load_experiment_config", must_not_load)
    assert cli.main(["--quick", "--max-tasks", "0", "--out", str(tmp_path)]) == 2
    assert "--max-tasks must be >= 1" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_with_plots_writes_them_under_out_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--no-plots`` the CLI hands every raw row to ``make_all_plots(<out>/plots)``
    and counts the figures in its final line (plotting itself is stubbed: it is tested in
    ``test_experiments.py``)."""
    cli = _cli_module()
    from rescue.experiments import plots

    tiny = _config(name="tiny_plots", workers=1,
                   blocks=(_block(name=spec.SLIP_SWEEP_EXPERIMENT, slip_p=(0.0,)),))
    monkeypatch.setattr(cli, "load_experiment_config", lambda path: tiny)
    calls: list[tuple[int, Path]] = []

    def fake_plots(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
        calls.append((len(rows), Path(out_dir)))
        return [Path(out_dir) / "a.png", Path(out_dir) / "b.png"]

    monkeypatch.setattr(plots, "make_all_plots", fake_plots)
    out = tmp_path / "out"
    assert cli.main(["--quick", "--quiet", "--workers", "1", "--max-tasks", "1",
                     "--out", str(out)]) == 0
    assert calls == [(len(MAIN_AGENTS), out / cli.PLOTS_DIR)]  # one task x three agents
    stdout = capsys.readouterr().out
    assert f"Wrote 5 files to {out.as_posix()}/" in stdout  # raw, summary, hypotheses + 2
    assert "Success rate, slip_sweep" in stdout


def test_console_summary_reads_textual_flags_and_marks_missing_cells() -> None:
    cli = _cli_module()
    rows = [
        {"experiment": spec.SLIP_SWEEP_EXPERIMENT, "agent": "andor", "slip_p": "0.1",
         "success": "True"},
        {"experiment": spec.SLIP_SWEEP_EXPERIMENT, "agent": "andor", "slip_p": "0.1",
         "success": "false"},
        {"experiment": spec.SLIP_SWEEP_EXPERIMENT, "agent": "astar_fixed", "slip_p": "0.3",
         "success": "yes"},
        {"experiment": spec.SLIP_SWEEP_EXPERIMENT, "agent": "dstar_lite", "slip_p": "0.1",
         "success": True},
    ]
    lines = cli.console_summary(rows, 2.0).splitlines()
    andor = next(line for line in lines if line.startswith("andor"))
    fixed = next(line for line in lines if line.startswith("astar_fixed"))
    assert andor.split()[1:] == ["50.0%", "n/a"]  # p = 0.1, p = 0.3
    assert fixed.split()[1:] == ["n/a", "100.0%"]
    assert not any(line.startswith("dstar_lite") for line in lines)  # main agents only
    assert lines[-1] == "4 runs, total wall time 2.0 s"


def test_console_summary_labels_runs_without_a_slip_sweep_as_all_runs() -> None:
    cli = _cli_module()
    rows = [{"experiment": "scaling", "agent": "andor", "slip_p": "0.2", "success": "True"}]
    lines = cli.console_summary(rows, 1.0).splitlines()
    assert lines[0].startswith("Success rate, all runs")
    with_sweep = [{**rows[0], "experiment": spec.SLIP_SWEEP_EXPERIMENT}]
    assert cli.console_summary(with_sweep, 1.0).startswith(
        f"Success rate, {spec.SLIP_SWEEP_EXPERIMENT}"
    )


def test_worker_failure_names_the_failing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    task = spec.expand_tasks(_config())[0]
    boom = ZeroDivisionError("planner exploded")

    def failing(t: spec.Task) -> list[dict[str, Any]]:
        raise boom

    monkeypatch.setattr(run, "run_task", failing)
    with pytest.raises(RuntimeError, match="task 7 failed") as excinfo:
        run._run_indexed((7, task))
    assert excinfo.value.__cause__ is boom
    assert task.scenario in str(excinfo.value)


@pytest.mark.parametrize(("cpus", "expected"), [(None, 1), (1, 1), (run.RESERVED_CPUS, 1),
                                                (run.RESERVED_CPUS + 6, 6)])
def test_default_workers_reserves_cpus_but_never_drops_below_one(
    monkeypatch: pytest.MonkeyPatch, cpus: int | None, expected: int
) -> None:
    monkeypatch.setattr(run.os, "cpu_count", lambda: cpus)
    assert run.default_workers() == expected


@pytest.mark.parametrize("error", [OSError("git not installed"),
                                   subprocess.CalledProcessError(128, ["git"])])
def test_git_commit_is_none_without_git(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(run.subprocess, "run", no_git)
    assert run.git_commit() is None


def test_raw_csv_writes_none_as_empty_and_rounds_only_timings() -> None:
    assert run._format_cell("failure_reason", None) == ""
    assert run._format_cell("steps", 1.23456789) == 1.23456789  # not a timing column
    timing = run.TIMING_COLUMNS[0]
    assert run._format_cell(timing, 1.23456789) == round(1.23456789, run.TIMING_DECIMALS)
    assert math.isnan(run._format_cell(timing, math.nan))


# --- stretch agents: unreachable survivor --------------------------------------------------------
@pytest.mark.parametrize("name", STRETCH_AGENT_NAMES)
def test_stretch_agent_reports_walled_off_survivor_as_unreachable(name: str) -> None:
    grid = TrueGrid.from_ascii(WALLED_OFF)
    world = WorldConfig(slip_p=0.2)
    mission = run_mission(grid, make_agent(name, make_transition(grid, world)), world, seed=0)
    assert mission.status is MissionStatus.UNREACHABLE
    assert str(mission.status) == "UNREACHABLE"
    metrics = mission.metrics()
    assert metrics.failure_reason == "unreachable"
    assert metrics.steps == 0 and not metrics.success
    assert math.isnan(metrics.wasted_moves)
    assert mission.events[-1].note.startswith("unreachable")


@pytest.mark.parametrize("name", STRETCH_AGENT_NAMES)
def test_stretch_agent_discovers_the_seal_without_bumping(name: str) -> None:
    grid = TrueGrid.from_ascii(HIDDEN_SEAL)
    world = WorldConfig(slip_p=0.2, sensor_radius=1)
    for seed in range(3):
        agent = make_agent(name, make_transition(grid, world))
        mission = run_mission(grid, agent, world, seed=seed)
        assert mission.status is MissionStatus.UNREACHABLE, (name, seed)
        assert mission.metrics().bumps == 0
        for p in ((1, 1), (1, 2), (2, 1)):
            assert agent.belief.get(p) == Cell.DEBRIS, (name, seed, p)


def test_dstar_lite_detail_and_search_handle() -> None:
    grid = TrueGrid.from_ascii(WALLED_OFF)
    world = WorldConfig(slip_p=0.0)
    agent = make_agent("dstar_lite", make_transition(grid, world))
    assert agent.dstar is None  # no search before the first plan
    mission = run_mission(grid, agent, world)
    assert agent.dstar is not None
    assert "D* Lite found no path from (1,1)" in mission.events[-1].note  # 1-indexed


def test_vi_expected_has_infinite_expected_steps_without_a_policy() -> None:
    grid = TrueGrid.from_ascii(WALLED_OFF)
    world = WorldConfig(slip_p=0.2)
    agent = make_agent("vi_expected", make_transition(grid, world))
    run_mission(grid, agent, world)
    assert agent.expected_steps == math.inf


# --- world ---------------------------------------------------------------------------------------
def test_world_true_cell_and_rescued() -> None:
    grid = TrueGrid.from_ascii("S#\n.P")
    world = World(grid, UniformSlip(0.0), seed=0)
    assert world.true_cell((0, 1)) == Cell.DEBRIS
    assert world.true_cell((1, 0)) == Cell.OPEN
    with pytest.raises(ValueError, match="off-grid"):
        world.true_cell((2, 0))
    with pytest.raises(ValueError, match="off-grid"):
        world.true_cell((0, -1))
    assert not world.rescued
    agent_world = WorldConfig(slip_p=0.0)
    mission = run_mission(grid, make_agent("andor", make_transition(grid, agent_world)),
                          agent_world)
    assert mission.world.rescued and mission.status is MissionStatus.RESCUED


# --- sim / replay CLIs: bad input files ----------------------------------------------------------
@pytest.mark.viz
@pytest.mark.parametrize(("argv", "message"), [
    (["--map", "definitely/missing_map.txt"], "map file not found"),
    (["--scenario", "volcano"], "unknown scenario"),
])
def test_sim_cli_turns_bad_maps_into_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("pygame")
    from rescue.sim import main

    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--headless", "--frames", "1"])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.viz
@pytest.mark.parametrize(("content", "detail"), [
    (None, "No such file"),
    ("{not json", "Expecting property name"),
    (json.dumps({"version": 999}), "unsupported run-file version 999"),
])
def test_replay_cli_turns_unreadable_run_files_into_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, detail: str
) -> None:
    pytest.importorskip("pygame")
    from rescue.replay import main

    path = tmp_path / "run.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main([str(path), "--headless", "--frames", "1"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot read run file {path}" in err
    assert detail in err
