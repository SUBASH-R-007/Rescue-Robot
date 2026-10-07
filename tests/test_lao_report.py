"""``rescue.experiments.lao_report``: the LAO* experiment's own analysis (plan §3.15.1)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rescue.experiments.lao_report import (
    LAO,
    LAO_REPORT_FILE,
    VI,
    expanded_per_plan,
    identical_share,
    lao_report,
    main,
    ms_per_plan,
    paired_ratios,
    write_lao_report,
)
from rescue.experiments.run import run_experiments
from rescue.experiments.spec import load_experiment_config
from rescue.experiments.stats import read_raw

#: A miniature experiments/lao.yaml: 2 scenarios x 2 sizes x 2 maps, in-process.
SMALL_CONFIG = """\
name: lao_small
base_seed: 2026
n_maps: 2
agents: [andor, vi_expected, lao_expected]
workers: 1
blocks:
  - name: lao_scaling
    scenarios: [random_rubble, maze_corridors]
    sizes: [8, 10]
    slip_p: [0.2]
"""
N_TASKS = 8
SIZES = (8, 10)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("lao_run")
    config = out / "lao_small.yaml"
    config.write_text(SMALL_CONFIG, encoding="utf-8")
    run_experiments(load_experiment_config(config), out, workers=1, progress=False,
                    config_path=config)
    return out


def test_report_sections_and_pairing(run_dir: Path) -> None:
    path = write_lao_report(run_dir)
    assert path == run_dir / LAO_REPORT_FILE
    text = path.read_text(encoding="utf-8")
    for heading in ("## 1. Plan quality", "## 2. Planning time per plan (ms)",
                    f"## 3. `{LAO}` states expanded per plan"):
        assert heading in text
    assert f"{N_TASKS} tasks, {3 * N_TASKS} missions, workers 1" in text
    assert "1 worker process(es)" in text  # the timing context is stated
    # One row per scenario x size plus one pooled row per size, in tables 2 and 3.
    pooled = [line for line in text.splitlines() if line.startswith("| all |")]
    assert len(pooled) == 2 * len(SIZES)
    rows = read_raw(run_dir / "raw.csv")
    same, total = identical_share(rows, LAO, VI)
    assert total == N_TASKS
    assert f"{same}/{total}" in text
    # Same optimum: the LAO*/VI paired step difference is reported as exactly 0 here.
    assert f"| `{LAO}` − `{VI}` | 0.000 [0.000, 0.000] |" in text
    ratios = paired_ratios(rows, LAO, VI, ms_per_plan)
    assert len(ratios) == N_TASKS and all(r > 0 for r in ratios)


def test_per_plan_quantities() -> None:
    row = {"planning_ms": 30.0, "nodes_expanded": 90, "plans_made": 3}
    assert ms_per_plan(row) == 10.0 and expanded_per_plan(row) == 30.0
    none = {"planning_ms": 0.0, "nodes_expanded": 0, "plans_made": 0}
    assert math.isnan(ms_per_plan(none)) and math.isnan(expanded_per_plan(none))


def test_missing_agent_is_an_error(run_dir: Path) -> None:
    rows = [r for r in read_raw(run_dir / "raw.csv") if r["agent"] != LAO]
    with pytest.raises(ValueError, match=LAO):
        lao_report(rows)


def test_cli(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(run_dir)]) == 0
    assert LAO_REPORT_FILE in capsys.readouterr().out
    assert main([str(tmp_path)]) == 2  # no raw.csv
