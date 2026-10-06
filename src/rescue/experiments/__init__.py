"""Monte Carlo experiment harness (brief §6.6, Phase 6).

``python -m rescue.experiments --config experiments/main.yaml`` (or ``--quick``) runs every
agent on seeded maps with common random numbers and writes ``raw.csv``, ``summary.csv``,
``HYPOTHESES.md`` and the plots. Modules: :mod:`.spec` (YAML configs → tasks), :mod:`.run`
(parallel runner → ``raw.csv``), :mod:`.stats` (CIs, paired tests, summary), :mod:`.hypotheses`
(the deck's claims, ✅/❌/⚠️) and :mod:`.plots` (matplotlib figures). Only the light modules are
re-exported here; ``stats``, ``hypotheses`` and ``plots`` are imported on demand.
"""

from __future__ import annotations

from rescue.experiments.run import RAW_COLUMNS, run_experiments, run_task
from rescue.experiments.spec import (
    ExperimentBlock,
    ExperimentConfig,
    Task,
    expand_tasks,
    load_experiment_config,
    task_seed,
)

__all__ = [
    "RAW_COLUMNS",
    "ExperimentBlock",
    "ExperimentConfig",
    "Task",
    "expand_tasks",
    "load_experiment_config",
    "run_experiments",
    "run_task",
    "task_seed",
]
