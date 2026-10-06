"""Experiment specifications: YAML configs, blocks and the expanded task list.

An :class:`ExperimentConfig` (one YAML file, e.g. ``experiments/main.yaml``) holds named
:class:`ExperimentBlock` s. Each block is a small factorial design (scenarios × sizes × slip p ×
sensor radius × replan mode × aftershock q × replicates); :func:`expand_tasks` turns the whole
config into a deterministic list of :class:`Task` s, one per map, each of which runs every agent.

Common random numbers: a task's seed depends **only** on ``(base_seed, scenario, rows, cols,
replicate)`` (:func:`task_seed`), so the same map and the same outcome / aftershock streams are
reused across slip p values, sensor radii, replan modes, aftershock q values, planner overrides
and all agents: every comparison between agents or between parameter values is paired, also
across blocks (the ``tiebreak_ablation`` block replays the ``slip_sweep`` missions with other
tie-breaks).

A block's optional ``planner`` mapping overrides :class:`rescue.config.PlannerConfig` fields for
every agent of the block; its canonical text (:func:`planner_label`, ``''`` for the defaults) is
the ``planner`` column of ``raw.csv``.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from rescue.config import MAPGEN_MIN_SIZE, MAX_SLIP_P, PlannerConfig, load_yaml

#: Repository root (``src/rescue/experiments/spec.py`` → three levels up from ``src``).
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
#: Config used by ``python -m rescue.experiments`` without ``--config``.
DEFAULT_CONFIG: Path = Path("experiments") / "main.yaml"
#: Config used by ``--quick``.
QUICK_CONFIG: Path = Path("experiments") / "quick.yaml"
#: Default output directory of a full run, and of a ``--quick`` run (so a quick run never
#: overwrites the committed full results).
DEFAULT_OUT_DIR: Path = Path("results")
QUICK_OUT_DIR: Path = Path("results") / "quick"
#: Name of the block holding the main slip-probability sweep (console summary, hypotheses).
SLIP_SWEEP_EXPERIMENT: str = "slip_sweep"
#: Name of the block that replays the slip sweep with swapped tie-breaks (H7, ablation plot).
TIEBREAK_ABLATION_EXPERIMENT: str = "tiebreak_ablation"
#: ``planner`` column text: ``key=value`` pairs sorted by key, joined by these separators.
PLANNER_ITEM_SEP: str = ";"
PLANNER_KV_SEP: str = "="
#: Replan modes an agent accepts (``rescue.config.AgentConfig``).
REPLAN_MODES: tuple[str, ...] = ("eager", "lazy")
#: Defaults of the optional block fields.
DEFAULT_SENSOR_RADII: tuple[int, ...] = (1,)
DEFAULT_REPLAN_MODES: tuple[str, ...] = ("eager",)
DEFAULT_AFTERSHOCK_Q: tuple[float, ...] = (0.0,)
DEFAULT_AFTERSHOCK_K: int = 2

_CONFIG_KEYS: frozenset[str] = frozenset(
    {"name", "base_seed", "n_maps", "agents", "blocks", "workers"}
)
_REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {"name", "base_seed", "n_maps", "agents", "blocks"}
)
_BLOCK_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "scenarios",
        "sizes",
        "slip_p",
        "sensor_radius",
        "replan_mode",
        "aftershock_q",
        "aftershock_k",
        "agents",
        "n_maps",
        "planner",
    }
)
_REQUIRED_BLOCK_KEYS: frozenset[str] = frozenset({"name", "scenarios", "sizes", "slip_p"})


@dataclass(frozen=True)
class ExperimentBlock:
    """One factorial block of an experiment (sizes are square: ``rows = cols = size``).

    ``agents`` / ``n_maps`` of ``None`` inherit the config's top-level values. ``planner``
    overrides :class:`~rescue.config.PlannerConfig` fields for every agent of the block (empty:
    the defaults); it does not change task seeds.
    """

    name: str
    scenarios: tuple[str, ...]
    sizes: tuple[int, ...]
    slip_p: tuple[float, ...]
    sensor_radius: tuple[int, ...] = DEFAULT_SENSOR_RADII
    replan_mode: tuple[str, ...] = DEFAULT_REPLAN_MODES
    aftershock_q: tuple[float, ...] = DEFAULT_AFTERSHOCK_Q
    aftershock_k: int = DEFAULT_AFTERSHOCK_K
    agents: tuple[str, ...] | None = None
    n_maps: int | None = None
    planner: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "planner", dict(sorted(dict(self.planner).items())))
        _validate_block(self)

    @property
    def planner_label(self) -> str:
        """Canonical text of :attr:`planner` (the ``planner`` column; ``''`` for defaults)."""
        return planner_label(self.planner)


@dataclass(frozen=True)
class ExperimentConfig:
    """A whole experiment: shared seed, map count and agents plus its blocks.

    ``workers`` of ``None`` lets the harness choose (``os.cpu_count() - 2``, at least 1).
    """

    name: str
    base_seed: int
    n_maps: int
    agents: tuple[str, ...]
    blocks: tuple[ExperimentBlock, ...]
    workers: int | None = None

    def __post_init__(self) -> None:
        _validate_config(self)

    def block_agents(self, block: ExperimentBlock) -> tuple[str, ...]:
        """Agents run in ``block`` (the block's own list, else the top-level one)."""
        return block.agents if block.agents is not None else self.agents

    def block_n_maps(self, block: ExperimentBlock) -> int:
        """Maps (replicates) per parameter combination of ``block``."""
        return block.n_maps if block.n_maps is not None else self.n_maps


@dataclass(frozen=True)
class Task:
    """One map of one parameter combination; :func:`rescue.experiments.run.run_task` runs
    every agent in ``agents`` on it."""

    experiment: str
    scenario: str
    rows: int
    cols: int
    slip_p: float
    sensor_radius: int
    replan_mode: str
    aftershock_q: float
    aftershock_k: int
    replicate: int
    seed: int
    agents: tuple[str, ...]
    #: Canonical text of the block's PlannerConfig overrides (``''`` = defaults).
    planner: str = ""
    #: The overrides themselves, as sorted ``(field, value)`` pairs (hashable, picklable).
    planner_overrides: tuple[tuple[str, Any], ...] = ()

    def planner_config(self) -> PlannerConfig:
        """``PlannerConfig(**overrides)`` of this task."""
        return PlannerConfig(**dict(self.planner_overrides))


# --- planner overrides -----------------------------------------------------------------------
def _format_override(value: Any) -> str:
    """YAML-like text of one override value (``true`` / ``false`` / ``null`` / str)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def planner_label(overrides: Mapping[str, Any]) -> str:
    """Canonical text of PlannerConfig overrides: ``'astar_tie_break=diagonal;tie_break=fixed'``.

    Keys sorted; ``''`` for no overrides. Used as the ``planner`` column of ``raw.csv``.
    """
    return PLANNER_ITEM_SEP.join(
        f"{key}{PLANNER_KV_SEP}{_format_override(overrides[key])}" for key in sorted(overrides)
    )


def parse_planner_label(label: str) -> dict[str, str]:
    """Inverse of :func:`planner_label` as text: ``{field: value text}`` (``{}`` for ``''``)."""
    out: dict[str, str] = {}
    text = (label or "").strip()
    if not text:
        return out
    for item in text.split(PLANNER_ITEM_SEP):
        key, sep, value = item.partition(PLANNER_KV_SEP)
        if not sep or not key.strip():
            raise ValueError(f"bad planner label item {item!r} in {label!r}")
        out[key.strip()] = value.strip()
    return out


def effective_planner_value(label: str, name: str) -> str:
    """Text of PlannerConfig field ``name`` under the overrides ``label`` (default if absent)."""
    overrides = parse_planner_label(label)
    if name in overrides:
        return overrides[name]
    return _format_override(getattr(PlannerConfig(), name))


# --- validation ------------------------------------------------------------------------------
def _known_scenarios() -> tuple[str, ...]:
    from rescue.mapgen import SCENARIOS

    return tuple(SCENARIOS)


def _known_agents() -> tuple[str, ...]:
    from rescue.agents import ALL_AGENT_NAMES

    return ALL_AGENT_NAMES


def _require_nonempty_unique(label: str, values: tuple[Any, ...]) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} has duplicates: {list(values)}")


def _check_agents(label: str, agents: tuple[str, ...]) -> None:
    _require_nonempty_unique(label, agents)
    known = _known_agents()
    unknown = [a for a in agents if a not in known]
    if unknown:
        raise ValueError(f"{label}: unknown agents {unknown}; choose from {', '.join(known)}")


def _check_count(label: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")


def _validate_block(block: ExperimentBlock) -> None:
    where = f"block {block.name!r}"
    if not block.name:
        raise ValueError("block name must not be empty")
    _require_nonempty_unique(f"{where} scenarios", block.scenarios)
    known = _known_scenarios()
    unknown = [s for s in block.scenarios if s not in known]
    if unknown:
        raise ValueError(f"{where}: unknown scenarios {unknown}; choose from {', '.join(known)}")
    _require_nonempty_unique(f"{where} sizes", block.sizes)
    for size in block.sizes:
        _check_count(f"{where} size", size, MAPGEN_MIN_SIZE)
    _require_nonempty_unique(f"{where} slip_p", block.slip_p)
    for p in block.slip_p:
        if not 0.0 <= p <= MAX_SLIP_P:
            raise ValueError(f"{where}: slip_p must be in [0, {MAX_SLIP_P}], got {p}")
    _require_nonempty_unique(f"{where} sensor_radius", block.sensor_radius)
    for r in block.sensor_radius:
        _check_count(f"{where} sensor_radius", r, 0)
    _require_nonempty_unique(f"{where} replan_mode", block.replan_mode)
    for mode in block.replan_mode:
        if mode not in REPLAN_MODES:
            raise ValueError(f"{where}: unknown replan_mode {mode!r}")
    _require_nonempty_unique(f"{where} aftershock_q", block.aftershock_q)
    for q in block.aftershock_q:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"{where}: aftershock_q must be in [0, 1], got {q}")
    _check_count(f"{where} aftershock_k", block.aftershock_k, 0)
    _check_planner(where, block.planner)
    if block.agents is not None:
        _check_agents(f"{where} agents", block.agents)
    if block.n_maps is not None:
        _check_count(f"{where} n_maps", block.n_maps, 1)


def _check_planner(where: str, planner: Mapping[str, Any]) -> None:
    """PlannerConfig overrides: known fields, scalar values, and a valid PlannerConfig."""
    allowed = {f.name for f in fields(PlannerConfig)}
    unknown = sorted(set(planner) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown planner keys {unknown}; choose from "
                         f"{', '.join(sorted(allowed))}")
    for key, value in planner.items():
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError(f"{where}: planner {key} must be a scalar, got {value!r}")
        if isinstance(value, str) and (PLANNER_ITEM_SEP in value or PLANNER_KV_SEP in value):
            raise ValueError(f"{where}: planner {key} value {value!r} contains "
                             f"{PLANNER_ITEM_SEP!r} or {PLANNER_KV_SEP!r}")
    try:
        PlannerConfig(**planner)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bad planner overrides {dict(planner)}: {exc}") from exc


def _validate_config(cfg: ExperimentConfig) -> None:
    if not cfg.name:
        raise ValueError("experiment name must not be empty")
    _check_count("base_seed", cfg.base_seed, 0)
    _check_count("n_maps", cfg.n_maps, 1)
    _check_agents("agents", cfg.agents)
    if not cfg.blocks:
        raise ValueError("an experiment needs at least one block")
    _require_nonempty_unique("block names", tuple(b.name for b in cfg.blocks))
    if cfg.workers is not None:
        _check_count("workers", cfg.workers, 1)


# --- YAML loading ----------------------------------------------------------------------------
def _as_tuple(value: Any, kind: type, label: str) -> tuple[Any, ...]:
    """A YAML scalar or list as a tuple of ``kind`` (ints accepted where floats are wanted)."""
    items = list(value) if isinstance(value, list | tuple) else [value]
    out: list[Any] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{label}: expected {kind.__name__}, got {item!r}")
        if kind is float and isinstance(item, int | float):
            out.append(float(item))
        elif isinstance(item, kind):
            out.append(item)
        else:
            raise ValueError(f"{label}: expected {kind.__name__}, got {item!r}")
    return tuple(out)


def _check_mapping_keys(label: str, data: Any, allowed: frozenset[str],
                        required: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a mapping, got {type(data).__name__}")
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{label}: unknown keys {sorted(unknown)}")
    missing = required - set(data)
    if missing:
        raise ValueError(f"{label}: missing keys {sorted(missing)}")
    return data


def block_from_dict(data: dict[str, Any]) -> ExperimentBlock:
    """Build an :class:`ExperimentBlock` from a parsed YAML mapping (unknown keys → ValueError)."""
    data = _check_mapping_keys("block", data, _BLOCK_KEYS, _REQUIRED_BLOCK_KEYS)
    name = str(data["name"])
    label = f"block {name!r}"
    kwargs: dict[str, Any] = {
        "name": name,
        "scenarios": _as_tuple(data["scenarios"], str, f"{label} scenarios"),
        "sizes": _as_tuple(data["sizes"], int, f"{label} sizes"),
        "slip_p": _as_tuple(data["slip_p"], float, f"{label} slip_p"),
    }
    if "sensor_radius" in data:
        kwargs["sensor_radius"] = _as_tuple(data["sensor_radius"], int, f"{label} sensor_radius")
    if "replan_mode" in data:
        kwargs["replan_mode"] = _as_tuple(data["replan_mode"], str, f"{label} replan_mode")
    if "aftershock_q" in data:
        kwargs["aftershock_q"] = _as_tuple(data["aftershock_q"], float, f"{label} aftershock_q")
    if "aftershock_k" in data:
        kwargs["aftershock_k"] = data["aftershock_k"]
    if data.get("agents") is not None:
        kwargs["agents"] = _as_tuple(data["agents"], str, f"{label} agents")
    if data.get("n_maps") is not None:
        kwargs["n_maps"] = data["n_maps"]
    if data.get("planner") is not None:
        if not isinstance(data["planner"], dict):
            raise ValueError(f"{label} planner must be a mapping of PlannerConfig fields")
        kwargs["planner"] = {str(k): v for k, v in data["planner"].items()}
    return ExperimentBlock(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a parsed YAML mapping (unknown keys → ValueError)."""
    data = _check_mapping_keys("experiment config", data, _CONFIG_KEYS, _REQUIRED_CONFIG_KEYS)
    blocks = data["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list of mappings")
    return ExperimentConfig(
        name=str(data["name"]),
        base_seed=data["base_seed"],
        n_maps=data["n_maps"],
        agents=_as_tuple(data["agents"], str, "agents"),
        blocks=tuple(block_from_dict(b) for b in blocks),
        workers=data.get("workers"),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file (unknown keys or bad values → ValueError)."""
    try:
        return config_from_dict(load_yaml(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def resolve_config_path(path: str | Path) -> Path:
    """``path`` as given if it exists, else relative to the repository root (if that exists)."""
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    fallback = REPO_ROOT / candidate
    return fallback if fallback.exists() else candidate


# --- task expansion --------------------------------------------------------------------------
def scenario_key(scenario: str) -> int:
    """Stable 32-bit key of a scenario name (``zlib.crc32``; ``hash()`` is salted per process)."""
    return zlib.crc32(scenario.encode("utf-8"))


def task_seed(base_seed: int, scenario: str, rows: int, cols: int, replicate: int) -> int:
    """Map and world seed of a task: depends only on these five values (common random numbers).

    The seed drives both ``mapgen.generate`` (its own ``SeedSequence([seed, KEY])`` stream, D18)
    and the ``World`` streams, so every slip p, sensor radius, replan mode, aftershock q,
    planner override and agent sees the same map and the same outcome uniforms.
    """
    entropy = [int(base_seed), scenario_key(scenario), int(rows), int(cols), int(replicate)]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def expand_tasks(cfg: ExperimentConfig) -> list[Task]:
    """Every task of ``cfg`` in a deterministic order.

    Order: blocks in config order, then scenario (block order), size, slip p, sensor radius,
    replan mode, aftershock q (each in the order listed) and replicate. ``raw.csv`` is sorted
    the same way (then by agent, in config order).
    """
    tasks: list[Task] = []
    for block in cfg.blocks:
        agents = cfg.block_agents(block)
        n_maps = cfg.block_n_maps(block)
        planner = block.planner_label
        overrides = tuple(block.planner.items())
        for scenario in block.scenarios:
            for size in block.sizes:
                seeds = [task_seed(cfg.base_seed, scenario, size, size, i) for i in range(n_maps)]
                for p in block.slip_p:
                    for radius in block.sensor_radius:
                        for mode in block.replan_mode:
                            for q in block.aftershock_q:
                                for replicate, seed in enumerate(seeds):
                                    tasks.append(
                                        Task(
                                            experiment=block.name,
                                            scenario=scenario,
                                            rows=size,
                                            cols=size,
                                            slip_p=p,
                                            sensor_radius=radius,
                                            replan_mode=mode,
                                            aftershock_q=q,
                                            aftershock_k=block.aftershock_k,
                                            replicate=replicate,
                                            seed=seed,
                                            agents=agents,
                                            planner=planner,
                                            planner_overrides=overrides,
                                        )
                                    )
    return tasks


def subsample_tasks(tasks: Sequence[Task], max_tasks: int | None) -> list[Task]:
    """At most ``max_tasks`` tasks, evenly spaced over ``tasks`` (order kept; all if None).

    Evenly spaced rather than the first ``max_tasks`` so that a sample covers every block,
    scenario, size and p (used for smoke runs and run-time estimates).
    """
    if max_tasks is None or max_tasks >= len(tasks):
        return list(tasks)
    if max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
    picks = np.unique(np.linspace(0, len(tasks) - 1, max_tasks).round().astype(int))
    return [tasks[int(i)] for i in picks]


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUT_DIR",
    "QUICK_CONFIG",
    "QUICK_OUT_DIR",
    "REPO_ROOT",
    "SLIP_SWEEP_EXPERIMENT",
    "TIEBREAK_ABLATION_EXPERIMENT",
    "ExperimentBlock",
    "ExperimentConfig",
    "Task",
    "block_from_dict",
    "config_from_dict",
    "effective_planner_value",
    "expand_tasks",
    "load_experiment_config",
    "parse_planner_label",
    "planner_label",
    "resolve_config_path",
    "scenario_key",
    "subsample_tasks",
    "task_seed",
]
