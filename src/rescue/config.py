"""Constants and configuration dataclasses.

Every tunable number used by the simulator, planners, agents and experiments lives here so
that logic modules never contain magic numbers. Configs are frozen dataclasses and can be
built from plain dicts (e.g. parsed YAML) with the ``*_from_dict`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

# --- Transition model -------------------------------------------------------------------
#: Fraction of the slip mass ``p`` that keeps the robot in place ("slipped").
SLIP_STAY_FRACTION: float = 0.75
#: Fraction of ``p`` that moves the robot sideways, split equally left/right ("skid").
SKID_FRACTION: float = 0.25
#: Default slip parameter; gives the deck's 0.80 / 0.15 / 0.05 split.
DEFAULT_SLIP_P: float = 0.2
#: Largest slip parameter the model accepts.
MAX_SLIP_P: float = 0.4
#: Slip parameter on high-slip terrain ('~' cells: water, mud).
DEFAULT_TERRAIN_HIGH_P: float = 0.4
#: Probabilities at or below this are treated as impossible outcomes.
PROB_EPS: float = 1e-12

# --- Sensing and mission ----------------------------------------------------------------
#: Manhattan radius of the robot's sensor (1 = 4-neighbours).
DEFAULT_SENSOR_RADIUS: int = 1
#: Step budget = STEP_BUDGET_FACTOR * rows * cols.
STEP_BUDGET_FACTOR: int = 4

# --- Map generation ---------------------------------------------------------------------
#: Regeneration attempts before a generator gives up on producing a solvable map.
MAPGEN_MAX_ATTEMPTS: int = 200
#: Default i.i.d. debris density for ``random_rubble``.
DEFAULT_RUBBLE_DENSITY: float = 0.25
#: Allowed debris densities for ``random_rubble``.
RUBBLE_DENSITY_RANGE: tuple[float, float] = (0.15, 0.35)

ReplanMode = Literal["eager", "lazy"]
TieBreak = Literal["diagonal", "fixed"]


@dataclass(frozen=True)
class AftershockConfig:
    """Stress simulation: with probability ``q`` per step, ``k`` debris cells appear."""

    q: float = 0.0
    k: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.q <= 1.0:
            raise ValueError(f"aftershock q must be in [0, 1], got {self.q}")
        if self.k < 0:
            raise ValueError(f"aftershock k must be >= 0, got {self.k}")

    @property
    def enabled(self) -> bool:
        """True when aftershocks can actually happen."""
        return self.q > 0.0 and self.k > 0


@dataclass(frozen=True)
class WorldConfig:
    """Parameters of the true world / simulator."""

    slip_p: float = DEFAULT_SLIP_P
    sensor_radius: int = DEFAULT_SENSOR_RADIUS
    terrain_high_p: float = DEFAULT_TERRAIN_HIGH_P
    aftershock: AftershockConfig = field(default_factory=AftershockConfig)
    budget_factor: int = STEP_BUDGET_FACTOR

    def __post_init__(self) -> None:
        for name in ("slip_p", "terrain_high_p"):
            value = getattr(self, name)
            if not 0.0 <= value <= MAX_SLIP_P:
                raise ValueError(f"{name} must be in [0, {MAX_SLIP_P}], got {value}")
        if self.sensor_radius < 0:
            raise ValueError(f"sensor_radius must be >= 0, got {self.sensor_radius}")
        if self.budget_factor < 1:
            raise ValueError(f"budget_factor must be >= 1, got {self.budget_factor}")

    def step_budget(self, rows: int, cols: int) -> int:
        """Maximum number of steps for a mission on a ``rows`` x ``cols`` grid."""
        return self.budget_factor * rows * cols


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters of the AND-OR planner."""

    #: Maximum OR-depth; ``None`` means rows * cols (no effective cut-off).
    depth_limit: int | None = None
    #: Secondary ordering key for equally distant actions (see docs/plan.md D13).
    tie_break: TieBreak = "diagonal"
    #: Validate every plan with ``is_strong_cyclic`` (debug mode and tests).
    validate: bool = False
    #: Fall back to the fixpoint reference planner if AND-OR fails or is invalid.
    fallback_to_reference: bool = True

    def __post_init__(self) -> None:
        if self.depth_limit is not None and self.depth_limit < 0:
            raise ValueError(f"depth_limit must be >= 0 or None, got {self.depth_limit}")
        if self.tie_break not in ("diagonal", "fixed"):
            raise ValueError(f"unknown tie_break {self.tie_break!r}")


@dataclass(frozen=True)
class AgentConfig:
    """Parameters shared by the agents."""

    #: ``eager`` replans on any contradiction; ``lazy`` only if it touches the plan envelope.
    replan_mode: ReplanMode = "eager"

    def __post_init__(self) -> None:
        if self.replan_mode not in ("eager", "lazy"):
            raise ValueError(f"unknown replan_mode {self.replan_mode!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (an empty file gives ``{}``)."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def _check_keys(cls: type, data: dict[str, Any]) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")


def aftershock_config_from_dict(data: dict[str, Any] | None) -> AftershockConfig:
    """Build an :class:`AftershockConfig` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    _check_keys(AftershockConfig, data)
    return AftershockConfig(**data)


def world_config_from_dict(data: dict[str, Any] | None) -> WorldConfig:
    """Build a :class:`WorldConfig` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    _check_keys(WorldConfig, data)
    if "aftershock" in data and not isinstance(data["aftershock"], AftershockConfig):
        data["aftershock"] = aftershock_config_from_dict(data["aftershock"])
    return WorldConfig(**data)


def planner_config_from_dict(data: dict[str, Any] | None) -> PlannerConfig:
    """Build a :class:`PlannerConfig` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    _check_keys(PlannerConfig, data)
    return PlannerConfig(**data)


def agent_config_from_dict(data: dict[str, Any] | None) -> AgentConfig:
    """Build an :class:`AgentConfig` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    _check_keys(AgentConfig, data)
    return AgentConfig(**data)
