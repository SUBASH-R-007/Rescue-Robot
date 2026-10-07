"""Constants and configuration dataclasses.

Every tunable number used by the simulator, planners, agents and experiments lives here so
that logic modules never contain magic numbers. Configs are frozen dataclasses and can be
built from plain dicts (e.g. parsed YAML) with the ``*_from_dict`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Integral
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
#: Step budget = STEP_BUDGET_FACTOR * rows * cols. Strong-cyclic plans succeed with probability
#: 1 only without a budget: a mission that keeps slipping can run out of it (worst case about
#: ``p ** budget``, e.g. ≈ 6.6e-4 on a 1x2 corridor at p = 0.4), see docs/plan.md D15.
STEP_BUDGET_FACTOR: int = 4

# --- Demo -------------------------------------------------------------------------------
#: World seed of ``python -m rescue.demo`` (andor, ``maps/deck_demo.txt``, p = 0.2, r = 1,
#: eager). Found by searching seeds 0..20000 for missions whose trace starts ``(1,1) E ok →
#: (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned]`` and ends in a rescue
#: (docs/plan.md §3.10): 1,948 match. Among them, the smallest seed with the deck-like summary
#: ``Replans: 2   Slips handled: 3   Collisions: 0`` (the deck's maximum of 3 handled slips),
#: fewest cells explored, then fewest steps. It explores 12 cells in 9 steps. The deck's
#: "Cells explored: 11" is not reachable: no matching seed explores fewer than 12.
DEMO_SEED: int = 76

# --- Map generation ---------------------------------------------------------------------
#: Regeneration attempts before a generator gives up on producing a solvable map.
MAPGEN_MAX_ATTEMPTS: int = 200
#: Default i.i.d. debris density for ``random_rubble``.
DEFAULT_RUBBLE_DENSITY: float = 0.25
#: Allowed debris densities for ``random_rubble``.
RUBBLE_DENSITY_RANGE: tuple[float, float] = (0.15, 0.35)

ReplanMode = Literal["eager", "lazy"]
TieBreak = Literal["diagonal", "fixed"]
AStarTieBreak = Literal["g", "diagonal"]
#: Accepted values of ``PlannerConfig.tie_break`` and ``PlannerConfig.astar_tie_break``.
TIE_BREAKS: tuple[str, ...] = ("diagonal", "fixed")
ASTAR_TIE_BREAKS: tuple[str, ...] = ("g", "diagonal")


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
    """Parameters of the AND-OR planner (and the A* baselines' tie-break)."""

    #: Maximum OR-depth; ``None`` means rows * cols (no effective cut-off).
    depth_limit: int | None = None
    #: Secondary ordering key for equally distant actions (see docs/plan.md D13).
    tie_break: TieBreak = "diagonal"
    #: Validate every plan with ``is_strong_cyclic`` (debug mode and tests).
    validate: bool = False
    #: Fall back to the fixpoint reference planner if AND-OR fails or is invalid.
    fallback_to_reference: bool = True
    #: f-tie-break of the A* agents (``astar_fixed``, ``astar_replan``): ``"g"`` = larger g,
    #: then FIFO; ``"diagonal"`` = larger g, then the D13 ``diag`` key (docs/plan.md D13, §3.7).
    #: D* Lite ignores it.
    astar_tie_break: AStarTieBreak = "g"

    def __post_init__(self) -> None:
        # Types are checked too: a quoted YAML override such as ``validate: "false"`` would
        # otherwise be a truthy string while the run's ``planner`` label said ``false``.
        if self.depth_limit is not None and (
            isinstance(self.depth_limit, bool) or not isinstance(self.depth_limit, Integral)
        ):
            raise ValueError(f"depth_limit must be an int or None, got {self.depth_limit!r}")
        for name in ("validate", "fallback_to_reference"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.depth_limit is not None and self.depth_limit < 0:
            raise ValueError(f"depth_limit must be >= 0 or None, got {self.depth_limit}")
        if not isinstance(self.tie_break, str) or self.tie_break not in TIE_BREAKS:
            raise ValueError(f"unknown tie_break {self.tie_break!r}")
        if (not isinstance(self.astar_tie_break, str)
                or self.astar_tie_break not in ASTAR_TIE_BREAKS):
            raise ValueError(f"unknown astar_tie_break {self.astar_tie_break!r}")


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


# --- Stretch planners (Phase 5) -----------------------------------------------------------
#: Value iteration stops once the largest change of any finite value is below this.
VI_TOLERANCE: float = 1e-10
#: Value iteration gives up (with a warning) after this many sweeps.
VI_MAX_ITERS: int = 100_000
#: Q-values within this of the minimum count as tied when extracting the greedy policy (the
#: first tied action in ``ACTIONS`` order wins). Well above the residual error left by
#: ``VI_TOLERANCE`` and well below any genuine difference between two actions.
VI_TIE_TOLERANCE: float = 1e-7

# --- Scenario library (rescue.mapgen, Phase 5) ---------------------------------------------
#: Second entropy word of the map generators' seed sequence: ``SeedSequence([seed, KEY])``
#: gives a stream independent of the World's ``SeedSequence(seed).spawn(3)`` streams.
MAPGEN_STREAM_KEY: int = 0x6D61_7067  # "mapg"
#: Smallest rows / cols a generated scenario accepts.
MAPGEN_MIN_SIZE: int = 5
#: Start and survivor are at least this fraction of ``rows + cols - 2`` apart (Manhattan).
MAPGEN_MIN_SEPARATION_FRACTION: float = 0.5
#: After this fraction of ``MAPGEN_MAX_ATTEMPTS`` failed attempts, the required separation is
#: relaxed linearly down to 1 (only ever reached on tiny or very cluttered maps).
MAPGEN_RELAX_START_FRACTION: float = 0.5
#: collapsed_building: target interior size of a room (cells per side).
BUILDING_ROOM_SIZE: int = 4
#: collapsed_building: smallest room interior; also the smallest accepted ``room_size``.
BUILDING_MIN_ROOM_SIZE: int = 2
#: collapsed_building: each room side is randomly lengthened / shortened by up to this much.
BUILDING_ROOM_JITTER: int = 1
#: collapsed_building: fraction of doorways (interior and entrances) blocked by debris.
BUILDING_COLLAPSE_FRACTION: float = 0.3
#: collapsed_building: debris density inside rooms (door approaches are kept clear).
BUILDING_RUBBLE_DENSITY: float = 0.08
#: collapsed_building: fraction of interior wall segments that have fallen (rooms merge).
BUILDING_MERGE_FRACTION: float = 0.1
#: collapsed_building: width of the open street around the building.
BUILDING_STREET_WIDTH: int = 1
#: collapsed_building: probability that an outer wall segment of a room has an entrance.
BUILDING_ENTRANCE_PROB: float = 0.15
#: collapsed_building: at least this many entrances (before collapses).
BUILDING_MIN_ENTRANCES: int = 2
#: flooded_zone: target fraction of the map covered by high-slip water / mud ('~').
FLOOD_WATER_FRACTION: float = 0.25
#: flooded_zone: debris density on dry land.
FLOOD_DEBRIS_DENSITY: float = 0.06
#: flooded_zone: Gaussian smoothing of the noise field, as a fraction of sqrt(rows * cols)
#: (bigger = fewer, larger pools).
FLOOD_BLOB_SCALE: float = 0.12
#: flooded_zone: smallest smoothing sigma in cells.
FLOOD_MIN_SIGMA: float = 1.2
#: flooded_zone: the noise is smoothed on a grid padded by this many sigmas per side and then
#: cropped, so the smoothed field itself is stationary (no wrap-around pools from the opposite
#: border). Water is the top ``water_fraction`` of each map's own field, and that per-map
#: rank threshold leaves a small border excess: about 8 % relative at 15x15 (P(water) 0.257 on
#: the outer ring vs 0.238 in the centre, 3000 fields), against a flat 0.25 for one global
#: threshold. Kept for the exact per-map water fraction.
FLOOD_PAD_SIGMAS: float = 2.0
#: flooded_zone: accepted ``water_fraction`` and ``debris_density`` ranges.
FLOOD_WATER_RANGE: tuple[float, float] = (0.0, 0.6)
FLOOD_DEBRIS_RANGE: tuple[float, float] = (0.0, 0.3)
#: maze_corridors: fraction of the remaining inner walls knocked out to add loops.
MAZE_LOOP_FRACTION: float = 0.08
#: Default map size (rows, cols) and seed of the ``python -m rescue.mapgen`` CLI.
MAPGEN_DEFAULT_SIZE: tuple[int, int] = (15, 15)
MAPGEN_DEFAULT_SEED: int = 0

# --- Head-to-head comparison (rescue.compare / viz/compare_app.py, Phase 5) ---------------
#: Scenario, world seed and generated map size (rows, cols) of ``python -m rescue.compare``.
COMPARE_DEFAULT_SCENARIO: str = "collapsed_building"
COMPARE_DEFAULT_SEED: int = 7
COMPARE_DEFAULT_SIZE: tuple[int, int] = (15, 15)
#: Number of agents (panes) the comparison accepts.
COMPARE_MIN_AGENTS: int = 2
COMPARE_MAX_AGENTS: int = 5
#: Initial index into ``viz.theme.SPEEDS`` (8 lock-step ticks per second).
COMPARE_DEFAULT_SPEED_INDEX: int = 4
#: Gap between panes and inner padding of a pane, in pixels.
COMPARE_PANE_GAP: int = 12
COMPARE_PANE_PADDING: int = 8
#: Columns of the per-pane metrics block.
COMPARE_METRIC_COLUMNS: int = 2
#: Lines the per-pane "last event" text may wrap over.
COMPARE_EVENT_LINES: int = 2
#: Agent colours missing from the viz palettes (which know the three main agents), keyed by
#: palette name: the stretch agents can be compared too (``--agents``).
COMPARE_AGENT_COLOURS: dict[str, dict[str, tuple[int, int, int]]] = {
    "default": {"dstar_lite": (236, 112, 160), "vi_expected": (110, 184, 250)},
    "high-contrast": {"dstar_lite": (255, 140, 255), "vi_expected": (255, 170, 60)},
}

# --- Phase 5 fix pass ------------------------------------------------------------------------
#: collapsed_building: width of an interior hallway (a corridor band between two room rows or
#: columns, with a doorway from every room along it and an entrance at both ends).
BUILDING_HALL_WIDTH: int = 1
#: collapsed_building: probability of a hallway along each axis that has room for one (at
#: least two rooms of ``BUILDING_MIN_ROOM_SIZE`` beside it); if no axis drew one, a random
#: fitting axis still gets one, so a building with room for a hallway always has one.
BUILDING_HALL_PROB: float = 0.5
#: flooded_zone: start candidates tried per attempt when looking for a start-survivor pair
#: with water on every shortest route (each try costs two BFS passes).
FLOOD_CROSSING_START_TRIES: int = 12
#: Compare panes: most rows of panes the layout may use; it picks the row count giving the
#: largest map cells (ties keep fewer rows), so wide maps with 4-5 agents get two rows.
COMPARE_MAX_PANE_ROWS: int = 2

# --- Experiments: tie-break ablation (H7, docs/plan.md D21) ----------------------------------
#: Equivalence margin (wasted moves per mission) of H7's ❌ rule. Not pre-registered: chosen
#: while the ablation was tried on quick runs, before the final full run (docs/plan.md D21).
#: H7 is refuted when either matched 95 % CI lies above 0, or when both
#: lie entirely inside [-margin, +margin] (equivalence, TOST-style: "no advantage once
#: tie-breaking is matched"). Half a move: below the smallest per-mission difference (one move)
#: and about a sixth of H3's full-run gap. A CI that contains 0 but is wider than the margin is
#: ⚠️ inconclusive, as for every other paired claim.
EQUIVALENCE_MARGIN_MOVES: float = 0.5

# --- Extensions (rescue.ext, docs/plan.md §3.14, brief §8.3–§8.5) ----------------------------
#: Second entropy word of the noisy sensor's stream: ``SeedSequence([seed, KEY])`` is
#: independent of the World's spawned streams and of the map generators' stream.
NOISE_STREAM_KEY: int = 0x6E6F_6973  # "nois"
#: Second entropy word of the extra-survivor placement stream of ``multi_survivor_grid``.
EXT_SCENARIO_STREAM_KEY: int = 0x6D75_6C74  # "mult"
#: Prior probability that a never-observed cell is debris (``ProbBelief``). It is the default
#: ``random_rubble`` density (``DEFAULT_RUBBLE_DENSITY``): the one scenario whose debris is
#: i.i.d., so the prior matches the base rate the robot would expect on unseen rubble. Being
#: below ``EXT_DEBRIS_THRESHOLD`` it keeps unknown cells passable (optimistic, like the core
#: belief's "UNKNOWN assumed open"), so with ``false_negative = 0`` a ``ProbBelief`` makes the
#: same passability decisions as a ``BeliefMap``.
EXT_PRIOR_DEBRIS: float = DEFAULT_RUBBLE_DENSITY
#: A cell is treated as debris (impassable) once its debris probability reaches this value.
EXT_DEBRIS_THRESHOLD: float = 0.5
#: Smallest pairwise Manhattan distance between survivors placed by ``multi_survivor_grid``
#: (relaxed one step at a time, down to 1, only when no free cell satisfies it).
EXT_MIN_SURVIVOR_SPACING: int = 3
#: The optimal tour (best visiting order) is brute-forced over all k! orders up to this k;
#: above it ``ExtMetrics.optimal_tour`` is NaN (6! = 720 orders).
EXT_MAX_BRUTE_FORCE_K: int = 6
#: Defaults of ``python -m rescue.ext.demo`` (map scenario, size, world / map seed).
EXT_DEMO_SCENARIO: str = "collapsed_building"
EXT_DEMO_SIZE: tuple[int, int] = (12, 12)
EXT_DEMO_SEED: int = 7
#: Survivors of the demo's ``multi`` mode, and false-negative rate of its ``noisy`` mode.
EXT_DEMO_MULTI_K: int = 3
EXT_DEMO_NOISE: float = 0.1
#: Longest execution trace the demo prints (segments, final position included).
EXT_DEMO_TRACE_SEGMENTS: int = 40
#: Claims of ``python -m rescue.ext.experiments`` (``results/ext/EXTENSIONS.md``), fixed before
#: the full run and never tuned toward a verdict.
#: X1 / X4: success-rate target (unknown-location search; noisy sensing at ``EXT_CLAIM_NOISE``).
EXT_CLAIM_SUCCESS_TARGET: float = 0.95
#: X3: the greedy nearest-first tour may be at most this many times the optimal tour.
EXT_CLAIM_TOUR_RATIO_MAX: float = 1.10
#: X4: false-negative rate at which the success target is checked.
EXT_CLAIM_NOISE: float = 0.1
#: X5: non-inferiority margin (bumps per mission): ``search_andor`` counts as "not worse" than
#: ``search_astar`` under noise when the CI upper bound of their paired difference is at most
#: this, pooled over ε and at every ε (and is refuted when a CI lies above 0).
#: Why 0.1, and why absolute: it allows at most one extra collision per ten missions, a
#: difference too small to matter for a rescue. A relative margin (a fraction of
#: ``search_astar``'s rate) was rejected because that rate is itself estimated from the same
#: runs (the threshold would move with the data) and is near 0 at small ε. For scale, the
#: 10×10 quick run measured ``search_astar`` at about 0.35 / 0.70 / 1.07 bumps per mission at
#: ε = 0.05 / 0.1 / 0.2, so 0.1 is roughly 30 % / 15 % / 10 % of that baseline: lenient at the
#: lowest noise level, strict at the highest. The value was set with the claim, before the
#: full run, and must not be changed after seeing its results.
EXT_CLAIM_BUMP_MARGIN: float = 0.1
