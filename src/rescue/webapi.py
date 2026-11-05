"""Python bridge of the web app (docs/plan.md §3.16): catalog, sessions and a JSON dispatcher.

The static web app runs this module, and through it the exact core, in Pyodide inside a Web
Worker. A :class:`WebSession` builds a :class:`~rescue.runner.Mission` (mode ``core``) or an
:class:`~rescue.ext.mission.ExtMission` (modes ``search`` / ``multi`` / ``noisy``) exactly as the
command-line tools do (``make_transition``, ``make_agent`` / ``make_ext_agent``, ``WorldConfig``,
the same seeds), so the same configuration gives the same mission in the browser and in
``python -m rescue.sim`` / ``python -m rescue.demo`` (tested event for event).

Every result is JSON-ready (lists, not tuples or numpy arrays; NaN and infinities become
``None``). Coordinates are 0-indexed ``[row, col]``; every text (log lines, explanations, plan
trees) is 1-indexed. :func:`dispatch` wraps the whole API in one JSON-in / JSON-out call for the
worker.

Imports: no pygame, matplotlib, scipy or PyYAML. :mod:`rescue.mapgen` (scipy) is imported only
for a generated map missing from the pre-generated library (:mod:`rescue.webmaps`) and LAO*
(``lao_expected``) uses scipy only while planning; :func:`needs_scipy` tells the worker in
advance.
"""

from __future__ import annotations

import json
import math
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rescue.agents import AGENT_NAMES, ALL_AGENT_NAMES, Agent, make_agent
from rescue.agents.astar_fixed import FixedAStarAgent
from rescue.config import (
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SLIP_P,
    DEFAULT_TERRAIN_HIGH_P,
    DEMO_SEED,
    EXT_DEMO_MULTI_K,
    EXT_DEMO_NOISE,
    EXT_DEMO_SCENARIO,
    EXT_DEMO_SEED,
    EXT_DEMO_SIZE,
    EXT_MAX_FALSE_POSITIVE,
    MAPGEN_DEFAULT_SIZE,
    MAPGEN_MIN_SIZE,
    MAX_SLIP_P,
    SKID_FRACTION,
    SLIP_STAY_FRACTION,
    STEP_BUDGET_FACTOR,
    AftershockConfig,
    AgentConfig,
    PlannerConfig,
    WorldConfig,
)
from rescue.demo import format_map, initial_plan_text
from rescue.ext.agents import EXT_AGENT_NAMES, SearchAgent, make_ext_agent
from rescue.ext.mission import ExtMission, ExtStepEvent, format_ext_trace
from rescue.ext.scenarios import add_survivors
from rescue.ext.sensing import ProbBelief
from rescue.grid import Pos, TrueGrid, fmt_pos
from rescue.runner import (
    Mission,
    MissionStatus,
    StepEvent,
    format_summary,
    format_trace,
    make_transition,
)
from rescue.webmaps import (
    EDIT_OPS,
    EDITOR_MIN_SIZE,
    GENERATED_SCENARIO_INFO,
    LIBRARY_SEEDS,
    LIBRARY_SIZES,
    MAP_FILE_INFO,
    WEB_MAX_MAP_SIZE,
    WEB_MAX_MAP_TEXT,
    check_map_size,
    edit_map,
    generated_grid,
    get_map_library,
    list_map_files,
    load_map_file,
    needs_generator,
    set_map_library,
    set_maps_dir,
    validate_map,
)
from rescue.webtext import (
    KIND_CONTINGENCY,
    KIND_EXPECTED,
    KIND_PATH,
    ExplainContext,
    core_header,
    core_log_lines,
    explain_event_toned,
    ext_header,
    ext_log_lines,
    friendly_header,
)

__all__ = [
    "API_VERSION",
    "WebSession",
    "catalog",
    "dispatch",
    "edit_map",
    "map_ascii",
    "needs_scipy",
    "normalize_config",
    "set_map_library",
    "set_maps_dir",
    "validate_map",
]

# --- constants ------------------------------------------------------------------------------
#: Version of the JSON shapes below; the front end checks it.
API_VERSION = 1
#: Mission modes: the core mission and the three extension modes (brief §8.3–§8.5).
MODES: tuple[str, ...] = ("core", "search", "multi", "noisy")
CORE_MODE = "core"
#: Map sources of a session config.
MAP_SOURCES: tuple[str, ...] = ("file", "scenario", "ascii")
#: Replanning modes (``AgentConfig.replan_mode``).
REPLAN_MODES: tuple[str, ...] = ("eager", "lazy")
#: Map of the core mode by default, and the scenario name given to an ASCII map.
DEFAULT_CORE_MAP = "deck_demo"
DEFAULT_ASCII_NAME = "custom"
#: Act depth of the plan trees in the state (as in run files).
WEB_TREE_DEPTH = 8
#: Largest sensor radius, aftershock k and survivor count the web app offers.
WEB_MAX_SENSOR_RADIUS = 4
WEB_MAX_AFTERSHOCK_K = 5
WEB_MAX_SURVIVORS = 6
#: Slider steps of the probability parameters.
SLIP_P_STEP = 0.05
NOISE_STEP = 0.05
AFTERSHOCK_Q_STEP = 0.01
#: Largest false-negative rate offered (the sensor model accepts up to 1; above one half a
#: reading no longer moves the belief the right way, as for false positives).
WEB_MAX_FALSE_NEGATIVE = 0.45
#: Largest offered false-positive rate (the model needs it strictly below
#: ``EXT_MAX_FALSE_POSITIVE``).
WEB_MAX_FALSE_POSITIVE = round(EXT_MAX_FALSE_POSITIVE - NOISE_STEP, 2)
#: Largest world seed (numpy seeds are unbounded; this keeps them exact in JavaScript).
WEB_MAX_SEED = 2**31 - 1
#: Open sessions kept by :func:`dispatch` (the oldest is closed beyond this).
WEB_MAX_SESSIONS = 16
#: Decimal places of the debris probabilities in the JSON state (the UI shades and quotes them
#: to 0.1 %; frames keep them as float32).
P_DEBRIS_DECIMALS = 6
#: Event-log tags (what the timeline marks): replans, survivors found, rescues, failures.
TAG_REPLAN, TAG_FOUND, TAG_RESCUE, TAG_FAIL = "replan", "found", "rescue", "fail"
#: Keys of a lean state (:meth:`WebSession.lean_state`) that a full state also has.
LEAN_DROPPED_KEYS: tuple[str, ...] = ("cells", "high_slip", "trace", "log", "plan", "config")

#: Catalog text of every agent: label, one-line description and kind.
AGENT_INFO: dict[str, tuple[str, str, str]] = {
    "andor": (
        "AND-OR contingency",
        "Plans for every slip and skid in advance (a strong-cyclic plan) and replans only when "
        "the map surprises it.",
        KIND_CONTINGENCY,
    ),
    "astar_fixed": (
        "Fixed A*",
        "Plans one shortest path and follows it blindly; gives up at the first slip, skid or "
        "blocked cell.",
        KIND_PATH,
    ),
    "astar_replan": (
        "A* + replan",
        "Follows a shortest path and re-runs A* whenever it is pushed off the path or sees new "
        "debris.",
        KIND_PATH,
    ),
    "dstar_lite": (
        "D* Lite",
        "Like A* + replan, but repairs its previous search incrementally instead of starting "
        "over.",
        KIND_PATH,
    ),
    "vi_expected": (
        "Value iteration (expected cost)",
        "Computes the policy with the fewest expected steps on the robot's map by value "
        "iteration over every cell.",
        KIND_EXPECTED,
    ),
    "lao_expected": (
        "LAO* (expected cost)",
        "The same optimal policy as value iteration, found by heuristic search from the "
        "robot's position.",
        KIND_EXPECTED,
    ),
    "search_andor": (
        "Search + AND-OR",
        "Explores frontier cells until survivors are seen, rescues the nearest first and plans "
        "every leg with AND-OR.",
        KIND_CONTINGENCY,
    ),
    "search_astar": (
        "Search + A*",
        "Explores frontier cells until survivors are seen, rescues the nearest first and plans "
        "every leg with A* + replan.",
        KIND_PATH,
    ),
}
#: Agents whose planning imports scipy.
SCIPY_AGENTS: frozenset[str] = frozenset({"lao_expected"})


@dataclass(frozen=True)
class ModeInfo:
    """A mission mode: catalog text and the defaults of ``rescue.ext.demo.MODES`` (tested)."""

    title: str
    description: str
    survivors_known: bool
    k: int
    false_negative: float


#: The modes. The extension defaults equal ``rescue.ext.demo.MODES`` (spelled out here because
#: that module imports scipy through ``rescue.mapgen``).
MODE_INFO: dict[str, ModeInfo] = {
    "core": ModeInfo(
        "Rescue mission",
        "One survivor at a known location; the robot discovers debris as it drives and copes "
        "with slips and skids.",
        True, 1, 0.0,
    ),
    "search": ModeInfo(
        "Hidden survivor",
        "survivor location unknown (frontier exploration)",
        False, 1, 0.0,
    ),
    "multi": ModeInfo(
        "Several survivors",
        "several known survivors (greedy nearest-first order)",
        True, EXT_DEMO_MULTI_K, 0.0,
    ),
    "noisy": ModeInfo(
        "Noisy sensors",
        "noisy sensing (debris missed with probability eps)",
        True, 1, EXT_DEMO_NOISE,
    ),
}

#: Pygame panel labels (``viz.sim_app`` / ``viz.ext_sim_app``).
STEPS_LABEL = "Steps / budget"
NO_VALUE = "–"
#: Plan-tree messages without a plan (the Pygame simulator's wording).
NO_PLAN_TEXT = "No plan: the survivor is unreachable from {pos} on the robot's map"
NO_TARGET_TEXT = "No target: no reachable survivor or frontier cell from {pos}"
NO_EXT_PLAN_TEXT = "No plan: the target {target} is unreachable from {pos} on the robot's map"
#: Config keys of a session.
CONFIG_KEYS: tuple[str, ...] = (
    "mode", "map", "agent", "seed", "slip_p", "sensor_radius", "replan_mode", "aftershock_q",
    "aftershock_k", "k", "false_negative", "false_positive",
)


# --- JSON helpers ---------------------------------------------------------------------------
def _pos(p: Pos | None) -> list[int] | None:
    return None if p is None else [int(p[0]), int(p[1])]


def _jsonable(value: Any) -> Any:
    """Tuples → lists, numpy scalars → Python, NaN / inf → None, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return float(value)


def _integer(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return int(default)
    if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value:
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    return int(value)


def _in_range(key: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}, got {value:g}")


# --- configuration --------------------------------------------------------------------------
def _default_map(mode: str) -> dict[str, Any]:
    if mode == CORE_MODE:
        return {"source": "file", "name": DEFAULT_CORE_MAP}
    return {"source": "scenario", "scenario": EXT_DEMO_SCENARIO, "rows": EXT_DEMO_SIZE[0],
            "cols": EXT_DEMO_SIZE[1], "seed": EXT_DEMO_SEED}


def _normalize_map(raw: Any, mode: str) -> dict[str, Any]:
    if raw is None:
        return _default_map(mode)
    if not isinstance(raw, Mapping):
        raise ValueError(f"map must be an object with a 'source', got {raw!r}")
    source = raw.get("source")
    if source == "file":
        name = raw.get("name")
        if not isinstance(name, str):
            raise ValueError("a file map needs a 'name' (one of the bundled maps)")
        return {"source": "file", "name": name}
    if source == "scenario":
        scenario = raw.get("scenario")
        if scenario not in GENERATED_SCENARIO_INFO:
            raise ValueError(f"unknown scenario {scenario!r}; choose from "
                             f"{', '.join(GENERATED_SCENARIO_INFO)}")
        size = MAPGEN_DEFAULT_SIZE if mode == CORE_MODE else EXT_DEMO_SIZE
        default_seed = DEMO_SEED if mode == CORE_MODE else EXT_DEMO_SEED
        rows, cols = _integer(raw, "rows", size[0]), _integer(raw, "cols", size[1])
        seed = _integer(raw, "seed", default_seed)
        for key, value in (("rows", rows), ("cols", cols)):
            _in_range(key, value, MAPGEN_MIN_SIZE, WEB_MAX_MAP_SIZE)
        _in_range("map seed", seed, 0, WEB_MAX_SEED)
        return {"source": "scenario", "scenario": scenario, "rows": rows, "cols": cols,
                "seed": seed}
    if source == "ascii":
        text = raw.get("text")
        if not isinstance(text, str):
            raise ValueError("an ascii map needs its 'text'")
        if len(text) > WEB_MAX_MAP_TEXT:
            raise ValueError(f"the map text is too long ({len(text)} characters; at most "
                             f"{WEB_MAX_MAP_TEXT})")
        name = raw.get("name") or DEFAULT_ASCII_NAME
        if not isinstance(name, str):
            raise ValueError(f"map name must be text, got {name!r}")
        return {"source": "ascii", "text": text, "name": name}
    raise ValueError(f"map source must be one of {', '.join(MAP_SOURCES)}, got {source!r}")


def normalize_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """The full session config with defaults filled in (``ValueError`` on bad input).

    Defaults: mode ``core``; map ``deck_demo`` (core) or the extension demo's
    ``collapsed_building`` 12x12 seed 7; agent ``andor`` / ``search_andor``; world ``seed``: the
    map seed for a generated map, else ``DEMO_SEED`` (core) / ``EXT_DEMO_SEED`` (extensions),
    as in ``python -m rescue.sim``; ``k`` / ``false_negative`` from the mode (core: 1 / 0),
    ``false_positive`` 0. The core mode rejects other ``k`` / noise values (like the CLI's "need
    --ext").
    """
    config = dict(raw or {})
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    mode = config.get("mode") or CORE_MODE
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    map_spec = _normalize_map(config.get("map"), mode)
    agents = ALL_AGENT_NAMES if mode == CORE_MODE else EXT_AGENT_NAMES
    agent = config.get("agent") or agents[0]
    if agent not in agents:
        raise ValueError(f"agent {agent!r} is not available in the {mode} mode; choose from "
                         f"{', '.join(agents)}")
    if map_spec["source"] == "scenario":
        default_seed = map_spec["seed"]
    else:
        default_seed = DEMO_SEED if mode == CORE_MODE else EXT_DEMO_SEED
    seed = _integer(config, "seed", default_seed)
    _in_range("seed", seed, 0, WEB_MAX_SEED)
    replan_mode = config.get("replan_mode") or REPLAN_MODES[0]
    if replan_mode not in REPLAN_MODES:
        raise ValueError(f"replan_mode must be one of {', '.join(REPLAN_MODES)}")
    info = MODE_INFO[mode]
    k = _integer(config, "k", info.k)
    false_negative = _number(config, "false_negative", info.false_negative)
    false_positive = _number(config, "false_positive", 0.0)
    if mode == CORE_MODE and (k != 1 or false_negative != 0.0 or false_positive != 0.0):
        raise ValueError("k, false_negative and false_positive need an extension mode "
                         "(search, multi or noisy)")
    _in_range("k", k, 1, WEB_MAX_SURVIVORS)
    _in_range("false_negative", false_negative, 0.0, 1.0)
    if not 0.0 <= false_positive < EXT_MAX_FALSE_POSITIVE:
        raise ValueError(f"false_positive must be in [0, {EXT_MAX_FALSE_POSITIVE:g}), got "
                         f"{false_positive:g}")
    sensor_radius = _integer(config, "sensor_radius", DEFAULT_SENSOR_RADIUS)
    _in_range("sensor_radius", sensor_radius, 0, WEB_MAX_SENSOR_RADIUS)
    aftershock_k = _integer(config, "aftershock_k", AftershockConfig().k)
    _in_range("aftershock_k", aftershock_k, 0, WEB_MAX_AFTERSHOCK_K)
    normalized = {
        "mode": mode,
        "map": map_spec,
        "agent": agent,
        "seed": seed,
        "slip_p": _number(config, "slip_p", DEFAULT_SLIP_P),
        "sensor_radius": sensor_radius,
        "replan_mode": replan_mode,
        "aftershock_q": _number(config, "aftershock_q", AftershockConfig().q),
        "aftershock_k": aftershock_k,
        "k": k,
        "false_negative": false_negative,
        "false_positive": false_positive,
    }
    _world_config(normalized)  # validates p, radius and the aftershock settings
    return normalized


def _world_config(config: Mapping[str, Any]) -> WorldConfig:
    return WorldConfig(
        slip_p=config["slip_p"],
        sensor_radius=config["sensor_radius"],
        aftershock=AftershockConfig(q=config["aftershock_q"], k=config["aftershock_k"]),
    )


def _base_grid(config: Mapping[str, Any]) -> TrueGrid:
    """The true map of a normalized config, with the extension's extra survivors.

    As the CLI builds it: a map file is ``TrueGrid.load`` (``rescue.sim --map``); a generated
    map is ``mapgen.generate`` (from the library when possible) and, in the extension modes,
    ``multi_survivor_grid`` = ``add_survivors(map, map seed, k)``; a map file or an ASCII map
    in an extension mode gets ``add_survivors(map, world seed, k)`` when it has fewer than
    ``k`` survivors (``rescue.sim --ext``).
    """
    spec = config["map"]
    mode = config["mode"]
    k = config["k"]
    if spec["source"] == "scenario":
        grid = generated_grid(spec["scenario"], spec["rows"], spec["cols"], spec["seed"])
        if mode != CORE_MODE and k > 1:
            grid = add_survivors(grid, spec["seed"], k)
        return grid
    if spec["source"] == "file":
        grid = load_map_file(spec["name"])
    else:
        grid = TrueGrid.from_ascii(spec["text"], name=spec["name"], multi=mode != CORE_MODE)
        check_map_size(grid.rows, grid.cols)
    if mode != CORE_MODE and k > len(grid.survivors):
        grid = add_survivors(grid, config["seed"], k)
    return grid


def map_ascii(spec: Mapping[str, Any] | None, mode: str = CORE_MODE) -> str:
    """The ASCII text of a map spec (as in a session config), e.g. to open it in the editor.

    The map itself (no extra survivors are added); comments are dropped.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    normalized = _normalize_map(spec, mode)
    return _base_grid({"map": normalized, "mode": mode, "k": 1, "seed": 0}).to_ascii()


def needs_scipy(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether a session with ``config`` will import scipy: ``{scipy, reasons, agent, map}``.

    True for ``lao_expected`` (``agent``: the agent's label) and for a generated map the
    library does not hold (``map`` True), so the worker can say which one it loads scipy for.
    """
    normalized = normalize_config(config)
    reasons: list[str] = []
    agent = None
    if normalized["agent"] in SCIPY_AGENTS:
        agent = AGENT_INFO[normalized["agent"]][0]
        reasons.append(f"{normalized['agent']} plans with scipy.sparse")
    spec = normalized["map"]
    generator = spec["source"] == "scenario" and needs_generator(
        spec["scenario"], spec["rows"], spec["cols"], spec["seed"])
    if generator:
        reasons.append(f"{spec['scenario']} {spec['rows']}x{spec['cols']} seed {spec['seed']} "
                       "is not in the pre-generated library (loading the map generator)")
    return {"scipy": bool(reasons), "reasons": reasons, "agent": agent, "map": generator}


# --- catalog --------------------------------------------------------------------------------
def _agent_entry(name: str) -> dict[str, Any]:
    label, description, kind = AGENT_INFO[name]
    modes = [CORE_MODE] if name in ALL_AGENT_NAMES else [m for m in MODES if m != CORE_MODE]
    return {
        "name": name,
        "label": label,
        "description": description,
        "kind": kind,
        "modes": modes,
        "needs_scipy": name in SCIPY_AGENTS,
        "main": name in AGENT_NAMES or name in EXT_AGENT_NAMES,
    }


def _presets() -> list[dict[str, Any]]:
    scenario_map = {"source": "scenario", "scenario": EXT_DEMO_SCENARIO,
                    "rows": EXT_DEMO_SIZE[0], "cols": EXT_DEMO_SIZE[1], "seed": EXT_DEMO_SEED}
    return [
        {
            "id": "slide9",
            "title": "Quick demo (4×4)",
            "description": "A tiny map: watch the AND-OR planner prepare for every slip and "
                           "skid and reach the survivor in a few steps.",
            "note": "The sample run of the proposal deck (slide 9): the same mission as "
                    "python -m rescue.demo.",
            "config": {"mode": "core", "map": {"source": "file", "name": DEFAULT_CORE_MAP},
                       "agent": "andor", "seed": DEMO_SEED},
        },
        {
            "id": "head_to_head",
            "title": "Planners head-to-head",
            "description": "AND-OR against fixed A* and A* + replan on one collapsed building, "
                           "with the same map and the same random outcomes for each.",
            "config": {"mode": "core",
                       "map": {"source": "scenario", "scenario": "collapsed_building",
                               "rows": MAPGEN_DEFAULT_SIZE[0], "cols": MAPGEN_DEFAULT_SIZE[1],
                               "seed": EXT_DEMO_SEED},
                       "agent": "andor"},
            "compare_agents": list(AGENT_NAMES),
        },
        {
            "id": "hidden_survivor",
            "title": "Hidden survivor",
            "description": "The survivor's location is unknown: the robot explores frontier "
                           "cells until it sees the survivor.",
            "config": {"mode": "search", "map": dict(scenario_map), "agent": "search_andor"},
        },
        {
            "id": "several_survivors",
            "title": "Several survivors",
            "description": f"{EXT_DEMO_MULTI_K} known survivors, rescued nearest-first.",
            "config": {"mode": "multi", "map": dict(scenario_map), "agent": "search_andor",
                       "k": EXT_DEMO_MULTI_K},
        },
        {
            "id": "noisy_sensors",
            "title": "Noisy sensors",
            "description": f"The debris sensor misses debris with probability "
                           f"{EXT_DEMO_NOISE:g}; the robot keeps a debris probability per cell.",
            "config": {"mode": "noisy", "map": dict(scenario_map), "agent": "search_andor",
                       "false_negative": EXT_DEMO_NOISE},
        },
        {
            "id": "blank_canvas",
            "title": "Blank canvas",
            "description": "An empty 10x10 map to draw your own disaster zone in the editor.",
            "config": {"mode": "core",
                       "map": {"source": "ascii", "name": DEFAULT_ASCII_NAME,
                               "text": _blank_ascii(LIBRARY_SIZES[0], LIBRARY_SIZES[0])},
                       "agent": "andor", "seed": DEMO_SEED},
            "edit": True,
        },
    ]


def _blank_ascii(rows: int, cols: int) -> str:
    """An all-open map with the start top-left and the survivor bottom-right."""
    lines = []
    for r in range(rows):
        row = ["."] * cols
        if r == 0:
            row[0] = "S"
        if r == rows - 1:
            row[-1] = "P"
        lines.append("".join(row))
    return "\n".join(lines) + "\n"


def catalog() -> dict[str, Any]:
    """Everything the UI offers: maps, scenarios, agents, modes, parameters and presets.

    Reads the bundled map files; needs neither scipy nor the map library (whose sizes and
    seeds are reported under ``library``).
    """
    library = get_map_library()
    library_info: dict[str, Any] = (
        library.summary() if library is not None
        else {"loaded": False, "scenarios": [], "sizes": list(LIBRARY_SIZES),
              "seeds": list(LIBRARY_SEEDS), "count": 0}
    )
    modes = []
    for name in MODES:
        info = MODE_INFO[name]
        agents = ALL_AGENT_NAMES if name == CORE_MODE else EXT_AGENT_NAMES
        modes.append({
            "name": name,
            "title": info.title,
            "description": info.description,
            "agents": list(agents),
            "default_agent": agents[0],
            "default_map": _default_map(name),
            "defaults": {"k": info.k, "false_negative": info.false_negative,
                         "false_positive": 0.0, "survivors_known": info.survivors_known},
        })
    return {
        "api": API_VERSION,
        "maps": list_map_files(),
        "scenarios": [
            {"name": name, "title": title, "description": description,
             "sizes": list(LIBRARY_SIZES), "min_size": MAPGEN_MIN_SIZE,
             "max_size": WEB_MAX_MAP_SIZE}
            for name, (title, description) in GENERATED_SCENARIO_INFO.items()
        ],
        "library": library_info,
        "agents": [_agent_entry(name) for name in (*ALL_AGENT_NAMES, *EXT_AGENT_NAMES)],
        "modes": modes,
        "params": {
            "slip_p": {"min": 0.0, "max": MAX_SLIP_P, "step": SLIP_P_STEP,
                       "default": DEFAULT_SLIP_P},
            "terrain_high_p": {"value": DEFAULT_TERRAIN_HIGH_P},
            "outcome_fractions": {"stay": SLIP_STAY_FRACTION, "skid": SKID_FRACTION},
            "sensor_radius": {"min": 0, "max": WEB_MAX_SENSOR_RADIUS, "step": 1,
                              "default": DEFAULT_SENSOR_RADIUS},
            "replan_mode": {"choices": list(REPLAN_MODES), "default": REPLAN_MODES[0]},
            "aftershock_q": {"min": 0.0, "max": 1.0, "step": AFTERSHOCK_Q_STEP,
                             "default": AftershockConfig().q},
            "aftershock_k": {"min": 0, "max": WEB_MAX_AFTERSHOCK_K, "step": 1,
                             "default": AftershockConfig().k},
            "k": {"min": 1, "max": WEB_MAX_SURVIVORS, "step": 1, "default": EXT_DEMO_MULTI_K},
            "false_negative": {"min": 0.0, "max": WEB_MAX_FALSE_NEGATIVE, "step": NOISE_STEP,
                               "default": EXT_DEMO_NOISE},
            "false_positive": {"min": 0.0, "max": WEB_MAX_FALSE_POSITIVE, "step": NOISE_STEP,
                               "default": 0.0, "exclusive_max": EXT_MAX_FALSE_POSITIVE},
            "seed": {"min": 0, "max": WEB_MAX_SEED, "default": DEMO_SEED},
            "map_size": {"min": MAPGEN_MIN_SIZE, "max": WEB_MAX_MAP_SIZE,
                         "default": MAPGEN_DEFAULT_SIZE[0], "library": list(LIBRARY_SIZES),
                         "editor_min": EDITOR_MIN_SIZE, "max_text": WEB_MAX_MAP_TEXT,
                         "library_seeds": len(LIBRARY_SEEDS)},
            "step_budget_factor": {"value": STEP_BUDGET_FACTOR},
        },
        "presets": _presets(),
        "edit_ops": list(EDIT_OPS),
    }


# --- sessions -------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Frame:
    """What the state needs of one recorded event (everything immutable or never mutated)."""

    index: int
    t: int
    status: str
    robot: Pos
    rescued: tuple[Pos, ...]
    target: Pos | None
    target_mode: str | None
    belief: np.ndarray
    p_debris: np.ndarray | None
    grid_version: int
    plan_id: int
    show_plan: bool
    plan_message: str | None
    policy: dict[str, str]
    metrics: dict[str, Any]
    panel: list[list[str]]
    log_end: int
    trace_end: int
    explanation: tuple[str, ...]
    tones: tuple[str, ...]
    event: dict[str, Any]


class WebSession:
    """One mission driven from the browser; see the module docstring.

    ``config`` is normalized by :func:`normalize_config`. The t = 0 event is recorded on
    construction. Every later :meth:`step` records one event; :meth:`state_at` rebuilds the
    state after any recorded event from the recording (it equals the state returned live).
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = normalize_config(config)
        cfg = self.config
        self.mode: str = cfg["mode"]
        grid = _base_grid(cfg)
        self.grid: TrueGrid = grid
        self.scenario: str = grid.name or cfg["map"].get("name", DEFAULT_ASCII_NAME)
        self.world_config = _world_config(cfg)
        transition = make_transition(grid, self.world_config)
        agent_config = AgentConfig(replan_mode=cfg["replan_mode"])
        self.mission: Mission | ExtMission
        if self.mode == CORE_MODE:
            agent: Agent = make_agent(cfg["agent"], transition, PlannerConfig(), agent_config)
            self.mission = Mission(grid, agent, self.world_config, seed=cfg["seed"],
                                   scenario=self.scenario)
        else:
            info = MODE_INFO[self.mode]
            ext_agent = make_ext_agent(cfg["agent"], transition, PlannerConfig(), agent_config)
            self.mission = ExtMission(
                grid, ext_agent, self.world_config, cfg["seed"],
                survivors_known=info.survivors_known,
                false_negative=cfg["false_negative"],
                false_positive=cfg["false_positive"],
                scenario=self.scenario,
                mode=self.mode,
            )
        self.config["k"] = len(grid.survivors)
        label, _, kind = AGENT_INFO[cfg["agent"]]
        self.agent_info = {"name": cfg["agent"], "label": label, "kind": kind}
        self._frames: list[_Frame] = []
        self._trace: list[dict[str, Any]] = []
        self._log: list[dict[str, Any]] = []
        self._grids: list[tuple[list[list[int]], list[list[bool]]]] = []
        self._plans: dict[int, dict[str, Any] | None] = {}
        self._paths: dict[int, list[list[int]] | None] = {}
        self._envelopes: dict[int, list[list[int]]] = {}
        self._rescued_count = 0
        self._previous_target: Pos | None = None
        # What the (one) client of a lean session already holds: trace and log prefixes, grid
        # versions and plan ids (see lean_state).
        self._sent_trace = self._sent_log = 0
        self._sent_grids: set[int] = set()
        self._sent_plans: set[int] = set()
        self._grid_snapshot()
        self._header()
        self._record(self.mission.start(), replans_before=0, fallbacks_before=0)

    # --- properties ------------------------------------------------------------------------
    @property
    def done(self) -> bool:
        """True once the mission reached a terminal status."""
        return self.mission.done

    @property
    def t(self) -> int:
        """World time of the latest event."""
        return self._frames[-1].t

    @property
    def agent(self) -> Agent:
        """The mission's agent."""
        return self.mission.agent

    # --- driving ---------------------------------------------------------------------------
    def step(self, lean: bool = False) -> dict[str, Any]:
        """One mission step: ``{event, explanation, state}`` (``RuntimeError`` once done);
        ``lean`` sends :meth:`lean_state` instead of the full state."""
        if self.done:
            raise RuntimeError(f"The mission is over ({self.mission.status.value}); start a new "
                               "one to run again")
        stats = self.agent.stats
        replans, fallbacks = int(stats.replans), self._fallbacks()
        event = self.mission.step()
        frame = self._record(event, replans_before=replans, fallbacks_before=fallbacks)
        state = self._lean(frame) if lean else self._assemble(frame)
        return {"event": frame.event, "explanation": list(frame.explanation), "state": state}

    def run(self, max_steps: int | None = None, lean: bool = False) -> dict[str, Any]:
        """Step until done or ``max_steps`` steps: ``{events, state}`` (the new events)."""
        # The mission ends by itself within its budget; the limit only guards the loop.
        limit = self.mission.budget + 1 if max_steps is None else max(0, int(max_steps))
        events: list[dict[str, Any]] = []
        while not self.done and len(events) < limit:
            stats = self.agent.stats
            replans, fallbacks = int(stats.replans), self._fallbacks()
            frame = self._record(self.mission.step(), replans_before=replans,
                                 fallbacks_before=fallbacks)
            events.append(frame.event)
        return {"events": events, "state": self.lean_state() if lean else self.state()}

    # --- states ----------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        """The state after the latest event (see docs/plan.md §3.16 for its keys)."""
        return self._assemble(self._frames[-1])

    def state_at(self, t: int) -> dict[str, Any]:
        """The state after the last event at world time ``t`` (``0 <= t <=`` :attr:`t`).

        A mission ending on an agent abort records two events at its last ``t`` (the move and
        the abort); ``state_at`` returns the later one, i.e. the live state at that ``t``.
        """
        return self._assemble(self._frames[self._frame_index(t)])

    def lean_state(self, t: int | None = None) -> dict[str, Any]:
        """The state at ``t`` (default: the latest) without what the client already holds.

        For the web worker, whose single client keeps everything it was sent: the state
        carries ``trace_end`` / ``log_end`` (the client slices its own copies) and only the
        trace and log entries it has not been sent yet (``trace_new`` / ``log_new``); the grid
        (``cells`` / ``high_slip``) only for a ``grid_version`` it has not been sent, and the
        plan only for a ``plan_id`` it has not been sent; no ``config``. Put back together it
        equals :meth:`state_at` (tested), so payloads stay small however long the mission.
        """
        if t is None:
            return self._lean(self._frames[-1])
        return self._lean(self._frames[self._frame_index(t)])

    def _lean(self, frame: _Frame) -> dict[str, Any]:
        state = self._assemble(frame)
        for key in LEAN_DROPPED_KEYS:
            state.pop(key, None)
        state["lean"] = True
        state["trace_new"] = self._trace[self._sent_trace:]
        state["log_new"] = self._log[self._sent_log:]
        self._sent_trace, self._sent_log = len(self._trace), len(self._log)
        if frame.grid_version not in self._sent_grids:
            cells, high_slip = self._grids[frame.grid_version]
            state["cells"], state["high_slip"] = cells, high_slip
            self._sent_grids.add(frame.grid_version)
        pid = frame.plan_id
        if frame.show_plan and pid >= 0 and pid not in self._sent_plans:
            state["plan"] = self._plans.get(pid)
            self._sent_plans.add(pid)
        state["has_plan"] = frame.show_plan and pid >= 0 and self._plans.get(pid) is not None
        return state

    def _frame_index(self, t: int) -> int:
        t = int(t)
        if not 0 <= t <= self.t:
            raise ValueError(f"t must be between 0 and {self.t}, got {t}")
        return max(i for i, f in enumerate(self._frames) if f.t == t)

    def frame_count(self) -> int:
        """Number of recorded events (t = 0 included)."""
        return len(self._frames)

    def report(self) -> dict[str, str]:
        """The command-line texts of this mission: ``map`` (1-indexed rulers), ``initial_plan``,
        ``trace`` (the deck trace) and ``summary`` (core: the demo's two lines)."""
        mission = self.mission
        if isinstance(mission, Mission):
            trace = format_trace(mission.events)
            summary = format_summary(mission.metrics())
        else:
            trace = format_ext_trace(mission.events)
            m = mission.metrics()
            verdict = ("Rescue successful" if m.success else
                       "Rescue in progress" if m.failure_reason is None else
                       f"Rescue failed ({m.failure_reason})")
            summary = (f"Rescued: {m.rescued}/{m.k}   Cells explored: {m.cells_explored}   "
                       f"Replans: {m.replans}   Collisions: {m.bumps}\n{verdict}")
        return {
            "map": format_map(self.grid),
            "initial_plan": initial_plan_text(mission),  # type: ignore[arg-type]
            "trace": trace,
            "summary": summary,
        }

    # --- recording -------------------------------------------------------------------------
    def _fallbacks(self) -> int:
        agent = self.agent
        return int(agent.fallback_replans) if isinstance(agent, SearchAgent) else 0

    def _grid_snapshot(self) -> None:
        grid = self.mission.world.grid if self.mission.started else self.grid
        self._grids.append((grid.cells.tolist(), grid.high_slip.tolist()))

    def _map_title(self) -> str:
        spec = self.config["map"]
        if spec["source"] == "file":
            return MAP_FILE_INFO.get(spec["name"], (spec["name"], ""))[0].split(" (")[0]
        if spec["source"] == "scenario":
            title = GENERATED_SCENARIO_INFO[spec["scenario"]][0]
            return f"{title} (map seed {spec['seed']})"
        return "your map"

    def _header(self) -> None:
        """The Pygame header lines; the first also gets a ``display`` text with labels."""
        cfg, grid = self.config, self.grid
        if self.mode == CORE_MODE:
            lines = [(core_header(cfg["agent"], self.scenario, grid.rows, grid.cols,
                                  cfg["seed"], cfg["slip_p"]), "info")]
            extra = ""
        else:
            lines = ext_header(cfg["agent"], self.mode, self.scenario, grid.rows, grid.cols,
                               cfg["seed"], cfg["slip_p"], len(grid.survivors),
                               MODE_INFO[self.mode].survivors_known, cfg["false_negative"],
                               cfg["false_positive"])
            extra = f", mode “{MODE_INFO[self.mode].title}”"
        display = friendly_header(self.agent_info["label"], self._map_title(), grid.rows,
                                  grid.cols, cfg["seed"], cfg["slip_p"], extra)
        for i, (text, kind) in enumerate(lines):
            entry: dict[str, Any] = {"t": 0, "kind": kind, "text": text}
            if i == 0:
                entry["display"] = display
            self._log.append(entry)

    def _snapshot(self):  # type: ignore[no-untyped-def]
        agent = self.agent
        pid = agent.current_plan_id
        return agent.snapshots[pid] if 0 <= pid < len(agent.snapshots) else None

    def _plan_json(self, plan_id: int) -> dict[str, Any] | None:
        """The plan of snapshot ``plan_id`` for the state (cached; None for path agents)."""
        if plan_id not in self._plans:
            snap = self.agent.snapshots[plan_id]
            plan = snap.plan
            if plan is None:
                self._plans[plan_id] = None
            else:
                stats = plan.stats()
                expected = getattr(self.agent, "expected_steps", None)
                self._plans[plan_id] = {
                    "id": plan_id,
                    "t": int(snap.t),
                    "made_at": _pos(snap.pos),
                    "pretty": plan.pretty(),
                    "tree": _jsonable(plan.to_tree(WEB_TREE_DEPTH)),
                    "stats": {"nodes": stats.nodes, "act_nodes": stats.act_nodes,
                              "depth": stats.depth, "loops": stats.loops, "refs": stats.refs,
                              "envelope_size": stats.envelope_size},
                    "expected_steps": _jsonable(float(expected)) if expected is not None
                    else None,
                }
            self._paths[plan_id] = (None if snap.path is None
                                    else [_pos(p) for p in snap.path])  # type: ignore[misc]
            self._envelopes[plan_id] = [_pos(p) for p in sorted(snap.envelope)]  # type: ignore[misc]
        return self._plans[plan_id]

    def _policy(self, previous: _Frame | None) -> dict[str, str]:
        arrows = {f"{r},{c}": a.name for (r, c), a in sorted(self.agent.policy_arrows().items())}
        if previous is not None and previous.policy == arrows:
            return previous.policy  # shared: frames never mutate it
        return arrows

    def _plan_view(self, robot: Pos) -> tuple[bool, str | None]:
        """Whether the state shows the current snapshot's plan / path, and the message
        without one (the Pygame plan-tree texts)."""
        snap = self._snapshot()
        if isinstance(self.agent, SearchAgent):
            target = self.agent.target
            if target is None:
                return False, NO_TARGET_TEXT.format(pos=fmt_pos(robot))
            if snap is None or snap.plan is not None or snap.path is not None:
                return snap is not None, None
            return True, NO_EXT_PLAN_TEXT.format(target=fmt_pos(target), pos=fmt_pos(snap.pos))
        if snap is None:
            return False, None
        if snap.plan is None and snap.path is None:
            return True, NO_PLAN_TEXT.format(pos=fmt_pos(snap.pos))
        return True, None

    def _panel(self) -> list[list[str]]:
        """The Pygame info-panel metrics as ``[label, value]`` pairs."""
        mission, agent = self.mission, self.agent
        if isinstance(mission, Mission):
            m = mission.metrics()
            return [
                [STEPS_LABEL, f"{m.steps}/{mission.budget}"],
                ["Replans", str(m.replans)],
                ["Slips", str(m.slips)],
                ["Skids", str(m.skids)],
                ["Explored", str(m.cells_explored)],
                ["Aftershocks", str(m.aftershock_events)],
                ["Plan ms", f"{m.planning_ms:.1f}"],
                ["Plan size", str(m.plan_size)],
                ["Unplanned", str(m.unplanned_states)],
                ["Bumps", str(m.bumps)],
                ["Plan #", str(agent.current_plan_id)],
            ]
        e = mission.metrics()
        assert isinstance(agent, SearchAgent)
        first = e.first_detection_step
        noise = f"{e.false_negative:g}"
        if e.false_positive > 0:
            noise += f" / fp {e.false_positive:g}"
        rows = [
            [STEPS_LABEL, f"{e.steps}/{mission.budget}"],
            ["Mode", "done" if mission.done else agent.mode],
            ["Rescued", f"{e.rescued}/{e.k}"],
            ["First found", NO_VALUE if math.isnan(first) else f"t = {int(first)}"],
            ["Replans", str(e.replans)],
            ["Slips", str(e.slips)],
            ["Skids", str(e.skids)],
            ["Bumps", str(e.bumps)],
            ["Explored", str(e.cells_explored)],
            ["Unplanned", str(e.unplanned_states)],
            ["Fallbacks", str(e.fallback_replans)],
            ["Plan ms", f"{e.planning_ms:.1f}"],
            ["Plan #", str(agent.current_plan_id)],
            ["Noise ε", noise],
        ]
        if self.config["aftershock_q"] > 0:
            rows.append(["Aftershocks", str(e.aftershock_events)])
        return rows

    def _event_json(self, event: StepEvent | ExtStepEvent) -> dict[str, Any]:
        data: dict[str, Any] = {
            "t": int(event.t),
            "pos": _pos(event.pos),
            "action": None if event.action is None else event.action.name,
            "outcome": None if event.outcome is None else event.outcome.value,
            "new_pos": _pos(event.new_pos),
            "blocked": bool(event.blocked),
            "bump": bool(event.bump),
            "contradictions": [_pos(p) for p in event.contradictions],
            "triggering_contradictions": [_pos(p) for p in event.triggering_contradictions],
            "replanned": bool(event.replanned),
            "unplanned": bool(event.unplanned),
            "aftershock_cells": [_pos(p) for p in event.aftershock_cells],
            "status": event.status.value,
            "note": event.note,
            "plan_id": int(self.agent.current_plan_id),
        }
        if isinstance(event, StepEvent):
            data["new_cells"] = [[_pos(p), int(cell)] for p, cell in event.new_cells]
        else:
            data.update({
                "target": _pos(event.target),
                "mode": event.mode,
                "rescued_now": _pos(event.rescued_now),
                "found_now": [_pos(p) for p in event.found_now],
            })
        return data

    def _context(self, event: StepEvent | ExtStepEvent, replans_before: int,
                 fallbacks_before: int) -> ExplainContext:
        agent, mission, cfg = self.agent, self.mission, self.config
        stats = agent.stats
        replans = int(stats.replans)
        new_plan = replans > replans_before or event.action is None
        snap = self._snapshot()
        plan_pretty = path_moves = None
        if new_plan and snap is not None:
            if snap.plan is not None:
                plan_pretty = snap.plan.pretty()
            elif snap.path is not None:
                path_moves = len(snap.path) - 1
        expected = getattr(agent, "expected_steps", None)
        next_action = None
        path_ahead: tuple[Pos, ...] = ()
        blocked_ahead: Pos | None = None
        if not mission.done:
            next_action = agent.policy_arrows().get(event.new_pos)
            if isinstance(agent, FixedAStarAgent):
                # Fixed A* moves next only from its expected path cell with open ground ahead;
                # otherwise its next decision is an abort, so there is no "next move".
                path_ahead = tuple(agent._path[agent._cursor + 1:]) if agent._path else ()
                on_path = agent._expected_cell() == event.new_pos
                if on_path and agent._next_is_known_debris():
                    blocked_ahead = agent._next_cell()
                if not on_path or blocked_ahead is not None:
                    next_action = None
        if isinstance(agent, SearchAgent):
            goal: Pos | None = agent.target
        else:
            goal = self.grid.survivor
        return ExplainContext(
            agent=cfg["agent"],
            kind=self.agent_info["kind"],
            mode=self.mode,
            rows=self.grid.rows,
            cols=self.grid.cols,
            replan_mode=cfg["replan_mode"],
            sensor_radius=cfg["sensor_radius"],
            slip_p=float(mission.transition.slip_p(event.pos)),
            high_slip=bool(self.grid.high_slip[event.pos]),
            replans=replans,
            budget=int(mission.budget),
            goal=goal,
            plan_pretty=plan_pretty,
            path_moves=path_moves,
            expected_steps=(float(expected) if expected is not None
                            and math.isfinite(float(expected)) else None),
            next_action=next_action,
            has_plan=snap is not None and (snap.plan is not None or snap.path is not None),
            k=len(self.grid.survivors),
            rescued_count=self._rescued_count,
            survivors_known=MODE_INFO[self.mode].survivors_known,
            previous_target=self._previous_target,
            false_negative=cfg["false_negative"],
            false_positive=cfg["false_positive"],
            fallback_started=self._fallbacks() > fallbacks_before,
            path_ahead=path_ahead,
            blocked_ahead=blocked_ahead,
        )

    def _record(self, event: StepEvent | ExtStepEvent, *, replans_before: int,
                fallbacks_before: int) -> _Frame:
        """Log, trace, explain and capture one event."""
        mission, agent = self.mission, self.agent
        previous = self._frames[-1] if self._frames else None
        if isinstance(event, ExtStepEvent):
            if event.rescued_now is not None:
                self._rescued_count += 1
            lines = ext_log_lines(event, previous_target=self._previous_target,
                                  rescued_count=self._rescued_count, k=len(self.grid.survivors))
        else:
            lines = core_log_lines(event)
        found = ({(f"Survivor found at {fmt_pos(p)}", "success") for p in event.found_now}
                 if isinstance(event, ExtStepEvent) else set())
        for text, kind in lines:
            entry: dict[str, Any] = {"t": int(event.t), "kind": kind, "text": text}
            tag = _log_tag(text, kind, found)
            if tag is not None:
                entry["tag"] = tag
            self._log.append(entry)
        if event.action is not None and event.outcome is not None:
            self._trace.append({
                "t": int(event.t),
                "from": _pos(event.pos),
                "to": _pos(event.new_pos),
                "action": event.action.name,
                "outcome": event.outcome.value,
                "blocked": bool(event.blocked),
                "bump": bool(event.bump),
                "replanned": bool(event.replanned),
                "rescued": _pos(getattr(event, "rescued_now", None)),
            })
        if event.aftershock_cells:
            self._grid_snapshot()
        toned = explain_event_toned(event, self._context(event, replans_before,
                                                         fallbacks_before))
        world = mission.world
        belief = agent.belief
        p_debris = (belief.p_debris.astype(np.float32) if isinstance(belief, ProbBelief)
                    else None)
        robot = world.pos
        show_plan, message = self._plan_view(robot)
        plan_id = int(agent.current_plan_id)
        if plan_id >= 0:
            self._plan_json(plan_id)
        if isinstance(agent, SearchAgent):
            assert isinstance(mission, ExtMission)
            rescued = tuple(mission.rescued)
            target = None if mission.done else agent.target
            target_mode: str | None = None if mission.done else agent.mode
        else:
            rescued = (self.grid.survivor,) if mission.status is MissionStatus.RESCUED else ()
            target = None if mission.done else self.grid.survivor
            target_mode = None
        frame = _Frame(
            index=len(self._frames),
            t=int(world.t),
            status=mission.status.value,
            robot=robot,
            rescued=rescued,
            target=target,
            target_mode=target_mode,
            belief=np.array(belief.grid, dtype=np.int8),
            p_debris=p_debris,
            grid_version=len(self._grids) - 1,
            plan_id=plan_id,
            show_plan=show_plan,
            plan_message=message,
            policy=self._policy(previous),
            metrics=_jsonable(mission.metrics().as_dict()),
            panel=self._panel(),
            log_end=len(self._log),
            trace_end=len(self._trace),
            explanation=tuple(text for text, _ in toned),
            tones=tuple(tone for _, tone in toned),
            event=self._event_json(event),
        )
        self._frames.append(frame)
        if isinstance(event, ExtStepEvent):
            self._previous_target = event.target
        return frame

    def _assemble(self, frame: _Frame) -> dict[str, Any]:
        """The JSON state of one recorded frame.

        The dict shares recorded data (grids, plan tree, policy, event, config) with the
        session and with other states: treat it as read-only (:func:`dispatch` serializes it).
        """
        cells, high_slip = self._grids[frame.grid_version]
        pid = frame.plan_id
        show = frame.show_plan and pid >= 0
        plan = self._plans.get(pid) if show else None
        path = self._paths.get(pid) if show else None
        envelope = self._envelopes.get(pid, []) if show else []
        return {
            "api": API_VERSION,
            "frame": frame.index,
            "t": frame.t,
            "status": frame.status,
            "done": frame.status != MissionStatus.RUNNING.value,
            "mode": self.mode,
            "rows": self.grid.rows,
            "cols": self.grid.cols,
            "cells": cells,
            "high_slip": high_slip,
            "belief": frame.belief.tolist(),
            "p_debris": (None if frame.p_debris is None
                         else np.round(frame.p_debris.astype(np.float64),
                                       P_DEBRIS_DECIMALS).tolist()),
            "robot": _pos(frame.robot),
            "start": _pos(self.grid.start),
            "survivors": [_pos(s) for s in self.grid.survivors],
            "survivors_known": MODE_INFO[self.mode].survivors_known,
            "rescued": [_pos(s) for s in frame.rescued],
            "target": _pos(frame.target),
            "target_mode": frame.target_mode,
            "trace": self._trace[:frame.trace_end],
            "plan_id": pid,
            "grid_version": frame.grid_version,
            "trace_end": frame.trace_end,
            "log_end": frame.log_end,
            "policy": frame.policy,
            "envelope": envelope,
            "plan": plan,
            "path": path,
            "plan_message": frame.plan_message,
            "metrics": frame.metrics,
            "panel": frame.panel,
            "budget": int(self.mission.budget),
            "log": self._log[:frame.log_end],
            "explanation": list(frame.explanation),
            "explanation_tones": list(frame.tones),
            "event": frame.event,
            "agent": self.agent_info,
            "seed": int(self.config["seed"]),
            "scenario": self.scenario,
            "config": self.config,
        }


def _log_tag(text: str, kind: str, found: set[tuple[str, str]]) -> str | None:
    """What the timeline marks for an event-log line (None: nothing). ``found`` holds this
    event's "Survivor found" lines, built from the event's own data."""
    if (text, kind) in found:
        return TAG_FOUND
    return {"success": TAG_RESCUE, "failure": TAG_FAIL, "replan": TAG_REPLAN}.get(kind)


# --- dispatcher -----------------------------------------------------------------------------
_sessions: dict[int, WebSession] = {}
_next_session_id = 1


def _open(config: Mapping[str, Any] | None = None, lean: bool = False) -> dict[str, Any]:
    global _next_session_id
    session = WebSession(config)
    session_id = _next_session_id
    _next_session_id += 1
    _sessions[session_id] = session
    while len(_sessions) > WEB_MAX_SESSIONS:
        del _sessions[min(_sessions)]
    # The first lean state holds everything sent so far (grid, plan, trace, log) and the config.
    state = {**session.lean_state(), "config": session.config} if lean else session.state()
    return {"id": session_id, "state": state}


def _session(session_id: int) -> WebSession:
    try:
        return _sessions[int(session_id)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"no open session {session_id!r} (it may have been closed)") from None


def _close(session_id: int) -> dict[str, Any]:
    return {"closed": _sessions.pop(int(session_id), None) is not None}


def _version() -> dict[str, Any]:
    import sys

    return {"api": API_VERSION, "python": sys.version.split()[0], "numpy": np.__version__}


def _load_library(path: str | None = None) -> dict[str, Any]:
    set_map_library(path)
    library = get_map_library()
    return library.summary() if library is not None else {"loaded": False}


#: Methods of :func:`dispatch`: name → callable taking the request's ``params``.
METHODS: dict[str, Callable[..., Any]] = {
    "version": _version,
    "catalog": catalog,
    "needs_scipy": lambda config=None: needs_scipy(config),
    "normalize_config": lambda config=None: normalize_config(config),
    "map_ascii": lambda spec, mode=CORE_MODE: map_ascii(spec, mode),
    "open": _open,
    "step": lambda id, lean=False: _session(id).step(lean),
    "run": lambda id, max_steps=None, lean=False: _session(id).run(max_steps, lean),
    "state": lambda id: _session(id).state(),
    "state_at": lambda id, t, lean=False: (_session(id).lean_state(t) if lean
                                           else _session(id).state_at(t)),
    "report": lambda id: _session(id).report(),
    "close": lambda id: _close(id),
    "edit_map": lambda ascii, op: edit_map(ascii, op),
    "validate_map": lambda ascii, mode=CORE_MODE: validate_map(ascii, mode),
    "load_map_library": _load_library,
}


def dispatch(request: str) -> str:
    """One JSON request ``{"method": name, "params": {...}}`` → one JSON response.

    ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": {"type", "message",
    "details"}}`` (``details`` is the Python traceback). Never raises.
    """
    try:
        data = json.loads(request)
        if not isinstance(data, dict):
            raise ValueError("a request must be a JSON object")
        method = data.get("method")
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
        result = METHODS[method](**params)
        return json.dumps({"ok": True, "result": _jsonable(result)}, ensure_ascii=False,
                          allow_nan=False)
    except Exception as exc:  # every error goes back to the UI as a toast
        return json.dumps({
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "details": traceback.format_exc()},
        }, ensure_ascii=False)
