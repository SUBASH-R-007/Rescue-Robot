# Implementation plan and interface contract

This is the Phase 0 deliverable: the file tree, key interfaces, algorithm sketch, test plan and
decisions. **It is the contract every module is written against.** If code and this document
disagree, fix one of them in the same change and say why.

Conventions: Python ≥ 3.11, type hints everywhere, `from __future__ import annotations`,
frozen dataclasses where possible, docstrings on public functions, no magic numbers in logic
(constants live in `rescue/config.py`). Internal coordinates are 0-indexed `(row, col)`;
everything shown to a user (traces, CLI, UI, plan pretty-print) is 1-indexed via `fmt_pos`.

---

## 1. File tree

```
src/rescue/
  __init__.py            version, re-exports nothing heavy
  config.py              constants + frozen config dataclasses + YAML loading
  grid.py                Pos, Cell, fmt_pos, neighbours, TrueGrid, ASCII I/O
  belief.py              BeliefMap, BeliefUpdate (sensing update, contradiction detection)
  model.py               Action, OutcomeKind, TransitionModel (UniformSlip, TerrainSlip),
                         Outcome, ProblemModel.results()
  world.py               Percept, StepResult, RngStreams, World (step, sensor, aftershocks)
  plan.py                Goal / Loop / Ref / Act / Branch, Plan (to_policy, pretty, to_tree,
                         stats, envelope), plan_from_policy
  planners/
    __init__.py
    bfs.py               bfs_distances, shortest_path, shortest_path_length, reachable
    astar.py             astar (Manhattan, tie-break on larger g)
    andor.py             and_or_search, AndOrPlanner (fallback + validation), PlannerStats
    validate.py          validate_policy, is_strong_cyclic
    strong_cyclic_fixpoint.py   reference planner (Cimatti et al. 2003)
    dstar_lite.py        [stretch] D* Lite
    value_iteration.py   [stretch] expected-cost optimal policy
  agents/
    __init__.py          AGENTS registry, make_agent(name, ...)
    base.py              Agent ABC, Abort, AgentStats, AgentUpdate, PlanSnapshot, Observation
    contingency.py       ContingencyAgent  (id "andor")
    astar_fixed.py       FixedAStarAgent   (id "astar_fixed")
    astar_replan.py      ReplanAStarAgent  (id "astar_replan")
    [dstar_lite.py, vi_expected.py]  stretch agents
  runner.py              Mission, StepEvent, MissionStatus, MissionMetrics, run_mission,
                         format_trace, format_summary, make_transition
  runs.py                save_run / load_run (JSON replay files)
  mapgen.py              scenario generators with solvability guarantee, SCENARIOS registry
  demo.py                `python -m rescue.demo`
  sim.py compare.py replay.py   thin CLI entry points → viz apps
  viz/  theme.py grid_view.py tree_view.py panel.py sim_app.py compare_app.py
        replay_app.py record.py export.py (plan-tree PNG via matplotlib)
  experiments/ __init__.py __main__.py run.py stats.py plots.py hypotheses.py
maps/ deck_demo.txt deck_figure.txt building_01.txt …
experiments/ main.yaml quick.yaml
tests/  one test module per source module, plus property and invariant tests
docs/   plan.md algorithm.md architecture.md results.md figures/
```

---

## 2. Decisions that resolve ambiguities (beyond prompt §2)

| # | Decision | Why |
|---|---|---|
| D1 | `Cell` is an `IntEnum`: `UNKNOWN=0, OPEN=1, DEBRIS=2, SURVIVOR=3`; arrays are `np.int8`. | `np.zeros` = all unknown. |
| D2 | High-slip terrain (`~`) is **not** a cell type; `TrueGrid.high_slip` is a separate bool mask. A `~` cell is `OPEN`. | Terrain changes dynamics, not occupancy. |
| D3 | Outcome draw uses **one uniform `u` per step** from the outcome stream, mapped by inverse CDF over the fixed order `(ARRIVED, SLIPPED, SKID_L, SKID_R)`. | Common random numbers: every agent consumes exactly one draw per step. |
| D4 | Slip depends on the cell the robot **stands on** (`TransitionModel.slip_p(pos)`). | "Mud under the wheels." |
| D5 | Agents are given the `TransitionModel` (the robot knows its own dynamics, including a prior terrain map); they are **not** given the debris map. The planner uses only the *support* (outcomes with probability > 0). | Sound: the plan covers every outcome that can happen. |
| D6 | **Bump / collision** = an outcome tried to enter a debris cell that the agent's belief did **not** mark as `DEBRIS` when it acted (an unexpected collision). A move/skid into *known* debris or off-grid is a **blocked** outcome (robot stays; it was anticipated and merged with "stay"). A bump also reveals the bumped cell (touch sensing). | Makes "Collisions: 0" a meaningful invariant: with r ≥ 1 it holds by construction; with r = 0 bumps really happen (tested). |
| D7 | Sensor footprint = all in-grid cells with Manhattan distance ≤ r **including the robot's own cell**. r = 0 is allowed (testing only). | Own cell is trivially observed. |
| D8 | **Contradiction** = a sensed `DEBRIS` where the belief held `UNKNOWN` (optimism violated) or `OPEN` (aftershock). `UNKNOWN→OPEN` is not a contradiction. | Prompt §2.5 plus aftershocks. |
| D9 | Step order in `World.step`: draw outcome → move → aftershock (if any) → sense. Aftershocks target cells **outside the robot's current sensor footprint** (the robot does not witness them), never the robot/survivor cell, never a cell that disconnects robot↔survivor on the true map. The aftershock stream draws `(u_event, permutation)` every step whether or not an event fires. | Belief can go stale (OPEN→DEBRIS), which exercises replanning; draws stay aligned across agents. |
| D10 | **Unplanned state** = after executing action `a` from `s`, the new position is not among the outcomes the agent's current plan anticipated for `(s, a)`. For AND-OR that is `Results(s,a)` on the planning model; for A\* agents it is the single expected next cell. | Same metric for every agent: A\* agents have one per deviation; AND-OR must have 0. |
| D11 | "Slips handled" in the summary = slips + skids (non-`arrived` outcomes) the agent survived. Metrics also keep `slips` and `skids` separately. | Matches deck wording. |
| D12 | `Replans` counts re-planning calls **after** the initial plan. | Deck semantics. |
| D13 | Action iteration order `ACTIONS = (N, E, S, W)`. AND-OR `ORDERED_ACTIONS(s)` sorts by `(bfs_dist(intended target), diag(intended target), index in ACTIONS)` where a blocked intended target counts as `s`, and `diag(p) = abs(abs(goal.r-p.r) - abs(goal.c-p.c))` (stay near the straight line to the goal, like A\*'s cross-product tie-break). | Deterministic and reproduces the deck's root plan (E at (1,1), S at (1,2)). |
| D14 | Deck root plan: the planner produces `E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}`. The deck's `N-plan` at (2,1) **cannot** come from a distance-ordered planner on that map: going N from (2,1) is never strictly better than going E, because `d(2,2) ≤ 1 + min(d(1,2), d(2,1)) = d(1,1)`. At t=0 the belief is optimistic, so `d(2,2)=4 < d(1,1)=6`. We reproduce the shape and document the one-letter deviation rather than hard-coding it. | Honesty over cosmetics. |
| D15 | Step budget default `4·R·C`. Depth limit default `R·C` (= no effective cut-off since path states are distinct). | Prompt §2.6, §3. |
| D16 | Wasted moves = `steps − shortest true-map path length`, recorded only for successful missions (NaN otherwise). | Undefined on failure. |
| D17 | Venv lives outside OneDrive (`%USERPROFILE%/.venvs/rescue-robot`); README documents a normal `pip install -e .[dev]`. | Avoid syncing ~200 MB of site-packages. |

---

## 3. Module contracts

### 3.1 `config.py`
Constants (module level, UPPER_CASE): `SLIP_STAY_FRACTION = 0.75`, `SKID_FRACTION = 0.25`,
`DEFAULT_SLIP_P = 0.2`, `MAX_SLIP_P = 0.4`, `DEFAULT_SENSOR_RADIUS = 1`,
`STEP_BUDGET_FACTOR = 4`, `DEFAULT_TERRAIN_HIGH_P = 0.4`, `MAPGEN_MAX_ATTEMPTS = 200`,
`DEFAULT_RUBBLE_DENSITY = 0.25`, `RUBBLE_DENSITY_RANGE = (0.15, 0.35)`, `PROB_EPS = 1e-12`.

```python
@dataclass(frozen=True)
class AftershockConfig:   q: float = 0.0; k: int = 1          # q = per-step event probability
@dataclass(frozen=True)
class WorldConfig:        slip_p: float = DEFAULT_SLIP_P; sensor_radius: int = DEFAULT_SENSOR_RADIUS
                          terrain_high_p: float = DEFAULT_TERRAIN_HIGH_P
                          aftershock: AftershockConfig = AftershockConfig()
                          budget_factor: int = STEP_BUDGET_FACTOR
                          def step_budget(self, rows, cols) -> int   # budget_factor*rows*cols
@dataclass(frozen=True)
class PlannerConfig:      depth_limit: int | None = None   # None → rows*cols
                          tie_break: Literal["diagonal", "fixed"] = "diagonal"
                          validate: bool = False           # validate every plan (debug/tests)
                          fallback_to_reference: bool = True
@dataclass(frozen=True)
class AgentConfig:        replan_mode: Literal["eager", "lazy"] = "eager"
def load_yaml(path) -> dict
def world_config_from_dict(d) -> WorldConfig   (and planner/agent equivalents; unknown keys → ValueError)
```
`WorldConfig.__post_init__` validates `0 ≤ slip_p ≤ MAX_SLIP_P` (ValueError otherwise) and
`sensor_radius ≥ 0`.

### 3.2 `grid.py`
```python
Pos = tuple[int, int]
class Cell(IntEnum): UNKNOWN = 0; OPEN = 1; DEBRIS = 2; SURVIVOR = 3
def fmt_pos(p: Pos) -> str            # (0,0) -> "(1,1)"
def manhattan(a: Pos, b: Pos) -> int
def in_bounds(p: Pos, rows: int, cols: int) -> bool
def neighbours4(p: Pos, rows: int, cols: int) -> list[Pos]   # in-grid, order N, E, S, W
def footprint(p: Pos, radius: int, rows: int, cols: int) -> list[Pos]  # Manhattan ≤ r, incl. p, row-major

ASCII: '.' open, '#' debris, 'S' start (open), 'P' survivor, '~' open + high slip.
Lines are stripped; blank lines and lines starting with ';' are ignored. All rows equal
length; exactly one 'S' and one 'P'; any other char → ValueError with line/col (1-indexed).

@dataclass
class TrueGrid:
    cells: np.ndarray        # int8 (rows, cols): OPEN / DEBRIS / SURVIVOR (never UNKNOWN)
    start: Pos
    survivor: Pos
    high_slip: np.ndarray    # bool (rows, cols)
    name: str = ""
    rows, cols, shape (properties)
    def in_bounds(p) -> bool;  def is_debris(p) -> bool  (False off-grid)
    def passable_mask() -> np.ndarray[bool]        # cells != DEBRIS
    def copy() -> TrueGrid                        # deep copy
    def add_debris(p) -> None                     # ValueError on start/survivor... (world checks)
    def to_ascii() -> str ; def save(path) ; @classmethod from_ascii(text, name="") ; @classmethod load(path)
```
`from_ascii(to_ascii(g))` round-trips exactly (start cell written as 'S' even though OPEN).
*Clarification (P1):* `TrueGrid` rejects (ValueError) any grid that could not round-trip: start
must be OPEN, survivor the only SURVIVOR cell, start ≠ survivor, and neither may be high-slip
(mapgen must not mark them `~`).

### 3.3 `model.py`
```python
class Action(Enum):  N=(-1,0); E=(0,1); S=(1,0); W=(0,-1)
    delta -> tuple[int,int]; left -> Action; right -> Action; __str__ -> "N"
    # left/right relative to heading: N:(W,E)  E:(N,S)  S:(E,W)  W:(S,N)
ACTIONS = (Action.N, Action.E, Action.S, Action.W)
def step_pos(p: Pos, a: Action) -> Pos
class OutcomeKind(Enum): ARRIVED="arrived"; SLIPPED="slipped"; SKID_L="skid-L"; SKID_R="skid-R"
OUTCOME_ORDER = (ARRIVED, SLIPPED, SKID_L, SKID_R)
def kind_direction(kind, action) -> Action | None        # ARRIVED→a, SLIPPED→None, SKID_L→a.left …
def outcome_distribution(p: float) -> tuple[tuple[OutcomeKind, float], ...]
    # (ARRIVED, 1-p), (SLIPPED, SLIP_STAY_FRACTION*p), (SKID_L, SKID_FRACTION*p/2), (SKID_R, same)
class TransitionModel(ABC):
    @abstractmethod def slip_p(self, pos: Pos) -> float
    def distribution(self, pos) -> tuple[tuple[OutcomeKind, float], ...]
    def support(self, pos) -> tuple[OutcomeKind, ...]     # kinds with prob > PROB_EPS, OUTCOME_ORDER
    def sample_kind(self, pos, u: float) -> OutcomeKind   # inverse CDF; u in [0,1)
    @abstractmethod def max_p(self) -> float
class UniformSlip(TransitionModel):  __init__(p)
class TerrainSlip(TransitionModel):  __init__(base_p, high_p, high_mask: np.ndarray)
@dataclass(frozen=True)
class Outcome:  state: Pos; labels: tuple[OutcomeKind, ...]   # labels in OUTCOME_ORDER
class ProblemModel:
    """Planner's model on a frozen passability mask (unknown assumed open)."""
    def __init__(self, passable: np.ndarray, goal: Pos, transition: TransitionModel)
    passable (read-only copy), goal, rows, cols
    def is_goal(s) -> bool; def is_free(p) -> bool   # in bounds and passable
    def actions(s) -> tuple[Action, ...]            # ACTIONS
    def results(s, a) -> tuple[Outcome, ...]
        # for each kind in transition.support(s): target = step_pos(s, dir) (or s for SLIPPED);
        # if not is_free(target): target = s.  Merge equal targets (labels concatenated in
        # OUTCOME_ORDER); order outcomes by their first label. Cached per (s, a).
    def states() -> list[Pos]                        # all free cells, row-major
    @classmethod def from_belief(cls, belief: BeliefMap, transition) -> ProblemModel
```
With `p = 0`, `results` has exactly one outcome (deterministic).

### 3.4 `belief.py`
```python
@dataclass(frozen=True)
class BeliefUpdate: newly_known: tuple[Pos, ...]; contradictions: tuple[Pos, ...]
class BeliefMap:
    def __init__(self, rows, cols, survivor: Pos)   # all UNKNOWN, survivor cell = SURVIVOR
    grid: np.ndarray (int8)  rows, cols, survivor
    def get(p) -> Cell ; def is_known(p) -> bool
    def update(observations: Mapping[Pos, Cell] | Iterable[tuple[Pos, Cell]]) -> BeliefUpdate
        # writes true values; newly_known = was UNKNOWN; contradiction per D8
    def passable_mask() -> np.ndarray[bool]          # optimistic: grid != DEBRIS
    def explored_count() -> int                      # non-UNKNOWN cells
    def copy() -> BeliefMap
```
*Clarification (P1):* `explored_count` includes the survivor cell (known from t = 0, so it
starts at 1); `update`/`get` raise ValueError on off-grid positions, `update` also on UNKNOWN.

### 3.5 `world.py`
```python
@dataclass(frozen=True)
class Percept: cells: tuple[tuple[Pos, Cell], ...];  def as_dict() -> dict[Pos, Cell]
@dataclass(frozen=True)
class StepResult:
    prev: Pos; action: Action; kind: OutcomeKind; target: Pos  # cell the outcome tried to enter (prev for SLIPPED)
    new_pos: Pos; blocked: bool          # tried to move but stayed (wall or debris)
    hit_debris: bool                     # target was true DEBRIS
    percept: Percept; aftershock_cells: tuple[Pos, ...]
class RngStreams:
    def __init__(self, seed: int)   # np.random.SeedSequence(seed).spawn(3) → outcome, aftershock, misc
    outcome: np.random.Generator; aftershock: np.random.Generator; misc: np.random.Generator
class World:
    def __init__(self, grid: TrueGrid, transition: TransitionModel, *, sensor_radius=1,
                 seed=0, aftershock: AftershockConfig = AftershockConfig())
    grid (private mutable copy), pos, t, aftershock_events
    def sense() -> Percept                 # footprint(pos, r) with TRUE cell values
    def step(action) -> StepResult         # D3, D9; bumps into debris add the bumped cell to the percept
```
*Clarification (P1):* aftershock candidates are OPEN cells outside `footprint(new_pos, r)` (the
post-move position) and never the start cell (`add_debris` refuses it); fewer than `k` cells are
placed if too few qualify; `aftershock_events` counts steps that placed ≥ 1 cell. With
aftershocks disabled (`not cfg.enabled`) no aftershock draws are made — the stream is private,
so CRN is unaffected. A SLIPPED step has `target == prev`, `blocked=False`; an off-grid target
gives `blocked=True, hit_debris=False`.

### 3.6 `plan.py`
```python
@dataclass(frozen=True) class Goal: state: Pos
@dataclass(frozen=True) class Loop: state: Pos      # back-edge to an ancestor (self = "retry")
@dataclass(frozen=True) class Ref:  state: Pos      # subplan solved elsewhere in this plan
@dataclass(frozen=True) class Branch: state: Pos; labels: tuple[OutcomeKind, ...]; node: PlanNode
@dataclass(frozen=True) class Act:  state: Pos; action: Action; branches: tuple[Branch, ...]
PlanNode = Goal | Loop | Ref | Act
@dataclass(frozen=True) class PlanStats: nodes: int; act_nodes: int; depth: int; loops: int; refs: int; envelope_size: int
@dataclass(frozen=True)
class Plan:
    root: PlanNode; start: Pos; goal: Pos
    def to_policy() -> dict[Pos, Action]          # every Act node's state → action
    def envelope() -> frozenset[Pos]              # every state appearing anywhere (incl. goal)
    def pretty() -> str                           # deck one-liner for the root, see below
    def pretty_node(node, parent_state) -> str    # used by UI
    def to_tree(depth: int) -> dict               # nested {"state","kind","action","label","children"}
    def stats() -> PlanStats
def branch_label(parent: Pos, br: Branch) -> str  # "arrived" | "slipped" | "blocked" | "skid→(r,c)"
def plan_from_policy(policy, model, start) -> Plan   # unfold a policy into Act/Loop/Ref (fallback)
```
Pretty format, root only: `"{a} → {{{label}: {target}, ...}}"` with branch labels
(`branch_label`): stay-branch → `"slipped"` (or `"blocked"` if it has no SLIPPED label);
else ARRIVED in labels → `"arrived"`; else `"skid→(r,c)"`. Targets: `Goal` → `"goal"`;
`Loop(parent)` → `"retry {a}"`; `Loop(other)` → `"↺(r,c)"`; `Act(b)` → `"{b}-plan"`;
`Ref(t)` → `"{policy[t]}-plan"`. Separator `", "`.

### 3.7 Planners
```python
# bfs.py
UNREACHABLE = -1
def bfs_distances(passable, goal) -> np.ndarray       # int32, UNREACHABLE where no path / blocked
def shortest_path(passable, start, goal) -> list[Pos] | None       # inclusive endpoints
def shortest_path_length(passable, start, goal) -> int | None
def reachable(passable, start, goal) -> bool
# astar.py
@dataclass(frozen=True) class AStarResult: path: list[Pos] | None; nodes_expanded: int
def astar(passable, start, goal) -> AStarResult   # Manhattan h, ties on f broken by larger g, then FIFO
# andor.py
@dataclass(frozen=True)
class PlannerStats: nodes_expanded: int; or_nodes: int; and_nodes: int; rollbacks: int
                    time_ms: float; plan_size: int; fallback_used: bool; validated: bool | None
@dataclass(frozen=True)
class PlanResult: plan: Plan | None; stats: PlannerStats;  success (property)
def and_or_search(model: ProblemModel, start: Pos, config=PlannerConfig()) -> PlanResult
class AndOrPlanner:  def __init__(self, config=PlannerConfig()); def plan(model, start) -> PlanResult
    # runs and_or_search; if config.validate: validate; if failed or invalid and
    # fallback_to_reference and the reference finds a plan → plan_from_policy(reference) with
    # fallback_used=True.
# validate.py
@dataclass(frozen=True) class ValidationReport: ok: bool; reachable: frozenset[Pos]; missing: frozenset[Pos]; dead: frozenset[Pos]
def validate_policy(policy, model, start) -> ValidationReport
def is_strong_cyclic(policy, model, start) -> bool
# strong_cyclic_fixpoint.py
def strong_cyclic_policy(model, start) -> dict[Pos, Action] | None
def winning_region(model) -> frozenset[Pos]
```

### 3.8 Agents (`agents/base.py`)
```python
@dataclass(frozen=True) class Observation: pos: Pos; t: int
@dataclass(frozen=True) class Abort: reason: Literal["aborted", "unreachable"]; detail: str
Decision = Action | Abort
@dataclass class AgentStats:  replans=0; plans_made=0; planning_ms=0.0; nodes_expanded=0
                              plan_size=0; max_plan_size=0; unplanned_states=0
                              planner_fallbacks=0; contradictions=0
@dataclass(frozen=True) class AgentUpdate: newly_known; contradictions; replanned: bool; unplanned: bool; bumped_cells
@dataclass(frozen=True) class PlanSnapshot: plan_id: int; t: int; pos: Pos; plan: Plan | None; path: tuple[Pos, ...] | None; envelope: frozenset[Pos]
class Agent(ABC):
    name: ClassVar[str]
    def __init__(self, transition: TransitionModel, planner_config=PlannerConfig(), agent_config=AgentConfig())
    belief: BeliefMap; pos: Pos; goal: Pos; stats: AgentStats; snapshots: list[PlanSnapshot]
    current_plan_id: int  (-1 before first plan)
    def reset(self, start: Pos, goal: Pos, belief: BeliefMap) -> None
    def on_outcome(self, new_pos: Pos, percept: Percept) -> AgentUpdate   # also called at t=0 with start
    def choose_action(self, obs: Observation) -> Decision
    def anticipated(self, pos: Pos, action: Action) -> frozenset[Pos]   # D10
    def policy_arrows() -> dict[Pos, Action]                          # for the UI
```
`make_agent(name, transition, planner_config, agent_config)` in `agents/__init__.py`.

### 3.9 Runner
```python
class MissionStatus(Enum): RUNNING, RESCUED, ABORTED, BUDGET, UNREACHABLE
@dataclass(frozen=True)
class StepEvent:
    t; pos; action: Action | None; outcome: OutcomeKind | None; new_pos; blocked; bump
    new_cells: tuple[tuple[Pos, Cell], ...]; contradictions: tuple[Pos, ...]
    replanned: bool; unplanned: bool; plan_id: int; aftershock_cells: tuple[Pos, ...]
    status: MissionStatus; note: str
@dataclass(frozen=True)
class MissionMetrics: agent, scenario, seed, rows, cols, slip_p, success, failure_reason (str|None),
    steps, shortest_path, wasted_moves (float, NaN if failed), replans, slips, skids, bumps,
    blocked_moves, unplanned_states, cells_explored, planning_ms, nodes_expanded, plan_size,
    planner_fallbacks, aftershock_events
class Mission:
    def __init__(self, grid, agent, world_config=WorldConfig(), seed=0, scenario="")
    def start() -> StepEvent ; def step() -> StepEvent ; done ; status ; events ; world ; agent
    def metrics() -> MissionMetrics
def run_mission(grid, agent, world_config, seed, scenario="", on_event=None) -> Mission
def make_transition(grid, world_config) -> TransitionModel  # TerrainSlip iff grid has '~'
def format_trace(events, max_segments=None) -> str
def format_summary(metrics) -> str
```
Trace: segments `"{fmt(pos)}{note} {action} {label}"` joined by `" → "`, ending with the final
position; `label ∈ {ok, slip, skid-L, skid-R}` with `" (blocked)"` appended to blocked moves;
`note = " [debris at (r,c), replanned]"` (comma-joined cells) on the position where a
contradiction triggered a replan. Summary:
```
Cells explored: 11   Replans: 2   Slips handled: 3   Collisions: 0
Rescue successful            | Rescue failed (<reason>)
```

### 3.10 Behavioural details

**Plan tree (`Plan.to_tree(depth)`)** — nested dicts with keys `state` (Pos), `kind`
(`"act" | "goal" | "loop" | "ref" | "truncated"`), `action` (str or None), `edge` (branch label
from the parent, None at the root), `text` (display string, 1-indexed, e.g. `"E @ (1,1)"`,
`"goal (4,4)"`, `"↺ retry E"`, `"↺ (1,1)"`, `"→ S-plan @ (2,1)"`) and `children` (list). Act
nodes deeper than `depth` become `kind="truncated"` with no children. `PlanStats.depth` =
max number of Act nodes on a root-to-leaf path; `nodes` counts every node incl. leaves.

**ContingencyAgent (`andor`).** `on_outcome(new_pos, percept)`: update belief; if a previous
action exists and `new_pos ∉ anticipated(prev, a)` → `unplanned_states += 1`, log a warning.
Replan from `new_pos` (unless it is the goal) when: no plan yet (initial plan, not a replan),
or unplanned, or `new_pos` has no policy entry, or there are contradictions and
(`eager` or any contradicted cell ∈ plan envelope). The planning model is
`ProblemModel.from_belief(belief, transition)` snapshotted at planning time; `anticipated`
uses that snapshot's `results`. A failed plan → `choose_action` returns
`Abort("unreachable", …)`. Stats accumulate planning ms, nodes expanded, plan size,
fallbacks.

**FixedAStarAgent (`astar_fixed`).** Plans one A\* path on the optimistic belief at t=0.
`choose_action` returns `Abort("aborted", …)` if the current position ≠ the expected path cell
(first deviation; also counted as one unplanned state) or the next path cell is known DEBRIS;
`Abort("unreachable", …)` if A\* found no path.

**ReplanAStarAgent (`astar_replan`).** Same as above but instead of aborting it re-runs A\*
from the current position whenever the outcome deviates from the expected cell (counted as an
unplanned state), the next cell is known debris, or there are contradictions and (`eager` or a
contradicted cell is on the remaining path). `anticipated(pos, a)` = {expected next cell}.

**Mission loop.** `start()`: build `World(grid, make_transition(grid, cfg), sensor_radius,
seed, aftershock)`, a fresh `BeliefMap(rows, cols, survivor)`, `agent.reset(start, survivor,
belief)`, `agent.on_outcome(start, world.sense())`, emit the t=0 event (`action=None`).
`step()`: `decision = agent.choose_action(Observation(pos, t))`; an `Abort` ends the mission
(ABORTED / UNREACHABLE). Otherwise `result = world.step(action)`; `bump = result.hit_debris and
agent.belief.get(result.target) != DEBRIS` (evaluated **before** `on_outcome`); then
`agent.on_outcome(result.new_pos, result.percept)`; status RESCUED when on the survivor,
BUDGET when `world.t ≥ budget`. `blocked_moves` counts blocked outcomes that are not bumps.
Map seeds and world seeds are independent (`World` uses spawned child streams).

**Demo (`python -m rescue.demo`).** Prints a header, the initial root plan (`Plan.pretty()`),
the full trace and the summary for `maps/deck_demo.txt` with `config.DEMO_SEED`, chosen so the
trace starts `(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned]` and
ends in a rescue.

---

## 4. AND-OR algorithm sketch (see docs/algorithm.md for the proof)

```
AND_OR_SEARCH(model, start):
    solved = {}            # state -> action  (graph search; transactional, see below)
    journal = []           # undo log of solved/grounded insertions
    return OR(start, path=[], depth=0)

OR(s, path, depth):
    if goal(s): return Goal(s)
    if s in solved: return Ref(s)
    if s in path: return Loop(s)
    if depth > depth_limit or bfs_dist(s) = ∞: return FAIL   # dead-state pruning
    for a in ORDERED_ACTIONS(s):                # D13
        mark = len(journal)
        branches = AND(Results(s, a), path + [s], depth + 1)
        if branches != FAIL:
            solved[s] = a                       # tentatively
            if HAS_PROGRESS(s, path):           # semantic check, below
                return Act(s, a, branches)
        rollback(journal, mark)                 # undo every solved entry made in this attempt
    return FAIL

AND(outcomes, path, depth):
    for o in outcomes: plan_o = OR(o.state, path, depth); if plan_o == FAIL: return FAIL
    return branches

HAS_PROGRESS(s, path):  in the provisional policy graph (solved table + s),
    s can reach the goal, a grounded state, or a proper ancestor (a state on `path`).
    States that reach the goal are cached as grounded (also journaled).
```
`HAS_PROGRESS` generalises the prompt's syntactic rule (“not only Loop labels”) to the AIMA
condition “a leaf is reachable from every point in the plan”. The syntactic rule is both
unsound (two states looping into each other both look like progress) and incomplete (a
dead-end alcove whose only exit is back to its parent is rejected). The wall-bump regression
(action whose only outcome is “stay”) is rejected by both. Rollback fixes the other classic
caching bug: a state solved with a Loop to an ancestor that later fails.

---

## 5. Test plan (per phase)

* **P1** `test_config`, `test_grid` (ASCII round trip, parse errors, neighbours/footprint),
  `test_model` (χ² over 100k draws at p = 0.2 vs 0.8/0.15/0.025/0.025; walls/debris collapse
  to stay with merged labels; p = 0 → single outcome; left/right table; TerrainSlip),
  `test_belief` (update, contradiction cases incl. OPEN→DEBRIS, explored count),
  `test_world` (sensor footprint exactly 4-neighbours + self at r=1, r=2 diamond, step
  determinism by seed, CRN: two worlds with the same seed draw identical u's, bump reveals cell,
  aftershocks never on robot/survivor/footprint and never disconnect).
* **P2** `test_plan` (pretty/to_policy/stats/to_tree), `test_andor` (deck_demo root pretty,
  p=0 reduces to shortest path, wall-bump regression, alcove completeness regression, circular
  pending regression, unreachable → FAIL, rollback), `test_validate`, `test_fixpoint`,
  `test_bfs_astar` (A\* length == BFS length on random maps), `test_properties` (hypothesis,
  ≥ 500 maps up to 8×8, random p: AND-OR succeeds ⇔ reference succeeds; every AND-OR plan
  passes `is_strong_cyclic`).
* **P3** `test_agents`, `test_runner` (trace/summary format, determinism: same seed ⇒ identical
  events), `test_invariants` (1,000 missions: andor `unplanned_states == 0`, `bumps == 0`;
  r = 0 shows bumps/unplanned are real), `test_demo`.
* **P4** headless viz smoke test (100 frames, screenshot), replay round trip, recorder.
* **P5** `test_mapgen` (5 scenarios × 100 seeds solvable), aftershock invariants, compare app
  smoke test.
* **P6** `test_experiments` (quick config end to end on a tiny matrix, CSV schema, Wilson CI,
  hypotheses table rendering).
* **P7** coverage ≥ 85 % on `src/rescue` excluding `viz/`.
