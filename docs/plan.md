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
| D11 | "Slips handled" in the summary = slips + skids (non-`arrived` outcomes) the agent survived: all of them except a final one the agent answered by giving up (`Abort("aborted")` as its next decision, i.e. `astar_fixed`'s first deviation). Recorded as `MissionMetrics.slips_handled`; metrics also keep the raw `slips` and `skids`. | Matches deck wording; the slip that made a baseline abort was not handled. |
| D12 | `Replans` counts re-planning calls **after** the initial plan. | Deck semantics. |
| D13 | Action iteration order `ACTIONS = (N, E, S, W)`. AND-OR `ORDERED_ACTIONS(s)` sorts by `(bfs_dist(intended target), diag(intended target), index in ACTIONS)` where a blocked intended target counts as `s`, and `diag(p) = abs(abs(goal.r-p.r) - abs(goal.c-p.c))` (stay near the straight line to the goal, like A\*'s cross-product tie-break); `PlannerConfig.tie_break = "fixed"` drops the `diag` key. The A\* baselines break f-ties by larger g, then FIFO (`PlannerConfig.astar_tie_break = "g"`, the default); `astar_tie_break = "diagonal"` inserts the same `diag(node)` key after `-g` (§3.7), so the two planners can be compared with **approximately matched** tie-breaking (the `tiebreak_ablation` experiment, H7, D21). The match is approximate: A\* still orders by Manhattan `f` and larger `g` before `diag` and rebuilds its path from first-generator parent pointers, while AND-OR orders forward by BFS distance on the belief, then `diag`; near belief obstacles the two can pick different equally short routes (H7 reports the share of matched pairs with identical steps and cells explored as a manipulation check; quick run: about 4–9 % of matched pairs still differ). | Deterministic and reproduces the deck's root plan (E at (1,1), S at (1,2)). |
| D14 | Deck root plan: the planner produces `E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}`. The deck's `N-plan` at (2,1) **cannot** come from a distance-ordered planner on that map: going N from (2,1) is never strictly better than going E, because `d(2,2) ≤ 1 + min(d(1,2), d(2,1)) = d(1,1)`. At t=0 the belief is optimistic, so `d(2,2)=4 < d(1,1)=6`. We reproduce the shape and document the one-letter deviation rather than hard-coding it. | Honesty over cosmetics. |
| D15 | Step budget default `4·R·C`. Depth limit default `R·C` (= no effective cut-off since path states are distinct). A strong-cyclic plan reaches the goal with probability 1 only in the limit, so **budget failures of `andor` are possible**, with probability ≈ `p^budget` per mission at worst (e.g. 1×2 corridor, p = 0.4, r = 1: budget 8, ≈ 6.6·10⁻⁴; seed 5582 fails). Negligible on the ≥ 5×5 maps of the tests and experiments; no minimum budget is added, and results must count such failures honestly. | Prompt §2.6, §3. |
| D16 | Wasted moves = `steps − shortest true-map path length`, recorded only for successful missions (NaN otherwise). | Undefined on failure. |
| D17 | Venv lives outside OneDrive (`%USERPROFILE%/.venvs/rescue-robot`); README documents a normal `pip install -e .[dev]`. | Avoid syncing ~200 MB of site-packages. |
| D18 | Map generators draw from `default_rng(SeedSequence([seed, MAPGEN_STREAM_KEY]))`, a stream independent of the World's `SeedSequence(seed).spawn(3)`, so a map depends only on `(scenario, rows, cols, seed, params)` and `sim` / `compare` can use one seed for map and world without correlating them. Start and survivor are at least `min_separation = ceil(MAPGEN_MIN_SEPARATION_FRACTION·(R+C−2))` apart (Manhattan); the requirement is relaxed linearly to 1 only after `MAPGEN_RELAX_START_FRACTION` of `MAPGEN_MAX_ATTEMPTS` failed attempts (never reached in the test sweeps). | Seeded, reproducible maps whose missions are not trivially short. |
| D19 | The stretch agents (`dstar_lite`, `vi_expected`) are **not** in `AGENT_NAMES` (the UI's `1/2/3` keys, the default comparison panes and brief §5's three compared agents); `ALL_AGENT_NAMES = AGENT_NAMES + STRETCH_AGENT_NAMES` are all buildable by `make_agent` and selectable with `compare --agents`. `vi_expected` reuses `andor`'s replan triggers and D10 anticipation (so 0 unplanned states is an invariant for it too); `dstar_lite` reuses `astar_replan`'s. Their `nodes_expanded` mean different things (VI: Bellman backups = sweeps × proper-region size; D\* Lite: vertices expanded by the incremental call) and are not comparable with AND-OR / A\* expansions. | Keeps the brief's three-way comparison intact while making the stretch agents first-class. |
| D20 | Comparison panes: every pane's `World` gets the same seed, so the outcome uniform `u_t` is identical in every pane (D3); the outcome **kind** is identical only between panes whose robots act from cells with the same slip p (`TerrainSlip`, D4). On maps without `~` every pane draws the same kind at every `t`. The aftershock draws `(u, permutation)` are shared the same way, but the debris lands only on cells outside each robot's own sensor footprint and keeps the survivor reachable from that robot (D9), so with `q > 0` the panes' true maps stay identical only while their robots stand in the same places (a pane that finished stops receiving debris); the footer then says so (`CRN_AFTERSHOCK_TEXT`). One lock-step tick = one `Mission.step()` per still-running mission, so every running pane has `t == ticks`; a finished pane keeps its own final `t`, and an agent's final Abort takes a tick but no world step, so the footer's `tick n` can exceed every pane's `t`. | Precise meaning of "same outcome stream" and "lock-step" for brief §6.2. |
| D21 | Tie-break ablation (H7). An experiment block may set `planner:` (a mapping of `PlannerConfig` overrides, validated; e.g. `{tie_break: fixed, astar_tie_break: diagonal}`) that applies to every agent of the block. Its canonical text (`key=value` sorted by key, joined by `;`; `''` = defaults) is the `planner` column of `raw.csv` (right after `replan_mode`), a summary grouping key and part of `stats.TASK_KEYS` (a **condition**, never part of `MAP_KEYS`: the task seed ignores it). `tiebreak_ablation` repeats `slip_sweep`'s scenarios × sizes × p × n_maps for `andor` and `astar_replan` with both tie-breaks swapped, so each ablation run pairs 1:1 (same map, same outcome stream) with a slip-sweep run; H7 pairs them across blocks on (scenario, rows, cols, p, r, replan mode, q, replicate, seed) with the map as the unit: (a) andor diagonal − astar_replan diagonal, (b) andor fixed − astar_replan larger-g, plus each tie-break's own effect. ✅ iff the CIs of (a) and (b) are both below 0 over p > 0; ❌ if either is above 0, **or** if both lie entirely inside ±`EQUIVALENCE_MARGIN_MOVES` (0.5 moves, `config.py`, fixed a priori; TOST-style equivalence: no practically relevant advantage once tie-breaking is matched); ⚠️ otherwise (a CI contains 0 but is too wide to show equivalence: underpowered, not refuted, as for every other paired claim). The ✅ test comes first. H7 prints both matched CI half-widths, the reason for its verdict, and a manipulation check (share of matched pairs with identical steps and cells explored at p = 0 and p > 0; the matching is approximate, D13). | At p = 0 nothing slips except on high-slip `~` terrain, which keeps its own slip parameter (`DEFAULT_TERRAIN_HIGH_P` = 0.4) at every block p and occurs only in `flooded_zone` (full run: 510/600 `andor` runs there slipped or skidded at p = 0, 0 elsewhere); on the other scenarios AND-OR has no contingency to plan for and reduces to greedy shortest-route search on the optimistic belief, yet it already wasted fewer moves than A\* + replan in the full run (H3's p = 0 row: −2.12 [−2.68, −1.56] moves overall and −2.74 [−3.48, −2.00] without `flooded_zone`, the clean no-contingency baseline that H3 also reports). So part of H3's gap is path choice (tie-breaking), which must be separated from the contingency-planning effect before H3 is reported. |

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
                          astar_tie_break: Literal["g", "diagonal"] = "g"   # A* agents (D13, §3.7)
@dataclass(frozen=True)
class AgentConfig:        replan_mode: Literal["eager", "lazy"] = "eager"
def load_yaml(path) -> dict
def world_config_from_dict(d) -> WorldConfig   (and planner/agent equivalents; unknown keys → ValueError)
```
`WorldConfig.__post_init__` validates `0 ≤ slip_p ≤ MAX_SLIP_P` (ValueError otherwise) and
`sensor_radius ≥ 0`. `PlannerConfig.__post_init__` validates `tie_break ∈ TIE_BREAKS` and
`astar_tie_break ∈ ASTAR_TIE_BREAKS` (ValueError otherwise); `astar_tie_break` is read by the
A\* agents only (`astar_fixed`, `astar_replan`; D\* Lite and AND-OR ignore it).

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
    def to_tree(depth: int) -> dict               # nested {"state","kind","action","edge","text","children"} (§3.10)
    def stats() -> PlanStats
def branch_label(parent: Pos, br: Branch) -> str  # "arrived" | "slipped" | "blocked" | "skid→(r,c)"
def plan_from_policy(policy, model, start) -> Plan   # unfold a policy into Act/Loop/Ref (fallback)
```
Pretty format, root only: `"{a} → {{{label}: {target}, ...}}"` with branch labels
(`branch_label`): stay-branch → `"slipped"` (or `"blocked"` if it has no SLIPPED label);
else ARRIVED in labels → `"arrived"`; else `"skid→(r,c)"`. Targets: `Goal` → `"goal"`;
`Loop(parent)` → `"retry {a}"`; `Loop(other)` → `"↺(r,c)"`; `Act(b)` → `"{b}-plan"`;
`Ref(t)` → `"{policy[t]}-plan"`. Separator `", "`.
*Clarification (P2):* `to_tree` dicts carry the §3.10 keys; `label` is kept as an alias of
`edge`. `Plan` equality/hash are structural but computed iteratively (plans can be hundreds of
Act levels deep), and its repr is the short `Plan(start=…, goal=…, root='<pretty>')`. `Act` and
`Branch` likewise define iterative structural `==`/`hash`, a one-level repr, and pickle/deepcopy
through a flat pre-order encoding (no recursion anywhere, P2 review).

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
def astar(passable, start, goal, tie_break="g") -> AStarResult
    # Manhattan h. tie_break="g": heap key (f, -g, counter) — larger g, then FIFO (bit-identical
    # to the original search). tie_break="diagonal": (f, -g, diag(node), counter) with D13's
    # diag(p) = abs(abs(goal.r-p.r) - abs(goal.c-p.c)), smaller first. The heuristic is unchanged,
    # so both stay optimal. PathAgent passes PlannerConfig.astar_tie_break.
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
*Clarification (P2 review):* `AndOrPlanner` stats: `time_ms` covers the whole call (search,
validation, fallback); `plan_size` is the returned plan's number of states with an action (0 when
no plan is returned); `fallback_used` means the returned plan came from the reference;
`validated` is None when validation is off or no plan was produced, else True iff every plan
produced in the call (AND-OR's and, if used, the fallback's) passed `is_strong_cyclic`. With
`validate` an invalid plan is never returned. The fallback's reachability check runs only when
a fallback is considered. `AndOrSearch.run` is thread-safe (reference-counted recursion-limit
raise). `strong_cyclic_policy` picks, among progressing pairs, the action whose `arrived`
outcome is in the lowest layer (then any progressing pair, then N, E, S, W).
*Clarification (P2):* `ValidationReport.dead` ⊇ `missing` (a state with no action has no
successors, so it cannot reach the goal). A\* `nodes_expanded` counts nodes closed and expanded;
popping the goal ends the search uncounted, so `start == goal` and blocked/off-grid endpoints
give 0.

### 3.8 Agents (`agents/base.py`)
```python
@dataclass(frozen=True) class Observation: pos: Pos; t: int
@dataclass(frozen=True) class Abort: reason: Literal["aborted", "unreachable"]; detail: str
Decision = Action | Abort
@dataclass class AgentStats:  replans=0; plans_made=0; planning_ms=0.0; nodes_expanded=0
                              plan_size=0; max_plan_size=0; unplanned_states=0
                              planner_fallbacks=0; contradictions=0
@dataclass(frozen=True) class AgentUpdate: newly_known; contradictions; replanned: bool; unplanned: bool
                                          triggering_contradictions: tuple[Pos, ...] = ()   # bumps are computed by the runner
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
*Clarification (P3 review):* every replan happens in `on_outcome` and is reported in its
`AgentUpdate` (so `stats.replans == Σ update.replanned`); `choose_action` never plans — a state
`on_outcome` should have replanned is a logic error (`RuntimeError`). `choose_action` raises
`RuntimeError` before the first `on_outcome` and on the goal (the mission is over), and
`ValueError` when `obs.pos` is not the agent's position; `reset` raises `ValueError` when `goal`
is not `belief.survivor`. `triggering_contradictions` are the contradicted cells that demanded
the replan (all of them in `eager` mode; in `lazy` mode those in the plan envelope / on the
remaining A\* path), empty when there was no replan or it had other causes only (deviation,
unplanned state, missing policy entry).

### 3.9 Runner
```python
class MissionStatus(Enum): RUNNING, RESCUED, ABORTED, BUDGET, UNREACHABLE
@dataclass(frozen=True)
class StepEvent:
    t; pos; action: Action | None; outcome: OutcomeKind | None; new_pos; blocked; bump
    new_cells: tuple[tuple[Pos, Cell], ...]; contradictions: tuple[Pos, ...]
    triggering_contradictions: tuple[Pos, ...]          # from AgentUpdate
    replanned: bool; unplanned: bool; plan_id: int; aftershock_cells: tuple[Pos, ...]
    status: MissionStatus; note: str
@dataclass(frozen=True)
class MissionMetrics: agent, scenario, seed, rows, cols, slip_p, success, failure_reason (str|None),
    steps, shortest_path, wasted_moves (float, NaN if failed), replans, slips, skids,
    slips_handled (D11), bumps, blocked_moves, unplanned_states, cells_explored, planning_ms,
    nodes_expanded, plan_size, planner_fallbacks, aftershock_events
    def as_dict() -> dict ; @classmethod from_dict(d) -> MissionMetrics   # None floats → NaN
class Mission:
    def __init__(self, grid, agent, world_config=WorldConfig(), seed=0, scenario="")
    def start() -> StepEvent ; def step() -> StepEvent ; done ; status ; events ; world ; agent
    def metrics() -> MissionMetrics
def run_mission(grid, agent, world_config, seed, scenario="", on_event=None) -> Mission
def make_transition(grid, world_config) -> TransitionModel  # TerrainSlip iff grid has '~'
def check_agent_transition(agent, world_transition, rows, cols) -> None   # D5, ValueError
def format_trace(events, max_segments=None) -> str
def format_summary(metrics) -> str
```
Trace: segments `"{fmt(pos)}{note} {action} {label}"` joined by `" → "`, ending with the final
position; `label ∈ {ok, slip, skid-L, skid-R}` with `" (blocked)"` appended to blocked moves;
`note = " [debris at (r,c), replanned]"` (comma-joined cells) on the position where a
contradiction triggered a replan: the cells listed are the event's `triggering_contradictions`,
so a replan caused only by a deviation (or a contradiction lazy mode ignores) is not
annotated. Summary (`Slips handled` = `slips_handled`, D11):
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
*Clarification (P2):* Act depth counts the root Act as 1, so `to_tree(stats().depth)` is the
full tree and `to_tree(0)` truncates the root; a truncated node keeps its `action`.

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
`Mission.__init__` raises `ValueError` if `agent.transition` differs from
`make_transition(grid, cfg)` in the slip p of any cell (D5: the agent is given the world's
dynamics; a mismatch would silently break the andor invariants).
`step()`: `decision = agent.choose_action(Observation(pos, t))`; an `Abort` ends the mission
(ABORTED / UNREACHABLE). Otherwise `result = world.step(action)`; `bump = result.hit_debris and
agent.belief.get(result.target) != DEBRIS` (evaluated **before** `on_outcome`); then
`agent.on_outcome(result.new_pos, result.percept)`; status RESCUED when on the survivor,
BUDGET when `world.t ≥ budget`. `blocked_moves` counts blocked outcomes that are not bumps.
Map seeds and world seeds are independent (`World` uses spawned child streams).

**Run files (`runs.py`).** `save_run(mission, path=None, runs_dir="runs") -> Path` (RuntimeError
for an unstarted mission; default name `<YYYYmmdd-HHMMSS>_<agent>_<seed>.json`, suffixed
`-1`, `-2`, … instead of overwriting); `load_run(path) -> RunRecord` (ValueError for another
format version). `RunRecord`: `grid` (the *initial* true map), `agent`, `seed`, `scenario`,
the three configs, `events: list[StepEvent]`, `metrics: MissionMetrics` (NaN restored), and
`snapshots` as JSON dicts (`plan_id`, `t`, `pos`, `pretty`, `policy`, `path`, `envelope`,
`tree`). Events and metrics round-trip exactly.

**Demo (`python -m rescue.demo`).** Prints a header, the initial root plan (`Plan.pretty()`),
the full trace and the summary for `maps/deck_demo.txt` with `config.DEMO_SEED`, chosen so the
trace starts `(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned]` and
ends in a rescue.

### 3.11 Visualisation (`viz/`, Phase 4–5)

The UI only renders state and forwards key presses; all behaviour comes from `Mission` /
`StepEvent`s. Every module must work under `SDL_VIDEODRIVER=dummy`.

```python
# theme.py — all UI constants (sizes, speeds, alphas, durations) live here, not in logic
@dataclass(frozen=True) class Palette: background, panel_bg, text, text_dim, grid_line, unknown,
    unknown_glyph, open, debris, debris_glyph, high_slip, survivor, robot, start, arrow, envelope,
    trace_ok, trace_slip, trace_skid, replan_flash, highlight, badge: dict[MissionStatus, RGB],
    agent: dict[str, RGB]
DEFAULT_PALETTE, HIGH_CONTRAST_PALETTE
WINDOW_SIZE = (1280, 800); PANEL_WIDTH; TARGET_FPS = 60; TRUE_MAP_ALPHA = round(0.35 * 255)
SPEEDS = (0.5, 1, 2, 4, 8, 16, 32)  # steps per second; DEFAULT_SPEED_INDEX
REPLAN_FLASH_MS; LOG_CAPACITY; SLIP_P_STEP = 0.05
def load_fonts() -> Fonts      # SysFont with Unicode-capable fallbacks (Segoe UI, DejaVu Sans)

# grid_view.py
@dataclass(frozen=True) class TraceMark: pos; new_pos; action; outcome: OutcomeKind; blocked: bool
@dataclass class GridScene: rows; cols; belief (int8 array or None = show truth); true_cells;
    high_slip; robot; start; survivor; show_true: bool; show_policy: bool;
    policy: Mapping[Pos, Action]; envelope: frozenset[Pos]; trace: Sequence[TraceMark];
    flash_cells: Sequence[Pos]; flash_alpha: float (0..1); editor: bool
class GridView: __init__(rect, palette, fonts); cell_rect(p) -> Rect; pos_at(xy) -> Pos | None;
    draw(surface, scene)
    # glyphs, not just colour: unknown = dark + "?", debris = cross-hatch, '~' = wave lines,
    # survivor = cross, robot = filled circle, start = "S"; arrived = green segment,
    # slipped = amber ring, skid = red arrow (red × when blocked); policy = small arrows.

# tree_view.py
class TreeView: __init__(rect, palette, fonts); set_plan(plan | None, current: Pos | None,
    max_depth=6); draw(surface); handle_event(event) -> bool   # click collapses, wheel scrolls
    # rows: "E @ (1,1)", "arrived → S @ (1,2)", "slipped → ↺ retry E", "skid→(2,1) → E-plan"

# panel.py
class EventLog: add(text, kind="info"); lines
@dataclass class PanelInfo: title; agent; status: MissionStatus; seed; slip_p; scenario;
    metrics: Mapping[str, str]; speed: float; paused: bool; recording: bool
class Panel: __init__(rect, palette, fonts); draw(surface, info, log, tree: TreeView)
def draw_help_overlay(surface, palette, fonts, lines)
def draw_badge(surface, rect, status, palette, fonts)

# record.py
class Recorder: start(path, fps); add_frame(surface); stop() -> Path | None; active
    # GIF via imageio; MP4 via imageio-ffmpeg when the suffix is .mp4

# export.py
def export_plan_tree(plan, path, max_depth=5, title=None) -> Path   # matplotlib, no Graphviz

# sim_app.py / replay_app.py / compare_app.py
@dataclass(frozen=True) class SimOptions: map_path | None; scenario | None; rows; cols; seed;
    agent; slip_p; sensor_radius; replan_mode; aftershock_q; aftershock_k; high_contrast;
    headless; frames; screenshot | None; record | None; autoplay
class SimApp: __init__(options); run() -> int; handle_key(key, mods); tick(dt_ms); render()
class ReplayApp: __init__(record: RunRecord, ...); same loop; ←/→ scrub, Home/End, Space
class CompareApp: 2–5 Missions (default AGENT_NAMES) on the same grid + seed, stepped in
    lock-step (D20); per-pane GridView + badge (RUNNING / RESCUED / ABORTED / BUDGET /
    UNREACHABLE) — see §3.12
```

Controls are exactly docs/brief.md §6.1 plus `C` (high-contrast palette) and `Esc` (quit).
In the simulator `M` cycles `maps/*.txt`, then the `mapgen.GENERATED_SCENARIOS` generators.
Headless use: `python -m rescue.sim --headless --frames 100 --screenshot out.png`.
*Clarification (P5):* brief §6.2's badge list omits UNREACHABLE; a pane whose agent proves the
survivor unreachable on its belief shows it like the simulator does.

### 3.12 Scenario library and stretch agents (Phase 5)

**Scenario library (`mapgen.py`, brief §6.3, D18).**
```python
class MapGenerationError(RuntimeError)          # no solvable map in MAPGEN_MAX_ATTEMPTS attempts
@dataclass(frozen=True)
class ScenarioSpec: name: str; description: str; generator: Callable[..., TrueGrid]
                    fixed_map: str | None = None   # repo-relative file of the deck_* scenarios
SCENARIOS: dict[str, ScenarioSpec]   # random_rubble, collapsed_building, flooded_zone,
                                     # maze_corridors, deck_demo, deck_figure (this order)
GENERATED_SCENARIOS: tuple[str, ...] # the four with fixed_map None
def generate(name, rows, cols, seed, **params) -> TrueGrid   # ValueError if unknown; grid.name = name
def random_rubble(rows, cols, seed, density=DEFAULT_RUBBLE_DENSITY)          # RUBBLE_DENSITY_RANGE
def collapsed_building(rows, cols, seed, room_size=BUILDING_ROOM_SIZE,
                       collapse_fraction=BUILDING_COLLAPSE_FRACTION,
                       rubble_density=BUILDING_RUBBLE_DENSITY)
def flooded_zone(rows, cols, seed, water_fraction=FLOOD_WATER_FRACTION,
                 debris_density=FLOOD_DEBRIS_DENSITY)                        # '~' = high slip
def maze_corridors(rows, cols, seed, loop_fraction=MAZE_LOOP_FRACTION)
def min_separation(rows, cols) -> int        # unrelaxed start–survivor distance (D18)
def map_header(name, rows, cols, seed, params) -> str   # ';' lines: generator + exact command
def build_parser(); def main(argv=None) -> int
    # python -m rescue.mapgen NAME [--rows R --cols C --seed N --param KEY=VALUE ... --out PATH]
    # prints the ASCII map; --out also writes it with the map_header
```
Every generated map has `rows, cols ≥ MAPGEN_MIN_SIZE` (else `ValueError`, as for a negative
seed or a parameter out of range), start ≠ survivor, start OPEN, the survivor BFS-reachable on
the true map (else the attempt is regenerated from the same stream), and no `~` on the start,
the survivor or debris. `deck_*` load their file and ignore rows, cols and seed. Shapes:
`collapsed_building` = a `BUILDING_STREET_WIDTH` street (dropped on maps too small for it)
around rooms behind 1-cell walls with 1-cell doorways, interior **hallways** (brief §6.3
"corridors"): along each axis with room for a `BUILDING_HALL_WIDTH` band beside ≥ 2 rooms of
`BUILDING_MIN_ROOM_SIZE` (15×15 and larger; not 10×10), a hallway crosses the whole building
with probability `BUILDING_HALL_PROB`, and at least one whenever one fits; every room beside it
has a doorway onto it and both ends are entrances. Then `BUILDING_MERGE_FRACTION` fallen walls,
`collapse_fraction` blocked doorways (entrances included) and rubble away from doors (hallways
included); start in the street, survivor in a room (never a hallway); a cut-off survivor gets a
collapsed doorway on the start's component boundary re-opened until connected, preferring one
whose other side is the survivor's component (otherwise random). `flooded_zone` = the top
`water_fraction` of Gaussian-smoothed noise (computed on a padded grid, then cropped) is `~`,
sparse debris only on dry land; start and survivor are placed so that **water lies on every
shortest route** between them (an all-dry route is strictly longer or absent; up to
`FLOOD_CROSSING_START_TRIES` starts per attempt, dropped together with the separation after
`MAPGEN_RELAX_START_FRACTION` of the attempts and on maps without water). Before this rule an
all-dry route was as short as the shortest one on ~70 % of seeds, so the scenario mostly behaved
like sparse rubble. The per-map top-`water_fraction` threshold leaves a small border excess of
water (≈ 8 % relative at 15×15; a single global threshold would be flat but would let the water
fraction vary per map). Missions use `TerrainSlip` through `make_transition`. `maze_corridors` = recursive-backtracker perfect maze
on the even lattice (spare line on a random side for even sizes) with `loop_fraction` of inner
walls knocked out. Float parameters must be real numbers (bools and strings raise
`ValueError`, like non-int sizes). Sample maps (each regenerates byte-for-byte from its header
command; `--out` and `TrueGrid.save` always write LF line endings): `maps/rubble_01.txt` (15×15,
seed 0), `building_01.txt` (15×20, seed 0: a vertical hallway with both ends open),
`flooded_01.txt` (15×15, seed 2: a water channel between start and survivor),
`maze_01.txt` (15×15, seed 1).

**Stretch planners.**
```python
# planners/value_iteration.py — expected-steps optimal policy on the frozen belief snapshot
class TransitionTables(NamedTuple): states: list[Pos]; goal_index: int
                                    succ: np.ndarray; prob: np.ndarray   # (len(ACTIONS), n, 4)
def transition_tables(model: ProblemModel) -> TransitionTables   # true P(o|s,a), vectorised:
    # slot k = outcome kind OUTCOME_ORDER[k] (kinds landing on one cell keep separate slots =
    # the same backups as their merged outcome); zero-probability kinds → (goal, 0)
def proper_region(tables) -> np.ndarray      # bool: strong-cyclic winning region (greatest fixpoint)
class ValueIterationResult(NamedTuple): values: dict[Pos, float]; policy: dict[Pos, Action]
                                        iterations: int        # inf outside the proper region
def value_iteration(model) -> ValueIterationResult
    # V = 0 start, Bellman sweeps V(s) = min_a 1 + Σ P·V until max Δ < VI_TOLERANCE
    # (≤ VI_MAX_ITERS, warning otherwise); greedy = first action in ACTIONS order within
    # VI_TIE_TOLERANCE of the minimum
class ValueIterationPlanner:   # same interface as AndOrPlanner
    __init__(config=PlannerConfig()); plan(model, start) -> PlanResult
    last_values: dict[Pos, float]; last_iterations: int
    # plan = plan_from_policy(policy); stats.nodes_expanded = Bellman backups; plan_size =
    # policy states reachable from start; with config.validate a policy failing
    # is_strong_cyclic is replaced by the reference policy (fallback_to_reference)
# planners/dstar_lite.py — Koenig & Likhachev 2002, optimized version, backward search
@dataclass(frozen=True) class DStarResult: path: list[Pos] | None; nodes_expanded: int
class DStarLite:
    __init__(passable, start, goal); g(p); rhs(p); is_passable(p); km; total_expanded
    move_start(p); set_passable(changes) -> tuple[Pos, ...]; sync(passable) -> tuple[Pos, ...]
    compute_shortest_path() -> int; extract_path() -> list[Pos] | None   # greedy argmin c + g
    plan(passable, start) -> DStarResult                                 # all of the above
```
*Clarification (P5):* the optimized termination test can leave `s_start` overconsistent
(`g = ∞`, `rhs` finite); `extract_path` therefore tests reachability with `rhs(s_start)` (exact
at that point), never `g(s_start)`. Tests check every replan's path length against BFS.

**Stretch agents (D19).**
```python
# agents/__init__.py
AGENT_NAMES = ("andor", "astar_fixed", "astar_replan")   # unchanged
STRETCH_AGENT_NAMES = ("dstar_lite", "vi_expected")
ALL_AGENT_NAMES = AGENT_NAMES + STRETCH_AGENT_NAMES      # AGENTS / make_agent cover all five
class DStarLiteAgent(ReplanAStarAgent)       # id "dstar_lite": one DStarLite per mission; every
    dstar: DStarLite | None                  # replan syncs all belief changes since the last call
class ValueIterationAgent(ContingencyAgent)  # id "vi_expected": planner = ValueIterationPlanner
    vi_planner (alias of planner)            # on ProblemModel.from_belief, same triggers as andor
    expected_steps: float                    # VI value at the planning position; inf without a
                                             # plan, nan when the reference fallback is in force
# ContingencyAgent hooks: _make_planner() -> planner; _plan_made(result) after every plan
```

**Head-to-head comparison (`compare.py`, `viz/compare_app.py`, brief §6.2, D20).**
```python
# python -m rescue.compare [--scenario NAME | --map PATH] [--rows R --cols C] [--seed N]
#     [--agents a,b,...] [--p P] [--radius R] [--lazy] [--aftershock-q Q --aftershock-k K]
#     [--high-contrast] [--headless --frames F --screenshot PNG] [--record GIF|MP4] [--mp4]
#     [--autoplay] [--speed INDEX]          # headless prints one status line per pane
def parse_agents(text | Sequence[str] | None) -> tuple[str, ...]   # 2–5 distinct ALL_AGENT_NAMES
@dataclass(frozen=True) class CompareOptions: map_path; scenario; rows; cols; seed; agents;
    slip_p; sensor_radius; replan_mode; aftershock_q; aftershock_k; high_contrast; headless;
    frames; screenshot; record; autoplay; mp4; speed_index; recordings_dir
@dataclass class ComparePane: agent; mission; view; trace; last_event; status; t (properties)
class CompareApp(BaseApp): panes; missions; ticks; all_done; reset_missions(); step_once();
    toggle_play(); new_seed(); next_scenario(); change_slip(delta); handle_key(key, mods);
    render() -> Surface; run() -> int
def scenario_entries(map_path) -> list[ScenarioEntry]   # M cycle: --map file first, then SCENARIOS
def outcome_rows(app) -> list[tuple[int, tuple[str | None, ...]]]   # per t, kind per pane (CRN)
def pane_event_line(event) -> tuple[str, str] | None   # the pane's line: most important of
    # event_messages (failure/success > warn > replan > slip/skid > info); an Abort's note is
    # folded into the failure line
# layout: 1..COMPARE_MAX_PANE_ROWS rows of panes, whichever gives the largest cells (ties: fewer
# rows; two rows only pay off for wide maps); one agent-name font shared by all panes
def agent_colour(palette, agent) -> RGB   # palette.agent, else COMPARE_AGENT_COLOURS
```
Keys: `Space`, `N` (one tick), `R`, `S` (seed + 1; regenerates a generated scenario, a map file
stays), `+/−`, `[ ]`, `M`, `T`, `P`, `C`, `G`, `H`, `Esc`. The figure
`docs/figures/compare_collapsed_building.png` comes from
`python -m rescue.compare --headless --frames 250 --screenshot docs/figures/compare_collapsed_building.png`.

**Constants (`config.py`, Phase 5).** Value iteration: `VI_TOLERANCE = 1e-10`,
`VI_MAX_ITERS = 100_000`, `VI_TIE_TOLERANCE = 1e-7`. Map generation: `MAPGEN_STREAM_KEY`,
`MAPGEN_MIN_SIZE = 5`, `MAPGEN_MIN_SEPARATION_FRACTION = 0.5`,
`MAPGEN_RELAX_START_FRACTION = 0.5`, `MAPGEN_DEFAULT_SIZE = (15, 15)`, `MAPGEN_DEFAULT_SEED = 0`
(CLI); `BUILDING_ROOM_SIZE = 4`, `BUILDING_MIN_ROOM_SIZE = 2`, `BUILDING_ROOM_JITTER = 1`,
`BUILDING_COLLAPSE_FRACTION = 0.3`, `BUILDING_RUBBLE_DENSITY = 0.08`,
`BUILDING_MERGE_FRACTION = 0.1`, `BUILDING_STREET_WIDTH = 1`, `BUILDING_ENTRANCE_PROB = 0.15`,
`BUILDING_MIN_ENTRANCES = 2`; `FLOOD_WATER_FRACTION = 0.25`, `FLOOD_DEBRIS_DENSITY = 0.06`,
`FLOOD_BLOB_SCALE = 0.12`, `FLOOD_MIN_SIGMA = 1.2`, `FLOOD_PAD_SIGMAS = 2.0`,
`FLOOD_WATER_RANGE = (0.0, 0.6)`, `FLOOD_DEBRIS_RANGE = (0.0, 0.3)`;
`MAZE_LOOP_FRACTION = 0.08`. Comparison: `COMPARE_DEFAULT_SCENARIO = "collapsed_building"`,
`COMPARE_DEFAULT_SEED = 7`, `COMPARE_DEFAULT_SIZE = (15, 15)`, `COMPARE_MIN_AGENTS = 2`,
`COMPARE_MAX_AGENTS = 5`, `COMPARE_DEFAULT_SPEED_INDEX = 4` (8 ticks/s, so a short headless
run gets somewhere), and the layout / colour constants `COMPARE_PANE_GAP`,
`COMPARE_PANE_PADDING`, `COMPARE_METRIC_COLUMNS`, `COMPARE_EVENT_LINES`,
`COMPARE_AGENT_COLOURS` (stretch-agent colours per palette), `COMPARE_MAX_PANE_ROWS = 2`.
Fix pass: `BUILDING_HALL_WIDTH = 1`, `BUILDING_HALL_PROB = 0.5`,
`FLOOD_CROSSING_START_TRIES = 12`. *Exception:* the layout / colour constants are
UI constants that §3.11 would put in `viz/theme.py`; they live in `config.py` with the other
comparison settings.

**Honest notes from the Phase 5 checks** (scratch sanity checks, superseded by the Phase 6
harness). On a frozen snapshot VI's exact expected steps are never worse than the AND-OR
plan's (property-tested), but in closed-loop missions `vi_expected` is not guaranteed to need
fewer steps than `andor`: each policy is optimal only for the optimistic belief it was planned
on, and the runs diverge after the first different action, so mission comparisons are not
paired (one 300-mission check had `vi_expected` slightly *more* steps on average). D\* Lite
expands fewer nodes than A\* from scratch but, in pure Python, spends more total planning time
than `astar_replan` because of its per-call overhead. `vi_expected` is the slowest planner: after
vectorising `transition_tables` (39 → 1.6 ms on a 30×30 open belief at p = 0.4; a plan 77 →
29 ms), each plan is still a full VI on the whole belief (tens of ms at 30×30, many replans
per mission), so the Phase 6 harness must budget for it.

`andor` vs `astar_replan` under CRN: both follow a shortest route on the optimistic belief
(with *different* tie-breaks among equally short routes — D13 diagonal vs larger-g/FIFO; the
Phase 6 ablation D21 / H7 measures what that alone contributes, with approximately matched
tie-breaks), so they take *identical*
trajectories until the contingency plan's
action first differs from A\*'s next move, and until then differ only in Replans, Unplanned and
Plan ms. Scratch check at 15×15, p = 0.2, seeds 0–39: identical whole trajectories on 5/40
`random_rubble`, 13/40 `collapsed_building`, 4/40 `flooded_zone` and 16/40 `maze_corridors`
maps (mean steps andor / astar_replan: 29.5 / 30.6, 53.2 / 57.8, 22.7 / 23.3, 65.9 / 61.2).
A "fewer wasted moves than A\* + replan" hypothesis can therefore come out as a tie or ❌ on
some scenarios; report it as measured, without tuning seeds or the default comparison
(`collapsed_building` 15×15 seed 7: 76 vs 81 steps).

Scenario context for cross-scenario numbers: `collapsed_building` always starts on the
perimeter street (≈ 40 % of a shortest route is street at 15×15), the others on any free cell;
mean BFS shortest paths at 15×15 over seeds 0–39: rubble 17.8, flooded 16.1, building 24.0,
maze 28.9.

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
condition “a leaf is reachable from every point in the plan”. With a transactional table the
literal syntactic rule is sound but incomplete (a dead-end alcove whose only exit is back to
its parent is rejected); relaxing it so a Loop to an ancestor counts as progress fixes the
alcove but is unsound (two states looping into each other). The wall-bump regression (action
whose only outcome is “stay”) is rejected by all three rules (docs/algorithm.md §5). Rollback fixes the other classic
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
  smoke test. *As built:* `test_mapgen` (4 generated scenarios × 100 seeds at 12×12, 10×17 and
  5×5 plus `deck_*`; determinism, structure, CLI, sample maps regenerate), `test_stress`
  (aftershocks never disconnect the survivor, terrain-slip frequencies, r = 1 vs 2, p sweep),
  `test_stretch_agents` (VI closed forms, Bellman optimality, VI ≤ AND-OR expected cost; D\* Lite
  = BFS over incremental replans; stretch missions), `test_compare` (lock-step, CRN, keys,
  headless CLI).
* **P6** `test_experiments` (quick config end to end on a tiny matrix, CSV schema, Wilson CI,
  hypotheses table rendering).
* **P7** coverage ≥ 85 % on `src/rescue` excluding `viz/`.
