# Project brief (original specification)

> This is the specification the project was built from. `docs/plan.md` records how each
> ambiguity was resolved; where the two differ, `docs/plan.md` explains why.

## 0. What we're building

The **Rescue Robot Navigation System** is the AI mini project for **CS23533 Foundations of
Artificial Intelligence** at Rajalakshmi Engineering College, Chennai (team: Srithan Saravanan
240701532, Subash R 240701538).

A rescue robot has to reach a survivor in a disaster-zone grid. It has two problems:

1. **Its actions are nondeterministic.** A move can arrive, slip and stay in place, or skid sideways.
2. **The map is partially known.** Cells are open, debris or unknown, and the robot only sees nearby cells.

A fixed A* path breaks the first time something goes wrong, so the agent uses **AND-OR graph
search** (Russell & Norvig, *AIMA* Ch. 4.3) to build a **cyclic contingency plan**. That plan
has an action for every possible outcome, and repeated slips become "retry" loops. The planner
runs inside a **plan → act → observe → update belief → replan** loop. On top of the agent we
build a **simulation suite**: an interactive Pygame simulator, head-to-head agent comparison,
scenario and stress simulations, and a Monte Carlo experiment harness that produces the
report's results.

## 1. How to work

- Phase 0: implementation plan (file tree, interfaces, algorithm sketch, test plan).
- At the end of each phase: run `pytest -q` and `ruff check .`, fix every failure, summarise,
  and commit (`feat(phaseN): …`).
- Python ≥ 3.11, type hints, `dataclasses` (frozen where possible), small modules, docstrings
  on public functions, no magic numbers in logic (they belong in `config.py`).
- Everything has to be **deterministic given a seed**.
- Be honest in results. If a hypothesis from the deck turns out false (for example, "fewer
  wasted moves than A* + replan"), report it and explain why. Never tune the experiment so the
  proposed agent wins.
- `CLAUDE.md` records conventions, commands and design decisions.

## 2. Problem formulation

### 2.1 World
- **R × C grid**, 0-indexed `(row, col)` internally; **everything shown to the user (traces,
  CLI, UI labels) is 1-indexed `(row, col)`**.
- True cell types `OPEN`, `DEBRIS`, `SURVIVOR` (survivor cells count as open). `START` is just
  the robot's initial position.
- **The survivor's location is known**; the map is not.
- Actions `N, S, E, W`.

### 2.2 Nondeterministic transition model (true world / simulator)
With slip parameter `p ∈ [0, 0.4]`: intended move `1 − p`; slip (stay) `0.75·p`; skid (move
perpendicular) `0.25·p`, split equally between the two perpendicular directions. At `p = 0.2`
this gives 0.80 / 0.15 / 0.05 (0.025 per side).
- If an outcome would put the robot into debris or off the grid, the robot stays in place. A
  move into debris counts as a **bump** (the deck calls these collisions).
- Pluggable `TransitionModel` interface so terrain-dependent slip can override `p` per cell.

### 2.3 Planner's model: `Results(s, a)`
- During a planning call the belief map is frozen, so **the planning state is the robot position**.
- `Results(s, a)` returns the **set** of possible next positions {intended, stay, perp-left,
  perp-right}, computed on the **belief map** with `UNKNOWN` **assumed open**; blocked or
  off-grid outcomes collapse to `s`.
- An outcome is only included if its probability is above 0 (with `p = 0` AND-OR reduces to
  ordinary search; test it).
- Each outcome carries a **label** (`arrived`, `slipped`, `skid-L`, `skid-R`). Outcomes that
  land on the same cell merge into one branch keeping both labels.

### 2.4 Sensing and belief
- After every step, including t = 0, the robot observes the true contents of its
  **4-neighbours** (Manhattan radius 1; configurable r = 1 default, r = 2 for experiments).
- Belief map: NumPy int8 array with `UNKNOWN / OPEN / DEBRIS / SURVIVOR`. "Cells explored" =
  count of non-UNKNOWN cells.
- With r ≥ 1 every possible outcome cell is sensed before acting, so **bumps never happen**
  and **unplanned states are 0 by construction** — invariants to assert in tests.

### 2.5 Replanning trigger
- A newly sensed cell that **contradicts the belief** (assumed open while unknown, turns out
  debris) triggers a replan from the current position.
- Modes: `eager` (any contradiction, default) and `lazy` (only if the contradicted cell is in
  the current plan's reachable envelope). Record replans.
- If the observed outcome state isn't covered by the current plan: increment
  `unplanned_states`, log a warning, replan (safety net; should never happen).

### 2.6 Mission termination
Success (on survivor) · Failure: unreachable (no strong-cyclic plan on current belief) ·
Failure: budget (default `4·R·C` steps) · Failure: aborted (baseline-specific).

### 2.7 Deck inconsistencies
Slides 1–2 show debris at (2,1), survivor at (3,4); slide 9's plan says `skid→(2,1): N-plan`
and the trace ends at (4,4). `maps/deck_demo.txt` reproduces the slide 9 output shape: start
(1,1), survivor (4,4), debris at (3,2), (2,1) open, (3,2) not visible until the robot reaches
(2,2). Root plan: `E → {arrived: S-plan, slipped: retry E, skid→(2,1): N-plan}` (the N-skid at
(1,1) merges into "slipped" because it hits the wall). `maps/deck_figure.txt` copies the slide
1 figure.

## 3. Core algorithm: AND-OR graph search with cyclic plans (`planners/andor.py`)

AIMA Fig. 4.11 plus the cyclic extension (AIMA §4.3.3, Hansen & Zilberstein solutions with loops).

```
AND_OR_SEARCH(problem) -> Plan | FAIL
    solved = {}                          # global: state -> action (makes it a GRAPH search)
    return OR_SEARCH(problem.initial, path=[], depth=0)

OR_SEARCH(s, path, depth)
    if goal(s):            return Goal
    if s in solved:        return Ref(s)
    if s in path:          return Loop(s)         # cyclic label = "retry", not FAIL
    if depth > depth_limit:return FAIL
    for a in ORDERED_ACTIONS(s):                  # order by BFS distance-to-goal on belief map
        branches = AND_SEARCH(Results(s,a), path+[s], depth+1)
        if branches != FAIL and HAS_PROGRESS(branches):
            solved[s] = a
            return Act(a, branches)
    return FAIL

AND_SEARCH(states, path, depth)
    for s_i in states:
        plan_i = OR_SEARCH(s_i, path, depth)
        if plan_i == FAIL: return FAIL
    return {s_i: plan_i}

HAS_PROGRESS(branches) := at least one branch is Goal, Ref, or Act (not only Loop labels)
```

Strong-cyclic: under fair nondeterminism the robot reaches the goal with probability 1
(argument in `docs/algorithm.md`).

- `plan.py`: `Goal`, `Loop(state)`, `Ref(state)`, `Act(action, branches, labels)`; `to_policy()`,
  `pretty()` (deck one-liner), `to_tree(depth)`, `stats()` (node count, depth, loops, envelope).
- `planners/validate.py`: `is_strong_cyclic(policy, model, goal)` — policy defined on every
  reachable state and every reachable state can reach the goal (reverse BFS).
- `planners/strong_cyclic_fixpoint.py`: fixpoint strong-cyclic planner (Cimatti, Pistore,
  Roveri & Traverso 2003), for completeness tests: AND-OR finds a plan iff the reference does.
  Last-resort fallback tracked as `planner_fallbacks`.
- Instrumentation: nodes expanded, OR/AND counts, planning time, plan size.
- Complexity note: ≤ R·C states; measure on 20×20 and 30×30.

## 4. Agent architecture

```
Environment (simulator)                      Rescue agent
├─ TrueGrid (open/debris/survivor)           ├─ BeliefMap (open/blocked/unknown)
├─ OutcomeSampler (pluggable TransitionModel)├─ ProblemModel: actions, Results(s,a), goal
├─ Sensor (radius r)                         ├─ Planner (AND-OR) → contingency plan
└─ Event hooks (aftershocks, etc.)           └─ Executor: act · observe · update · replan
            ── actions ──▶            ◀── percepts ──
Outputs: plan tree · execution trace · metrics · SUCCESS/FAILED
```

- Common agent interface: `reset(start, goal, belief)`, `choose_action(obs) -> Action`,
  `on_outcome(new_pos, percept)`, `stats`.
- `runner.py`: one loop for CLI, UI and experiments; each step emits a `StepEvent`.
- Trace: `(1,1) E ok → (1,2) S slip → (1,2) S ok → (2,2) [debris at (3,2), replanned] → … → (4,4)`
- Summary: `Cells explored: 11   Replans: 2   Slips handled: 3   Collisions: 0` then
  `Rescue successful` / `Rescue failed (<reason>)`.

## 5. Agents to compare

| ID | Agent | Behaviour |
|---|---|---|
| `andor` | Contingency agent | AND-OR cyclic plan; follows the branch matching the outcome; replans only on map contradictions. |
| `astar_fixed` | Fixed A* | Plans once on the optimistic belief, executes open-loop. **Aborts** the first time actual ≠ expected state, or when the next cell is sensed as debris. |
| `astar_replan` | A* + replan | Replans whenever the outcome deviates **or** the map is contradicted. |
| `dstar_lite` (stretch) | D* Lite | Koenig & Likhachev 2002, same trigger as `astar_replan`. |
| `vi_expected` (stretch) | Expected-cost optimal | Value iteration on the belief map with true probabilities. |

A* uses Manhattan heuristic with tie-breaking on larger g, tested for optimality against BFS.
All agents share sensor, belief update, optimistic-unknown assumption and step budget.

## 6. Simulation suite

### 6.1 Interactive simulator — `python -m rescue.sim`
Pygame (pygame-ce), window ≥ 1280×800. Left: grid with fog of war ("?" on unknown), `T`
reveals true map at 35% opacity, policy arrows (`P`), executed trace coloured by outcome
(arrived green, slipped amber ring, skid red arrow), replan flash. Right panel: collapsible
depth-limited plan tree (loop/ref labels as back-edges or "↺ retry", current node
highlighted), live metrics (steps, replans, slips, skids, cells explored, planning ms, plan
size), scrolling event log. Controls: `Space` play/pause · `N` step · `R` reset same seed · `S`
new seed · `+/−` speed · `[ ]` slip p · `1/2/3` agent · `M` scenario · `E` map editor (click
toggles debris; shift-click start; ctrl-click survivor) · `Ctrl+S` save map · `G` record
GIF/MP4 · `H` help. Accessibility: glyphs/shapes besides colour, high-contrast palette.

### 6.2 Head-to-head comparison — `python -m rescue.compare`
Three panes (`andor` | `astar_fixed` | `astar_replan`), same true map and same outcome stream
(common random numbers), lock-step, status badges RUNNING / RESCUED / ABORTED / BUDGET.

### 6.3 Scenario library — `rescue/mapgen.py` + `maps/`
Generators `(rows, cols, seed, **params)` guarantee solvability on the true map (BFS check,
regenerate, capped attempts): `random_rubble` (density 0.15–0.35), `collapsed_building`
(rooms, corridors, collapsed doorways), `flooded_zone` (high-slip regions), `maze_corridors`
(long dead-ends), plus `deck_demo` / `deck_figure`. ASCII: `.` open, `#` debris, `S` start,
`P` survivor, `~` high-slip; no `?` in true-map files.

### 6.4 Stress simulations (off by default)
Aftershocks (prob `q` per step, `k` new debris at random unsensed cells; never on robot,
survivor, or disconnecting the survivor; track `aftershock_events`); terrain-dependent slip;
sensor range sweep r = 1 vs 2; slip sweep p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

### 6.5 Replay and recording
`runs/<timestamp>_<agent>_<seed>.json` (map, config, seed, StepEvents, plan snapshots);
`python -m rescue.replay <file>` with scrubbing (`←/→`); plan trees exported as PNG
(matplotlib).

### 6.6 Monte Carlo harness — `python -m rescue.experiments --config experiments/main.yaml`
`multiprocessing`; agents × scenarios × p × sizes (10, 15, 20) × N = 200 seeded maps, common
random numbers. Metrics per run: success, failure reason, steps, wasted moves (steps − BFS
oracle shortest path), replans, slips, skids, bumps, unplanned_states, cells explored,
planning ms, nodes expanded, plan size, planner_fallbacks. Outputs `results/raw.csv`,
`results/summary.csv` (mean ± 95% CI; Wilson for success), plots (success % vs p; steps and
wasted moves vs p; replans vs p; cells explored vs p; planning time vs grid size; success under
aftershocks), `results/HYPOTHESES.md` (✅/❌/⚠️: ≥ 95% success at p = 0.3; 0 unplanned
states; fewer wasted moves than A* + replan). `--quick` (N = 20, one size) < 1 minute.

### 6.7 Stretch: browser demo with `pygbag`.

## 7. Repository layout
See `docs/plan.md` §1. `python -m rescue.demo` prints the slide 9 sample output for
`deck_demo` with a fixed, recorded seed (trace with a slip, a replan at (2,2), a rescue).

## 8. Optional extensions (after §3–6 are green)
LAO*/value iteration; terrain-aware cost; noisy sensing; unknown survivor location (frontier
exploration); multiple survivors.

## 9. Phases and acceptance criteria

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | grid, belief, model, world, ASCII I/O, RNG streams, config | χ² test over 100k samples matches 0.8/0.15/0.025/0.025 at p = 0.2; wall/debris collapse to stay; p = 0 deterministic; sensor reveals exactly 4-neighbours; contradiction detection. |
| 2 | AND-OR, plan structures, validator, fixpoint reference, BFS, A* | deck_demo root plan pretty-prints in the deck's format. Property tests (hypothesis, ≥ 500 random maps up to 8×8, random p): AND-OR finds a plan ⇔ reference does; every AND-OR plan passes `is_strong_cyclic`. A* matches BFS optimal length. Wall-bump "plan" regression test. |
| 3 | Agents, runner, metrics, trace formatting, `rescue.demo` | Demo reproduces slide 9 shape. Over 1,000 missions: `unplanned_states == 0`, `bumps == 0` for andor. `astar_fixed` aborts on first deviation. Same seed ⇒ identical trace. |
| 4 | Interactive simulator + replay + recording | ≥ 30 FPS on 30×30; every control works; headless smoke test renders 100 frames and saves a screenshot; GIF of deck_demo in `docs/figures/`. |
| 5 | Comparison app, scenario library, stress simulations | All 5 scenarios solvable for 100 seeds each; synced panes; aftershocks never make the survivor unreachable. |
| 6 | Experiment harness, plots, HYPOTHESES.md | `--quick` < 60 s; full run completes; `docs/results.md` interprets results honestly. |
| 7 | Docs and polish | README, `docs/algorithm.md`, `docs/architecture.md`; coverage ≥ 85% on `src/rescue` excluding `viz/`; ruff clean. |

## 10. Definition of done
`pip install -e .[dev] && pytest -q` passes on a fresh clone; `python -m rescue.demo`,
`python -m rescue.sim --map maps/deck_demo.txt`, `python -m rescue.compare --scenario
collapsed_building --seed 7` and `python -m rescue.experiments --quick` all work. Every number
in README and `docs/results.md` comes from a committed CSV generated by the harness.
