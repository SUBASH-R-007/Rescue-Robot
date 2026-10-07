# Extensions: unknown survivor, several survivors, noisy sensing

This chapter covers the optional extensions of `docs/brief.md` §8 that were built after the core
system (§3–§6) was finished and measured. Brief §8 is a single list of five items; §8.1–§8.5
number them in order (LAO\*/value iteration, terrain-aware cost, noisy sensing, unknown survivor
location, multiple survivors), as `docs/plan.md` does:

* **§8.4 Unknown survivor location.** The robot does not know where the survivor is and finds
  it by frontier exploration.
* **§8.5 Multiple survivors.** The robot rescues k survivors in a greedy nearest-first order.
  The order is re-evaluated after every rescue and compared with the brute-force optimal tour.
* **§8.3 Noisy sensing.** The debris sensor misses debris (false negatives), and the robot
  keeps a Bayesian per-cell debris belief. A second round added **false positives** (free
  cells read as debris), so the noise is two-sided.

§1 says what each extension does and why. §2 gives the algorithms. §3 describes the hooks into
the core and the rule that the core results must not move. §4 is the experiment design, §5 the
results claim by claim, and §6 honest notes. §1–§6 describe the first round (claims X1–X5,
one-sided noise). §7 covers the second round's two-sided noise and claim X6, and §8 shows the
extensions in the interactive simulator. §9 lists the limitations and future work, and §10 is
the summary. The interface contract is `docs/plan.md` §3.14 (round 1) and §3.15 (round 2). The
core results are in `docs/results.md`, whose conventions this chapter follows.

Every number comes from one of four places:

* `results/ext/summary.csv`, which holds per-condition means with 95 % CIs;
* `results/ext/EXTENSIONS.md`, which holds the claim verdicts (X1–X6) and the paired tests;
* `results/ext/run_meta.json`, which describes the run;
* a reproducible computation on `results/ext/raw.csv` (once also on the core
  `results/raw.csv`, and once on the previously committed `results/ext/raw.csv`, read with
  `git show`).

Numbers of the last kind come with the one-line Python (standard library only) that produced
them, in an HTML comment beside them. Run it from the repository root. Design constants (the
prior, the threshold, the claim thresholds) are quoted from `src/rescue/config.py` by name.

> **Key findings**
>
> 1. **Frontier search always found the survivor.** `search_andor` rescued an unknown survivor
>    in 1600 of 1600 missions, with 0 bumps and 0 unplanned states (X1).
> 2. **Not knowing the location costs about 2.9 times the steps:** +76.27 steps
>    [+71.61, +80.93] per mission for `search_andor` (X2). Almost the whole mission is
>    exploration. With sensor radius 1 the survivor can only be seen from a neighbouring cell.
> 3. **In unknown-location search the planner hardly matters.** `search_andor` and
>    `search_astar` took exactly the same number of steps in 1525 of 1600 paired missions.
>    Following the nearest frontier dominates. AND-OR still costs about 284 times as much
>    planning time per mission (§6.1).
> 4. **Greedy nearest-first is close to the optimal visiting order:** 2.7 % longer on average
>    at k = 4 (×1.027 [1.022, 1.033]) and never more than ×1.444 on any map (X3).
> 5. **Noisy sensing makes collisions possible but not fatal.** There were 2719 bumps in 2400
>    `search_andor` runs at ε > 0 and still 100 % success. AND-OR is not measurably worse at
>    avoiding them than A\* + replan (−0.04 bumps per mission [−0.11, +0.03], X4, X5).
> 6. **False positives cost steps, not rescues.** With phantom walls added (false-positive
>    rate ε_fp up to 0.2, on top of ε = 0.1) both agents still rescued the survivor in every
>    mission (X6). At ε_fp = 0.1 `search_andor` needed +46.53 steps [+40.79, +52.27] more than
>    without false positives, and it lost its step advantage over A\* + replan (§7).

---

## 1. What was added and why

### 1.1 Unknown survivor location (§8.4): frontier exploration

The core robot knows where the survivor is and only has to find a way there through unknown
debris. In a real disaster the location is usually unknown. In this extension the belief starts
with **every cell UNKNOWN**, and the robot only knows *how many* survivors there are (a radio
count; §9 discusses dropping it). Until it senses a survivor, the robot explores.

Exploration is **frontier-based**. A *frontier cell* is a known, believed-passable cell with at
least one UNKNOWN 4-neighbour. The robot drives to the nearest frontier cell. Its sensor then
reveals the cells around it, which pushes the frontier outwards. Frontier exploration was
chosen because it fits the core loop without new machinery. A frontier cell is just another
goal for the core planners, so the AND-OR planner, its strong-cyclic guarantee and the
replanning triggers all carry over unchanged. And because the sensor has no false positives
(§2.2), it terminates. (False positives were added later as an option, `false_positive`,
default 0; every argument and result in §1–§6 is for the default. See §7.)

### 1.2 Multiple survivors (§8.5): greedy order, re-evaluated after each rescue

With k survivors the robot must also choose an **order**. Finding the best order is a
travelling-salesman-type problem. The robot uses the simplest reasonable rule: go to the
**nearest** known unrescued survivor (BFS distance on its current belief), and choose again
after each rescue. Distances change as the robot learns the map, so re-evaluating keeps the
rule honest about what it knows. The rule's quality is measured against an oracle: the
**optimal tour**, found by brute force over all k! visiting orders on the true map
(k ≤ `EXT_MAX_BRUTE_FORCE_K` = 6).

### 1.3 Noisy sensing (§8.3): false negatives and a Bayesian belief

The core sensor is perfect. Here, the sensor reports a true DEBRIS cell as OPEN with
probability ε (a **false negative**: dust or smoke hides the rubble). In round 1 it never
reports debris where there is none, and it never hides a survivor. (Round 2 added false
positives as an option, `false_positive`, default 0; they are covered in §7. Everything in
§1–§6 is one-sided.) Missed debris can now surprise the robot, so
**collisions (bumps, D6) become possible**. Under the core's perfect sensor with r ≥ 1 they are
impossible. A bump reveals the debris by touch.

The belief becomes a per-cell debris probability, `ProbBelief`, updated by Bayes' rule under
this sensor model (§2.4). A cell is treated as impassable once its probability reaches a
threshold.

---

## 2. Algorithms

### 2.1 The search agent and its target

`ext/agents.py` defines one agent class, `SearchAgent`, with two planners:

| Agent | Planner | Follows | Replans on |
|---|---|---|---|
| `search_andor` | AND-OR strong-cyclic plan to the current target (the core `andor`, with a moving goal) | the policy branch that matches each outcome | target change, contradiction (eager), unplanned state, no policy entry for the current cell |
| `search_astar` | A\* path to the current target (the core `astar_replan`, with a moving goal) | the path | target change, contradiction (eager), any deviation from the path, next path cell known debris |

The **target** is chosen at t = 0 and re-evaluated **only** when one of the following happens:

1. a survivor has just been rescued;
2. a survivor has been seen for the first time;
3. the current target is no longer a valid target. A rescue target must still be a known,
   unrescued survivor. An explore target must still be a frontier cell. Either kind must still
   be reachable on the optimistic belief and must not be the robot's own cell.

Re-evaluation then picks, in this order:

1. **rescue mode:** the nearest known, unrescued, reachable survivor by BFS distance on the
   agent's optimistic passability (unknown cells count as open; ties row-major). This is the
   greedy nearest-first order.
2. **explore mode:** otherwise, the nearest reachable frontier cell (ties row-major).
3. otherwise the agent aborts with `unreachable`. This never happened in the run (§5.1).

Only cells the belief holds as SURVIVOR are ever rescue targets. An unknown cell is never
assumed to hide a survivor. Between re-evaluations the target is fixed, so the agent does not
dither between equally near frontier cells.

### 2.2 Why exploration terminates

The argument is informal. It is written out in full in the `SearchAgent` docstring.

* **Between target changes.** The agent follows a strong-cyclic plan (or a replanned A\* path)
  to a fixed, reachable target. Under fair nondeterminism it gets there with probability 1,
  unless the belief changes first.
* **Each cause of a target change can happen only finitely often.**
  1. *The target stopped being a frontier cell.* With sensor radius r ≥ 1 this happens at the
     latest on arrival: standing on the target, the robot senses all of the target's
     4-neighbours, so at least one UNKNOWN cell becomes known. There are finitely many cells.
  2. *The target became impassable or unreachable.* Some cell became debris in the belief. That
     happens at most once per cell, because debris is never "un-seen": the sensor has no false
     positives, and in `ProbBelief` a probability of 1 stays 1 (§2.4).
  3. *A survivor was seen or rescued.* This happens at most twice per survivor.
* **Exploration cannot run out of frontier while a survivor is still hidden** (on a solvable
  map). Take a true path from the robot to an unfound survivor. Its cells are truly passable,
  so they are believed passable (no false positives). The survivor cell itself is still
  UNKNOWN. So the last known cell before the first UNKNOWN cell on that path is a reachable
  frontier cell.

So after finitely many target changes every survivor is rescued, with probability 1. The
argument relies on the sensor having no false positives (`false_positive = 0`, the default and
the setting of every run in §1–§6; with false positives it fails, see §7.4) and on unknown
cells being optimistic (prior below the threshold). For this reason `ExtMission` rejects `prior_debris ≥ threshold`,
and it rejects the categorical `BeliefMap` when ε > 0, because a noisy OPEN reading would
overwrite known debris there. The step budget is `4·R·C·k` (the core D15 budget times k).

### 2.3 Greedy order and the oracle tours

The agent applies the greedy rule on its **belief** while it moves. For the evaluation, each
mission also records two oracle tour lengths, both computed on the **initial true map** with
BFS distances:

* `greedy_tour`: the length of the nearest-first visiting order from the start (ties
  row-major);
* `optimal_tour`: the length of the shortest of all k! orders.

X3 compares these two. They measure the rule itself, not the agent's realised route, which also
pays for slips and for discovering debris. The realised route shows up as steps per rescue
(§5.3). For k ≤ 2 greedy is optimal by construction. Both orders share the survivor-to-survivor
leg, and greedy takes the shorter first leg. So only k ≥ 3 can test the rule.

Extra survivors come from `ext/scenarios.py`. `multi_survivor_grid` takes the core scenario map
(same start, same primary survivor, same debris) and adds k − 1 survivors on reachable, open,
non-mud cells. The cells are visited in one seeded permutation and must be pairwise
`EXT_MIN_SURVIVOR_SPACING` = 3 apart in Manhattan distance (relaxed one step at a time if no
cell qualifies; this never happened in the run: all 400 k = 4 maps keep spacing 3). Survivor i depends only on
survivors 1…i−1, so the survivors for k are a prefix of those for k + 1.

### 2.4 Noisy sensor and Bayes update

**Sensor (`NoisySensor`).** Each true DEBRIS cell in the percept is reported OPEN with
probability ε. OPEN and SURVIVOR cells are never changed (in round 1; §7.1 adds false
positives on OPEN cells). The cell a bump has just touched is
exempt, because touch is reliable. The sensor has its own stream,
`SeedSequence([seed, NOISE_STREAM_KEY])`, independent of the world's outcome and aftershock
streams.

**Belief (`ProbBelief`).** Let P be a cell's debris probability and e = ε. Every cell starts at
the prior P₀ = `EXT_PRIOR_DEBRIS`. On each observation:

| Observation | Update |
|---|---|
| reads OPEN | P ← e·P / (e·P + (1 − P)) |
| reads DEBRIS, or touched by a bump | P ← 1 |
| reads SURVIVOR | P ← 0, and the cell is a known survivor |

The OPEN rule is Bayes' rule with likelihoods P(OPEN | debris) = e and P(OPEN | free) = 1. An
OPEN reading never raises P, and P = 1 stays 1. A cell is **passable** while
P < `EXT_DEBRIS_THRESHOLD`. A *contradiction*, which triggers a replan, is a cell that goes from
passable to impassable.

**Prior and threshold.**

* `EXT_PRIOR_DEBRIS` = `DEFAULT_RUBBLE_DENSITY` = 0.25. This is the density of
  `random_rubble`, the one scenario whose debris is i.i.d., so the prior matches the base rate
  the robot would expect on unseen rubble.
* `EXT_DEBRIS_THRESHOLD` = 0.5, so a cell counts as debris once debris is at least as likely as not.
* Because the prior is below the threshold, unknown cells stay passable. This is the core's
  "unknown means open" assumption.
* Worked example at ε = 0.1: one OPEN reading takes a never-seen cell from 0.25 to
  0.1·0.25 / (0.025 + 0.75) ≈ 0.032. A second OPEN reading takes it to about 0.0033.

With ε = 0 an OPEN reading sets P to 0, and `ProbBelief` makes exactly the same passability
decisions as the core `BeliefMap`. In the experiments, ε = 0 runs use the `BeliefMap` itself,
and ε > 0 runs use `ProbBelief`.

**Noise pairing.** Every sensor call draws **one uniform per grid cell** (a fixed `rows × cols`
block, whatever the percept contains). Cell (r, c) uses the draw at index `r·cols + c` and is
missed when that draw is below ε. So the noise at a given (step, cell) is the same in every
mission on a map: for both agents, whatever path led there, and at every ε. Because "missed"
means u < ε, the misses are also **nested** across noise levels. A cell missed at ε = 0.05 at
step t is also missed at ε = 0.1 and ε = 0.2 at step t.

---

## 3. Core hooks and rule zero

**Rule zero: the core results must not move.** With default arguments every core module
behaves bit-identically to the pre-extension commit `c709053`. Two checks confirm it, run before
and after the change (`docs/plan.md` §3.14):

* `python -m rescue.demo` prints byte-identical output;
* `python -m rescue.experiments --quick` writes a `raw.csv` that is identical apart from the
  timing columns.

The extensions live in `src/rescue/ext/`. No core module imports them. They use the core only
through these backward-compatible hooks:

| Module | Hook | Default behaviour |
|---|---|---|
| `grid.py` | `TrueGrid.extra_survivors` (default `()`), `survivors`, `from_ascii` / `load(multi=True)` | one survivor, as before |
| `belief.py` | `BeliefMap(rows, cols, None)` (nothing pre-marked), `survivors_seen()` | survivor pre-marked, as before |
| `model.py` | `ProblemModel.from_belief` raises `ValueError` when the belief has no survivor | unchanged |
| `world.py` | aftershocks keep **every** survivor reachable and debris-free; `World.on_survivor` | single-survivor maps: same calls, same draws, same result |

`tests/test_ext_core_hooks.py` covers the hooks. `tests/test_ext.py` and
`tests/test_ext_experiments.py` cover the package.

The full run also gives an end-to-end check. The ext run uses the core harness's task seeds
with the same base seed (2026), so its known-location `search` missions replay core slip-sweep
missions: same 15×15 maps, same p, same outcome stream. With the survivor location known and
k = 1, `search_andor` reproduced the core `andor` run exactly (steps, cells explored, replans,
slips and skids) in **1600 of 1600** missions. `search_astar` reproduced `astar_replan` in
1600 of 1600.
<!-- python -c "import csv;C=[r for r in csv.DictReader(open('results/raw.csv')) if r['experiment']=='slip_sweep'];E=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='search' and r['survivors_known']=='True'];k=lambda r:(r['scenario'],r['rows'],r['slip_p'],r['replicate'],r['seed']);f=lambda r:(r['steps'],r['cells_explored'],r['replans'],r['slips'],r['skids']);[print(a,b,sum(f(r)==f(M[k(r)]) for r in E if r['agent']==b),sum(r['agent']==b for r in E)) for a,b in (('andor','search_andor'),('astar_replan','search_astar')) for M in [{k(r):r for r in C if r['agent']==a}]]" -->
So the search agents are the core agents with a target-selection layer on top. Any difference
in §5 comes from the extension, not from a re-implementation.

---

## 4. Experiment design

### 4.1 What was run

`python -m rescue.ext.experiments` ran `experiments/ext.yaml`. It has four blocks, all on
15×15 maps of the four generated scenarios (random_rubble, collapsed_building, flooded_zone,
maze_corridors), with sensor radius r = 1, eager replanning and no aftershocks. The first three
are round 1; `noisy_fp` was appended in round 2 (§7).

| Block | Brief | Slip p | Factor | Survivor locations | Agents | Maps per cell | Tasks | Runs |
|---|---|---|---|---|---|---|---|---|
| `search` | §8.4 | 0.1, 0.3 | location known vs unknown, k = 1 | known / unknown | `search_andor`, `search_astar` | 200 | 3,200 | 6,400 |
| `multi` | §8.5 | 0.2 | k ∈ {1, 2, 3, 4} | known | `search_andor` | 100 | 1,600 | 1,600 |
| `noisy` | §8.3 | 0.2 | false-negative rate ε ∈ {0, 0.05, 0.1, 0.2}, k = 1 | known | `search_andor`, `search_astar` | 200 | 3,200 | 6,400 |
| `noisy_fp` | §8.3 | 0.2 | false-positive rate ε_fp ∈ {0, 0.05, 0.1, 0.2} at ε = 0.1, k = 1 | known | `search_andor`, `search_astar` | 200 | 3,200 | 6,400 |

<!-- python -c "import csv,collections as c;print(sorted(c.Counter((r['experiment'],r['agent']) for r in csv.DictReader(open('results/ext/raw.csv'))).items()))" -->

From `results/ext/run_meta.json`:

| Item | Value |
|---|---|
| Config | `ext` (`experiments/ext.yaml`) |
| Base seed | 2026 |
| Tasks / runs | 11,200 / 20,800 |
| Workers | 18 processes |
| Wall time | 450.3 s (started 2026-10-07 22:14:31 +05:30, finished 22:22:01) |
| Git commit of the code | `cfb50a2` |
| Python / platform | 3.11.9 / Windows-10-10.0.26200-SP0 (Windows 11) |

**The round-2 rerun reproduced round 1 exactly.** The committed run above is a full rerun of
the whole config with the round-2 code. The first run (commit `fdb8af0`, code `762a944`, 8,000
tasks / 14,400 runs, 277.1 s) had only the first three blocks. Every one of its 14,400 rows
reappears in the rerun, in the same order, with every column equal except the two timing
columns (`planning_ms`, `elapsed_ms`). The rerun only appends two columns (`false_positive`
= 0 and `fallback_replans` = 0 on all those rows) and the 6,400 `noisy_fp` rows. So every
number in §1–§6 holds for both runs; the timing values quoted in §6.1 come from the rerun.
<!-- python -c "import csv,io,subprocess as p;O=list(csv.DictReader(io.StringIO(p.run(['git','show','fdb8af0:results/ext/raw.csv'],capture_output=True,text=True).stdout)));N=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']!='noisy_fp'];c=[k for k in O[0] if k not in('planning_ms','elapsed_ms')];print(len(O),len(N),sum(all(a[k]==b[k] for k in c) for a,b in zip(O,N)),{(r['false_positive'],r['fallback_replans']) for r in N})"
prints 14400 14400 14400 {('0.0', '0')} -->


### 4.2 Seeding and pairing

A task's seed is the core harness's `task_seed(base_seed, scenario, rows, cols, replicate)`.
The seed is the map seed, the World seed (outcome stream) and the `NoisySensor` seed. The
condition (known or unknown location, k, ε, agent) is **not** part of the seed. So:

* known and unknown location replay the same map and outcome stream;
* every k uses the same base map, and the survivors for k are a prefix of those for k + 1;
* every ε and both agents see the same outcome draws and the same per-(step, cell) noise draws
  (§2.4).

Comparisons are therefore **paired**. A map appears at several p values (X2) or several ε
(X5), and its differences there are correlated. So pooled paired tests use **the map as the
unit**: a map's differences are averaged first, and n = number of maps, not pairs.

### 4.3 Statistical methods and claim rules

The methods are those of the core harness (`docs/results.md` §1.3):

* success rates use Wilson 95 % intervals;
* means use Student-t 95 % intervals;
* paired differences use the mean of per-map differences with a Student-t CI, a paired t-test
  and a Wilcoxon signed-rank test;
* X3 uses one tour ratio per map with a Student-t CI of the mean.

Steps are averaged over successful missions. Every mission succeeded, so this makes no
difference here.

| # | Claim | ✅ supported if | ❌ refuted if |
|---|---|---|---|
| X1 | `search_andor` rescues an unknown survivor in ≥ 95 % of missions | rate ≥ `EXT_CLAIM_SUCCESS_TARGET` (0.95), the rule of the core H1 | the Wilson upper bound is < 0.95 |
| X2 | an unknown location costs extra steps (paired unknown − known) | the CI of the mean difference is > 0 | the CI is < 0 |
| X3 | greedy order within 10 % of the optimal tour | the CI upper bound of the mean ratio is ≤ `EXT_CLAIM_TOUR_RATIO_MAX` (1.10) for every k | some CI lower bound is > 1.10 |
| X4 | noise makes bumps possible, yet success stays ≥ 95 % at ε = `EXT_CLAIM_NOISE` (0.1) | bumps at ε > 0, none at ε = 0, and success ≥ 0.95 | no bumps at ε > 0, any bump at ε = 0 (a core invariant broken), or Wilson upper bound < 0.95 |
| X5 | non-inferiority: `search_andor` has at most 0.1 more bumps per mission than `search_astar` | every CI upper bound (pooled and per ε) is ≤ `EXT_CLAIM_BUMP_MARGIN` (0.1) | any CI lies above 0 |
| X6 (round 2) | with false positives up to ε_fp = `EXT_CLAIM_FP` (0.1), `search_andor` still rescues ≥ 95 % | success ≥ 0.95 at every level 0 < ε_fp ≤ 0.1 of block `noisy_fp` | the Wilson upper bound is < 0.95 at some level |

Anything else is ⚠️ inconclusive. The thresholds are constants in `config.py`. The X1–X5
thresholds were fixed before the first full run, in commit `762a944`, which that run used, and
none of the round-1 `EXT_*` constants has changed since. Round 2 only appended new constants
(`EXT_CLAIM_FP`, `EXT_FALLBACK_MAX_P`, `EXT_MAX_FALSE_POSITIVE`) in commit `cfb50a2`, before the
rerun that used it. §6.4 and §7.6 say which choices were made with data in view.

### 4.4 Reproducing

```bash
python -m rescue.ext.experiments            # full run: experiments/ext.yaml → results/ext/
python -m rescue.ext.experiments --quick    # experiments/ext_quick.yaml → results/ext/quick/ (git-ignored)
python -m rescue.ext.demo --mode search     # one mission, printed (also: multi, noisy)
```

The full run writes `raw.csv`, `summary.csv`, `EXTENSIONS.md`, `run_meta.json` and
`plots/*.png`. Two runs of one config give identical `raw.csv` files except for the timing
columns (`planning_ms`, `elapsed_ms`). The quick run uses the same factors on 10×10 maps with 10
maps per cell. Its output is not committed, so this chapter reports no quick-run numbers.

---

## 5. Results

### 5.1 Verdicts

The table below copies `results/ext/EXTENSIONS.md`. Its verdict symbols are written out as
words.

| # | Claim | Verdict | Measured |
|---|---|---|---|
| X1 | frontier search finds and rescues an unknown survivor in ≥ 95 % of missions (`search_andor`) | **supported** | 100.0% [99.8%, 100.0%] (1600/1600); Σ bumps = 0, Σ unplanned states = 0 |
| X2 | not knowing the survivor's location costs extra steps (paired unknown − known) | **supported** | `search_andor` +76.27 steps [+71.61, +80.93] over 800 maps, ×2.89; `search_astar` +72.47 [+67.79, +77.15], ×2.64 |
| X3 | the greedy nearest-first order is within 10 % of the optimal tour | **supported** | k = 3: ×1.011 [1.008, 1.015]; k = 4: ×1.027 [1.022, 1.033] (k ≤ 2: ×1.000 by construction) |
| X4 | with noisy sensing collisions become possible but success stays ≥ 95 % at ε = 0.1 | **supported** | Σ bumps = 2719 in 2400 runs at ε > 0 vs 0 in 800 runs at ε = 0; success at ε = 0.1: 100.0% [99.5%, 100.0%] (800/800) |
| X5 | under noise `search_andor` has at most 0.1 more collisions per mission than `search_astar` | **supported** (non-inferiority) | pooled −0.04 bumps per mission [−0.11, +0.03] over 800 maps; largest per-ε upper bound +0.06 at ε = 0.2 |
| X6 | with false positives up to ε_fp = 0.1 `search_andor` still rescues ≥ 95 % of missions | **supported** | ε_fp = 0.05 and 0.1: 100.0% [99.5%, 100.0%] (800/800 each); steps cost at ε_fp = 0.1 +46.53 [+40.79, +52.27] (informational, §7.5) |

All 14,400 missions of the round-1 blocks succeeded. The longest used 36 % of its step budget
(320 of 900 steps: `search_andor`, unknown location, random_rubble, p = 0.3). So the budget
never bound, and no agent ever aborted. The 6,400 `noisy_fp` missions of round 2 also all
succeeded; the longest used 65 % of its budget (583 of 900 steps: `search_astar`,
maze_corridors, ε_fp = 0.1).
<!-- python -c "import csv;R=list(csv.DictReader(open('results/ext/raw.csv')));f=lambda r:int(r['steps'])/(4*int(r['rows'])*int(r['cols'])*int(r['k']));[print(b,len(X),sum(r['success']=='True' for r in X),round(f(m),3),m['experiment'],m['agent'],m['scenario'],m['slip_p'],m['k'],m['survivors_known'],m['false_positive'],m['steps']) for b in (0,1) for X in [[r for r in R if (r['experiment']=='noisy_fp')==b]] for m in [max(X,key=f)]]" -->

### 5.2 X1 and X2: the cost of an unknown location

**X1.** `search_andor` rescued the unknown survivor in all 1600 missions (Wilson
[99.8 %, 100 %]): every scenario, both p. It had Σ bumps = 0 and Σ unplanned states = 0. So the
core invariants (r ≥ 1) survive exploration. `search_astar` also succeeded in all 1600
(`EXTENSIONS.md`, X1 table). As with the core H1, success does not separate the agents.

**X2.** Not knowing the location costs `search_andor` **+76.27 steps** [+71.61, +80.93] per
mission (800 maps, paired t p = 4.7e-146). Mean steps rise by ×2.89. The cost depends on the
scenario:

| Scenario | `search_andor` unknown − known (95 % CI) | ratio |
|---|---|---|
| random_rubble | +80.06 [+71.29, +88.83] | ×3.74 |
| collapsed_building | +67.78 [+60.23, +75.32] | ×2.33 |
| flooded_zone | +105.13 [+95.27, +114.99] | ×5.64 |
| maze_corridors | +52.10 [+42.54, +61.67] | ×1.89 |

Each row covers 200 maps (`EXTENSIONS.md`, X2). The ratio is largest where the known-location
mission is short. flooded_zone has the shortest known-location missions (the plot below), so
the same exploration effort multiplies its steps the most. maze_corridors, where the
known-location route is already long and winding, has the smallest ratio. The cost also grows
with slip: +67.19 [+63.04, +71.35] at p = 0.1 against +85.34 [+80.08, +90.60] at p = 0.3.

![Steps per mission with the survivor location known and unknown, per scenario, for search_andor and search_astar](../results/ext/plots/search_steps.png)

*Unpaired view, 200 maps per point, steps averaged per map first. The paired test is X2. In
every scenario the two agents' unknown-location points nearly coincide. The known-location
points are further apart: `search_astar` is higher, most clearly in collapsed_building (the core
H3/H7 tie-break effect).*

**Almost the whole unknown-location mission is search.** In both agents' unknown-location
missions, 98.9 % of all steps come before the survivor is first sensed. Only 1.33
(`search_andor`) and 1.32 (`search_astar`) steps per mission come after it. With r = 1 the
survivor can only be sensed from a neighbouring cell, so once it is seen it is one step away
(the extra third of a step is slips). Exploration also roughly doubles the map the robot has to look at: 136.9 cells explored
per mission (of 225) against 68.2 with the location known (`search_andor`).
<!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='search'];[print(a,z,round(s.mean(int(r['cells_explored']) for r in X),1),round(s.mean(int(r['replans']) for r in X),1),round(s.mean(float(r['planning_ms']) for r in X),1),round(s.mean(int(r['unplanned_states']) for r in X),2),round(s.mean(int(r['steps'])-float(r['first_detection_step']) for r in X),2),round(sum(float(r['first_detection_step']) for r in X)/sum(int(r['steps']) for r in X),3)) for a in ('search_andor','search_astar') for z in ('True','False') for X in [[r for r in R if r['agent']==a and r['survivors_known']==z]]]" -->

### 5.3 X3: greedy versus optimal visiting order

| k | maps | greedy / optimal (95 % CI) | max ratio | greedy is optimal | `search_andor` steps per rescue |
|---|---|---|---|---|---|
| 1 | 400 | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 41.02 [38.24, 43.81] |
| 2 | 400 | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 26.93 [25.24, 28.62] |
| 3 | 400 | 1.011 [1.008, 1.015] | 1.250 | 88.2% | 22.49 [21.20, 23.78] |
| 4 | 400 | 1.027 [1.022, 1.033] | 1.444 | 72.5% | 19.61 [18.60, 20.63] |

(`EXTENSIONS.md`, X3.) The greedy order was optimal on 88.2 % of maps at k = 3 and 72.5 % at
k = 4. On average it was 1.1 % and 2.7 % longer than the optimal tour. That is far inside the
10 % limit, so X3 is supported. The average hides a tail, though. Excess tour length averaged
0.45 cells at k = 3 and 1.25 cells at k = 4, but reached 12 and 17 cells on the worst maps
(ratios up to ×1.250 and ×1.444). The gap grows with k, as expected for a greedy tour.
<!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='multi'];[print(k,len(X),round(s.mean(int(r['steps'])/float(r['optimal_tour']) for r in X),2),round(s.mean(float(r['greedy_tour'])-float(r['optimal_tour']) for r in X),2),max(float(r['greedy_tour'])-float(r['optimal_tour']) for r in X),round(s.mean(float(r['planning_ms']) for r in X),1),sum(int(r['unplanned_states'])+int(r['bumps']) for r in X)) for k in '1234' for X in [[r for r in R if r['k']==k]]]" -->

The order is not where most of the steps go. `search_andor` rescued all k survivors in every
mission (400/400 per k). Its actual mission took 1.84 (k = 1) and 1.71–1.73 (k = 2–4) times the
optimal oracle tour. Those factors come mostly from slips at p = 0.2 and from detours around debris
discovered on the way. The order accounts for at most a few percent (X3); this split was not
measured directly. Steps per rescue fall with k (41.02 → 19.61), plausibly
because later survivors are reached from nearby cells rather than from the start. There were no bumps and no
unplanned states in any `multi` mission (same one-liner).

![Left: greedy / optimal tour ratio per k with the 1.1 claim limit; right: steps per rescue of search_andor and the optimal tour per survivor](../results/ext/plots/multi_tours.png)

*Left: one value per map, 400 maps per k. The dashed line is the claim limit ×1.1. Right: the
grey dashed series is the oracle optimal tour divided by k (true map, no slips). The gap to the
blue series is mostly the price of slips and of an initially unknown map; the order adds at most
a few percent (left).*

### 5.4 X4: noisy sensing makes collisions possible, not fatal

| ε | `search_andor` success | Σ bumps | missions bumped | bumps per mission | steps | Σ unplanned |
|---|---|---|---|---|---|---|
| 0 | 100.0% (800/800) | 0 | 0 | 0.00 | 39.48 [37.60, 41.36] | 0 |
| 0.05 | 100.0% (800/800) | 409 | 269 | 0.51 [0.45, 0.57] | 40.33 [38.37, 42.28] | 0 |
| 0.1 | 100.0% (800/800) | 760 | 385 | 0.95 [0.86, 1.04] | 40.88 [38.89, 42.86] | 0 |
| 0.2 | 100.0% (800/800) | 1550 | 500 | 1.94 [1.77, 2.10] | 42.07 [39.98, 44.15] | 0 |

(`EXTENSIONS.md`, X4.) Both parts of X4 hold:

* **Collisions.** There are no bumps at ε = 0, the core invariant. At ε > 0 there are 2719
  bumps in 2400 runs. The bump rate grows roughly in proportion to ε (0.51 → 0.95 → 1.94 per
  mission).
* **Success.** It stays at 100 % (Wilson [99.5 %, 100 %] at ε = 0.1). So does `search_astar`'s.

A bump costs little. The robot stays where it is, the touched cell becomes certain debris, and
the agent replans. Steps rise by 2.59 per mission from ε = 0 to ε = 0.2. Steps minus bumps rise
by only 0.65 (39.48 → 40.13). So about three quarters of the extra steps are the bumped moves
themselves, and the rest are detours.
<!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='noisy'];[print(a,e,round(s.mean(int(r['steps']) for r in X),2),round(s.mean(int(r['bumps']) for r in X),2),round(s.mean(int(r['steps'])-int(r['bumps']) for r in X),2),round(s.mean(int(r['replans']) for r in X),2)) for a in ('search_andor','search_astar') for e in ('0.0','0.05','0.1','0.2') for X in [[r for r in R if r['agent']==a and r['false_negative']==e]]]" -->

`search_andor` still has **0 unplanned states** under noise. A bump leaves the robot in place,
and "stay in place" (the slip outcome) is in every AND-OR `Results(s, a)` set (with p > 0, as
here, p = 0.2), so a bump is
never an outcome the plan did not cover. It is a bump by D6 (debris the agent did not believe
was there), but not a surprise for the plan. `search_astar`'s unplanned states rise from 7355
(ε = 0) to 9350 (ε = 0.2) in 800 runs, because for a path follower a bump is one more
deviation.

![Bumps per mission and rescue success against the false-negative rate, for both agents](../results/ext/plots/noisy.png)

*Left: all missions, Student-t 95 % CIs. Right: Wilson 95 % CIs. 800 missions per agent and
point. The two agents' bump lines overlap at every ε.*

### 5.5 X5: AND-OR is not worse at avoiding collisions

| ε | `search_andor` − `search_astar` bumps per mission (95 % CI) | Σ bumps andor / astar |
|---|---|---|
| 0.05 | −0.01 [−0.07, +0.05] | 409 / 417 |
| 0.1 | −0.05 [−0.13, +0.04] | 760 / 797 |
| 0.2 | −0.07 [−0.19, +0.06] | 1550 / 1602 |
| pooled (800 maps) | −0.04 [−0.11, +0.03], paired t p = 0.28, Wilcoxon p = 0.32 | 2719 / 2816 |

(`EXTENSIONS.md`, X5.) Every CI upper bound is at most +0.06, below the 0.1 margin, so the
non-inferiority claim is supported. ✅ here means "not worse by more than 0.1 bumps per
mission". It does **not** mean "fewer collisions". Every CI contains 0, and the point estimates
are small. Both agents face the same noise draws at a given (step, cell), and neither planner
uses the debris probabilities to avoid uncertain cells, so neither has a way to avoid debris it
believes is open. The beliefs themselves differ, because the agents walk different paths and
sense different cells (equal step counts in only 224–282 of 800 pairs per ε). The tendency
towards slightly fewer bumps for `search_andor` is not significant.

---

## 6. Honest notes

### 6.1 In unknown-location search the two planners behave almost identically

`search_andor` and `search_astar` took exactly the same number of steps in **1525 of 1600**
paired unknown-location missions. In 1514 of these they also explored the same cells. The
mean difference is −0.06 steps. With the location known, only 564 of 1600 pairs tie, and
`search_andor` is 3.85 steps shorter on average.
<!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='search'];k=lambda r:(r['scenario'],r['slip_p'],r['replicate'],r['survivors_known']);A={k(r):r for r in R if r['agent']=='search_andor'};B={k(r):r for r in R if r['agent']=='search_astar'};[print(z,len(Q),sum(A[q]['steps']==B[q]['steps'] for q in Q),sum(A[q]['steps']==B[q]['steps'] and A[q]['cells_explored']==B[q]['cells_explored'] for q in Q),sum(int(A[q]['steps'])<int(B[q]['steps']) for q in Q),sum(int(A[q]['steps'])>int(B[q]['steps']) for q in Q),round(s.mean(int(A[q]['steps'])-int(B[q]['steps']) for q in Q),2)) for z in ('True','False') for Q in [[q for q in A if q[3]==z]]]" -->

The reason is that **following the frontier dominates**. With r = 1 the robot always knows its
4-neighbours, and they usually border unknown cells, so the nearest frontier cell is usually
adjacent. A one-step target has a single shortest move, so both planners usually choose the same
action. The tie-break difference that gives the core
`andor` its known-location step advantage (core H3/H7, `docs/results.md` §2.4) has no room to
act. The same fact explains why X2's ratio is larger for `search_andor` (×2.89) than for
`search_astar` (×2.64). The unknown-location missions are about equally long, but
`search_andor`'s known-location baseline is shorter.

The AND-OR agent still pays its planning cost. In unknown-location search it used
1307.9 ms of planning time per mission against 4.60 ms for `search_astar`, **about 284 times as
much**. It made fewer replanning calls (74.4 against 100.6 per mission) and had no unplanned
states (A\*: 26.92 per mission). Neither bought a single step.
<!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='search' and r['survivors_known']=='False'];a,b=[s.mean(float(r['planning_ms']) for r in R if r['agent']==x) for x in ('search_andor','search_astar')];print(round(a,1),round(b,2),round(a/b))" -->
With the location known, on the same maps and slip probabilities, the ratio is about 74 times
(336.6 ms against 4.54 ms, same one-liner with `'True'`). It grows to 284 times because an A\*
call becomes cheaper when its target is adjacent, while an AND-OR call does not. Per planning
call, `search_andor` needed 18.47 ms with the location known and 17.34 ms with it unknown.
`search_astar` went from 0.155 ms to 0.045 ms.
<!-- python -c "import csv;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='search'];[print(a,z,round(sum(float(r['planning_ms']) for r in X)/sum(int(r['plans_made']) for r in X),3)) for a in ('search_andor','search_astar') for z in ('True','False') for X in [[r for r in R if r['agent']==a and r['survivors_known']==z]]]" -->
Timings are wall-clock times of a pure-Python implementation on one machine with 18 worker
processes. The ratio is more reliable than the absolute values. These timings are the round-2
rerun's (§4.1). The first run, on identical missions, measured 1117.1 ms against 4.12 ms
(about 271 times) and 267.1 ms against 4.12 ms (about 65 times) for the same quantities (the
same one-liners on `git show fdb8af0:results/ext/raw.csv`). So between two runs of the same
missions the absolute times moved by 10–26 % and the ratios by 5–14 %.
<!-- python -c "import csv,subprocess as sp,io,statistics as s;f=lambda t:{(a,z):s.mean(float(r['planning_ms']) for r in csv.DictReader(io.StringIO(t)) if r['experiment']=='search' and r['agent']==a and r['survivors_known']==z) for a in ('search_andor','search_astar') for z in ('True','False')};o=f(sp.run(['git','show','fdb8af0:results/ext/raw.csv'],capture_output=True,text=True).stdout);n=f(open('results/ext/raw.csv').read());[print(k,round(100*(n[k]/o[k]-1),1)) for k in o];[print(z,round(100*((n['search_andor',z]/n['search_astar',z])/(o['search_andor',z]/o['search_astar',z])-1),1)) for z in ('True','False')]" -->

### 6.2 The success claims do not discriminate

X1 and part (b) of X4 are supported with 100 % success, but every mission of both agents
succeeded in every block. The maps are solvable, sensing has no false positives in these blocks, and the budget
is generous. As in the core H1, these claims show that the extensions are reliable. They do not
show that AND-OR is needed for reliability.

### 6.3 X4(a) is close to true by construction

Bumps at ε > 0 are possible by design. Part (a) of X4 mainly checks the ε = 0 side: that the
harness, the noisy sensor and the `BeliefMap` path do not break the core's zero-bump invariant.
The measured content of X4 is the size of the bump rate and the fact that success holds.

### 6.4 Which thresholds saw data first

All four round-1 claim thresholds were fixed in `config.py` before the first full run, and the
round-1 `EXT_*` constants in `config.py` have not changed since the commit that run used
(`762a944`). They are still **not pre-registered in the strict sense**, because the quick run
was executed during development:

* **X5's margin** (0.1 bumps per mission, absolute) was chosen with the quick run's
  `search_astar` bump rates in view. The rationale in `config.py` quotes them to justify an
  absolute rather than a relative margin. With the full run's CIs this wide, a margin chosen
  differently could change the verdict. A margin of 0.05 would make it ⚠️, because the pooled
  upper bound is +0.03 but the ε = 0.2 upper bound is +0.06.
* **X1 and X4** reuse the deck's 95 % success target (core H1).
* **X3's 10 % limit** is a round number. Its CIs end far below it.

No experiment, seed or threshold was changed after the first full run. (Round 2's choices
are in §7.6.)

### 6.5 Replans do not grow with noise

`search_andor` made 17.11 replanning calls per mission at ε = 0 and 17.17 at ε = 0.2. Bumps rise
from 0 to 1.94 per mission over the same range. A plausible reading: a missed debris cell does
not add a contradiction, it delays it. The cell is believed open after the false reading, and
the contradiction (and its replan) comes later, when the cell is read again or touched. The run
does not measure this directly (same one-liner as §5.4).

### 6.6 The Bayes belief makes the same decisions as a "sticky debris" rule

With one-sided noise an OPEN reading only lowers P, and a DEBRIS reading or a touch sets it to 1.
With P₀ = 0.25 < 0.5, a cell is therefore impassable exactly when it has ever been read as
DEBRIS or touched. The probabilities themselves never change a decision of these agents. They
would matter for a planner that uses them, such as an expected-cost planner that avoids
uncertain cells, or for a sensor with false positives. With false positives they do: a
DEBRIS reading is no longer final, and the agent needs the actual value of P to tell a
suspected cell from a near-certain one (§7.2–§7.3). In the round-1 blocks, though, the
`ProbBelief`'s decisions match the simpler rule.

---

## 7. Two-sided noise (round 2)

Round 1's sensor only ever *hides* debris. Real sensors also see things that are not there:
dust, shadows or a fallen sheet can look like rubble. Round 2 adds this **false positive**: a
free cell read as DEBRIS with probability ε_fp. The contract is `docs/plan.md` §3.15.2. With
the default ε_fp = 0 every class, mission and harness row is the round-1 one (§4.1).

### 7.1 Sensor model: one draw per cell, used in both directions

`NoisySensor(false_negative, seed, rows, cols, false_positive=0.0)` still draws exactly one
uniform u per grid cell per call, and cell (r, c) still uses the draw at index `r·cols + c`
(§2.4). The same draw now serves both directions, because a cell is either debris or free:

| True cell | Reported as | When |
|---|---|---|
| DEBRIS | OPEN (false negative) | u < ε |
| OPEN | DEBRIS (false positive, a *phantom*) | u < ε_fp |
| SURVIVOR, the robot's own cell, the cell a bump just touched | the truth | always |

Three properties follow from the shared draw:

* **ε_fp = 0 changes nothing.** The sensor draws the same uniforms and reports the same cells
  as in round 1, so the round-1 rows are reproduced exactly (§4.1).
* **Phantoms are nested across levels.** A cell that is a phantom at ε_fp = 0.05 at step t is
  also one at ε_fp = 0.1 and 0.2 at step t. Every ε_fp level and both agents meet the same noise
  at a given (step, cell), so comparisons across levels and agents are paired.
* **Block `noisy_fp` at ε_fp = 0 is block `noisy` at ε = 0.1.** The two blocks share task
  seeds, and all 1,600 of these `noisy_fp` rows equal their `noisy` rows in every column apart
  from the block label and the timing columns.
  <!-- python -c "import csv;R=list(csv.DictReader(open('results/ext/raw.csv')));k=lambda r:(r['scenario'],r['replicate'],r['agent']);A={k(r):r for r in R if r['experiment']=='noisy' and r['false_negative']=='0.1'};F=[r for r in R if r['experiment']=='noisy_fp' and r['false_positive']=='0.0'];c=[x for x in R[0] if x not in('experiment','planning_ms','elapsed_ms')];print(len(F),sum(all(r[x]==A[k(r)][x] for x in c) for r in F))"
  prints 1600 1600 -->

The rate is limited to ε_fp < `EXT_MAX_FALSE_POSITIVE` = 0.5. Below one half (with ε below one
half too) a DEBRIS reading is more likely on a debris cell than on a free one, so every reading
still moves the belief in the right direction.

### 7.2 Two-sided Bayes update

With likelihoods P(OPEN | debris) = ε, P(OPEN | free) = 1 − ε_fp, P(DEBRIS | debris) = 1 − ε and
P(DEBRIS | free) = ε_fp, Bayes' rule gives (`ext/sensing.py`, `debris_posterior`):

| Observation | Update |
|---|---|
| reads OPEN | P ← ε·P / (ε·P + (1 − ε_fp)·(1 − P)) |
| reads DEBRIS | P ← (1 − ε)·P / ((1 − ε)·P + ε_fp·(1 − P)) |
| touched by a bump | P ← 1 |
| reads SURVIVOR | P ← 0, and the cell is a known survivor |
| the robot's own cell (only when ε_fp > 0) | P ← 0 |

With ε_fp = 0 the OPEN row is the round-1 rule (§2.4) and the DEBRIS row gives P = 1, so the
probabilities are bit-identical to round 1. (The own-cell rule is skipped there for the same
reason; it would not change any passability decision.) A reading that is impossible under the
model (a zero denominator) keeps the prior. Passability is still P < `EXT_DEBRIS_THRESHOLD`
= 0.5.

**Worked example** at ε = ε_fp = 0.1 and the prior P₀ = 0.25. Each DEBRIS reading multiplies the
odds P / (1 − P) by (1 − ε) / ε_fp = 9, and each OPEN reading divides them by 9. So at ε = ε_fp
the belief depends only on the number of DEBRIS readings minus the number of OPEN readings:

| DEBRIS − OPEN readings | −1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|
| P(debris) | 0.036 | 0.25 (prior) | 0.75 | 0.964 | 0.996 | 0.9995 |

One false DEBRIS reading of a free, never-seen cell therefore makes it impassable (0.75 ≥ 0.5),
and one later OPEN reading makes it passable again. This is the first change to round 1 that
matters: a DEBRIS reading is no longer final, so the "sticky debris" equivalence of §6.6 no
longer holds.

### 7.3 Phantom walls and the relaxed-passability fallback

A false positive can turn a free cell into a **phantom wall**. With r = 1 the robot reads a
cell only while it stands next to it. A phantom in a one-cell corridor or a doorway can
therefore cut the robot off from every target on its strict passability, and the phantom stays
until the robot comes back and reads the cell again. In round 1, "no reachable target" proved
that no survivor or frontier cell could be reached, and the agent aborted with `unreachable`.
With false positives that proof is gone: the wall may not exist.

`SearchAgent` therefore has a **fallback**. When the strict search (P < 0.5) finds no target,
the agent switches to a **relaxed passability** in which only cells with
P ≥ `EXT_FALLBACK_MAX_P` = 0.999 block. Such a cell has been touched (P = 1), or it has four
more DEBRIS than OPEN readings at ε = ε_fp = 0.1 (§7.2). The agent then re-selects its target on
the relaxed passability and plans to it. A cell suspected from fewer readings may thus be
tried. The relaxed mode lasts until the next target re-selection, which tries the strict
passability first. While relaxed, a cell of the plan in force that becomes impassable even on
the relaxed passability (typically a suspected cell the robot bumps into, which sets P = 1)
also triggers a re-selection and a replan. A *fallback* in the results is one strict → relaxed
switch (an *episode*). A re-selection made while already relaxed that falls back again
continues the episode and is not counted again.

With ε_fp = 0 the fallback never triggers. Every cell is then unobserved (P = prior < 0.5),
read OPEN only (P ≤ prior) or read DEBRIS or touched (P = 1 exactly), so the strict and relaxed
passabilities are the same mask. This is proved in the `SearchAgent` docstring and checked per
step in `tests/test_ext_fp.py`; all 14,400 round-1 rows have `fallback_replans` = 0 (§4.1).

### 7.4 What no longer holds

* **The termination argument of §2.2.** Its cause 2 relies on debris never being "un-seen". With
  false positives P can fall below the threshold again, so a target can be blocked and freed
  repeatedly. There is no rule against such flip-flopping (no hysteresis, no minimum number of
  readings). Only the step budget `4·R·C·k` bounds a mission. In the run it never bound: the
  longest `noisy_fp` mission used 65 % of it (§5.1).
* **Zero unplanned states for `search_andor`.** The AND-OR plan now covers the outcomes of a
  *wrong* model. A skid into a cell the belief holds as debris, but which is free, puts the
  robot where the plan said it could not go. The agent replans (D10), and the event is logged
  at DEBUG level as an expected model mismatch. `search_andor` met 54, 100 and 249 unplanned
  states in 800 missions each at ε_fp = 0.05, 0.1 and 0.2, against 0 at ε_fp = 0.
  <!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='noisy_fp'];[print(a,f,sum(int(r['unplanned_states']) for r in X),round(s.mean(int(r['replans']) for r in X),2),max(int(r['steps']) for r in X)) for a in ('search_andor','search_astar') for f in ('0.0','0.05','0.1','0.2') for X in [[r for r in R if r['agent']==a and r['false_positive']==f]]]" -->
* **A static map.** The Bayes filter assumes that cells do not change. §7.6 describes the patch
  for aftershocks.

### 7.5 X6: false positives cost steps, not rescues

Block `noisy_fp`: the four scenarios, 15×15, p = 0.2, survivor location known, ε = 0.1,
ε_fp ∈ {0, 0.05, 0.1, 0.2}, both agents, 200 maps per scenario (`EXTENSIONS.md`, X6):

| Agent | ε_fp | Success | Bumps per mission | Missions with a fallback | Fallbacks per mission | Steps |
|---|---|---|---|---|---|---|
| `search_andor` | 0 | 100.0% (800/800) | 0.95 [0.86, 1.04] | 0 | 0.00 | 40.88 [38.89, 42.86] |
| `search_andor` | 0.05 | 100.0% (800/800) | 1.75 [1.59, 1.91] | 147 | 0.32 [0.26, 0.37] | 71.79 [65.99, 77.58] |
| `search_andor` | 0.1 | 100.0% (800/800) | 2.11 [1.92, 2.30] | 266 | 0.84 [0.73, 0.95] | 87.41 [80.86, 93.95] |
| `search_andor` | 0.2 | 100.0% (800/800) | 3.59 [3.31, 3.87] | 412 | 2.27 [2.05, 2.50] | 110.26 [102.90, 117.62] |
| `search_astar` | 0 | 100.0% (800/800) | 1.00 [0.90, 1.09] | 0 | 0.00 | 45.08 [42.97, 47.19] |
| `search_astar` | 0.05 | 100.0% (800/800) | 1.78 [1.62, 1.94] | 141 | 0.21 [0.18, 0.25] | 71.00 [65.50, 76.49] |
| `search_astar` | 0.1 | 100.0% (800/800) | 1.97 [1.80, 2.14] | 265 | 0.42 [0.37, 0.46] | 81.02 [75.30, 86.74] |
| `search_astar` | 0.2 | 100.0% (800/800) | 2.75 [2.54, 2.96] | 411 | 0.55 [0.51, 0.59] | 87.27 [81.91, 92.63] |

Every success interval is Wilson [99.5 %, 100 %]. **X6 is supported**: `search_andor` rescued
the survivor in all 800 missions at ε_fp = 0.05 and at ε_fp = 0.1, and also at 0.2, which the
claim does not cover. As with X1 and X4, the claim shows that the fallback works. It does not
separate the agents: `search_astar` also succeeded in every mission.

The costs were fixed as informational only (no threshold before the run). Paired against the
same agent at ε_fp = 0 on the same map:

| Agent | ε_fp | Steps(ε_fp) − steps(0) (95 % CI) | Bumps(ε_fp) − bumps(0) (95 % CI) |
|---|---|---|---|
| `search_andor` | 0.05 | +30.91 [+26.00, +35.82] | +0.80 [+0.67, +0.93] |
| `search_andor` | 0.1 | +46.53 [+40.79, +52.27] | +1.16 [+1.00, +1.32] |
| `search_andor` | 0.2 | +69.38 [+62.92, +75.85] | +2.64 [+2.39, +2.90] |
| `search_astar` | 0.05 | +25.92 [+21.24, +30.60] | +0.79 [+0.65, +0.92] |
| `search_astar` | 0.1 | +35.94 [+31.14, +40.75] | +0.97 [+0.82, +1.13] |
| `search_astar` | 0.2 | +42.19 [+37.51, +46.86] | +1.75 [+1.55, +1.95] |

(800 maps each, paired t p ≤ 1e-25 in every row; `EXTENSIONS.md`, X6.)

![Success, bumps per mission, steps and fallbacks per mission against the false-positive rate, for both agents](../results/ext/plots/noisy_fp.png)

*800 missions per agent and point, ε = 0.1, 15×15, p = 0.2, four scenarios pooled. Success
(top left) stays at 100 % for both agents. At ε_fp = 0 `search_andor` takes fewer steps than
`search_astar` (and has slightly fewer bumps). The step and bump lines cross near ε_fp = 0.05
and pull apart as ε_fp grows, with `search_andor` above; paired on the same maps, its step
excess is significant from ε_fp = 0.1 on and its bump excess only at 0.2 (honest notes below).
The fallback panel separates most: `search_andor` is above from ε_fp = 0.05 on.*

Honest notes on these numbers:

* **Most of the cost sits in the missions that fell back.** Compared with the same maps at
  ε_fp = 0, `search_andor` at ε_fp = 0.1 took 38.9 steps against 32.2 (+6.7) and 0.85 bumps
  against 0.57 in the 534 missions without a fallback. So even these missions pay a visible
  false-positive cost. The 266 missions with a fallback took 184.8 steps against 58.2 and 4.63
  bumps against 1.71. Those maps were already harder at ε_fp = 0, and 90 % of the extra steps
  (33,655 of 37,224) fall in them (`search_astar`: 35.4 → 39.3 steps without a fallback, 64.6 →
  165.2 with one, 93 % of its extra steps). This split is correlational: a mission falls back
  because a phantom wall blocked it, and that mission would be harder in any case. Plausibly the extra bumps come from the relaxed mode itself,
  which deliberately drives through suspected cells and bumps into the ones that are real
  debris. The run does not separate these causes.
  <!-- python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='noisy_fp'];[print(a,f,b,len(X),round(s.mean(int(r['bumps']) for r in X),2),round(s.mean(int(r['steps']) for r in X),1)) for a in ('search_andor','search_astar') for f in ('0.05','0.1','0.2') for b in (0,1) for X in [[r for r in R if r['agent']==a and r['false_positive']==f and (int(r['fallback_replans'])>0)==b]]]" -->
  <!-- paired with the same agent, scenario and replicate at ε_fp = 0 (columns: n, steps at 0, steps, bumps at 0, bumps, extra steps in the subset, extra steps in all): python -c "import csv,statistics as s;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='noisy_fp'];k=lambda r:(r['agent'],r['scenario'],r['replicate']);Z={k(r):r for r in R if r['false_positive']=='0.0'};d=lambda r,c:int(r[c])-int(Z[k(r)][c]);[print(a,f,b,len(X),round(s.mean(int(Z[k(r)]['steps']) for r in X),1),round(s.mean(int(r['steps']) for r in X),1),round(s.mean(int(Z[k(r)]['bumps']) for r in X),2),round(s.mean(int(r['bumps']) for r in X),2),sum(d(r,'steps') for r in X),sum(d(r,'steps') for r in A)) for a in ('search_andor','search_astar') for f in ('0.05','0.1','0.2') for A in [[r for r in R if r['agent']==a and r['false_positive']==f]] for b in (0,1) for X in [[r for r in A if (int(r['fallback_replans'])>0)==b]]]" -->
* **AND-OR loses its step advantage.** Paired on the same maps, `search_andor` − `search_astar`
  steps are −4.20 [−5.58, −2.83] at ε_fp = 0 (the known-location advantage also seen in §6.1), +0.79
  [−4.22, +5.81] at 0.05, +6.38 [+0.87, +11.90] at 0.1 and +22.99 [+16.61, +29.37] at 0.2. Its
  bumps are +0.84 [+0.60, +1.08] per mission higher at ε_fp = 0.2 (+0.14 [−0.02, +0.30] at
  0.1). No claim was fixed for this comparison. X5's non-inferiority margin of 0.1 bumps per
  mission was defined for false negatives only, but applied here it would be inconclusive at
  ε_fp = 0.1 and refuted at 0.2.
  <!-- python -c "import csv,statistics as s;from scipy.stats import t;R=[r for r in csv.DictReader(open('results/ext/raw.csv')) if r['experiment']=='noisy_fp'];k=lambda r:(r['scenario'],r['replicate'],r['false_positive']);A={k(r):r for r in R if r['agent']=='search_andor'};B={k(r):r for r in R if r['agent']=='search_astar'};[print(f,c,len(x),round(m,2),round(m-h,2),round(m+h,2)) for f in ('0.0','0.05','0.1','0.2') for c in ('steps','bumps') for x in [[int(A[q][c])-int(B[q][c]) for q in A if q[2]==f]] for m in [s.mean(x)] for h in [t.ppf(.975,len(x)-1)*s.stdev(x)/len(x)**.5]]" -->
* **The fallback counts of the two agents are not directly comparable.** Almost the same number
  of missions fell back (147 / 141, 266 / 265 and 412 / 411), but `search_andor` had about twice
  as many episodes per mission at ε_fp = 0.1 (0.84 against 0.42) and about four times as many at
  0.2 (2.27 against 0.55). Its plan envelope also holds the skid cells beside the main path, so
  a suspected cell there that turns near-certain triggers a re-selection, which an A\* path
  never contains. That is a plausible reason why AND-OR pays more steps under false positives.
  It is not measured directly.

### 7.6 Choices made with data in view

* **The aftershock debris floor was chosen after measurements.** With ε_fp > 0 a cell at P = 0
  (the robot's own cell, once it has stood there) stays at 0 under every DEBRIS reading, because
  (1 − ε)·0 / (0 + ε_fp) = 0. Debris that an aftershock drops on a visited cell could then never
  be *seen*, only bumped into. Development missions with aftershocks showed exactly this
  (`docs/plan.md` §3.15.2). Only then was the floor added: with aftershocks enabled and
  ε_fp > 0, a DEBRIS reading starts from max(P, P₀), so it counts at least as much as a reading
  of an unseen cell. It adds no constant (the floor is the prior), and its value was not tuned.
  For `search_astar` it is only a partial patch: on the same 60 development missions
  (random_rubble 12×12, ε = ε_fp = 0.1, q = 0.3, k = 2) it had 108 bumps before the floor and
  72 after, against 45 without aftershocks (`search_andor`: 116 before, 45 after; `docs/plan.md`
  §3.15.2). These are development measurements, not rows of the committed run.
  It changes no row of the committed run, which has no aftershocks, and with ε_fp = 0 a DEBRIS
  reading gives P = 1 whatever the floor. The run does not test it.
* **The fallback threshold** `EXT_FALLBACK_MAX_P` = 0.999 is justified in `config.py` by the
  reading count of §7.2 (four net DEBRIS readings at ε = ε_fp = 0.1). It was fixed before the
  rerun, and the run used only this value. No record shows whether other values were tried in
  the quick runs executed during development, so, as in §6.4, it is not pre-registered in the
  strict sense.
* **Two fallback details were changed after the first quick runs.** The re-selection while
  relaxed when a plan cell becomes impassable (§7.3) was added because the robot otherwise
  bumped into the same suspected cell until the step budget ran out. This was found on the first
  quick run and fixed before any result was kept. Counting fallbacks per episode rather than per
  re-selection was also chosen after seeing quick-run counts, in which re-entries inflated
  `search_andor`'s numbers (plan §3.15.2). The counting changes only the reported
  `fallback_replans`, not any mission.
* **X6's threshold** (`EXT_CLAIM_FP` = 0.1, with the 95 % success rule of X1) was fixed before
  the rerun. The step and bump costs and the fallback usage have no verdict, because no
  threshold was fixed for them in advance.

---

## 8. Watching the extensions

Round 2 brings the extension missions into the interactive simulator (`docs/plan.md`
§3.15.3):

```bash
python -m rescue.sim --ext search            # unknown survivor location (frontier exploration)
python -m rescue.sim --ext multi --k 4       # k known survivors, greedy nearest-first order
python -m rescue.sim --ext noisy --fp 0.1    # noisy sensing (ε = 0.1 by default), with phantoms
python -m rescue.sim --ext search --headless --frames 120 --screenshot recordings/ext.png
```

Without further options a mode plays the mission of `python -m rescue.ext.demo --mode <mode>`
on the ext demo map (`collapsed_building`, 12×12, seed 7): one unknown survivor (`search`),
three known survivors (`multi`) or one known survivor with ε = 0.1 (`noisy`). `--k`, `--noise`
and `--fp` override the mode's values in every mode. The usual options (`--scenario`, `--map`,
`--rows`, `--cols`, `--seed`, `--p`, `--radius`, `--aftershock-q`, `--headless`, …) still work.
`--agent` picks `search_andor` (default) or `search_astar`. `--save-run` is not available with
`--ext`.

| Key | Action |
|---|---|
| `X` | next mode: search → multi → noisy (same overrides; map rebuilt, mission reset) |
| `1` / `2` | agent: `search_andor` / `search_astar` |
| `M` | next generated scenario |
| `S` | new seed (seed + 1; the map and its extra survivors are rebuilt) |
| `Space`, `N`, `R`, `+`/`-`, `[`/`]`, `T`, `P`, `C`, `G`, `H`, `Esc` | as in the core simulator |
| `E`, `Ctrl`+`S` | disabled (the map editor is not available in extension mode) |

The panel adds the mode (explore / rescue / done), survivors rescued, the step at which a
survivor was first found, bumps, cells explored, unplanned states, fallbacks (§7.3) and the
noise rates. The current target is a dashed square with a flag, and a rescued survivor gets a
green ring with a tick. With noisy sensing, seen-open cells get a hatch whose density shows
P(debris) on a log scale, with a legend under the metrics. With `--fp`, debris drawn lighter
with a `?` badge may be a phantom (P < 1). The browser build stays core-only (§9).

The three figures below come from `python scripts/make_ext_figures.py`, which renders them
headlessly and deterministically into `docs/figures/`.

![Search mode: the robot explores the edge of a collapsed building; the frontier target is marked with a flag](figures/ext_search.png)

*`search`, step 57 (step budget 576 = 4·R·C·k). The robot (blue) has circled the outside of
the building and is now inside it, heading for the frontier target at (10,6), flagged. The `?`
cells in the middle are still unknown, and the survivor is somewhere among them. Policy arrows show the
AND-OR plan to the target, and the plan tree on the right is its contingency plan.*

![Multi mode: one survivor rescued, the robot heading to the second of three](figures/ext_multi.png)

*`multi` with k = 3, step 20 (step budget 1,728). The survivor at (3,9) has just been rescued
(green ring and tick). The robot now heads to the nearest remaining known survivor at (4,4), flagged. The
third survivor is at the top left.*

![Noisy mode: three bumps into unseen debris, and seen-open cells shaded by their debris probability](figures/ext_noisy.png)

*`noisy` with ε = 0.1, step 18, just after the third bump (red crosses). Each bump revealed
debris the sensor had missed. The hatched cells were read OPEN, and the denser the hatch, the
higher their remaining debris probability (legend under the metrics, 0.001 to 0.5).*

---

## 9. Limitations and future work

* **Two-sided noise is only partly handled.** The fallback keeps the robot from giving up at a
  phantom wall, but there is no rule against flip-flopping (§7.4), and the step budget is the
  only termination guarantee. The belief assumes a static map; the aftershock floor (§7.6) is a
  patch, not a model of change. Block `noisy_fp` runs only one false-negative rate (ε = 0.1),
  the known-location mission and no aftershocks. No claim compares the two agents under false
  positives.
* **Planning in belief space (POMDP).** The planners still see a thresholded, deterministic
  passability map. Neither uses P to prefer certainly-open cells, and neither plans actions
  that gather information, such as looking before committing to a corridor. A POMDP or
  belief-space planner, or simply an expected-cost planner over `p_debris` in the style of
  `vi_expected`, is the natural next step. With false positives this matters more: the relaxed
  fallback is a crude two-level version of "prefer cells that are probably free".
* **Unknown survivor count.** The agent knows how many survivors there are, and that is how it
  knows when to stop. Without the count it would have to explore until no frontier is left (full
  coverage), or use a stopping rule based on the probability that a survivor is still hidden.
* **Smarter exploration.** Frontier choice is nearest-first with a row-major tie-break. Heuristics
  based on information gain, priors over where survivors are likely (rooms, not corridors), or
  coverage were not implemented or compared.
* **Tours.** The optimal tour is brute-forced only up to k = 6 (`EXT_MAX_BRUTE_FORCE_K`). X3
  evaluates the greedy rule on the true map, not the order the agent actually followed on its
  belief. Local improvement (2-opt) or re-ordering with travel-time estimates was not tried.
* **Combinations and conditions not run.** Each block varies one extension. The code supports
  unknown location, several survivors and noise together, but the run does not test them
  together. The ext run used only 15×15 maps, r = 1, eager replanning and no aftershocks.
* **UI support.** The desktop simulator plays the extension missions (§8). The comparison and
  replay apps do not, a run cannot be saved in extension mode, and the browser build
  (`docs/web.md`) leaves the extensions out entirely.
* **The rest of brief §8.** LAO\* (§8.1) was built in round 2 as the stretch agent
  `lao_expected` (`docs/algorithm.md` §8). It finds the same plans as `vi_expected` but did not
  plan faster on these maps. Terrain-aware AND-OR costs (§8.2) were not built.

---

## 10. Summary

* **§8.4 unknown location:** frontier exploration with the core planners rescued the survivor in
  every mission (X1, 1600/1600). It needs about 2.9 times the steps of the known-location
  mission (X2, +76.27 [+71.61, +80.93]), almost all of it spent searching. The choice of planner
  hardly matters here (1525/1600 identical pairs), but AND-OR costs about 284 times the planning
  time.
* **§8.5 several survivors:** the greedy nearest-first order is optimal on most maps and 1.1 %
  (k = 3) to 2.7 % (k = 4) longer on average (X3), with a tail of up to ×1.444. Mostly slips and an
  unknown map, with the order adding at most a few percent, make the realised route about 1.7
  times the optimal tour.
* **§8.3 noisy sensing, false negatives:** missing debris with probability ε causes about one
  bump per mission per 0.1 of ε (X4). Success stays at 100 %, AND-OR keeps zero unplanned
  states, and it collides no more often than A\* + replan (X5, non-inferior within 0.1 bumps per
  mission).
* **§8.3 noisy sensing, false positives (round 2):** with phantom walls and the
  relaxed-passability fallback both agents still rescued every survivor (X6, 800/800 at every
  ε_fp). The price is steps: +46.53 [+40.79, +52.27] for `search_andor` at ε_fp = 0.1, 90 %
  of it in the third of the missions that fell back (§7.5). Under false positives AND-OR no longer
  takes fewer steps than A\* + replan, and from ε_fp = 0.1 on it takes more.
* **Simulator:** `python -m rescue.sim --ext search|multi|noisy` shows all three extensions.
* **Rule zero:** the core is unchanged. With the location known, the search agents replay the
  core `andor` and `astar_replan` missions exactly (1600/1600 each), and the round-2 rerun
  reproduced every one of the 14,400 round-1 rows.
