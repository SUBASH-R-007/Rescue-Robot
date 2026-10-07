# Hypotheses — `lao`

- Date of run: 2026-10-07
- Missions (rows of raw.csv): 1500; agents: `andor`, `vi_expected`, `lao_expected`
- Config file: `experiments/lao.yaml`
- Base seed: 2026
- Tasks (maps × conditions): 500
- Workers: 18
- Wall time: 179.6 s
- Git commit: `cfb50a2`
- Methods: success rates with Wilson 95 % intervals; means with Student-t 95 % intervals; agent comparisons paired on identical tasks (common random numbers: same map, same outcome stream) with a paired t-test and a Wilcoxon signed-rank test (Pratt zero handling). Because each map is reused at every slip p, pooled comparisons use the map as the unit (per-map mean difference; n = maps, not pairs).
- Verdicts: ✅ supported, ❌ refuted, ⚠️ inconclusive or not measured. Thresholds come from the proposal deck; no experiment was tuned toward a verdict.

## Claims

| # | Claim | Verdict | Measured |
|---|---|---|---|
| H1 | `andor` rescues the survivor in ≥ 95.0% of missions at p = 0.3 | ⚠️ | not measured: no `andor` runs at p = 0.3 |
| H2 | `andor` never reaches an unplanned state (and never bumps into debris) | ✅ | Σ unplanned states = 0, Σ bumps = 0 over 500 `andor` runs (500 unique missions) |
| H3 | `andor` wastes fewer moves than `astar_replan` | ⚠️ | not measured: no `andor`, `astar_replan` runs in the slip sweep |
| H4 | `astar_fixed` fails the first time an action misbehaves (slip or skid) | ⚠️ | not measured: `astar_fixed` not run at both p = 0 and p ≥ 0.1 |
| H5 | `andor` replans less often than `astar_replan` | ⚠️ | not measured: no `andor`, `astar_replan` runs in the slip sweep |
| H6 | Reference: `vi_expected` (expected-steps optimal policy) needs fewer steps than `andor` | ⚠️ | +1.39 steps [-0.71, +3.50] over 500 maps (500 pairs), paired t p = 0.19, Wilcoxon p = 0.0073 (293/500 maps with a non-zero difference); per scenario ✅ 0 / ⚠️ 1 / ❌ 1; CI above 0 (claim refuted) in random_rubble; mean above 0 (against the claim) in random_rubble (+1.39), maze_corridors (+1.40) |
| H7 | `andor`'s wasted-move advantage over `astar_replan` survives matched tie-breaking | ⚠️ | not measured: no slip-sweep experiment in the data |

## Details

### H1. ⚠️ `andor` rescues the survivor in ≥ 95.0% of missions at p = 0.3

not measured: no `andor` runs at p = 0.3.

### H2. ✅ `andor` never reaches an unplanned state (and never bumps into debris)

All experiments (every condition, every size). Rule: ✅ iff the sum of unplanned states over all runs is 0. Bumps (D6) are reported alongside: with r ≥ 1 both are invariants of the design.

| experiment | sensor r | `andor` runs | Σ unplanned | Σ bumps |
|---|---|---|---|---|
| lao_scaling | 1 | 500 | 0 | 0 |

Blocks share task seeds (the seed depends only on scenario, size and replicate), so a block's default-condition runs replay missions of other blocks exactly (same map, outcome stream and parameters). Deduplicated on (scenario, size, seed, p, r, replan mode, planner overrides, q, k), the 500 runs are 500 unique missions.

For reference, `vi_expected` (same anticipation, D19): Σ unplanned = 0, Σ bumps = 0 over 500 runs.

### H3. ⚠️ `andor` wastes fewer moves than `astar_replan`

not measured: no `andor`, `astar_replan` runs in the slip sweep.

### H4. ⚠️ `astar_fixed` fails the first time an action misbehaves (slip or skid)

not measured: `astar_fixed` not run at both p = 0 and p ≥ 0.1.

### H5. ⚠️ `andor` replans less often than `astar_replan`

not measured: no `andor`, `astar_replan` runs in the slip sweep.

### H6. ⚠️ Reference: `vi_expected` (expected-steps optimal policy) needs fewer steps than `andor`

Experiment `lao_scaling`; sizes pooled: 10×10, 15×15, 20×20, 25×25, 30×30; scenarios: random_rubble, maze_corridors; slip p ∈ {0.2}.

Difference = `vi_expected` − `andor` in steps on identical tasks (same map, same outcome stream), both successful, pooled over p > 0: +1.39 steps [-0.71, +3.50] over 500 maps (500 pairs), paired t p = 0.19, Wilcoxon p = 0.0073 (293/500 maps with a non-zero difference).

Unit of analysis: the map. Every map and its outcome stream are reused at each slip p (common random numbers), so a map's differences at the p > 0 values are averaged first; the Student-t CI, the paired t-test and the Wilcoxon signed-rank test (Pratt: zero differences are ranked, not discarded) run over those per-map means. Rule: ✅ if the 95 % CI lies below 0, ❌ if it lies above 0, ⚠️ if it contains 0 — applied to the pooled verdict and to every row below.

By slip p (one pair per map):

| slip p | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 0.2 | 500 | 500 | +1.39 | [-0.71, +3.50] | 0.19 | 0.0073 | ⚠️ |

By scenario (p > 0):

| scenario | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| random_rubble | 250 | 250 | +1.39 | [+0.29, +2.50] | 0.014 | 0.0013 | ❌ |
| maze_corridors | 250 | 250 | +1.40 | [-2.69, +5.48] | 0.5 | 0.48 | ⚠️ |

By grid size (p > 0):

| size | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 10×10 | 100 | 100 | +1.10 | [-0.69, +2.89] | 0.23 | 0.38 | ⚠️ |
| 15×15 | 100 | 100 | +2.74 | [-1.01, +6.49] | 0.15 | 0.09 | ⚠️ |
| 20×20 | 100 | 100 | -0.02 | [-4.48, +4.44] | 0.99 | 0.95 | ⚠️ |
| 25×25 | 100 | 100 | +3.14 | [-2.94, +9.22] | 0.31 | 0.22 | ⚠️ |
| 30×30 | 100 | 100 | +0.01 | [-6.28, +6.30] | 1 | 0.049 | ⚠️ |

Informational (no deck claim). Success: `vi_expected` 100.0% [99.2%, 100.0%] (500/500), `andor` 100.0% [99.2%, 100.0%] (500/500) (all p).

### H7. ⚠️ `andor`'s wasted-move advantage over `astar_replan` survives matched tie-breaking

not measured: no slip-sweep experiment in the data.
