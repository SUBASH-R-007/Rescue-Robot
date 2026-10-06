# Hypotheses — `main`

- Date of run: 2026-10-06
- Missions (rows of raw.csv): 99400; agents: `andor`, `astar_fixed`, `astar_replan`, `dstar_lite`, `vi_expected`
- Config file: `experiments/main.yaml`
- Base seed: 2026
- Tasks (maps × conditions): 27450
- Workers: 18
- Wall time: 873.9 s
- Git commit: `a6d6484`
- Methods: success rates with Wilson 95 % intervals; means with Student-t 95 % intervals; agent comparisons paired on identical tasks (common random numbers: same map, same outcome stream) with a paired t-test and a Wilcoxon signed-rank test (Pratt zero handling). Because each map is reused at every slip p, pooled comparisons use the map as the unit (per-map mean difference; n = maps, not pairs).
- Verdicts: ✅ supported, ❌ refuted, ⚠️ inconclusive or not measured. Thresholds come from the proposal deck; no experiment was tuned toward a verdict.

## Claims

| # | Claim | Verdict | Measured |
|---|---|---|---|
| H1 | `andor` rescues the survivor in ≥ 95.0% of missions at p = 0.3 | ✅ | 100.0% [99.8%, 100.0%] (2400/2400); Wilson lower bound 99.8% ≥ 95.0% |
| H2 | `andor` never reaches an unplanned state (and never bumps into debris) | ✅ | Σ unplanned states = 0, Σ bumps = 0 over 27450 `andor` runs (26100 unique missions) |
| H3 | `andor` wastes fewer moves than `astar_replan` | ✅ | -3.09 moves [-3.81, -2.37] over 2400 maps (9600 pairs), paired t p = 8.6e-17, Wilcoxon p = 4e-45 (1661/2400 maps with a non-zero difference); per scenario ✅ 4 / ⚠️ 0 / ❌ 0 |
| H4 | `astar_fixed` fails the first time an action misbehaves (slip or skid) | ✅ | success 4.6% [3.8%, 5.5%] (110/2400) at p = 0 vs 0.6% (61/9600 runs on 2400 maps) at p ≥ 0.1; map-paired sign test p = 3.1e-33 (109 maps worse, 0 better at p ≥ 0.1); 6141/6141 runs with a slip or skid failed |
| H5 | `andor` replans less often than `astar_replan` | ✅ | -14.14 replans [-14.73, -13.56] over 2400 maps (9600 pairs), paired t p = 0, Wilcoxon p = 0 (2394/2400 maps with a non-zero difference); per scenario ✅ 4 / ⚠️ 0 / ❌ 0 |
| H6 | Reference: `vi_expected` (expected-steps optimal policy) needs fewer steps than `andor` | ❌ | +1.31 steps [+0.59, +2.04] over 2400 maps (9600 pairs), paired t p = 0.00039, Wilcoxon p = 0.76 (1479/2400 maps with a non-zero difference); per scenario ✅ 1 / ⚠️ 2 / ❌ 1; CI above 0 (claim refuted) in collapsed_building; mean above 0 (against the claim) in random_rubble (+0.44), collapsed_building (+6.00) |
| H7 | `andor`'s wasted-move advantage over `astar_replan` survives matched tie-breaking | ❌ | matched diagonal +0.11 [-0.12, +0.35] ⚠️ (CI ±0.24), matched plain +0.56 [+0.20, +0.91] ❌ (CI ±0.36) moves over p > 0 (2400 maps; equivalence margin ±0.5); single-factor effects: andor diagonal − fixed -3.64 [-4.40, -2.89] ✅, A* diagonal − g -3.20 [-3.91, -2.49] ✅; unmatched (H3) -3.09 [-3.81, -2.37] ✅; identical steps + cells explored at p > 0: (a) 92.1% (8839/9600), (b) 87.9% (8442/9600), (H3) 25.6% (2454/9600) (tie-break matching is approximate); refuted: a matched CI lies above 0 |

## Details

### H1. ✅ `andor` rescues the survivor in ≥ 95.0% of missions at p = 0.3

Experiment `slip_sweep`; sizes pooled: 10×10, 15×15, 20×20; scenarios: random_rubble, collapsed_building, flooded_zone, maze_corridors; slip p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

Overall at p = 0.3: 100.0% [99.8%, 100.0%] (2400/2400) (Wilson 95 % CI). Rule: ✅ if the point estimate ≥ 95.0%; ⚠️ if it is below but the CI upper bound reaches 95.0%; ❌ if the upper bound is below 95.0%. The ✅ rule rests on the point estimate, so the claims table also states whether the Wilson lower bound clears 95.0% and names any scenario below it.

| scenario | success (Wilson 95 % CI) | failures |
|---|---|---|
| random_rubble | 100.0% [99.4%, 100.0%] (600/600) | — |
| collapsed_building | 100.0% [99.4%, 100.0%] (600/600) | — |
| flooded_zone | 100.0% [99.4%, 100.0%] (600/600) | — |
| maze_corridors | 100.0% [99.4%, 100.0%] (600/600) | — |

| size | success (Wilson 95 % CI) |
|---|---|
| 10×10 | 100.0% [99.5%, 100.0%] (800/800) |
| 15×15 | 100.0% [99.5%, 100.0%] (800/800) |
| 20×20 | 100.0% [99.5%, 100.0%] (800/800) |

### H2. ✅ `andor` never reaches an unplanned state (and never bumps into debris)

All experiments (every condition, every size). Rule: ✅ iff the sum of unplanned states over all runs is 0. Bumps (D6) are reported alongside: with r ≥ 1 both are invariants of the design.

| experiment | sensor r | `andor` runs | Σ unplanned | Σ bumps |
|---|---|---|---|---|
| slip_sweep | 1 | 12000 | 0 | 0 |
| aftershocks | 1 | 1600 | 0 | 0 |
| sensor_range | 1 | 400 | 0 | 0 |
| sensor_range | 2 | 400 | 0 | 0 |
| replan_mode | 1 | 800 | 0 | 0 |
| scaling | 1 | 250 | 0 | 0 |
| tiebreak_ablation | 1 | 12000 | 0 | 0 |

Blocks share task seeds (the seed depends only on scenario, size and replicate), so a block's default-condition runs replay missions of other blocks exactly (same map, outcome stream and parameters). Deduplicated on (scenario, size, seed, p, r, replan mode, planner overrides, q, k), the 27450 runs are 26100 unique missions.

For reference, `vi_expected` (same anticipation, D19): Σ unplanned = 0, Σ bumps = 0 over 14650 runs.

### H3. ✅ `andor` wastes fewer moves than `astar_replan`

Experiment `slip_sweep`; sizes pooled: 10×10, 15×15, 20×20; scenarios: random_rubble, collapsed_building, flooded_zone, maze_corridors; slip p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

Difference = `andor` − `astar_replan` in wasted_moves on identical tasks (same map, same outcome stream), both successful, pooled over p > 0: -3.09 moves [-3.81, -2.37] over 2400 maps (9600 pairs), paired t p = 8.6e-17, Wilcoxon p = 4e-45 (1661/2400 maps with a non-zero difference).

Unit of analysis: the map. Every map and its outcome stream are reused at each slip p (common random numbers), so a map's differences at the p > 0 values are averaged first; the Student-t CI, the paired t-test and the Wilcoxon signed-rank test (Pratt: zero differences are ranked, not discarded) run over those per-map means. Rule: ✅ if the 95 % CI lies below 0, ❌ if it lies above 0, ⚠️ if it contains 0 — applied to the pooled verdict and to every row below.

By slip p (one pair per map):

| slip p | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 0 (p = 0, not in the verdict) | 2400 | 2400 | -2.12 | [-2.68, -1.56] | 1.6e-13 | 2.2e-26 | ✅ |
| 0.1 | 2400 | 2400 | -2.38 | [-3.00, -1.77] | 5.2e-14 | 1.6e-29 | ✅ |
| 0.2 | 2400 | 2400 | -2.86 | [-3.56, -2.16] | 1.4e-15 | 1.7e-33 | ✅ |
| 0.3 | 2400 | 2400 | -3.01 | [-3.81, -2.21] | 2.7e-13 | 4.2e-36 | ✅ |
| 0.4 | 2400 | 2400 | -4.10 | [-5.02, -3.18] | 5e-18 | 1.4e-48 | ✅ |

By scenario (p > 0):

| scenario | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| random_rubble | 600 | 2400 | -3.27 | [-4.17, -2.37] | 2.8e-12 | 2.2e-23 | ✅ |
| collapsed_building | 600 | 2400 | -5.58 | [-7.73, -3.43] | 4.6e-07 | 2.4e-10 | ✅ |
| flooded_zone | 600 | 2400 | -1.25 | [-1.49, -1.01] | 3.6e-22 | 3.7e-20 | ✅ |
| maze_corridors | 600 | 2400 | -2.26 | [-3.94, -0.58] | 0.0086 | 0.0021 | ✅ |

By grid size (p > 0):

| size | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 10×10 | 800 | 3200 | -0.99 | [-1.64, -0.35] | 0.0026 | 8e-13 | ✅ |
| 15×15 | 800 | 3200 | -4.20 | [-5.55, -2.85] | 1.6e-09 | 3e-22 | ✅ |
| 20×20 | 800 | 3200 | -4.08 | [-5.64, -2.51] | 3.9e-07 | 4.4e-14 | ✅ |

At p = 0 nothing slips except on high-slip '~' terrain, which keeps its own slip parameter (0.4) at every block p (only in flooded_zone here: 975/1200 of the runs there at p = 0 had a slip or skid); elsewhere `andor` has no contingency to plan for and reduces to a greedy shortest-route search on the optimistic belief. The p = 0 difference is already -2.12 [-2.68, -1.56] moves over 2400 maps (row p = 0 above); without flooded_zone (the clean no-contingency baseline) it is -2.74 [-3.48, -2.00] moves over 1800 maps. So part of the gap is path choice under the optimistic-unknown assumption (different tie-breaks), not contingency planning; H7 repeats the comparison with matched tie-breaking.

### H4. ✅ `astar_fixed` fails the first time an action misbehaves (slip or skid)

Experiment `slip_sweep`; sizes pooled: 10×10, 15×15, 20×20; scenarios: random_rubble, collapsed_building, flooded_zone, maze_corridors; slip p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

Rule: ✅ if the success rate at p ≥ 0.1 is below the one at p = 0, the drop is significant in an exact two-sided sign test over maps (α = 0.05; per map: success at p = 0 minus mean success at p ≥ 0.1; with one p this is the exact McNemar test) **and** every run with at least one slip or skid failed; ⚠️ if the drop is not significant or some runs survived a slip or skid; ❌ if the success rate does not drop. The map is the unit because every p reuses the same maps (common random numbers); the per-p Wilson intervals below are each over distinct maps.

Runs with ≥ 1 non-arrived outcome: 6141; survived: 0 (a run can survive only if the deviation itself lands on the survivor).

Abort reasons: *after a slip/skid* = `aborted` with slips + skids > 0 (deviation from the path); *debris ahead* = `aborted` with none (the next path cell turned out to be debris; no replanning).

| slip p | success (Wilson 95 % CI) | aborted after slip/skid | aborted at debris ahead | budget | unreachable |
|---|---|---|---|---|---|
| 0 | 4.6% [3.8%, 5.5%] (110/2400) | 342 | 1948 | 0 | 0 |
| 0.1 | 1.8% [1.3%, 2.4%] (42/2400) | 988 | 1370 | 0 | 0 |
| 0.2 | 0.5% [0.3%, 0.9%] (12/2400) | 1374 | 1014 | 0 | 0 |
| 0.3 | 0.2% [0.1%, 0.5%] (6/2400) | 1628 | 766 | 0 | 0 |
| 0.4 | 0.0% [0.0%, 0.2%] (1/2400) | 1809 | 590 | 0 | 0 |

Caveats: (1) p = 0 is not free of misbehaviour: high-slip terrain ('~' cells) keeps its own slip parameter (0.4) at every p, so 342/2400 runs at p = 0 had a slip or skid. (2) `astar_fixed` also aborts when it senses debris on its next path cell, so its success at p = 0 is already only 4.6% [3.8%, 5.5%] (110/2400) (1948 aborts at debris ahead): most of its failures come from debris it did not know about, not from a misbehaving action. The verdict covers the slip/skid part of the claim only.

### H5. ✅ `andor` replans less often than `astar_replan`

Experiment `slip_sweep`; sizes pooled: 10×10, 15×15, 20×20; scenarios: random_rubble, collapsed_building, flooded_zone, maze_corridors; slip p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

Difference = `andor` − `astar_replan` in replans on identical tasks (same map, same outcome stream), all runs, pooled over p > 0: -14.14 replans [-14.73, -13.56] over 2400 maps (9600 pairs), paired t p = 0, Wilcoxon p = 0 (2394/2400 maps with a non-zero difference).

Unit of analysis: the map. Every map and its outcome stream are reused at each slip p (common random numbers), so a map's differences at the p > 0 values are averaged first; the Student-t CI, the paired t-test and the Wilcoxon signed-rank test (Pratt: zero differences are ranked, not discarded) run over those per-map means. Rule: ✅ if the 95 % CI lies below 0, ❌ if it lies above 0, ⚠️ if it contains 0 — applied to the pooled verdict and to every row below.

By slip p (one pair per map):

| slip p | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 0 (p = 0, not in the verdict) | 2400 | 2400 | -1.85 | [-2.18, -1.51] | 3.5e-27 | 6.9e-79 | ✅ |
| 0.1 | 2400 | 2400 | -5.51 | [-5.90, -5.13] | 1.2e-152 | 1.5e-274 | ✅ |
| 0.2 | 2400 | 2400 | -10.23 | [-10.72, -9.73] | 5.6e-272 | 0 | ✅ |
| 0.3 | 2400 | 2400 | -16.09 | [-16.76, -15.42] | 0 | 0 | ✅ |
| 0.4 | 2400 | 2400 | -24.75 | [-25.68, -23.82] | 0 | 0 | ✅ |

By scenario (p > 0):

| scenario | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| random_rubble | 600 | 2400 | -9.78 | [-10.36, -9.20] | 3.5e-138 | 3.7e-99 | ✅ |
| collapsed_building | 600 | 2400 | -19.77 | [-21.34, -18.20] | 1.4e-93 | 2.3e-84 | ✅ |
| flooded_zone | 600 | 2400 | -8.10 | [-8.47, -7.74] | 2.9e-188 | 5.8e-100 | ✅ |
| maze_corridors | 600 | 2400 | -18.93 | [-20.28, -17.58] | 1.4e-108 | 1.5e-95 | ✅ |

By grid size (p > 0):

| size | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 10×10 | 800 | 3200 | -7.93 | [-8.40, -7.45] | 1.1e-149 | 6.4e-125 | ✅ |
| 15×15 | 800 | 3200 | -14.24 | [-15.20, -13.28] | 1.7e-127 | 1.7e-122 | ✅ |
| 20×20 | 800 | 3200 | -20.27 | [-21.51, -19.02] | 5.7e-145 | 2.5e-124 | ✅ |

### H6. ❌ Reference: `vi_expected` (expected-steps optimal policy) needs fewer steps than `andor`

Experiment `slip_sweep`; sizes pooled: 10×10, 15×15, 20×20; scenarios: random_rubble, collapsed_building, flooded_zone, maze_corridors; slip p ∈ {0, 0.1, 0.2, 0.3, 0.4}.

Difference = `vi_expected` − `andor` in steps on identical tasks (same map, same outcome stream), both successful, pooled over p > 0: +1.31 steps [+0.59, +2.04] over 2400 maps (9600 pairs), paired t p = 0.00039, Wilcoxon p = 0.76 (1479/2400 maps with a non-zero difference).

Unit of analysis: the map. Every map and its outcome stream are reused at each slip p (common random numbers), so a map's differences at the p > 0 values are averaged first; the Student-t CI, the paired t-test and the Wilcoxon signed-rank test (Pratt: zero differences are ranked, not discarded) run over those per-map means. Rule: ✅ if the 95 % CI lies below 0, ❌ if it lies above 0, ⚠️ if it contains 0 — applied to the pooled verdict and to every row below.

By slip p (one pair per map):

| slip p | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 0 (p = 0, not in the verdict) | 2400 | 2400 | +2.06 | [+1.47, +2.65] | 7.6e-12 | 2.4e-09 | ❌ |
| 0.1 | 2400 | 2400 | +1.12 | [+0.45, +1.80] | 0.0012 | 0.077 | ❌ |
| 0.2 | 2400 | 2400 | +1.23 | [+0.49, +1.97] | 0.0011 | 0.61 | ❌ |
| 0.3 | 2400 | 2400 | +1.25 | [+0.44, +2.05] | 0.0023 | 0.27 | ❌ |
| 0.4 | 2400 | 2400 | +1.65 | [+0.74, +2.56] | 0.00039 | 0.0074 | ❌ |

By scenario (p > 0):

| scenario | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| random_rubble | 600 | 2400 | +0.44 | [-0.26, +1.14] | 0.22 | 0.15 | ⚠️ |
| collapsed_building | 600 | 2400 | +6.00 | [+3.88, +8.12] | 4e-08 | 6.4e-06 | ❌ |
| flooded_zone | 600 | 2400 | -0.54 | [-0.70, -0.38] | 2.6e-10 | 1.5e-11 | ✅ |
| maze_corridors | 600 | 2400 | -0.65 | [-2.45, +1.15] | 0.48 | 0.59 | ⚠️ |

By grid size (p > 0):

| size | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| 10×10 | 800 | 3200 | +0.48 | [+0.01, +0.95] | 0.044 | 0.43 | ❌ |
| 15×15 | 800 | 3200 | +2.77 | [+1.55, +3.99] | 9.2e-06 | 0.31 | ❌ |
| 20×20 | 800 | 3200 | +0.68 | [-1.05, +2.42] | 0.44 | 0.49 | ⚠️ |

Informational (no deck claim). Success: `vi_expected` 100.0% [100.0%, 100.0%] (12000/12000), `andor` 100.0% [100.0%, 100.0%] (12000/12000) (all p).

### H7. ❌ `andor`'s wasted-move advantage over `astar_replan` survives matched tie-breaking

Experiments `slip_sweep` (default planner: AND-OR D13 diagonal tie-break, A* larger-g then FIFO) and `tiebreak_ablation` (the same missions with the tie-breaks swapped), restricted to r = 1, eager replanning, no aftershocks. Task seeds do not depend on the planner overrides, so every ablation run pairs 1:1 with the slip-sweep run of the same map, outcome stream and slip p (pairing across blocks on scenario, rows, cols, slip_p, sensor_radius, replan_mode, aftershock_q, replicate, seed).

Why: at p = 0 nothing slips except on high-slip '~' terrain, which keeps its own slip parameter (0.4) at every block p (only in flooded_zone here: 1949/2400 of the runs there at p = 0 had a slip or skid); everywhere else AND-OR has no contingency to plan for and reduces to a greedy shortest-route search on the optimistic belief, so a difference there (H3's p = 0 row; the (H3) column of the p = 0 row below) is path choice, not contingency planning. Under the optimistic-unknown assumption many routes are equally short, and which one an agent takes (its tie-break) decides how often it walks into unseen debris. (a) and (b) compare the agents with matched tie-breaks (only approximately matched: see the manipulation check below); (c) and (d) measure each tie-break's own effect. On common pairs H3 = (a) + (d) = (b) + (c) (approximately here, since each difference uses the runs where both of its sides succeeded).

| comparison | a | b |
|---|---|---|
| (a) matched, diagonal | `andor` tie_break=diagonal (`slip_sweep`) | `astar_replan` astar_tie_break=diagonal (`tiebreak_ablation`) |
| (b) matched, plain | `andor` tie_break=fixed (`tiebreak_ablation`) | `astar_replan` astar_tie_break=g (`slip_sweep`) |
| (c) andor diagonal − fixed | `andor` tie_break=diagonal (`slip_sweep`) | `andor` tie_break=fixed (`tiebreak_ablation`) |
| (d) A* diagonal − g | `astar_replan` astar_tie_break=diagonal (`tiebreak_ablation`) | `astar_replan` astar_tie_break=g (`slip_sweep`) |
| (H3) unmatched | `andor` tie_break=diagonal (`slip_sweep`) | `astar_replan` astar_tie_break=g (`slip_sweep`) |

Difference = a − b in wasted moves (successful runs on both sides), map as the unit (per-map mean over the p > 0 values, as in H3); Student-t 95 % CI, paired t-test and Wilcoxon signed-rank test over maps. Rule: ✅ if the CIs of both matched comparisons (a) and (b) lie below 0 over p > 0; ❌ if either lies above 0, or if both lie entirely inside ±0.5 moves (equivalence margin fixed a priori, TOST-style: no practically relevant advantage once tie-breaking is matched); ⚠️ otherwise (a CI contains 0 but is too wide to show equivalence: underpowered, not refuted). p = 0 is shown separately and is not part of the verdict. CI half-widths: (a) ±0.24, (b) ±0.36 moves.

Pooled over p > 0:

| comparison | maps | pairs | mean diff | 95 % CI | paired t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| (a) matched, diagonal | 2400 | 9600 | +0.11 | [-0.12, +0.35] | 0.35 | 0.57 | ⚠️ |
| (b) matched, plain | 2400 | 9600 | +0.56 | [+0.20, +0.91] | 0.0024 | 0.0001 | ❌ |
| (c) andor diagonal − fixed | 2400 | 9600 | -3.64 | [-4.40, -2.89] | 6.9e-21 | 1.2e-49 | ✅ |
| (d) A* diagonal − g | 2400 | 9600 | -3.20 | [-3.91, -2.49] | 1.7e-18 | 1e-49 | ✅ |
| (H3) unmatched | 2400 | 9600 | -3.09 | [-3.81, -2.37] | 8.6e-17 | 4e-45 | ✅ |

By slip p (mean diff [95 % CI] verdict; one pair per map):

| slip p | (a) matched, diagonal | (b) matched, plain | (c) andor diagonal − fixed | (d) A* diagonal − g | (H3) unmatched |
|---|---|---|---|---|---|
| 0 (not in the verdict) | +0.03 [-0.15, +0.21] ⚠️ | +0.38 [+0.11, +0.66] ❌ | -2.50 [-3.09, -1.92] ✅ | -2.15 [-2.70, -1.60] ✅ | -2.12 [-2.68, -1.56] ✅ |
| 0.1 | +0.06 [-0.15, +0.27] ⚠️ | +0.40 [+0.09, +0.71] ❌ | -2.79 [-3.43, -2.14] ✅ | -2.44 [-3.04, -1.84] ✅ | -2.38 [-3.00, -1.77] ✅ |
| 0.2 | +0.05 [-0.16, +0.26] ⚠️ | +0.44 [+0.08, +0.80] ❌ | -3.30 [-4.04, -2.56] ✅ | -2.91 [-3.60, -2.23] ✅ | -2.86 [-3.56, -2.16] ✅ |
| 0.3 | +0.27 [-0.00, +0.55] ⚠️ | +0.71 [+0.30, +1.11] ❌ | -3.72 [-4.56, -2.88] ✅ | -3.28 [-4.06, -2.50] ✅ | -3.01 [-3.81, -2.21] ✅ |
| 0.4 | +0.08 [-0.27, +0.43] ⚠️ | +0.68 [+0.22, +1.13] ❌ | -4.78 [-5.74, -3.82] ✅ | -4.18 [-5.08, -3.27] ✅ | -4.10 [-5.02, -3.18] ✅ |

By scenario (p > 0, map as the unit):

| scenario | (a) matched, diagonal | (b) matched, plain | (c) andor diagonal − fixed | (d) A* diagonal − g | (H3) unmatched |
|---|---|---|---|---|---|
| random_rubble | -0.03 [-0.23, +0.18] ⚠️ | +0.04 [-0.22, +0.29] ⚠️ | -3.30 [-4.19, -2.42] ✅ | -3.24 [-4.10, -2.38] ✅ | -3.27 [-4.17, -2.37] ✅ |
| collapsed_building | +0.31 [-0.12, +0.73] ⚠️ | +1.15 [+0.30, +2.00] ❌ | -6.73 [-8.93, -4.53] ✅ | -5.89 [-8.06, -3.72] ✅ | -5.58 [-7.73, -3.43] ✅ |
| flooded_zone | +0.00 [-0.00, +0.01] ⚠️ | +0.00 [-0.02, +0.03] ⚠️ | -1.25 [-1.49, -1.01] ✅ | -1.25 [-1.50, -1.01] ✅ | -1.25 [-1.49, -1.01] ✅ |
| maze_corridors | +0.17 [-0.65, +0.99] ⚠️ | +1.03 [-0.09, +2.16] ⚠️ | -3.29 [-5.13, -1.45] ✅ | -2.43 [-4.01, -0.85] ✅ | -2.26 [-3.94, -0.58] ✅ |

Manipulation check (the tie-break matching is approximate): A* orders its open list by f = g + Manhattan h first, then larger g, and only then by `diag` (D13), and rebuilds its path from the parent pointer of each node's first cheapest generator; AND-OR orders actions forward by BFS distance on the belief, then by `diag` of the intended target. Near belief obstacles the two can still pick different equally short routes. Share of paired runs (successful or not) whose steps and cells explored are both identical (a proxy for the same trajectory):

| comparison | p = 0 | p > 0 |
|---|---|---|
| (a) matched, diagonal | 93.2% (2238/2400) | 92.1% (8839/9600) |
| (b) matched, plain | 89.2% (2140/2400) | 87.9% (8442/9600) |
| (H3) unmatched | 25.8% (620/2400) | 25.6% (2454/9600) |
