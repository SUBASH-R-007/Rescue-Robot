# Extension claims — `ext`

- Date of run: 2026-10-07
- Missions (rows of raw.csv): 14400; agents: `search_andor`, `search_astar`
- Config file: `experiments/ext.yaml`
- Base seed: 2026
- Tasks (maps × conditions): 8000
- Workers: 18
- Wall time: 277.1 s
- Git commit: `762a944`
- Methods: success rates with Wilson 95 % intervals; means with Student-t 95 % intervals; paired comparisons on identical maps and outcome / noise streams (a task seed depends only on base seed, scenario, size and replicate) with a paired t-test and a Wilcoxon signed-rank test; pooled paired comparisons use the map as the unit (per-map mean difference; n = maps, not pairs). The noise draws are paired per (step, cell): one uniform per grid cell per step, the same in every mission on that map.
- Verdicts: ✅ supported, ❌ refuted, ⚠️ inconclusive or not measured. Thresholds are constants in `rescue/config.py` fixed before the full run; no experiment was tuned toward a verdict.

## Claims

| # | Claim | Verdict | Measured |
|---|---|---|---|
| X1 | frontier search finds and rescues an unknown survivor in ≥ 95.0% of missions (`search_andor`) | ✅ | 100.0% [99.8%, 100.0%] (1600/1600); Wilson lower bound 99.8%; Σ bumps = 0, Σ unplanned states = 0 |
| X2 | not knowing the survivor's location costs extra steps (paired unknown − known) | ✅ | `search_andor` +76.27 steps [+71.61, +80.93] over 800 maps (1600 pairs), paired t p = 4.7e-146, Wilcoxon p = 6.2e-116, ratio of mean steps ×2.89; `search_astar` +72.47 steps [+67.79, +77.15] over 800 maps (1600 pairs), paired t p = 1.3e-135, Wilcoxon p = 2.9e-112, ratio of mean steps ×2.64 |
| X3 | the greedy nearest-first order is within 10.0% of the optimal tour | ✅ | k = 1: ×1.000 [1.000, 1.000]; k = 2: ×1.000 [1.000, 1.000]; k = 3: ×1.011 [1.008, 1.015]; k = 4: ×1.027 [1.022, 1.033] |
| X4 | with noisy sensing collisions become possible (bumps > 0) but success stays ≥ 95.0% at ε = 0.1 (`search_andor`) | ✅ | collisions ✅: Σ bumps = 2719 in 2400 runs at ε > 0 (1154 missions bumped) vs 0 in 800 runs at ε = 0; success at ε = 0.1 ✅: 100.0% [99.5%, 100.0%] (800/800) |
| X5 | under noise `search_andor` has at most 0.1 more collisions per mission than `search_astar` (non-inferiority, every ε) | ✅ | pooled -0.04 bumps per mission [-0.11, +0.03] over 800 maps (2400 pairs), paired t p = 0.28, Wilcoxon p = 0.32; largest per-ε upper bound +0.06 at ε = 0.2; Σ bumps 2719 vs 2816 |

## Details

### X1. ✅ frontier search finds and rescues an unknown survivor in ≥ 95.0% of missions (`search_andor`)

Block `search`, survivor location unknown (the belief starts all UNKNOWN; the agent knows only how many survivors there are). Pooled over scenarios and p: 100.0% [99.8%, 100.0%] (1600/1600).

Rule: ✅ if the success rate is ≥ 95.0%; ⚠️ if it is below but the Wilson 95 % upper bound reaches 95.0%; ❌ if the upper bound is below it.

| agent | p | scenario | success (Wilson 95 % CI) | first detection step | steps (successful) | Σ bumps |
|---|---|---|---|---|---|---|
| `search_andor` | 0.1 | random_rubble | 100.0% [98.1%, 100.0%] (200/200) | 94.14 [86.47, 101.82] (n = 200) | 95.23 [87.56, 102.91] (n = 200) | 0 |
| `search_andor` | 0.1 | collapsed_building | 100.0% [98.1%, 100.0%] (200/200) | 101.39 [95.06, 107.72] (n = 200) | 102.52 [96.19, 108.85] (n = 200) | 0 |
| `search_andor` | 0.1 | flooded_zone | 100.0% [98.1%, 100.0%] (200/200) | 115.02 [105.80, 124.24] (n = 200) | 116.33 [107.11, 125.55] (n = 200) | 0 |
| `search_andor` | 0.1 | maze_corridors | 100.0% [98.1%, 100.0%] (200/200) | 94.33 [86.32, 102.34] (n = 200) | 95.44 [87.42, 103.45] (n = 200) | 0 |
| `search_andor` | 0.3 | random_rubble | 100.0% [98.1%, 100.0%] (200/200) | 121.85 [111.67, 132.03] (n = 200) | 123.42 [113.22, 133.61] (n = 200) | 0 |
| `search_andor` | 0.3 | collapsed_building | 100.0% [98.1%, 100.0%] (200/200) | 133.34 [124.84, 141.85] (n = 200) | 134.88 [126.37, 143.40] (n = 200) | 0 |
| `search_andor` | 0.3 | flooded_zone | 100.0% [98.1%, 100.0%] (200/200) | 137.68 [126.86, 148.49] (n = 200) | 139.24 [128.44, 150.04] (n = 200) | 0 |
| `search_andor` | 0.3 | maze_corridors | 100.0% [98.1%, 100.0%] (200/200) | 124.46 [113.80, 135.12] (n = 200) | 125.78 [115.13, 136.44] (n = 200) | 0 |
| `search_astar` | 0.1 | random_rubble | 100.0% [98.1%, 100.0%] (200/200) | 94.14 [86.46, 101.82] (n = 200) | 95.22 [87.55, 102.90] (n = 200) | 0 |
| `search_astar` | 0.1 | collapsed_building | 100.0% [98.1%, 100.0%] (200/200) | 101.42 [95.09, 107.75] (n = 200) | 102.55 [96.22, 108.88] (n = 200) | 0 |
| `search_astar` | 0.1 | flooded_zone | 100.0% [98.1%, 100.0%] (200/200) | 114.83 [105.64, 124.03] (n = 200) | 116.15 [106.95, 125.35] (n = 200) | 0 |
| `search_astar` | 0.1 | maze_corridors | 100.0% [98.1%, 100.0%] (200/200) | 94.34 [86.33, 102.35] (n = 200) | 95.44 [87.43, 103.45] (n = 200) | 0 |
| `search_astar` | 0.3 | random_rubble | 100.0% [98.1%, 100.0%] (200/200) | 122.59 [112.43, 132.76] (n = 200) | 124.13 [113.96, 134.30] (n = 200) | 0 |
| `search_astar` | 0.3 | collapsed_building | 100.0% [98.1%, 100.0%] (200/200) | 133.38 [124.86, 141.89] (n = 200) | 134.91 [126.39, 143.43] (n = 200) | 0 |
| `search_astar` | 0.3 | flooded_zone | 100.0% [98.1%, 100.0%] (200/200) | 137.53 [126.75, 148.32] (n = 200) | 139.10 [128.33, 149.87] (n = 200) | 0 |
| `search_astar` | 0.3 | maze_corridors | 100.0% [98.1%, 100.0%] (200/200) | 124.48 [113.82, 135.15] (n = 200) | 125.80 [115.14, 136.46] (n = 200) | 0 |

### X2. ✅ not knowing the survivor's location costs extra steps (paired unknown − known)

Block `search`: every map is run with the survivor location known (pre-marked, the core setting) and unknown, with the same outcome stream. Difference = steps(unknown) − steps(known) on the same map, slip p and agent, both missions successful; the map is the unit (per-map mean over p). Ratio = mean steps unknown / mean steps known over the same pairs.

Rule (on `search_andor`): ✅ if the 95 % CI of the mean difference lies above 0; ❌ if it lies below 0; ⚠️ otherwise.

| agent | scenario | unknown − known steps (95 % CI) | ratio |
|---|---|---|---|
| `search_andor` | random_rubble | +80.06 steps [+71.29, +88.83] over 200 maps (400 pairs), paired t p = 1.2e-43, Wilcoxon p = 3.4e-33 | ×3.74 |
| `search_andor` | collapsed_building | +67.78 steps [+60.23, +75.32] over 200 maps (400 pairs), paired t p = 9e-43, Wilcoxon p = 4.9e-31 | ×2.33 |
| `search_andor` | flooded_zone | +105.13 steps [+95.27, +114.99] over 200 maps (400 pairs), paired t p = 1.9e-52, Wilcoxon p = 3.7e-34 | ×5.64 |
| `search_andor` | maze_corridors | +52.10 steps [+42.54, +61.67] over 200 maps (400 pairs), paired t p = 1.6e-21, Wilcoxon p = 1.2e-20 | ×1.89 |
| `search_andor` | all, p = 0.1 | +67.19 steps [+63.04, +71.35] over 800 maps (800 pairs), paired t p = 1.3e-143, Wilcoxon p = 1.4e-115 | ×2.91 |
| `search_andor` | all, p = 0.3 | +85.34 steps [+80.08, +90.60] over 800 maps (800 pairs), paired t p = 3.1e-144, Wilcoxon p = 4.1e-115 | ×2.88 |
| `search_astar` | random_rubble | +77.01 steps [+68.37, +85.66] over 200 maps (400 pairs), paired t p = 2.5e-42, Wilcoxon p = 2.1e-33 | ×3.36 |
| `search_astar` | collapsed_building | +60.04 steps [+52.61, +67.48] over 200 maps (400 pairs), paired t p = 2.4e-37, Wilcoxon p = 5.1e-30 | ×2.02 |
| `search_astar` | flooded_zone | +103.66 steps [+93.86, +113.46] over 200 maps (400 pairs), paired t p = 5.4e-52, Wilcoxon p = 4.2e-34 | ×5.33 |
| `search_astar` | maze_corridors | +49.17 steps [+39.40, +58.94] over 200 maps (400 pairs), paired t p = 4e-19, Wilcoxon p = 3.6e-18 | ×1.80 |
| `search_astar` | all, p = 0.1 | +63.72 steps [+59.53, +67.90] over 800 maps (800 pairs), paired t p = 3.2e-132, Wilcoxon p = 3.2e-111 | ×2.65 |
| `search_astar` | all, p = 0.3 | +81.22 steps [+75.95, +86.50] over 800 maps (800 pairs), paired t p = 1.6e-134, Wilcoxon p = 8.8e-112 | ×2.63 |

### X3. ✅ the greedy nearest-first order is within 10.0% of the optimal tour

Block `multi`: k survivors at known locations (the core map plus k − 1 extra survivors). Greedy tour = true-map BFS length of the nearest-first visiting order from the start; optimal tour = the shortest over all k! orders; both on the initial true map (the oracle the agent's greedy order approximates on its belief). One value per map; Student-t 95 % CI of the mean ratio. For k ≤ 2 greedy is optimal by construction (both orders share the survivor-to-survivor leg), so only k ≥ 3 can test the claim.

Rule: ✅ if the CI upper bound of the mean ratio is ≤ 1.1 for every k; ❌ if the CI lower bound exceeds 1.1 for some k; ⚠️ otherwise (also when some k has fewer than 2 maps, so no CI).

| k | maps | greedy / optimal (95 % CI) | max ratio | greedy optimal | `search_andor` success | `search_andor` steps per rescue |
|---|---|---|---|---|---|---|
| 1 | 400 | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 100.0% [99.0%, 100.0%] (400/400) | 41.02 [38.24, 43.81] (n = 400) |
| 2 | 400 | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 100.0% [99.0%, 100.0%] (400/400) | 26.93 [25.24, 28.62] (n = 400) |
| 3 | 400 | 1.011 [1.008, 1.015] | 1.250 | 88.2% | 100.0% [99.0%, 100.0%] (400/400) | 22.49 [21.20, 23.78] (n = 400) |
| 4 | 400 | 1.027 [1.022, 1.033] | 1.444 | 72.5% | 100.0% [99.0%, 100.0%] (400/400) | 19.61 [18.60, 20.63] (n = 400) |

### X4. ✅ with noisy sensing collisions become possible (bumps > 0) but success stays ≥ 95.0% at ε = 0.1 (`search_andor`)

Block `noisy`: the debris sensor reports a true DEBRIS cell as OPEN with probability ε (false negatives only); the belief is a Bayes map (`ProbBelief`) for ε > 0. A bump = entering debris the agent did not believe was there (D6); with ε = 0 and r ≥ 1 it is impossible (core invariant).

Rule: part (a) ✅ if `search_andor` bumps at least once at ε > 0 and never at ε = 0 (❌ if it never bumps at ε > 0, or if it bumps at ε = 0, which would break the core invariant and point to a harness or agent bug; ⚠️ if there are no ε = 0 runs); part (b) ✅ if the success rate is ≥ 95.0%; ⚠️ if it is below but the Wilson 95 % upper bound reaches 95.0%; ❌ if the upper bound is below it. Overall ✅ only if both parts are ✅, ❌ if either is ❌, else ⚠️.

| agent | ε | success (Wilson 95 % CI) | Σ bumps | missions bumped | bumps per mission | Σ unplanned | steps (successful) |
|---|---|---|---|---|---|---|---|
| `search_andor` | 0 | 100.0% [99.5%, 100.0%] (800/800) | 0 | 0 | 0.00 [0.00, 0.00] (n = 800) | 0 | 39.48 [37.60, 41.36] (n = 800) |
| `search_andor` | 0.05 | 100.0% [99.5%, 100.0%] (800/800) | 409 | 269 | 0.51 [0.45, 0.57] (n = 800) | 0 | 40.33 [38.37, 42.28] (n = 800) |
| `search_andor` | 0.1 | 100.0% [99.5%, 100.0%] (800/800) | 760 | 385 | 0.95 [0.86, 1.04] (n = 800) | 0 | 40.88 [38.89, 42.86] (n = 800) |
| `search_andor` | 0.2 | 100.0% [99.5%, 100.0%] (800/800) | 1550 | 500 | 1.94 [1.77, 2.10] (n = 800) | 0 | 42.07 [39.98, 44.15] (n = 800) |
| `search_astar` | 0 | 100.0% [99.5%, 100.0%] (800/800) | 0 | 0 | 0.00 [0.00, 0.00] (n = 800) | 7355 | 43.49 [41.50, 45.49] (n = 800) |
| `search_astar` | 0.05 | 100.0% [99.5%, 100.0%] (800/800) | 417 | 277 | 0.52 [0.46, 0.58] (n = 800) | 7875 | 44.24 [42.21, 46.28] (n = 800) |
| `search_astar` | 0.1 | 100.0% [99.5%, 100.0%] (800/800) | 797 | 390 | 1.00 [0.90, 1.09] (n = 800) | 8372 | 45.08 [42.97, 47.19] (n = 800) |
| `search_astar` | 0.2 | 100.0% [99.5%, 100.0%] (800/800) | 1602 | 518 | 2.00 [1.84, 2.16] (n = 800) | 9350 | 46.35 [44.17, 48.54] (n = 800) |

### X5. ✅ under noise `search_andor` has at most 0.1 more collisions per mission than `search_astar` (non-inferiority, every ε)

Block `noisy`, ε > 0: both agents run every map with the same outcome and noise streams. Difference = bumps(search_andor) − bumps(search_astar) per mission, all missions; pooled over ε with the map as the unit (per ε each map appears once), and separately at every ε.

Rule (non-inferiority, margin 0.1 bumps per mission, an absolute margin fixed before the full run): ❌ if the 95 % CI lies above 0 (significantly more collisions), pooled or at any ε; ✅ if the CI upper bound is ≤ 0.1, pooled and at every ε; ⚠️ otherwise. ✅ means "not worse by more than the margin", not "no more collisions".

| ε | search_andor − search_astar (95 % CI) | Σ bumps `search_andor` | Σ bumps `search_astar` |
|---|---|---|---|
| 0.05 | -0.01 bumps [-0.07, +0.05] over 800 maps (800 pairs), paired t p = 0.73, Wilcoxon p = 0.73 | 409 | 417 |
| 0.1 | -0.05 bumps [-0.13, +0.04] over 800 maps (800 pairs), paired t p = 0.28, Wilcoxon p = 0.39 | 760 | 797 |
| 0.2 | -0.07 bumps [-0.19, +0.06] over 800 maps (800 pairs), paired t p = 0.32, Wilcoxon p = 0.5 | 1550 | 1602 |
