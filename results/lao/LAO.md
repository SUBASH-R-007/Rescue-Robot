# LAO\* experiment (plan §3.15.1)

Config `lao`, 500 tasks, 1500 missions, workers 18, commit `cfb50a2`. Descriptive only: no claim of the deck depends on these numbers.

## 1. Plan quality

| steps (both successful) | mean diff [95 % CI] | maps | maps ≠ 0 |
|---|---|---|---|
| `lao_expected` − `vi_expected` | 0.000 [0.000, 0.000] | 500 | 0 |
| `lao_expected` − `andor` | 1.394 [-0.713, 3.501] | 500 | 293 |

Success: `andor` 500/500, `vi_expected` 500/500, `lao_expected` 500/500.

Identical `lao_expected` / `vi_expected` missions (every metric except timing and `nodes_expanded`): 500/500. Both planners compute the same optimum but may break exact ties differently, so this is an observation, not a guarantee.

## 2. Planning time per plan (ms)

Wall-clock means with 95 % CIs, measured with 18 worker process(es) running at once (`run_meta.json`); `LAO*/VI` = ratio of the means, `paired` = mean of the per-mission ratios.

| scenario | size | `andor` | `vi_expected` | `lao_expected` | LAO*/VI | paired |
|---|---|---|---|---|---|---|
| maze_corridors | 10×10 | 4.5 [3.6, 5.5] | 10.3 [9.2, 11.3] | 21.0 [19.2, 22.8] | 2.04 | 2.13 [1.99, 2.28] |
| maze_corridors | 15×15 | 14.0 [11.4, 16.6] | 20.8 [19.3, 22.2] | 40.6 [37.5, 43.8] | 1.95 | 1.98 [1.86, 2.11] |
| maze_corridors | 20×20 | 29.3 [24.6, 34.0] | 35.9 [33.6, 38.3] | 57.9 [52.9, 62.9] | 1.61 | 1.63 [1.51, 1.74] |
| maze_corridors | 25×25 | 49.8 [39.9, 59.7] | 49.0 [45.7, 52.4] | 78.9 [72.0, 85.9] | 1.61 | 1.62 [1.51, 1.73] |
| maze_corridors | 30×30 | 99.1 [77.4, 120.9] | 72.2 [68.2, 76.3] | 114.8 [105.7, 124.0] | 1.59 | 1.58 [1.51, 1.65] |
| random_rubble | 10×10 | 5.3 [4.4, 6.2] | 9.6 [8.8, 10.4] | 22.8 [20.4, 25.1] | 2.38 | 2.40 [2.24, 2.55] |
| random_rubble | 15×15 | 16.7 [13.8, 19.6] | 20.2 [18.8, 21.7] | 43.1 [39.5, 46.7] | 2.13 | 2.15 [2.02, 2.27] |
| random_rubble | 20×20 | 44.5 [34.3, 54.6] | 41.9 [37.6, 46.3] | 72.0 [64.1, 79.8] | 1.72 | 1.78 [1.63, 1.93] |
| random_rubble | 25×25 | 44.3 [33.8, 54.8] | 46.9 [45.2, 48.6] | 86.6 [81.4, 91.8] | 1.85 | 1.84 [1.76, 1.93] |
| random_rubble | 30×30 | 88.7 [63.7, 113.6] | 71.9 [68.5, 75.3] | 125.0 [116.5, 133.5] | 1.74 | 1.74 [1.65, 1.83] |
| all | 10×10 | 4.9 [4.3, 5.6] | 9.9 [9.3, 10.6] | 21.9 [20.4, 23.3] | 2.21 | 2.27 [2.16, 2.37] |
| all | 15×15 | 15.4 [13.4, 17.3] | 20.5 [19.5, 21.5] | 41.9 [39.5, 44.2] | 2.04 | 2.06 [1.98, 2.15] |
| all | 20×20 | 36.9 [31.2, 42.6] | 38.9 [36.4, 41.4] | 64.9 [60.1, 69.7] | 1.67 | 1.70 [1.61, 1.80] |
| all | 25×25 | 47.1 [39.9, 54.2] | 48.0 [46.1, 49.8] | 82.8 [78.4, 87.1] | 1.72 | 1.73 [1.66, 1.80] |
| all | 30×30 | 93.9 [77.6, 110.2] | 72.1 [69.4, 74.7] | 119.9 [113.7, 126.1] | 1.66 | 1.66 [1.60, 1.72] |

## 3. `lao_expected` states expanded per plan

Share = expanded per plan / free cells of the true map (unknown cells are planned as open, so it can exceed 100 %).

| scenario | size | expanded / plan | share of free cells (%) |
|---|---|---|---|
| maze_corridors | 10×10 | 53.5 [50.1, 57.0] | 107.1 [100.3, 113.9] |
| maze_corridors | 15×15 | 131.8 [123.9, 139.7] | 100.6 [94.6, 106.6] |
| maze_corridors | 20×20 | 220.7 [205.2, 236.2] | 107.7 [100.1, 115.2] |
| maze_corridors | 25×25 | 352.7 [328.9, 376.5] | 101.1 [94.2, 107.9] |
| maze_corridors | 30×30 | 532.0 [500.8, 563.2] | 114.4 [107.7, 121.1] |
| random_rubble | 10×10 | 70.1 [66.2, 74.0] | 93.8 [88.9, 98.8] |
| random_rubble | 15×15 | 165.4 [156.9, 174.0] | 96.9 [92.0, 101.9] |
| random_rubble | 20×20 | 294.6 [277.2, 312.1] | 98.1 [92.3, 104.0] |
| random_rubble | 25×25 | 456.2 [432.9, 479.5] | 97.8 [92.8, 102.9] |
| random_rubble | 30×30 | 640.8 [606.6, 675.0] | 94.9 [89.9, 100.0] |
| all | 10×10 | 61.8 [58.8, 64.9] | 100.5 [96.1, 104.8] |
| all | 15×15 | 148.6 [142.0, 155.2] | 98.8 [94.9, 102.6] |
| all | 20×20 | 257.7 [244.0, 271.3] | 102.9 [98.1, 107.7] |
| all | 25×25 | 404.5 [385.1, 423.8] | 99.5 [95.3, 103.7] |
| all | 30×30 | 586.4 [561.2, 611.6] | 104.7 [100.1, 109.2] |
