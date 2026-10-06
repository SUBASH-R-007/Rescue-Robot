# CLAUDE.md — Rescue Robot Navigation System

CS23533 Foundations of AI mini project (Rajalakshmi Engineering College). A rescue robot plans
with **AND-OR graph search** (AIMA §4.3, cyclic plans) under nondeterministic moves (slip /
skid) and a partially known map, inside a plan → act → observe → update belief → replan loop.
A simulation suite (Pygame simulator, head-to-head comparison, scenarios, stress tests,
Monte Carlo harness) produces the report's results.

**Source of truth:** `docs/plan.md` (file tree §1, decisions D1–D21 in §2, module contracts
§3.1–§3.13: §3.11 viz, §3.12 scenarios / stretch agents / compare, §3.13 experiments). Read it
before changing any public interface. If code and plan disagree, fix one of them in the same
change. `docs/brief.md` is the original specification; `docs/architecture.md` the module
map; `docs/algorithm.md` the algorithm and proofs; `docs/results.md` the measured results
(its §1 lists where every number comes from: the committed `results/` run).

## Commands

The dev venv lives outside OneDrive: `C:/Users/SUBASH/.venvs/rescue-robot` (Python 3.11).
Use its interpreter explicitly (Git Bash path shown), from the repository root:

```bash
PY=C:/Users/SUBASH/.venvs/rescue-robot/Scripts/python.exe
$PY -m pip install -e ".[dev,video]"         # install / refresh (video = imageio-ffmpeg, MP4)
$PY -m pytest -q                              # all tests (viz tests pick the SDL dummy driver)
$PY -m pytest -q -n auto                      # same, parallel (pytest-xdist): the fast way
$PY -m pytest -q -m "not slow"                # skips the few @pytest.mark.slow tests only
$PY -m ruff check .                           # lint (must be clean)
$PY -m pytest --cov=rescue --cov-report=term-missing   # coverage (viz/ excluded; P7 ≥ 85 %)
$PY -m rescue.demo                            # slide 9 sample output (exit 0 on rescue)
$PY -m rescue.demo --save                     # ... and write runs/<timestamp>_andor_<seed>.json
$PY -m rescue.sim --map maps/deck_demo.txt    # interactive simulator (H = help, --save-run)
$PY -m rescue.compare --scenario collapsed_building --seed 7   # head-to-head panes
$PY -m rescue.replay runs/<file>.json         # scrub a saved run (←/→, Home/End)
$PY -m rescue.mapgen random_rubble --rows 15 --cols 15 --seed 0  # print a scenario map
$PY -m rescue.mapgen maze_corridors --seed 1 --out maps/<name>.txt   # save it (+ header)
$PY -m rescue.experiments --quick             # < 60 s, writes results/quick/ (git-ignored)
$PY -m rescue.experiments --config experiments/main.yaml   # full run → results/ (committed)
$PY scripts/make_figures.py                   # simulator figures → docs/figures/ (deterministic)
$PY scripts/make_figures.py --only gif tree --out <dir>    # a subset, somewhere else
```

Headless runs (CI, screenshots, smoke checks): `sim`, `compare` and `replay` take
`--headless --frames N --screenshot PNG` and select the SDL dummy drivers themselves (a
headless run plays automatically; `compare --headless` prints one status line per pane). The
examples write into the git-ignored `recordings/` (parent directories are created):

```bash
$PY -m rescue.sim --map maps/deck_demo.txt --headless --frames 100 --screenshot recordings/out.png
$PY -m rescue.compare --scenario collapsed_building --seed 7 --headless --frames 100 --screenshot recordings/out.png
$PY -m rescue.demo --save runs/demo.json && $PY -m rescue.replay runs/demo.json --headless --frames 60 --screenshot recordings/out.png
```

Other scripts that import pygame directly need `SDL_VIDEODRIVER=dummy` (and
`SDL_AUDIODRIVER=dummy`) in the environment. `--quick` and `make_figures.py` without `--out`
overwrite `results/quick/` and the committed `docs/figures/` respectively; `--out DIR` writes
elsewhere. `--max-tasks N --out DIR` runs an evenly spaced sample of the full config.

## Conventions

- Python ≥ 3.11, `from __future__ import annotations`, type hints everywhere, frozen
  dataclasses where possible, docstrings on public functions, small modules.
- **No magic numbers in logic** — constants live in `src/rescue/config.py`.
- **Coordinates:** internal `Pos = (row, col)` 0-indexed. Anything a human sees (traces, CLI,
  UI labels, plan pretty-print) is 1-indexed via `grid.fmt_pos`.
- **Determinism:** every random draw comes from a seeded `numpy.random.Generator`
  (`world.RngStreams`, mapgen seeds). Same seed ⇒ identical trace. Never use `random` or
  unseeded numpy.
- **Common random numbers:** the world consumes exactly one outcome uniform per step and one
  `(u, permutation)` aftershock draw per step while aftershocks are enabled (none when q = 0 or
  k = 0; the aftershock stream is private, so the outcome stream is unaffected), regardless of
  agent or outcome.
- The UI never contains logic: it subscribes to `runner.StepEvent`s.
- CLI entry points call `rescue.cli_utils.ensure_utf8_stdout()` (Windows consoles + arrows).
- Honest results: never tune experiments so the proposed agent wins; report failed claims.
- Each phase ends with `pytest -q` + `ruff check .` green and a commit `feat(phaseN): …`.

## Key design decisions (details in docs/plan.md §2)

- Transition: intended `1−p`, stay `0.75p`, skid `0.25p` split equally left/right; blocked
  outcomes (wall/debris) stay in place. Slip depends on the cell the robot stands on.
- Planner state = robot position only (belief frozen during a planning call). `Results(s,a)`
  uses the belief with **UNKNOWN assumed open**; equal outcome cells merge with both labels.
- Sensor: Manhattan radius r (default 1) including the robot's own cell, at t=0 and after
  every step. ⇒ with r ≥ 1, `unplanned_states == 0` and `bumps == 0` are **invariants**.
- Bump/collision = entering debris the agent did not know about (D6). Known-debris / wall
  outcomes are "blocked" (anticipated).
- Contradiction = sensed DEBRIS where belief had UNKNOWN or OPEN. Replan modes: `eager`
  (default) or `lazy` (only if the cell is in the current plan envelope).
- AND-OR: transactional solved table (rollback on failed attempts) + semantic `HAS_PROGRESS`
  (state reaches goal / grounded state / proper ancestor) — sound and complete; validated
  against the fixpoint strong-cyclic reference planner by property tests.
- Action ordering: BFS distance of intended target, then diagonal tie-break, then (N,E,S,W).
- Deck root plan reproduces as `E → {arrived: S-plan, slipped: retry E, skid→(2,1): E-plan}`
  (the deck's `N-plan` is not reachable by a distance-ordered planner — see D14).
- Aftershocks hit cells outside the robot's current sensor footprint and never disconnect the
  survivor on the true map.
- Stretch agents `dstar_lite` / `vi_expected` are not in `AGENT_NAMES` (UI `1/2/3`, default
  compare panes); `ALL_AGENT_NAMES` holds all five for `make_agent`, `compare --agents` and the
  harness (D19).
- Experiments: a task seed depends only on `(base_seed, scenario, size, replicate)`, so every
  agent, slip p, sensor radius, replan mode, aftershock q and `planner` override replays the
  same map and outcome stream; paired tests use the map as the unit (D21, plan §3.13).
