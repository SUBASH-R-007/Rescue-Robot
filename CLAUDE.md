# CLAUDE.md — Rescue Robot Navigation System

CS23533 Foundations of AI mini project (Rajalakshmi Engineering College). A rescue robot plans
with **AND-OR graph search** (AIMA §4.3, cyclic plans) under nondeterministic moves (slip /
skid) and a partially known map, inside a plan → act → observe → update belief → replan loop.
A simulation suite (Pygame simulator, head-to-head comparison, scenarios, stress tests,
Monte Carlo harness) produces the report's results.

**Source of truth:** `docs/plan.md` (interface contract + decisions D1–D17). Read it before
changing any public interface. If code and plan disagree, fix one of them in the same change.

## Commands

The dev venv lives outside OneDrive: `C:/Users/SUBASH/.venvs/rescue-robot` (Python 3.11).
Use its interpreter explicitly (Git Bash path shown):

```bash
PY=C:/Users/SUBASH/.venvs/rescue-robot/Scripts/python.exe
$PY -m pip install -e ".[dev,video]"         # install / refresh
$PY -m pytest -q                              # all tests
$PY -m pytest -q -m "not slow"                # fast subset
$PY -m pytest -q -n auto                      # parallel (pytest-xdist)
$PY -m ruff check .                           # lint (must be clean)
$PY -m pytest --cov=rescue --cov-report=term-missing   # coverage (viz/ excluded)
$PY -m rescue.demo                            # slide 9 sample output
$PY -m rescue.sim --map maps/deck_demo.txt    # interactive simulator
$PY -m rescue.compare --scenario collapsed_building --seed 7
$PY -m rescue.replay runs/<file>.json
$PY -m rescue.experiments --quick             # < 60 s
$PY -m rescue.experiments --config experiments/main.yaml
```

Headless pygame (tests, CI, recording): set `SDL_VIDEODRIVER=dummy` (and `SDL_AUDIODRIVER=dummy`).

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
  `(u, permutation)` aftershock draw per step, regardless of agent or outcome.
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
