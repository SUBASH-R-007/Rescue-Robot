"""``python -m rescue.experiments``: run the Monte Carlo harness end to end.

Examples::

    python -m rescue.experiments --quick                       # < 60 s, writes results/quick/
    python -m rescue.experiments --config experiments/main.yaml
    python -m rescue.experiments --config experiments/main.yaml --max-tasks 200 --out /tmp/x

Pipeline: :func:`~rescue.experiments.run.run_experiments` (``raw.csv`` + ``run_meta.json``) →
``stats.write_summary`` (``summary.csv``) → ``hypotheses.write_hypotheses`` (``HYPOTHESES.md``)
→ ``plots.make_all_plots`` (``plots/*.png``, unless ``--no-plots``), then a short console
summary (success of the three main agents at each slip p, wall time).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rescue.cli_utils import ensure_utf8_stdout
from rescue.experiments.spec import (
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    QUICK_CONFIG,
    QUICK_OUT_DIR,
    SLIP_SWEEP_EXPERIMENT,
    load_experiment_config,
    resolve_config_path,
)

#: Output file names inside the output directory.
SUMMARY_FILE: str = "summary.csv"
HYPOTHESES_FILE: str = "HYPOTHESES.md"
PLOTS_DIR: str = "plots"
#: Agents shown in the console summary (brief §5's three compared agents).
CONSOLE_AGENTS: tuple[str, ...] = ("andor", "astar_fixed", "astar_replan")
#: Width of the agent column in the console summary.
CONSOLE_AGENT_WIDTH: int = 14


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the experiment harness."""
    parser = argparse.ArgumentParser(
        prog="python -m rescue.experiments",
        description="Monte Carlo experiments: agents × scenarios × slip p × sizes with common "
        "random numbers; writes raw.csv, summary.csv, HYPOTHESES.md and plots.",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--config", metavar="PATH", default=None,
                       help=f"experiment YAML (default {DEFAULT_CONFIG.as_posix()})")
    which.add_argument("--quick", action="store_true",
                       help=f"use {QUICK_CONFIG.as_posix()} (N = 20, one size; < 60 s) and "
                            f"write to {QUICK_OUT_DIR.as_posix()}/ by default")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="worker processes (default: config value, else CPUs - 2; "
                             "1 = in-process)")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help=f"output directory (default {DEFAULT_OUT_DIR.as_posix()}, "
                             f"or {QUICK_OUT_DIR.as_posix()} with --quick)")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    parser.add_argument("--max-tasks", type=int, default=None, metavar="N",
                        help="run an evenly spaced sample of at most N tasks (smoke tests, "
                             "run-time estimates)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return parser


def _truthy(value: Any) -> bool:
    """Success flag from a typed or textual CSV value."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def console_summary(rows: Sequence[dict[str, Any]], wall_s: float) -> str:
    """Success % of the main agents at each slip p (slip-sweep block, all sizes and scenarios
    pooled) plus the run size and wall time."""
    sweep = [r for r in rows if r.get("experiment") == SLIP_SWEEP_EXPERIMENT] or list(rows)
    ps = sorted({float(r["slip_p"]) for r in sweep})
    agents = [a for a in CONSOLE_AGENTS if any(r["agent"] == a for r in sweep)]
    lines = [
        f"Success rate, {SLIP_SWEEP_EXPERIMENT if sweep is not rows else 'all runs'} "
        "(scenarios and sizes pooled):",
        " " * CONSOLE_AGENT_WIDTH + "".join(f"  p={p:<5g}" for p in ps),
    ]
    for agent in agents:
        cells = []
        for p in ps:
            runs = [r for r in sweep if r["agent"] == agent and float(r["slip_p"]) == p]
            ok = sum(_truthy(r["success"]) for r in runs)
            rate = 100.0 * ok / len(runs) if runs else math.nan
            cells.append(f"  {rate:6.1f}%" if runs else "      n/a")
        lines.append(f"{agent:<{CONSOLE_AGENT_WIDTH}}" + "".join(cells))
    lines.append(f"{len(rows)} runs, total wall time {wall_s:.1f} s")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.workers is not None and args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2
    if args.max_tasks is not None and args.max_tasks < 1:
        print("error: --max-tasks must be >= 1", file=sys.stderr)
        return 2
    config_arg = QUICK_CONFIG if args.quick else Path(args.config or DEFAULT_CONFIG)
    config_path = resolve_config_path(config_arg)
    try:
        cfg = load_experiment_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else (QUICK_OUT_DIR if args.quick else DEFAULT_OUT_DIR)

    from rescue.experiments import hypotheses, stats
    from rescue.experiments.run import run_experiments

    t0 = time.perf_counter()
    raw_path = run_experiments(
        cfg,
        out_dir,
        workers=args.workers,
        progress=not args.quiet,
        max_tasks=args.max_tasks,
        config_path=config_path.as_posix(),
    )
    rows = stats.read_raw(raw_path)
    stats.write_summary(rows, out_dir / SUMMARY_FILE)
    hypotheses.write_hypotheses(rows, out_dir / HYPOTHESES_FILE)
    written = [raw_path, out_dir / SUMMARY_FILE, out_dir / HYPOTHESES_FILE]
    if not args.no_plots:
        from rescue.experiments import plots

        written.extend(plots.make_all_plots(rows, out_dir / PLOTS_DIR))
    wall_s = time.perf_counter() - t0
    print(console_summary(rows, wall_s))
    print(f"Wrote {len(written)} files to {out_dir.as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
