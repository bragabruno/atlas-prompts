"""gate.py — Regression gate CLI (REG-11).

Compares a candidate eval run's metrics to a production baseline and exits
non-zero if any BLOCKING metric regresses beyond the configured threshold.

Usage
-----
    python gate.py <candidate_run_id> <baseline_run_id> [--store-root PATH]

    python gate.py <candidate_run_id> --no-baseline   # first-run (always passes)

Arguments
---------
candidate_run_id
    Run ID for the candidate (produced by the eval runner / REG-8).
baseline_run_id
    Run ID for the production baseline.  Omit (with --no-baseline) for the
    first ever eval run where there is no production baseline to compare.

Options
-------
--store-root PATH
    Root directory of the results store (default: eval_runs/eval_results/).
--no-baseline
    Skip baseline comparison — gate always passes.  Use only for the first run
    of a prompt with no prior production record.

Exit codes
----------
0   Gate passed — no blocking regressions.
1   Gate failed — at least one BLOCKING metric regressed beyond its threshold.
2   Usage / argument error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.gate.comparator import run_gate
from evals.gate.gate_config import default_gate_config
from evals.runner.results_store import JsonlResultsStore


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gate.py",
        description="Regression gate: compare candidate metrics to production baseline.",
    )
    parser.add_argument("candidate_run_id", help="Candidate eval run ID.")
    parser.add_argument(
        "baseline_run_id",
        nargs="?",
        default=None,
        help="Production baseline eval run ID (omit with --no-baseline).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison (first-run mode). Gate always passes.",
    )
    parser.add_argument(
        "--store-root",
        default=None,
        help="Override the results store root directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns exit code (0 = pass, 1 = fail, 2 = usage error)."""
    args = _parse_args(argv)

    if args.baseline_run_id is None and not args.no_baseline:
        print(
            "gate.py: error: provide baseline_run_id or pass --no-baseline for first-run mode.",
            file=sys.stderr,
        )
        return 2

    store_root = Path(args.store_root) if args.store_root else None
    store = JsonlResultsStore(root=store_root) if store_root else JsonlResultsStore()

    # Load candidate results
    try:
        candidate_results = store.load_run(args.candidate_run_id)
    except FileNotFoundError as exc:
        print(f"gate.py: error loading candidate run: {exc}", file=sys.stderr)
        return 2

    # Load baseline results (may be None)
    baseline_results = None
    baseline_run_id: str | None = None
    if not args.no_baseline and args.baseline_run_id:
        try:
            baseline_results = store.load_run(args.baseline_run_id)
            baseline_run_id = args.baseline_run_id
        except FileNotFoundError as exc:
            print(f"gate.py: error loading baseline run: {exc}", file=sys.stderr)
            return 2

    config = default_gate_config()
    result = run_gate(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        config=config,
        candidate_run_id=args.candidate_run_id,
        baseline_run_id=baseline_run_id,
    )

    print(result.format_table())

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
