"""Local results store (REG-8).

Persists eval run results to a JSONL file under ``eval_runs/eval_results/``.

Design seam
-----------
``ResultsStore`` is a thin Protocol.  ``JsonlResultsStore`` is the local
implementation — no database required.  When the eval_results DB (REG-6,
deferred) ships, add a ``SqliteResultsStore`` or ``PostgresResultsStore``
that satisfies the same interface.

The JSONL format is intentional:
- Human-readable, diff-friendly in git.
- Append-only: each ``save_run`` call appends one JSON line per metric result.
- Zero-dependency: stdlib ``json`` only.

File layout
-----------
    eval_runs/eval_results/<run_id>.jsonl

Each line is a flat JSON object:

    {
      "run_id":          "<uuid>",
      "prompt_version":  "<name>/<semver>",
      "dataset_name":    "<name>",
      "dataset_version": "<version>",
      "triggered_by":    "ci" | "manual:<user>" | ...,
      "created_at":      "<ISO-8601 UTC>",
      "case_index":      <int>,
      "input_snippet":   "<first 120 chars of input>",
      "metric":          "exact_match" | "semantic_match" | "citation_validity"
                         | "latency_ms" | "cost_usd",
      "value":           <float>,
      "baseline_value":  <float | null>,
      "passed":          <bool>
    }

Loading
-------
    store = JsonlResultsStore()
    rows = store.load_run(run_id)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_STORE_ROOT = Path(__file__).parent.parent.parent / "eval_runs" / "eval_results"


class MetricResult(BaseModel):
    """One metric row for a single eval case within a run."""

    run_id: str
    prompt_version: str  # "<name>/<semver>"
    dataset_name: str
    dataset_version: str
    triggered_by: str
    created_at: str  # ISO-8601 UTC
    case_index: int
    input_snippet: str  # first 120 chars of the input
    metric: str
    value: float
    baseline_value: float | None = None
    passed: bool


class EvalRunSummary(BaseModel):
    """Aggregated summary for a completed eval run."""

    run_id: str
    prompt_version: str
    dataset_name: str
    dataset_version: str
    triggered_by: str
    created_at: str
    total_cases: int
    metrics_passed: int
    metrics_total: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    all_passed: bool


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResultsStore(Protocol):
    """Seam for the result-persistence layer.

    Both ``JsonlResultsStore`` (local) and any future DB-backed implementation
    must satisfy this interface.
    """

    def save_run(self, results: list[MetricResult]) -> str:
        """Persist a list of metric results for one run.  Returns the run_id."""
        ...

    def load_run(self, run_id: str) -> list[MetricResult]:
        """Load all metric results for a previously saved run_id.

        Raises
        ------
        FileNotFoundError
            If no results exist for ``run_id``.
        """
        ...


# ---------------------------------------------------------------------------
# JSONL implementation
# ---------------------------------------------------------------------------


class JsonlResultsStore:
    """Append-only JSONL file store for eval results.

    One file per run, stored as ``<root>/<run_id>.jsonl``.

    Parameters
    ----------
    root:
        Directory to store result files.  Created on first write if absent.
        Defaults to ``eval_runs/eval_results/`` at the repo root.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else _STORE_ROOT

    def save_run(self, results: list[MetricResult]) -> str:
        if not results:
            raise ValueError("Cannot save an empty results list")

        run_id = results[0].run_id
        # Validate all rows share the same run_id
        if any(r.run_id != run_id for r in results):
            raise ValueError("All MetricResult rows in a single save_run call must share run_id")

        self._root.mkdir(parents=True, exist_ok=True)
        out_path = self._root / f"{run_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row.model_dump()) + "\n")
        return run_id

    def load_run(self, run_id: str) -> list[MetricResult]:
        path = self._root / f"{run_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"No eval results found for run_id={run_id!r} at {path}")
        results: list[MetricResult] = []
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt results file {path}, line {lineno}: {exc}") from exc
            results.append(MetricResult.model_validate(obj))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Generate a fresh UUID-4 string for a new eval run."""
    return str(uuid.uuid4())


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def summarise_run(results: list[MetricResult]) -> EvalRunSummary:
    """Compute aggregate pass/fail counts over a completed run's results."""
    if not results:
        raise ValueError("Cannot summarise an empty results list")

    first = results[0]
    metrics_passed = sum(1 for r in results if r.passed)
    metrics_total = len(results)
    # total_cases = number of distinct case_index values
    total_cases = len({r.case_index for r in results})

    return EvalRunSummary(
        run_id=first.run_id,
        prompt_version=first.prompt_version,
        dataset_name=first.dataset_name,
        dataset_version=first.dataset_version,
        triggered_by=first.triggered_by,
        created_at=first.created_at,
        total_cases=total_cases,
        metrics_passed=metrics_passed,
        metrics_total=metrics_total,
        pass_rate=round(metrics_passed / metrics_total, 6),
        all_passed=(metrics_passed == metrics_total),
    )
