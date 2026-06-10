"""POL-3 — Nightly shadow/drift eval job.

Samples recent production traffic from ``atlas.shadow.v1``, converts it to
eval-compatible ``MetricResult`` rows (exact match not available — latency and
cost are the primary drift signals), compares against a stored baseline, logs
the drift report to MLflow, and optionally fires an alert when a score drops
beyond the configured threshold.

Pipeline
--------
1. Pull shadow records from the source (Kafka or JSONL in tests).
2. Convert each record to a ``MetricResult`` using direct-value metrics
   (latency_ms, cost_usd) — no LLM inference needed for drift tracking.
3. Group by ``prompt_version`` and compare each version against its baseline.
4. Log to MLflow via ``MlflowResultsStore`` (wraps ``JsonlResultsStore``).
5. Return a ``DriftReport`` summarising all comparisons.  Callers check
   ``report.has_alerts`` to decide whether to fire downstream notifications.

Alert threshold
---------------
A score alert fires when any metric's candidate mean regresses beyond
``alert_threshold_pct`` relative to the baseline (default 10%).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evals.drift.shadow_source import ShadowRecord, ShadowSource
from evals.gate.comparator import GateResult, run_gate
from evals.runner.mlflow_store import MlflowResultsStore
from evals.runner.results_store import JsonlResultsStore, MetricResult

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_DEFAULT_ALERT_THRESHOLD_PCT = 10.0


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class VersionDrift:
    """Drift comparison result for one prompt version."""

    prompt_version: str
    run_id: str
    n_samples: int
    gate_result: GateResult
    alerted: bool


@dataclass
class DriftReport:
    """Aggregated drift report across all prompt versions evaluated."""

    versions: list[VersionDrift] = field(default_factory=list)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat()
    )

    @property
    def has_alerts(self) -> bool:
        return any(v.alerted for v in self.versions)

    def format_summary(self) -> str:
        lines = [
            "",
            f"  Drift report — {self.evaluated_at}",
            f"  Versions evaluated: {len(self.versions)}",
            "",
        ]
        for v in self.versions:
            status = "ALERT" if v.alerted else ("FAIL" if not v.gate_result.passed else "OK")
            lines.append(f"  {v.prompt_version:<35} n={v.n_samples:<6} status={status}")
            lines.append(v.gate_result.format_table())
        if self.has_alerts:
            lines.append("  *** DRIFT ALERT: score drop detected — check MLflow dashboard ***")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shadow record → MetricResult conversion
# ---------------------------------------------------------------------------


def _shadow_to_metric_results(
    records: list[ShadowRecord],
    run_id: str,
    created_at: str,
) -> list[MetricResult]:
    """Convert shadow records to MetricResult rows for latency and cost."""
    results: list[MetricResult] = []
    for idx, rec in enumerate(records):
        common = {
            "run_id": run_id,
            "prompt_version": rec.prompt_version,
            "dataset_name": "shadow",
            "dataset_version": "live",
            "triggered_by": "drift-eval",
            "created_at": created_at,
            "case_index": idx,
            "input_snippet": rec.input[:120],
            "baseline_value": None,
            "passed": True,
        }
        results.append(MetricResult(**common, metric="latency_ms", value=rec.latency_ms))
        results.append(MetricResult(**common, metric="cost_usd", value=rec.cost_usd))
    return results


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def run_drift_eval(
    source: ShadowSource,
    *,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "atlas-drift",
    alert_threshold_pct: float = _DEFAULT_ALERT_THRESHOLD_PCT,
    sample_limit: int | None = 1000,
    baseline_results: dict[str, list[MetricResult]] | None = None,
) -> DriftReport:
    """Run the nightly drift eval and return a ``DriftReport``.

    Parameters
    ----------
    source:
        Shadow traffic source (Kafka in production, JSONL in tests).
    mlflow_tracking_uri:
        MLflow server URI.  Falls back to ``MLFLOW_TRACKING_URI`` env var.
    mlflow_experiment:
        MLflow experiment name (default ``atlas-drift``).
    alert_threshold_pct:
        Trigger an alert when any metric regresses by more than this % vs baseline.
    sample_limit:
        Maximum number of shadow records to consume per run.
    baseline_results:
        Mapping of ``prompt_version`` → list of ``MetricResult`` from a prior
        production run.  If None, the gate runs without baselines (first-run mode).
    """
    created_at = datetime.now(tz=UTC).isoformat()

    # ---- 1. Consume shadow records ----------------------------------------
    all_records: list[ShadowRecord] = list(source.records(limit=sample_limit))
    log.info("drift_eval_start n_records=%d", len(all_records))

    if not all_records:
        log.warning("drift_eval_no_records source=%s", type(source).__name__)
        return DriftReport()

    # ---- 2. Group by prompt_version ---------------------------------------
    by_version: dict[str, list[ShadowRecord]] = {}
    for rec in all_records:
        by_version.setdefault(rec.prompt_version, []).append(rec)

    inner_store = JsonlResultsStore()
    mlflow_store = MlflowResultsStore(
        inner=inner_store,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        raise_on_error=False,
    )

    report = DriftReport(evaluated_at=created_at)

    for version, records in by_version.items():
        run_id = str(uuid.uuid4())
        metric_rows = _shadow_to_metric_results(records, run_id, created_at)

        # ---- 3. Persist + log to MLflow -----------------------------------
        mlflow_store.save_run(metric_rows)

        # ---- 4. Compare against baseline ----------------------------------
        base_rows = baseline_results.get(version) if baseline_results else None
        gate_result = run_gate(
            candidate_results=metric_rows,
            baseline_results=base_rows,
            candidate_run_id=run_id,
            baseline_run_id=base_rows[0].run_id if base_rows else None,
        )

        # ---- 5. Alert check -----------------------------------------------
        alerted = False
        for diff in gate_result.diffs:
            if diff.baseline_mean is not None and diff.baseline_mean > 0:
                pct_change = abs(diff.delta / diff.baseline_mean) * 100
                if diff.regression and pct_change > alert_threshold_pct:
                    alerted = True
                    log.warning(
                        "drift_alert version=%s metric=%s delta=%.4f (%.1f%%)",
                        version,
                        diff.metric,
                        diff.delta,
                        pct_change,
                    )

        report.versions.append(VersionDrift(
            prompt_version=version,
            run_id=run_id,
            n_samples=len(records),
            gate_result=gate_result,
            alerted=alerted,
        ))
        log.info(
            "drift_version_done version=%s n=%d passed=%s alerted=%s",
            version,
            len(records),
            gate_result.passed,
            alerted,
        )

    log.info(
        "drift_eval_complete versions=%d has_alerts=%s",
        len(report.versions),
        report.has_alerts,
    )
    return report
