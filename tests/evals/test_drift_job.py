"""POL-3 — Nightly drift eval tests.

All tests use ``JsonlShadowSource`` with a tmp JSONL file and mock out
MLflow so no server is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.drift.shadow_source import JsonlShadowSource, ShadowRecord
from evals.drift.drift_job import (
    DriftReport,
    VersionDrift,
    _shadow_to_metric_results,
    run_drift_eval,
)
from evals.runner.results_store import MetricResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _shadow_line(
    prompt_version: str = "regdoc-qa/1.0.0",
    latency_ms: float = 150.0,
    cost_usd: float = 0.001,
) -> str:
    return json.dumps({
        "prompt_version": prompt_version,
        "input": "What does clause 4.2 require?",
        "output": "Clause 4.2 requires ...",
        "model": "claude-sonnet-4-6",
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    })


@pytest.fixture()
def shadow_file(tmp_path: Path) -> Path:
    p = tmp_path / "shadow.jsonl"
    p.write_text("\n".join([
        _shadow_line("regdoc-qa/1.0.0", latency_ms=150.0, cost_usd=0.001),
        _shadow_line("regdoc-qa/1.0.0", latency_ms=160.0, cost_usd=0.0012),
        _shadow_line("regdoc-qa/2.0.0", latency_ms=200.0, cost_usd=0.002),
    ]))
    return p


# ---------------------------------------------------------------------------
# shadow_source
# ---------------------------------------------------------------------------

def test_jsonl_shadow_source_yields_records(shadow_file: Path) -> None:
    source = JsonlShadowSource(shadow_file)
    records = list(source.records())
    assert len(records) == 3
    assert all(isinstance(r, ShadowRecord) for r in records)


def test_jsonl_shadow_source_respects_limit(shadow_file: Path) -> None:
    source = JsonlShadowSource(shadow_file)
    records = list(source.records(limit=2))
    assert len(records) == 2


# ---------------------------------------------------------------------------
# _shadow_to_metric_results
# ---------------------------------------------------------------------------

def test_shadow_to_metric_results_produces_latency_and_cost() -> None:
    records = [
        ShadowRecord(
            prompt_version="regdoc-qa/1.0.0",
            input="Q",
            output="A",
            model="m",
            latency_ms=100.0,
            cost_usd=0.001,
        )
    ]
    rows = _shadow_to_metric_results(records, run_id="r1", created_at="2026-06-10T00:00:00+00:00")
    metrics = {r.metric: r.value for r in rows}
    assert metrics["latency_ms"] == 100.0
    assert metrics["cost_usd"] == 0.001


# ---------------------------------------------------------------------------
# run_drift_eval
# ---------------------------------------------------------------------------

def _mock_mlflow() -> MagicMock:
    mock = MagicMock()
    mock.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


def test_run_drift_eval_returns_report_per_version(shadow_file: Path) -> None:
    source = JsonlShadowSource(shadow_file)
    with patch.dict("sys.modules", {"mlflow": _mock_mlflow()}):
        report = run_drift_eval(source)
    assert isinstance(report, DriftReport)
    versions = {v.prompt_version for v in report.versions}
    assert "regdoc-qa/1.0.0" in versions
    assert "regdoc-qa/2.0.0" in versions


def test_run_drift_eval_no_alerts_without_baseline(shadow_file: Path) -> None:
    source = JsonlShadowSource(shadow_file)
    with patch.dict("sys.modules", {"mlflow": _mock_mlflow()}):
        report = run_drift_eval(source)
    assert not report.has_alerts


def test_run_drift_eval_alert_fires_on_regression(tmp_path: Path) -> None:
    f = tmp_path / "shadow.jsonl"
    # candidate: latency_ms=300 (big regression vs baseline of 100)
    f.write_text(_shadow_line("regdoc-qa/1.0.0", latency_ms=300.0, cost_usd=0.001))

    # baseline at 100ms
    baseline_row = MetricResult(
        run_id="base-run",
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="shadow",
        dataset_version="live",
        triggered_by="drift-eval",
        created_at="2026-06-09T00:00:00+00:00",
        case_index=0,
        input_snippet="Q",
        metric="latency_ms",
        value=100.0,
        baseline_value=None,
        passed=True,
    )

    source = JsonlShadowSource(f)
    with patch.dict("sys.modules", {"mlflow": _mock_mlflow()}):
        report = run_drift_eval(
            source,
            baseline_results={"regdoc-qa/1.0.0": [baseline_row]},
            alert_threshold_pct=10.0,
        )
    assert report.has_alerts


def test_run_drift_eval_empty_source_returns_empty_report(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    source = JsonlShadowSource(f)
    with patch.dict("sys.modules", {"mlflow": _mock_mlflow()}):
        report = run_drift_eval(source)
    assert report.versions == []
    assert not report.has_alerts
