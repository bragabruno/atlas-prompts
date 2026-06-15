"""REG-10 — MlflowResultsStore tests.

MLflow is mocked out entirely so no server or installation is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from atlas_prompts.evals.runner.mlflow_store import MlflowResultsStore
from atlas_prompts.evals.runner.results_store import MetricResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RUN_ID = "test-run-001"
_CREATED = "2026-06-10T00:00:00+00:00"


def _make_result(metric: str, value: float, passed: bool) -> MetricResult:
    return MetricResult(
        run_id=_RUN_ID,
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="regdoc-golden",
        dataset_version="1.0.0",
        triggered_by="ci",
        created_at=_CREATED,
        case_index=0,
        input_snippet="What are the requirements?",
        metric=metric,
        value=value,
        baseline_value=None,
        passed=passed,
    )


_RESULTS = [
    _make_result("exact_match", 1.0, True),
    _make_result("latency_ms", 120.0, True),
    _make_result("cost_usd", 0.001, True),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_run_delegates_to_inner() -> None:
    inner = MagicMock()
    inner.save_run.return_value = _RUN_ID

    with patch.dict("sys.modules", {"mlflow": MagicMock()}):
        store = MlflowResultsStore(inner=inner)
        result = store.save_run(_RESULTS)

    inner.save_run.assert_called_once_with(_RESULTS)
    assert result == _RUN_ID


def test_save_run_logs_params_and_metrics() -> None:
    inner = MagicMock()
    inner.save_run.return_value = _RUN_ID

    mock_mlflow = MagicMock()
    mock_run_ctx = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        store = MlflowResultsStore(inner=inner, experiment_name="test-experiment")
        store.save_run(_RESULTS)

    mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
    mock_mlflow.log_params.assert_called_once()
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["prompt_version"] == "regdoc-qa/1.0.0"
    assert params["dataset_name"] == "regdoc-golden"
    assert params["triggered_by"] == "ci"

    # Should log overall_pass_rate and all_passed
    metric_calls = {c[0][0]: c[0][1] for c in mock_mlflow.log_metric.call_args_list}
    assert "overall_pass_rate" in metric_calls
    assert "all_passed" in metric_calls
    assert metric_calls["all_passed"] == 1.0


def test_mlflow_error_is_swallowed_by_default() -> None:
    inner = MagicMock()
    inner.save_run.return_value = _RUN_ID

    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.side_effect = RuntimeError("server down")

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        store = MlflowResultsStore(inner=inner)
        result = store.save_run(_RESULTS)  # should not raise

    assert result == _RUN_ID


def test_mlflow_error_is_raised_when_strict() -> None:
    inner = MagicMock()
    inner.save_run.return_value = _RUN_ID

    mock_mlflow = MagicMock()
    mock_mlflow.set_experiment.side_effect = RuntimeError("server down")

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        store = MlflowResultsStore(inner=inner, raise_on_error=True)
        with pytest.raises(RuntimeError, match="server down"):
            store.save_run(_RESULTS)


def test_load_run_delegates_to_inner() -> None:
    inner = MagicMock()
    inner.load_run.return_value = _RESULTS

    store = MlflowResultsStore(inner=inner)
    result = store.load_run(_RUN_ID)

    inner.load_run.assert_called_once_with(_RUN_ID)
    assert result == _RESULTS


def test_tracking_uri_is_set_when_provided() -> None:
    inner = MagicMock()
    inner.save_run.return_value = _RUN_ID

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        store = MlflowResultsStore(inner=inner, tracking_uri="http://mlflow.local:5000")
        store.save_run(_RESULTS)

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow.local:5000")
