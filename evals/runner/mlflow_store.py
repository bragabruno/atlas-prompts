"""REG-10 — MLflow tracking store.

Wraps any inner ``ResultsStore`` and additionally logs each eval run to MLflow
so runs are comparable across prompt versions and dataset versions.

Usage
-----
    from evals.runner.mlflow_store import MlflowResultsStore
    from evals.runner.results_store import JsonlResultsStore

    store = MlflowResultsStore(
        inner=JsonlResultsStore(),
        tracking_uri="http://mlflow.atlas.internal:5000",  # or $MLFLOW_TRACKING_URI
        experiment_name="atlas-evals",
    )
    runner = EvalRunner(client=..., store=store)

What gets logged per run
------------------------
- **params**: prompt_version, dataset_name, dataset_version, triggered_by
- **metrics**: per-metric mean value + per-metric pass_rate (e.g. ``exact_match``,
  ``semantic_match``, ``citation_validity``, ``latency_ms``, ``cost_usd``,
  ``llm_judge``); plus ``overall_pass_rate`` and ``all_passed`` (0/1).
- **tags**: run_id (UUID), triggered_by

The inner store's ``save_run()`` is always called first so results are
persisted locally even if the MLflow server is unreachable (non-fatal log
on MLflow errors by default — set ``raise_on_error=True`` for strict mode).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from evals.runner.results_store import (
    EvalRunSummary,
    MetricResult,
    ResultsStore,
    summarise_run,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_EXPERIMENT = "atlas-evals"


class MlflowResultsStore:
    """Decorator store that logs to MLflow after delegating to an inner store.

    Parameters
    ----------
    inner:
        Primary store (e.g. ``JsonlResultsStore``).
    tracking_uri:
        MLflow tracking server URI.  Falls back to the ``MLFLOW_TRACKING_URI``
        environment variable (which MLflow reads automatically) when omitted.
    experiment_name:
        MLflow experiment name (created on first use if absent).
    raise_on_error:
        Re-raise MLflow errors instead of logging and continuing.  Default
        ``False`` so a tracking outage does not break the eval pipeline.
    """

    def __init__(
        self,
        inner: ResultsStore,
        *,
        tracking_uri: str | None = None,
        experiment_name: str = _EXPERIMENT,
        raise_on_error: bool = False,
    ) -> None:
        self._inner = inner
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._raise_on_error = raise_on_error

    # ------------------------------------------------------------------
    # ResultsStore protocol
    # ------------------------------------------------------------------

    def save_run(self, results: list[MetricResult]) -> str:
        run_id = self._inner.save_run(results)
        try:
            self._log_to_mlflow(results)
        except Exception as exc:  # noqa: BLE001
            if self._raise_on_error:
                raise
            log.warning("mlflow_log_failed run_id=%s error=%s", run_id, exc)
        return run_id

    def load_run(self, run_id: str) -> list[MetricResult]:
        return self._inner.load_run(run_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, results: list[MetricResult]) -> None:
        import mlflow

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        # mlflow types set_experiment's signature with an internal Unknown; the
        # call itself is correct, so pin just this member access.
        mlflow.set_experiment(self._experiment_name)  # pyright: ignore[reportUnknownMemberType]

        summary: EvalRunSummary = summarise_run(results)
        first = results[0]

        with mlflow.start_run(run_name=f"{first.prompt_version}@{first.dataset_version}"):
            # ---- params ------------------------------------------------
            mlflow.log_params(
                {
                    "prompt_version": summary.prompt_version,
                    "dataset_name": summary.dataset_name,
                    "dataset_version": summary.dataset_version,
                    "triggered_by": summary.triggered_by,
                }
            )

            # ---- tags --------------------------------------------------
            mlflow.set_tags(
                {
                    "run_id": summary.run_id,
                    "triggered_by": summary.triggered_by,
                }
            )

            # ---- per-metric aggregates ---------------------------------
            metric_values: dict[str, list[float]] = defaultdict(list)
            metric_passes: dict[str, list[bool]] = defaultdict(list)
            for row in results:
                metric_values[row.metric].append(row.value)
                metric_passes[row.metric].append(row.passed)

            for metric_name, values in metric_values.items():
                mean_val = sum(values) / len(values)
                pass_rate = sum(1 for p in metric_passes[metric_name] if p) / len(
                    metric_passes[metric_name]
                )
                mlflow.log_metric(metric_name, mean_val)
                mlflow.log_metric(f"{metric_name}_pass_rate", pass_rate)

            # ---- summary metrics ---------------------------------------
            mlflow.log_metric("overall_pass_rate", summary.pass_rate)
            mlflow.log_metric("all_passed", 1.0 if summary.all_passed else 0.0)

        log.info(
            "mlflow_run_logged experiment=%s prompt=%s pass_rate=%.3f",
            self._experiment_name,
            summary.prompt_version,
            summary.pass_rate,
        )
