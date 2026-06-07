"""Regression comparator (REG-11).

Compares a *candidate* eval run against a *production baseline* run.  Each
metric is compared by its mean value across all cases.  The result is a list of
:class:`MetricDiff` objects — one per metric spec in the :class:`GateConfig`.
:func:`run_gate` returns a :class:`GateResult` that aggregates the diffs and
tells the caller whether to exit non-zero.

Typical usage
-------------
    from evals.gate.comparator import run_gate
    from evals.gate.gate_config import default_gate_config

    result = run_gate(
        candidate_results=candidate_rows,
        baseline_results=baseline_rows,
        config=default_gate_config(),
    )
    print(result.format_table())
    sys.exit(0 if result.passed else 1)
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.gate.gate_config import GateConfig, MetricSpec, default_gate_config
from evals.runner.results_store import MetricResult  # noqa: TC001

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDiff:
    """Comparison result for one metric."""

    metric: str
    candidate_mean: float
    baseline_mean: float | None
    delta: float
    """candidate_mean − baseline_mean (signed; None baseline → 0.0)."""
    regression: bool
    """True when the candidate exceeds the allowed threshold."""
    blocking: bool
    """Whether this metric is configured as blocking."""
    threshold: float
    higher_is_better: bool

    @property
    def status(self) -> str:
        if self.baseline_mean is None:
            return "NO_BASELINE"
        if self.regression:
            return "FAIL" if self.blocking else "WARN"
        return "PASS"


@dataclass(frozen=True)
class GateResult:
    """Aggregated result of running the regression gate."""

    diffs: list[MetricDiff]
    candidate_run_id: str
    baseline_run_id: str | None

    @property
    def passed(self) -> bool:
        """True iff no blocking metric regressed."""
        return not any(d.regression and d.blocking for d in self.diffs)

    @property
    def blocking_failures(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.regression and d.blocking]

    @property
    def advisory_warnings(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.regression and not d.blocking]

    def format_table(self) -> str:
        """Return a human-readable metric-diff table."""
        lines: list[str] = [
            "",
            f"  Candidate run : {self.candidate_run_id}",
            f"  Baseline run  : {self.baseline_run_id or 'none (first run)'}",
            "",
            f"  {'Metric':<22} {'Candidate':>10} {'Baseline':>10} {'Delta':>10}  {'Status':<12}",
            f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 12}",
        ]
        for d in self.diffs:
            baseline_str = f"{d.baseline_mean:.6f}" if d.baseline_mean is not None else "     n/a"
            lines.append(
                f"  {d.metric:<22} {d.candidate_mean:>10.6f} {baseline_str:>10}"
                f" {d.delta:>+10.6f}  {d.status:<12}"
            )
        lines.append("")
        if self.passed:
            lines.append("  Gate: PASS — no blocking regressions detected.")
        else:
            lines.append("  Gate: FAIL — blocking regressions detected:")
            for f_ in self.blocking_failures:
                lines.append(
                    f"    [BLOCKING] {f_.metric}: delta={f_.delta:+.6f}, threshold={f_.threshold}"
                )
        if self.advisory_warnings:
            lines.append("  Advisory warnings:")
            for w in self.advisory_warnings:
                lines.append(
                    f"    [ADVISORY] {w.metric}: delta={w.delta:+.6f}, threshold={w.threshold}"
                )
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def _mean_metric(results: list[MetricResult], metric: str) -> float | None:
    """Return the mean ``value`` for ``metric`` across all rows, or None if absent."""
    values = [r.value for r in results if r.metric == metric]
    if not values:
        return None
    return sum(values) / len(values)


_FLOAT_EPS = 1e-9
"""Epsilon used to absorb floating-point rounding when comparing against thresholds.

Without this, a drop of exactly ``threshold`` may be flagged as a regression
due to binary floating-point representation (e.g. 0.90 - 0.85 = 5.0e-2 + 4.4e-17).
"""


def _is_regression(
    spec: MetricSpec,
    candidate_mean: float,
    baseline_mean: float,
) -> bool:
    """Return True if the candidate regresses *beyond* ``spec.threshold``.

    A delta equal to the threshold is **not** a regression — the gate is
    exclusive on the boundary.
    """
    if spec.higher_is_better:
        # regression = candidate dropped below baseline by more than threshold
        delta = candidate_mean - baseline_mean
        return delta < -(spec.threshold + _FLOAT_EPS)
    else:
        # regression = candidate exceeded baseline by more than threshold
        delta = candidate_mean - baseline_mean
        return delta > spec.threshold + _FLOAT_EPS


def compare_runs(
    candidate_results: list[MetricResult],
    baseline_results: list[MetricResult] | None,
    config: GateConfig,
    candidate_run_id: str = "",
    baseline_run_id: str | None = None,
) -> GateResult:
    """Compare candidate vs baseline and return a :class:`GateResult`.

    Parameters
    ----------
    candidate_results:
        All :class:`~evals.runner.results_store.MetricResult` rows from the
        candidate eval run.
    baseline_results:
        All rows from the production baseline run, or None for first-run
        (no baseline).  When None every metric is reported as NO_BASELINE
        and the gate always passes.
    config:
        :class:`~evals.gate.gate_config.GateConfig` describing which metrics
        are blocking and what thresholds apply.
    candidate_run_id:
        Informational identifier for the candidate run (included in the diff
        table header).
    baseline_run_id:
        Informational identifier for the baseline run (included in the diff
        table header).  May be None.
    """
    diffs: list[MetricDiff] = []

    for spec in config.metrics:
        cand_mean = _mean_metric(candidate_results, spec.metric)
        base_mean = _mean_metric(baseline_results or [], spec.metric) if baseline_results else None

        if cand_mean is None:
            # metric not present in candidate — treat as perfect pass with zero mean
            cand_mean = 0.0

        if base_mean is None:
            # no baseline data → no regression possible
            delta = 0.0
            regression = False
        else:
            delta = cand_mean - base_mean
            regression = _is_regression(spec, cand_mean, base_mean)

        diffs.append(
            MetricDiff(
                metric=spec.metric,
                candidate_mean=cand_mean,
                baseline_mean=base_mean,
                delta=delta,
                regression=regression,
                blocking=spec.blocking,
                threshold=spec.threshold,
                higher_is_better=spec.higher_is_better,
            )
        )

    return GateResult(
        diffs=diffs,
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
    )


def run_gate(
    candidate_results: list[MetricResult],
    baseline_results: list[MetricResult] | None = None,
    config: GateConfig | None = None,
    candidate_run_id: str = "",
    baseline_run_id: str | None = None,
) -> GateResult:
    """Convenience wrapper around :func:`compare_runs`.

    Uses :func:`~evals.gate.gate_config.default_gate_config` when ``config`` is None.
    """
    return compare_runs(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        config=config if config is not None else default_gate_config(),
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
    )
