"""Gate configuration types (REG-11).

A ``GateConfig`` lists every metric the gate should evaluate.  Each entry is a
``MetricSpec`` that describes:

- ``metric``      — metric name matching the ``MetricResult.metric`` field from
                    the results store (e.g. ``"exact_match"``, ``"latency_ms"``).
- ``blocking``    — if True a regression beyond ``threshold`` causes a non-zero
                    exit; if False the metric is advisory-only (reported but never
                    blocks).
- ``threshold``   — the maximum *allowed regression* expressed as an absolute
                    delta of the metric value.  Positive = the candidate may be
                    *worse* than the baseline by at most this amount before it
                    triggers a failure.  Use 0.0 to require the candidate is at
                    least as good as the baseline.
- ``higher_is_better`` — when True a decrease in the mean metric value is a
                    regression (e.g. pass_rate, exact_match).  When False an
                    *increase* is a regression (e.g. latency_ms, cost_usd).

Default gate
-----------
``default_gate_config()`` returns a ready-to-use config that matches the metrics
defined in ADR-017 and implemented in REG-8.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    """Specification for one metric in the gate."""

    metric: str
    """Metric name (matches ``MetricResult.metric``)."""

    blocking: bool
    """True  → regression blocks the gate (non-zero exit).
    False → regression is reported but advisory only."""

    threshold: float
    """Maximum allowed regression (absolute delta).
    e.g. 0.05 on pass_rate means the candidate may drop by at most 5 pp."""

    higher_is_better: bool = True
    """True  → lower candidate mean = regression (quality metrics).
    False → higher candidate mean = regression (cost/latency metrics)."""


@dataclass
class GateConfig:
    """Full gate configuration.

    Parameters
    ----------
    metrics:
        Ordered list of :class:`MetricSpec` objects.  The gate evaluates them
        in order; *all* blocking regressions are collected before deciding the
        exit code (no early exit).
    """

    metrics: list[MetricSpec] = field(default_factory=list)


def default_gate_config() -> GateConfig:
    """Return the canonical ADR-017 gate configuration.

    Blocking metrics
    ----------------
    - exact_match    pass-rate: candidate must not drop by more than 5 pp.
    - citation_validity pass-rate: candidate must not drop by more than 5 pp.
    - semantic_match mean score: candidate must not drop by more than 0.05.

    Advisory metrics (reported, never block)
    ----------------------------------------
    - latency_ms    mean value: advisory regression flag if candidate > baseline.
    - cost_usd      mean value: advisory regression flag if candidate > baseline.
    """
    return GateConfig(
        metrics=[
            MetricSpec(
                metric="exact_match",
                blocking=True,
                threshold=0.05,
                higher_is_better=True,
            ),
            MetricSpec(
                metric="citation_validity",
                blocking=True,
                threshold=0.05,
                higher_is_better=True,
            ),
            MetricSpec(
                metric="semantic_match",
                blocking=True,
                threshold=0.05,
                higher_is_better=True,
            ),
            MetricSpec(
                metric="latency_ms",
                blocking=False,
                threshold=0.0,
                higher_is_better=False,
            ),
            MetricSpec(
                metric="cost_usd",
                blocking=False,
                threshold=0.0,
                higher_is_better=False,
            ),
        ]
    )
