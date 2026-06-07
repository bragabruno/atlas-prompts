"""Core eval metrics (REG-8) — pure-Python implementations.

DeepEval integration status
----------------------------
ADR-017 adopts DeepEval for metric implementations.  REG-8 implements the
required metrics in **pure Python** because DeepEval 4.0.x (latest safe pin:
4.0.3, published 2026-05-21) pulls in transitive dependencies published *after*
the repo's 14-day floor (2026-05-24):

    posthog==7.18.0     → 2026-06-05
    sentry-sdk==2.61.1  → 2026-06-01
    grpcio==1.81.0      → 2026-06-01
    aiohttp==3.14.0     → 2026-06-01
    typer==0.26.7       → 2026-06-03
    tqdm==4.68.1        → 2026-06-05

Swapping in DeepEval is deferred to a follow-up ticket; the seam is the
``MetricsEngine`` interface — replace the method bodies with ``deepeval``
calls once the dep tree is safe.

Metrics implemented
-------------------
- ``exact_match``     — string equality (normalised whitespace + case).
- ``semantic_match``  — cosine similarity of gateway embeddings (numpy dot).
- ``citation_validity`` — fraction of required citations present in the
  agent response, weighted by min_count.
- ``latency_delta_ms`` / ``cost_delta_usd`` — absolute deltas vs. a baseline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas_prompts.datasets import CitationRequirement


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactMatchResult:
    passed: bool
    candidate: str
    expected: str


@dataclass(frozen=True, slots=True)
class SemanticMatchResult:
    score: float  # cosine similarity [0, 1]
    passed: bool
    threshold: float


@dataclass(frozen=True, slots=True)
class CitationValidityResult:
    score: float  # fraction of required citations satisfied [0, 1]
    passed: bool
    threshold: float
    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class JudgeScoreResult:
    """Result of an LLM-as-judge evaluation (REG-9).

    The score is the mean of ``n_samples`` independent judge calls, which
    bounds per-sample flakiness.  The metric is always *advisory* — the
    ``passed`` flag merely records whether the mean score reached
    ``pass_threshold``; the gate never blocks on it.
    """

    score: float
    """Mean judge score, normalised to [0, 1] from the rubric scale."""

    raw_scores: list[float]
    """Individual scores from each sample call (before normalisation)."""

    rubric_version: str
    """Rubric file version used for this evaluation."""

    judge_model: str
    """Model alias used as the judge."""

    n_samples: int
    """Number of independent judge calls made."""

    margin: float
    """Allowed variance: max(raw_scores) − min(raw_scores) must be ≤ margin.
    When the spread exceeds the margin the result is still valid but the
    caller may log a variance warning."""

    spread_within_margin: bool
    """True when max(raw_scores) − min(raw_scores) ≤ margin."""

    passed: bool
    """True when the normalised mean score ≥ pass_threshold."""

    pass_threshold: float
    """Minimum normalised score required to pass (e.g. 0.6)."""


@dataclass(frozen=True, slots=True)
class DeltaResult:
    """Absolute delta of a scalar metric vs. a baseline value."""

    delta: float  # candidate − baseline  (negative means improvement for latency/cost)
    passed: bool
    metric_name: str
    candidate_value: float
    baseline_value: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for exact-match comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity.  Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def exact_match(candidate: str, expected: str) -> ExactMatchResult:
    """Check whether ``candidate`` equals ``expected`` after normalisation.

    Normalisation: strip leading/trailing whitespace, collapse internal
    whitespace runs to a single space, lowercase.
    """
    passed = _normalise(candidate) == _normalise(expected)
    return ExactMatchResult(passed=passed, candidate=candidate, expected=expected)


def semantic_match(
    candidate_embedding: list[float],
    reference_embedding: list[float],
    threshold: float = 0.85,
) -> SemanticMatchResult:
    """Cosine similarity of two embedding vectors.

    Parameters
    ----------
    candidate_embedding:
        Embedding of the model's actual response text.
    reference_embedding:
        Embedding of the golden ``expected_semantic`` reference text.
    threshold:
        Minimum cosine similarity to consider the match passed.
        Default 0.85 is a pragmatic starting point; tune per dataset.

    Notes
    -----
    DeepEval follow-up: replace the body with::

        from deepeval.metrics import AnswerRelevancyMetric
        # or GEval with semantic criterion

    The ``threshold`` parameter maps directly to DeepEval's ``minimum_score``.
    """
    score = _cosine_similarity(candidate_embedding, reference_embedding)
    return SemanticMatchResult(
        score=round(score, 6),
        passed=score >= threshold,
        threshold=threshold,
    )


def citation_validity(
    response_text: str,
    required: list[CitationRequirement],
    threshold: float = 1.0,
) -> CitationValidityResult:
    """Check whether all required citation source_ids appear in ``response_text``.

    A citation is considered *satisfied* if its ``source_id`` string appears
    verbatim in ``response_text`` at least ``min_count`` times (case-sensitive,
    because source IDs are structured codes such as ``EUR-REACH-Annex-XVII``).

    Parameters
    ----------
    response_text:
        The full text of the agent's response (citations expected to be embedded
        or listed at the end, depending on agent convention).
    required:
        List of :class:`~atlas_prompts.datasets.CitationRequirement` objects
        from the golden record.
    threshold:
        Minimum fraction of required citations that must be satisfied for the
        metric to pass.  Default 1.0 = all citations required.

    Returns
    -------
    CitationValidityResult
        ``score`` is the weighted fraction satisfied:
        ``sum(satisfied_weight) / sum(total_weight)`` where each citation's
        weight equals its ``min_count``.

    Notes
    -----
    DeepEval follow-up: the ``BiasMetric`` / ``FaithfulnessMetric`` cover
    adjacent ground.  Citation checking maps to a custom ``GEval`` criterion
    or a thin wrapper around ``FaithfulnessMetric``.
    """
    if not required:
        return CitationValidityResult(
            score=1.0, passed=True, threshold=threshold, satisfied=[], missing=[]
        )

    total_weight = sum(c.min_count for c in required)
    satisfied_weight = 0
    satisfied_ids: list[str] = []
    missing_ids: list[str] = []

    for cit in required:
        count = response_text.count(cit.source_id)
        if count >= cit.min_count:
            satisfied_weight += cit.min_count
            satisfied_ids.append(cit.source_id)
        else:
            missing_ids.append(cit.source_id)

    score = satisfied_weight / total_weight if total_weight > 0 else 1.0
    return CitationValidityResult(
        score=round(score, 6),
        passed=score >= threshold,
        threshold=threshold,
        satisfied=satisfied_ids,
        missing=missing_ids,
    )


def latency_delta(
    candidate_ms: float,
    baseline_ms: float | None,
    max_regression_pct: float = 20.0,
) -> DeltaResult:
    """Compute latency delta vs. baseline.

    A regression is defined as candidate latency exceeding baseline by more
    than ``max_regression_pct`` percent.  When ``baseline_ms`` is None (first
    run), the result is always passed.

    Parameters
    ----------
    candidate_ms:
        Observed wall-clock latency for the candidate in milliseconds.
    baseline_ms:
        Baseline latency.  None means no baseline (first eval run).
    max_regression_pct:
        Allowable percentage increase before the metric fails.  Default 20%.
    """
    if baseline_ms is None:
        return DeltaResult(
            delta=0.0,
            passed=True,
            metric_name="latency_ms",
            candidate_value=candidate_ms,
            baseline_value=None,
        )
    delta = candidate_ms - baseline_ms
    if baseline_ms == 0.0:
        passed = delta <= 0.0
    else:
        passed = (delta / baseline_ms) * 100.0 <= max_regression_pct
    return DeltaResult(
        delta=round(delta, 3),
        passed=passed,
        metric_name="latency_ms",
        candidate_value=candidate_ms,
        baseline_value=baseline_ms,
    )


def cost_delta(
    candidate_usd: float,
    baseline_usd: float | None,
    max_regression_pct: float = 20.0,
) -> DeltaResult:
    """Compute cost delta vs. baseline.

    Analogous to :func:`latency_delta` but for estimated cost in USD.
    """
    if baseline_usd is None:
        return DeltaResult(
            delta=0.0,
            passed=True,
            metric_name="cost_usd",
            candidate_value=candidate_usd,
            baseline_value=None,
        )
    delta = candidate_usd - baseline_usd
    if baseline_usd == 0.0:
        passed = delta <= 0.0
    else:
        passed = (delta / baseline_usd) * 100.0 <= max_regression_pct
    return DeltaResult(
        delta=round(delta, 9),
        passed=passed,
        metric_name="cost_usd",
        candidate_value=candidate_usd,
        baseline_value=baseline_usd,
    )
