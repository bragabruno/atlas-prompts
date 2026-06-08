"""Tests for eval metrics (REG-8) — pure-Python implementations."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math

import pytest

from atlas_prompts.datasets import CitationRequirement
from atlas_prompts.evals.runner.metrics import (
    SemanticMatchResult,
    citation_validity,
    cost_delta,
    exact_match,
    latency_delta,
    semantic_match,
)

# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical_strings_pass(self) -> None:
        r = exact_match("0.05% by weight", "0.05% by weight")
        assert r.passed is True

    def test_case_insensitive(self) -> None:
        r = exact_match("No", "no")
        assert r.passed is True

    def test_whitespace_normalised(self) -> None:
        r = exact_match("  hello   world  ", "hello world")
        assert r.passed is True

    def test_different_strings_fail(self) -> None:
        r = exact_match("0.05%", "0.1%")
        assert r.passed is False

    def test_partial_match_fails(self) -> None:
        r = exact_match("yes, confirmed", "yes")
        assert r.passed is False

    def test_empty_strings_pass_each_other(self) -> None:
        r = exact_match("", "")
        assert r.passed is True

    def test_candidate_and_expected_stored(self) -> None:
        r = exact_match("A", "B")
        assert r.candidate == "A"
        assert r.expected == "B"


# ---------------------------------------------------------------------------
# semantic_match
# ---------------------------------------------------------------------------


class TestSemanticMatch:
    def _unit(self, *values: float) -> list[float]:
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def test_identical_vectors_score_one(self) -> None:
        v = self._unit(1.0, 2.0, 3.0)
        r = semantic_match(v, v, threshold=0.85)
        assert r.score == pytest.approx(1.0, abs=1e-6)
        assert r.passed is True

    def test_orthogonal_vectors_score_zero(self) -> None:
        r = semantic_match([1.0, 0.0], [0.0, 1.0], threshold=0.5)
        assert r.score == pytest.approx(0.0, abs=1e-6)
        assert r.passed is False

    def test_antiparallel_vectors_score_minus_one(self) -> None:
        r = semantic_match([1.0, 0.0], [-1.0, 0.0], threshold=0.0)
        assert r.score == pytest.approx(-1.0, abs=1e-6)
        assert r.passed is False

    def test_threshold_respected_pass(self) -> None:
        v1 = self._unit(1.0, 1.0)
        v2 = self._unit(1.0, 1.0)
        r = semantic_match(v1, v2, threshold=0.99)
        assert r.passed is True

    def test_threshold_respected_fail(self) -> None:
        v1 = self._unit(1.0, 0.0)
        v2 = self._unit(0.9, 0.436)  # angle ~25.8°, cos≈0.9
        r = semantic_match(v1, v2, threshold=0.95)
        assert r.passed is False

    def test_zero_vector_returns_zero(self) -> None:
        r = semantic_match([0.0, 0.0], [1.0, 0.0], threshold=0.0)
        assert r.score == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            semantic_match([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_result_is_frozen(self) -> None:
        r: SemanticMatchResult = semantic_match([1.0], [1.0], threshold=0.5)
        with pytest.raises((AttributeError, TypeError)):
            r.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# citation_validity
# ---------------------------------------------------------------------------


class TestCitationValidity:
    def _req(self, source_id: str, min_count: int = 1) -> CitationRequirement:
        return CitationRequirement(source_id=source_id, min_count=min_count)

    def test_all_citations_present_passes(self) -> None:
        text = "See EUR-REACH-Annex-XVII-Entry-27 for details."
        r = citation_validity(text, [self._req("EUR-REACH-Annex-XVII-Entry-27")])
        assert r.score == pytest.approx(1.0)
        assert r.passed is True
        assert "EUR-REACH-Annex-XVII-Entry-27" in r.satisfied

    def test_missing_citation_fails(self) -> None:
        text = "No relevant citations here."
        r = citation_validity(text, [self._req("EUR-CLP-1272-2008-Art17")])
        assert r.score == pytest.approx(0.0)
        assert r.passed is False
        assert "EUR-CLP-1272-2008-Art17" in r.missing

    def test_partial_citations_gives_partial_score(self) -> None:
        text = "EUR-A mentioned here."
        r = citation_validity(
            text,
            [self._req("EUR-A"), self._req("EUR-B")],
            threshold=1.0,
        )
        assert r.score == pytest.approx(0.5)
        assert r.passed is False

    def test_empty_required_always_passes(self) -> None:
        r = citation_validity("any text", [], threshold=1.0)
        assert r.score == pytest.approx(1.0)
        assert r.passed is True

    def test_min_count_below_threshold_fails(self) -> None:
        # source appears once but min_count=2 requires two occurrences
        text = "EUR-REACH mentioned once."
        r = citation_validity(text, [self._req("EUR-REACH", min_count=2)])
        assert r.passed is False
        assert "EUR-REACH" in r.missing

    def test_min_count_satisfied(self) -> None:
        text = "EUR-REACH appears twice: EUR-REACH."
        r = citation_validity(text, [self._req("EUR-REACH", min_count=2)])
        assert r.passed is True

    def test_threshold_below_full_allows_partial(self) -> None:
        text = "Only EUR-A here."
        r = citation_validity(
            text,
            [self._req("EUR-A"), self._req("EUR-B")],
            threshold=0.5,
        )
        assert r.score == pytest.approx(0.5)
        assert r.passed is True

    def test_weighted_by_min_count(self) -> None:
        # EUR-A has min_count=2, EUR-B has min_count=1
        # total weight = 3, EUR-A satisfied (weight 2), EUR-B missing
        text = "EUR-A is cited twice: EUR-A."
        r = citation_validity(
            text,
            [self._req("EUR-A", min_count=2), self._req("EUR-B", min_count=1)],
            threshold=1.0,
        )
        assert r.score == pytest.approx(2 / 3)
        assert r.passed is False


# ---------------------------------------------------------------------------
# latency_delta
# ---------------------------------------------------------------------------


class TestLatencyDelta:
    def test_no_baseline_always_passes(self) -> None:
        r = latency_delta(1000.0, None)
        assert r.passed is True
        assert r.baseline_value is None

    def test_improvement_passes(self) -> None:
        r = latency_delta(80.0, 100.0, max_regression_pct=20.0)
        assert r.passed is True
        assert r.delta == pytest.approx(-20.0, abs=1e-3)

    def test_within_threshold_passes(self) -> None:
        r = latency_delta(115.0, 100.0, max_regression_pct=20.0)
        assert r.passed is True

    def test_above_threshold_fails(self) -> None:
        r = latency_delta(125.0, 100.0, max_regression_pct=20.0)
        assert r.passed is False

    def test_exactly_at_threshold_passes(self) -> None:
        r = latency_delta(120.0, 100.0, max_regression_pct=20.0)
        assert r.passed is True

    def test_metric_name_is_latency_ms(self) -> None:
        r = latency_delta(100.0, 100.0)
        assert r.metric_name == "latency_ms"

    def test_zero_baseline_regression_check(self) -> None:
        # baseline=0 → only improvement (delta<=0) passes
        r = latency_delta(1.0, 0.0)
        assert r.passed is False
        r2 = latency_delta(0.0, 0.0)
        assert r2.passed is True


# ---------------------------------------------------------------------------
# cost_delta
# ---------------------------------------------------------------------------


class TestCostDelta:
    def test_no_baseline_always_passes(self) -> None:
        r = cost_delta(0.01, None)
        assert r.passed is True

    def test_improvement_passes(self) -> None:
        r = cost_delta(0.008, 0.010, max_regression_pct=20.0)
        assert r.passed is True

    def test_regression_fails(self) -> None:
        r = cost_delta(0.013, 0.010, max_regression_pct=20.0)
        assert r.passed is False

    def test_metric_name_is_cost_usd(self) -> None:
        r = cost_delta(0.01, 0.01)
        assert r.metric_name == "cost_usd"
