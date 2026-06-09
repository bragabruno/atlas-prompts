"""Tests for the eval-gate PR-comment poster (REG-12).

Coverage
--------
- ``format_metric_diff_comment`` renders correctly for:
    (a) a passing comparison,
    (b) a regressing comparison,
    (c) a no-baseline / first-run comparison.
  Header verdict, per-metric rows, and run ids are all asserted.
- The poster is a clean no-op when PR-context env is absent (no HTTP).
- The poster fails fast when PR context is half-set (real misconfig).
- ``post_comment`` with an injected fake poster posts on both pass and
  regression — proving the comment is emitted regardless of gate verdict.

No real network calls are made: the HTTP provider is never constructed in
no-PR-context tests, and the post path is exercised with an in-memory fake.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_prompts.evals.gate.comparator import run_gate
from atlas_prompts.evals.gate.pr_comment import (
    CommentPoster,
    format_metric_diff_comment,
    main,
    post_comment,
)
from atlas_prompts.evals.runner.results_store import MetricResult, new_run_id

_TS = "2026-06-09T00:00:00+00:00"

# Every Bitbucket env var the poster reads — cleared before each test so the
# host environment can never leak a real PR context into a unit test.
_BITBUCKET_ENV = (
    "BITBUCKET_WORKSPACE",
    "BITBUCKET_REPO_SLUG",
    "BITBUCKET_PR_ID",
    "BITBUCKET_TOKEN",
    "BITBUCKET_ACCESS_TOKEN",
)


@pytest.fixture(autouse=True)
def _clear_bitbucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _BITBUCKET_ENV:
        monkeypatch.delenv(key, raising=False)


def _row(run_id: str, metric: str, value: float) -> MetricResult:
    return MetricResult(
        run_id=run_id,
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="regdoc-qa",
        dataset_version="1.0.0",
        triggered_by="test",
        created_at=_TS,
        case_index=0,
        input_snippet="Q?",
        metric=metric,
        value=value,
        baseline_value=None,
        passed=True,
    )


def _rows(run_id: str, exact: float, semantic: float, citation: float) -> list[MetricResult]:
    return [
        _row(run_id, "exact_match", exact),
        _row(run_id, "semantic_match", semantic),
        _row(run_id, "citation_validity", citation),
    ]


class _FakePoster:
    """In-memory ``CommentPoster`` — records bodies instead of hitting the API."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def post(self, body: str) -> None:
        self.bodies.append(body)


# ---------------------------------------------------------------------------
# format_metric_diff_comment — the pure core
# ---------------------------------------------------------------------------


class TestFormatPassing:
    def test_passing_header_and_rows(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        body = format_metric_diff_comment(result)

        assert "PASS ✅" in body
        assert "REGRESSION ❌" not in body
        # per-metric rows present, all PASS
        assert "`exact_match`" in body
        assert "`semantic_match`" in body
        assert "`citation_validity`" in body
        assert "REGRESSION" not in body.replace("REGRESSION ❌", "")
        assert body.count("PASS") >= 3  # header + (at least) one verdict cell
        # signed delta rendered for a present baseline
        assert "+0.1000" in body
        # run ids
        assert cid in body
        assert bid in body

    def test_deterministic(self) -> None:
        cid = new_run_id()
        bid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        assert format_metric_diff_comment(result) == format_metric_diff_comment(result)


class TestFormatRegressing:
    def test_regression_header_and_verdict(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            # exact_match collapses 0.9 -> 0.3 (delta -0.6, threshold 0.05)
            candidate_results=_rows(cid, exact=0.3, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        body = format_metric_diff_comment(result)

        assert "REGRESSION ❌" in body
        assert "PASS ✅" not in body
        # the regressed metric's row shows FAIL and the negative delta
        assert "FAIL" in body
        assert "-0.6000" in body
        assert cid in body
        assert bid in body


class TestFormatNoBaseline:
    def test_first_run_renders_as_pass_with_dashes(self) -> None:
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.5, semantic=0.5, citation=0.5),
            baseline_results=None,
            candidate_run_id=cid,
            baseline_run_id=None,
        )
        body = format_metric_diff_comment(result)

        # No baseline → still passes, baseline cells + delta render as em dashes
        assert "PASS ✅" in body
        assert "NO_BASELINE" in body
        assert "—" in body
        assert "none (first run)" in body
        assert cid in body


# ---------------------------------------------------------------------------
# post_comment — env gating + provider seam
# ---------------------------------------------------------------------------


class TestPostCommentNoContext:
    def test_no_pr_context_is_noop(self) -> None:
        """No Bitbucket env at all → skip cleanly, no HTTP, return False."""
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=None,
            candidate_run_id=cid,
        )
        # poster=None forces resolution from the (empty) environment.
        assert post_comment(result, poster=None) is False


class TestPostCommentMisconfig:
    def test_partial_pr_context_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PR id present but workspace/repo missing → explicit error, no silent skip."""
        monkeypatch.setenv("BITBUCKET_PR_ID", "42")
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=None,
            candidate_run_id=cid,
        )
        with pytest.raises(RuntimeError, match="Partial Bitbucket PR context"):
            post_comment(result, poster=None)

    def test_full_context_missing_token_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BITBUCKET_WORKSPACE", "acme")
        monkeypatch.setenv("BITBUCKET_REPO_SLUG", "atlas-prompts")
        monkeypatch.setenv("BITBUCKET_PR_ID", "42")
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=None,
            candidate_run_id=cid,
        )
        with pytest.raises(RuntimeError, match="no token found"):
            post_comment(result, poster=None)


class TestPostCommentWithFakePoster:
    def test_posts_on_pass(self) -> None:
        cid = new_run_id()
        bid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        poster = _FakePoster()
        posted = post_comment(result, poster=poster)
        assert posted is True
        assert len(poster.bodies) == 1
        assert "PASS ✅" in poster.bodies[0]

    def test_posts_on_regression(self) -> None:
        cid = new_run_id()
        bid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.3, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        poster = _FakePoster()
        posted = post_comment(result, poster=poster)
        assert posted is True
        assert "REGRESSION ❌" in poster.bodies[0]

    def test_fake_poster_satisfies_protocol(self) -> None:
        poster: CommentPoster = _FakePoster()
        poster.post("hello")  # type-checks against the seam


# ---------------------------------------------------------------------------
# CLI (main) — reads the gate's --json output
# ---------------------------------------------------------------------------


class TestCli:
    def _write_comparison(self, tmp_path: Path) -> Path:
        cid = new_run_id()
        bid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        path = tmp_path / "diff.json"
        path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        return path

    def test_cli_noop_without_pr_context(self, tmp_path: Path) -> None:
        """Reads a real comparison file; with no PR context it skips and returns 0."""
        path = self._write_comparison(tmp_path)
        assert main([str(path)]) == 0

    def test_cli_usage_error_on_wrong_args(self) -> None:
        assert main([]) == 2

    def test_cli_missing_file(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nope.json")]) == 2

    def test_cli_misconfig_returns_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BITBUCKET_PR_ID", "42")  # half-set context
        path = self._write_comparison(tmp_path)
        assert main([str(path)]) == 2
