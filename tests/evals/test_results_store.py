"""Tests for the local JSONL results store (REG-8)."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_prompts.evals.runner.results_store import (
    EvalRunSummary,
    JsonlResultsStore,
    MetricResult,
    ResultsStore,
    new_run_id,
    now_utc_iso,
    summarise_run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    run_id: str,
    case_index: int = 0,
    metric: str = "exact_match",
    value: float = 1.0,
    passed: bool = True,
) -> MetricResult:
    return MetricResult(
        run_id=run_id,
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="regdoc-qa",
        dataset_version="1.0.0",
        triggered_by="test",
        created_at="2026-06-07T00:00:00+00:00",
        case_index=case_index,
        input_snippet="Under REACH Annex XVII...",
        metric=metric,
        value=value,
        baseline_value=None,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestResultsStoreProtocol:
    def test_jsonl_store_satisfies_protocol(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        assert isinstance(store, ResultsStore)


# ---------------------------------------------------------------------------
# JsonlResultsStore — save and reload
# ---------------------------------------------------------------------------


class TestJsonlResultsStore:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        store.save_run([_make_result(run_id)])
        assert (tmp_path / f"{run_id}.jsonl").is_file()

    def test_load_returns_same_rows(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        row = _make_result(run_id, metric="citation_validity", value=0.75, passed=False)
        store.save_run([row])

        loaded = store.load_run(run_id)
        assert len(loaded) == 1
        assert loaded[0].metric == "citation_validity"
        assert loaded[0].value == pytest.approx(0.75)
        assert loaded[0].passed is False

    def test_multiple_rows_round_trip(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        rows = [
            _make_result(run_id, case_index=0, metric="exact_match", value=1.0, passed=True),
            _make_result(run_id, case_index=0, metric="semantic_match", value=0.92, passed=True),
            _make_result(run_id, case_index=1, metric="exact_match", value=0.0, passed=False),
        ]
        store.save_run(rows)
        loaded = store.load_run(run_id)
        assert len(loaded) == 3
        metrics = [r.metric for r in loaded]
        assert "exact_match" in metrics
        assert "semantic_match" in metrics

    def test_load_missing_run_raises(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        with pytest.raises(FileNotFoundError, match="run_id"):
            store.load_run("nonexistent-uuid")

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            store.save_run([])

    def test_mixed_run_ids_raises(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        row1 = _make_result("id-A")
        row2 = _make_result("id-B")
        with pytest.raises(ValueError, match="run_id"):
            store.save_run([row1, row2])

    def test_root_created_on_save(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        store = JsonlResultsStore(root=nested)
        run_id = new_run_id()
        store.save_run([_make_result(run_id)])
        assert nested.is_dir()

    def test_baseline_value_null_round_trips(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        row = _make_result(run_id)
        assert row.baseline_value is None
        store.save_run([row])
        loaded = store.load_run(run_id)
        assert loaded[0].baseline_value is None

    def test_baseline_value_float_round_trips(self, tmp_path: Path) -> None:
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        row = MetricResult(
            run_id=run_id,
            prompt_version="p/1.0.0",
            dataset_name="ds",
            dataset_version="1.0.0",
            triggered_by="ci",
            created_at="2026-06-07T00:00:00+00:00",
            case_index=0,
            input_snippet="Q?",
            metric="latency_ms",
            value=150.0,
            baseline_value=120.0,
            passed=True,
        )
        store.save_run([row])
        loaded = store.load_run(run_id)
        assert loaded[0].baseline_value == pytest.approx(120.0)

    def test_file_is_valid_jsonl(self, tmp_path: Path) -> None:
        """Each line in the saved file must be valid JSON."""
        store = JsonlResultsStore(root=tmp_path)
        run_id = new_run_id()
        store.save_run([_make_result(run_id, case_index=0), _make_result(run_id, case_index=1)])
        path = tmp_path / f"{run_id}.jsonl"
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        for ln in lines:
            obj = json.loads(ln)  # must not raise
            assert "metric" in obj
            assert "passed" in obj


# ---------------------------------------------------------------------------
# new_run_id / now_utc_iso
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_new_run_id_returns_uuid_string(self) -> None:
        rid = new_run_id()
        assert isinstance(rid, str)
        # UUID4 has 36 chars with hyphens
        assert len(rid) == 36
        assert rid.count("-") == 4

    def test_new_run_ids_are_unique(self) -> None:
        ids = {new_run_id() for _ in range(20)}
        assert len(ids) == 20

    def test_now_utc_iso_is_string(self) -> None:
        ts = now_utc_iso()
        assert isinstance(ts, str)
        assert "T" in ts
        assert ts.endswith("+00:00")


# ---------------------------------------------------------------------------
# summarise_run
# ---------------------------------------------------------------------------


class TestSummariseRun:
    def test_all_passed(self) -> None:
        run_id = new_run_id()
        rows = [_make_result(run_id, case_index=i, passed=True) for i in range(3)]
        summary = summarise_run(rows)
        assert isinstance(summary, EvalRunSummary)
        assert summary.all_passed is True
        assert summary.pass_rate == pytest.approx(1.0)

    def test_partial_pass(self) -> None:
        run_id = new_run_id()
        rows = [
            _make_result(run_id, case_index=0, passed=True),
            _make_result(run_id, case_index=1, passed=False),
        ]
        summary = summarise_run(rows)
        assert summary.all_passed is False
        assert summary.pass_rate == pytest.approx(0.5)
        assert summary.metrics_passed == 1
        assert summary.metrics_total == 2

    def test_total_cases_counts_distinct_indices(self) -> None:
        run_id = new_run_id()
        # 2 metrics per case (2 cases) = 4 rows total but 2 distinct cases
        rows = [
            _make_result(run_id, case_index=0, metric="exact_match"),
            _make_result(run_id, case_index=0, metric="semantic_match"),
            _make_result(run_id, case_index=1, metric="exact_match"),
            _make_result(run_id, case_index=1, metric="semantic_match"),
        ]
        summary = summarise_run(rows)
        assert summary.total_cases == 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            summarise_run([])
