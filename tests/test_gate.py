"""Tests for the regression gate (REG-11).

Coverage requirements
---------------------
- Regressed candidate → non-zero exit / GateResult.passed == False
- Improved / equal candidate → exit 0 / GateResult.passed == True
- Thresholds are configurable (custom GateConfig respected)
- Blocking vs advisory split is honoured
- gate.py CLI exits with correct code
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_prompts.evals.gate.comparator import MetricDiff, run_gate
from atlas_prompts.evals.gate.gate_config import GateConfig, MetricSpec, default_gate_config
from atlas_prompts.evals.runner.results_store import JsonlResultsStore, MetricResult, new_run_id
from atlas_prompts.gate import main as gate_main

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TS = "2026-06-07T00:00:00+00:00"


def _row(
    run_id: str,
    metric: str,
    value: float,
    passed: bool = True,
    case_index: int = 0,
) -> MetricResult:
    return MetricResult(
        run_id=run_id,
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="regdoc-qa",
        dataset_version="1.0.0",
        triggered_by="test",
        created_at=_TS,
        case_index=case_index,
        input_snippet="Q?",
        metric=metric,
        value=value,
        baseline_value=None,
        passed=passed,
    )


def _rows(run_id: str, exact: float, semantic: float, citation: float) -> list[MetricResult]:
    """Three blocking-metric rows for one run."""
    return [
        _row(run_id, "exact_match", exact),
        _row(run_id, "semantic_match", semantic),
        _row(run_id, "citation_validity", citation),
    ]


def _rows_with_perf(
    run_id: str,
    exact: float,
    semantic: float,
    citation: float,
    latency: float,
    cost: float,
) -> list[MetricResult]:
    return [
        _row(run_id, "exact_match", exact),
        _row(run_id, "semantic_match", semantic),
        _row(run_id, "citation_validity", citation),
        _row(run_id, "latency_ms", latency),
        _row(run_id, "cost_usd", cost),
    ]


# ---------------------------------------------------------------------------
# compare_runs — core logic
# ---------------------------------------------------------------------------


class TestCompareRunsNoBaseline:
    """When baseline is None the gate must always pass (first-run mode)."""

    def test_no_baseline_always_passes(self) -> None:
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.5, semantic=0.5, citation=0.5),
            baseline_results=None,
        )
        assert result.passed is True

    def test_no_baseline_diffs_show_no_baseline(self) -> None:
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.8, semantic=0.8, citation=0.8),
            baseline_results=None,
        )
        for d in result.diffs:
            assert d.baseline_mean is None
            assert d.regression is False
            assert d.status == "NO_BASELINE"


class TestCompareRunsImproved:
    """Candidate better than baseline → gate passes."""

    def test_improved_candidate_passes(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
        )
        assert result.passed is True

    def test_equal_candidate_passes(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.8, semantic=0.8, citation=0.8),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
        )
        assert result.passed is True

    def test_improved_has_no_blocking_failures(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=1.0, semantic=1.0, citation=1.0),
            baseline_results=_rows(bid, exact=0.5, semantic=0.5, citation=0.5),
        )
        assert result.blocking_failures == []


class TestCompareRunsRegressed:
    """Candidate worse than baseline beyond threshold → gate fails."""

    def test_blocking_regression_fails_gate(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            # exact_match drops from 0.9 → 0.5 (delta −0.4, threshold 0.05)
            candidate_results=_rows(cid, exact=0.5, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
        )
        assert result.passed is False

    def test_blocking_failures_listed(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.3, semantic=0.3, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
        )
        failing_names = {d.metric for d in result.blocking_failures}
        assert "exact_match" in failing_names
        assert "semantic_match" in failing_names

    def test_regression_within_threshold_passes(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        # Drop of 0.03 is within the default 0.05 threshold
        result = run_gate(
            candidate_results=_rows(cid, exact=0.87, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.90, semantic=0.9, citation=0.9),
        )
        assert result.passed is True

    def test_regression_exactly_at_threshold_passes(self) -> None:
        """Delta equal to threshold is not a regression (strictly greater)."""
        bid = new_run_id()
        cid = new_run_id()
        # Drop of exactly 0.05 (the default threshold) must not fail
        result = run_gate(
            candidate_results=_rows(cid, exact=0.85, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.90, semantic=0.9, citation=0.9),
        )
        assert result.passed is True

    def test_regression_just_beyond_threshold_fails(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        # Drop of 0.051 is just beyond the 0.05 threshold
        result = run_gate(
            candidate_results=_rows(cid, exact=0.849, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.900, semantic=0.9, citation=0.9),
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Thresholds configurable
# ---------------------------------------------------------------------------


class TestThresholdsConfigurable:
    def test_strict_threshold_fails_small_regression(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        strict_config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="exact_match",
                    blocking=True,
                    threshold=0.0,  # zero tolerance
                    higher_is_better=True,
                )
            ]
        )
        result = run_gate(
            candidate_results=[_row(cid, "exact_match", 0.89)],
            baseline_results=[_row(bid, "exact_match", 0.90)],
            config=strict_config,
        )
        assert result.passed is False

    def test_generous_threshold_passes_large_regression(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        generous_config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="exact_match",
                    blocking=True,
                    threshold=0.5,  # allow up to 50 pp drop
                    higher_is_better=True,
                )
            ]
        )
        result = run_gate(
            candidate_results=[_row(cid, "exact_match", 0.6)],
            baseline_results=[_row(bid, "exact_match", 0.9)],
            config=generous_config,
        )
        assert result.passed is True

    def test_custom_config_only_evaluates_listed_metrics(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        single_metric_config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="exact_match", blocking=True, threshold=0.05, higher_is_better=True
                )
            ]
        )
        # semantic_match regresses badly but is not in the config
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.1, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
            config=single_metric_config,
        )
        assert result.passed is True
        assert len(result.diffs) == 1
        assert result.diffs[0].metric == "exact_match"


# ---------------------------------------------------------------------------
# Blocking vs advisory split
# ---------------------------------------------------------------------------


class TestBlockingVsAdvisory:
    def test_advisory_regression_does_not_block(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="latency_ms",
                    blocking=False,  # advisory
                    threshold=0.0,
                    higher_is_better=False,
                )
            ]
        )
        # latency doubles — a clear regression
        result = run_gate(
            candidate_results=[_row(cid, "latency_ms", 200.0)],
            baseline_results=[_row(bid, "latency_ms", 100.0)],
            config=config,
        )
        assert result.passed is True  # gate passes
        assert len(result.advisory_warnings) == 1
        assert result.advisory_warnings[0].metric == "latency_ms"

    def test_advisory_regression_appears_in_warnings(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="cost_usd",
                    blocking=False,
                    threshold=0.0,
                    higher_is_better=False,
                )
            ]
        )
        result = run_gate(
            candidate_results=[_row(cid, "cost_usd", 0.05)],
            baseline_results=[_row(bid, "cost_usd", 0.01)],
            config=config,
        )
        assert result.advisory_warnings[0].status == "WARN"

    def test_blocking_failure_and_advisory_warning_coexist(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="exact_match",
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
            ]
        )
        result = run_gate(
            candidate_results=[
                _row(cid, "exact_match", 0.5),  # regressed
                _row(cid, "latency_ms", 200.0),  # advisory regression
            ],
            baseline_results=[
                _row(bid, "exact_match", 0.9),
                _row(bid, "latency_ms", 100.0),
            ],
            config=config,
        )
        assert result.passed is False
        assert len(result.blocking_failures) == 1
        assert len(result.advisory_warnings) == 1

    def test_advisory_only_run_always_passes(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        advisory_only_config = GateConfig(
            metrics=[
                MetricSpec(
                    metric="exact_match",
                    blocking=False,  # advisory, not blocking
                    threshold=0.0,
                    higher_is_better=True,
                )
            ]
        )
        result = run_gate(
            candidate_results=[_row(cid, "exact_match", 0.0)],  # terrible score
            baseline_results=[_row(bid, "exact_match", 1.0)],
            config=advisory_only_config,
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# MetricDiff / GateResult properties
# ---------------------------------------------------------------------------


class TestMetricDiffStatus:
    def test_pass_status(self) -> None:
        d = MetricDiff(
            metric="exact_match",
            candidate_mean=0.9,
            baseline_mean=0.8,
            delta=0.1,
            regression=False,
            blocking=True,
            threshold=0.05,
            higher_is_better=True,
        )
        assert d.status == "PASS"

    def test_fail_status(self) -> None:
        d = MetricDiff(
            metric="exact_match",
            candidate_mean=0.5,
            baseline_mean=0.9,
            delta=-0.4,
            regression=True,
            blocking=True,
            threshold=0.05,
            higher_is_better=True,
        )
        assert d.status == "FAIL"

    def test_warn_status(self) -> None:
        d = MetricDiff(
            metric="latency_ms",
            candidate_mean=200.0,
            baseline_mean=100.0,
            delta=100.0,
            regression=True,
            blocking=False,
            threshold=0.0,
            higher_is_better=False,
        )
        assert d.status == "WARN"

    def test_no_baseline_status(self) -> None:
        d = MetricDiff(
            metric="exact_match",
            candidate_mean=0.9,
            baseline_mean=None,
            delta=0.0,
            regression=False,
            blocking=True,
            threshold=0.05,
            higher_is_better=True,
        )
        assert d.status == "NO_BASELINE"


class TestGateResultFormatTable:
    def test_format_table_contains_metric_names(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.8, semantic=0.8, citation=0.8),
            candidate_run_id=cid,
            baseline_run_id=bid,
        )
        table = result.format_table()
        assert "exact_match" in table
        assert "semantic_match" in table
        assert "citation_validity" in table

    def test_format_table_shows_gate_pass(self) -> None:
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.9, semantic=0.9, citation=0.9),
            baseline_results=None,
        )
        assert "PASS" in result.format_table()

    def test_format_table_shows_gate_fail(self) -> None:
        bid = new_run_id()
        cid = new_run_id()
        result = run_gate(
            candidate_results=_rows(cid, exact=0.3, semantic=0.9, citation=0.9),
            baseline_results=_rows(bid, exact=0.9, semantic=0.9, citation=0.9),
        )
        assert "FAIL" in result.format_table()


# ---------------------------------------------------------------------------
# CLI (gate.py main()) — exit codes
# ---------------------------------------------------------------------------


class TestGateMainCli:
    """Integration tests for the gate.py CLI via gate_main()."""

    def _write_run(self, tmp_path: Path, run_id: str, rows: list[MetricResult]) -> None:
        store = JsonlResultsStore(root=tmp_path)
        for r in rows:
            assert r.run_id == run_id
        store.save_run(rows)

    def test_cli_exits_0_on_pass(self, tmp_path: Path) -> None:
        bid = new_run_id()
        cid = new_run_id()
        self._write_run(tmp_path, bid, _rows(bid, exact=0.8, semantic=0.8, citation=0.8))
        self._write_run(tmp_path, cid, _rows(cid, exact=0.9, semantic=0.9, citation=0.9))

        rc = gate_main([cid, bid, "--store-root", str(tmp_path)])
        assert rc == 0

    def test_cli_exits_1_on_blocking_regression(self, tmp_path: Path) -> None:
        bid = new_run_id()
        cid = new_run_id()
        self._write_run(tmp_path, bid, _rows(bid, exact=0.9, semantic=0.9, citation=0.9))
        self._write_run(tmp_path, cid, _rows(cid, exact=0.3, semantic=0.9, citation=0.9))

        rc = gate_main([cid, bid, "--store-root", str(tmp_path)])
        assert rc == 1

    def test_cli_exits_0_no_baseline_mode(self, tmp_path: Path) -> None:
        cid = new_run_id()
        self._write_run(tmp_path, cid, _rows(cid, exact=0.1, semantic=0.1, citation=0.1))

        rc = gate_main([cid, "--no-baseline", "--store-root", str(tmp_path)])
        assert rc == 0

    def test_cli_exits_2_missing_baseline_arg(self, tmp_path: Path) -> None:
        cid = new_run_id()
        self._write_run(tmp_path, cid, _rows(cid, exact=0.9, semantic=0.9, citation=0.9))

        rc = gate_main([cid, "--store-root", str(tmp_path)])
        assert rc == 2

    def test_cli_exits_2_unknown_candidate(self, tmp_path: Path) -> None:
        rc = gate_main(["nonexistent-uuid", "--no-baseline", "--store-root", str(tmp_path)])
        assert rc == 2

    def test_cli_exits_2_unknown_baseline(self, tmp_path: Path) -> None:
        cid = new_run_id()
        self._write_run(tmp_path, cid, _rows(cid, exact=0.9, semantic=0.9, citation=0.9))

        rc = gate_main([cid, "nonexistent-baseline", "--store-root", str(tmp_path)])
        assert rc == 2

    def test_cli_advisory_regression_does_not_block(self, tmp_path: Path) -> None:
        """Latency regression alone (advisory) must not cause exit 1."""
        bid = new_run_id()
        cid = new_run_id()
        self._write_run(
            tmp_path,
            bid,
            _rows_with_perf(bid, exact=0.9, semantic=0.9, citation=0.9, latency=100.0, cost=0.01),
        )
        self._write_run(
            tmp_path,
            cid,
            _rows_with_perf(cid, exact=0.9, semantic=0.9, citation=0.9, latency=500.0, cost=0.10),
        )
        rc = gate_main([cid, bid, "--store-root", str(tmp_path)])
        assert rc == 0

    def test_cli_all_metrics_present_in_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bid = new_run_id()
        cid = new_run_id()
        self._write_run(tmp_path, bid, _rows(bid, exact=0.8, semantic=0.8, citation=0.8))
        self._write_run(tmp_path, cid, _rows(cid, exact=0.9, semantic=0.9, citation=0.9))

        gate_main([cid, bid, "--store-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "exact_match" in out
        assert "semantic_match" in out
        assert "citation_validity" in out


# ---------------------------------------------------------------------------
# default_gate_config smoke test
# ---------------------------------------------------------------------------


class TestDefaultGateConfig:
    def test_default_config_has_six_metrics(self) -> None:
        # REG-9 added llm_judge as the 6th metric (advisory).
        cfg = default_gate_config()
        assert len(cfg.metrics) == 6

    def test_quality_metrics_are_blocking(self) -> None:
        cfg = default_gate_config()
        blocking = {s.metric for s in cfg.metrics if s.blocking}
        assert "exact_match" in blocking
        assert "citation_validity" in blocking
        assert "semantic_match" in blocking

    def test_perf_metrics_are_advisory(self) -> None:
        cfg = default_gate_config()
        advisory = {s.metric for s in cfg.metrics if not s.blocking}
        assert "latency_ms" in advisory
        assert "cost_usd" in advisory
