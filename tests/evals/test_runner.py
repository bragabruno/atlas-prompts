"""Tests for the EvalRunner (REG-8).

All tests are offline — they use FakeGatewayClient and JsonlResultsStore
over tmp_path.  No live gateway or network required.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_prompts.dataset_loader import LocalFileSource
from evals.runner.gateway_client import FakeGatewayClient, default_embedding_fn
from evals.runner.results_store import JsonlResultsStore, MetricResult
from evals.runner.runner import EvalRunner, RunConfig

# ---------------------------------------------------------------------------
# Dataset root for the committed seed
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
SEED_NAME = "regdoc-qa"
SEED_VERSION = "1.0.0"


def _make_runner(
    tmp_path: Path,
    completion_map: dict[str, str] | None = None,
    default_completion: str = "Answer EUR-REACH-Annex-XVII-Entry-27 cited.",
) -> tuple[EvalRunner, JsonlResultsStore]:
    client = FakeGatewayClient(
        completion_map=completion_map or {},
        default_completion=default_completion,
        embedding_fn=default_embedding_fn,
    )
    store = JsonlResultsStore(root=tmp_path)
    runner = EvalRunner(client=client, store=store)
    return runner, store


def _base_config(**kwargs: object) -> RunConfig:
    defaults: dict[str, object] = {
        "prompt_version": "regdoc-qa/1.0.0",
        "dataset_name": SEED_NAME,
        "dataset_version": SEED_VERSION,
        "completion_model": "test-model",
        "embedding_model": "test-emb",
        "triggered_by": "test",
    }
    defaults.update(kwargs)
    return RunConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy-path: full run over the seed dataset
# ---------------------------------------------------------------------------


class TestRunnerSeedDataset:
    def test_run_returns_summary(self, tmp_path: Path) -> None:
        runner, _ = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        assert summary.total_cases == 4  # seed has 4 records
        assert summary.metrics_total > 0
        assert 0.0 <= summary.pass_rate <= 1.0

    def test_results_written_to_store(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        assert len(loaded) > 0
        assert all(r.run_id == summary.run_id for r in loaded)

    def test_results_reloadable(self, tmp_path: Path) -> None:
        """Results written to JSONL can be read back and the summary matches."""
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())

        reloaded = store.load_run(summary.run_id)
        assert len(reloaded) == summary.metrics_total

    def test_all_four_seed_cases_processed(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        case_indices = {r.case_index for r in loaded}
        assert case_indices == {0, 1, 2, 3}

    def test_prompt_version_stored_in_results(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        assert all(r.prompt_version == "regdoc-qa/1.0.0" for r in loaded)

    def test_triggered_by_stored(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        config = _base_config(triggered_by="ci")
        summary = runner.run(config)
        loaded = store.load_run(summary.run_id)
        assert all(r.triggered_by == "ci" for r in loaded)


# ---------------------------------------------------------------------------
# Exact-match metric
# ---------------------------------------------------------------------------


class TestRunnerExactMatch:
    def test_exact_match_passes_when_correct(self, tmp_path: Path) -> None:
        # Seed record[0] expects exact: "0.05% by weight (500 mg/kg)"
        client = FakeGatewayClient(
            completion_map={
                "Under REACH Annex XVII entry 27, what is the maximum permitted "
                "concentration of lead in articles supplied to the general public?": (
                    "0.05% by weight (500 mg/kg)"
                )
            },
            default_completion="some other answer",
        )
        store = JsonlResultsStore(root=tmp_path)
        runner = EvalRunner(client=client, store=store)
        summary = runner.run(_base_config())

        loaded = store.load_run(summary.run_id)
        exact_rows = [r for r in loaded if r.metric == "exact_match" and r.case_index == 0]
        assert len(exact_rows) == 1
        assert exact_rows[0].passed is True

    def test_exact_match_fails_when_wrong(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path, default_completion="completely wrong answer")
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        # case_index=0 and case_index=2 have expected_exact
        exact_rows = [r for r in loaded if r.metric == "exact_match"]
        assert all(not r.passed for r in exact_rows)

    def test_no_exact_match_row_when_no_expected_exact(self, tmp_path: Path) -> None:
        """Cases without expected_exact must not produce an exact_match row."""
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        # case_index=1 has no expected_exact (only expected_semantic + citations)
        exact_case1 = [r for r in loaded if r.metric == "exact_match" and r.case_index == 1]
        assert exact_case1 == []


# ---------------------------------------------------------------------------
# Semantic-match metric
# ---------------------------------------------------------------------------


class TestRunnerSemanticMatch:
    def test_semantic_match_present_for_cases_with_expected_semantic(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        # All 4 seed cases have expected_semantic
        sem_indices = {r.case_index for r in loaded if r.metric == "semantic_match"}
        assert sem_indices == {0, 1, 2, 3}

    def test_semantic_score_in_range(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        for row in loaded:
            if row.metric == "semantic_match":
                assert -1.0 <= row.value <= 1.0

    def test_identical_response_gives_high_score(self, tmp_path: Path) -> None:
        """When the response is the expected_semantic text, cosine must be 1."""
        # Seed record[1] expected_semantic:
        expected_text = (
            "CLP requires suppliers of hazardous mixtures to affix a label bearing "
            "hazard pictograms, signal words, hazard statements, precautionary "
            "statements, and the supplier's identity before placing the product on "
            "the market."
        )
        client = FakeGatewayClient(
            default_completion=expected_text,
            embedding_fn=default_embedding_fn,
        )
        store = JsonlResultsStore(root=tmp_path)
        runner = EvalRunner(client=client, store=store)
        config = _base_config(semantic_threshold=0.0)  # threshold=0 so it always passes
        summary = runner.run(config)
        loaded = store.load_run(summary.run_id)

        sem_case1 = [r for r in loaded if r.metric == "semantic_match" and r.case_index == 1]
        assert len(sem_case1) == 1
        # Identical text → embedding is the same vector → cosine = 1.0
        assert sem_case1[0].value == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Citation-validity metric
# ---------------------------------------------------------------------------


class TestRunnerCitationValidity:
    def test_citation_rows_present_for_cases_with_required_citations(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        cit_indices = {r.case_index for r in loaded if r.metric == "citation_validity"}
        # All 4 seed cases have required_citations
        assert cit_indices == {0, 1, 2, 3}

    def test_citation_passes_when_source_id_in_response(self, tmp_path: Path) -> None:
        # Seed record[0] requires EUR-REACH-Annex-XVII-Entry-27
        client = FakeGatewayClient(
            default_completion=(
                "The limit is 0.05%. See EUR-REACH-Annex-XVII-Entry-27 for the full text."
            )
        )
        store = JsonlResultsStore(root=tmp_path)
        runner = EvalRunner(client=client, store=store)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        cit_case0 = [r for r in loaded if r.metric == "citation_validity" and r.case_index == 0]
        assert cit_case0[0].passed is True
        assert cit_case0[0].value == pytest.approx(1.0)

    def test_citation_fails_when_source_id_absent(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path, default_completion="No citations here.")
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        cit_rows = [r for r in loaded if r.metric == "citation_validity"]
        # All should fail since no source IDs appear in "No citations here."
        assert all(not r.passed for r in cit_rows)


# ---------------------------------------------------------------------------
# Latency and cost delta metrics
# ---------------------------------------------------------------------------


class TestRunnerDeltaMetrics:
    def test_latency_row_always_present(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        lat_indices = {r.case_index for r in loaded if r.metric == "latency_ms"}
        assert lat_indices == {0, 1, 2, 3}

    def test_cost_row_always_present(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config())
        loaded = store.load_run(summary.run_id)
        cost_indices = {r.case_index for r in loaded if r.metric == "cost_usd"}
        assert cost_indices == {0, 1, 2, 3}

    def test_latency_passes_without_baseline(self, tmp_path: Path) -> None:
        runner, store = _make_runner(tmp_path)
        summary = runner.run(_base_config(baseline_latency_ms=None))
        loaded = store.load_run(summary.run_id)
        for row in loaded:
            if row.metric == "latency_ms":
                assert row.passed is True
                assert row.baseline_value is None

    def test_latency_regression_detected(self, tmp_path: Path) -> None:
        # FakeClient returns latency_ms=50, baseline=10 → 400% regression → fail
        client = FakeGatewayClient(completion_latency_ms=50.0)
        store = JsonlResultsStore(root=tmp_path)
        runner = EvalRunner(client=client, store=store)
        config = _base_config(
            baseline_latency_ms=10.0,
            latency_regression_pct=20.0,
        )
        summary = runner.run(config)
        loaded = store.load_run(summary.run_id)
        lat_rows = [r for r in loaded if r.metric == "latency_ms"]
        assert all(not r.passed for r in lat_rows)


# ---------------------------------------------------------------------------
# Custom dataset (tmp_path — deterministic input/output)
# ---------------------------------------------------------------------------


class TestRunnerCustomDataset:
    def _write_dataset(self, root: Path, records: list[dict[str, object]]) -> None:
        path = root / "myds" / "1.0.0.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def test_single_record_exact_match(self, tmp_path: Path) -> None:
        ds_root = tmp_path / "datasets"
        self._write_dataset(ds_root, [{"input": "Q?", "expected_exact": "A"}])

        client = FakeGatewayClient(completion_map={"Q?": "A"})
        store = JsonlResultsStore(root=tmp_path / "results")
        runner = EvalRunner(client=client, store=store)

        config = RunConfig(
            prompt_version="test/1.0.0",
            dataset_name="myds",
            dataset_version="1.0.0",
            dataset_source=LocalFileSource(root=ds_root),
            completion_model="m",
            embedding_model="e",
        )
        summary = runner.run(config)
        assert summary.total_cases == 1

        loaded = store.load_run(summary.run_id)
        exact_rows = [r for r in loaded if r.metric == "exact_match"]
        assert len(exact_rows) == 1
        assert exact_rows[0].passed is True

    def test_run_id_is_unique_across_runs(self, tmp_path: Path) -> None:
        ds_root = tmp_path / "datasets"
        self._write_dataset(ds_root, [{"input": "Q?", "expected_exact": "A"}])

        client = FakeGatewayClient()
        store = JsonlResultsStore(root=tmp_path / "results")
        runner = EvalRunner(client=client, store=store)
        config = RunConfig(
            prompt_version="test/1.0.0",
            dataset_name="myds",
            dataset_version="1.0.0",
            dataset_source=LocalFileSource(root=ds_root),
            completion_model="m",
            embedding_model="e",
        )
        s1 = runner.run(config)
        s2 = runner.run(config)
        assert s1.run_id != s2.run_id

    def test_deterministic_metrics_given_same_inputs(self, tmp_path: Path) -> None:
        """Same client + same dataset → same metric values (no randomness)."""
        ds_root = tmp_path / "datasets"
        self._write_dataset(
            ds_root,
            [{"input": "Q?", "expected_exact": "A", "expected_semantic": "A answer."}],
        )

        def run_once(results_dir: Path) -> list[MetricResult]:
            client = FakeGatewayClient(completion_map={"Q?": "A"})
            store = JsonlResultsStore(root=results_dir)
            runner = EvalRunner(client=client, store=store)
            config = RunConfig(
                prompt_version="test/1.0.0",
                dataset_name="myds",
                dataset_version="1.0.0",
                dataset_source=LocalFileSource(root=ds_root),
                completion_model="m",
                embedding_model="e",
            )
            summary = runner.run(config)
            return store.load_run(summary.run_id)

        run1 = {r.metric: r.value for r in run_once(tmp_path / "r1")}
        run2 = {r.metric: r.value for r in run_once(tmp_path / "r2")}
        assert run1 == run2
