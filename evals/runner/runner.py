"""Eval runner (REG-8).

Loads a golden dataset (REG-7), executes a prompt version over every record by
calling the gateway, computes all configured metrics, and writes results to the
local results store.

Public entry point
------------------
    from evals.runner.runner import EvalRunner, RunConfig

    runner = EvalRunner(client=gateway_client, store=JsonlResultsStore())
    summary = runner.run(RunConfig(...))

The runner is fully offline when given a ``FakeGatewayClient`` — no live
gateway or network required.

Metrics computed per golden record
-----------------------------------
1. ``exact_match``       — if ``expected_exact`` is set.
2. ``semantic_match``    — if ``expected_semantic`` is set; requires an
                           embedding call per record (candidate + reference).
3. ``citation_validity`` — if ``required_citations`` is non-empty.
4. ``latency_ms``        — latency delta vs. ``baseline_latency_ms``.
5. ``cost_usd``          — cost delta vs. ``baseline_cost_usd``.

Baseline
--------
Pass ``baseline_latency_ms`` / ``baseline_cost_usd`` in ``RunConfig`` to
enable delta metrics.  Omit (leave ``None``) for first-run scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from atlas_prompts.dataset_loader import DatasetSource, load_dataset
from atlas_prompts.datasets import GoldenRecord
from evals.runner.gateway_client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    GatewayClient,
)
from evals.runner.metrics import (
    citation_validity,
    cost_delta,
    exact_match,
    latency_delta,
    semantic_match,
)
from evals.runner.results_store import (
    EvalRunSummary,
    MetricResult,
    ResultsStore,
    new_run_id,
    now_utc_iso,
    summarise_run,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Parameters that define one eval run.

    Parameters
    ----------
    prompt_version:
        Human-readable identifier ``"<name>/<semver>"``, e.g.
        ``"regdoc-qa/1.0.0"``.  Stored in result rows for traceability;
        the runner does not validate it against the prompts directory.
    dataset_name:
        Dataset name as recognised by :func:`~atlas_prompts.dataset_loader.load_dataset`.
    dataset_version:
        Dataset version string, e.g. ``"1.0.0"``.
    completion_model:
        Model alias sent in the ``/v1/chat/completions`` request body.
    embedding_model:
        Model alias sent in the ``/v1/embeddings`` request body.
    triggered_by:
        Free-text trigger label — ``"ci"``, ``"manual:<user>"``, etc.
    semantic_threshold:
        Minimum cosine similarity for the semantic-match metric (default 0.85).
    citation_threshold:
        Minimum citation-validity score to pass (default 1.0 = all required).
    latency_regression_pct:
        Maximum % increase in latency before the delta metric fails (default 20).
    cost_regression_pct:
        Maximum % increase in cost before the delta metric fails (default 20).
    baseline_latency_ms:
        Per-case average latency of the production baseline, or ``None``.
    baseline_cost_usd:
        Per-case average cost of the production baseline, or ``None``.
    dataset_source:
        Optional alternative :class:`~atlas_prompts.dataset_loader.DatasetSource`;
        defaults to local file.
    system_prompt:
        Optional system message prepended to every completion request.
        Defaults to a minimal instruction that includes the golden-record input.
    extra_messages:
        Optional additional ``ChatMessage`` objects inserted between the system
        message and the user message (e.g. few-shot examples).
    """

    prompt_version: str
    dataset_name: str
    dataset_version: str
    completion_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    triggered_by: str = "manual"
    semantic_threshold: float = 0.85
    citation_threshold: float = 1.0
    latency_regression_pct: float = 20.0
    cost_regression_pct: float = 20.0
    baseline_latency_ms: float | None = None
    baseline_cost_usd: float | None = None
    dataset_source: DatasetSource | None = None
    system_prompt: str = (
        "You are a regulatory-compliance assistant. "
        "Answer the user's question precisely and cite your sources."
    )
    extra_messages: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Executes an eval run against a golden dataset.

    Parameters
    ----------
    client:
        Any object satisfying :class:`~evals.runner.gateway_client.GatewayClient`.
    store:
        Any object satisfying :class:`~evals.runner.results_store.ResultsStore`.
    """

    def __init__(self, client: GatewayClient, store: ResultsStore) -> None:
        self._client = client
        self._store = store

    def run(self, config: RunConfig) -> EvalRunSummary:
        """Execute one complete eval run and return the aggregated summary.

        Parameters
        ----------
        config:
            Full run configuration.

        Returns
        -------
        EvalRunSummary
            Aggregated pass/fail counts plus the run_id for downstream use.
        """
        run_id = new_run_id()
        created_at = now_utc_iso()

        records: list[GoldenRecord] = load_dataset(
            config.dataset_name,
            config.dataset_version,
            source=config.dataset_source,
        )
        logger.info(
            "eval_run_start run_id=%s prompt=%s dataset=%s/%s n_cases=%d",
            run_id,
            config.prompt_version,
            config.dataset_name,
            config.dataset_version,
            len(records),
        )

        all_results: list[MetricResult] = []

        for case_idx, record in enumerate(records):
            case_results = self._run_case(
                run_id=run_id,
                created_at=created_at,
                config=config,
                case_idx=case_idx,
                record=record,
            )
            all_results.extend(case_results)

        self._store.save_run(all_results)
        summary = summarise_run(all_results)
        logger.info(
            "eval_run_complete run_id=%s pass_rate=%.3f all_passed=%s",
            run_id,
            summary.pass_rate,
            summary.all_passed,
        )
        return summary

    def _run_case(
        self,
        *,
        run_id: str,
        created_at: str,
        config: RunConfig,
        case_idx: int,
        record: GoldenRecord,
    ) -> list[MetricResult]:
        """Execute a single golden case and return its metric rows."""
        messages = self._build_messages(config, record.input)
        completion: CompletionResponse = self._client.chat_completion(
            CompletionRequest(
                model=config.completion_model,
                messages=messages,
                temperature=0.0,
            )
        )
        candidate_text = completion.content

        def _base(metric: str, value: float, baseline: float | None, passed: bool) -> MetricResult:
            return MetricResult(
                run_id=run_id,
                prompt_version=config.prompt_version,
                dataset_name=config.dataset_name,
                dataset_version=config.dataset_version,
                triggered_by=config.triggered_by,
                created_at=created_at,
                case_index=case_idx,
                input_snippet=record.input[:120],
                metric=metric,
                value=value,
                baseline_value=baseline,
                passed=passed,
            )

        results: list[MetricResult] = []

        # ---- 1. Exact match ------------------------------------------------
        if record.expected_exact is not None:
            em = exact_match(candidate_text, record.expected_exact)
            results.append(_base("exact_match", 1.0 if em.passed else 0.0, None, em.passed))

        # ---- 2. Semantic match ---------------------------------------------
        if record.expected_semantic is not None:
            candidate_emb = self._client.embed(
                EmbeddingRequest(model=config.embedding_model, input=candidate_text)
            )
            reference_emb = self._client.embed(
                EmbeddingRequest(model=config.embedding_model, input=record.expected_semantic)
            )
            sm = semantic_match(
                candidate_emb.embedding,
                reference_emb.embedding,
                threshold=config.semantic_threshold,
            )
            results.append(_base("semantic_match", sm.score, None, sm.passed))

        # ---- 3. Citation validity ------------------------------------------
        if record.required_citations:
            cv = citation_validity(
                candidate_text,
                record.required_citations,
                threshold=config.citation_threshold,
            )
            results.append(_base("citation_validity", cv.score, None, cv.passed))

        # ---- 4. Latency delta ----------------------------------------------
        ld = latency_delta(
            completion.latency_ms,
            config.baseline_latency_ms,
            max_regression_pct=config.latency_regression_pct,
        )
        results.append(
            _base(
                "latency_ms",
                completion.latency_ms,
                config.baseline_latency_ms,
                ld.passed,
            )
        )

        # ---- 5. Cost delta -------------------------------------------------
        cd = cost_delta(
            completion.cost_usd,
            config.baseline_cost_usd,
            max_regression_pct=config.cost_regression_pct,
        )
        results.append(
            _base(
                "cost_usd",
                completion.cost_usd,
                config.baseline_cost_usd,
                cd.passed,
            )
        )

        return results

    @staticmethod
    def _build_messages(config: RunConfig, user_input: str) -> list[ChatMessage]:
        """Construct the message list for a completion request."""
        messages: list[ChatMessage] = [ChatMessage(role="system", content=config.system_prompt)]
        for extra in config.extra_messages:
            messages.append(ChatMessage(role=extra["role"], content=extra["content"]))
        messages.append(ChatMessage(role="user", content=user_input))
        return messages
