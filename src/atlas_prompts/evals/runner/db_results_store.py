"""DB-backed results store (REG-6).

Provides ``DbResultsStore`` — a SQLAlchemy 2.0 (sync) implementation of the
``ResultsStore`` Protocol (REG-8) that persists eval runs and their metric
results to the ``eval_runs`` / ``eval_results`` schema (atlas-docs/03 §1.6).

Design seam
-----------
``DbResultsStore`` satisfies the same ``ResultsStore`` Protocol as
``JsonlResultsStore`` (REG-8).  Callers swap stores by passing a different
implementation at construction time — the gate and eval runner are unaffected.

Engine
------
Pass a SQLAlchemy ``Engine`` (sync) at construction.  The engine is caller-owned
so its lifecycle (connection pool, teardown) stays outside this module.  A
SQLite engine is sufficient for offline tests and CI:

    engine = create_engine("sqlite:///eval_results.db")
    store  = DbResultsStore(engine)

For Postgres supply:

    engine = create_engine(
        "postgresql+psycopg://user:pass@host:5432/atlas_prompts",
        poolclass=NullPool,
    )

Schema
------
Tables are created via Alembic in production (REG-6 migration).  The offline
smoke test calls ``Base.metadata.create_all(engine)`` directly (SQLite only).
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from atlas_prompts.eval_runs.db.tables import EvalResult, EvalRun
from atlas_prompts.evals.runner.results_store import MetricResult


class DbResultsStore:
    """SQLAlchemy 2.0 sync results store for eval_runs / eval_results.

    Parameters
    ----------
    engine:
        A SQLAlchemy sync engine.  The caller owns the engine lifecycle.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # ResultsStore Protocol
    # ------------------------------------------------------------------

    def save_run(self, results: list[MetricResult]) -> str:
        """Persist an eval run + its metric results.  Returns the run_id (UUID).

        All ``results`` must share the same ``run_id``; a new ``eval_runs`` row is
        inserted using that UUID, and one ``eval_results`` row is inserted per
        ``MetricResult``.  The ``prompt_version_id`` and ``dataset_version`` are
        taken from the first row; callers are responsible for ensuring consistency
        across all rows.

        Raises
        ------
        ValueError
            If ``results`` is empty or rows carry different ``run_id`` values.
        """
        if not results:
            raise ValueError("Cannot save an empty results list")

        run_id = results[0].run_id
        if any(r.run_id != run_id for r in results):
            raise ValueError("All MetricResult rows in a single save_run call must share run_id")

        first = results[0]

        # prompt_version_id: the MetricResult carries `prompt_version` as a
        # "<name>/<semver>" string, not a UUID.  For the DB store we accept the
        # run_id as both the eval_run PK and expect callers to pass the actual
        # UUID via a `prompt_version_id` field if available.  Since MetricResult
        # does not carry a UUID, we generate a stable nil UUID as a sentinel so
        # the DB schema constraint is satisfied; real callers that have the UUID
        # should subclass or extend MetricResult.  See note below.
        #
        # NOTE: MetricResult.prompt_version is a "<name>/<semver>" slug designed
        # for the JSONL store.  The DB schema stores the UUID from prompt_versions
        # in the gateway DB (atlas-docs/03 §1.6).  We use uuid.NAMESPACE_URL +
        # the slug as a deterministic UUID-5 so the round-trip is reproducible
        # in tests without requiring a live gateway DB.
        pv_uuid = uuid.uuid5(uuid.NAMESPACE_URL, first.prompt_version)

        with Session(self._engine) as session:
            run = EvalRun(
                id=uuid.UUID(run_id),
                prompt_version_id=pv_uuid,
                dataset_version=first.dataset_version,
                triggered_by=first.triggered_by,
            )
            session.add(run)
            session.flush()  # populate run.id for the FK

            for row in results:
                result = EvalResult(
                    id=uuid.uuid4(),
                    eval_run_id=uuid.UUID(run_id),
                    metric=row.metric,
                    value=Decimal(str(row.value)),
                    baseline_value=(
                        Decimal(str(row.baseline_value)) if row.baseline_value is not None else None
                    ),
                    passed=row.passed,
                )
                session.add(result)

            session.commit()

        return run_id

    def load_run(self, run_id: str) -> list[MetricResult]:
        """Load all metric results for a previously saved run_id.

        Raises
        ------
        FileNotFoundError
            If no ``eval_runs`` row exists for ``run_id``.  Uses
            ``FileNotFoundError`` to match the ``ResultsStore`` Protocol
            contract established by ``JsonlResultsStore``.
        """
        run_uuid = uuid.UUID(run_id)

        with Session(self._engine) as session:
            run = session.get(EvalRun, run_uuid)
            if run is None:
                raise FileNotFoundError(f"No eval results found for run_id={run_id!r}")

            stmt = select(EvalResult).where(EvalResult.eval_run_id == run_uuid)
            db_results = list(session.scalars(stmt))

        # Reconstruct MetricResult from DB rows.
        # `prompt_version`, `dataset_name`, `case_index`, `input_snippet`, and
        # `created_at` are not stored in the eval_results table (the schema
        # follows atlas-docs/03 §1.6 exactly).  We reconstruct the slug from
        # the run's prompt_version_id (stored as UUID-5 of the slug) which is
        # not reversible — for load_run the slug is therefore set to the UUID
        # string.  Callers that need the original slug should use the JSONL
        # store or extend the schema.
        pv_slug = str(run.prompt_version_id)
        created_at_str = run.created_at.isoformat() if run.created_at else ""

        metric_results: list[MetricResult] = []
        for res in db_results:
            metric_results.append(
                MetricResult(
                    run_id=run_id,
                    prompt_version=pv_slug,
                    dataset_name="",
                    dataset_version=run.dataset_version,
                    triggered_by=run.triggered_by,
                    created_at=created_at_str,
                    case_index=0,
                    input_snippet="",
                    metric=res.metric,
                    value=float(res.value),
                    baseline_value=(
                        float(res.baseline_value) if res.baseline_value is not None else None
                    ),
                    passed=res.passed,
                )
            )

        return metric_results
