"""Tests for the DB-backed results store (REG-6).

Covers:
- Schema smoke: metadata.create_all builds both tables on SQLite.
- DbResultsStore satisfies the ResultsStore Protocol.
- Round-trip: save_run + load_run returns correct metric rows.
- FK constraint: eval_results.eval_run_id references eval_runs.id.
- Indexes: idx_eval_runs_prompt_version and idx_eval_results_eval_run_id present.
- Error paths: empty results raises, mixed run_ids raises, missing run_id raises.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import uuid
from collections.abc import Generator
from decimal import Decimal

import pytest
from sqlalchemy import Engine, create_engine, event, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from eval_runs.db.base import Base
from eval_runs.db.tables import EvalResult, EvalRun
from evals.runner.db_results_store import DbResultsStore
from evals.runner.results_store import MetricResult, ResultsStore, new_run_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> Generator[Engine, None, None]:
    """In-memory SQLite engine with the eval schema applied.

    Foreign-key enforcement is enabled via the ``PRAGMA foreign_keys = ON``
    event hook — SQLite disables FK checks by default.
    """
    eng = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(eng, "connect")
    def _set_sqlite_pragma(dbapi_conn: object, _connection_record: object) -> None:
        import sqlite3

        assert isinstance(dbapi_conn, sqlite3.Connection)
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def store(engine: Engine) -> DbResultsStore:
    return DbResultsStore(engine)


def _make_result(
    run_id: str,
    case_index: int = 0,
    metric: str = "exact_match",
    value: float = 1.0,
    passed: bool = True,
    baseline_value: float | None = None,
) -> MetricResult:
    return MetricResult(
        run_id=run_id,
        prompt_version="regdoc-qa/1.0.0",
        dataset_name="regdoc-qa",
        dataset_version="v1.0",
        triggered_by="ci",
        created_at="2026-06-07T00:00:00+00:00",
        case_index=case_index,
        input_snippet="Under REACH Annex XVII...",
        metric=metric,
        value=value,
        baseline_value=baseline_value,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Schema smoke: metadata + create_all
# ---------------------------------------------------------------------------


def test_create_all_builds_both_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "eval_runs" in tables
    assert "eval_results" in tables


def test_eval_runs_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("eval_runs")}
    assert {"id", "prompt_version_id", "dataset_version", "triggered_by", "created_at"} <= cols


def test_eval_results_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("eval_results")}
    assert {"id", "eval_run_id", "metric", "value", "baseline_value", "passed"} <= cols


def test_eval_runs_index_present(engine: Engine) -> None:
    inspector = inspect(engine)
    index_names = {idx["name"] for idx in inspector.get_indexes("eval_runs")}
    assert "idx_eval_runs_prompt_version" in index_names


def test_eval_results_index_present(engine: Engine) -> None:
    inspector = inspect(engine)
    index_names = {idx["name"] for idx in inspector.get_indexes("eval_results")}
    assert "idx_eval_results_eval_run_id" in index_names


def test_eval_results_fk_to_eval_runs(engine: Engine) -> None:
    inspector = inspect(engine)
    fks = inspector.get_foreign_keys("eval_results")
    assert len(fks) == 1
    fk = fks[0]
    assert fk["referred_table"] == "eval_runs"
    assert "eval_run_id" in fk["constrained_columns"]


def test_base_metadata_tables() -> None:
    """Base.metadata must contain exactly the two eval tables."""
    assert "eval_runs" in Base.metadata.tables
    assert "eval_results" in Base.metadata.tables
    # Gateway tables must NOT bleed into this metadata.
    assert "model_aliases" not in Base.metadata.tables
    assert "api_keys" not in Base.metadata.tables


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_db_store_satisfies_results_store_protocol(store: DbResultsStore) -> None:
    assert isinstance(store, ResultsStore)


# ---------------------------------------------------------------------------
# DbResultsStore — save and reload
# ---------------------------------------------------------------------------


def test_save_run_returns_run_id(store: DbResultsStore) -> None:
    run_id = new_run_id()
    returned = store.save_run([_make_result(run_id)])
    assert returned == run_id


def test_save_creates_eval_run_row(store: DbResultsStore, engine: Engine) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id)])
    with Session(engine) as session:
        run = session.get(EvalRun, uuid.UUID(run_id))
    assert run is not None
    assert run.dataset_version == "v1.0"
    assert run.triggered_by == "ci"


def test_save_creates_eval_result_rows(store: DbResultsStore, engine: Engine) -> None:
    run_id = new_run_id()
    rows = [
        _make_result(run_id, metric="exact_match", value=1.0, passed=True),
        _make_result(run_id, metric="semantic_match", value=0.9, passed=True),
    ]
    store.save_run(rows)
    with Session(engine) as session:
        results = list(
            session.scalars(select(EvalResult).where(EvalResult.eval_run_id == uuid.UUID(run_id)))
        )
    assert len(results) == 2
    metrics = {r.metric for r in results}
    assert metrics == {"exact_match", "semantic_match"}


def test_load_run_returns_metric_results(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id, metric="cost_usd", value=0.05, passed=True)])
    loaded = store.load_run(run_id)
    assert len(loaded) == 1
    assert loaded[0].metric == "cost_usd"
    assert loaded[0].value == pytest.approx(0.05)
    assert loaded[0].passed is True


def test_load_run_returns_all_rows(store: DbResultsStore) -> None:
    run_id = new_run_id()
    rows = [_make_result(run_id, metric=f"metric_{i}", value=float(i)) for i in range(5)]
    store.save_run(rows)
    loaded = store.load_run(run_id)
    assert len(loaded) == 5


def test_baseline_value_null_round_trips(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id, baseline_value=None)])
    loaded = store.load_run(run_id)
    assert loaded[0].baseline_value is None


def test_baseline_value_float_round_trips(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id, value=0.75, baseline_value=0.80)])
    loaded = store.load_run(run_id)
    assert loaded[0].baseline_value == pytest.approx(0.80)


def test_load_run_preserves_run_id(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id)])
    loaded = store.load_run(run_id)
    assert all(r.run_id == run_id for r in loaded)


def test_load_run_preserves_dataset_version(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id)])
    loaded = store.load_run(run_id)
    assert loaded[0].dataset_version == "v1.0"


def test_load_run_preserves_triggered_by(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id)])
    loaded = store.load_run(run_id)
    assert loaded[0].triggered_by == "ci"


def test_load_missing_run_raises_file_not_found(store: DbResultsStore) -> None:
    with pytest.raises(FileNotFoundError, match="run_id"):
        store.load_run(new_run_id())


def test_empty_results_raises_value_error(store: DbResultsStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.save_run([])


def test_mixed_run_ids_raises_value_error(store: DbResultsStore) -> None:
    row1 = _make_result("id-" + new_run_id())
    row2 = _make_result("id-" + new_run_id())
    assert row1.run_id != row2.run_id
    with pytest.raises(ValueError, match="run_id"):
        store.save_run([row1, row2])


def test_passed_false_persisted(store: DbResultsStore) -> None:
    run_id = new_run_id()
    store.save_run([_make_result(run_id, value=0.0, passed=False)])
    loaded = store.load_run(run_id)
    assert loaded[0].passed is False


def test_value_precision_preserved(store: DbResultsStore) -> None:
    """Numeric(14,6) should round-trip six decimal places."""
    run_id = new_run_id()
    store.save_run([_make_result(run_id, value=0.123456)])
    loaded = store.load_run(run_id)
    assert loaded[0].value == pytest.approx(0.123456, rel=1e-5)


# ---------------------------------------------------------------------------
# FK + integrity constraints
# ---------------------------------------------------------------------------


def test_orphan_eval_result_raises_integrity_error(engine: Engine) -> None:
    """Inserting an eval_result without a matching eval_runs row must fail."""
    with Session(engine) as session:
        orphan = EvalResult(
            id=uuid.uuid4(),
            eval_run_id=uuid.uuid4(),  # no corresponding EvalRun
            metric="exact_match",
            value=Decimal("1.0"),
            baseline_value=None,
            passed=True,
        )
        session.add(orphan)
        with pytest.raises(IntegrityError):
            session.commit()


def test_cascade_delete_removes_results(store: DbResultsStore, engine: Engine) -> None:
    """Deleting an eval_run must cascade to its eval_results rows."""
    run_id = new_run_id()
    store.save_run(
        [
            _make_result(run_id, metric="exact_match"),
            _make_result(run_id, metric="latency_ms"),
        ]
    )
    run_uuid = uuid.UUID(run_id)
    with Session(engine) as session:
        run = session.get(EvalRun, run_uuid)
        assert run is not None
        session.delete(run)
        session.commit()

    with Session(engine) as session:
        remaining = list(
            session.scalars(select(EvalResult).where(EvalResult.eval_run_id == run_uuid))
        )
    assert remaining == []
