"""SQLAlchemy 2.0 typed models for eval_runs and eval_results (REG-6).

Mirrors the DDL in atlas-docs/03 §1.6.  This repo owns both tables (ADR-015);
`prompt_versions` lives in the gateway DB, so `prompt_version_id` is a plain
UUID column — a documented logical reference, not a hard cross-DB FK constraint.
Models use SQLAlchemy 2.0 typed `Mapped[...]` columns and form the schema source
of truth Alembic diffs against.  Column types are chosen to compile to the
atlas-docs Postgres DDL in production while remaining valid under SQLite for the
offline schema smoke test.  See atlas-docs/03 §1.6.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from eval_runs.db.base import Base


class EvalRun(Base):
    """One evaluation run for a specific prompt version (atlas-docs/03 §1.6).

    `prompt_version_id` references `prompt_versions.id` in the **gateway** DB
    (ADR-015).  There is no hard cross-DB FK — the column stores the UUID as a
    logical reference only.  Applications are responsible for supplying a valid
    UUID that exists in the gateway DB.

    `triggered_by` encodes the origin: ``"ci"``, ``"manual:<user>"``, or
    ``"shadow_drift"``.
    """

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Logical reference to prompt_versions.id in the gateway DB (no cross-DB FK).
    prompt_version_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    dataset_version: Mapped[str] = mapped_column(Text, nullable=False)
    triggered_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_eval_runs_prompt_version", "prompt_version_id", "created_at"),)


class EvalResult(Base):
    """One metric measurement within an eval run (atlas-docs/03 §1.6).

    Each row records one metric (e.g. ``"accuracy"``, ``"latency_p95"``,
    ``"cost_usd"``) for a completed `EvalRun`.  `baseline_value` is NULL when
    this is the first run for the prompt.  `passed` is set by comparing `value`
    against `baseline_value` using metric-specific thresholds defined in the
    eval dataset manifest (atlas-docs/03 §7.2).
    """

    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(14, 6), nullable=False)
    baseline_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (Index("idx_eval_results_eval_run_id", "eval_run_id"),)
