"""Initial eval schema: eval_runs, eval_results (REG-6).

Creates the two atlas-prompts-owned tables (ADR-015) with the columns, indexes,
and FK specified in atlas-docs/03 §1.6.  `prompt_version_id` on `eval_runs` is a
plain UUID column — a documented logical reference to `prompt_versions.id` in the
gateway DB; there is no hard cross-DB FK (the gateway DB owns that table per
ADR-015).  `eval_results.eval_run_id` is a local FK to `eval_runs.id` with
ON DELETE CASCADE.  Targets Azure Database for PostgreSQL via the psycopg3 sync
driver (ADR-010); column types fall back gracefully to SQLite for offline smoke.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-06-07

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Logical reference to prompt_versions.id in the gateway DB (ADR-015).
        # No hard cross-DB FK — enforced at the application layer.
        sa.Column("prompt_version_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.Text(), nullable=False),
        sa.Column("triggered_by", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_eval_runs_prompt_version",
        "eval_runs",
        ["prompt_version_id", "created_at"],
    )

    op.create_table(
        "eval_results",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "eval_run_id",
            sa.Uuid(),
            sa.ForeignKey("eval_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("value", sa.Numeric(14, 6), nullable=False),
        sa.Column("baseline_value", sa.Numeric(14, 6), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
    )
    op.create_index("idx_eval_results_eval_run_id", "eval_results", ["eval_run_id"])


def downgrade() -> None:
    op.drop_index("idx_eval_results_eval_run_id", table_name="eval_results")
    op.drop_table("eval_results")

    op.drop_index("idx_eval_runs_prompt_version", table_name="eval_runs")
    op.drop_table("eval_runs")
