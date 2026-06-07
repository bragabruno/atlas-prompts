"""SQLAlchemy 2.0 declarative base for the atlas-prompts eval schema (REG-6).

`Base` is the single declarative registry every ORM model in this package
inherits from, so `Base.metadata` is the schema source of truth Alembic diffs
against (ADR-010).  This repo owns `eval_runs` and `eval_results`; the gateway
DB owns `prompt_versions` (ADR-015).  Per ADR-016 nothing outside
`eval_runs.db` imports these models directly.  See atlas-docs/03 §1.6.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base; `Base.metadata` is the authoritative eval schema."""
