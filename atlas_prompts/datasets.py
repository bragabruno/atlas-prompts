"""Pydantic v2 models for golden-set JSONL records (REG-7).

Each record in a golden-set file represents one eval case: the input sent to the
agent, the expected output (for exact- and/or semantic-match), and the citation
requirements the eval runner (REG-8) must verify.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class CitationRequirement(BaseModel):
    """One required citation in the agent's response.

    The eval runner checks that every required source_id appears in the list of
    citations returned by the agent.  ``min_count`` lets a case assert that a
    particular source must be cited at least N times (default 1).
    """

    source_id: Annotated[str, Field(min_length=1)]
    """Opaque identifier matching the source document (e.g. 'EUR-2024-REACH-Annex-XVII')."""

    min_count: Annotated[int, Field(ge=1)] = 1
    """Minimum number of times this source must appear in the citations list."""


class GoldenRecord(BaseModel):
    """One row in a golden-set JSONL file.

    Design contract for the eval runner (REG-8):
    - ``expected_exact`` → string equality check (optional; omit for open-ended cases).
    - ``expected_semantic`` → reference text for embedding cosine-similarity check
      (optional; omit when exact match is sufficient).
    - ``required_citations`` → each entry must appear in the agent response citations.

    At least one of ``expected_exact`` or ``expected_semantic`` MUST be present.
    """

    input: Annotated[str, Field(min_length=1)]
    """The user query or test prompt sent verbatim to the agent."""

    expected_exact: str | None = None
    """Expected verbatim answer; None means skip the exact-match metric."""

    expected_semantic: str | None = None
    """Reference text for semantic similarity; None means skip the semantic-match metric."""

    required_citations: list[CitationRequirement] = Field(default_factory=list)
    """Sources that must appear in the agent's citation list."""

    tags: list[str] = Field(default_factory=list)
    """Free-form labels for filtering (e.g. ['reach', 'annex-xvii'])."""

    @model_validator(mode="after")
    def _at_least_one_expected(self) -> GoldenRecord:
        if self.expected_exact is None and self.expected_semantic is None:
            raise ValueError(
                "at least one of 'expected_exact' or 'expected_semantic' must be provided"
            )
        return self
