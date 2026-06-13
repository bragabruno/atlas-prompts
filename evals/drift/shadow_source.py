"""POL-3 — Shadow traffic source abstraction.

Defines the ``ShadowRecord`` domain type and the ``ShadowSource`` Protocol that
decouples the drift-eval job from any specific transport (Kafka, file, etc.).

Implementations
---------------
- ``JsonlShadowSource`` — reads from a local JSONL file (tests / dry runs).
- ``KafkaShadowSource`` — streams from the ``atlas.shadow.v1`` Kafka topic;
  lives in ``evals.drift.kafka_shadow_source`` to keep aiokafka import optional.

Shadow message format (``atlas.shadow.v1``)
-------------------------------------------
Each Kafka message is a JSON object:
    {
      "prompt_version":  "<name>/<semver>",
      "input":           "<user message>",
      "output":          "<model response>",
      "model":           "<model alias>",
      "latency_ms":      <float>,
      "cost_usd":        <float>
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ShadowRecord(BaseModel):
    """One sampled production request/response from ``atlas.shadow.v1``."""

    prompt_version: str
    input: str
    output: str
    model: str
    latency_ms: float
    cost_usd: float


@runtime_checkable
class ShadowSource(Protocol):
    """Structural protocol for any shadow-traffic source."""

    def records(self, *, limit: int | None = None) -> Iterator[ShadowRecord]:
        """Yield up to ``limit`` records (or all if limit is None)."""
        ...


class JsonlShadowSource:
    """Read shadow records from a local JSONL file.

    Each line must be a JSON object with the ``ShadowRecord`` fields.
    Blank lines are skipped.  Used in tests and dry-run scenarios.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def records(self, *, limit: int | None = None) -> Iterator[ShadowRecord]:
        count = 0
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            yield ShadowRecord.model_validate(json.loads(line))
            count += 1
            if limit is not None and count >= limit:
                break
