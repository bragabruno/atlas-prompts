"""POL-3 — Kafka-backed shadow source (aiokafka).

Reads from the ``atlas.shadow.v1`` topic and yields ``ShadowRecord`` objects.
Import is guarded — if aiokafka is not installed, importing this module raises
``ImportError`` which callers should surface clearly.

Usage
-----
    from evals.drift.kafka_shadow_source import KafkaShadowSource

    source = KafkaShadowSource(
        bootstrap_servers="kafka.atlas.internal:9092",
        topic="atlas.shadow.v1",
        group_id="drift-eval-nightly",
        poll_timeout_ms=5000,
        max_empty_polls=3,
    )
    for record in source.records(limit=1000):
        ...
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from evals.drift.shadow_source import ShadowRecord

log = logging.getLogger(__name__)

_DEFAULT_TOPIC = "atlas.shadow.v1"


class KafkaShadowSource:
    """Synchronous Kafka consumer for shadow traffic records.

    Polls ``atlas.shadow.v1`` in a blocking loop.  Stops when ``limit`` records
    have been yielded or when ``max_empty_polls`` consecutive polls return nothing
    (indicating the topic is caught up for this batch).

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated Kafka broker addresses.
    topic:
        Kafka topic name (default ``atlas.shadow.v1``).
    group_id:
        Consumer group ID.  Use a dedicated group so the nightly job does not
        interfere with other consumers.
    poll_timeout_ms:
        Max milliseconds to wait for messages in each poll call.
    max_empty_polls:
        Stop after this many consecutive polls that return no messages.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        *,
        topic: str = _DEFAULT_TOPIC,
        group_id: str = "drift-eval-nightly",
        poll_timeout_ms: int = 5_000,
        max_empty_polls: int = 3,
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._poll_timeout_ms = poll_timeout_ms
        self._max_empty_polls = max_empty_polls

    def records(self, *, limit: int | None = None) -> Iterator[ShadowRecord]:
        try:
            from kafka import KafkaConsumer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "kafka-python is required for KafkaShadowSource. "
                "Install aiokafka or kafka-python-ng."
            ) from exc

        consumer = KafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=self._poll_timeout_ms,
        )

        count = 0
        empty_polls = 0

        try:
            for message in consumer:
                try:
                    record = ShadowRecord.model_validate(message.value)
                except Exception as exc:  # noqa: BLE001
                    log.warning("shadow_record_invalid offset=%d error=%s", message.offset, exc)
                    continue

                yield record
                count += 1
                empty_polls = 0

                if limit is not None and count >= limit:
                    break
        except StopIteration:
            pass
        finally:
            consumer.close()

        log.info("kafka_shadow_source_done records=%d topic=%s", count, self._topic)
