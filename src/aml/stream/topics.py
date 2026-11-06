"""Topic definitions, created idempotently ("create if missing")."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TOPIC_ALREADY_EXISTS = 36  # Kafka protocol error code


@dataclass(frozen=True)
class TopicSpec:
    name: str
    partitions: int
    retention_days: int


def specs(raw_topic: str, dlq_topic: str) -> list[TopicSpec]:
    return [
        # 3 partitions: up to 3 consumers in parallel; ordering is per account (message key).
        TopicSpec(raw_topic, partitions=3, retention_days=7),
        # Dead letters are kept longer: someone has to investigate and replay them.
        TopicSpec(dlq_topic, partitions=1, retention_days=30),
    ]


def ensure_topics(admin: Any, wanted: list[TopicSpec], replication: int = 1) -> list[str]:
    """Create missing topics. replication=1 for one local broker; production uses 3."""
    from confluent_kafka.admin import NewTopic  # type: ignore[attr-defined]

    new = [
        NewTopic(
            t.name,
            num_partitions=t.partitions,
            replication_factor=replication,
            config={"retention.ms": str(t.retention_days * 86_400_000)},
        )
        for t in wanted
    ]
    created = []
    for name, future in admin.create_topics(new).items():
        try:
            future.result()
            created.append(name)
        except Exception as e:  # KafkaException wrapping a KafkaError
            err = e.args[0] if e.args else None
            if getattr(err, "code", lambda: None)() != TOPIC_ALREADY_EXISTS:
                raise
    return created
