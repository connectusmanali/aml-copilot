"""Validating consumer: the data contract's gate for streamed transactions.

For each batch of messages:
  1. decode + validate every message against the Transaction contract
  2. send failures to the dead-letter topic, with the reason in a header, and wait for the ack
  3. write valid rows (and the rejects) to Postgres in ONE database transaction
  4. only then commit the Kafka offsets

Delivery guarantee: at-least-once + idempotent sink = each transaction stored exactly once.
If the process dies after step 3 but before step 4, the batch is consumed again on restart;
ON CONFLICT DO NOTHING on the deterministic txn_id makes the second write a no-op. The DLQ
topic may then hold a duplicate, but its Postgres copy (raw.rejected_records) is de-duplicated
by hash. Committing offsets BEFORE the database write would instead lose data on a crash.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import ValidationError

from aml.data.schemas import Transaction
from aml.data.validate import format_errors
from aml.stream.clients import Consumer, Message, Producer
from aml.stream.serde import DecodeError, decode

PARTITION_EOF = -191  # librdkafka "reached end of partition": informational, not an error


class BrokerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeadLetter:
    value: bytes | None
    key: bytes | None
    error: str
    topic: str
    partition: int
    offset: int
    payload: dict[str, Any] | None  # decoded record when it was at least valid JSON

    def headers(self) -> list[tuple[str, bytes]]:
        return [
            ("error", self.error[:1000].encode()),
            ("source.topic", self.topic.encode()),
            ("source.partition", str(self.partition).encode()),
            ("source.offset", str(self.offset).encode()),
        ]


@dataclass
class Batch:
    rows: list[dict[str, Any]] = field(default_factory=list)
    dead: list[DeadLetter] = field(default_factory=list)


def process(messages: Sequence[Message]) -> Batch:
    """Pure: messages in, valid rows and dead letters out. No I/O, so it is easy to test."""
    batch = Batch()
    for m in messages:
        err = m.error()
        if err is not None:
            if err.code() == PARTITION_EOF:
                continue
            raise BrokerError(err.str())
        payload: dict[str, Any] | None = None
        try:
            payload = decode(m.value())
            txn = Transaction.model_validate(payload)
        except DecodeError as e:
            reason = f"decode: {e}"
        except ValidationError as e:
            reason = format_errors(e)
        else:
            batch.rows.append(txn.model_dump())
            continue
        batch.dead.append(
            DeadLetter(
                m.value(),
                m.key(),
                reason,
                m.topic() or "",
                m.partition() or 0,
                m.offset() or 0,
                payload,
            )
        )
    return batch


class Sink(Protocol):
    def write(self, rows: list[dict[str, Any]], dead: list[DeadLetter]) -> int: ...


@dataclass
class ConsumeStats:
    consumed: int = 0
    valid: int = 0
    dead: int = 0
    inserted: int = 0
    batches: int = 0


def run(
    consumer: Consumer,
    dlq_producer: Producer,
    sink: Sink,
    *,
    dlq_topic: str,
    batch_size: int = 500,
    poll_timeout_s: float = 1.0,
    idle_timeout_s: float = 10.0,
    max_messages: int | None = None,
    stop: Callable[[], bool] = lambda: False,
    on_batch: Callable[[ConsumeStats], None] | None = None,
) -> ConsumeStats:
    stats = ConsumeStats()
    idle = 0.0
    while not stop():
        messages = consumer.consume(num_messages=batch_size, timeout=poll_timeout_s)
        if not messages:
            idle += poll_timeout_s
            if idle >= idle_timeout_s:
                break
            continue
        idle = 0.0
        batch = process(messages)

        for d in batch.dead:
            dlq_producer.produce(dlq_topic, value=d.value, key=d.key, headers=d.headers())
        if batch.dead and dlq_producer.flush(10) > 0:
            raise BrokerError("dead letters not acknowledged by the broker; offsets NOT committed")

        stats.inserted += sink.write(batch.rows, batch.dead)  # one DB transaction
        consumer.commit(asynchronous=False)  # only after the data is safe

        stats.consumed += len(messages)
        stats.valid += len(batch.rows)
        stats.dead += len(batch.dead)
        stats.batches += 1
        if on_batch:
            on_batch(stats)
        if max_messages is not None and stats.consumed >= max_messages:
            break
    return stats
