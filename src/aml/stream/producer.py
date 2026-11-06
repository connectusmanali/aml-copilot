"""Replay producer: publishes transactions to Kafka as if a core banking system were emitting them.

It sends rows in time order at a chosen rate, keyed by sender account. Kafka keeps messages with
the same key in the same partition, in order, so each account's payments are consumed in the
order they happened. Step 3's real-time scoring relies on that ordering.

The producer does not validate (a source system wouldn't). --corrupt-every N deliberately
breaks every Nth message, so the consumer's dead-letter path can be exercised end to end.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from aml.stream.clients import Producer
from aml.stream.serde import encode

CORRUPTIONS = ("bad_json", "negative_amount", "unknown_format", "missing_account")


@dataclass
class ReplayStats:
    sent: int = 0
    corrupted: int = 0
    failed: int = 0


def corrupt(value: bytes, kind: str) -> bytes:
    if kind == "bad_json":
        return value[: len(value) // 2]  # truncated message
    record = json.loads(value)
    if kind == "negative_amount":
        record["amount_paid"] = record["amount_received"] = "-50.00"
    elif kind == "unknown_format":
        record["payment_format"] = "Carrier pigeon"
    elif kind == "missing_account":
        record["from_account"] = ""
    return json.dumps(record, separators=(",", ":")).encode()


def replay(
    rows: Iterable[Mapping[str, Any]],
    producer: Producer,
    topic: str,
    *,
    rate: float | None = None,
    corrupt_every: int = 0,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ReplayStats:
    stats = ReplayStats()

    def on_delivery(err: Any, _msg: Any) -> None:
        if err is not None:
            stats.failed += 1

    start = clock()
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        value = encode(row)
        if corrupt_every and (i + 1) % corrupt_every == 0:
            value = corrupt(value, CORRUPTIONS[(i // corrupt_every) % len(CORRUPTIONS)])
            stats.corrupted += 1
        producer.produce(
            topic, value=value, key=str(row.get("from_account")).encode(), on_delivery=on_delivery
        )
        producer.poll(0)  # serve delivery callbacks
        stats.sent += 1
        if rate:
            ahead = start + stats.sent / rate - clock()
            if ahead > 0:
                sleep(ahead)
    stats.failed += producer.flush(30)  # messages still undelivered after 30 s
    return stats
