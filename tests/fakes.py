"""In-memory stand-ins for Kafka and Neo4j, matching the Protocols in aml.stream.clients.

FakeBroker keeps partitions, offsets and per-group committed offsets the way Kafka does, so tests
can simulate a crash between "data written" and "offset committed" and check nothing is lost or
duplicated.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeError:
    _code: int
    _text: str = "fake broker error"

    def code(self) -> int:
        return self._code

    def str(self) -> str:
        return self._text


@dataclass
class FakeMessage:
    _topic: str
    _partition: int
    _offset: int
    _key: bytes | None
    _value: bytes | None
    headers: list[tuple[str, bytes]] = field(default_factory=list)
    _error: FakeError | None = None

    def error(self) -> FakeError | None:
        return self._error

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class FakeBroker:
    def __init__(self, partitions: dict[str, int]) -> None:
        self.log: dict[str, list[list[FakeMessage]]] = {
            t: [[] for _ in range(n)] for t, n in partitions.items()
        }
        self.committed: dict[tuple[str, str, int], int] = {}

    def append(
        self, topic: str, key: bytes | None, value: bytes | None, headers: list[tuple[str, bytes]]
    ) -> FakeMessage:
        parts = self.log[topic]
        p = zlib.crc32(key or b"") % len(parts)  # same key -> same partition, like Kafka
        msg = FakeMessage(topic, p, len(parts[p]), key, value, headers)
        parts[p].append(msg)
        return msg

    def messages(self, topic: str) -> list[FakeMessage]:
        return [m for part in self.log[topic] for m in part]


class FakeProducer:
    def __init__(self, broker: FakeBroker, undelivered_on_flush: int = 0) -> None:
        self.broker = broker
        self.undelivered_on_flush = undelivered_on_flush

    def produce(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
        on_delivery: Callable[[Any, Any], None] | None = None,
    ) -> None:
        msg = self.broker.append(topic, key, value, headers or [])
        if on_delivery:
            on_delivery(None, msg)

    def poll(self, timeout: float = 0) -> int:
        return 0

    def flush(self, timeout: float = 0) -> int:
        return self.undelivered_on_flush


class FakeConsumer:
    def __init__(self, broker: FakeBroker, group: str) -> None:
        self.broker = broker
        self.group = group
        self.position: dict[tuple[str, int], int] = {}

    def subscribe(self, topics: Sequence[str]) -> None:
        for t in topics:
            for p in range(len(self.broker.log[t])):
                self.position[(t, p)] = self.broker.committed.get((self.group, t, p), 0)

    def consume(self, num_messages: int = 1, timeout: float = -1) -> list[FakeMessage]:
        out: list[FakeMessage] = []
        progressed = True
        while len(out) < num_messages and progressed:  # round-robin across partitions
            progressed = False
            for (t, p), pos in self.position.items():
                part = self.broker.log[t][p]
                if pos < len(part) and len(out) < num_messages:
                    out.append(part[pos])
                    self.position[(t, p)] = pos + 1
                    progressed = True
        return out

    def commit(self, asynchronous: bool = True) -> None:
        for (t, p), pos in self.position.items():
            self.broker.committed[(self.group, t, p)] = pos

    def close(self) -> None:
        pass


class MemorySink:
    """Idempotent in-memory sink keyed by txn_id, with optional failure injection."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.dead: list[Any] = []
        self.calls = 0
        self.fail_on_call = fail_on_call

    def write(self, rows: list[dict[str, Any]], dead: list[Any]) -> int:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise ConnectionError("database unavailable")
        new = [r for r in rows if r["txn_id"] not in self.rows]
        for r in new:
            self.rows[r["txn_id"]] = r
        self.dead.extend(dead)
        return len(new)


class FakeResult(list[dict[str, Any]]):
    def single(self) -> dict[str, Any] | None:
        return self[0] if self else None


class FakeSession:
    """Records every Cypher statement and its parameters."""

    def __init__(self, single: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._single = single or {"nodes": 0, "rels": 0}

    def run(self, query: str, parameters: dict[str, Any] | None = None, **kw: Any) -> FakeResult:
        self.calls.append((query, parameters or {}))
        return FakeResult([self._single])


def by_topic(broker: FakeBroker, topic: str) -> dict[int, list[FakeMessage]]:
    out: dict[int, list[FakeMessage]] = defaultdict(list)
    for m in broker.messages(topic):
        out[m.partition()].append(m)
    return out
