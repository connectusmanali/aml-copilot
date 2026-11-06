"""The small slice of the Kafka client API this project uses, as typing Protocols.

Production code is written against these Protocols; confluent_kafka's Producer and Consumer
satisfy them, and so do the in-memory fakes in tests/fakes.py. That is how the consumer's
delivery guarantees are tested without a broker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

Headers = list[tuple[str, bytes]]


class KafkaError(Protocol):
    def code(self) -> int: ...
    def str(self) -> str: ...


class Message(Protocol):
    def error(self) -> KafkaError | None: ...
    def value(self) -> bytes | None: ...
    def key(self) -> bytes | None: ...
    def topic(self) -> str | None: ...
    def partition(self) -> int | None: ...
    def offset(self) -> int | None: ...


class Producer(Protocol):
    def produce(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        headers: Headers | None = None,
        on_delivery: Callable[[Any, Any], None] | None = None,
    ) -> None: ...
    def poll(self, timeout: float = ...) -> int: ...
    def flush(self, timeout: float = ...) -> int: ...


class Consumer(Protocol):
    def subscribe(self, topics: Sequence[str]) -> None: ...
    def consume(self, num_messages: int = ..., timeout: float = ...) -> list[Any]: ...
    def commit(self, asynchronous: bool = ...) -> Any: ...
    def close(self) -> None: ...


def producer_config(bootstrap: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": bootstrap,
        # Idempotent producer: broker de-duplicates retries, so a network blip never writes a
        # message twice. Requires acks=all.
        "enable.idempotence": True,
        "acks": "all",
        "compression.type": "zstd",
        "linger.ms": 20,  # wait up to 20 ms to batch messages: far fewer requests
    }


def consumer_config(bootstrap: str, group: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        # Offsets are committed by our code, only after the database commit (at-least-once).
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "enable.partition.eof": False,
    }
