"""Streaming ingestion: serialization, dead letters, and delivery guarantees (no broker needed)."""

import json
from decimal import Decimal

import pandas as pd
import pytest

from aml.stream.consumer import BrokerError, process, run
from aml.stream.producer import CORRUPTIONS, corrupt, replay
from aml.stream.serde import DecodeError, decode, encode
from tests.fakes import (
    FakeBroker,
    FakeConsumer,
    FakeError,
    FakeMessage,
    FakeProducer,
    MemorySink,
    by_topic,
)

RAW, DLQ = "transactions.raw", "transactions.dlq"


@pytest.fixture
def rows(txns: pd.DataFrame) -> list[dict]:
    return txns.head(300).to_dict(orient="records")


def broker() -> FakeBroker:
    return FakeBroker({RAW: 3, DLQ: 1})


def test_money_travels_as_exact_strings(rows: list[dict]) -> None:
    msg = decode(encode(rows[0]))
    assert isinstance(msg["amount_paid"], str)
    assert Decimal(msg["amount_paid"]) == Decimal(str(rows[0]["amount_paid"])).quantize(
        Decimal("0.01")
    )
    assert msg["ts"].startswith(str(rows[0]["ts"].date()))


@pytest.mark.parametrize("value", [b"", b"{not json", b"[1, 2]", None])
def test_decode_rejects_non_objects(value: bytes | None) -> None:
    with pytest.raises(DecodeError):
        decode(value)


def test_same_account_same_partition_in_time_order(rows: list[dict]) -> None:
    b = broker()
    replay(rows, FakeProducer(b), RAW)
    for msgs in by_topic(b, RAW).values():
        seen: dict[str, str] = {}
        for m in msgs:
            rec = decode(m.value())
            acct = rec["from_account"]
            assert rec["ts"] >= seen.get(acct, "")  # per-account order is preserved
            seen[acct] = rec["ts"]
    for acct in {r["from_account"] for r in rows}:
        partitions = {
            m.partition() for m in b.messages(RAW) if decode(m.value())["from_account"] == acct
        }
        assert len(partitions) == 1


def test_every_corruption_is_dead_lettered_with_a_reason(rows: list[dict]) -> None:
    messages = [
        FakeMessage(RAW, 0, i, b"k", corrupt(encode(rows[i]), kind))
        for i, kind in enumerate(CORRUPTIONS)
    ]
    batch = process(messages)
    assert batch.rows == []
    assert len(batch.dead) == len(CORRUPTIONS)
    reasons = " | ".join(d.error for d in batch.dead)
    for expected in ["decode", "amount_paid", "payment_format", "from_account"]:
        assert expected in reasons
    assert batch.dead[0].payload is None  # undecodable: raw bytes kept
    assert batch.dead[1].payload is not None  # decodable: record kept for inspection


def test_partition_eof_is_ignored_other_broker_errors_raise() -> None:
    eof = FakeMessage(RAW, 0, 0, None, None, _error=FakeError(-191))
    assert process([eof]).rows == []
    with pytest.raises(BrokerError):
        process([FakeMessage(RAW, 0, 0, None, None, _error=FakeError(-195, "transport"))])


def test_end_to_end_valid_rows_stored_bad_rows_dead_lettered(rows: list[dict]) -> None:
    b = broker()
    stats_p = replay(rows, FakeProducer(b), RAW, corrupt_every=25)
    consumer = FakeConsumer(b, "g")
    consumer.subscribe([RAW])
    sink = MemorySink()
    stats = run(consumer, FakeProducer(b), sink, dlq_topic=DLQ, batch_size=64, idle_timeout_s=1)

    assert stats_p.corrupted == len(rows) // 25
    assert stats.consumed == len(rows)
    assert stats.dead == stats_p.corrupted == len(b.messages(DLQ))
    assert len(sink.rows) == len(rows) - stats_p.corrupted
    dlq = b.messages(DLQ)[0]
    assert dict(dlq.headers)["error"]  # the reason travels with the message
    assert dict(dlq.headers)["source.topic"] == RAW.encode()


def test_crash_after_write_before_commit_loses_nothing_duplicates_nothing(
    rows: list[dict],
) -> None:
    """At-least-once delivery + idempotent sink = each transaction stored exactly once."""
    b = broker()
    replay(rows, FakeProducer(b), RAW)
    sink = MemorySink()

    crashing = FakeConsumer(b, "g")
    crashing.subscribe([RAW])
    crashing.commit = lambda asynchronous=True: (_ for _ in ()).throw(SystemExit("killed"))
    with pytest.raises(SystemExit):
        run(crashing, FakeProducer(b), sink, dlq_topic=DLQ, batch_size=100)
    assert len(sink.rows) == 100  # written, but offsets never committed

    restarted = FakeConsumer(b, "g")  # same group: resumes from the last COMMITTED offset
    restarted.subscribe([RAW])
    stats = run(restarted, FakeProducer(b), sink, dlq_topic=DLQ, batch_size=100, idle_timeout_s=1)
    assert stats.consumed == len(rows)  # the first 100 were redelivered...
    assert len(sink.rows) == len(rows)  # ...but stored once


def test_database_down_means_offsets_not_committed(rows: list[dict]) -> None:
    b = broker()
    replay(rows, FakeProducer(b), RAW)
    consumer = FakeConsumer(b, "g")
    consumer.subscribe([RAW])
    with pytest.raises(ConnectionError):
        run(consumer, FakeProducer(b), MemorySink(fail_on_call=1), dlq_topic=DLQ, batch_size=50)
    assert all(v == 0 for v in b.committed.values())


def test_unacknowledged_dead_letters_block_the_commit(rows: list[dict]) -> None:
    b = broker()
    replay(rows, FakeProducer(b), RAW, corrupt_every=5)
    consumer = FakeConsumer(b, "g")
    consumer.subscribe([RAW])
    sink = MemorySink()
    with pytest.raises(BrokerError):
        run(consumer, FakeProducer(b, undelivered_on_flush=1), sink, dlq_topic=DLQ, batch_size=50)
    assert sink.calls == 0 and all(v == 0 for v in b.committed.values())


def test_replay_respects_rate(rows: list[dict]) -> None:
    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        clock["t"] += s

    replay(rows[:50], FakeProducer(broker()), RAW, rate=10, sleep=sleep, clock=lambda: clock["t"])
    assert clock["t"] == pytest.approx(5.0)  # 50 messages at 10/s


def test_corrupt_keeps_message_shape() -> None:
    value = encode({"txn_id": "a" * 32, "amount_paid": 5, "amount_received": 5})
    assert json.loads(corrupt(value, "unknown_format"))["payment_format"] == "Carrier pigeon"
