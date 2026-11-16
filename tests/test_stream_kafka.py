"""Real broker end to end: topics -> producer -> consumer -> Postgres + DLQ.

Runs in CI against Redpanda (and locally after `make up-lite`). Topic and group names are unique
per run, so it never interferes with your own topics.
"""

import uuid

import pandas as pd
import psycopg
import pytest

kafka = pytest.importorskip("confluent_kafka", reason="needs --extra stream")

from confluent_kafka.admin import AdminClient  # noqa: E402

from aml.config import get_settings  # noqa: E402
from aml.stream.clients import consumer_config, producer_config  # noqa: E402
from aml.stream.consumer import run  # noqa: E402
from aml.stream.producer import replay  # noqa: E402
from aml.stream.sink import PostgresSink  # noqa: E402
from aml.stream.topics import ensure_topics, specs  # noqa: E402

pytestmark = pytest.mark.kafka  # also needs Postgres (pg_dsn)


def test_real_broker_round_trip(pg_dsn: str, txns: pd.DataFrame) -> None:
    s = get_settings()
    tag = uuid.uuid4().hex[:8]
    raw, dlq = f"test.{tag}.raw", f"test.{tag}.dlq"
    admin = AdminClient({"bootstrap.servers": s.kafka_bootstrap})
    assert set(ensure_topics(admin, specs(raw, dlq))) == {raw, dlq}
    assert ensure_topics(admin, specs(raw, dlq)) == []  # idempotent

    rows = txns.head(300).to_dict(orient="records")
    sent = replay(rows, kafka.Producer(producer_config(s.kafka_bootstrap)), raw, corrupt_every=20)
    assert sent.failed == 0

    consumer = kafka.Consumer(consumer_config(s.kafka_bootstrap, f"test-{tag}"))
    consumer.subscribe([raw])
    sink = PostgresSink(pg_dsn, f"stream:{raw}")
    try:
        stats = run(
            consumer,
            kafka.Producer(producer_config(s.kafka_bootstrap)),
            sink,
            dlq_topic=dlq,
            batch_size=100,
            idle_timeout_s=20,
            max_messages=len(rows),
        )
    finally:
        consumer.close()
        sink.close()

    assert stats.consumed == len(rows)
    assert stats.dead == sent.corrupted
    with psycopg.connect(pg_dsn) as conn:
        (stored,) = conn.execute(
            "SELECT count(*) FROM raw.transactions t JOIN raw.ingest_runs r USING (run_id) "
            "WHERE r.source = %s",
            (f"stream:{raw}",),
        ).fetchone()
    assert stored == len(rows) - sent.corrupted

    reader = kafka.Consumer(consumer_config(s.kafka_bootstrap, f"test-{tag}-dlq"))
    reader.subscribe([dlq])
    dead = []
    for _ in range(30):
        dead += [m for m in reader.consume(num_messages=100, timeout=1.0) if m.error() is None]
        if len(dead) >= sent.corrupted:
            break
    reader.close()
    assert len(dead) == sent.corrupted
    assert all(dict(m.headers())["error"] for m in dead)
    admin.delete_topics([raw, dlq])
