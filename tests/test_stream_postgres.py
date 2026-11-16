"""The consumer's REAL Postgres sink, fed by the in-memory broker: crash recovery end to end."""

import pandas as pd
import psycopg
import pytest

from aml.stream.consumer import run
from aml.stream.producer import replay
from aml.stream.sink import PostgresSink
from tests.fakes import FakeBroker, FakeConsumer, FakeProducer

pytestmark = pytest.mark.integration
RAW, DLQ = "transactions.raw", "transactions.dlq"


def test_stream_to_postgres_survives_a_crash(pg_dsn: str, txns: pd.DataFrame) -> None:
    rows = txns.head(400).to_dict(orient="records")
    broker = FakeBroker({RAW: 3, DLQ: 1})
    sent = replay(rows, FakeProducer(broker), RAW, corrupt_every=40)

    sink = PostgresSink(pg_dsn, f"stream:{RAW}")
    crashing = FakeConsumer(broker, "g")
    crashing.subscribe([RAW])
    crashing.commit = lambda asynchronous=True: (_ for _ in ()).throw(SystemExit("killed"))
    with pytest.raises(SystemExit):
        run(crashing, FakeProducer(broker), sink, dlq_topic=DLQ, batch_size=150)
    sink.close("failed")

    sink2 = PostgresSink(pg_dsn, f"stream:{RAW}")
    restarted = FakeConsumer(broker, "g")
    restarted.subscribe([RAW])
    stats = run(
        restarted, FakeProducer(broker), sink2, dlq_topic=DLQ, batch_size=150, idle_timeout_s=1
    )
    sink2.close()

    valid = len(rows) - sent.corrupted
    assert stats.consumed == len(rows)  # first batch redelivered after the crash
    with psycopg.connect(pg_dsn) as conn:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT txn_id FROM raw.transactions WHERE run_id IN "
                "(SELECT run_id FROM raw.ingest_runs WHERE source = %s)",
                (f"stream:{RAW}",),
            )
        ]
        assert len(ids) == len(set(ids)) == valid  # each stored exactly once
        (rejected,) = conn.execute(
            "SELECT count(*) FROM raw.rejected_records WHERE source = %s", (f"stream:{RAW}",)
        ).fetchone()
        assert rejected == sent.corrupted  # dead letters persisted once despite redelivery
        statuses = [
            r[0]
            for r in conn.execute(
                "SELECT status FROM raw.ingest_runs WHERE source = %s ORDER BY started_at",
                (f"stream:{RAW}",),
            )
        ]
        assert statuses == ["failed", "succeeded"]
