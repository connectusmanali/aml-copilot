"""Streaming command line (needs `uv sync --extra stream` and a running broker).

aml-stream topics                                        # create topics if missing
aml-stream produce --source synthetic --rate 200         # replay transactions
aml-stream produce --source synthetic --corrupt-every 50 # ...with broken messages
aml-stream consume                                       # validate -> Postgres / DLQ
"""

from __future__ import annotations

import argparse
import signal
import sys
from typing import Any

from aml.config import get_settings
from aml.data import ibm, synthetic
from aml.stream.clients import consumer_config, producer_config
from aml.stream.consumer import ConsumeStats, run
from aml.stream.producer import replay
from aml.stream.topics import ensure_topics, specs


def _kafka() -> Any:
    try:
        import confluent_kafka
    except ImportError:
        sys.exit("confluent-kafka is not installed: run `uv sync --extra stream`")
    return confluent_kafka


def topics() -> None:
    s = get_settings()
    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": s.kafka_bootstrap})
    created = ensure_topics(admin, specs(s.kafka_topic_raw, s.kafka_topic_dlq))
    print(f"created: {', '.join(created)}" if created else "topics already exist")


def produce(
    source: str, size: ibm.Size, rate: float | None, corrupt_every: int, limit: int | None
) -> None:
    s = get_settings()
    kafka = _kafka()
    if source == "synthetic":
        n_acc, n_norm, n_pat = synthetic.SIZES[size]
        raw = synthetic.generate(n_acc, n_norm, n_pat, seed=s.random_seed)
    else:
        n = ibm.SIZES[size]
        raw = ibm.read_raw(s.raw_dir / "HI-Small_Trans.csv", nrows=n)
    rows: list[dict[str, Any]] = ibm.normalize(raw).to_dict(orient="records")  # type: ignore[assignment]  # time order, not validated
    producer = kafka.Producer(producer_config(s.kafka_bootstrap))
    stats = replay(
        rows, producer, s.kafka_topic_raw, rate=rate, corrupt_every=corrupt_every, limit=limit
    )
    print(
        f"sent {stats.sent:,} to {s.kafka_topic_raw} "
        f"({stats.corrupted} deliberately corrupted, {stats.failed} failed)"
    )
    if stats.failed:
        sys.exit(1)


def consume(batch_size: int, idle_timeout: float, max_messages: int | None) -> None:
    from aml.stream.sink import PostgresSink

    s = get_settings()
    kafka = _kafka()
    stopping = False

    def request_stop(*_: Any) -> None:  # Ctrl-C / docker stop: finish the batch, then exit
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    consumer = kafka.Consumer(consumer_config(s.kafka_bootstrap, s.kafka_group))
    consumer.subscribe([s.kafka_topic_raw])
    dlq = kafka.Producer(producer_config(s.kafka_bootstrap))
    sink = PostgresSink(s.postgres_dsn, f"stream:{s.kafka_topic_raw}")

    def progress(st: ConsumeStats) -> None:
        print(
            f"batch {st.batches}: consumed {st.consumed:,}, valid {st.valid:,}, "
            f"dead-lettered {st.dead:,}, new rows {st.inserted:,}"
        )

    status = "failed"
    try:
        stats = run(
            consumer,
            dlq,
            sink,
            dlq_topic=s.kafka_topic_dlq,
            batch_size=batch_size,
            idle_timeout_s=idle_timeout,
            max_messages=max_messages,
            stop=lambda: stopping,
            on_batch=progress,
        )
        status = "succeeded"
        print(
            f"done (run {sink.run_id}): {stats.consumed:,} consumed, {stats.inserted:,} new, "
            f"{stats.dead:,} dead-lettered"
        )
    finally:
        consumer.close()
        sink.close(status)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-stream")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("topics", help="create topics if missing")
    pp = sub.add_parser("produce", help="replay transactions into Kafka")
    pp.add_argument("--source", choices=["synthetic", "ibm"], default="synthetic")
    pp.add_argument("--size", choices=list(ibm.SIZES), default="small")
    pp.add_argument("--rate", type=float, default=None, help="messages per second (default: max)")
    pp.add_argument("--corrupt-every", type=int, default=0)
    pp.add_argument("--limit", type=int, default=None)
    cp = sub.add_parser("consume", help="validate messages into Postgres; failures to the DLQ")
    cp.add_argument("--batch-size", type=int, default=500)
    cp.add_argument(
        "--idle-timeout",
        type=float,
        default=10.0,
        help="exit after this many seconds without messages",
    )
    cp.add_argument("--max-messages", type=int, default=None)
    args = p.parse_args(argv)

    if args.cmd == "topics":
        topics()
    elif args.cmd == "produce":
        produce(args.source, args.size, args.rate, args.corrupt_every, args.limit)
    else:
        consume(args.batch_size, args.idle_timeout, args.max_messages)


if __name__ == "__main__":
    main()
