"""Postgres sink for the consumer: valid rows and rejects, written in one transaction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import psycopg

from aml.data.ibm import NORMALIZED_COLUMNS
from aml.data.load import REJECTED_TABLE, add_to_run, finish_run, merge_frame, start_run
from aml.stream.consumer import DeadLetter


def dead_letter_records(dead: list[DeadLetter], source: str, run_id: str) -> pd.DataFrame:
    payloads = []
    for d in dead:
        body: dict[str, Any] = (
            d.payload
            if d.payload is not None
            else {"raw": (d.value or b"").decode("utf-8", errors="replace")}
        )
        payloads.append(json.dumps(body, sort_keys=True, default=str))
    return pd.DataFrame(
        {
            "record_hash": [hashlib.sha256(f"{source}|{p}".encode()).hexdigest() for p in payloads],
            "run_id": run_id,
            "source": source,
            "payload": payloads,
            "error": [d.error for d in dead],
        }
    )


class PostgresSink:
    def __init__(self, dsn: str, source: str) -> None:
        self.conn = psycopg.connect(dsn)
        self.source = source
        self.run_id = start_run(self.conn, source)

    def write(self, rows: list[dict[str, Any]], dead: list[DeadLetter]) -> int:
        valid = (
            pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
            .drop_duplicates("txn_id")
            .assign(run_id=self.run_id)
        )
        rejected = dead_letter_records(dead, self.source, self.run_id).drop_duplicates(
            "record_hash"
        )
        with self.conn.transaction(), self.conn.cursor() as cur:
            inserted = merge_frame(cur, "raw.transactions", "txn_id", valid)
            merge_frame(cur, *REJECTED_TABLE, rejected)
            add_to_run(cur, self.run_id, len(valid), len(rejected), inserted)
        return inserted

    def close(self, status: str = "succeeded") -> None:
        finish_run(self.conn, self.run_id, status)
        self.conn.close()
