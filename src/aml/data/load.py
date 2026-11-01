"""Load prepared Parquet files into Postgres, idempotently, with lineage.

Pattern: COPY into a temporary staging table (fast), then
INSERT ... ON CONFLICT DO NOTHING into the real table (safe to re-run).
Running `make load` twice leaves exactly the same rows.

Every load is recorded as a run in raw.ingest_runs; each transaction keeps the run_id of the run
that first inserted it, and rejected rows go to raw.rejected_records with their reason.
The stream consumer reuses merge_frame, start_run and finish_run.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

TABLES: dict[str, tuple[str, str]] = {
    # file stem -> (target table, conflict key). Order matters: foreign keys.
    "customers": ("ref.customers", "customer_id"),
    "accounts": ("ref.accounts", "account_id"),
    "watchlist": ("ref.watchlist", "entry_id"),
    "transactions": ("raw.transactions", "txn_id"),
}
REJECTED_TABLE = ("raw.rejected_records", "record_hash")
META_FILE = "_meta.json"


def _ident(qualified: str) -> sql.Identifier:
    schema, name = qualified.split(".")
    return sql.Identifier(schema, name)


def merge_frame(cur: psycopg.Cursor[Any], table: str, key: str, df: pd.DataFrame) -> int:
    """Stage df with COPY, then insert rows whose key is new. Returns rows actually inserted.

    Names go through sql.Identifier, never string formatting, so a hostile column name in a
    Parquet file cannot inject SQL.
    """
    if df.empty:
        return 0
    stage = sql.Identifier(f"stage_{table.split('.')[1]}")
    cols = sql.SQL(", ").join(map(sql.Identifier, df.columns))
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(stage))
    cur.execute(
        sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP").format(
            stage, _ident(table)
        )
    )
    with cur.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(stage, cols)) as copy:
        for row in df.itertuples(index=False, name=None):
            copy.write_row(row)
    cur.execute(
        sql.SQL("INSERT INTO {t} ({c}) SELECT {c} FROM {s} ON CONFLICT ({k}) DO NOTHING").format(
            t=_ident(table), c=cols, s=stage, k=sql.Identifier(key)
        )
    )
    return int(cur.rowcount)


def rejected_records(rejected: pd.DataFrame, source: str, run_id: str) -> pd.DataFrame:
    """Rejected rows -> raw.rejected_records rows. The hash makes reloading them idempotent."""
    if rejected.empty:
        return pd.DataFrame(columns=["record_hash", "run_id", "source", "payload", "error"])
    body = rejected.drop(columns=["error"]).astype(object)
    body = body.where(body.notna(), None)  # NaN is not valid JSON
    payloads = [
        json.dumps(rec, default=str, sort_keys=True) for rec in body.to_dict(orient="records")
    ]
    hashes = [hashlib.sha256(f"{source}|{p}".encode()).hexdigest() for p in payloads]
    return pd.DataFrame(
        {
            "record_hash": hashes,
            "run_id": run_id,
            "source": source,
            "payload": payloads,
            "error": rejected["error"].astype(str).to_numpy(),
        }
    )


def start_run(conn: psycopg.Connection[Any], source: str) -> str:
    run_id = str(uuid.uuid4())
    with conn.transaction():
        conn.execute(
            "INSERT INTO raw.ingest_runs (run_id, source) VALUES (%s, %s)", (run_id, source)
        )
    return run_id


def add_to_run(
    cur: psycopg.Cursor[Any], run_id: str, valid: int, rejected: int, inserted: int
) -> None:
    cur.execute(
        "UPDATE raw.ingest_runs SET rows_valid = rows_valid + %s, "
        "rows_rejected = rows_rejected + %s, rows_inserted = rows_inserted + %s "
        "WHERE run_id = %s",
        (valid, rejected, inserted, run_id),
    )


def finish_run(conn: psycopg.Connection[Any], run_id: str, status: str) -> None:
    with conn.transaction():
        conn.execute(
            "UPDATE raw.ingest_runs SET status = %s, finished_at = now() WHERE run_id = %s",
            (status, run_id),
        )


def read_meta(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / META_FILE
    return json.loads(path.read_text()) if path.exists() else {"source": "unknown"}


def load_all(dsn: str, processed_dir: Path) -> dict[str, int]:
    """Load customers, accounts, transactions and rejected rows. Returns rows inserted per table."""
    frames: dict[str, pd.DataFrame] = {}
    for stem in [*TABLES, "rejected"]:
        path = processed_dir / f"{stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `make data` first")
        frames[stem] = pd.read_parquet(path)

    source = f"batch:{read_meta(processed_dir)['source']}"
    inserted: dict[str, int] = {}
    with psycopg.connect(dsn) as conn:
        run_id = start_run(conn, source)
        try:
            for stem, (table, key) in TABLES.items():
                df = frames[stem]
                if stem == "transactions":
                    df = df.assign(run_id=run_id)
                t0 = time.perf_counter()
                with conn.transaction(), conn.cursor() as cur:
                    inserted[stem] = merge_frame(cur, table, key, df)
                    if stem == "transactions":
                        add_to_run(cur, run_id, len(df), 0, inserted[stem])
                print(
                    f"{table:<22} {len(df):>9,} rows read, {inserted[stem]:>9,} new "
                    f"({time.perf_counter() - t0:.1f}s)"
                )
            rej = rejected_records(frames["rejected"], source, run_id)
            with conn.transaction(), conn.cursor() as cur:
                inserted["rejected"] = merge_frame(cur, *REJECTED_TABLE, rej)
                add_to_run(cur, run_id, 0, len(rej), 0)
            print(
                f"{REJECTED_TABLE[0]:<22} {len(rej):>9,} rows read, {inserted['rejected']:>9,} new"
            )
        except Exception:
            finish_run(conn, run_id, "failed")
            raise
        finish_run(conn, run_id, "succeeded")
        print(f"run {run_id} ({source}) succeeded")
    return inserted
