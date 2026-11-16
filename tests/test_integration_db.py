"""Integration tests against a real Postgres (a throwaway database per test session)."""

from pathlib import Path

import pandas as pd
import psycopg
import pytest

from aml.data.load import load_all

pytestmark = pytest.mark.integration


def test_load_is_idempotent_and_records_lineage(pg_dsn: str, processed_dir: Path) -> None:
    first = load_all(pg_dsn, processed_dir)
    second = load_all(pg_dsn, processed_dir)
    assert first["transactions"] > 0
    assert second == {
        "customers": 0,
        "accounts": 0,
        "watchlist": 0,
        "transactions": 0,
        "rejected": 0,
    }
    with psycopg.connect(pg_dsn) as conn:
        runs = conn.execute(
            "SELECT status, rows_inserted FROM raw.ingest_runs ORDER BY started_at"
        ).fetchall()
        assert [r[0] for r in runs] == ["succeeded", "succeeded"]
        assert runs[0][1] == first["transactions"] and runs[1][1] == 0
        # every row keeps the run that FIRST inserted it
        (n_runs,) = conn.execute("SELECT count(DISTINCT run_id) FROM raw.transactions").fetchone()
        assert n_runs == 1


def test_rejected_rows_are_persisted_once(pg_dsn: str, processed_dir: Path, tmp_path: Path) -> None:
    for f in processed_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    txns = pd.read_parquet(tmp_path / "transactions.parquet")
    bad = txns.head(2).assign(error="amount_paid: Input should be greater than 0")
    bad.to_parquet(tmp_path / "rejected.parquet", index=False)

    assert load_all(pg_dsn, tmp_path)["rejected"] == 2
    assert load_all(pg_dsn, tmp_path)["rejected"] == 0  # same bad rows: not duplicated
    with psycopg.connect(pg_dsn) as conn:
        payload, error = conn.execute(
            "SELECT payload, error FROM raw.rejected_records LIMIT 1"
        ).fetchone()
    assert payload["txn_id"] in set(bad["txn_id"]) and "greater than 0" in error


def test_audit_log_is_append_only(pg_dsn: str) -> None:
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO cases.audit_log (actor, entity, action) VALUES ('test', 'case:0', 'ping')"
        )
        (before,) = conn.execute("SELECT count(*) FROM cases.audit_log").fetchone()
        for statement in [
            "DELETE FROM cases.audit_log",
            "UPDATE cases.audit_log SET actor='x'",
            "TRUNCATE cases.audit_log",
        ]:
            with pytest.raises(psycopg.errors.RaiseException):
                conn.execute(statement.encode())
        (count,) = conn.execute("SELECT count(*) FROM cases.audit_log").fetchone()
        assert count == before
