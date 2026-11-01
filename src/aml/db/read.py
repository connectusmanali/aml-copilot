"""Read query results from Postgres into DataFrames.

Postgres is the system of record: features, quality checks and the graph load read from here,
so data that arrived through the stream is included, not just the last batch file.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import psycopg

# Amounts are cast to float8 for analytics (features, graph). Exact NUMERIC stays in the table.
TRANSACTIONS_SQL = """
SELECT txn_id, ts, from_account, to_account,
       amount_paid::float8 AS amount_paid, payment_currency,
       amount_received::float8 AS amount_received, receiving_currency,
       payment_format, is_laundering
FROM raw.transactions
ORDER BY ts, txn_id
"""

ACCOUNTS_SQL = """
SELECT a.account_id, a.customer_id, a.bank_id, c.segment, c.occupation, c.country,
       c.risk_rating, c.expected_monthly_volume::float8 AS expected_monthly_volume,
       c.onboarded_at
FROM ref.accounts a JOIN ref.customers c USING (customer_id)
"""


def fetch_frame(dsn: str, query: str, params: Any = None) -> pd.DataFrame:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(query.encode(), params)
        cols = [d.name for d in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_count(dsn: str, query: str) -> int:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(query.encode()).fetchone()
    return int(row[0]) if row else 0
