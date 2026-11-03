"""Load accounts and transactions into Neo4j for investigation, idempotently.

Model:
    (:Account {id, customer_id, bank_id, segment, occupation, country, risk_rating,
               expected_monthly_volume})
    (:Account)-[:SENT {txn_id, ts, amount, currency, amount_usd, format}]->(:Account)

Design rules:
  - MERGE everywhere: loading twice changes nothing (same idea as ON CONFLICT DO NOTHING).
  - Batches of rows with UNWIND: one round trip per few thousand rows, not per row.
  - The laundering LABEL is never loaded. The graph is what an investigator, and the Step 4
    agent, may look at; if it contained the answer, every evaluation of the agent would be
    meaningless. Confirmed cases get their own property later, from the case workflow.
  - Customer attributes sit on the Account node (one customer per account today), which halves
    the node count: AuraDB Free allows 200k nodes / 400k relationships, and the loader checks
    that before writing anything.
  - Amounts are floats here: the graph is for exploration. Exact amounts live in Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import pandas as pd

from aml.data.fx import to_usd

SCHEMA = [
    "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
    "CREATE INDEX sent_txn_id IF NOT EXISTS FOR ()-[r:SENT]-() ON (r.txn_id)",
    "CREATE INDEX sent_ts IF NOT EXISTS FOR ()-[r:SENT]-() ON (r.ts)",
]

MERGE_ACCOUNTS = """
UNWIND $rows AS row
MERGE (a:Account {id: row.account_id})
SET a.customer_id = row.customer_id, a.bank_id = row.bank_id, a.segment = row.segment,
    a.occupation = row.occupation, a.country = row.country, a.risk_rating = row.risk_rating,
    a.expected_monthly_volume = row.expected_monthly_volume
"""

MERGE_TRANSACTIONS = """
UNWIND $rows AS row
MERGE (s:Account {id: row.from_account})
MERGE (r:Account {id: row.to_account})
MERGE (s)-[t:SENT {txn_id: row.txn_id}]->(r)
SET t.ts = localdatetime(row.ts), t.amount = row.amount_paid, t.currency = row.payment_currency,
    t.amount_usd = row.amount_usd, t.format = row.payment_format
"""

ACCOUNT_FIELDS = [
    "account_id",
    "customer_id",
    "bank_id",
    "segment",
    "occupation",
    "country",
    "risk_rating",
    "expected_monthly_volume",
]
TXN_FIELDS = [
    "txn_id",
    "ts",
    "from_account",
    "to_account",
    "amount_paid",
    "payment_currency",
    "amount_usd",
    "payment_format",
]


class Session(Protocol):
    def run(self, query: str, parameters: dict[str, Any] | None = None, **kw: Any) -> Any: ...


class CapacityError(RuntimeError):
    pass


def _batches(records: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(records), size):
        yield records[i : i + size]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict(orient="records")  # type: ignore[return-value]


def account_rows(accounts: pd.DataFrame) -> list[dict[str, Any]]:
    df = accounts[ACCOUNT_FIELDS].copy()
    df["expected_monthly_volume"] = df["expected_monthly_volume"].astype(float)
    return _records(df)


def transaction_rows(txns: pd.DataFrame) -> list[dict[str, Any]]:
    """Only the fields in TXN_FIELDS: the label column is dropped here, on purpose."""
    df = txns.assign(
        amount_usd=to_usd(txns["amount_paid"], txns["payment_currency"]),
        ts=pd.to_datetime(txns["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S"),
        amount_paid=txns["amount_paid"].astype(float),
    )[TXN_FIELDS]
    df["amount_usd"] = df["amount_usd"].replace({np.nan: None})
    return _records(df)


def check_capacity(
    session: Session, txns: pd.DataFrame, accounts: pd.DataFrame, max_nodes: int, max_rels: int
) -> None:
    """Refuse to load past the configured limits (AuraDB Free by default)."""
    # Two independent counts: one combined MATCH would return NO row when there are nodes but
    # no relationships yet, silently reporting an occupied graph as empty.
    existing = session.run(
        "CALL { MATCH (n) RETURN count(n) AS nodes } "
        "CALL { MATCH ()-[r]->() RETURN count(r) AS rels } "
        "RETURN nodes, rels"
    ).single()
    nodes = int(existing["nodes"]) if existing else 0
    rels = int(existing["rels"]) if existing else 0
    new_nodes = len(
        set(accounts["account_id"]) | set(txns["from_account"]) | set(txns["to_account"])
    )
    if max(nodes, new_nodes) > max_nodes or rels + len(txns) > max_rels:
        raise CapacityError(
            f"would exceed the graph limits ({max_nodes:,} nodes / {max_rels:,} relationships): "
            f"{nodes:,} nodes and {rels:,} relationships exist, loading {new_nodes:,} accounts and "
            f"{len(txns):,} transactions. Load a smaller window (--since/--until) or a smaller "
            "SIZE, or raise AML_NEO4J_MAX_NODES / AML_NEO4J_MAX_RELATIONSHIPS on a bigger instance."
        )


def load(
    session: Session, accounts: pd.DataFrame, txns: pd.DataFrame, batch_size: int = 5_000
) -> dict[str, int]:
    for statement in SCHEMA:
        session.run(statement)
    acc = account_rows(accounts)
    for chunk in _batches(acc, batch_size):
        session.run(MERGE_ACCOUNTS, {"rows": chunk})
    tx = transaction_rows(txns)
    for chunk in _batches(tx, batch_size):
        session.run(MERGE_TRANSACTIONS, {"rows": chunk})
    return {"accounts": len(acc), "transactions": len(tx)}
