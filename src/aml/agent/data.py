"""Read-only data access for the agent's tools: Parquet files (dev, tests) or Postgres (prod).

The agent never gets a database connection or a writable handle, only these read methods.
That is how "the agent has read-only tools" is enforced in code, not just in a prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from aml.data.fx import to_usd


class CaseData(Protocol):
    def alert(self, txn_id: str) -> dict[str, Any] | None: ...
    def account(self, account_id: str) -> dict[str, Any] | None: ...
    def transactions(
        self, account_id: str, since: datetime, until: datetime, limit: int = 200
    ) -> list[dict[str, Any]]: ...
    def watchlist(self, account_ids: list[str]) -> list[dict[str, Any]]: ...


def _clean(rec: Mapping[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, pd.Timestamp | datetime):
            v = v.isoformat(sep=" ")
        elif hasattr(v, "item"):
            v = v.item()
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        out[str(k)] = None if isinstance(v, float) and v != v else v
    return out


class FrameData:
    """In-memory implementation over DataFrames (processed Parquet + scored alerts)."""

    def __init__(
        self,
        txns: pd.DataFrame,
        accounts: pd.DataFrame,
        alerts: pd.DataFrame,
        watchlist: pd.DataFrame,
    ) -> None:
        t = txns.copy()
        t["ts"] = pd.to_datetime(t["ts"])
        t["amount_usd"] = to_usd(t["amount_paid"], t["payment_currency"]).round(2)
        self.txns = t.sort_values("ts", kind="stable")
        self._by_from = {a: g for a, g in self.txns.groupby("from_account")}
        self._by_to = {a: g for a, g in self.txns.groupby("to_account")}
        self.accounts = accounts.set_index("account_id")
        self.alerts = alerts.set_index("txn_id")
        self.wl = watchlist

    @classmethod
    def from_dirs(cls, processed: Path, features: Path) -> FrameData:
        acc = pd.read_parquet(processed / "accounts.parquet").merge(
            pd.read_parquet(processed / "customers.parquet"), on="customer_id"
        )
        wl_path = processed / "watchlist.parquet"
        wl = (
            pd.read_parquet(wl_path)
            if wl_path.exists()
            else pd.DataFrame(columns=["entry_id", "account_id", "list", "reason", "listed_at"])
        )
        return cls(
            pd.read_parquet(processed / "transactions.parquet"),
            acc,
            pd.read_parquet(features / "alerts.parquet"),
            wl,
        )

    def alert(self, txn_id: str) -> dict[str, Any] | None:
        if txn_id not in self.alerts.index:
            return None
        rec = {"txn_id": txn_id, **_clean(self.alerts.loc[txn_id].to_dict())}
        if isinstance(rec.get("reasons"), str):
            rec["reasons"] = json.loads(rec["reasons"]) if rec["reasons"] else []
        return rec

    def account(self, account_id: str) -> dict[str, Any] | None:
        if account_id not in self.accounts.index:
            return None
        return {"account_id": account_id, **_clean(self.accounts.loc[account_id].to_dict())}

    def transactions(
        self, account_id: str, since: datetime, until: datetime, limit: int = 200
    ) -> list[dict[str, Any]]:
        empty = self.txns.iloc[:0]
        both = pd.concat([self._by_from.get(account_id, empty), self._by_to.get(account_id, empty)])
        both = both[(both["ts"] >= since) & (both["ts"] < until)].drop_duplicates("txn_id")
        both = both.sort_values("ts", kind="stable").head(limit)
        cols = [
            "txn_id",
            "ts",
            "from_account",
            "to_account",
            "amount_paid",
            "payment_currency",
            "amount_usd",
            "payment_format",
        ]
        return [_clean(r) for r in both[cols].to_dict(orient="records")]

    def watchlist(self, account_ids: list[str]) -> list[dict[str, Any]]:
        hits = self.wl[self.wl["account_id"].isin(account_ids)]
        return [_clean(r) for r in hits.to_dict(orient="records")]


class PostgresData:
    """Same interface over Postgres. Connects as the configured (read) role."""

    def __init__(self, dsn: str) -> None:
        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)

    def _rows(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [_clean(dict(r)) for r in self.conn.execute(query.encode(), params).fetchall()]

    def alert(self, txn_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM cases.alerts WHERE txn_id = %s", (txn_id,))
        return rows[0] if rows else None

    def account(self, account_id: str) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT a.account_id, a.customer_id, a.bank_id, c.segment, c.occupation, c.country, "
            "c.risk_rating, c.expected_monthly_volume::float8 AS expected_monthly_volume, "
            "c.onboarded_at FROM ref.accounts a JOIN ref.customers c USING (customer_id) "
            "WHERE a.account_id = %s",
            (account_id,),
        )
        return rows[0] if rows else None

    def transactions(
        self, account_id: str, since: datetime, until: datetime, limit: int = 200
    ) -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT txn_id, ts, from_account, to_account, amount_paid::float8 AS amount_paid, "
            "payment_currency, payment_format FROM raw.transactions "
            "WHERE (from_account = %s OR to_account = %s) AND ts >= %s AND ts < %s "
            "ORDER BY ts, txn_id LIMIT %s",
            (account_id, account_id, since, until, limit),
        )
        if rows:
            df = pd.DataFrame(rows)
            usd = to_usd(df["amount_paid"], df["payment_currency"]).round(2)
            for r, u in zip(rows, usd, strict=True):
                r["amount_usd"] = None if u != u else float(u)
        return rows

    def watchlist(self, account_ids: list[str]) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        return self._rows(
            "SELECT entry_id, account_id, list, reason, listed_at FROM ref.watchlist "
            "WHERE account_id = ANY(%s)",
            (account_ids,),
        )
