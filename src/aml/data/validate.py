"""Validate DataFrames against the data contracts.

Each function returns (valid, rejected). Rejected rows keep an `error` column explaining why, so
bad data is quarantined and inspectable instead of silently dropped. The stream consumer uses the
same contract (and `format_errors`) record by record before routing failures to the dead-letter
topic.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from aml.data.schemas import Customer, Transaction


def format_errors(e: ValidationError) -> str:
    """One readable line per failed field: 'amount_paid: Input should be greater than 0; ...'."""
    return "; ".join(f"{'.'.join(map(str, x['loc'])) or 'record'}: {x['msg']}" for x in e.errors())


def _split(
    df: pd.DataFrame, model: type[BaseModel], prepare: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors: list[str | None] = []
    for rec in df.to_dict(orient="records"):
        try:
            model.model_validate(prepare(rec))
            errors.append(None)
        except ValidationError as e:
            errors.append(format_errors(e))
    err = pd.Series(errors, index=df.index, dtype="object")
    ok = err.isna()
    rejected = df[~ok].assign(error=err[~ok])
    return df[ok].reset_index(drop=True), rejected.reset_index(drop=True)


def _decimal(x: Any) -> Any:
    # Via str(): Decimal(100.1) would capture the float's binary error, Decimal("100.1") is exact.
    return Decimal(str(x)) if isinstance(x, float | int) else x


def _prepare_txn(rec: dict[str, Any]) -> dict[str, Any]:
    ts = rec["ts"]
    return {
        **rec,
        "ts": None if pd.isna(ts) else ts.to_pydatetime(),
        "amount_paid": _decimal(rec["amount_paid"]),
        "amount_received": _decimal(rec["amount_received"]),
        "is_laundering": None if pd.isna(rec["is_laundering"]) else rec["is_laundering"],
    }


def validate_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid, rejected = _split(df, Transaction, _prepare_txn)
    # normalize() can leave object dtypes behind (NaN-able columns); valid rows are clean.
    valid["is_laundering"] = valid["is_laundering"].astype(bool)
    return valid, rejected


def _prepare_customer(rec: dict[str, Any]) -> dict[str, Any]:
    return {**rec, "expected_monthly_volume": _decimal(rec["expected_monthly_volume"])}


def validate_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _split(df, Customer, _prepare_customer)
