"""Read IBM's synthetic AML transactions and normalize them into our transaction contract.

Source: Kaggle dataset ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Raw columns (note the duplicated "Account" header):
    Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency,
    Amount Paid, Payment Currency, Payment Format, Is Laundering
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from aml.data.schemas import make_txn_id

RAW_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

NORMALIZED_COLUMNS = [
    "txn_id",
    "ts",
    "from_account",
    "to_account",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
    "is_laundering",
]

Size = Literal["small", "medium", "full"]

# Row counts per sample size. We take a contiguous time window (not a random sample)
# so account histories and laundering chains stay intact.
SIZES: dict[Size, int | None] = {"small": 100_000, "medium": 1_000_000, "full": None}


def read_raw(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=0,
        names=RAW_COLUMNS,
        nrows=nrows,
        dtype={"from_bank": str, "from_acct": str, "to_bank": str, "to_acct": str},
    )


def _hash_ids(df: pd.DataFrame) -> pd.Series:
    """Deterministic txn IDs, built row by row with the one shared schemas.make_txn_id."""
    seq = df.groupby(
        ["ts", "from_account", "to_account", "amount_paid"], sort=False, dropna=False
    ).cumcount()
    ids = [
        make_txn_id(ts, frm, to, amt, int(n))
        for ts, frm, to, amt, n in zip(
            df["ts"], df["from_account"], df["to_account"], df["amount_paid"], seq, strict=True
        )
    ]
    return pd.Series(ids, index=df.index, dtype="string")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw IBM rows -> our contract shape. Pure function: no I/O, easy to test.

    This maps shape only; it never decides what is valid. Unparseable values become NaN/NaT and
    are rejected, with a reason, by validate.validate_transactions. Nothing is silently dropped.
    """
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(raw["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce"),
            # Account hex IDs are only unique within a bank, so prefix with the bank ID.
            "from_account": raw["from_bank"].str.strip() + "-" + raw["from_acct"].str.strip(),
            "to_account": raw["to_bank"].str.strip() + "-" + raw["to_acct"].str.strip(),
            "amount_paid": pd.to_numeric(raw["amount_paid"], errors="coerce").round(2),
            "payment_currency": raw["payment_currency"].str.strip(),
            "amount_received": pd.to_numeric(raw["amount_received"], errors="coerce").round(2),
            "receiving_currency": raw["receiving_currency"].str.strip(),
            "payment_format": raw["payment_format"].str.strip(),
            "is_laundering": pd.to_numeric(raw["is_laundering"], errors="coerce").map(
                {0: False, 1: True}
            ),
        }
    )
    df["txn_id"] = _hash_ids(df)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)[NORMALIZED_COLUMNS]


def take_window(df: pd.DataFrame, size: Size) -> pd.DataFrame:
    """First N rows in time order."""
    n = SIZES[size]
    return df if n is None else df.head(n).reset_index(drop=True)
