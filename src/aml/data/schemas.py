"""Data contracts. Every record entering the system is validated against these models.

In Step 2 the stream consumer uses `Transaction` to route bad records to a dead-letter topic.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PAYMENT_FORMATS = frozenset(
    {"ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"}
)


class Transaction(BaseModel):
    """One money movement between two accounts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    txn_id: str = Field(min_length=16, max_length=64)
    ts: datetime
    from_account: str = Field(min_length=1)
    to_account: str = Field(min_length=1)
    amount_paid: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    payment_currency: str = Field(min_length=1)
    amount_received: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    receiving_currency: str = Field(min_length=1)
    payment_format: str
    is_laundering: bool

    @field_validator("payment_format")
    @classmethod
    def _known_format(cls, v: str) -> str:
        if v not in PAYMENT_FORMATS:
            raise ValueError(f"unknown payment_format {v!r}")
        return v

    @model_validator(mode="after")
    def _same_currency_same_amount(self) -> Transaction:
        if (
            self.payment_currency == self.receiving_currency
            and self.amount_paid != self.amount_received
        ):
            raise ValueError("same-currency transfer must have equal paid and received amounts")
        return self


class Customer(BaseModel):
    """Synthetic KYC profile. The IBM dataset has accounts but no people, so we generate these."""

    model_config = ConfigDict(frozen=True)

    customer_id: str
    segment: Literal["individual", "business"]
    occupation: str
    country: str = Field(min_length=2, max_length=2)
    risk_rating: Literal["low", "medium", "high"]
    expected_monthly_volume: Decimal = Field(ge=0)
    onboarded_at: date


TS_KEY_FORMAT = "%Y-%m-%dT%H:%M:%S"


def format_ts(ts: datetime | None) -> str:
    """Timestamp as it appears in an ID key. Missing timestamps (None/NaT) become 'NaT'."""
    if ts is None or ts != ts:  # NaT is the only value not equal to itself
        return "NaT"
    return ts.strftime(TS_KEY_FORMAT)


def format_amount(amount: Decimal | float | str) -> str:
    """Amount as it appears in an ID key: always exactly two decimals ('100.00', never '100.0')."""
    try:
        d = Decimal(str(amount))
    except InvalidOperation:
        return "NaN"
    return f"{d:.2f}" if d.is_finite() else "NaN"


def make_txn_id(
    ts: datetime | None,
    from_account: object,
    to_account: object,
    amount_paid: Decimal | float | str,
    seq: int,
) -> str:
    """Deterministic ID: loading the same row twice yields the same ID, so loads are idempotent.

    This is the ONLY place IDs are built; ibm.normalize calls it row by row, so the batch and
    streaming paths can never disagree. `seq` disambiguates genuinely identical rows (same time,
    parties and amount). Malformed fields still get a stable ID, so bad rows can be quarantined
    and de-duplicated too.
    """
    key = "|".join(
        [format_ts(ts), str(from_account), str(to_account), format_amount(amount_paid), str(seq)]
    )
    return hashlib.sha256(key.encode()).hexdigest()[:32]
