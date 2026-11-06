"""Wire format for transaction messages: JSON, with money as STRINGS.

JSON numbers are binary floats in most readers: 9999.99 can arrive as 9999.990000000002.
Sending "9999.99" as a string keeps cents exact end to end; the consumer turns it into a
Decimal through the data contract. Timestamps are ISO 8601 strings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from aml.data.schemas import format_amount


class DecodeError(ValueError):
    """The message is not a JSON object (corrupt, truncated, or not ours)."""


def _ts(value: Any) -> str | None:
    if value is None or value != value:  # None or NaT
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _amount(value: Any) -> str | None:
    text = format_amount(value)
    return None if text == "NaN" else text


def encode(row: Mapping[str, Any]) -> bytes:
    """A normalized transaction row -> message bytes. Never validates: that's the consumer's job."""
    payload = {
        "txn_id": row.get("txn_id"),
        "ts": _ts(row.get("ts")),
        "from_account": row.get("from_account"),
        "to_account": row.get("to_account"),
        "amount_paid": _amount(row.get("amount_paid")),
        "payment_currency": row.get("payment_currency"),
        "amount_received": _amount(row.get("amount_received")),
        "receiving_currency": row.get("receiving_currency"),
        "payment_format": row.get("payment_format"),
        "is_laundering": row.get("is_laundering"),
    }
    payload = {k: (None if isinstance(v, float) and v != v else v) for k, v in payload.items()}
    return json.dumps(payload, separators=(",", ":"), default=str).encode()


def decode(value: bytes | None) -> dict[str, Any]:
    if value is None:
        raise DecodeError("empty message")
    try:
        obj = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DecodeError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise DecodeError(f"expected a JSON object, got {type(obj).__name__}")
    return obj
