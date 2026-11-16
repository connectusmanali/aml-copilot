from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from aml.data.schemas import Transaction, format_amount, make_txn_id


def _txn(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "txn_id": "a" * 32,
        "ts": datetime(2022, 9, 1, 10, 0),
        "from_account": "010-ABC",
        "to_account": "020-DEF",
        "amount_paid": Decimal("100.00"),
        "payment_currency": "US Dollar",
        "amount_received": Decimal("100.00"),
        "receiving_currency": "US Dollar",
        "payment_format": "Wire",
        "is_laundering": False,
    }
    return base | over


def test_valid_transaction() -> None:
    assert Transaction.model_validate(_txn()).amount_paid == Decimal("100.00")


@pytest.mark.parametrize(
    "bad",
    [
        {"amount_paid": Decimal("0"), "amount_received": Decimal("0")},
        {"payment_format": "Carrier pigeon"},
        {"from_account": ""},
        {"amount_received": Decimal("99.00")},  # same currency, different amounts
    ],
)
def test_contract_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Transaction.model_validate(_txn(**bad))


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (100, "100.00"),
        (100.0, "100.00"),
        (100.1, "100.10"),
        (Decimal("9.5"), "9.50"),
        ("12.30", "12.30"),
        (float("nan"), "NaN"),
        ("abc", "NaN"),
    ],
)
def test_format_amount_is_always_two_decimals(value: object, text: str) -> None:
    assert format_amount(value) == text  # type: ignore[arg-type]


def test_txn_id_ignores_amount_representation() -> None:
    ts = datetime(2022, 9, 1)
    assert make_txn_id(ts, "a", "b", 100.0, 0) == make_txn_id(ts, "a", "b", Decimal("100.00"), 0)


def test_txn_id_is_deterministic() -> None:
    ts = datetime(2022, 9, 1)
    a = make_txn_id(ts, "x", "y", Decimal("5.00"), 0)
    assert a == make_txn_id(ts, "x", "y", Decimal("5.00"), 0)
    assert a != make_txn_id(ts, "x", "y", Decimal("5.00"), 1)
