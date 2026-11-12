import numpy as np
import pandas as pd
import pytest

from aml.data import fx, ibm, synthetic
from aml.data.profiles import build_profiles
from aml.data.schemas import make_txn_id
from aml.data.validate import validate_customers, validate_transactions


@pytest.fixture(scope="module")
def normalized() -> pd.DataFrame:
    return ibm.normalize(synthetic.generate(n_accounts=200, n_normal=2_000, seed=7))


def test_normalize_shape_and_order(normalized: pd.DataFrame) -> None:
    assert list(normalized.columns) == ibm.NORMALIZED_COLUMNS
    assert normalized["ts"].is_monotonic_increasing
    assert normalized["from_account"].str.contains("-").all()  # bank-prefixed


def test_txn_ids_unique_stable_and_from_one_function(normalized: pd.DataFrame) -> None:
    assert normalized["txn_id"].is_unique
    again = ibm.normalize(synthetic.generate(n_accounts=200, n_normal=2_000, seed=7))
    assert (again["txn_id"] == normalized["txn_id"]).all()
    # The batch path and the single-record function produce the same ID (flag #1).
    row = normalized.iloc[0]
    assert row["txn_id"] == make_txn_id(
        row["ts"], row["from_account"], row["to_account"], row["amount_paid"], 0
    )


def test_patterns_injected(normalized: pd.DataFrame) -> None:
    rate = normalized["is_laundering"].astype(bool).mean()
    assert 0 < rate < 0.1
    bad = normalized[normalized["is_laundering"].astype(bool)]
    structuring = bad[bad["payment_format"] == "Cash"]
    assert structuring["amount_paid"].between(9_000, 10_000).all()


def test_fan_out_mules_are_distinct_and_never_the_sender() -> None:
    raw = synthetic.generate(n_accounts=50, n_normal=0, n_patterns=30, seed=3)
    txns = ibm.normalize(raw)
    wires = txns[txns["payment_format"] == "Wire"]
    assert (wires["from_account"] != wires["to_account"]).all()
    for _, burst in wires.groupby("from_account"):
        # each burst pays distinct mules (several bursts per sender may repeat a mule)
        assert burst["to_account"].nunique() >= min(5, len(burst))


RAW_COLS = [
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


def test_bad_rows_are_quarantined_not_dropped() -> None:
    """Flag #3: normalize maps shape only; every bad row ends up rejected WITH a reason."""
    good = [
        "2022/09/01 10:00",
        "010",
        "A1",
        "020",
        "B1",
        100,
        "US Dollar",
        100,
        "US Dollar",
        "Wire",
        0,
    ]
    raw = pd.DataFrame(
        [
            good,
            ["not a date", *good[1:]],
            [*good[:5], 0, "US Dollar", 0, "US Dollar", "Wire", 0],  # zero amount
            [*good[:5], "abc", "US Dollar", "abc", "US Dollar", "Wire", 0],  # not a number
            [*good[:9], "Carrier pigeon", 0],
            [*good[:10], 7],  # label is not 0/1
        ],
        columns=RAW_COLS,
    ).astype({"from_bank": str, "from_acct": str, "to_bank": str, "to_acct": str})
    normalized = ibm.normalize(raw)
    assert len(normalized) == len(raw)  # nothing dropped
    valid, rejected = validate_transactions(normalized)
    assert len(valid) == 1 and len(rejected) == 5
    assert rejected["error"].str.len().gt(0).all()
    joined = " ".join(rejected["error"])
    for field in ["ts", "amount_paid", "payment_format", "is_laundering"]:
        assert field in joined


def test_validation_quarantines_bad_rows(normalized: pd.DataFrame) -> None:
    broken = normalized.copy()
    broken.loc[0, "payment_format"] = "Carrier pigeon"
    valid, rejected = validate_transactions(broken)
    assert len(rejected) == 1
    assert "payment_format" in rejected.loc[0, "error"]
    assert len(valid) == len(normalized) - 1
    assert valid["is_laundering"].dtype == bool


def test_profiles_cover_every_account_and_pass_contract(normalized: pd.DataFrame) -> None:
    customers, accounts = build_profiles(normalized)
    used = set(normalized["from_account"]) | set(normalized["to_account"])
    assert set(accounts["account_id"]) == used
    assert customers["customer_id"].is_unique
    _, bad = validate_customers(customers)
    assert bad.empty  # flag #6: generated profiles satisfy the Customer contract


def test_kyc_under_declaration_is_a_signal_not_a_giveaway() -> None:
    """Flag #5: some laundering accounts declare honestly and some honest ones under-declare."""
    txns = ibm.normalize(synthetic.generate(n_accounts=400, n_normal=8_000, n_patterns=60, seed=1))
    customers, accounts = build_profiles(txns)
    usd = fx.to_usd(txns["amount_paid"], txns["payment_currency"])
    sent = txns.assign(usd=usd).groupby("from_account")["usd"].sum()
    days = max((txns["ts"].max() - txns["ts"].min()).days, 1)
    df = accounts.merge(customers, on="customer_id").set_index("account_id")
    df["actual"] = sent.reindex(df.index).fillna(0) / days * 30
    df = df[df["actual"] > 2_000]  # the $500 floor hides small accounts
    under = df["expected_monthly_volume"] < 0.55 * df["actual"]
    flagged = txns[txns["is_laundering"].astype(bool)]
    bad = df.index.isin(set(flagged["from_account"]) | set(flagged["to_account"]))
    assert under[bad].mean() > under[~bad].mean()  # still a real red flag...
    assert under[bad].mean() < 0.9  # ...but not every launderer under-declares
    assert under[~bad].sum() > 0  # ...and some honest customers do


def test_customer_contract_rejects_bad_profiles(normalized: pd.DataFrame) -> None:
    customers, _ = build_profiles(normalized)
    broken = customers.copy()
    broken.loc[0, "segment"] = "corporate"
    broken.loc[1, "expected_monthly_volume"] = -5
    _, bad = validate_customers(broken)
    assert len(bad) == 2


def test_customer_ids_stable_per_account(normalized: pd.DataFrame) -> None:
    _, a1 = build_profiles(normalized, seed=1)
    _, a2 = build_profiles(normalized.sample(frac=0.5, random_state=0), seed=2)
    merged = a1.merge(a2, on="account_id", suffixes=("_1", "_2"))
    assert (merged["customer_id_1"] == merged["customer_id_2"]).all()


def test_fx_converts_and_flags_unknown_currencies() -> None:
    amounts = pd.Series([100.0, 1_000.0, 5.0])
    usd = fx.to_usd(amounts, pd.Series(["US Dollar", "Yen", "Doubloon"]))
    assert usd.iloc[0] == 100.0
    assert usd.iloc[1] == pytest.approx(7.0)
    assert np.isnan(usd.iloc[2])
    assert fx.unknown_currencies(pd.Series(["Euro", "Doubloon"])) == ["Doubloon"]


def test_take_window_respects_size(normalized: pd.DataFrame) -> None:
    assert len(ibm.take_window(normalized, "full")) == len(normalized)
    big = pd.concat([normalized] * 60, ignore_index=True)  # > 100k rows
    small = ibm.take_window(big, "small")
    assert len(small) == ibm.SIZES["small"]
    assert small.equals(big.head(len(small)))  # the earliest rows, in order
