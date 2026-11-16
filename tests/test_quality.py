from datetime import datetime

import pandas as pd

from aml.data.profiles import build_profiles
from aml.data.quality import check_transactions


def failed(report) -> set[str]:  # type: ignore[no-untyped-def]
    return {c.name for c in report.failures}


def test_clean_data_passes(txns: pd.DataFrame) -> None:
    _, accounts = build_profiles(txns)
    report = check_transactions(txns, accounts=accounts)
    assert report.ok and not report.failures
    assert "PASSED" in report.render()


def test_critical_problems_fail_the_report(txns: pd.DataFrame) -> None:
    _, accounts = build_profiles(txns)
    broken = pd.concat([txns, txns.head(3)], ignore_index=True)  # duplicates
    broken.loc[5, "to_account"] = None  # null
    broken.loc[6, "ts"] = datetime(2099, 1, 1)  # future
    report = check_transactions(broken, accounts=accounts.iloc[1:])  # one profile missing
    assert not report.ok
    assert {
        "duplicate_txn_ids",
        "required_nulls",
        "future_timestamps",
        "accounts_without_profile",
    } <= failed(report)


def test_warnings_do_not_fail_the_report(txns: pd.DataFrame) -> None:
    odd = txns.copy()
    odd.loc[odd.index[:5], "payment_currency"] = "Doubloon"
    report = check_transactions(odd, rejected_count=len(txns))  # 50% rejected
    assert report.ok
    assert {"unknown_currencies", "rejected_rate"} <= failed(report)


def test_empty_dataset_is_critical() -> None:
    report = check_transactions(pd.DataFrame(columns=["txn_id"]))
    assert not report.ok
