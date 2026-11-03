"""Point-in-time joins: attach to each transaction the features that existed at its moment.

This is how Step 3 builds training data without leakage. Feast's get_historical_features does
the same as-of join; test_feast.py checks the two agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml.data.fx import to_usd
from aml.features.definitions import ACCOUNT_STATE_NAMES, GRAPH_DAILY_NAMES
from aml.features.graph import UNREACHABLE

STATE_TTL = pd.Timedelta(days=30)
RISK_ORDINAL = {"low": 0, "medium": 1, "high": 2}


def as_of_join(
    events: pd.DataFrame,
    entity_col: str,
    ts_col: str,
    features: pd.DataFrame,
    feature_ts_col: str,
    names: list[str],
    prefix: str,
    tolerance: pd.Timedelta | None = STATE_TTL,
) -> pd.DataFrame:
    """For each event row: the latest feature row of its entity with timestamp <= the event's.

    Returns `names` prefixed with `prefix`, aligned to `events.index`. No match within
    `tolerance` gives NaN (Feast calls this the TTL).
    """
    left = pd.DataFrame(
        {
            "_row": np.arange(len(events)),
            "account_id": events[entity_col].to_numpy(),
            "_ts": pd.to_datetime(events[ts_col]).to_numpy(),
        }
    ).sort_values("_ts", kind="stable")
    right = features[["account_id", feature_ts_col, *names]].rename(columns={feature_ts_col: "_ts"})
    right = right.assign(_ts=pd.to_datetime(right["_ts"])).sort_values("_ts", kind="stable")
    joined = pd.merge_asof(
        left,
        right,
        on="_ts",
        by="account_id",
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    ).sort_values("_row")
    out = joined[names].rename(columns={n: f"{prefix}{n}" for n in names})
    out.index = events.index
    return out


def _graph_for(
    txns: pd.DataFrame, graph: pd.DataFrame, entity_col: str, prefix: str
) -> pd.DataFrame:
    # Exact match on the transaction's day: snapshot D covers [D-30d, D). An account absent from
    # today's snapshot had no payments in the window, so its features are 0, not yesterday's.
    keys = pd.DataFrame(
        {
            "account_id": txns[entity_col].to_numpy(),
            "snapshot_ts": pd.to_datetime(txns["ts"]).dt.floor("D").to_numpy(),
        },
        index=txns.index,
    )
    merged = keys.merge(graph, on=["account_id", "snapshot_ts"], how="left")
    merged.index = txns.index
    out = merged[GRAPH_DAILY_NAMES].copy()
    out["g_hops_to_known_bad"] = out["g_hops_to_known_bad"].fillna(UNREACHABLE)
    out = out.fillna(0).astype(np.int32)
    return out.rename(columns={n: f"{prefix}{n}" for n in GRAPH_DAILY_NAMES})


def training_frame(
    txns: pd.DataFrame,
    state: pd.DataFrame,
    graph: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per transaction: its own attributes, sender (s_) and receiver (r_) features at
    the transaction's timestamp, and the label. Input to Step 3."""
    base = pd.DataFrame(
        {
            "txn_id": txns["txn_id"],
            "ts": pd.to_datetime(txns["ts"]),
            "from_account": txns["from_account"],
            "to_account": txns["to_account"],
            "amount_usd": to_usd(txns["amount_paid"], txns["payment_currency"]).astype(np.float32),
            "payment_format": txns["payment_format"],
            "is_cash": (txns["payment_format"] == "Cash").astype(np.int8),
            "is_xccy": (txns["payment_currency"] != txns["receiving_currency"]).astype(np.int8),
            "is_laundering": txns["is_laundering"].astype(bool),
        },
        index=txns.index,
    )
    parts = [
        base,
        as_of_join(txns, "from_account", "ts", state, "event_timestamp", ACCOUNT_STATE_NAMES, "s_"),
        as_of_join(txns, "to_account", "ts", state, "event_timestamp", ACCOUNT_STATE_NAMES, "r_"),
        _graph_for(txns, graph, "from_account", "s_"),
        _graph_for(txns, graph, "to_account", "r_"),
    ]
    frame = pd.concat(parts, axis=1)

    if accounts is not None:  # KYC context for the sender
        kyc = accounts.set_index("account_id")
        sender = kyc.reindex(txns["from_account"].to_numpy())
        declared = sender["expected_monthly_volume"].astype(float).to_numpy()
        frame["s_kyc_actual_vs_declared"] = np.divide(
            frame["s_out_amt_30d"].to_numpy(dtype=np.float64),
            declared,
            out=np.zeros(len(frame)),
            where=declared > 0,
        ).astype(np.float32)
        onboarded = pd.to_datetime(sender["onboarded_at"]).to_numpy()
        frame["s_account_age_days"] = (
            (frame["ts"].to_numpy() - onboarded) / np.timedelta64(1, "D")
        ).astype(np.float32)
        frame["s_risk_rating"] = (
            sender["risk_rating"].map(RISK_ORDINAL).fillna(-1).astype(np.int8).to_numpy()
        )
        frame["s_is_business"] = (sender["segment"] == "business").astype(np.int8).to_numpy()
    return frame
