"""Account state: rolling-window behaviour of every account, as of every moment it transacts.

One row per (account_id, event_timestamp) at which the account sent or received money. Each row
describes the account's behaviour in windows ending at, and including, that moment. Scoring a
transaction means reading the sender's (and receiver's) row at the transaction's own timestamp.

Point-in-time correct by construction: a row for time T only uses events at or before T, so
adding later data never changes an earlier row (tested).

Serving note: the online store keeps each account's LATEST row. Its windows are as of the
account's last event, so a stale account looks as active as it last was; scoring always happens
on a new transaction, which first updates the state. In production this state would be
maintained by a stream processor; the definitions are the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml.data.fx import to_usd
from aml.features.definitions import ACCOUNT_STATE_NAMES
from aml.features.windows import DAY, HOUR, Events, distinct_in_window, time_keys

EPOCH = pd.Timestamp("1970-01-01")
NEAR_THRESHOLD_USD = (9_000.0, 10_000.0)
PASSTHROUGH_CAP = 10.0


def _seconds(ts: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(ts) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def build_account_state(txns: pd.DataFrame) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["account_id", "event_timestamp", *ACCOUNT_STATE_NAMES])

    secs = _seconds(txns["ts"])
    # Unknown currencies contribute 0 USD; the quality report counts them.
    usd = to_usd(txns["amount_paid"], txns["payment_currency"]).fillna(0.0).to_numpy()
    codes, accounts = pd.factorize(pd.concat([txns["from_account"], txns["to_account"]]))
    n = len(txns)
    src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)

    is_cash = (txns["payment_format"] == "Cash").to_numpy()
    lo, hi = NEAR_THRESHOLD_USD
    near = is_cash & (usd >= lo) & (usd < hi)
    xccy = (txns["payment_currency"] != txns["receiving_currency"]).to_numpy()

    out_events = Events.build(
        time_keys(src, secs),
        {"amt": usd, "cash_amt": usd * is_cash, "near": near, "xccy": xccy},
    )
    in_events = Events.build(time_keys(dst, secs), {"amt": usd})

    # Evaluate at every moment each account sent or received anything.
    q = np.unique(np.concatenate([time_keys(src, secs), time_keys(dst, secs)]))

    o1h, s1h = out_events.window(q, HOUR)
    o24, s24 = out_events.window(q, DAY)
    o7d, s7d = out_events.window(q, 7 * DAY)
    o30, s30 = out_events.window(q, 30 * DAY)
    i24, si24 = in_events.window(q, DAY)
    _, si7 = in_events.window(q, 7 * DAY)

    out_amt_30d = s30["amt"]
    feats = {
        "out_cnt_1h": o1h,
        "out_amt_1h": s1h["amt"],
        "out_cnt_24h": o24,
        "out_amt_24h": s24["amt"],
        "out_cnt_7d": o7d,
        "out_amt_7d": s7d["amt"],
        "out_amt_30d": out_amt_30d,
        "in_cnt_24h": i24,
        "in_amt_24h": si24["amt"],
        "in_amt_7d": si7["amt"],
        "out_uniq_cpty_7d": distinct_in_window(src, dst, secs, q, 7 * DAY),
        "in_uniq_cpty_7d": distinct_in_window(dst, src, secs, q, 7 * DAY),
        "near_threshold_cash_24h": np.rint(s24["near"]).astype(np.int64),
        "cash_out_ratio_30d": _ratio(s30["cash_amt"], out_amt_30d),
        "xccy_out_share_30d": _ratio(s30["xccy"], o30.astype(np.float64)),
        "spike_ratio_24h": _ratio(s24["amt"], out_amt_30d / 30.0),
        "passthrough_ratio_24h": np.minimum(_ratio(s24["amt"], si24["amt"]), PASSTHROUGH_CAP),
    }

    df = pd.DataFrame(
        {
            "account_id": np.asarray(accounts)[q // np.int64(1 << 34)],
            "event_timestamp": EPOCH + pd.to_timedelta(q % np.int64(1 << 34), unit="s"),
        }
    )
    for name in ACCOUNT_STATE_NAMES:
        values = feats[name]
        df[name] = (
            values.astype(np.int32) if values.dtype.kind in "iu" else values.astype(np.float32)
        )
    return df
