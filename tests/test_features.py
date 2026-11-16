"""Feature correctness: brute-force agreement, and above all no leakage from the future."""

import numpy as np
import pandas as pd
import pytest

from aml.data.fx import to_usd
from aml.features.account_state import build_account_state
from aml.features.definitions import ACCOUNT_STATE_NAMES, GRAPH_DAILY_NAMES
from aml.features.graph import UNREACHABLE, build_graph_snapshots
from aml.features.pit import training_frame
from aml.features.windows import DAY, distinct_in_window, time_keys


@pytest.fixture(scope="module")
def state(txns: pd.DataFrame) -> pd.DataFrame:
    return build_account_state(txns)


@pytest.fixture(scope="module")
def graph(txns: pd.DataFrame) -> pd.DataFrame:
    return build_graph_snapshots(txns)


def test_distinct_in_window_by_hand() -> None:
    # account 0 pays counterparty 1 at t=10 and t=40, counterparty 2 at t=20; window = 15 s
    acct = np.array([0, 0, 0])
    cpty = np.array([1, 2, 1])
    t = np.array([10, 20, 40])
    q = time_keys(np.zeros(6, dtype=np.int64), np.array([10, 20, 24, 26, 40, 60]))
    assert distinct_in_window(acct, cpty, t, q, 15).tolist() == [1, 2, 2, 1, 1, 0]


def test_state_matches_brute_force(txns: pd.DataFrame, state: pd.DataFrame) -> None:
    tx = txns.assign(usd=to_usd(txns["amount_paid"], txns["payment_currency"]))
    sample = state.sample(300, random_state=0)
    for r in sample.itertuples(index=False):
        t = r.event_timestamp
        out = tx[(tx["from_account"] == r.account_id) & (tx["ts"] <= t)]
        inn = tx[(tx["to_account"] == r.account_id) & (tx["ts"] <= t)]
        o24 = out[out["ts"] > t - pd.Timedelta(days=1)]
        o7 = out[out["ts"] > t - pd.Timedelta(days=7)]
        i7 = inn[inn["ts"] > t - pd.Timedelta(days=7)]
        assert r.out_cnt_24h == len(o24)
        assert r.out_amt_24h == pytest.approx(o24["usd"].sum(), rel=1e-4, abs=0.05)
        assert r.out_uniq_cpty_7d == o7["to_account"].nunique()
        assert r.in_uniq_cpty_7d == i7["from_account"].nunique()


def test_no_future_leakage(txns: pd.DataFrame, state: pd.DataFrame, graph: pd.DataFrame) -> None:
    """THE leakage test: features up to time T must not change when later data is added."""
    cutoff = txns["ts"].quantile(0.6)
    past = txns[txns["ts"] <= cutoff]

    full_state = state[state["event_timestamp"] <= cutoff].reset_index(drop=True)
    past_state = build_account_state(past)
    pd.testing.assert_frame_equal(
        full_state.sort_values(["account_id", "event_timestamp"]).reset_index(drop=True),
        past_state.sort_values(["account_id", "event_timestamp"]).reset_index(drop=True),
    )

    day = cutoff.floor("D")
    full_graph = graph[graph["snapshot_ts"] <= day]
    past_graph = build_graph_snapshots(past)
    keys = ["account_id", "snapshot_ts"]
    pd.testing.assert_frame_equal(
        full_graph.sort_values(keys).reset_index(drop=True),
        past_graph.sort_values(keys).reset_index(drop=True),
    )

    full_tf = training_frame(txns, state, graph)
    past_tf = training_frame(past, past_state, past_graph)
    pd.testing.assert_frame_equal(
        full_tf[full_tf["ts"] <= cutoff].reset_index(drop=True),
        past_tf.reset_index(drop=True),
    )


def test_graph_snapshot_excludes_its_own_day(txns: pd.DataFrame, graph: pd.DataFrame) -> None:
    # Move one transaction to 00:00 of a day: it must not appear in that day's snapshot.
    day = txns["ts"].iloc[len(txns) // 2].floor("D")
    edge = txns.iloc[[0]].assign(
        ts=day, from_account="NEW-SENDER", to_account="NEW-RECEIVER", txn_id="x" * 32
    )
    g = build_graph_snapshots(pd.concat([txns, edge], ignore_index=True))
    assert "NEW-SENDER" not in set(g.loc[g["snapshot_ts"] == day, "account_id"])
    assert "NEW-SENDER" in set(g.loc[g["snapshot_ts"] == day + pd.Timedelta(days=1), "account_id"])


def test_cycle_accounts_have_three_loops(txns: pd.DataFrame, graph: pd.DataFrame) -> None:
    cycles = txns[txns["is_laundering"] & (txns["payment_format"] == "ACH")]
    assert len(cycles) > 0
    first = cycles.iloc[0]
    next_day = first["ts"].floor("D") + pd.Timedelta(days=1)
    snap = graph[
        (graph["snapshot_ts"] == next_day) & (graph["account_id"] == first["from_account"])
    ]
    assert int(snap["g_cycles3_30d"].iloc[0]) >= 1


def test_known_bad_respects_reporting_lag(txns: pd.DataFrame) -> None:
    bad = txns[txns["is_laundering"]]
    acct, when = bad["from_account"].iloc[0], bad["ts"].iloc[0]
    g = build_graph_snapshots(txns, label_lag_days=3)
    rows = g[g["account_id"] == acct].set_index("snapshot_ts")["g_is_known_bad"]
    before_lag = rows[rows.index <= when + pd.Timedelta(days=3)]
    assert (before_lag == 0).all()  # not "known" until the lag has passed
    later = rows[rows.index > when.floor("D") + pd.Timedelta(days=4)]
    if len(later):
        assert later.iloc[0] == 1


def test_training_frame_shape(txns: pd.DataFrame, state: pd.DataFrame, graph: pd.DataFrame) -> None:
    tf = training_frame(txns, state, graph)
    assert len(tf) == len(txns)
    assert tf["txn_id"].tolist() == txns["txn_id"].tolist()  # same order
    for prefix in ["s_", "r_"]:
        assert all(f"{prefix}{n}" in tf for n in ACCOUNT_STATE_NAMES + GRAPH_DAILY_NAMES)
    # Sender's own payment is always inside its 1-hour window (state includes time T).
    assert (tf["s_out_cnt_1h"] >= 1).all()
    assert tf["s_g_hops_to_known_bad"].between(0, UNREACHABLE).all()
    assert tf.drop(columns=["payment_format"]).notna().all().all()


def test_empty_input_gives_empty_features() -> None:
    empty = pd.DataFrame(
        columns=[
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
    )
    assert build_account_state(empty).empty
    assert build_graph_snapshots(empty).empty
    assert DAY == 86_400
