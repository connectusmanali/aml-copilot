"""Feast serves exactly what our own as-of join computes: no training/serving skew."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

feast_store = pytest.importorskip("aml.features.feast_store", reason="needs --extra feast")

from aml.features.account_state import build_account_state  # noqa: E402
from aml.features.definitions import ACCOUNT_STATE_NAMES, GRAPH_DAILY_NAMES  # noqa: E402
from aml.features.graph import build_graph_snapshots  # noqa: E402
from aml.features.pipeline import GRAPH_FILE, STATE_FILE  # noqa: E402
from aml.features.pit import as_of_join  # noqa: E402


@pytest.fixture(scope="module")
def store(txns: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    features = tmp_path_factory.mktemp("features")
    state = build_account_state(txns)
    graph = build_graph_snapshots(txns)
    state.to_parquet(features / STATE_FILE, index=False)
    graph.to_parquet(features / GRAPH_FILE, index=False)
    repo: Path = tmp_path_factory.mktemp("feast")
    fs = feast_store.apply_and_materialize(repo, features)
    return fs, state, graph


def test_historical_keeps_every_row_and_matches_our_join(store: tuple, txns: pd.DataFrame) -> None:
    fs, state, graph = store
    events = (
        txns.sample(200, random_state=1)[["from_account", "ts"]]
        .rename(columns={"from_account": "account_id", "ts": "event_timestamp"})
        .reset_index(drop=True)
    )
    theirs = feast_store.historical(fs, events)
    assert len(theirs) == len(events)  # raw Feast drops rows here; historical() must not
    assert (theirs["account_id"] == events["account_id"]).all()

    ours = as_of_join(
        events, "account_id", "event_timestamp", state, "event_timestamp", ACCOUNT_STATE_NAMES, ""
    )
    for name in ACCOUNT_STATE_NAMES:
        assert np.allclose(ours[name], theirs[name], rtol=1e-4, atol=1e-2), name

    # Graph: where a same-day snapshot exists, Feast and our join agree exactly.
    day = events["event_timestamp"].dt.floor("D")
    snap = events.assign(snapshot_ts=day).merge(graph, on=["account_id", "snapshot_ts"], how="left")
    has = snap[GRAPH_DAILY_NAMES[0]].notna()
    assert has.sum() > 0 and (~has).sum() > 0  # both cases are exercised
    for name in GRAPH_DAILY_NAMES:
        assert (theirs.loc[has, name].astype(int) == snap.loc[has, name].astype(int)).all(), name
    assert theirs.loc[~has, GRAPH_DAILY_NAMES].isna().all().all()  # null, not dropped


def test_online_returns_latest_values(store: tuple) -> None:
    fs, state, graph = store
    accounts = state["account_id"].drop_duplicates().head(20).tolist()
    online = feast_store.online(fs, accounts).set_index("account_id").loc[accounts]
    latest = state.sort_values("event_timestamp").groupby("account_id").tail(1)
    latest = latest.set_index("account_id").loc[accounts]
    for name in ACCOUNT_STATE_NAMES:
        assert np.allclose(online[name].astype(float), latest[name], rtol=1e-4, atol=1e-2), name
    # No TTL on read: graph features are the account's latest snapshot, however old.
    last_snap = graph.sort_values("snapshot_ts").groupby("account_id").tail(1)
    last_snap = last_snap.set_index("account_id").loc[accounts]
    for name in GRAPH_DAILY_NAMES:
        assert (online[name].astype(int) == last_snap[name].astype(int)).all(), name
