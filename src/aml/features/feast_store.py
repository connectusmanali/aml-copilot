"""Feast feature store: one definition, served offline (training) and online (scoring).

Why Feast on top of our Parquet files: the online store answers "latest features for account X"
in milliseconds for scoring, and Feast guarantees training and serving read the SAME feature
definitions. The classic production bug it prevents is training/serving skew: a feature
computed one way in the training notebook and another way in the scoring service.

Configured in code rather than feature_store.yaml so paths come from settings:
offline store = our Parquet files, online store = SQLite (local) under data/feast/.
In the cloud the online store would be Postgres or Redis; only the config changes.

Two Feast behaviours found while testing (both covered by tests/test_feast.py):
  1. The file offline store DROPS entity rows that have no match in a feature view, instead of
     returning nulls. For us that meant transactions of accounts with no graph snapshot that
     day (new or quiet accounts, i.e. typical mules) silently vanished from training data.
     `historical()` restores left-join semantics and checks the row count.
     Training data comes from pit.training_frame (tested to keep every transaction).
  2. The online store does not apply the TTL when READING: it serves the last materialized
     value however old. Online graph features therefore mean "latest daily snapshot"; a daily
     materialization job (Step 9) keeps them current.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, RepoConfig, ValueType
from feast.types import Float32, Int32, String

from aml.features.definitions import ACCOUNT_STATE, GRAPH_DAILY, FeatureSpec
from aml.features.pipeline import GRAPH_FILE, STATE_FILE
from aml.features.pit import STATE_TTL

# A graph snapshot is valid for its own day only (timestamps are whole minutes).
GRAPH_TTL = timedelta(days=1) - timedelta(seconds=1)
_DTYPES = {"int": Int32, "float": Float32}


def _schema(specs: list[FeatureSpec]) -> list[Field]:
    # The join key is declared explicitly; left to inference, Feast types it as JSON.
    return [Field(name="account_id", dtype=String)] + [
        Field(name=f.name, dtype=_DTYPES[f.dtype], description=f.description) for f in specs
    ]


def definitions(features_dir: Path) -> list[Any]:
    account = Entity(
        name="account",
        join_keys=["account_id"],
        value_type=ValueType.STRING,
        description="Bank account",
    )
    state = FeatureView(
        name="account_state",
        entities=[account],
        ttl=STATE_TTL.to_pytimedelta(),
        schema=_schema(ACCOUNT_STATE),
        source=FileSource(
            name="account_state_source",
            path=str((features_dir / STATE_FILE).resolve()),
            timestamp_field="event_timestamp",
        ),
        online=True,
        description="Rolling-window behaviour of an account as of each of its transactions",
    )
    graph = FeatureView(
        name="account_graph_daily",
        entities=[account],
        ttl=GRAPH_TTL,
        schema=_schema(GRAPH_DAILY),
        source=FileSource(
            name="account_graph_source",
            path=str((features_dir / GRAPH_FILE).resolve()),
            timestamp_field="snapshot_ts",
        ),
        online=True,
        description="Position in the 30-day money network, as of the start of each day",
    )
    return [account, state, graph]


def store(repo_dir: Path) -> FeatureStore:
    repo_dir.mkdir(parents=True, exist_ok=True)
    config = RepoConfig(
        project="aml",
        provider="local",
        registry=str((repo_dir / "registry.db").resolve()),
        online_store={"type": "sqlite", "path": str((repo_dir / "online.db").resolve())},
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
    )
    return FeatureStore(config=config, repo_path=str(repo_dir.resolve()))


def apply_and_materialize(repo_dir: Path, features_dir: Path) -> FeatureStore:
    fs = store(repo_dir)
    fs.apply(definitions(features_dir))
    state = pd.read_parquet(features_dir / STATE_FILE, columns=["event_timestamp"])
    graph = pd.read_parquet(features_dir / GRAPH_FILE, columns=["snapshot_ts"])
    start = min(state["event_timestamp"].min(), graph["snapshot_ts"].min()).to_pydatetime()
    end = max(state["event_timestamp"].max(), graph["snapshot_ts"].max()).to_pydatetime()
    fs.materialize(start_date=start - timedelta(seconds=1), end_date=end + timedelta(seconds=1))
    print(f"feast: registered 2 feature views, materialized {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    return fs


FEATURE_REFS = [f"account_state:{f.name}" for f in ACCOUNT_STATE] + [
    f"account_graph_daily:{f.name}" for f in GRAPH_DAILY
]


class FeastRowLossError(RuntimeError):
    pass


def historical(fs: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """As-of join through Feast, with LEFT-join semantics: one output row per entity row.

    Each feature view is fetched separately and left-joined: if Feast drops a row because one
    view has no match, that must not also erase the row's features from the other view.
    entity_df: account_id, event_timestamp (naive timestamps are treated as UTC).
    """
    keys = ["account_id", "event_timestamp"]
    ents = entity_df[keys].drop_duplicates().reset_index(drop=True)
    result = ents
    for view in ("account_state", "account_graph_daily"):
        refs = [r for r in FEATURE_REFS if r.startswith(f"{view}:")]
        out = fs.get_historical_features(entity_df=ents, features=refs).to_df()
        out["event_timestamp"] = pd.to_datetime(out["event_timestamp"]).dt.tz_localize(None)
        names = [r.split(":", 1)[1] for r in refs]
        result = result.merge(out[keys + names], on=keys, how="left")
    if len(result) != len(ents):
        raise FeastRowLossError(f"expected {len(ents)} rows, got {len(result)}")
    return entity_df[keys].merge(result, on=keys, how="left")


def online(fs: FeatureStore, account_ids: list[str]) -> pd.DataFrame:
    rows = [{"account_id": a} for a in account_ids]
    return pd.DataFrame(fs.get_online_features(features=FEATURE_REFS, entity_rows=rows).to_dict())


def check_account(repo_dir: Path, features_dir: Path, account_id: str) -> bool:
    """Online value == offline value at the account's latest state timestamp."""
    fs = store(repo_dir)
    state = pd.read_parquet(features_dir / STATE_FILE)
    rows = state[state["account_id"] == account_id]
    if rows.empty:
        print(f"{account_id}: no features")
        return False
    latest: datetime = rows["event_timestamp"].max().to_pydatetime()
    offline = historical(
        fs, pd.DataFrame({"account_id": [account_id], "event_timestamp": [latest]})
    )
    on = online(fs, [account_id])
    names = [f.name for f in ACCOUNT_STATE]
    diffs = [n for n in names if not _close(offline[n].iloc[0], on[n].iloc[0])]
    for n in names:
        mark = "" if n not in diffs else "   <-- differs"
        print(f"{n:<26} offline={offline[n].iloc[0]!s:<14} online={on[n].iloc[0]!s}{mark}")
    print("online == offline" if not diffs else f"MISMATCH in {len(diffs)} features")
    return not diffs


def _close(a: Any, b: Any) -> bool:
    a_missing, b_missing = a is None or bool(pd.isna(a)), b is None or bool(pd.isna(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(a)))
