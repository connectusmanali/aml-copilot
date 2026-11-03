"""Build and store features: account state, daily graph snapshots, training frame.

    aml-features build  [--from db|parquet]   -> data/features/*.parquet
    aml-features feast-apply                  -> register views, materialize the online store
    aml-features feast-check --account ID     -> online vs offline values for one account

The Parquet files are the OFFLINE store (training, backtests). Feast's online store serves the
latest values per account at low latency (scoring in Steps 3-5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from aml.config import get_settings
from aml.features.account_state import build_account_state
from aml.features.graph import build_graph_snapshots
from aml.features.pit import training_frame

STATE_FILE = "account_state.parquet"
GRAPH_FILE = "account_graph_daily.parquet"
TRAINING_FILE = "training_frame.parquet"


def load_inputs(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = get_settings()
    if source == "db":
        from aml.db.read import ACCOUNTS_SQL, TRANSACTIONS_SQL, fetch_frame

        return fetch_frame(s.postgres_dsn, TRANSACTIONS_SQL), fetch_frame(
            s.postgres_dsn, ACCOUNTS_SQL
        )
    d = s.processed_dir
    accounts = pd.read_parquet(d / "accounts.parquet").merge(
        pd.read_parquet(d / "customers.parquet"), on="customer_id"
    )
    return pd.read_parquet(d / "transactions.parquet"), accounts


def build(source: str, out_dir: Path) -> dict[str, int]:
    t0 = time.perf_counter()
    txns, accounts = load_inputs(source)
    state = build_account_state(txns)
    graph = build_graph_snapshots(txns)
    frame = training_frame(txns, state, graph, accounts)

    out_dir.mkdir(parents=True, exist_ok=True)
    state.to_parquet(out_dir / STATE_FILE, index=False)
    graph.to_parquet(out_dir / GRAPH_FILE, index=False)
    frame.to_parquet(out_dir / TRAINING_FILE, index=False)
    counts = {
        "transactions": len(txns),
        "state_rows": len(state),
        "graph_rows": len(graph),
        "training_rows": len(frame),
    }
    (out_dir / "_meta.json").write_text(json.dumps({"source": source, **counts}))
    print(
        f"features from {source}: "
        + ", ".join(f"{k}={v:,}" for k, v in counts.items())
        + f" ({time.perf_counter() - t0:.1f}s) -> {out_dir}/"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-features")
    sub = p.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build", help="compute features into data/features/")
    bp.add_argument("--from", dest="source", choices=["db", "parquet"], default="db")
    sub.add_parser("feast-apply", help="register feature views and materialize online store")
    cp = sub.add_parser("feast-check", help="compare online and offline features for an account")
    cp.add_argument("--account", required=True)
    args = p.parse_args(argv)

    s = get_settings()
    if args.cmd == "build":
        build(args.source, s.features_dir)
        return
    try:
        from aml.features import feast_store
    except ImportError:
        sys.exit("Feast is not installed: run `uv sync --extra feast`")
    if args.cmd == "feast-apply":
        feast_store.apply_and_materialize(s.feast_dir, s.features_dir)
    else:
        ok = feast_store.check_account(s.feast_dir, s.features_dir, args.account)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
