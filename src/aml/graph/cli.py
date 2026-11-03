"""Graph command line (needs `uv sync --extra graph` and Neo4j: local `make up` or AuraDB).

aml-graph load [--from db|parquet] [--since 2022-09-01 --until 2022-09-15]
aml-graph cycles --account 010-8000EBD30 [--since ... --until ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from aml.config import get_settings
from aml.features.pipeline import load_inputs
from aml.graph import queries
from aml.graph.load import CapacityError, check_capacity, load


def _driver() -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("neo4j driver is not installed: run `uv sync --extra graph`")
    s = get_settings()
    return GraphDatabase.driver(
        s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password.get_secret_value())
    )


def _window(txns: pd.DataFrame, since: str | None, until: str | None) -> pd.DataFrame:
    ts = pd.to_datetime(txns["ts"])
    keep = pd.Series(True, index=txns.index)
    if since:
        keep &= ts >= pd.Timestamp(since)
    if until:
        keep &= ts < pd.Timestamp(until)
    return txns[keep]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-graph")
    sub = p.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("load", help="load accounts and transactions into Neo4j (idempotent)")
    lp.add_argument("--from", dest="source", choices=["db", "parquet"], default="db")
    lp.add_argument("--since")
    lp.add_argument("--until")
    cp = sub.add_parser("cycles", help="money that loops back to an account")
    cp.add_argument("--account", required=True)
    cp.add_argument("--since", default="2000-01-01")
    cp.add_argument("--until", default="2100-01-01")
    cp.add_argument("--max-hops", type=int, default=3)
    args = p.parse_args(argv)

    s = get_settings()
    with _driver() as driver, driver.session(database=s.neo4j_database) as session:
        if args.cmd == "load":
            txns, accounts = load_inputs(args.source)
            txns = _window(txns, args.since, args.until)
            used = set(txns["from_account"]) | set(txns["to_account"])
            accounts = accounts[accounts["account_id"].isin(used)]
            try:
                check_capacity(
                    session, txns, accounts, s.neo4j_max_nodes, s.neo4j_max_relationships
                )
            except CapacityError as e:
                sys.exit(str(e))
            counts = load(session, accounts, txns)
            print(
                f"graph: merged {counts['accounts']:,} accounts, "
                f"{counts['transactions']:,} transactions"
            )
        else:
            rows = queries.cycles_through(
                session,
                args.account,
                datetime.fromisoformat(args.since),
                datetime.fromisoformat(args.until),
                max_hops=args.max_hops,
            )
            print(json.dumps(rows, indent=2, default=str) if rows else "no cycles found")


if __name__ == "__main__":
    main()
