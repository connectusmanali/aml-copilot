"""Real Neo4j: idempotent load and the injected laundering cycle is found.

Runs in CI against a Neo4j service (and locally after `make up`). Every id is prefixed with a
unique tag and only those nodes are deleted afterwards: your own graph is never wiped.
"""

import uuid
from datetime import datetime

import pandas as pd
import pytest

neo4j = pytest.importorskip("neo4j", reason="needs --extra graph")

from aml.config import get_settings  # noqa: E402
from aml.data.profiles import build_profiles  # noqa: E402
from aml.graph import queries  # noqa: E402
from aml.graph.load import load  # noqa: E402

pytestmark = pytest.mark.neo4j


def test_load_twice_and_find_the_cycle(txns: pd.DataFrame) -> None:
    s = get_settings()
    tag = f"T{uuid.uuid4().hex[:6]}-"
    t = txns.assign(
        from_account=tag + txns["from_account"],
        to_account=tag + txns["to_account"],
        txn_id=tag + txns["txn_id"],
    )
    customers, accounts = build_profiles(t)
    accounts = accounts.merge(customers, on="customer_id")

    driver = neo4j.GraphDatabase.driver(
        s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password.get_secret_value())
    )
    with driver, driver.session(database=s.neo4j_database) as session:
        try:
            load(session, accounts, t)
            load(session, accounts, t)  # MERGE: no duplicates
            nodes = session.run(
                "MATCH (a:Account) WHERE a.id STARTS WITH $tag RETURN count(a) AS n", tag=tag
            ).single()["n"]
            rels = session.run(
                "MATCH (:Account)-[r:SENT]->() WHERE r.txn_id STARTS WITH $tag "
                "RETURN count(r) AS n",
                tag=tag,
            ).single()["n"]
            assert nodes == len(accounts)
            assert rels == len(t)
            labels = session.run(
                "MATCH ()-[r:SENT]->() WHERE r.txn_id STARTS WITH $tag "
                "RETURN count(r.is_laundering) AS n",
                tag=tag,
            ).single()["n"]
            assert labels == 0  # the answer key never reaches the graph

            cycle = t[t["is_laundering"] & (t["payment_format"] == "ACH")].iloc[0]
            found = queries.cycles_through(
                session, cycle["from_account"], datetime(2022, 8, 1), datetime(2022, 11, 1)
            )
            assert any(
                r["accounts"][0] == r["accounts"][-1] == cycle["from_account"] for r in found
            )
        finally:
            session.run("MATCH (a:Account) WHERE a.id STARTS WITH $tag DETACH DELETE a", tag=tag)
