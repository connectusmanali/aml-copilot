"""Neo4j loader and queries, against a recording fake session (no Neo4j needed)."""

from datetime import datetime

import pandas as pd
import pytest

from aml.data.profiles import build_profiles
from aml.graph import queries
from aml.graph.load import CapacityError, check_capacity, load, transaction_rows
from tests.fakes import FakeSession


@pytest.fixture(scope="module")
def accounts(txns: pd.DataFrame) -> pd.DataFrame:
    customers, accts = build_profiles(txns)
    return accts.merge(customers, on="customer_id")


def test_label_is_never_sent_to_the_graph(txns: pd.DataFrame) -> None:
    rows = transaction_rows(txns)
    assert all("is_laundering" not in r for r in rows)
    assert rows[0]["ts"].count("T") == 1  # ISO string for localdatetime()


def test_load_creates_schema_then_batches(txns: pd.DataFrame, accounts: pd.DataFrame) -> None:
    session = FakeSession()
    counts = load(session, accounts, txns, batch_size=500)
    statements = [q for q, _ in session.calls]
    assert statements[0].startswith("CREATE CONSTRAINT account_id IF NOT EXISTS")
    merges = [p for q, p in session.calls if "MERGE (s)-[t:SENT" in q]
    assert len(merges) == -(-len(txns) // 500)  # ceil: one call per batch
    assert sum(len(p["rows"]) for p in merges) == counts["transactions"] == len(txns)
    assert all("MERGE" in q or q.startswith("CREATE") for q in statements)  # idempotent only


def test_capacity_guard(txns: pd.DataFrame, accounts: pd.DataFrame) -> None:
    check_capacity(FakeSession(), txns, accounts, max_nodes=10_000, max_rels=10_000)
    with pytest.raises(CapacityError, match="smaller window"):
        check_capacity(FakeSession(), txns, accounts, max_nodes=10, max_rels=10_000)
    full = FakeSession(single={"nodes": 100, "rels": 9_000})
    with pytest.raises(CapacityError):
        check_capacity(full, txns, accounts, max_nodes=10_000, max_rels=10_000)


@pytest.mark.parametrize("hops", [0, 5, "3", 2.0])
def test_hop_count_is_validated_before_it_reaches_cypher(hops: object) -> None:
    with pytest.raises(ValueError):
        queries.cycles_through(
            FakeSession(),
            "A",
            datetime(2022, 9, 1),
            datetime(2022, 9, 2),
            max_hops=hops,  # type: ignore[arg-type]
        )


def test_queries_are_bounded_and_parameterized() -> None:
    session = FakeSession()
    since, until = datetime(2022, 9, 1), datetime(2022, 9, 8)
    queries.cycles_through(session, "010-ABC'; DROP", since, until, max_hops=3)
    queries.counterparties(session, "010-ABC", since, until)
    queries.neighbourhood(session, "010-ABC", since, until, hops=2)
    for query, params in session.calls:
        assert "LIMIT $limit" in query
        assert "$since" in query and "$until" in query
        assert "DROP" not in query  # the account id travels as a parameter, never as text
        assert params["since"] == "2022-09-01T00:00:00"
    with pytest.raises(ValueError):
        queries.counterparties(session, "A", until, since)
