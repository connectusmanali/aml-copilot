"""Investigation queries over the money network. Step 4's agent calls these as read-only tools.

Every query is bounded: a time window, a hop limit and a row LIMIT. An unbounded graph query on a
busy account can touch millions of paths; an agent must never be able to start one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aml.graph.load import Session

MAX_HOPS = 4


def _hops(hops: int) -> int:
    # Cypher cannot take a hop count as a parameter, so it is formatted into the query text.
    # Validating it as a small int first is what makes that safe (no injection possible).
    if not isinstance(hops, int) or not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"hops must be an int between 1 and {MAX_HOPS}")
    return hops


def _window(since: datetime, until: datetime) -> dict[str, str]:
    if since >= until:
        raise ValueError("since must be before until")
    return {
        "since": since.strftime("%Y-%m-%dT%H:%M:%S"),
        "until": until.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def cycles_through(
    session: Session,
    account_id: str,
    since: datetime,
    until: datetime,
    max_hops: int = 3,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Money that leaves the account and comes back, hop by hop forward in time (layering)."""
    hops = _hops(max_hops)
    query = f"""
    MATCH p = (a:Account {{id: $account_id}})-[rs:SENT*2..{hops}]->(a)
    WHERE all(r IN rs WHERE r.ts >= localdatetime($since) AND r.ts < localdatetime($until))
      AND all(i IN range(0, size(rs) - 2) WHERE rs[i].ts <= rs[i + 1].ts)
    RETURN [n IN nodes(p) | n.id] AS accounts,
           [r IN rs | r.txn_id] AS txn_ids,
           [r IN rs | r.amount_usd] AS amounts_usd,
           [r IN rs | toString(r.ts)] AS timestamps
    LIMIT $limit
    """
    params = {"account_id": account_id, "limit": limit, **_window(since, until)}
    return [dict(r) for r in session.run(query, params)]


def counterparties(
    session: Session, account_id: str, since: datetime, until: datetime, limit: int = 50
) -> list[dict[str, Any]]:
    """Who the account paid and was paid by, with totals: the fan-out / fan-in picture."""
    query = """
    MATCH (a:Account {id: $account_id})-[r:SENT]-(c:Account)
    WHERE r.ts >= localdatetime($since) AND r.ts < localdatetime($until)
    WITH c, CASE WHEN startNode(r) = a THEN 'out' ELSE 'in' END AS direction, r
    RETURN c.id AS counterparty, direction, count(r) AS payments,
           round(sum(r.amount_usd), 2) AS total_usd, c.risk_rating AS risk_rating
    ORDER BY total_usd DESC
    LIMIT $limit
    """
    params = {"account_id": account_id, "limit": limit, **_window(since, until)}
    return [dict(r) for r in session.run(query, params)]


def neighbourhood(
    session: Session,
    account_id: str,
    since: datetime,
    until: datetime,
    hops: int = 2,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Accounts within N hops in the window, with their distance (for the investigator's map)."""
    h = _hops(hops)
    query = f"""
    MATCH p = (a:Account {{id: $account_id}})-[rs:SENT*1..{h}]-(b:Account)
    WHERE b <> a
      AND all(r IN rs WHERE r.ts >= localdatetime($since) AND r.ts < localdatetime($until))
    RETURN b.id AS account, min(length(p)) AS hops, b.risk_rating AS risk_rating
    ORDER BY hops, account
    LIMIT $limit
    """
    params = {"account_id": account_id, "limit": limit, **_window(since, until)}
    return [dict(r) for r in session.run(query, params)]
