"""Daily graph snapshots: each account's place in the money network, as of the start of a day.

The snapshot for day D uses only transactions in [D - 30 days, D), so a transaction on day D
is scored against yesterday's network: never against itself or anything later that day.

Computed with sparse matrices (scipy), not Neo4j. Features must be reproducible, versioned and
point-in-time correct for training; Neo4j serves the investigator's and the agent's ad-hoc
questions ("show me this account's network") in Step 4. Same data, two jobs.

"Known bad" uses labels, so it carries a reporting LAG: an account counts as confirmed only if
its laundering happened more than `label_lag_days` before D. This mimics a bank learning about
cases from filed reports, days after the activity. Without the lag, the feature would leak the
answer. Step 3 measures the model with and without these features (ablation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from aml.features.definitions import GRAPH_DAILY_NAMES

UNREACHABLE = 99
MAX_HOPS = 6


def _snapshot(
    src: np.ndarray, dst: np.ndarray, bad_codes: np.ndarray, hub_degree: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    nodes = np.unique(np.concatenate([src, dst]))
    s, d = np.searchsorted(nodes, src), np.searchsorted(nodes, dst)
    n = len(nodes)
    keep = s != d  # self-transfers (e.g. reinvestment) are not network edges
    a = sparse.csr_matrix((np.ones(keep.sum()), (s[keep], d[keep])), shape=(n, n))
    a.data[:] = 1.0  # binary: "any payment i -> j in the window"

    out_deg = np.diff(a.indptr)
    in_deg = np.diff(a.tocsc().indptr)
    reciprocal = np.asarray(a.multiply(a.T).sum(axis=1)).ravel()

    # Three-account loops through i: paths i->j->k with an edge k->i. Hubs (thousands of
    # counterparties, e.g. payroll accounts) are excluded: their loop counts explode
    # combinatorially and carry no signal.
    small = (out_deg + in_deg) <= hub_degree
    mask = sparse.diags(small.astype(np.float64))
    a_small = mask @ a @ mask
    cycles3 = np.asarray((a_small @ a_small).multiply(a_small.T).sum(axis=1)).ravel()

    sym = ((a + a.T) > 0).astype(np.float64)
    is_bad = np.isin(nodes, bad_codes)
    bad_neighbours = np.asarray(sym @ is_bad.astype(np.float64)).ravel()
    hops = np.full(n, UNREACHABLE, dtype=np.int64)
    if is_bad.any():
        dist = dijkstra(
            sym,
            directed=False,
            indices=np.flatnonzero(is_bad),
            unweighted=True,
            limit=MAX_HOPS,
            min_only=True,
        )
        finite = np.isfinite(dist)
        hops[finite] = dist[finite].astype(np.int64)

    return nodes, {
        "g_out_degree_30d": out_deg,
        "g_in_degree_30d": in_deg,
        "g_reciprocal_30d": reciprocal,
        "g_cycles3_30d": cycles3,
        "g_known_bad_neighbours": bad_neighbours,
        "g_hops_to_known_bad": hops,
        "g_is_known_bad": is_bad.astype(np.int64),
    }


def build_graph_snapshots(
    txns: pd.DataFrame,
    *,
    window_days: int = 30,
    label_lag_days: int = 3,
    hub_degree: int = 1_000,
) -> pd.DataFrame:
    cols = ["account_id", "snapshot_ts", *GRAPH_DAILY_NAMES]
    if txns.empty:
        return pd.DataFrame(columns=cols)

    ts = pd.to_datetime(txns["ts"])
    codes, accounts = pd.factorize(pd.concat([txns["from_account"], txns["to_account"]]))
    n = len(txns)
    src, dst = codes[:n], codes[n:]
    bad_txn = txns["is_laundering"].astype(bool).to_numpy()
    ts_np = ts.to_numpy()

    frames = []
    for day in pd.date_range(ts.min().floor("D"), ts.max().floor("D"), freq="D"):
        in_window = (ts_np >= (day - pd.Timedelta(days=window_days)).to_datetime64()) & (
            ts_np < day.to_datetime64()
        )
        if not in_window.any():
            continue
        confirmed = bad_txn & (ts_np < (day - pd.Timedelta(days=label_lag_days)).to_datetime64())
        bad_codes = np.unique(np.concatenate([src[confirmed], dst[confirmed]]))
        nodes, feats = _snapshot(src[in_window], dst[in_window], bad_codes, hub_degree)
        frame = pd.DataFrame({"account_id": np.asarray(accounts)[nodes], "snapshot_ts": day})
        for name in GRAPH_DAILY_NAMES:
            frame[name] = np.asarray(feats[name]).astype(np.int32)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
