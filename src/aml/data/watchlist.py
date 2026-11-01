"""Synthetic watchlist (sanctions, PEP, adverse media) for the agent's screening tool.

Real screening matches NAMES against vendor lists with fuzzy matching; our synthetic customers
have no names, so entries reference accounts directly. As with KYC, the label is used only to
make hits MORE LIKELY on laundering accounts, never certain: most laundering accounts are not
listed and some listed accounts are clean, like real lists.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

P_LISTED_LAUNDERING = 0.08
P_LISTED_OTHER = 0.005
LISTS = ["sanctions", "pep", "adverse_media"]
LIST_P = [0.15, 0.35, 0.50]
REASONS = {
    "sanctions": "Name match on a consolidated sanctions list (synthetic entry)",
    "pep": "Politically exposed person or close associate (synthetic entry)",
    "adverse_media": "Negative news coverage alleging financial crime (synthetic entry)",
}


def build_watchlist(txns: pd.DataFrame, accounts: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    bad_txn = txns["is_laundering"].astype(bool)
    bad = set(txns.loc[bad_txn, "from_account"]) | set(txns.loc[bad_txn, "to_account"])
    ids = accounts["account_id"].to_numpy()
    p = np.where(np.isin(ids, list(bad)), P_LISTED_LAUNDERING, P_LISTED_OTHER)
    listed = ids[rng.random(len(ids)) < p]
    kinds = rng.choice(LISTS, size=len(listed), p=LIST_P)
    first = pd.to_datetime(txns["ts"]).min().date()
    return pd.DataFrame(
        {
            "entry_id": [f"WL{i:05d}" for i in range(len(listed))],
            "account_id": listed,
            "list": kinds,
            "reason": [REASONS[k] for k in kinds],
            "listed_at": [
                first - timedelta(days=int(d)) for d in rng.integers(30, 1500, len(listed))
            ],
        }
    )
