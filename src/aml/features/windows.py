"""Vectorized time-window aggregation over many accounts at once.

The trick: give every event one sortable integer key, account_code * SPAN + epoch_seconds.
Keys of different accounts never overlap, so a single sorted array plus np.searchsorted answers
"how many events, and what total, did account A have in (T - w, T]?" for millions of (A, T)
queries without a Python loop over accounts.

Windows are (T - w, T]: they INCLUDE events at time T (a transaction is known when it happens)
and EXCLUDE anything after T. Nothing here can see the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPAN = np.int64(1 << 34)  # > any epoch second this century; keeps accounts apart
HOUR, DAY = 3_600, 86_400


def time_keys(codes: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    return codes.astype(np.int64) * SPAN + seconds.astype(np.int64)


@dataclass(frozen=True)
class Events:
    """Sorted event keys plus cumulative sums of their values (with a leading 0)."""

    keys: np.ndarray
    cums: dict[str, np.ndarray]

    @classmethod
    def build(cls, keys: np.ndarray, values: dict[str, np.ndarray]) -> Events:
        order = np.argsort(keys, kind="stable")
        cums = {
            name: np.concatenate([[0.0], np.cumsum(np.asarray(v, dtype=np.float64)[order])])
            for name, v in values.items()
        }
        return cls(keys[order], cums)

    def window(
        self, query_keys: np.ndarray, width_s: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Count and per-value sums of events in (T - width, T] for each query key."""
        hi = np.searchsorted(self.keys, query_keys, side="right")
        lo = np.searchsorted(self.keys, query_keys - width_s, side="right")
        return hi - lo, {name: c[hi] - c[lo] for name, c in self.cums.items()}


def distinct_in_window(
    acct: np.ndarray, cpty: np.ndarray, seconds: np.ndarray, query_keys: np.ndarray, width_s: int
) -> np.ndarray:
    """Distinct counterparties per account in (T - width, T], fully vectorized.

    A pair (account, counterparty) is counted at T if its latest event at or before T is newer
    than T - width. Each event "covers" the interval [t, min(next event of the same pair,
    t + width)); the distinct count at T is the number of intervals covering T, i.e.
    (#interval starts <= T) - (#interval ends <= T). Using the NEXT event only to end an
    interval never changes the answer for any T before that next event, so results for time T
    are identical whether or not later data exists (tested in test_features.py).
    """
    order = np.lexsort((seconds, cpty, acct))
    a, p, t = acct[order], cpty[order], seconds[order].astype(np.int64)
    same_next = (a[1:] == a[:-1]) & (p[1:] == p[:-1])
    nxt = np.full(len(t), np.iinfo(np.int64).max, dtype=np.int64)
    nxt[:-1][same_next] = t[1:][same_next]
    end = np.minimum(nxt, t + width_s)

    starts = np.sort(time_keys(a, t))
    ends = np.sort(time_keys(a, end))
    base = query_keys - (query_keys % SPAN)  # start of each query's account range
    starts_le = np.searchsorted(starts, query_keys, "right") - np.searchsorted(starts, base, "left")
    ends_le = np.searchsorted(ends, query_keys, "right") - np.searchsorted(ends, base, "left")
    return (starts_le - ends_le).astype(np.int64)
