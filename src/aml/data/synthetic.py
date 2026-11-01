"""Synthetic generator in IBM raw format.

Used by tests and CI (no Kaggle download needed), and for experiments. It injects three classic
laundering typologies so later steps have something to find:
  - structuring: several cash deposits just under the 10,000 reporting threshold
  - fan-out:     one account sprays funds to many mules within hours
  - cycle:       funds travel A -> B -> C -> A and return to the origin

...and the same number of DECOYS: legitimate behaviour that looks like a typology on the surface.
Without decoys a model only learns "cash near 10,000 = laundering", which real data punishes:
  - payroll:         a business pays many employees the same day (looks like fan-out)
  - honest cash:     a single cash deposit just under 10,000 (looks like structuring)
  - refund:          A pays B, B refunds A (money comes back, like a cycle)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from aml.data.ibm import RAW_COLUMNS

FORMATS = ["ACH", "Wire", "Credit Card", "Cheque", "Cash"]
FORMAT_P = [0.40, 0.15, 0.25, 0.10, 0.10]


def _acct(rng: np.random.Generator) -> tuple[str, str]:
    bank = f"{rng.integers(1, 40):03d}"
    acct = f"{rng.integers(0x10000000, 0xFFFFFFFF):08X}"
    return bank, acct


# (accounts, normal transactions, laundering patterns) per size
SIZES: dict[str, tuple[int, int, int]] = {
    "small": (500, 5_000, 10),
    "medium": (3_000, 120_000, 600),
    "full": (10_000, 600_000, 3_000),
}


def generate(
    n_accounts: int = 500,
    n_normal: int = 5_000,
    n_patterns: int = 10,
    seed: int = 42,
    n_decoys: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2022, 9, 1)
    accounts = [_acct(rng) for _ in range(n_accounts)]
    rows: list[list[object]] = []

    def add(
        ts: datetime, a: tuple[str, str], b: tuple[str, str], amt: float, fmt: str, bad: bool
    ) -> None:
        amt = round(amt, 2)
        rows.append(
            [
                ts.strftime("%Y/%m/%d %H:%M"),
                a[0],
                a[1],
                b[0],
                b[1],
                amt,
                "US Dollar",
                amt,
                "US Dollar",
                fmt,
                int(bad),
            ]
        )

    # Normal traffic: log-normal amounts, random pairs over 30 days.
    for _ in range(n_normal):
        i, j = rng.choice(n_accounts, size=2, replace=False)
        ts = start + timedelta(minutes=int(rng.integers(0, 30 * 24 * 60)))
        fmt = str(rng.choice(FORMATS, p=FORMAT_P))
        add(ts, accounts[i], accounts[j], float(rng.lognormal(6.5, 1.2)), fmt, False)

    # Injected laundering patterns.
    for p in range(n_patterns):
        ts = start + timedelta(days=int(rng.integers(1, 28)), hours=int(rng.integers(0, 20)))
        kind = p % 3
        if kind == 0:  # structuring
            i, j = rng.choice(n_accounts, size=2, replace=False)
            src, dst = accounts[i], accounts[j]
            for k in range(int(rng.integers(3, 6))):
                add(
                    ts + timedelta(hours=3 * k),
                    src,
                    dst,
                    float(rng.uniform(9_000, 9_950)),
                    "Cash",
                    True,
                )
        elif kind == 1:  # fan-out
            src_i = int(rng.integers(n_accounts))
            # Distinct mules, never the sender itself.
            others = np.delete(np.arange(n_accounts), src_i)
            mules = rng.choice(others, size=int(rng.integers(5, 10)), replace=False)
            for k, mule_i in enumerate(mules):
                add(
                    ts + timedelta(minutes=20 * k),
                    accounts[src_i],
                    accounts[int(mule_i)],
                    float(rng.uniform(2_000, 8_000)),
                    "Wire",
                    True,
                )
        else:  # cycle
            idx = rng.choice(n_accounts, size=3, replace=False)
            amt = float(rng.uniform(20_000, 60_000))
            path = [accounts[i] for i in idx] + [accounts[idx[0]]]
            for k in range(3):
                add(ts + timedelta(hours=k), path[k], path[k + 1], amt * (0.98**k), "ACH", True)

    # Decoys: legitimate look-alikes (label 0).
    for p in range(n_patterns if n_decoys is None else n_decoys):
        ts = start + timedelta(days=int(rng.integers(1, 28)), hours=int(rng.integers(6, 18)))
        kind = p % 3
        if kind == 0:  # payroll: one business, many employees, same morning
            src_i = int(rng.integers(n_accounts))
            others = np.delete(np.arange(n_accounts), src_i)
            staff = rng.choice(others, size=int(rng.integers(6, 13)), replace=False)
            for k, emp in enumerate(staff):
                add(
                    ts + timedelta(minutes=2 * k),
                    accounts[src_i],
                    accounts[int(emp)],
                    float(rng.uniform(1_500, 6_000)),
                    "ACH",
                    False,
                )
        elif kind == 1:  # one honest cash deposit just under the threshold
            i, j = rng.choice(n_accounts, size=2, replace=False)
            add(ts, accounts[i], accounts[j], float(rng.uniform(9_000, 9_950)), "Cash", False)
        else:  # payment and refund
            i, j = rng.choice(n_accounts, size=2, replace=False)
            amt = float(rng.uniform(500, 5_000))
            add(ts, accounts[i], accounts[j], amt, "Credit Card", False)
            add(
                ts + timedelta(days=int(rng.integers(1, 4))),
                accounts[j],
                accounts[i],
                amt,
                "Credit Card",
                False,
            )

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
