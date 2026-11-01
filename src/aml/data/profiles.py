"""Generate synthetic KYC customer profiles for every account in the transactions.

The IBM data has accounts but no customers. Investigators always compare activity against
what the customer *declared* at onboarding, so we create that declaration here.

Realism choices (each is a talking point in interviews):
  - expected_monthly_volume is derived from actual activity (in USD) with noise, so most
    customers look consistent. Under-declaring is a real red flag, so laundering accounts
    under-declare MORE OFTEN than others, but not always, and some honest customers do too.
    An earlier version made every laundering account under-declare, which leaked the label
    into a feature; see ADR 0003.
  - risk_rating comes from a simple, explainable KYC rule (country + segment + occupation),
    the way first-line onboarding teams actually assign it.

Known limitation of synthetic data (documented in the model card in Step 3): declared volume and
segment are derived from activity across the whole window, i.e. partly from the "future" relative
to onboarding. Real KYC is captured before any activity.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd

from aml.data.fx import to_usd

COUNTRIES = ["CA", "US", "GB", "MX", "AE", "HK", "PA", "CY"]
COUNTRY_P = [0.78, 0.10, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]
HIGH_RISK_COUNTRIES = {"PA", "CY", "AE", "HK"}  # illustrative only, not an official list

INDIVIDUAL_JOBS = [
    "teacher",
    "nurse",
    "software developer",
    "student",
    "retired",
    "accountant",
    "electrician",
    "sales associate",
    "self-employed",
]
BUSINESS_TYPES = [
    "restaurant",
    "construction",
    "import/export",
    "money service business",
    "car dealership",
    "consulting",
    "retail",
    "real estate holding",
]
# Probability that a customer under-declares (declares 10-50% of real activity).
P_UNDER_DECLARE_LAUNDERING = 0.40
P_UNDER_DECLARE_OTHER = 0.10

HIGH_RISK_BUSINESS = {
    "money service business",
    "import/export",
    "car dealership",
    "real estate holding",
}


def _risk(country: str, segment: str, occupation: str) -> str:
    score = 0
    score += 2 if country in HIGH_RISK_COUNTRIES else 0
    score += 2 if occupation in HIGH_RISK_BUSINESS else 0
    score += 1 if segment == "business" else 0
    return "high" if score >= 3 else "medium" if score >= 1 else "low"


def build_profiles(txns: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (customers, accounts) DataFrames. One customer per account for now."""
    rng = np.random.default_rng(seed)

    usd = to_usd(txns["amount_paid"], txns["payment_currency"]).fillna(0.0)
    sent = (
        txns.assign(amount_usd=usd)
        .groupby("from_account")
        .agg(n_sent=("txn_id", "size"), vol_sent=("amount_usd", "sum"))
    )
    bad = pd.concat(
        [
            txns.loc[txns["is_laundering"], "from_account"],
            txns.loc[txns["is_laundering"], "to_account"],
        ]
    ).unique()
    all_accounts = pd.Index(pd.concat([txns["from_account"], txns["to_account"]]).unique())
    stats = sent.reindex(all_accounts).fillna(0)

    span_days = max((txns["ts"].max() - txns["ts"].min()).days, 1)
    monthly_actual = stats["vol_sent"] / span_days * 30

    n = len(all_accounts)
    # Busiest ~15% of accounts are businesses.
    busy_cut = stats["n_sent"].quantile(0.85)
    segment = np.where(stats["n_sent"] > busy_cut, "business", "individual")
    occupation = [
        str(rng.choice(BUSINESS_TYPES)) if s == "business" else str(rng.choice(INDIVIDUAL_JOBS))
        for s in segment
    ]
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_P)

    # Declared volume = actual * noise, except for under-declarers (10-50% of actual).
    # Laundering accounts under-declare more often than others, but neither group always does.
    noise = rng.lognormal(0.0, 0.3, size=n)
    is_bad = all_accounts.isin(bad)
    p_under = np.where(is_bad, P_UNDER_DECLARE_LAUNDERING, P_UNDER_DECLARE_OTHER)
    under = rng.random(n) < p_under
    declared = monthly_actual * np.where(under, rng.uniform(0.1, 0.5, size=n), noise)
    declared = np.maximum(declared, 500.0).round(2)

    first_seen = txns["ts"].min().date()
    onboarded = [first_seen - timedelta(days=int(d)) for d in rng.integers(30, 3650, size=n)]

    # Derived from the account ID, not a counter: the same account always maps to the same
    # customer, so loading a second dataset can never attach accounts to someone else's profile.
    customer_ids = [
        "C" + hashlib.sha256(str(a).encode()).hexdigest()[:12].upper() for a in all_accounts
    ]
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "segment": segment,
            "occupation": occupation,
            "country": country,
            "risk_rating": [
                _risk(c, s, o) for c, s, o in zip(country, segment, occupation, strict=True)
            ],
            "expected_monthly_volume": declared,
            "onboarded_at": onboarded,
        }
    )
    accounts = pd.DataFrame(
        {
            "account_id": all_accounts.astype(str),
            "customer_id": customer_ids,
            "bank_id": [a.split("-", 1)[0] for a in all_accounts],
        }
    )
    return customers, accounts
