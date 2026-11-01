"""Convert amounts to US dollars so they can be added up and compared.

IBM's data mixes 15 currencies. Summing 10,000 yen and 10,000 dollars as "20,000" is meaningless,
so every volume, feature and threshold uses USD.

These are APPROXIMATE rates for September 2022 (the dataset's period), fixed on purpose: features
must be reproducible, so the rate cannot depend on the day you run the pipeline. A production
system would join the daily rate for each transaction's date from a rates table.

A currency missing from this table converts to NaN; the data quality report counts those rows,
so a new currency in the source data is noticed instead of silently ignored.
"""

from __future__ import annotations

import pandas as pd

USD_PER_UNIT: dict[str, float] = {
    "US Dollar": 1.0,
    "Euro": 1.00,
    "UK Pound": 1.15,
    "Swiss Franc": 1.03,
    "Canadian Dollar": 0.75,
    "Australian Dollar": 0.67,
    "Yen": 0.0070,
    "Yuan": 0.145,
    "Rupee": 0.0123,
    "Ruble": 0.0165,
    "Mexican Peso": 0.050,
    "Brazil Real": 0.19,
    "Saudi Riyal": 0.266,
    "Shekel": 0.29,
    "Bitcoin": 19_500.0,
}


def to_usd(amount: pd.Series, currency: pd.Series) -> pd.Series:
    """Element-wise conversion. Unknown currencies give NaN (counted by the quality report)."""
    rate = currency.map(USD_PER_UNIT).astype(float)
    return amount.astype(float) * rate


def unknown_currencies(currency: pd.Series) -> list[str]:
    return sorted(set(currency.dropna().unique()) - set(USD_PER_UNIT))
