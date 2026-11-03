"""Every feature, defined once: name, type, and what it means to an investigator.

Feast's schema, the model card (Step 10) and the investigator UI's "why was this flagged"
text (Step 8) all read from this list, so a feature can never mean different things in
different places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: Literal["int", "float"]
    description: str


ACCOUNT_STATE: list[FeatureSpec] = [
    FeatureSpec("out_cnt_1h", "int", "Payments sent in the last hour (velocity)"),
    FeatureSpec("out_amt_1h", "float", "USD sent in the last hour"),
    FeatureSpec("out_cnt_24h", "int", "Payments sent in the last 24 hours"),
    FeatureSpec("out_amt_24h", "float", "USD sent in the last 24 hours"),
    FeatureSpec("out_cnt_7d", "int", "Payments sent in the last 7 days"),
    FeatureSpec("out_amt_7d", "float", "USD sent in the last 7 days"),
    FeatureSpec("out_amt_30d", "float", "USD sent in the last 30 days"),
    FeatureSpec("in_cnt_24h", "int", "Payments received in the last 24 hours"),
    FeatureSpec("in_amt_24h", "float", "USD received in the last 24 hours"),
    FeatureSpec("in_amt_7d", "float", "USD received in the last 7 days"),
    FeatureSpec("out_uniq_cpty_7d", "int", "Distinct accounts paid in 7 days (fan-out)"),
    FeatureSpec("in_uniq_cpty_7d", "int", "Distinct accounts paying in, 7 days (fan-in)"),
    FeatureSpec(
        "near_threshold_cash_24h",
        "int",
        "Cash payments of USD 9,000-9,999 in 24 hours (structuring)",
    ),
    FeatureSpec("cash_out_ratio_30d", "float", "Share of USD sent as cash, 30 days"),
    FeatureSpec(
        "xccy_out_share_30d", "float", "Share of payments sent in a different currency, 30 days"
    ),
    FeatureSpec(
        "spike_ratio_24h",
        "float",
        "USD sent in 24h vs the account's 30-day daily average (1 = normal day)",
    ),
    FeatureSpec(
        "passthrough_ratio_24h",
        "float",
        "USD sent / USD received in 24h, capped at 10 (money in, straight out)",
    ),
]

GRAPH_DAILY: list[FeatureSpec] = [
    FeatureSpec("g_out_degree_30d", "int", "Distinct accounts paid, 30 days before today"),
    FeatureSpec("g_in_degree_30d", "int", "Distinct accounts that paid in, 30 days"),
    FeatureSpec("g_reciprocal_30d", "int", "Counterparties money flowed both ways with"),
    FeatureSpec("g_cycles3_30d", "int", "Three-account loops (A->B->C->A) through this account"),
    FeatureSpec(
        "g_known_bad_neighbours",
        "int",
        "Direct counterparties confirmed as laundering before the reporting lag",
    ),
    FeatureSpec(
        "g_hops_to_known_bad",
        "int",
        "Shortest hops to a confirmed account (0 = itself, 99 = none within 6)",
    ),
    FeatureSpec("g_is_known_bad", "int", "Account itself confirmed before the reporting lag"),
]

ACCOUNT_STATE_NAMES = [f.name for f in ACCOUNT_STATE]
GRAPH_DAILY_NAMES = [f.name for f in GRAPH_DAILY]
