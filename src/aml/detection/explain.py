"""Turn TreeSHAP contributions into reasons an investigator can read.

LightGBM computes exact SHAP values for tree models natively (pred_contrib=True), so there is no
need for the separate `shap` package. Each reason says which feature moved the score, which way,
by how much, and what the feature means in plain words.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from aml.detection.model import ModelBundle
from aml.features.definitions import ACCOUNT_STATE, GRAPH_DAILY

_BASE = {f.name: f.description for f in ACCOUNT_STATE + GRAPH_DAILY}
_BASE.update(
    {
        "amount_usd": "Payment amount in USD",
        "is_cash": "Paid in cash",
        "is_xccy": "Paid in a different currency than received",
        "kyc_actual_vs_declared": "30-day outflow divided by the volume declared at onboarding",
        "account_age_days": "Days since the account was opened",
        "risk_rating": "KYC risk rating (0 low, 1 medium, 2 high)",
        "is_business": "Business customer",
    }
)


def describe(feature: str) -> str:
    for prefix, who in (("s_", "Sender"), ("r_", "Receiver")):
        if feature.startswith(prefix):
            base = feature[len(prefix) :]
            return f"{who}: {_BASE.get(base, base)}"
    return _BASE.get(feature, feature)


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float
    contribution: float  # log-odds; positive raises risk
    direction: str
    text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def reasons(bundle: ModelBundle, frame: pd.DataFrame, top: int = 5) -> list[list[Reason]]:
    contrib = bundle.contributions(frame)[:, :-1]  # drop the bias column
    values = frame[bundle.features].to_numpy(dtype=np.float64)
    out: list[list[Reason]] = []
    for row_c, row_v in zip(contrib, values, strict=True):
        order = np.argsort(-np.abs(row_c))[:top]
        out.append(
            [
                Reason(
                    feature=bundle.features[i],
                    value=float(row_v[i]),
                    contribution=round(float(row_c[i]), 4),
                    direction="raises risk" if row_c[i] > 0 else "lowers risk",
                    text=f"{describe(bundle.features[i])} = {row_v[i]:,.2f}",
                )
                for i in order
                if row_c[i] != 0
            ]
        )
    return out
