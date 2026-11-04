"""Shadow and canary scoring for a challenger model.

Shadow: the challenger scores the same traffic as the champion; its scores are stored next to
the champion's (cases.alerts.shadow_score) but never route a case. We compare alert overlap,
the alert rate and, once investigators decide, precision on the decided cases.
Canary: the challenger routes a small, deterministic slice of accounts (hash-based, so the same
account always gets the same model). Promotion needs the gates below AND a human sign-off.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aml.detection.model import ModelBundle


@dataclass
class ShadowReport:
    champion: str
    challenger: str
    champion_alerts: int
    challenger_alerts: int
    overlap: float  # Jaccard of the two alert sets
    rank_corr: float  # Spearman correlation of scores
    precision_champion: float | None
    precision_challenger: float | None

    def promotable(self, min_overlap: float = 0.5, max_alert_change: float = 0.2) -> bool:
        change = abs(self.challenger_alerts - self.champion_alerts) / max(self.champion_alerts, 1)
        better = (
            self.precision_champion is None
            or self.precision_challenger is None
            or self.precision_challenger >= self.precision_champion
        )
        return self.overlap >= min_overlap and change <= max_alert_change and better


def shadow(
    frame: pd.DataFrame,
    champion: ModelBundle,
    challenger: ModelBundle,
    labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, ShadowReport]:
    a, b = champion.predict(frame), challenger.predict(frame)
    ha, hb = a >= champion.threshold, b >= challenger.threshold
    union = (ha | hb).sum()

    def prec(hit: np.ndarray) -> float | None:
        if labels is None or hit.sum() == 0:
            return None
        return round(float(labels.to_numpy()[hit].mean()), 4)

    rank = float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    out = pd.DataFrame(
        {
            "txn_id": frame["txn_id"].to_numpy(),
            "score": a,
            "shadow_score": b,
            "shadow_version": challenger.version,
        }
    )
    return out, ShadowReport(
        champion.version,
        challenger.version,
        int(ha.sum()),
        int(hb.sum()),
        round(float((ha & hb).sum() / union) if union else 1.0, 4),
        round(rank, 4),
        prec(ha),
        prec(hb),
    )


def canary_bucket(account_id: str, percent: float) -> bool:
    """True if this account is routed by the challenger. Stable across runs and processes."""
    h = int(hashlib.sha256(account_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < percent * 100
