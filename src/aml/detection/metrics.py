"""Metrics an AML team actually uses: alert budgets, not just AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MINUTES_PER_ALERT = 30  # typical L1 review time for a false positive (vendor/industry surveys)


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if y.sum() == 0:
        return 0.0
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted probability and observed rate, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def summary(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    has_both = 0 < y.sum() < len(y)
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if has_both else float("nan"),
        "ece": expected_calibration_error(y, prob),
        "positives": int(y.sum()),
        "n": len(y),
    }


def threshold_for_capacity(score: np.ndarray, days: float, alerts_per_day: int) -> float:
    """Score threshold that produces about `alerts_per_day` alerts on this data."""
    budget = round(alerts_per_day * days)
    if budget >= len(score):
        return float(score.min())
    if budget <= 0:
        return float(score.max()) + 1e-9
    return float(np.sort(score)[::-1][budget - 1])


def hours(alerts: int) -> float:
    return round(alerts * MINUTES_PER_ALERT / 60, 1)
