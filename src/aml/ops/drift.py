"""Drift monitoring with the Population Stability Index (PSI).

PSI compares a feature's distribution now with the distribution the model was trained on:
    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%)
Industry rule of thumb (also used in bank model-monitoring standards):
    < 0.10 stable · 0.10-0.25 watch · > 0.25 act (investigate, consider retraining)
We monitor the inputs (features), the output (score distribution) and the business metric
(alerts per day against investigator capacity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WATCH, ACT = 0.10, 0.25


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < bins // 2 + 1:  # many tied values (e.g. calibrated scores): equal-width bins
        lo, hi = float(ref.min()), float(ref.max())
        if hi <= lo:  # constant reference: share of values equal to it
            r, c = np.mean(ref == lo), np.mean(cur == lo)
            a, b = np.clip([r, 1 - r], 1e-4, None), np.clip([c, 1 - c], 1e-4, None)
            return float(np.sum((b - a) * np.log(b / a)))
        edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "act" if value > ACT else "watch" if value > WATCH else "stable"


@dataclass
class DriftReport:
    features: pd.DataFrame  # feature, psi, level
    score_psi: float
    alerts_per_day_ref: float
    alerts_per_day_cur: float

    @property
    def worst(self) -> str:
        levels = [*self.features["level"], level(self.score_psi)]
        return "act" if "act" in levels else "watch" if "watch" in levels else "stable"

    def render(self) -> str:
        top = self.features.sort_values("psi", ascending=False).head(10)
        rows = "\n".join(f"| {r.feature} | {r.psi:.3f} | {r.level} |" for r in top.itertuples())
        return (
            f"# Drift report\n\nOverall: **{self.worst}**\n\n"
            f"- Score PSI: {self.score_psi:.3f} ({level(self.score_psi)})\n"
            f"- Alerts/day: reference {self.alerts_per_day_ref:.1f}, "
            f"current {self.alerts_per_day_cur:.1f}\n\n"
            "| Feature (top 10 by PSI) | PSI | Level |\n|---|---|---|\n" + rows + "\n"
        )


def drift_report(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    features: list[str],
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    threshold: float,
) -> DriftReport:
    rows = [
        {"feature": f, "psi": psi(ref[f].to_numpy(float), cur[f].to_numpy(float))} for f in features
    ]
    table = pd.DataFrame(rows)
    table["level"] = table["psi"].map(level)

    def per_day(frame: pd.DataFrame, s: np.ndarray) -> float:
        days = max((frame["ts"].max() - frame["ts"].min()) / pd.Timedelta(days=1), 1e-9)
        return float((s >= threshold).sum() / days)

    return DriftReport(
        table, psi(ref_scores, cur_scores), per_day(ref, ref_scores), per_day(cur, cur_scores)
    )
