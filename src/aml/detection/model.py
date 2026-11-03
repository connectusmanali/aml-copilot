"""The risk scorer as a self-contained, versioned bundle.

A bundle is everything needed to reproduce a score: the LightGBM model, the calibration curve,
the exact feature list, the alert threshold and the training metadata. Saved as plain files
(model.txt + JSON, no pickle), so a score from months ago can be replayed exactly (Step 9) and a
model validator can read every part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = {"txn_id", "ts", "from_account", "to_account", "payment_format", "is_laundering"}
LABEL_DERIVED = [
    f"{side}g_{name}"
    for side in ("s_", "r_")
    for name in ("known_bad_neighbours", "hops_to_known_bad", "is_known_bad")
]
KYC = ["s_kyc_actual_vs_declared", "s_account_age_days", "s_risk_rating", "s_is_business"]
FEATURE_GROUPS: dict[str, list[str]] = {"label_derived": LABEL_DERIVED, "kyc": KYC}


def feature_columns(frame: pd.DataFrame, exclude_groups: tuple[str, ...] = ()) -> list[str]:
    excluded = set(NON_FEATURES)
    for g in exclude_groups:
        excluded |= set(FEATURE_GROUPS[g])
    return [
        c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]


@dataclass
class ModelBundle:
    version: str
    features: list[str]
    booster: lgb.Booster
    calib_x: list[float]
    calib_y: list[float]
    threshold: float
    meta: dict[str, Any] = field(default_factory=dict)

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in frame]
        if missing:
            raise KeyError(f"model {self.version} needs features missing from input: {missing}")
        return frame[self.features].to_numpy(dtype=np.float64)

    def raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self._x(frame)), dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that the transaction is suspicious (0.8 ~ 80% of such alerts)."""
        return calibrate(self.raw_probability(frame), self.calib_x, self.calib_y)

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """TreeSHAP values in log-odds, one column per feature plus the bias (last column)."""
        return np.asarray(self.booster.predict(self._x(frame), pred_contrib=True))

    def save(self, directory: Path) -> Path:
        out = directory / self.version
        out.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out / "model.txt"))
        (out / "bundle.json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "features": self.features,
                    "calib_x": self.calib_x,
                    "calib_y": self.calib_y,
                    "threshold": self.threshold,
                    "meta": self.meta,
                },
                indent=2,
                default=str,
            )
        )
        (directory / "CURRENT").write_text(self.version)
        return out

    @classmethod
    def load(cls, directory: Path, version: str | None = None) -> ModelBundle:
        version = version or (directory / "CURRENT").read_text().strip()
        d = directory / version
        spec = json.loads((d / "bundle.json").read_text())
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        return cls(
            spec["version"],
            spec["features"],
            booster,
            spec["calib_x"],
            spec["calib_y"],
            spec["threshold"],
            spec["meta"],
        )


def calibrate(raw: np.ndarray, xs: list[float], ys: list[float]) -> np.ndarray:
    """Apply a stored isotonic calibration curve (monotone, piecewise linear)."""
    out: np.ndarray = np.clip(np.interp(raw, xs, ys), 0.0, 1.0)
    return out
