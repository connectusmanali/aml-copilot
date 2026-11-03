"""Score transactions into alerts: model score + rule hits + reasons, one row per transaction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aml.detection.explain import reasons
from aml.detection.model import ModelBundle
from aml.detection.rules import Scenario, evaluate, fired_names, load_scenarios


def score_frame(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    scenarios: list[Scenario] | None = None,
    with_reasons: bool = True,
) -> pd.DataFrame:
    """Alerts = model score at/above the capacity threshold OR any rule scenario fired.

    priority 'high'  -> score >= threshold: full investigation by the agent
    priority 'low'   -> rule hit only: light review (Step 5 routes these differently)
    """
    scenarios = scenarios or load_scenarios()
    fired = evaluate(frame, scenarios)
    typ = {s.id: s.typology for s in scenarios}
    score = bundle.predict(frame)
    out = pd.DataFrame(
        {
            "txn_id": frame["txn_id"].to_numpy(),
            "ts": pd.to_datetime(frame["ts"]).to_numpy(),
            "from_account": frame["from_account"].to_numpy(),
            "to_account": frame["to_account"].to_numpy(),
            "amount_usd": frame["amount_usd"].to_numpy(),
            "score": score,  # unrounded: rounding can drop a score below the threshold
            "rules": fired_names(fired).to_numpy(),
            "model_version": bundle.version,
        }
    )
    out["typologies"] = [
        ",".join(sorted({typ[r] for r in rs.split(",") if r})) for rs in out["rules"]
    ]
    high = score >= bundle.threshold
    out["is_alert"] = high | fired.any(axis=1).to_numpy()
    out["priority"] = np.where(high, "high", np.where(out["is_alert"], "low", "none"))
    if with_reasons:
        alert_idx = np.flatnonzero(out["is_alert"].to_numpy())
        texts = [""] * len(out)
        if len(alert_idx):
            for i, rs in zip(alert_idx, reasons(bundle, frame.iloc[alert_idx]), strict=True):
                texts[i] = json.dumps([r.as_dict() for r in rs])
        out["reasons"] = texts
    return out
