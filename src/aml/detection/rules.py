"""Rules engine: evaluates the YAML scenarios over the training frame, vectorized."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RULES_FILE = Path(__file__).with_name("rules.yaml")
OPS: dict[str, Callable[[Any, Any], Any]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float


@dataclass(frozen=True)
class Scenario:
    id: str
    typology: str
    description: str
    conditions: tuple[Condition, ...]


def load_scenarios(path: Path = RULES_FILE) -> list[Scenario]:
    raw = yaml.safe_load(path.read_text())
    scenarios = []
    for s in raw["scenarios"]:
        conds = tuple(Condition(c["feature"], c["op"], float(c["value"])) for c in s["all"])
        for c in conds:
            if c.op not in OPS:
                raise ValueError(f"scenario {s['id']}: unknown operator {c.op!r}")
        scenarios.append(Scenario(s["id"], s["typology"], s["description"], conds))
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique")
    return scenarios


def evaluate(frame: pd.DataFrame, scenarios: list[Scenario]) -> pd.DataFrame:
    """One boolean column per scenario (True = fired), aligned to frame.index."""
    fired = {}
    for sc in scenarios:
        hit = np.ones(len(frame), dtype=bool)
        for c in sc.conditions:
            if c.feature not in frame:
                raise KeyError(f"scenario {sc.id}: feature {c.feature!r} not in the frame")
            hit &= OPS[c.op](frame[c.feature].fillna(0).to_numpy(), c.value)
        fired[sc.id] = hit
    return pd.DataFrame(fired, index=frame.index)


def fired_names(fired: pd.DataFrame) -> pd.Series:
    """Comma-separated scenario ids per row ('' when none fired)."""
    cols = fired.columns.to_numpy()
    return pd.Series(
        [",".join(cols[row]) for row in fired.to_numpy()], index=fired.index, dtype="object"
    )


def rule_metrics(fired: pd.DataFrame, y: pd.Series, days: float) -> pd.DataFrame:
    """Per scenario and overall: alerts, alerts/day, precision, recall."""
    labels = y.astype(bool).to_numpy()
    rows = []
    for name in [*fired.columns, "ANY"]:
        hit: np.ndarray = fired.any(axis=1).to_numpy() if name == "ANY" else fired[name].to_numpy()
        tp = int((hit & labels).sum())
        rows.append(
            {
                "scenario": name,
                "alerts": int(hit.sum()),
                "alerts_per_day": round(hit.sum() / max(days, 1e-9), 1),
                "precision": round(tp / hit.sum(), 4) if hit.sum() else 0.0,
                "recall": round(tp / labels.sum(), 4) if labels.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)
