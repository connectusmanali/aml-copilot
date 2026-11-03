"""Train, calibrate and evaluate the risk scorer, against the rules baseline.

Discipline (each point is something a model validator checks):
  - Time-based split (train on earlier days, test on later ones), never random: a random split
    lets the model learn from the future of the very laundering chains it is tested on.
  - Class imbalance handled with weights, not by inventing synthetic positives.
  - Early stopping on the validation period; the test period is touched once, at the end.
  - Calibration (isotonic) fitted on validation, so a score of 0.8 means ~80%.
  - Threshold chosen from investigator CAPACITY (alerts per day), not from a metric.
  - Compared with the rules at the same alert budget, and ablated: the label-derived graph
    features and the KYC features are each removed to show how much the model leans on them.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from aml.detection import metrics
from aml.detection.model import ModelBundle, feature_columns
from aml.detection.rules import evaluate, load_scenarios, rule_metrics

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}
MAX_ROUNDS = 1_000


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @staticmethod
    def days(frame: pd.DataFrame) -> float:
        span = frame["ts"].max() - frame["ts"].min()
        return max(float(span / pd.Timedelta(days=1)), 1.0)


def time_split(frame: pd.DataFrame, train: float = 0.6, valid: float = 0.2) -> Split:
    """Split by calendar time: first 60% of the period, next 20%, last 20%."""
    ts = pd.to_datetime(frame["ts"])
    start, end = ts.min(), ts.max()
    t1 = start + (end - start) * train
    t2 = start + (end - start) * (train + valid)
    return Split(frame[ts < t1], frame[(ts >= t1) & (ts < t2)], frame[ts >= t2])


def _fit(split: Split, features: list[str], seed: int) -> lgb.Booster:
    y_tr = split.train["is_laundering"].astype(int).to_numpy()
    pos = max(int(y_tr.sum()), 1)
    params = {**PARAMS, "seed": seed, "scale_pos_weight": (len(y_tr) - pos) / pos}
    train_set = lgb.Dataset(split.train[features].to_numpy(np.float64), y_tr, feature_name=features)
    valid_set = lgb.Dataset(
        split.valid[features].to_numpy(np.float64),
        split.valid["is_laundering"].astype(int).to_numpy(),
        reference=train_set,
    )
    return lgb.train(
        params,
        train_set,
        MAX_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def _data_hash(frame: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(frame[["txn_id", "is_laundering"]], index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:8]


def train_scorer(
    frame: pd.DataFrame,
    *,
    alerts_per_day: int = 100,
    exclude_groups: tuple[str, ...] = (),
    seed: int = 42,
) -> tuple[ModelBundle, dict[str, Any]]:
    split = time_split(frame)
    features = feature_columns(frame, exclude_groups)
    booster = _fit(split, features, seed)

    raw_valid = booster.predict(split.valid[features].to_numpy(np.float64))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        raw_valid, split.valid["is_laundering"].astype(int)
    )
    version = f"scorer-{datetime.now():%Y%m%d}-{_data_hash(frame)}" + (
        f"-no-{'-'.join(exclude_groups)}" if exclude_groups else ""
    )
    bundle = ModelBundle(
        version=version,
        features=features,
        booster=booster,
        calib_x=[float(x) for x in iso.X_thresholds_],
        calib_y=[float(y) for y in iso.y_thresholds_],
        threshold=0.0,
        meta={},
    )
    p_valid = bundle.predict(split.valid)
    bundle.threshold = metrics.threshold_for_capacity(
        p_valid, Split.days(split.valid), alerts_per_day
    )

    report = evaluate_on_test(bundle, split, alerts_per_day)
    bundle.meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows": {"train": len(split.train), "valid": len(split.valid), "test": len(split.test)},
        "periods": {
            name: [str(part["ts"].min()), str(part["ts"].max())]
            for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test))
        },
        "best_iteration": booster.best_iteration,
        "params": PARAMS,
        "alerts_per_day": alerts_per_day,
        "excluded_feature_groups": list(exclude_groups),
        "test": report["model"],
    }
    return bundle, report


def evaluate_on_test(bundle: ModelBundle, split: Split, alerts_per_day: int) -> dict[str, Any]:
    test = split.test
    y = test["is_laundering"].astype(int).to_numpy()
    p = bundle.predict(test)
    days = Split.days(test)

    fired = evaluate(test, load_scenarios())
    rules = rule_metrics(fired, test["is_laundering"], days)
    any_rule = rules[rules["scenario"] == "ANY"].iloc[0]
    k = int(any_rule["alerts"])  # same alert budget as the rules

    at_threshold = p >= bundle.threshold
    tp = int((at_threshold & (y == 1)).sum())
    model = {
        **metrics.summary(y, p),
        "threshold": bundle.threshold,
        "alerts": int(at_threshold.sum()),
        "alerts_per_day": round(at_threshold.sum() / days, 1),
        "precision_at_threshold": round(tp / max(at_threshold.sum(), 1), 4),
        "recall_at_threshold": round(tp / max(y.sum(), 1), 4),
        "precision_at_rules_budget": round(metrics.precision_at_k(y, p, k), 4),
        "recall_at_rules_budget": round(metrics.recall_at_k(y, p, k), 4),
        "investigator_hours": metrics.hours(int(at_threshold.sum())),
    }
    return {
        "model": model,
        "rules": rules.to_dict(orient="records"),
        "rules_budget": k,
        "rules_investigator_hours": metrics.hours(k),
        "test_days": round(days, 1),
    }


def ablations(frame: pd.DataFrame, alerts_per_day: int, seed: int = 42) -> list[dict[str, Any]]:
    """Retrain without each feature group; the drop in PR-AUC is how much the model needs it."""
    out = []
    for groups in [(), ("label_derived",), ("kyc",), ("label_derived", "kyc")]:
        _, rep = train_scorer(
            frame, alerts_per_day=alerts_per_day, exclude_groups=groups, seed=seed
        )
        m = rep["model"]
        out.append(
            {
                "without": ", ".join(groups) or "(full model)",
                "pr_auc": round(m["pr_auc"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "precision_at_rules_budget": m["precision_at_rules_budget"],
                "recall_at_rules_budget": m["recall_at_rules_budget"],
            }
        )
    return out


def render_report(report: dict[str, Any], abl: list[dict[str, Any]], bundle: ModelBundle) -> str:
    m = report["model"]
    lines = [
        f"# Scorer evaluation: {bundle.version}",
        "",
        f"Test period: {bundle.meta['periods']['test'][0]} to {bundle.meta['periods']['test'][1]}"
        f" ({report['test_days']} days, {m['n']:,} transactions, {m['positives']:,} laundering).",
        "",
        "## Model vs rules at the same alert budget",
        "",
        "| | Alerts | Precision | Recall | Investigator hours |",
        "| --- | --- | --- | --- | --- |",
    ]
    any_rule = next(r for r in report["rules"] if r["scenario"] == "ANY")
    lines.append(
        f"| Rules (any scenario) | {any_rule['alerts']:,} | {any_rule['precision']:.1%} | "
        f"{any_rule['recall']:.1%} | {report['rules_investigator_hours']:,} |"
    )
    lines.append(
        f"| Model, top {report['rules_budget']:,} | {report['rules_budget']:,} | "
        f"{m['precision_at_rules_budget']:.1%} | {m['recall_at_rules_budget']:.1%} | "
        f"{report['rules_investigator_hours']:,} |"
    )
    lines.append(
        f"| Model at capacity threshold ({bundle.meta['alerts_per_day']}/day) | {m['alerts']:,} | "
        f"{m['precision_at_threshold']:.1%} | {m['recall_at_threshold']:.1%} | "
        f"{m['investigator_hours']:,} |"
    )
    lines += [
        "",
        f"ROC-AUC {m['roc_auc']:.4f} · PR-AUC {m['pr_auc']:.4f} · "
        f"calibration error (ECE) {m['ece']:.4f} · threshold {m['threshold']:.4f}",
        "",
        "## Rule scenarios (test period)",
        "",
        "| Scenario | Alerts | Alerts/day | Precision | Recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["rules"]:
        lines.append(
            f"| {r['scenario']} | {r['alerts']:,} | {r['alerts_per_day']} | "
            f"{r['precision']:.1%} | {r['recall']:.1%} |"
        )
    lines += [
        "",
        "## Ablation: what the model leans on",
        "",
        "| Model without | PR-AUC | ROC-AUC | Precision @ rules budget | Recall @ rules budget |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a in abl:
        lines.append(
            f"| {a['without']} | {a['pr_auc']:.4f} | {a['roc_auc']:.4f} | "
            f"{a['precision_at_rules_budget']:.1%} | {a['recall_at_rules_budget']:.1%} |"
        )
    lines += [
        "",
        "Synthetic data: these numbers show the method works, not real-world performance. "
        "See docs/model-cards/risk-scorer.md for limitations.",
    ]
    return "\n".join(lines) + "\n"


def log_to_mlflow(
    bundle: ModelBundle, report: dict[str, Any], tracking_uri: str, bundle_dir: str
) -> str | None:
    """Track the run and register the model. Optional: needs `--extra mlflow`."""
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aml-risk-scorer")
    with mlflow.start_run(run_name=bundle.version) as run:
        mlflow.log_params({k: v for k, v in bundle.meta["params"].items()})
        mlflow.log_param("alerts_per_day", bundle.meta["alerts_per_day"])
        mlflow.log_param("features", len(bundle.features))
        flat = {k: v for k, v in report["model"].items() if isinstance(v, int | float)}
        mlflow.log_metrics({k: float(v) for k, v in flat.items() if v == v})
        mlflow.log_artifacts(bundle_dir, artifact_path="bundle")
        mlflow.set_tag("bundle_version", bundle.version)
        # A store without a registry backend still records the run.
        with contextlib.suppress(Exception):
            mlflow.register_model(f"runs:/{run.info.run_id}/bundle", "aml-risk-scorer")
        return str(run.info.run_id)
