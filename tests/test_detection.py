"""Step 3: rules, training discipline, calibration, explanations, bundle round-trip, API."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aml.data import ibm, synthetic
from aml.data.profiles import build_profiles
from aml.data.validate import validate_transactions
from aml.detection import metrics
from aml.detection.explain import describe, reasons
from aml.detection.model import LABEL_DERIVED, ModelBundle, feature_columns
from aml.detection.rules import evaluate, load_scenarios, rule_metrics
from aml.detection.scoring import score_frame
from aml.detection.service import create_app
from aml.detection.train import time_split, train_scorer
from aml.features.account_state import build_account_state
from aml.features.graph import build_graph_snapshots
from aml.features.pit import training_frame


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    txns, _ = validate_transactions(ibm.normalize(synthetic.generate(800, 16_000, 160, seed=11)))
    customers, accounts = build_profiles(txns)
    acc = accounts.merge(customers, on="customer_id")
    return training_frame(txns, build_account_state(txns), build_graph_snapshots(txns), acc)


@pytest.fixture(scope="session")
def trained(frame: pd.DataFrame) -> tuple[ModelBundle, dict]:
    return train_scorer(frame, alerts_per_day=20)


def test_scenarios_load_and_fire(frame: pd.DataFrame) -> None:
    scenarios = load_scenarios()
    fired = evaluate(frame, scenarios)
    assert list(fired.columns) == [s.id for s in scenarios]
    m = rule_metrics(fired, frame["is_laundering"], days=30)
    structuring = m[m["scenario"] == "structuring"].iloc[0]
    assert structuring["alerts"] > 0 and structuring["precision"] > 0.5


def test_bad_scenario_config_is_refused(tmp_path: Path) -> None:
    (tmp_path / "r.yaml").write_text(
        "scenarios:\n"
        "  - {id: x, typology: t, description: d, all: [{feature: a, op: '~', value: 1}]}"
    )
    with pytest.raises(ValueError, match="unknown operator"):
        load_scenarios(tmp_path / "r.yaml")


def test_time_split_never_trains_on_the_future(frame: pd.DataFrame) -> None:
    s = time_split(frame)
    assert s.train["ts"].max() < s.valid["ts"].min() <= s.valid["ts"].max() < s.test["ts"].min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(frame)


def test_features_never_include_the_label_or_ids(frame: pd.DataFrame) -> None:
    cols = feature_columns(frame)
    assert "is_laundering" not in cols and "txn_id" not in cols
    assert not set(LABEL_DERIVED) & set(feature_columns(frame, ("label_derived",)))


def test_model_beats_rules_at_same_budget(trained: tuple[ModelBundle, dict]) -> None:
    _, rep = trained
    any_rule = next(r for r in rep["rules"] if r["scenario"] == "ANY")
    m = rep["model"]
    assert m["precision_at_rules_budget"] >= any_rule["precision"]
    assert m["recall_at_rules_budget"] >= any_rule["recall"]
    assert m["roc_auc"] > 0.9


def test_calibration_is_monotone_and_honest(trained: tuple[ModelBundle, dict]) -> None:
    bundle, rep = trained
    assert np.all(np.diff(bundle.calib_y) >= 0)
    assert rep["model"]["ece"] < 0.05


def test_bundle_round_trip_reproduces_scores(
    trained: tuple[ModelBundle, dict], frame: pd.DataFrame, tmp_path: Path
) -> None:
    bundle, _ = trained
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    assert loaded.version == bundle.version
    np.testing.assert_array_equal(loaded.predict(frame), bundle.predict(frame))


def test_reasons_explain_the_score(trained: tuple[ModelBundle, dict], frame: pd.DataFrame) -> None:
    bundle, _ = trained
    rows = frame[frame["is_laundering"]].head(5)
    contrib = bundle.contributions(rows)
    raw = bundle.raw_probability(rows)
    # SHAP values + bias add up to the model's log-odds: the explanation is exact.
    np.testing.assert_allclose(1 / (1 + np.exp(-contrib.sum(axis=1))), raw, rtol=1e-6)
    rs = reasons(bundle, rows, top=3)
    assert all(len(r) <= 3 for r in rs)
    assert rs[0][0].text and rs[0][0].direction in {"raises risk", "lowers risk"}
    assert describe("s_out_cnt_24h").startswith("Sender:")


def test_score_frame_priorities(trained: tuple[ModelBundle, dict], frame: pd.DataFrame) -> None:
    bundle, _ = trained
    scored = score_frame(frame.head(3_000), bundle)
    high = scored[scored["priority"] == "high"]
    assert (high["score"] >= bundle.threshold).all()
    low = scored[scored["priority"] == "low"]
    assert (low["rules"] != "").all()
    assert (scored.loc[scored["is_alert"], "reasons"] != "").all()


def test_capacity_threshold() -> None:
    score = np.linspace(0, 1, 1_000)
    t = metrics.threshold_for_capacity(score, days=10, alerts_per_day=5)
    assert (score >= t).sum() == 50


def test_scoring_api(trained: tuple[ModelBundle, dict], frame: pd.DataFrame) -> None:
    bundle, _ = trained
    client = TestClient(create_app(bundle))
    assert client.get("/health").json()["model"] == bundle.version
    rows = frame.head(3).assign(ts=frame["ts"].head(3).astype(str)).to_dict(orient="records")
    body = client.post("/score", json={"rows": rows}).json()
    assert len(body["results"]) == 3 and "score" in body["results"][0]
    bad = client.post("/score", json={"rows": [{"txn_id": "x"}]})
    assert bad.status_code == 422
