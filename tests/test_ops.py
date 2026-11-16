"""Step 9: tracing, drift (PSI), shadow/canary scoring, and exact case replay."""

import numpy as np
import pandas as pd
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aml.agent.graph import InvestigationAgent
from aml.detection.model import ModelBundle
from aml.detection.train import train_scorer
from aml.gateway.gateway import LLMGateway
from aml.ops import tracing
from aml.ops.drift import drift_report, level, psi
from aml.ops.replay import diff, replay_case
from aml.ops.shadow import canary_bucket, shadow
from tests.test_agent import ScriptedProvider, first, world  # noqa: F401
from tests.test_detection import frame, trained  # noqa: F401


def test_psi_levels() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.01
    assert level(psi(ref, rng.normal(0.45, 1, 20_000))) == "watch"
    assert level(psi(ref, rng.normal(1.5, 1, 20_000))) == "act"
    assert psi(np.zeros(100), np.zeros(100)) < 0.01  # constant features do not crash


def test_drift_report(trained: tuple[ModelBundle, dict], frame: pd.DataFrame) -> None:  # noqa: F811
    bundle, _ = trained
    half = frame["ts"].min() + (frame["ts"].max() - frame["ts"].min()) / 2
    ref, cur = frame[frame["ts"] < half], frame[frame["ts"] >= half].copy()
    shifted = cur.copy()
    col = bundle.features[0]
    shifted[col] = shifted[col] * 5 + 100
    rep = drift_report(
        ref,
        shifted,
        bundle.features,
        bundle.predict(ref),
        bundle.predict(shifted),
        bundle.threshold,
    )
    assert rep.features.set_index("feature").loc[col, "level"] == "act"
    assert "Drift report" in rep.render()


def test_shadow_and_canary(trained: tuple[ModelBundle, dict], frame: pd.DataFrame) -> None:  # noqa: F811
    champion, _ = trained
    # a challenger tuned for a much larger alert budget: similar ranking, very different volume
    challenger, _ = train_scorer(frame, alerts_per_day=2_000, seed=7)
    out, rep = shadow(frame, champion, challenger, frame["is_laundering"].astype(int))
    assert len(out) == len(frame) and rep.rank_corr > 0.8
    assert rep.challenger_alerts > rep.champion_alerts and not rep.promotable()
    _, same = shadow(frame, champion, champion)
    assert same.overlap == 1.0 and same.promotable()
    picks = [canary_bucket(f"acct-{i}", 10) for i in range(5_000)]
    assert 0.07 < np.mean(picks) < 0.13 and canary_bucket("x", 10) == canary_bucket("x", 10)


def test_agent_emits_one_trace_per_case(world: dict) -> None:  # noqa: F811
    exporter = InMemorySpanExporter()
    provider = tracing.init_tracing("test", exporter)
    assert provider is not None
    InvestigationAgent(world["data"]).investigate(first(world, "Wire"))
    provider.force_flush()
    names = [s.name for s in exporter.get_finished_spans()]
    assert {"agent.investigate", "agent.plan", "agent.gather", "agent.finalize"} <= set(names)
    root = next(s for s in exporter.get_finished_spans() if s.name == "agent.investigate")
    assert root.attributes is not None and root.attributes["aml.status"] == "complete"


def test_case_replay_is_exact(world: dict) -> None:  # noqa: F811
    txn = first(world, "Cash")
    stored = InvestigationAgent(world["data"]).investigate(txn).model_dump(mode="json")
    ok, changed = replay_case(stored, world["data"], {})
    assert ok and changed == []
    tampered = {**stored, "suggested_disposition": "close_candidate"}
    assert diff(tampered, stored) == ["suggested_disposition"]


def test_llm_case_replays_from_recorded_calls(world: dict) -> None:  # noqa: F811
    txn = first(world, "Wire")
    recorded: dict[str, str] = {}
    answer = InvestigationAgent(world["data"]).investigate(txn)
    import json

    reply = json.dumps(
        {
            "risk_factors": [rf.model_dump() for rf in answer.risk_factors],
            "typology_match": answer.typology_match,
            "open_questions": answer.open_questions,
            "suggested_disposition": answer.suggested_disposition,
        }
    )
    gw = LLMGateway(
        ScriptedProvider([reply]), recorder=lambda r: recorded.update({r.call_id: r.response})
    )
    stored = InvestigationAgent(world["data"], gw).investigate(txn).model_dump(mode="json")
    # replay: a provider that would fail if called; every answer must come from the record
    ok, changed = replay_case(
        stored, world["data"], recorded, ScriptedProvider([RuntimeError("must not be called")])
    )
    assert ok, changed
