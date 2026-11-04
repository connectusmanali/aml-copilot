"""Exact case replay: re-run an investigation from recorded inputs and compare.

A regulator or model validator asks "why did the system say this on that day?". We re-run the
agent with the gateway in REPLAY mode: every LLM call is answered from cases.llm_calls (the
recorded, masked response for the same call id), so the run is deterministic. If the new case
file matches the stored one, the decision path is proven reproducible; if not, the diff shows
exactly what changed (data, prompt version, code).
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

from aml.agent.data import CaseData
from aml.agent.graph import InvestigationAgent
from aml.gateway.gateway import CallRecord, LLMGateway
from aml.gateway.providers import OfflineProvider, Provider

VOLATILE = {"budget", "versions"}  # timings and versions legitimately differ between runs


def llm_recorder(dsn: str) -> Any:
    def record(rec: CallRecord) -> None:
        with psycopg.connect(dsn) as conn:
            conn.execute(
                "INSERT INTO cases.llm_calls (call_id, case_id, task, provider, model, "
                "prompt_version, response, input_tokens, output_tokens, latency_ms, cost_usd) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (call_id) DO NOTHING",
                (
                    rec.call_id,
                    rec.case_id,
                    rec.task,
                    rec.provider,
                    rec.model,
                    rec.prompt_version,
                    rec.response,
                    rec.input_tokens,
                    rec.output_tokens,
                    rec.latency_ms,
                    rec.cost_usd,
                ),
            )

    return record


def load_replay(dsn: str, case_id: str) -> dict[str, str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT call_id, response FROM cases.llm_calls WHERE case_id = %s", (case_id,)
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k in VOLATILE:
            continue
        if json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(
            b.get(k), sort_keys=True, default=str
        ):
            out.append(k)
    return out


def replay_case(
    stored: dict[str, Any],
    data: CaseData,
    recorded: dict[str, str],
    provider: Provider | None = None,
) -> tuple[bool, list[str]]:
    provider = provider or OfflineProvider()
    gw = LLMGateway(provider, replay=recorded) if provider.name != "offline" else None
    new = InvestigationAgent(data, gw).investigate(stored["alert_txn_id"], stored["case_id"])
    changed = diff(stored, new.model_dump(mode="json"))
    return not changed, changed
