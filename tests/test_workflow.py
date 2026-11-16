"""Step 5: the case workflow on a real (local) Temporal server, plus store and audit rules."""

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.testing import WorkflowEnvironment

from aml.cases.audit import verify
from aml.cases.service import CaseService
from aml.cases.store import DecisionRefused, IllegalTransition, MemoryStore
from aml.workflow.activities import DecisionInput
from aml.workflow.case_workflow import CaseWorkflow
from aml.workflow.worker import start_case, workers
from tests.test_agent import first, world  # noqa: F401  (fixture reuse)


def alert_row(world: dict, fmt: str, priority: str = "high") -> dict[str, Any]:  # noqa: F811
    txn = first(world, fmt)
    a = world["alerts"].set_index("txn_id").loc[txn].to_dict()
    return {**a, "txn_id": txn, "priority": priority, "ts": str(a["ts"])}


# ---------------- store rules (no Temporal) ----------------
def test_state_machine_and_decision_rules(world: dict) -> None:  # noqa: F811
    svc = CaseService(MemoryStore(), world["data"])
    cid = svc.open(alert_row(world, "Wire"))
    assert svc.open(alert_row(world, "Wire")) == cid  # idempotent
    with pytest.raises(IllegalTransition):
        svc.store.transition(cid, "filed", "user:x")
    assert svc.investigate(cid) == "complete"
    assert svc.investigate(cid) == "complete"  # retried: no second case file
    assert len(svc.store.case_files[cid]) == 1  # type: ignore[attr-defined]
    assert svc.draft(cid) == "in_review"
    with pytest.raises(DecisionRefused, match="final report"):
        svc.decide(cid, "file_str", "structuring", "alice", "investigator")
    with pytest.raises(DecisionRefused, match="may not decide"):
        svc.decide(cid, "close", "fine", "carol", "auditor")
    with pytest.raises(DecisionRefused, match="written reason"):
        svc.decide(cid, "close", "  ", "alice", "investigator")
    svc.approve_report(cid, "alice", "investigator")
    assert svc.decide(cid, "file_str", "pattern confirmed", "alice", "investigator") == "filed"
    assert svc.store.decision(cid)["edit_distance"] == 0.0  # type: ignore[index]


def test_audit_chain_detects_tampering(world: dict) -> None:  # noqa: F811
    store = MemoryStore()
    svc = CaseService(store, world["data"])
    cid = svc.open(alert_row(world, "Cash"))
    svc.investigate(cid)
    svc.draft(cid)
    assert verify(store.audit()).ok
    from dataclasses import replace

    store.log[2] = replace(store.log[2], details={"route": "rewritten"})
    report = verify(store.audit())
    assert not report.ok and report.first_bad_id == store.log[2].id


# ---------------- Temporal ----------------
def run_with_temporal(test: Callable[[Client, str], Awaitable[None]], service: CaseService) -> None:
    async def go() -> None:
        cli = os.environ.get("TEMPORAL_CLI")
        env = await WorkflowEnvironment.start_local(
            dev_server_existing_path=cli if cli and Path(cli).exists() else None,
            dev_server_extra_args=["--log-level", "error"],
        )
        async with env:
            q = f"cases-{uuid.uuid4().hex[:6]}"
            ws = workers(
                env.client,
                service,
                2,
                main_queue=q,
                agent_queue=f"{q}-agent",
                executor=ThreadPoolExecutor(4),
            )
            async with ws[0], ws[1]:
                await test(env.client, q)

    asyncio.run(go())


async def wait_for(handle: Any, pred: Callable[[Any], bool], seconds: float = 30) -> Any:
    for _ in range(int(seconds * 10)):
        st = await handle.query(CaseWorkflow.status)
        if pred(st):
            return st
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out; last state {st}")


@pytest.mark.temporal
def test_high_priority_case_runs_agent_then_waits_for_human(world: dict) -> None:  # noqa: F811
    store = MemoryStore()
    svc = CaseService(store, world["data"])
    alert = alert_row(world, "Wire")

    async def test(client: Client, q: str) -> None:
        h = await start_case(client, alert, sla_seconds=3600, queue=q, agent_queue=f"{q}-agent")
        with pytest.raises(WorkflowAlreadyStartedError):  # one workflow per alert
            await start_case(client, alert, sla_seconds=3600, queue=q, agent_queue=f"{q}-agent")
        st = await wait_for(h, lambda s: s.status == "in_review")
        assert st.route == "agent"
        await h.signal(
            CaseWorkflow.decide,
            DecisionInput(st.case_id, "file_str", "confirmed", "alice", "investigator"),
        )
        st = await wait_for(h, lambda s: bool(s.refusals))
        assert "final report" in st.refusals[0]  # refused, workflow still waiting
        svc.approve_report(st.case_id, "alice", "investigator")
        await h.signal(
            CaseWorkflow.decide,
            DecisionInput(st.case_id, "file_str", "confirmed", "alice", "investigator"),
        )
        result = await h.result()
        assert result.status == "filed"

    run_with_temporal(test, svc)
    assert verify(store.audit()).ok


@pytest.mark.temporal
def test_sla_breach_escalates_and_needs_a_senior(world: dict) -> None:  # noqa: F811
    store = MemoryStore()
    svc = CaseService(store, world["data"])
    alert = alert_row(world, "ACH", priority="low")

    async def test(client: Client, q: str) -> None:
        h = await start_case(client, alert, sla_seconds=1, queue=q, agent_queue=f"{q}-agent")
        st = await wait_for(h, lambda s: s.escalated)
        assert st.route == "light" and st.status == "escalated"
        await h.signal(
            CaseWorkflow.decide,
            DecisionInput(st.case_id, "close", "payroll", "alice", "investigator"),
        )
        st = await wait_for(h, lambda s: bool(s.refusals))
        assert "senior" in st.refusals[0]
        await h.signal(
            CaseWorkflow.decide, DecisionInput(st.case_id, "close", "payroll", "bob", "senior")
        )
        assert (await h.result()).status == "closed"

    run_with_temporal(test, svc)
    actions = [e.action for e in store.audit()]
    assert "status:in_review->escalated" in actions and actions[-1] == "status:escalated->closed"
