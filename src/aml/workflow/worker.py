"""Run the workers and start workflows.

aml-workflow worker            # case worker + agent worker (separate queues)
aml-workflow start --limit 20  # start one CaseWorkflow per scored alert
aml-workflow status --case case-<txn_id>
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from temporalio.client import Client, WorkflowHandle
from temporalio.common import WorkflowIDReusePolicy
from temporalio.worker import Worker

from aml.agent.data import CaseData
from aml.cases.service import CaseService
from aml.workflow.activities import CaseActivities
from aml.workflow.case_workflow import AGENT_QUEUE, MAIN_QUEUE, CaseInput, CaseWorkflow


def workers(
    client: Client,
    service: CaseService,
    agent_concurrency: int = 4,
    main_queue: str = MAIN_QUEUE,
    agent_queue: str = AGENT_QUEUE,
    executor: ThreadPoolExecutor | None = None,
) -> list[Worker]:
    acts = CaseActivities(service)
    pool = executor or ThreadPoolExecutor(max_workers=16)
    return [
        Worker(
            client,
            task_queue=main_queue,
            workflows=[CaseWorkflow],
            activities=acts.all(),
            activity_executor=pool,
        ),
        Worker(
            client,
            task_queue=agent_queue,
            activities=[acts.investigate],
            activity_executor=pool,
            max_concurrent_activities=agent_concurrency,
        ),
    ]


def alert_payload(row: Mapping[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, pd.Timestamp):
            v = v.isoformat(sep=" ")
        elif hasattr(v, "item"):
            v = v.item()
        out[str(k)] = v
    return out


async def start_case(
    client: Client,
    alert: Mapping[Any, Any],
    sla_seconds: float,
    queue: str = MAIN_QUEUE,
    agent_queue: str = AGENT_QUEUE,
) -> WorkflowHandle[Any, Any]:
    return await client.start_workflow(
        CaseWorkflow.run,
        CaseInput(alert_payload(alert), sla_seconds, agent_queue),
        id=f"case-{alert['txn_id']}",
        task_queue=queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
    )


def case_data() -> CaseData:
    """Scored alerts from Parquet when present (dev), else straight from Postgres (prod)."""
    from aml.agent.data import FrameData, PostgresData
    from aml.config import get_settings

    s = get_settings()
    if (s.features_dir / "alerts.parquet").exists():
        return FrameData.from_dirs(s.processed_dir, s.features_dir)
    return PostgresData(s.postgres_dsn)


def build_service() -> CaseService:
    from aml.agent.graph import AgentConfig
    from aml.cases.store import PostgresStore
    from aml.config import get_settings
    from aml.gateway.gateway import build_gateway
    from aml.ops.replay import llm_recorder

    s = get_settings()
    return CaseService(
        PostgresStore(s.postgres_dsn),
        case_data(),
        build_gateway(recorder=llm_recorder(s.postgres_dsn)),
        AgentConfig(s.agent_max_tool_calls, s.agent_max_seconds, s.case_token_budget),
        s.case_sla_hours,
    )


async def _main(args: argparse.Namespace) -> None:
    from temporalio.service import RPCError

    from aml.config import get_settings

    s = get_settings()
    client = await Client.connect(s.temporal_host)
    if args.cmd == "worker":
        ws = workers(client, build_service(), args.agent_concurrency)
        print(f"workers polling {MAIN_QUEUE} and {AGENT_QUEUE} on {s.temporal_host}")
        await asyncio.gather(*(w.run() for w in ws))
    elif args.cmd == "start":
        alerts = pd.read_parquet(s.features_dir / "alerts.parquet")
        alerts = alerts[alerts["priority"].isin(["high", "low"])]
        alerts = alerts.sort_values("score", ascending=False).head(args.limit)
        started = 0
        for row in alerts.to_dict("records"):
            try:
                await start_case(client, row, s.case_sla_hours * 3600)
                started += 1
            except RPCError:
                pass  # already started: one workflow per alert
        print(f"started {started} case workflows ({len(alerts) - started} already running)")
    elif args.cmd == "status":
        st = await client.get_workflow_handle(args.case).query(CaseWorkflow.status)
        print(st)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml-workflow")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--agent-concurrency", type=int, default=4)
    st = sub.add_parser("start")
    st.add_argument("--limit", type=int, default=20)
    q = sub.add_parser("status")
    q.add_argument("--case", required=True)
    asyncio.run(_main(p.parse_args(argv)))


if __name__ == "__main__":
    main()
