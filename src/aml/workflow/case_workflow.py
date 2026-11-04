"""CaseWorkflow: one durable workflow per alert.

    open_case -> (high priority) investigate -> draft_report -> wait for a human
              -> (low priority)  light_review --------------> wait for a human
    wait: a `decide` signal from the investigator, or the SLA timer fires -> escalate -> wait
    record_decision -> filed | closed

Temporal persists every step, so a crash or deploy mid-case resumes exactly where it stopped;
timers survive restarts; and the full history of each case is replayable.
The agent runs on its own task queue with a concurrency limit, so a burst of alerts queues up
instead of overloading the LLM provider (backpressure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from aml.workflow.activities import DecisionInput

MAIN_QUEUE = "aml-cases"
AGENT_QUEUE = "aml-agent"

DB_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=10,
)
AGENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=4,
    non_retryable_error_types=["KeyError"],
)


@dataclass
class CaseInput:
    alert: dict[str, Any]
    sla_seconds: float = 48 * 3600
    agent_queue: str = AGENT_QUEUE


@dataclass
class CaseState:
    case_id: str = ""
    status: str = "new"
    route: str = ""
    escalated: bool = False
    pending: DecisionInput | None = None
    refusals: list[str] = field(default_factory=list)


@workflow.defn(name="CaseWorkflow")
class CaseWorkflow:
    def __init__(self) -> None:
        self.state = CaseState()

    @workflow.signal
    def decide(self, d: DecisionInput) -> None:
        self.state.pending = d

    @workflow.query
    def status(self) -> CaseState:
        return self.state

    async def _act(
        self,
        name: str,
        arg: Any,
        *,
        queue: str | None = None,
        timeout: timedelta = timedelta(seconds=60),
        retry: RetryPolicy = DB_RETRY,
    ) -> str:
        result: str = await workflow.execute_activity(
            name,
            arg,
            task_queue=queue or workflow.info().task_queue,
            start_to_close_timeout=timeout,
            retry_policy=retry,
        )
        return result

    @workflow.run
    async def run(self, inp: CaseInput) -> CaseState:
        s = self.state
        s.case_id = await self._act("open_case", inp.alert)
        if inp.alert.get("priority") == "high":
            s.route = "agent"
            await self._act(
                "investigate",
                s.case_id,
                queue=inp.agent_queue,
                timeout=timedelta(minutes=5),
                retry=AGENT_RETRY,
            )
            s.status = await self._act(
                "draft_report", s.case_id, timeout=timedelta(minutes=3), retry=AGENT_RETRY
            )
        else:
            s.route = "light"
            s.status = await self._act("light_review", s.case_id)

        sla = timedelta(seconds=inp.sla_seconds)
        while True:
            try:
                await workflow.wait_condition(
                    lambda: s.pending is not None, timeout=None if s.escalated else sla
                )
            except TimeoutError:
                s.escalated = True
                s.status = await self._act("escalate", s.case_id)
                continue
            d, s.pending = s.pending, None
            assert d is not None
            try:
                s.status = await self._act("record_decision", d)
                return s
            except ActivityError as e:
                cause = e.cause
                msg = cause.message if isinstance(cause, ApplicationError) else str(e)
                s.refusals.append(f"{d.user}: {msg}")  # refused decision: keep waiting
