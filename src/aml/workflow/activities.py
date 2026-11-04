"""Temporal activities: thin wrappers over CaseService.

Activities do the I/O (database, agent, LLM); the workflow only decides what happens next.
Each activity is idempotent, so Temporal's automatic retries are always safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from aml.cases.service import CaseService
from aml.cases.store import DecisionRefused, IllegalTransition


@dataclass
class DecisionInput:
    case_id: str
    decision: str
    reason: str
    user: str
    role: str


class CaseActivities:
    def __init__(self, service: CaseService) -> None:
        self.service = service

    @activity.defn(name="open_case")
    def open_case(self, alert: dict[str, Any]) -> str:
        return self.service.open(alert)

    @activity.defn(name="light_review")
    def light_review(self, case_id: str) -> str:
        return self._safe(self.service.light_review, case_id)

    @activity.defn(name="investigate")
    def investigate(self, case_id: str) -> str:
        return self.service.investigate(case_id)

    @activity.defn(name="draft_report")
    def draft_report(self, case_id: str) -> str:
        return self.service.draft(case_id)

    @activity.defn(name="escalate")
    def escalate(self, case_id: str) -> str:
        return self.service.escalate(case_id)

    @activity.defn(name="record_decision")
    def record_decision(self, d: DecisionInput) -> str:
        return self._safe(self.service.decide, d.case_id, d.decision, d.reason, d.user, d.role)

    @staticmethod
    def _safe(fn: Any, *args: Any) -> str:
        """Business-rule refusals are not transient: tell Temporal not to retry them."""
        try:
            return str(fn(*args))
        except (DecisionRefused, IllegalTransition) as e:
            raise ApplicationError(str(e), type=type(e).__name__, non_retryable=True) from e

    def all(self) -> list[Any]:
        return [
            self.open_case,
            self.light_review,
            self.draft_report,
            self.escalate,
            self.record_decision,
        ]
