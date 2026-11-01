"""Case business logic, shared by the Temporal activities and the investigator API.

Each method is idempotent (safe to retry): it checks the case's current state first, so a
retried activity or a double-clicked button does not do the work twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aml.agent.casefile import CaseFile
from aml.agent.data import CaseData
from aml.agent.graph import AgentConfig, InvestigationAgent
from aml.cases.store import CaseStore, DecisionRefused
from aml.gateway.gateway import LLMGateway
from aml.ops.tracing import traced
from aml.report.draft import DRAFTER_VERSION, draft_report, edit_distance
from aml.report.validate import validate


@dataclass
class CaseService:
    store: CaseStore
    data: CaseData
    gateway: LLMGateway | None = None
    agent_config: AgentConfig | None = None
    sla_hours: float = 48.0

    # ---- workflow steps ----
    @traced("case.open")
    def open(self, alert: dict[str, Any]) -> str:
        cid = self.store.open_case(alert, self.sla_hours)
        case = self.store.get_case(cid)
        assert case is not None
        if case["status"] == "new":
            self.store.transition(
                cid,
                "scored",
                "system:scorer",
                {
                    "score": float(alert["score"]),
                    "priority": alert["priority"],
                    "model_version": alert.get("model_version"),
                },
            )
        return cid

    @traced("case.light_review")
    def light_review(self, case_id: str) -> str:
        """Low-priority alerts (a rule fired, the model score is low) skip the agent."""
        self.store.transition(case_id, "in_review", "system:workflow", {"route": "light"})
        return "in_review"

    @traced("case.investigate")
    def investigate(self, case_id: str) -> str:
        case = self._case(case_id)
        if case["status"] in {"scored"}:
            self.store.transition(case_id, "investigating", "system:workflow", {"route": "agent"})
        existing = self.store.latest_case_file(case_id)
        if existing is None:
            agent = InvestigationAgent(self.data, self.gateway, config=self.agent_config)
            cf = agent.investigate(case["txn_id"], case_id)
            self.store.save_case_file(case_id, cf.model_dump(mode="json"), cf.versions)
            return cf.status
        return str(existing["payload"]["status"])

    @traced("case.draft")
    def draft(self, case_id: str) -> str:
        case = self._case(case_id)
        if case["status"] not in {"investigating", "drafted"}:
            return str(case["status"])  # already past this step
        cf = self.case_file(case_id)
        if not any(r["kind"] == "draft" for r in self.store.reports(case_id)):
            result = draft_report(cf, self.gateway)
            self.store.save_report(
                case_id,
                "draft",
                result.text,
                {
                    **result.validation.as_dict(),
                    "mode": result.mode,
                    "attempts": result.attempts,
                    "prompt_version": result.prompt_version,
                },
                f"agent:{DRAFTER_VERSION}",
            )
            ok = result.validation.ok
        else:
            ok = bool(self.store.reports(case_id)[-1]["validation"].get("ok"))
        if cf.status == "needs_human" or not ok:
            why = "investigation incomplete" if cf.status == "needs_human" else "draft invalid"
            self.store.transition(case_id, "needs_human", "system:workflow", {"why": why})
            return "needs_human"
        self.store.transition(case_id, "drafted", "system:workflow", {})
        self.store.transition(case_id, "in_review", "system:workflow", {})
        return "in_review"

    @traced("case.escalate")
    def escalate(self, case_id: str) -> str:
        case = self._case(case_id)
        if case["status"] in {"in_review", "needs_human"}:
            self.store.transition(
                case_id, "escalated", "system:sla", {"why": "SLA breached without a decision"}
            )
        return str(self._case(case_id)["status"])

    # ---- investigator actions ----
    def edit_report(self, case_id: str, body: str, user: str) -> dict[str, Any]:
        v = validate(body, self.case_file(case_id)).as_dict()
        version = self.store.save_report(case_id, "edit", body, v, f"user:{user}")
        return {"version": version, "validation": v}

    def approve_report(self, case_id: str, user: str, role: str) -> dict[str, Any]:
        if role not in {"investigator", "senior"}:
            raise DecisionRefused(f"role {role!r} may not approve reports")
        rows = [r for r in self.store.reports(case_id) if r["kind"] in {"draft", "edit"}]
        if not rows:
            raise DecisionRefused("no report to approve")
        latest = rows[-1]
        v = validate(latest["body"], self.case_file(case_id))
        if not v.ok:
            raise DecisionRefused(f"report fails validation: {[i.problem for i in v.issues]}")
        version = self.store.save_report(
            case_id, "final", latest["body"], v.as_dict(), f"user:{user}"
        )
        return {"version": version}

    @traced("case.decide")
    def decide(self, case_id: str, decision: str, reason: str, user: str, role: str) -> str:
        rows = self.store.reports(case_id)
        draft = next((r["body"] for r in rows if r["kind"] == "draft"), None)
        final = next((r["body"] for r in reversed(rows) if r["kind"] == "final"), None)
        dist = edit_distance(draft, final) if draft and final else None
        self.store.record_decision(case_id, decision, reason, user, role, dist)
        return str(self._case(case_id)["status"])

    # ---- helpers ----
    def case_file(self, case_id: str) -> CaseFile:
        row = self.store.latest_case_file(case_id)
        if row is None:
            raise KeyError(f"{case_id} has no case file")
        return CaseFile.model_validate(row["payload"])

    def _case(self, case_id: str) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return case
