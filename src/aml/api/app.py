"""Investigator API.

GET  /cases                         queue, highest score first (any role)
GET  /cases/{id}                    case, case file (evidence), reports, decision
POST /cases/{id}/report             save an edited draft; returns the validator's verdict
POST /cases/{id}/report/approve     freeze the latest draft/edit as final (must validate)
POST /cases/{id}/decision           file_str | close, with a written reason
POST /cases/{id}/feedback           does the investigator agree with the score?
GET  /cases/{id}/audit              the case's audit trail
GET  /audit/verify                  verify the whole hash chain (senior, auditor)

Auth: `X-API-Key` maps to a user and a role (a stand-in for the bank's SSO). Roles are checked
here AND again in the case store, so a bug in one layer cannot let an auditor decide a case.
Decisions go through a `Decider`: directly to the store in dev, or as a signal to the case's
Temporal workflow in production (so the workflow, not the API, owns the case lifecycle).
"""

# No `from __future__ import annotations`: FastAPI must resolve the local Annotated aliases.
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from aml.cases.audit import verify
from aml.cases.service import CaseService
from aml.cases.store import DecisionRefused, IllegalTransition
from aml.config import get_settings

Decider = Callable[[str, str, str, str, str], Awaitable[str]]


@dataclass(frozen=True)
class User:
    id: str
    role: Literal["investigator", "senior", "auditor"]


def parse_keys(spec: str) -> dict[str, User]:
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, _, who = item.partition("=")
        uid, _, role = who.partition(":")
        if role not in {"investigator", "senior", "auditor"}:
            raise ValueError(f"bad role in api_keys: {item}")
        out[key] = User(uid, role)  # type: ignore[arg-type]
    return out


class ReportEdit(BaseModel):
    body: str = Field(min_length=1, max_length=50_000)


class DecisionIn(BaseModel):
    decision: Literal["file_str", "close"]
    reason: str = Field(min_length=1, max_length=4_000)


class FeedbackIn(BaseModel):
    agrees_with_score: bool
    comment: str = Field(default="", max_length=4_000)


def create_app(
    service: CaseService, decider: Decider | None = None, api_keys: str | None = None
) -> FastAPI:
    app = FastAPI(title="AML Investigation Copilot", version="1.0")
    keys = parse_keys(api_keys if api_keys is not None else get_settings().api_keys)
    store = service.store

    async def direct(case_id: str, decision: str, reason: str, user: str, role: str) -> str:
        return service.decide(case_id, decision, reason, user, role)

    decide = decider or direct

    def user(x_api_key: Annotated[str | None, Header()] = None) -> User:
        if x_api_key is None or x_api_key not in keys:
            raise HTTPException(401, "missing or unknown API key")
        return keys[x_api_key]

    def role(*allowed: str) -> Callable[[User], User]:
        def check(u: Annotated[User, Depends(user)]) -> User:
            if u.role not in allowed:
                raise HTTPException(403, f"role {u.role} cannot do this")
            return u

        return check

    Any_ = Annotated[User, Depends(user)]  # noqa: N806  (type aliases)
    Worker = Annotated[User, Depends(role("investigator", "senior"))]  # noqa: N806
    Oversight = Annotated[User, Depends(role("senior", "auditor"))]  # noqa: N806

    def case_or_404(case_id: str) -> dict[str, Any]:
        c = store.get_case(case_id)
        if c is None:
            raise HTTPException(404, "no such case")
        return c

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/me")
    def me(u: Any_) -> dict[str, str]:
        return asdict(u)

    @app.get("/cases")
    def queue(u: Any_, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return store.queue(status.split(",") if status else None, min(limit, 500))

    @app.get("/cases/{case_id}")
    def detail(case_id: str, u: Any_) -> dict[str, Any]:
        c = case_or_404(case_id)
        cf = store.latest_case_file(case_id)
        return {
            "case": c,
            "case_file": cf["payload"] if cf else None,
            "reports": store.reports(case_id),
            "decision": store.decision(case_id),
        }

    @app.post("/cases/{case_id}/report")
    def edit(case_id: str, body: ReportEdit, u: Worker) -> dict[str, Any]:
        case_or_404(case_id)
        try:
            return service.edit_report(case_id, body.body, u.id)
        except KeyError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/cases/{case_id}/report/approve")
    def approve(case_id: str, u: Worker) -> dict[str, Any]:
        case_or_404(case_id)
        try:
            return service.approve_report(case_id, u.id, u.role)
        except (DecisionRefused, KeyError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/cases/{case_id}/decision")
    async def decision(case_id: str, body: DecisionIn, u: Worker) -> dict[str, str]:
        case_or_404(case_id)
        try:
            status = await decide(case_id, body.decision, body.reason, u.id, u.role)
        except (DecisionRefused, IllegalTransition) as e:
            raise HTTPException(409, str(e)) from e
        return {"case_id": case_id, "status": status}

    @app.post("/cases/{case_id}/feedback")
    def feedback(case_id: str, body: FeedbackIn, u: Worker) -> dict[str, str]:
        case_or_404(case_id)
        store.add_feedback(case_id, u.id, body.agrees_with_score, body.comment)
        return {"status": "recorded"}

    @app.get("/cases/{case_id}/audit")
    def case_audit(case_id: str, u: Any_) -> list[dict[str, Any]]:
        return [asdict(e) for e in store.audit(f"case:{case_id}")]

    @app.get("/audit/verify")
    def audit_verify(u: Oversight) -> dict[str, Any]:
        return asdict(verify(store.audit()))

    return app


def temporal_decider(temporal_host: str) -> Decider:
    """Production path: the decision is a signal to the case's workflow."""
    from temporalio.client import Client

    from aml.workflow.activities import DecisionInput
    from aml.workflow.case_workflow import CaseWorkflow

    client: list[Client] = []

    async def decide(case_id: str, decision: str, reason: str, user: str, role: str) -> str:
        if not client:
            client.append(await Client.connect(temporal_host))
        h = client[0].get_workflow_handle(case_id)
        await h.signal(CaseWorkflow.decide, DecisionInput(case_id, decision, reason, user, role))
        return "decision_submitted"

    return decide
