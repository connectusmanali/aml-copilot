"""The agent's output: a structured case file in which every claim points at evidence.

Evidence ids are typed so a reader (and the validator) knows where each one came from:
    alert:<txn_id>   the alert itself (score, rules, model reasons)
    acct:<account>   KYC profile
    txn:<txn_id>     a transaction record
    wl:<entry_id>    a watchlist entry
    net:<account>    a network summary computed by the graph tool
    pol:<chunk_id>   a passage of policy guidance (supports reasoning, never a fact)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AGENT_VERSION = "agent-v1"
Severity = Literal["low", "medium", "high"]
Disposition = Literal["escalate", "review", "close_candidate"]


class Evidence(BaseModel):
    id: str = Field(pattern=r"^(alert|acct|txn|wl|net|pol):.+$")
    kind: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class RiskFactor(BaseModel):
    statement: str = Field(min_length=10)
    evidence_ids: list[str] = Field(min_length=1)
    severity: Severity
    typology: str


class Budget(BaseModel):
    tool_calls: int = 0
    max_tool_calls: int
    tokens: int = 0
    max_tokens: int
    seconds: float = 0.0
    max_seconds: float
    breached: str | None = None


class CaseFile(BaseModel):
    case_id: str
    alert_txn_id: str
    subject_account: str
    status: Literal["complete", "needs_human"]
    evidence: list[Evidence]
    risk_factors: list[RiskFactor]
    typology_match: list[str]
    open_questions: list[str]
    suggested_disposition: Disposition
    analysis_mode: Literal["rules", "llm", "llm_fallback_to_rules"]
    tools_called: list[str]
    budget: Budget
    versions: dict[str, str]

    @model_validator(mode="after")
    def _evidence_exists(self) -> CaseFile:
        """Every risk factor must cite evidence that is actually in the case file."""
        known = {e.id for e in self.evidence}
        for rf in self.risk_factors:
            missing = [i for i in rf.evidence_ids if i not in known]
            if missing:
                raise ValueError(f"risk factor cites unknown evidence {missing}: {rf.statement}")
        return self

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}
