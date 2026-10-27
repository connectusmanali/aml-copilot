"""The investigation agent as an explicit LangGraph state machine.

    plan -> gather -> analyze -> finalize
                \\-> (budget breached) -> finalize as "needs_human"

A FIXED graph, not a free-form "think, call any tool, repeat" loop: every run takes the same
path, the tool plan is computed from the alert's typologies, and each node is testable alone.
The model is used where judgement helps (analysis); control flow stays deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from aml.agent.analyzer import Analysis, LLMAnalyzer, RuleAnalyzer, load_prompt
from aml.agent.casefile import AGENT_VERSION, Budget, CaseFile, Evidence
from aml.agent.data import CaseData
from aml.agent.policy import PolicyIndex, TfidfPolicyIndex
from aml.agent.tools import ToolBudgetExceeded, ToolRunner
from aml.gateway.gateway import Budget as TokenBudget
from aml.gateway.gateway import BudgetExceeded, LLMGateway
from aml.ops.tracing import span, traced

POLICY_QUERIES = {
    "structuring": "cash just under reporting threshold split payments",
    "fan_out": "payments to many accounts money mules payroll false positive",
    "layering": "funds return through intermediaries cycle pass-through",
    "unusual_activity": "activity above declared expected volume spike",
    "high_risk_customer": "high risk customer sanctions screening",
}


class State(TypedDict, total=False):
    case_id: str
    txn_id: str
    subject: str
    when: datetime
    plan: list[tuple[str, dict[str, Any]]]
    evidence: list[Evidence]
    breached: str | None
    analysis: Analysis
    mode: str
    case_file: CaseFile


@dataclass
class AgentConfig:
    max_tool_calls: int = 12
    max_seconds: float = 60.0
    max_tokens: int = 20_000


class InvestigationAgent:
    def __init__(
        self,
        data: CaseData,
        gateway: LLMGateway | None = None,
        policy: PolicyIndex | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.data, self.gateway = data, gateway
        self.policy = policy or TfidfPolicyIndex()
        self.config = config or AgentConfig()
        self.graph = self._build()

    # ---------------- nodes ----------------
    def _plan(self, state: State) -> State:
        alert = self.data.alert(state["txn_id"])
        if alert is None:
            raise KeyError(f"no alert for transaction {state['txn_id']}")
        subject = str(alert["from_account"])
        typologies = [t for t in str(alert.get("typologies") or "").split(",") if t]
        plan: list[tuple[str, dict[str, Any]]] = [
            ("get_alert", {"txn_id": state["txn_id"]}),
            ("get_customer_profile", {"account_id": subject}),
            ("get_transactions", {"account_id": subject}),
            ("explore_network", {"account_id": subject}),
            ("screen_watchlist", {"account_ids": [subject, str(alert["to_account"])]}),
        ]
        for t in typologies or ["unusual_activity"]:
            if t in POLICY_QUERIES:
                plan.append(("search_policy", {"query": POLICY_QUERIES[t]}))
        return {
            "subject": subject,
            "when": datetime.fromisoformat(str(alert["ts"])),
            "plan": plan,
            "evidence": [],
            "breached": None,
        }

    def _gather(self, state: State) -> State:
        runner = self._runner
        evidence: dict[str, Evidence] = {}
        try:
            for name, kwargs in state["plan"]:
                if name in {"get_transactions", "explore_network"}:
                    kwargs = {**kwargs, "around": state["when"]}
                for e in getattr(runner, name)(**kwargs):
                    evidence.setdefault(e.id, e)
            counterparties = sorted(
                {
                    e.data["to_account"]
                    for e in evidence.values()
                    if e.kind == "transaction" and e.data.get("from_account") == state["subject"]
                }
            )[:20]
            if counterparties:  # screen who the subject paid, too
                for e in runner.screen_watchlist(counterparties):
                    evidence.setdefault(e.id, e)
            breached = None
        except ToolBudgetExceeded as e:
            breached = str(e)
        return {"evidence": list(evidence.values()), "breached": breached}

    def _analyze(self, state: State) -> State:
        analyzer: RuleAnalyzer | LLMAnalyzer = (
            LLMAnalyzer(self.gateway, self._tokens, state["case_id"])
            if self.gateway is not None and self.gateway.provider.name != "offline"
            else RuleAnalyzer()
        )
        try:
            analysis = analyzer.analyze(state["subject"], state["txn_id"], state["evidence"])
        except BudgetExceeded as e:
            return {
                "breached": str(e),
                "analysis": RuleAnalyzer().analyze(
                    state["subject"], state["txn_id"], state["evidence"]
                ),
                "mode": "rules",
            }
        mode = analyzer.mode
        if isinstance(analyzer, LLMAnalyzer) and analyzer.used_fallback:
            mode = "llm_fallback_to_rules"
        return {"analysis": analysis, "mode": mode}

    def _finalize(self, state: State) -> State:
        analysis = state.get("analysis") or RuleAnalyzer().analyze(
            state["subject"], state["txn_id"], state.get("evidence", [])
        )
        breached = state.get("breached")
        questions = list(analysis.open_questions)
        if breached:
            questions.insert(0, f"Investigation incomplete: {breached}")
        cf = CaseFile(
            case_id=state["case_id"],
            alert_txn_id=state["txn_id"],
            subject_account=state["subject"],
            status="needs_human" if breached else "complete",
            evidence=state.get("evidence", []),
            risk_factors=analysis.risk_factors,
            typology_match=analysis.typology_match,
            open_questions=questions,
            suggested_disposition=analysis.suggested_disposition,
            analysis_mode=state.get("mode", "rules"),
            tools_called=list(self._runner.calls),
            budget=Budget(
                tool_calls=len(self._runner.calls),
                max_tool_calls=self.config.max_tool_calls,
                tokens=self._tokens.used_tokens,
                max_tokens=self.config.max_tokens,
                seconds=round(time.monotonic() - self._runner.started, 3),
                max_seconds=self.config.max_seconds,
                breached=breached,
            ),
            versions={
                "agent": AGENT_VERSION,
                "prompt": load_prompt("analyze_v1")[2],
                "llm": self.gateway.provider.model if self.gateway else "none",
                "scorer": str((self.data.alert(state["txn_id"]) or {}).get("model_version")),
            },
        )
        return {"case_file": cf}

    def _build(self) -> Any:
        g = StateGraph(State)
        g.add_node("plan", traced("agent.plan")(self._plan))
        g.add_node("gather", traced("agent.gather")(self._gather))
        g.add_node("analyze", traced("agent.analyze")(self._analyze))
        g.add_node("finalize", traced("agent.finalize")(self._finalize))
        g.set_entry_point("plan")
        g.add_edge("plan", "gather")
        g.add_conditional_edges(
            "gather",
            lambda s: "finalize" if s.get("breached") else "analyze",
            {"analyze": "analyze", "finalize": "finalize"},
        )
        g.add_edge("analyze", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def investigate(self, txn_id: str, case_id: str | None = None) -> CaseFile:
        self._runner = ToolRunner(
            self.data, self.policy, self.config.max_tool_calls, self.config.max_seconds
        )
        self._tokens = TokenBudget(max_tokens=self.config.max_tokens)
        cid = case_id or f"case-{txn_id}"
        with span("agent.investigate", **{"aml.case_id": cid, "aml.txn_id": txn_id}) as sp:
            out = self.graph.invoke({"case_id": cid, "txn_id": txn_id})
            result: CaseFile = out["case_file"]
            sp.set_attribute("aml.status", result.status)
            sp.set_attribute("aml.disposition", result.suggested_disposition)
            sp.set_attribute("aml.tool_calls", len(result.tools_called))
        return result
