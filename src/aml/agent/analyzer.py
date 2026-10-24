"""Turn gathered evidence into risk factors.

RuleAnalyzer  deterministic red-flag checks: the baseline, the offline mode, and the fallback
              whenever the LLM answer is unusable. Every factor cites real evidence by design.
LLMAnalyzer   asks the model (through the gateway) for the same structure, then validates it:
              factors citing unknown evidence are dropped, and hard rules (a sanctions hit
              always escalates) override whatever the model said.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from aml.agent.casefile import Disposition, Evidence, RiskFactor
from aml.gateway.gateway import Budget, LLMGateway

PROMPT_DIR = Path(__file__).with_name("prompts")
NEAR_THRESHOLD = (9_000.0, 10_000.0)


class Analysis(BaseModel):
    risk_factors: list[RiskFactor]
    typology_match: list[str]
    open_questions: list[str]
    suggested_disposition: Disposition


def load_prompt(name: str) -> tuple[str, str, str]:
    """Returns (system, user template, version). Version = name + content hash: a prompt edit
    is a new version, recorded with every call."""
    text = (PROMPT_DIR / f"{name}.md").read_text()
    sections = {
        part.split("\n", 1)[0].strip(): part.split("\n", 1)[1].strip()
        for part in text.split("## ")[1:]
    }
    version = f"{name}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    return sections["system"], sections["user"], version


class Analyzer(Protocol):
    mode: str

    def analyze(self, subject: str, alert_txn_id: str, evidence: list[Evidence]) -> Analysis: ...


def _ts(t: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(t["ts"]))


def _disposition(factors: list[RiskFactor], evidence: list[Evidence]) -> Disposition:
    if any(e.kind == "watchlist" and e.data.get("list") == "sanctions" for e in evidence):
        return "escalate"  # hard rule: a sanctions match is never closed by the system
    if any(f.severity == "high" for f in factors):
        return "escalate"
    if any(f.severity == "medium" for f in factors):
        return "review"
    return "close_candidate"


class RuleAnalyzer:
    mode = "rules"

    def analyze(self, subject: str, alert_txn_id: str, evidence: list[Evidence]) -> Analysis:
        by_kind: dict[str, list[Evidence]] = {}
        for e in evidence:
            by_kind.setdefault(e.kind, []).append(e)
        factors: list[RiskFactor] = []
        questions: list[str] = []
        alert = next((e for e in by_kind.get("alert", [])), None)
        txns = [e for e in by_kind.get("transaction", []) if e.data]
        alert_txn = next((e for e in txns if e.id == f"txn:{alert_txn_id}"), None)
        when = (
            _ts(alert_txn.data)
            if alert_txn
            else (datetime.fromisoformat(str(alert.data["ts"])) if alert else None)
        )
        outs = [e for e in txns if e.data["from_account"] == subject]
        ins = [e for e in txns if e.data["to_account"] == subject]

        def near(e: Evidence, hours: int) -> bool:
            return when is not None and abs(_ts(e.data) - when) <= timedelta(hours=hours)

        # Structuring
        lo, hi = NEAR_THRESHOLD
        cash = [
            e
            for e in outs
            if e.data["payment_format"] == "Cash"
            and near(e, 48)
            and lo <= (e.data.get("amount_usd") or 0) < hi
        ]
        if len(cash) >= 2:
            total = sum(e.data["amount_usd"] for e in cash)
            factors.append(
                RiskFactor(
                    statement=f"{len(cash)} cash payments between USD 9,000 and 9,999 within "
                    f"48 hours, USD {total:,.2f} in total, each just under the reporting threshold",
                    evidence_ids=[e.id for e in cash],
                    severity="high",
                    typology="structuring",
                )
            )
            questions.append("What is the source of the cash, and why was it split?")
        elif len(cash) == 1:
            factors.append(
                RiskFactor(
                    statement="One cash payment just under USD 10,000; no repetition observed",
                    evidence_ids=[cash[0].id],
                    severity="low",
                    typology="structuring",
                )
            )

        # Fan-out
        burst = [e for e in outs if near(e, 24)]
        receivers = {e.data["to_account"] for e in burst}
        if len(receivers) >= 5:
            profile = next(
                (e for e in by_kind.get("profile", []) if e.id == f"acct:{subject}"), None
            )
            payroll_like = (
                profile is not None
                and profile.data.get("segment") == "business"
                and all(e.data["payment_format"] == "ACH" for e in burst)
            )
            factors.append(
                RiskFactor(
                    statement=f"Payments to {len(receivers)} distinct accounts within 24 hours "
                    f"({', '.join(sorted({e.data['payment_format'] for e in burst}))})"
                    + (", consistent with payroll by a business customer" if payroll_like else ""),
                    evidence_ids=[e.id for e in burst] + ([profile.id] if profile else []),
                    severity="medium" if payroll_like else "high",
                    typology="fan_out",
                )
            )
            questions.append("Are the recipients employees, suppliers or new relationships?")

        # Layering: loops back to the subject
        net = next((e for e in by_kind.get("network", []) if e.id == f"net:{subject}"), None)
        if net and net.data.get("cycles"):
            ids = [f"txn:{t}" for cyc in net.data["cycles"] for t in cyc]
            known = {e.id for e in evidence}
            factors.append(
                RiskFactor(
                    statement=f"Funds left the account and returned through intermediaries "
                    f"{len(net.data['cycles'])} time(s) in the window",
                    evidence_ids=[net.id, *[i for i in dict.fromkeys(ids) if i in known]],
                    severity="high",
                    typology="layering",
                )
            )

        # Pass-through
        if when is not None:
            in24 = sum(e.data.get("amount_usd") or 0 for e in ins if near(e, 24))
            out24 = sum(e.data.get("amount_usd") or 0 for e in outs if near(e, 24))
            if in24 >= 10_000 and out24 >= 0.8 * in24:
                factors.append(
                    RiskFactor(
                        statement=f"USD {in24:,.2f} received and USD {out24:,.2f} sent on within "
                        "24 hours (pass-through)",
                        evidence_ids=[e.id for e in ins + outs if near(e, 24)],
                        severity="medium",
                        typology="layering",
                    )
                )

        # KYC mismatch and profile risk
        profile = next((e for e in by_kind.get("profile", []) if e.id == f"acct:{subject}"), None)
        if profile is None:
            questions.append("No KYC profile found for the subject account")
        else:
            declared = float(profile.data.get("expected_monthly_volume") or 0)
            sent = sum(e.data.get("amount_usd") or 0 for e in outs) * 30 / 16  # 16-day window
            if declared > 0 and sent >= 3 * declared and sent >= 10_000:
                factors.append(
                    RiskFactor(
                        statement=f"Outflow runs at about USD {sent:,.0f}/month against USD "
                        f"{declared:,.0f}/month declared at onboarding",
                        evidence_ids=[profile.id, *[e.id for e in outs][:20]],
                        severity="medium",
                        typology="unusual_activity",
                    )
                )
                questions.append("Has the customer's situation changed since onboarding?")
            if profile.data.get("risk_rating") == "high":
                factors.append(
                    RiskFactor(
                        statement=f"Customer is rated high risk ({profile.data.get('occupation')}, "
                        f"{profile.data.get('country')})",
                        evidence_ids=[profile.id],
                        severity="low",
                        typology="high_risk_customer",
                    )
                )
            if any(_has_injection(v) for v in profile.data.values()):
                questions.append(
                    "The customer record contains instruction-like text; it was "
                    "treated as data and ignored"
                )

        # Watchlist
        for e in by_kind.get("watchlist", []):
            lst = e.data.get("list")
            factors.append(
                RiskFactor(
                    statement=(
                        f"{e.data['account_id']} appears on the {lst} list: {e.data['reason']}"
                    ),
                    evidence_ids=[e.id],
                    severity="high" if lst == "sanctions" else "medium",
                    typology="watchlist",
                )
            )

        # Model signal
        if alert is not None and alert.data.get("score", 0) >= 0.5:
            top = ", ".join(r["feature"] for r in (alert.data.get("reasons") or [])[:3])
            factors.append(
                RiskFactor(
                    statement=f"Risk model score {alert.data['score']:.2f}"
                    + (f"; strongest drivers: {top}" if top else ""),
                    evidence_ids=[alert.id],
                    severity="medium",
                    typology="model",
                )
            )

        typologies = list(
            dict.fromkeys(
                f.typology for f in factors if f.severity != "low" and f.typology != "model"
            )
        )
        return Analysis(
            risk_factors=factors,
            typology_match=typologies,
            open_questions=list(dict.fromkeys(questions)),
            suggested_disposition=_disposition(factors, evidence),
        )


def _has_injection(value: object) -> bool:
    from aml.agent.tools import INJECTION

    return isinstance(value, str) and bool(INJECTION.search(value))


class LLMAnalyzer:
    mode = "llm"

    def __init__(self, gateway: LLMGateway, budget: Budget, case_id: str) -> None:
        self.gateway, self.budget, self.case_id = gateway, budget, case_id
        self.system, self.user_tpl, self.prompt_version = load_prompt("analyze_v1")
        self.fallback = RuleAnalyzer()
        self.used_fallback = False

    def analyze(self, subject: str, alert_txn_id: str, evidence: list[Evidence]) -> Analysis:
        payload = [{"id": e.id, "kind": e.kind, "summary": e.summary} for e in evidence]
        user = (
            self.user_tpl.replace("{alert_txn_id}", alert_txn_id)
            .replace("{subject}", subject)
            .replace("{evidence_json}", json.dumps(payload, indent=1))
        )
        known = {e.id for e in evidence}
        for _attempt in range(2):  # one retry on an unusable answer
            text = self.gateway.complete(
                "analyze",
                self.system,
                user,
                prompt_version=self.prompt_version,
                budget=self.budget,
                case_id=self.case_id,
                json_mode=True,
            )
            try:
                raw = json.loads(text[text.find("{") : text.rfind("}") + 1])
                kept, dropped = [], 0
                for rf in raw.get("risk_factors", []):
                    rf["evidence_ids"] = [i for i in rf.get("evidence_ids", []) if i in known]
                    if rf["evidence_ids"]:
                        kept.append(rf)
                    else:
                        dropped += 1
                raw["risk_factors"] = kept
                result = Analysis.model_validate(raw)
            except (ValueError, ValidationError):
                continue
            if dropped:
                result.open_questions.append(
                    f"{dropped} model statement(s) without valid evidence were removed"
                )
            result.suggested_disposition = _strictest(
                result.suggested_disposition, _disposition(result.risk_factors, evidence)
            )
            return result
        self.used_fallback = True
        return self.fallback.analyze(subject, alert_txn_id, evidence)


def _strictest(a: Disposition, b: Disposition) -> Disposition:
    order: list[Disposition] = ["close_candidate", "review", "escalate"]
    return a if order.index(a) >= order.index(b) else b
