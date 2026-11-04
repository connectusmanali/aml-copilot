"""Draft the STR narrative from the case file, then validate it.

TemplateDrafter  deterministic sentences built from the evidence: always cites, always valid.
                 Used offline, and as the safe degraded mode when the LLM is unavailable.
LLMDrafter       the model writes the narrative from the case file only; the validator checks
                 it; on failure the model gets ONE retry with the validator's findings, then
                 the case goes to a human with the problems highlighted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from aml.agent.casefile import CaseFile, Evidence
from aml.gateway.gateway import Budget, BudgetExceeded, LLMGateway
from aml.report.validate import Validation, validate

DRAFTER_VERSION = "report-v1"
MAX_TXNS_LISTED = 10


@dataclass
class DraftResult:
    text: str
    validation: Validation
    mode: Literal["template", "llm", "llm_needs_human"]
    attempts: int
    prompt_version: str


def _usd(x: Any) -> str:
    return f"USD {float(x):,.2f}"


def _date(ts: Any) -> str:
    return str(ts)[:10]


class TemplateDrafter:
    mode = "template"

    def draft(self, case: CaseFile) -> str:
        ev = case.evidence_by_id()
        subject = case.subject_account
        lines: list[str] = []
        prof = ev.get(f"acct:{subject}")
        alert = ev.get(f"alert:{case.alert_txn_id}")

        lines.append("## Who")
        if prof:
            d = prof.data
            lines.append(
                f"The subject is account {subject}, a {d['segment']} customer "
                f"({d['occupation']}) in {d['country']} rated {d['risk_rating']} risk, who "
                f"declared {_usd(d['expected_monthly_volume'])} of monthly activity at onboarding "
                f"[{prof.id}]."
            )
        else:
            lines.append(
                f"The subject is account {subject}; no KYC profile was found "
                f"[alert:{case.alert_txn_id}]."
            )
        for e in (x for x in case.evidence if x.kind == "watchlist"):
            lines.append(
                f"Account {e.data['account_id']} appears on the {e.data['list']} list [{e.id}]."
            )

        lines.append("")
        lines.append("## What happened")
        cited_txns: list[Evidence] = []
        for rf in case.risk_factors:
            if rf.typology == "model":
                continue
            refs = " ".join(f"[{i}]" for i in rf.evidence_ids)
            lines.append(f"{rf.statement} {refs}.")
            cited_txns += [ev[i] for i in rf.evidence_ids if i.startswith("txn:")]
        if not cited_txns:
            cited_txns = [
                e
                for e in case.evidence
                if e.kind == "transaction" and e.id == f"txn:{case.alert_txn_id}"
            ]
        key_txns = list({e.id: e for e in cited_txns}.values())
        key_txns.sort(key=lambda e: str(e.data["ts"]))
        for e in key_txns[:MAX_TXNS_LISTED]:
            t = e.data
            amount = (
                _usd(t["amount_usd"])
                if t.get("amount_usd") is not None
                else (f"{t['amount_paid']} {t['payment_currency']}")
            )
            lines.append(
                f"On {_date(t['ts'])}, account {t['from_account']} paid account "
                f"{t['to_account']} {amount} by {t['payment_format']} [{e.id}]."
            )

        lines.append("")
        lines.append("## When and where")
        if key_txns:
            first, last = key_txns[0], key_txns[-1]
            banks = sorted(
                {
                    str(e.data[k]).split("-")[0]
                    for e in key_txns
                    for k in ("from_account", "to_account")
                }
            )
            lines.append(
                f"The activity took place between {_date(first.data['ts'])} and "
                f"{_date(last.data['ts'])} [{first.id}] [{last.id}]."
            )
            lines.append(
                f"Funds moved between accounts held at {len(banks)} bank(s) "
                + " ".join(f"[{e.id}]" for e in key_txns[:MAX_TXNS_LISTED])
                + "."
            )

        lines.append("")
        lines.append("## Why it is suspicious")
        names = {
            "structuring": "structuring below the reporting threshold",
            "fan_out": "rapid distribution of funds to many accounts",
            "layering": "layering of funds through intermediary accounts",
            "unusual_activity": "activity inconsistent with the customer's profile",
            "watchlist": "a watchlist match",
            "high_risk_customer": "a high-risk customer",
        }
        for typ in case.typology_match:
            support = next((rf for rf in case.risk_factors if rf.typology == typ), None)
            if support:
                lines.append(
                    f"The pattern is consistent with {names.get(typ, typ)} "
                    f"[{support.evidence_ids[0]}]."
                )
        if alert:
            lines.append(
                f"The bank's risk model scored the alerted transaction "
                f"{float(alert.data['score']):.2f} on a scale of 0 to 1 [{alert.id}]."
            )

        lines.append("")
        lines.append("## Action")
        lines.append(
            "This case was prepared by the investigation assistant for review by an "
            f"investigator, and no report has been filed [alert:{case.alert_txn_id}]."
        )
        return "\n".join(lines) + "\n"


def _case_payload(case: CaseFile) -> str:
    keep = [
        {
            "id": e.id,
            "kind": e.kind,
            "summary": e.summary,
            **(
                {"data": {k: v for k, v in e.data.items() if k != "reasons"}}
                if e.kind != "policy"
                else {}
            ),
        }
        for e in case.evidence
    ]
    return json.dumps(
        {
            "subject_account": case.subject_account,
            "alert_txn_id": case.alert_txn_id,
            "risk_factors": [rf.model_dump() for rf in case.risk_factors],
            "typology_match": case.typology_match,
            "open_questions": case.open_questions,
            "evidence": keep,
        },
        indent=1,
        default=str,
    )


def draft_report(
    case: CaseFile, gateway: LLMGateway | None = None, budget: Budget | None = None
) -> DraftResult:
    if gateway is None or gateway.provider.name == "offline":
        text = TemplateDrafter().draft(case)
        return DraftResult(text, validate(text, case), "template", 1, DRAFTER_VERSION)

    system, user_tpl, version = _load_draft_prompt()
    budget = budget or Budget()
    payload = _case_payload(case)
    feedback = ""
    text, result = "", None
    for attempt in (1, 2):
        user = user_tpl.replace("{case_json}", payload).replace("{feedback}", feedback)
        try:
            text = gateway.complete(
                "draft",
                system,
                user,
                prompt_version=version,
                budget=budget,
                case_id=case.case_id,
                max_tokens=2_000,
            )
        except BudgetExceeded:
            break
        result = validate(text, case)
        if result.ok:
            return DraftResult(text, result, "llm", attempt, version)
        feedback = (
            "\nYour previous draft failed validation. Fix exactly these problems:\n"
            + "\n".join(f'- {i.problem}: "{i.sentence[:120]}"' for i in result.issues)
        )
    return DraftResult(text, result or validate(text, case), "llm_needs_human", 2, version)


def _load_draft_prompt() -> tuple[str, str, str]:
    import hashlib
    from pathlib import Path

    text = (Path(__file__).with_name("prompts") / "draft_v1.md").read_text()
    sections = {
        p.split("\n", 1)[0].strip(): p.split("\n", 1)[1].strip()
        for p in text.split("## ")[1:]
        if p.split("\n", 1)[0].strip() in {"system", "user"}
    }
    return (
        sections["system"],
        sections["user"],
        f"draft_v1-{hashlib.sha256(text.encode()).hexdigest()[:8]}",
    )


def edit_distance(a: str, b: str) -> float:
    """0 = identical, 1 = completely rewritten (how much the investigator changed the draft)."""
    from difflib import SequenceMatcher

    return round(1 - SequenceMatcher(None, a, b).ratio(), 4)


__all__ = ["DraftResult", "TemplateDrafter", "draft_report", "edit_distance"]
