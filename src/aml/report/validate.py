"""Deterministic validation of a drafted report against its case file.

The validator, not the model, decides whether a draft can go to an investigator:
  1. every sentence carries at least one citation, e.g. [txn:abc123]
  2. every cited id exists in the case file
  3. a sentence cannot rest on policy text alone (pol: supports reasoning, not facts)
  4. every amount in a sentence matches a cited record (or the sum of the cited records)
  5. every date in a sentence matches a cited record
  6. every account id in a sentence appears in a cited record

Figures a sentence may state: values in the cited records (to the cent, or rounded to whole
dollars when the sentence gives no cents), the sum of the cited transactions, and figures the
deterministic rule analyzer computed from exactly those records (only when the risk factors
came from code, not from a model).

No LLM judges this: a rule either holds or it does not, and the same draft always gets the
same verdict.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from aml.agent.casefile import CaseFile, Evidence

CITATION = re.compile(r"\[((?:alert|acct|txn|wl|net|pol):[^\]\s]+)\]")
AMOUNT = re.compile(r"(?:\bUSD\s?|\$\s?)(\d[\d,]*(?:\.\d{1,2})?)")
DATE = re.compile(r"\b((?:19|20)\d\d-\d\d-\d\d)\b")
ACCOUNT = re.compile(r"\b(?:T[0-9a-f]{6}-)?\d{3}-[0-9A-F]{8,9}\b")
# A sentence ends after ". " or after a citation, when the next word starts with a capital.
SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z])")


@dataclass
class Issue:
    sentence: str
    problem: str


@dataclass
class Validation:
    ok: bool
    sentences: int
    cited_sentences: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def citation_rate(self) -> float:
        return self.cited_sentences / self.sentences if self.sentences else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "citation_rate": round(self.citation_rate, 4)}


def sentences(text: str) -> list[str]:
    """Body sentences, without headings. A sentence keeps its trailing citations."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lstrip("-* ").strip()
        out += [s.strip() for s in SPLIT.split(line) if s.strip()]
    return out


def _record_values(e: Evidence) -> tuple[set[str], set[str], set[str]]:
    """Amounts (2dp strings), dates and account ids that this evidence record supports."""
    amounts, dates, accounts = set(), set(), set()
    for k, v in e.data.items():
        if k in {
            "amount_usd",
            "amount_paid",
            "amount_received",
            "expected_monthly_volume",
            "score",
        } and isinstance(v, int | float):
            amounts.add(f"{float(v):,.2f}")
        if isinstance(v, str):
            dates.update(DATE.findall(v))
            accounts.update(ACCOUNT.findall(v))
    accounts.update(ACCOUNT.findall(e.summary))
    dates.update(DATE.findall(e.summary))
    return amounts, dates, accounts


def _computed(case: CaseFile, ids: set[str]) -> set[str]:
    """Figures stated by rule-computed risk factors whose evidence the sentence cites."""
    if case.analysis_mode != "rules":
        return set()
    out: set[str] = set()
    for rf in case.risk_factors:
        if set(rf.evidence_ids) <= ids:
            out |= {_norm(a) for a in AMOUNT.findall(rf.statement)}
            out |= {_norm(a) for a in re.findall(r"\band ([\d,]+(?:\.\d\d)?)\b", rf.statement)}
    return out


def _norm(raw: str) -> str:
    return f"{float(raw.replace(',', '')):,.2f}"


def validate(text: str, case: CaseFile) -> Validation:
    evidence = case.evidence_by_id()
    issues: list[Issue] = []
    body = sentences(text)
    cited = 0
    for s in body:
        ids = CITATION.findall(s)
        if not ids:
            issues.append(Issue(s, "no citation"))
            continue
        cited += 1
        unknown = [i for i in ids if i not in evidence]
        if unknown:
            issues.append(Issue(s, f"cites unknown evidence {unknown}"))
            continue
        if all(i.startswith("pol:") for i in ids):
            issues.append(Issue(s, "rests on policy text only; cite the bank's records"))
        amounts, dates, accounts = set(), set(), {case.subject_account}
        total = 0.0
        for i in ids:
            a, d, acc = _record_values(evidence[i])
            amounts |= a
            dates |= d
            accounts |= acc
            usd = evidence[i].data.get("amount_usd")
            if i.startswith("txn:") and isinstance(usd, int | float):
                total += usd
        amounts.add(f"{total:,.2f}")
        amounts |= _computed(case, set(ids))
        whole = {f"{round(float(a.replace(',', ''))):,.2f}" for a in amounts}
        claim = CITATION.sub("", s)
        for raw_amt in AMOUNT.findall(claim):
            amt = _norm(raw_amt)
            ok = amt in amounts or ("." not in raw_amt and amt in whole)
            if not ok:
                issues.append(Issue(s, f"amount USD {amt} not found in the cited records"))
        for d in DATE.findall(claim):
            if d not in dates:
                issues.append(Issue(s, f"date {d} not found in the cited records"))
        for acc in ACCOUNT.findall(claim):
            if acc not in accounts:
                issues.append(Issue(s, f"account {acc} not found in the cited records"))
    return Validation(
        ok=not issues and bool(body), sentences=len(body), cited_sentences=cited, issues=issues
    )
