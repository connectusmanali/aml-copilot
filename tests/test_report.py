"""Step 6: report drafting is cited, and the validator catches fabricated facts."""

import re

import pytest

from aml.agent.casefile import CaseFile
from aml.agent.graph import InvestigationAgent
from aml.gateway.gateway import LLMGateway
from aml.report.draft import TemplateDrafter, draft_report, edit_distance
from aml.report.validate import sentences, validate
from tests.test_agent import ScriptedProvider, first, world  # noqa: F401  (fixture reuse)


@pytest.fixture(scope="module")
def cases(world: dict) -> dict[str, CaseFile]:  # noqa: F811
    agent = InvestigationAgent(world["data"])
    return {f: agent.investigate(first(world, f)) for f in ("Cash", "Wire", "ACH")}


def test_template_draft_is_valid_and_fully_cited(cases: dict[str, CaseFile]) -> None:
    for case in cases.values():
        result = draft_report(case)
        assert result.mode == "template"
        assert result.validation.ok, result.validation.issues
        assert result.validation.citation_rate == 1.0
        for h in (
            "## Who",
            "## What happened",
            "## When and where",
            "## Why it is suspicious",
            "## Action",
        ):
            assert h in result.text


def test_validator_catches_a_changed_amount(cases: dict[str, CaseFile]) -> None:
    case = cases["Cash"]
    text = TemplateDrafter().draft(case)
    amounts = re.findall(r"USD ([\d,]+\.\d\d)", text.split("## What happened")[1])
    assert amounts
    tampered = text.replace(f"USD {amounts[0]}", "USD 1,234,567.89", 1)
    v = validate(tampered, case)
    assert not v.ok and any("1,234,567.89" in i.problem for i in v.issues)


def test_validator_catches_unknown_citation_missing_citation_and_wrong_date(
    cases: dict[str, CaseFile],
) -> None:
    case = cases["Wire"]
    v = validate("## Who\nThe customer moved money [txn:does-not-exist].\n", case)
    assert any("unknown evidence" in i.problem for i in v.issues)
    v = validate("## Who\nThe customer moved a lot of money.\n", case)
    assert any(i.problem == "no citation" for i in v.issues)
    alert = f"alert:{case.alert_txn_id}"
    v = validate(f"## When\nIt happened on 1999-01-01 [{alert}].\n", case)
    assert any("date 1999-01-01" in i.problem for i in v.issues)


def test_policy_only_sentences_are_rejected(cases: dict[str, CaseFile]) -> None:
    case = cases["Cash"]
    pol = next(e.id for e in case.evidence if e.kind == "policy")
    v = validate(f"The customer is structuring [{pol}].", case)
    assert any("policy text only" in i.problem for i in v.issues)


def test_sentence_split_keeps_citations_with_their_sentence() -> None:
    out = sentences("## Who\nA paid B [txn:1]. Then C paid D [txn:2] [txn:3].\n")
    assert out == ["A paid B [txn:1].", "Then C paid D [txn:2] [txn:3]."]


def test_llm_draft_retries_once_with_feedback_then_passes(cases: dict[str, CaseFile]) -> None:
    case = cases["Wire"]
    good = TemplateDrafter().draft(case)
    provider = ScriptedProvider(["## Who\nThe customer did crimes.\n", good])
    result = draft_report(case, LLMGateway(provider))
    assert result.mode == "llm" and result.attempts == 2 and result.validation.ok
    assert "failed validation" in provider.seen[1] and "no citation" in provider.seen[1]


def test_llm_draft_goes_to_human_after_two_failures(cases: dict[str, CaseFile]) -> None:
    case = cases["ACH"]
    bad = f"## Who\nThe account sent USD 9.99 [alert:{case.alert_txn_id}].\n"
    result = draft_report(case, LLMGateway(ScriptedProvider([bad, bad])))
    assert result.mode == "llm_needs_human" and not result.validation.ok


def test_edit_distance() -> None:
    assert edit_distance("abc", "abc") == 0.0
    assert 0 < edit_distance("hello world", "hello there") < 1
