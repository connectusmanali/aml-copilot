"""Step 7: the eval harness itself (smoke-sized world) and the judge calibration maths."""

from pathlib import Path

import pytest

from aml.evals import harness
from aml.evals.judge import Grade, load_human_grades, weighted_kappa, write_template


@pytest.fixture(scope="module")
def result() -> harness.EvalResult:
    return harness.run("smoke", per_kind=3)


def test_safety_gates_hold_on_smoke_world(result: harness.EvalResult) -> None:
    gates = {g["metric"]: g for g in result.gates}
    for name in (
        "agent.laundering_not_closed",
        "agent.citations_resolve",
        "adversarial.injection_resisted",
        "adversarial.sanctions_escalated",
        "adversarial.starved_goes_to_human",
        "report.mutations_caught",
    ):
        assert gates[name]["passed"], gates[name]


def test_report_is_written_with_history(result: harness.EvalResult, tmp_path: Path) -> None:
    path = harness.save(result, tmp_path, "offline")
    harness.save(result, tmp_path, "offline")
    assert "| Metric |" in path.read_text()
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2


def test_failing_gate_is_reported(tmp_path: Path) -> None:
    gates = tmp_path / "g.yaml"
    gates.write_text("gates:\n  x: {min: 0.9}\n  missing: {min: 0}\n")
    out = harness.check_gates({"x": 0.5}, gates)
    assert [g["passed"] for g in out] == [False, False]


def test_weighted_kappa() -> None:
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0


def test_template_has_empty_grades_and_is_ignored_until_filled(tmp_path: Path) -> None:
    p = write_template({"c1": "text"}, tmp_path / "g.csv")
    assert load_human_grades(p) == {}
    p.write_text(p.read_text().replace("c1,text,,,,", "c1,text,4,3,5,alice"))
    assert load_human_grades(p) == {"c1": ("text", Grade(4, 3, 5))}
