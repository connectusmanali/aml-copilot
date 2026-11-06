"""LLM-as-judge for narrative quality, and the calibration that decides whether to trust it.

The deterministic validator checks FACTS. Whether a narrative is clear, complete and neutral
is a judgement, so a model grades it on a rubric (1-5 per criterion). A judge is only used as
a gate after it agrees with human graders on a calibration set:

    1. `aml-eval judge-template` writes evals/human_grades.csv with drafts and EMPTY grade cells
    2. investigators fill in the grades (the repo never ships invented human grades)
    3. `aml-eval judge-calibrate` runs the judge on the same drafts and reports agreement
       (exact match, within-one, quadratic-weighted kappa). Kappa >= 0.6 -> usable as a gate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aml.gateway.gateway import Budget, LLMGateway

CRITERIA = ("clarity", "completeness", "neutrality")
RUBRIC = """You grade the narrative of a suspicious transaction report written for a bank's
financial intelligence unit. Grade each criterion from 1 (poor) to 5 (excellent):
- clarity: a reader understands who did what, when, and why it matters, on first reading
- completeness: covers who, what, when/where, why suspicious; nothing material omitted
- neutrality: states facts without speculation about guilt or loaded language
Reply with JSON only: {"clarity": n, "completeness": n, "neutrality": n, "comment": "..."}"""
PROMPT_VERSION = "judge_v1"
USABLE_KAPPA = 0.6


@dataclass
class Grade:
    clarity: int
    completeness: int
    neutrality: int


def judge(gateway: LLMGateway, narrative: str, case_id: str = "") -> Grade:
    raw = gateway.complete(
        "judge",
        RUBRIC,
        f"<narrative>\n{narrative}\n</narrative>",
        prompt_version=PROMPT_VERSION,
        budget=Budget(),
        case_id=case_id,
        max_tokens=300,
        json_mode=True,
    )
    d = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    return Grade(*(min(5, max(1, int(d[c]))) for c in CRITERIA))


def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal grades 1..k."""
    if not a:
        return float("nan")
    o = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        o[x - 1, y - 1] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    denom = (w * e).sum()
    return float(1 - (w * o).sum() / denom) if denom else 1.0


def write_template(drafts: dict[str, str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "narrative", *CRITERIA, "grader"])
        for cid, text in drafts.items():
            wr.writerow([cid, text, "", "", "", ""])
    return path


def load_human_grades(path: Path) -> dict[str, tuple[str, Grade]]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if all(row.get(c, "").strip() for c in CRITERIA):
                out[row["case_id"]] = (row["narrative"], Grade(*(int(row[c]) for c in CRITERIA)))
    return out


def calibrate(gateway: LLMGateway, grades_csv: Path) -> dict[str, object]:
    human = load_human_grades(grades_csv)
    if len(human) < 20:
        raise ValueError(
            f"{len(human)} graded drafts in {grades_csv}; calibration needs >= 20 "
            "drafts graded by investigators"
        )
    report: dict[str, object] = {"n": len(human)}
    machine = {cid: judge(gateway, text, cid) for cid, (text, _) in human.items()}
    for c in CRITERIA:
        h = [getattr(g, c) for _, g in human.values()]
        m = [getattr(machine[cid], c) for cid in human]
        report[c] = {
            "exact": round(float(np.mean([x == y for x, y in zip(h, m, strict=True)])), 3),
            "within_one": round(
                float(np.mean([abs(x - y) <= 1 for x, y in zip(h, m, strict=True)])), 3
            ),
            "kappa": round(weighted_kappa(h, m), 3),
        }
    report["usable_as_gate"] = all(
        isinstance(v, dict) and v["kappa"] >= USABLE_KAPPA
        for k, v in report.items()
        if k in CRITERIA
    )
    return report
