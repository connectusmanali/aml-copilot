# ADR 0008: Evaluation discipline — golden set, adversarial cases, release gates

- Status: accepted
- Date: 2026-10-20

## Decisions
1. **Golden set from a fixed-seed synthetic world** where the ground truth is exact: laundering
   patterns (structuring, fan-out, layering) and decoys (payroll, single honest cash deposit).
2. **Adversarial cases**: prompt injection in the customer record, a planted sanctions hit on a
   decoy, and a starved tool budget.
3. **Three layers of metrics**: scorer (PR-AUC, precision/recall at capacity, ECE), agent
   (typology accuracy, no laundering case suggested for closing, decoys not escalated,
   citation integrity, latency), report (validator pass rate, citation rate, mutations caught).
4. **Gates in `gates.yaml`**: safety gates are absolute (1.0); quality gates sit just below the
   measured baseline. `aml-eval run` exits non-zero on any failure, and CI runs it on every PR.
5. **History** (`history.jsonl`) so a slow decline is visible even when every run passes.
6. **LLM-as-judge only after calibration** against investigator grades (quadratic-weighted
   kappa ≥ 0.6). The repo ships an empty grading template, never invented human grades.
