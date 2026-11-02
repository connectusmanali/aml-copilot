# Evaluation baseline

Produced by `aml-eval run` (offline mode, eval world seed 2024, 2026-10-20). CI runs the same
command on every pull request and fails on any gate. Thresholds: `src/aml/evals/gates.yaml`.

| Metric | Value | Gate |
|---|---|---|
| scorer.pr_auc | 0.8354 | ≥ 0.75 |
| scorer.roc_auc | 0.9811 | — |
| scorer.ece | 0.0013 | ≤ 0.02 |
| scorer.precision_at_capacity | 0.2545 | ≥ 0.20 |
| scorer.recall_at_capacity | 0.9256 | ≥ 0.85 |
| agent.typology_accuracy | 1.0 | ≥ 0.90 |
| agent.laundering_not_closed | 1.0 | = 1.0 |
| agent.laundering_escalated | 0.8333 | — |
| agent.decoy_not_escalated | 0.375 | ≥ 0.30 |
| adversarial.injection_resisted | 1.0 | = 1.0 |
| adversarial.sanctions_escalated | 1.0 | = 1.0 |
| adversarial.starved_goes_to_human | 1.0 | = 1.0 |
| agent.citations_resolve | 1.0 | = 1.0 |
| agent.latency_p95_s | 1.245 | ≤ 10 |
| report.validator_pass_rate | 1.0 | ≥ 0.95 |
| report.citation_rate | 1.0 | ≥ 0.98 |
| report.mutations_caught | 1.0 (541/541) | = 1.0 |

Golden set: 24 laundering, 16 decoys, 4 injection, 3 sanctions, 3 starved-budget cases.
