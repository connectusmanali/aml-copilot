# Model risk assessment — AML Investigation Copilot

Written in the structure of a bank model risk management (MRM) submission, following the
principles of OSFI Guideline E-23 (Model Risk Management, which explicitly covers AI/ML) and
US SR 11-7. Check the current text of each guideline before relying on this mapping.

## 1. Model inventory
| Id | Component | Type | Tier | Decision role |
|---|---|---|---|---|
| M1 | Risk scorer | LightGBM + isotonic calibration | High | Prioritises alerts |
| M2 | Rule scenarios | Configured thresholds | Medium | Raises low-priority alerts |
| M3 | Investigation agent | LLM + deterministic rules (LangGraph) | High | Suggests disposition |
| M4 | STR drafter | LLM or template + deterministic validator | High | Drafts narrative |
| M5 | Policy retrieval | TF-IDF / embeddings | Low | Supplies guidance passages |

Tiering rationale: M1, M3 and M4 influence regulatory reporting outcomes (which activity is
investigated and how it is described). No component takes a final decision.

## 2. Purpose and use
Reduce time per alert and the false-positive burden while keeping every reporting decision with
a human investigator. Users: L1/L2 AML investigators; senior investigators for escalations.

## 3. Conceptual soundness
- Detection: point-in-time features, time-based validation, calibration, capacity threshold;
  compared against the incumbent rules at equal budget (ADR 0004).
- Agent: fixed graph (not free-form), read-only tools, evidence-cited outputs, hard rules for
  sanctions and budgets (ADR 0005).
- Drafting: deterministic validator as the control, not a model judge (ADR 0007).

## 4. Data
Synthetic only (IBM AML simulation and a generator with labelled decoys). Quality gate before
any write; quarantine for rejects; lineage per load; contracts in code. Real deployment requires
a data-sourcing review, PII classification and retention rules.

## 5. Performance and validation evidence
Source: `aml-eval run` (all gates pass) and the training report.

| Area | Evidence | Result |
|---|---|---|
| Scorer discrimination | Test-period PR-AUC | 0.886 (medium), 0.835 (eval world) |
| Scorer business value | Precision at capacity vs rules | 75.3% vs 7.8% |
| Calibration | ECE | 0.0021 |
| Agent safety | Laundering never suggested for closing | 24/24 |
| Robustness | Injection, sanctions, starved budget | 4/4, 3/3, 3/3 |
| Report integrity | Planted errors caught | 541/541 |
| Reproducibility | Case replay | identical case file |
| Audit integrity | Hash chain after tampering | tampering detected at the exact entry |

## 6. Limitations and compensating controls
| Limitation | Impact | Compensating control |
|---|---|---|
| Synthetic data only | Real performance unknown | Pilot in shadow mode on real alerts before any routing |
| Label bias in real outcomes | New typologies missed | Rules remain active; periodic below-threshold sampling |
| Agent over-escalates decoys | Investigator time | Tracked gate; tune with investigator feedback |
| LLM vendor change | Behaviour drift | Prompt/model versions recorded; eval gates on every change |
| LLM judge uncalibrated | Narrative quality unproven | Not used as a gate until kappa ≥ 0.6 with human grades |

## 7. Ongoing monitoring
| Signal | Threshold | Action |
|---|---|---|
| Feature/score PSI | > 0.10 watch, > 0.25 act | Investigate, consider retraining |
| Alerts/day vs capacity | ±20% for 5 days | Recalibrate threshold |
| Precision on decided cases | below validation − 10 pts | Model owner review |
| Validator failure rate (LLM mode) | > 5% | Review prompt/model version |
| Investigator disagreement with score | > 30% | Model owner review |
| Audit chain verification | any failure | Incident; security review |
| Eval gates | any failure | Release blocked (CI) |

## 8. Change management
Versioned scorer bundles, prompts (content hash in the version) and agent graph. Any change:
eval gates → shadow → canary (hash-bucketed accounts) → promotion with sign-off by model owner
and MRM. Rollback = repoint `CURRENT`.

## 9. Human oversight and accountability
Investigator decides with a written reason; escalated (SLA-breached) cases need a senior;
filing needs an approved final report that passes validation; auditors read but cannot decide.
All of it in a tamper-evident audit log.

## 10. Sign-off (template)
| Role | Name | Date | Decision |
|---|---|---|---|
| Model owner | | | |
| Independent validator | | | |
| Business owner (AML) | | | |
