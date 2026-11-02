# Model card: investigation agent and STR drafter

| | |
|---|---|
| Component | `agent-v1` (LangGraph), prompts `analyze_v1`, `draft_v1`, drafter `report-v1` |
| LLM | Configurable via the gateway: offline (deterministic), Gemini, Anthropic |
| Decision role | **Assistive.** Gathers evidence, suggests a disposition, drafts a narrative. Never decides |

## What it does
For one alert: fetch the alert, KYC profile, ±14/2-day transactions, network loops, watchlist
hits for the subject and counterparties, and relevant policy passages; produce a case file of
typed evidence, cited risk factors, typologies, open questions and a suggested disposition
(`escalate` / `review` / `close_candidate`); draft a five-section STR narrative in which every
sentence cites evidence.

## Controls built in
| Risk | Control |
|---|---|
| Invented facts | Case-file schema rejects unknown evidence ids; report validator checks every amount, date, account and citation |
| Prompt injection via customer data | Data wrapped as data; instruction-like text flagged; dispositions bounded by rules |
| Missed sanctions | Hard rule: sanctions hit → escalate, whatever the model says |
| Runaway cost or latency | Tool-call, time and token budgets; breach → `needs_human` |
| PII leaving the bank | Gateway masks account and customer ids before any external call |
| Non-reproducibility | Every LLM call recorded; exact case replay |

## Evaluation (offline mode, eval world seed 2024; 50 golden cases)
| Metric | Value | Gate |
|---|---|---|
| Typology accuracy (laundering cases) | 1.00 | ≥ 0.90 |
| Laundering cases not suggested for closing | 1.00 | = 1.00 |
| Laundering cases escalated | 0.83 | — |
| Decoys not escalated | 0.375 | ≥ 0.30 (known weakness) |
| Injection resisted | 1.00 | = 1.00 |
| Sanctions escalated | 1.00 | = 1.00 |
| Starved budget → needs_human | 1.00 | = 1.00 |
| Citations resolve | 1.00 | = 1.00 |
| Report validator pass rate | 1.00 | ≥ 0.95 |
| Planted errors caught by validator | 541 / 541 | = 100% |
| Latency p95 | 1.2 s | ≤ 10 s |

## Known weaknesses
- **Over-escalation of decoys** (62.5% escalated): payroll runs by customers whose KYC segment
  is "retail" are treated as fan-out, and any counterparty watchlist hit raises severity. This
  is conservative (it costs investigator time, not missed laundering) and is tracked by a gate.
- LLM mode has not been evaluated with real investigator grades; the judge must be calibrated
  first (see `aml-eval judge-template`).
