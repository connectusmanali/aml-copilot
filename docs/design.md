# Design document — AML Investigation Copilot

## Problem
Transaction monitoring at large banks produces thousands of alerts a day; most are false
positives, and each takes an investigator 30–60 minutes of evidence gathering and writing.
Regulators expect timely, well-supported suspicious transaction reports (STRs) and expect the
bank to explain and validate every model involved.

**Goal:** cut investigator time per alert and the number of wasted alerts, without ever letting
software decide whether something is reported. **Non-goals:** automated filing, customer exits,
sanctions screening.

## Principles
1. A deterministic workflow wraps a non-deterministic agent. The workflow owns state; the agent
   only gathers and drafts; a human decides.
2. Every claim cites evidence, and a deterministic check verifies it.
3. Safety rules are code, not prompts (sanctions, budgets, roles, state transitions).
4. Everything is versioned, traced, replayable and in a tamper-evident audit log.
5. Nothing ships unless the evaluation gates pass.

## Architecture
```
 core banking ──► Redpanda ──► validating consumer ──► Postgres (raw, ref)      ◄── batch loads
                      │ bad                           │
                      ▼                               ▼
                 dead-letter topic          point-in-time features (Parquet / Feast), Neo4j graph
                                                      │
                                         rules + LightGBM scorer (calibrated, SHAP reasons)
                                                      │ alerts (high / low priority)
                                                      ▼
                              Temporal CaseWorkflow (one per alert, SLA timer, signals)
                         high │                                         │ low
                              ▼                                         ▼
          agent task queue (rate-limited):                       light review
          LangGraph plan→gather→analyze→finalize
          read-only tools · policy RAG (pgvector) · LLM gateway (masking, budgets, recording)
                              │ case file (typed evidence, cited risk factors)
                              ▼
                     STR drafter + deterministic validator (1 retry → needs_human)
                              │
                              ▼
        FastAPI investigator API (roles) ◄── Streamlit UI      decision signal ──► workflow
                              │
     Postgres cases.* : cases · case_files · reports · decisions · feedback · llm_calls ·
                        hash-chained audit_log
     Ops: OpenTelemetry traces · PSI drift · eval gates in CI · replay · shadow/canary
```

## Case lifecycle
`new → scored → investigating → drafted → in_review → filed | closed`, with `needs_human`
(incomplete investigation or invalid draft) and `escalated` (SLA breach; senior only). Illegal
transitions raise; every transition writes an audit entry in the same database transaction.

## Key trade-offs
| Choice | Alternative | Why |
|---|---|---|
| Fixed agent graph | ReAct loop | Predictable, testable, explainable to validators |
| Deterministic validator | LLM judge | Same input, same verdict; catches 100% of planted errors |
| Temporal | Cron + status column | Durable timers, retries, signals, replayable history |
| Capacity threshold | 0.5 cut-off | Matches how AML teams actually staff |
| Synthetic data | Real data | No access to real data; ground truth is exact |
| Free tier (Oracle VM, Neon, AuraDB) | Paid cloud | $0/month; same patterns as a bank platform |

## Scale and cost (estimates)
- Scoring: vectorised features + LightGBM, ~125k transactions scored in seconds on a laptop.
- Agent: ~7 tool calls and ≤ 20k tokens per case (budget); offline mode p95 1.2 s.
- Backpressure: agent queue concurrency caps LLM calls; Temporal queues the rest durably.

## Security and privacy
PII masked before any LLM call; secrets from env/secret store; API keys → roles (SSO stand-in);
container runs non-root with read-only root FS; default-deny network policy; append-only,
hash-chained audit log.
