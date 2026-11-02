# ADR 0005: Investigation agent and LLM gateway

- Status: accepted
- Date: 2026-10-20

## Context
An investigator spends most of an alert's time collecting evidence across systems. An LLM can
help, but a free-form agent is unpredictable, can be manipulated by text inside customer data,
and can invent facts. The design must make those failures impossible or visible.

## Decisions
1. **Fixed LangGraph state machine**: plan → gather → analyze → finalize. The tool plan comes
   from the alert's typologies, not from the model. A breached budget routes to `needs_human`.
2. **Read-only tools** over a `CaseData` protocol: the agent never receives a database handle.
3. **Typed, citable evidence ids** (`alert:`, `acct:`, `txn:`, `wl:`, `net:`, `pol:`). The case
   file schema refuses any risk factor that cites an id that was not gathered.
4. **Budgets enforced at the tool boundary** (calls, seconds) and in the gateway (tokens).
5. **Deterministic overrides**: a sanctions hit always escalates; the strictest of model and
   rule dispositions wins; LLM factors citing unknown evidence are dropped.
6. **Prompt-injection defence in layers**: customer text is wrapped as data, instruction-like
   text is detected and surfaced as an open question, and dispositions are bounded by rules.
7. **LLM gateway**: PII masking (accounts/customers become `ACCT_n`/`CUST_n` before leaving the
   bank), call ids for caching and exact replay, retries with backoff, per-case token budget,
   cost accounting, and a recorder that writes every call to `cases.llm_calls`.
8. **Offline provider** (deterministic, no key) so the whole system runs and tests in CI.
9. **Policy RAG**: TF-IDF in memory or pgvector in Postgres; policy passages support reasoning
   but cannot be the only citation for a fact.

## Consequences
- Less "autonomous" than a ReAct loop, by design: every run takes a path a validator can read.
- The rule analyzer doubles as the fallback when the LLM fails or its budget runs out.
