# ADR 0009: Operations — tracing, drift, tamper-evident audit, replay, safe model change

- Status: accepted
- Date: 2026-10-20

## Decisions
1. **OpenTelemetry** spans for workflow steps, agent nodes and LLM calls (GenAI attributes:
   model, tokens, cost; never prompt text). Any OTLP backend works (Langfuse, Tempo, Jaeger).
2. **PSI drift** on features and the score distribution, plus alerts/day against capacity;
   0.10 watch, 0.25 act.
3. **Hash-chained audit log**: each entry stores the previous hash and its own; triggers block
   UPDATE/DELETE/TRUNCATE, and `aml-ops audit-verify` detects tampering by anyone who bypasses
   them (tested by disabling the trigger and rewriting a row).
4. **Exact case replay**: rerun an investigation with the gateway answering from recorded calls
   and diff the case file (`aml-ops replay --case`).
5. **Shadow, then canary** for a new scorer: shadow scores stored beside champion scores;
   canary by stable account hash; promotion needs overlap/volume/precision gates and sign-off.
6. **One image, several roles** (api, worker, scorer, ui); docker compose on the free VM,
   Kubernetes manifests (restricted pod security, default-deny network policy, external
   secrets) for a bank platform.
