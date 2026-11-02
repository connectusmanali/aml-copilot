# ADR 0001: Overall architecture

- Status: accepted
- Date: 2026-10-05

## Context

AML investigators at large banks face alert queues where 85-95% of rule-based alerts are false
positives. Each alert takes 20-30 minutes of manual evidence gathering before a decision.
Regulators require every decision to be explainable and reproducible, and the bank remains
accountable for any automated step.

We want an AI system that cuts investigator time per alert without taking the decision away
from humans and without creating an unexplainable black box.

## Decision

1. **A deterministic workflow wraps a non-deterministic agent.** Case state lives in a durable
   workflow engine (Temporal). The LLM agent is one activity inside it. It gathers evidence and
   drafts; it never changes case state on its own.
2. **Humans make every final decision.** File, close and escalate are human actions.
3. **Read-only agent tools.** The agent can query data; it cannot write anywhere except its
   own output.
4. **Grounded output.** Every factual sentence in a draft report cites a source record ID.
   A deterministic validator checks the citations.
5. **Single LLM gateway.** All model calls pass through one service for PII masking, routing,
   budgets and logging.
6. **Evals gate releases.** Golden-set evaluations run in CI; regressions block merges.
7. **Everything is versioned** (data, features, models, prompts, eval sets) so any past case can
   be replayed for an auditor.
8. **Local-first.** The whole stack runs on one machine with Docker Compose. A `lite` profile
   keeps Steps 1-3 workable on 8 GB RAM.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Fully autonomous agent that closes low-risk alerts | Regulator and accountability risk; hard to explain; no human accountable for the decision |
| Free-form ReAct loop | Unbounded cost and behaviour; hard to test. A fixed graph (plan, gather, analyze, summarize) is predictable |
| Celery/cron for orchestration | No durable state; a crash mid-case loses work; no built-in human-wait signals or timers |
| Rules only, no ML | Leaves the false-positive problem unsolved; used as the baseline instead |

## Consequences

- More moving parts than a notebook: Postgres, Redpanda, Neo4j, Temporal, MLflow.
- Every component is replaceable behind an interface (e.g. Redpanda <-> Kafka, LiteLLM <-> a
  bank's internal gateway), which mirrors how a bank would adopt it.
- The data is synthetic (IBM AML dataset + generated KYC profiles). Results show the method,
  not real-world performance.
