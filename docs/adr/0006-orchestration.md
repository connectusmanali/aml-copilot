# ADR 0006: Case orchestration with Temporal

- Status: accepted
- Date: 2026-10-20

## Context
A case lives for days: it is scored, investigated, drafted, reviewed, maybe escalated, and
decided. Steps call slow, failure-prone services (LLM APIs). Cron jobs and status columns lose
work on crashes and cannot express "wait for a human, but escalate after 48 hours".

## Decisions
1. **One `CaseWorkflow` per alert**, workflow id = case id (`case-<txn_id>`), duplicate starts
   rejected — so an alert can never open two cases.
2. **Routing by score**: high priority → agent investigation + drafting; low priority (rule hit
   only) → light review without the agent.
3. **Human decision as a signal**; an **SLA timer** escalates to a senior if no decision arrives.
   Refused decisions (wrong role, no final report) are recorded and the workflow keeps waiting.
4. **Activities are idempotent** (they check state first) so automatic retries are safe;
   business-rule refusals are `non_retryable`.
5. **Separate agent task queue** with `max_concurrent_activities`: LLM-bound work is
   rate-limited (backpressure) independently of database work.
6. **State machine enforced in the store** (`TRANSITIONS`), with the audit entry written in the
   same transaction as the change.

## Consequences
- Temporal is one more service to run (dev server locally; on the VM it shares an ops Postgres).
- Tests run against a real local Temporal server, including a 1-second SLA escalation.
