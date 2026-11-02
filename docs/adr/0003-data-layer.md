# ADR 0003: Data layer — streaming ingestion, point-in-time features, investigation graph

- Status: accepted
- Date: 2026-10-19
- Amends: ADR 0001 (adds schema migrations; Postgres is the system of record)

## Context

Step 2 turns a one-off batch load into a data layer the later steps can trust: transactions
arrive as a stream, bad records are quarantined, features are computed without leakage, and
investigators (and the Step 4 agent) can explore the money network. Reviewing Step 1 also found
weaknesses to fix first: a silent row drop, an audit log that `TRUNCATE` could empty, a KYC
field that leaked the label, money summed across currencies, and SQL built with f-strings.

## Decisions

### Schema and storage
1. **Numbered SQL migrations** (`infra/postgres/migrations/NNNN_*.sql`) applied by our own
   ~100-line runner (`aml-db migrate`): one transaction per migration, recorded with a
   checksum, refused if edited after being applied, serialized by an advisory lock. Alembic
   was rejected: we have no ORM models, and plain SQL is what a bank DBA reviews.
2. **Postgres is the system of record.** Features, quality checks and the graph load read from
   it, so streamed and batch data are treated the same.
3. **Lineage:** every load and stream session is a row in `raw.ingest_runs`; each transaction
   keeps the run that first inserted it.
4. **Quarantine, persisted:** rejected records go to `raw.rejected_records` with the reason,
   keyed by a hash so reprocessing never duplicates them.

### Streaming
5. **The consumer is the contract gate.** The producer (a stand-in for a core banking system)
   never validates; the consumer validates each message against the same `Transaction` contract
   as the batch path.
6. **At-least-once delivery + idempotent sink.** Order per batch: dead letters acknowledged →
   database transaction committed → Kafka offsets committed. A crash anywhere replays the batch;
   `ON CONFLICT DO NOTHING` on the deterministic `txn_id` stores each transaction once. Tested
   with a simulated crash, both in memory and against real Postgres.
7. **Money travels as strings** in JSON; messages are keyed by sender account so each account's
   payments stay in order within a partition.
8. **Client code depends on small Protocols**, not on `confluent_kafka` directly, so delivery
   guarantees are tested without a broker. Real-broker tests run in CI.

### Features
9. **Computed in Python (numpy/scipy), not in Neo4j or SQL.** One sorted key per event
   (`account * 2^34 + epoch_seconds`) answers windowed counts, sums and distinct counts for all
   accounts at once. Verified against brute force (0 mismatches on every row).
10. **Point-in-time correctness is a test, not a promise:** features up to time T are identical
    whether or not later data exists (`test_no_future_leakage`).
11. **Graph features are daily snapshots** built from `[D - 30d, D)`, so a transaction is never
    scored against its own day's network. Label-derived features ("known bad") use a 3-day
    reporting lag. Step 3 reports the model with and without them (ablation).
12. **All amounts in USD** using a fixed, approximate September 2022 rate table; unknown
    currencies are reported by the quality check, never silently mixed.
13. **Feast is optional** (`--extra feast`) and serves the same definitions online. Training
    data comes from our tested `training_frame`, because testing revealed two Feast behaviours:
    its file offline store drops rows with no match in a view (biased against new and quiet
    accounts), and its online store does not apply TTL on read. Our wrapper restores left-join
    semantics and checks row counts.

### Graph
14. **Neo4j is for investigation, not features.** Bounded, parameterized queries (time window,
    hop limit validated as an int, row LIMIT) for the investigator and the Step 4 agent.
15. **The laundering label is never loaded into the graph.** If the agent could see the answer
    key, every evaluation of it would be meaningless.
16. **A capacity guard** refuses loads beyond AuraDB Free's 200k nodes / 400k relationships;
    customer attributes live on Account nodes to halve the node count.

### Quality
17. **Dataset-level quality checks** (duplicates, nulls, future timestamps, missing profiles are
    critical; rejected rate, label rate, unknown currencies, outliers are warnings). `prepare`
    writes nothing when a critical check fails; `aml-data quality` exits non-zero for CI.

## Consequences

- More moving parts: a broker and a graph database. Both are optional for unit tests.
- Kafka and Neo4j integration tests could not run in the authoring environment (no Docker
  daemon); they run in CI against real Redpanda and Neo4j. Their logic is covered by fakes.
- Synthetic data limitations (KYC derived from the whole window; dense random traffic producing
  many incidental 3-account loops) are documented for the Step 3 model card.
- `numpy` is held below 2.5 while Feast holds `pandas` below 3.
