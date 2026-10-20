# AML Investigation Copilot

An AI system that helps anti-money-laundering investigators triage alerts, gather evidence and
draft suspicious transaction reports, with a human making every final decision.

> Core principle: **a deterministic workflow wraps a non-deterministic agent.**
> The agent gathers and drafts; the workflow controls state; a human decides.

Status: **all 10 steps complete.** 129 tests · all evaluation gates pass · runs fully offline.

| Read | |
|---|---|
| Design | [`docs/design.md`](docs/design.md) |
| Decisions | [`docs/adr/`](docs/adr/) (0001–0009) |
| Model risk | [`docs/model-risk/model-risk-assessment.md`](docs/model-risk/model-risk-assessment.md), [evaluation baseline](docs/model-risk/evaluation.md) |
| Model cards | [risk scorer](docs/model-cards/risk-scorer.md), [investigation agent](docs/model-cards/investigation-agent.md) |
| Operations | [`docs/runbook.md`](docs/runbook.md) |

## Results (synthetic data)

| | Alerts | Precision | Recall | Investigator hours |
|---|---|---|---|---|
| Rule scenarios | 4,185 | 7.8% | 82.6% | 2,092 |
| ML scorer at 100 alerts/day | 457 | 75.3% | 86.9% | 229 |

Agent and report (golden set of 50 cases incl. adversarial): typology accuracy 100%, no
laundering case suggested for closing, prompt injection resisted 4/4, sanctions always
escalated, 541/541 planted errors caught by the report validator.

## Architecture

| Layer | Purpose | Tech |
| --- | --- | --- |
| Data | Stream, validate, store, features, entity graph | Redpanda, Postgres, Parquet, Feast, Neo4j |
| Detection | Rules baseline + ML risk scorer with explanations | LightGBM, SHAP, MLflow |
| Orchestration | Durable case workflow, retries, SLAs, human signals | Temporal |
| Agent | Read-only tools, bounded graph, structured case file | LangGraph, pgvector |
| LLM gateway | PII masking, budgets, retries, caching, recording, replay | own gateway (httpx): offline, Gemini, Anthropic |
| Report | Cited STR narrative, deterministic validator | template / LLM + validator |
| Human review | Queue, evidence, cited draft, edit/approve, decide | FastAPI, Streamlit |
| Governance | Eval gates, tracing, drift, hash-chained audit, replay, shadow/canary | pytest, OpenTelemetry, PSI |

## Quick start

Prerequisites: Docker Desktop (or Docker Engine), [uv](https://docs.astral.sh/uv/), `make`, git.
Windows users: run everything inside WSL2.

```bash
cp .env.example .env          # then edit passwords
make install                  # Python deps (all extras) + pre-commit hooks
make up-lite                  # Postgres + Redpanda, then schema migrations (or `make up`)
make data-synthetic           # small generated dataset (quality-gated), no download needed
make load                     # into Postgres with run lineage; safe to run twice
make check                    # lint + types + unit tests
make test-int                 # Postgres integration tests (throwaway database)
```

### Step 2: streaming, features, graph

```bash
make stream-demo              # topics -> replay with 1 in 50 corrupted -> consume -> quality
make features                 # point-in-time features + training frame -> data/features/
make feast                    # register Feast views, fill the online store
uv run aml-features feast-check --account <account_id>   # online == offline
make up && make graph         # Neo4j (full stack): load the investigation graph
uv run aml-graph cycles --account <account_id>           # money that loops back
make test-services            # Kafka + Neo4j tests (full stack)
```

Bad messages land in the `transactions.dlq` topic (reason in the `error` header) and in
`raw.rejected_records`; every load or stream session is recorded in `raw.ingest_runs`.

### Steps 3–9: the copilot, offline in five commands

```bash
make pipeline SIZE=medium     # data -> features -> train scorer -> score alerts -> eval gates
make demo N=20                # API on :8000 with 20 investigated cases (in-memory, no services)
make ui                       # Streamlit UI on :8501 (key: dev-investigator / dev-senior / dev-auditor)
make investigate TXN=<txn_id> # one investigation, case file as JSON
make drift                    # PSI drift report
```

Full stack (Postgres + Temporal):

```bash
make up && make load          # services, schema, data
make worker                   # terminal 1: case + agent workers
make workflow N=20            # start one durable case workflow per alert
make api                      # terminal 2: decisions become workflow signals
make audit-verify             # hash chain intact?
uv run aml-ops replay --case case-<txn_id>   # exact re-run of an investigation
```

To use a real LLM, set `AML_LLM_PROVIDER=gemini` (free tier) or `anthropic` and
`AML_LLM_API_KEY` in `.env`. Without a key everything runs in deterministic offline mode.

### Running in the cloud (free tier)

The production-style setup uses Oracle Cloud Always Free (VM via Terraform), Neon Postgres,
Neo4j AuraDB Free, Tailscale and GitHub Actions, all at $0/month.
Full walkthrough: [`docs/cloud-setup.md`](docs/cloud-setup.md). Design rationale:
[`docs/adr/0002-cloud-free-tier.md`](docs/adr/0002-cloud-free-tier.md).

```bash
make cloud-init cloud-plan cloud-apply   # Oracle VM, network, bucket, budget alert
make migrate                             # schema -> Neon
make cloud-deploy                        # Redpanda, Temporal, MLflow -> VM + smoke tests
make cloud-load cloud-test-int           # data -> Neon, integration tests
```

### Using the real IBM dataset

1. Create a Kaggle API token (kaggle.com -> Settings -> API) and save it to `~/.kaggle/kaggle.json`.
2. `make download` (about 475 MB)
3. `make data SIZE=small` (100k rows), `SIZE=medium` (1M) or `SIZE=full` (~5M)
4. `make load`

## Service URLs (full stack)

| Service | URL |
| --- | --- |
| Postgres | `localhost:5432` |
| Redpanda (Kafka API) | `localhost:19092` |
| Redpanda Console (cloud VM) | http://aml-vm:8081 |
| Neo4j browser | http://localhost:7474 |
| Temporal UI | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Investigator API (docs) | http://localhost:8000/docs |
| Investigator UI | http://localhost:8501 |

## Repo layout

```
src/aml/
  config.py      typed settings (AML_* env vars)
  data/          contracts, IBM loader, synthetic generator, KYC profiles, FX, quality, loader
  db/            schema migrations runner, Postgres readers
  stream/        Kafka topics, replay producer, validating consumer, dead letters, sink
  features/      windowed account state, daily graph snapshots, as-of joins, Feast
  graph/         Neo4j loader and bounded investigation queries
  detection/     rule scenarios (YAML), LightGBM scorer, calibration, SHAP reasons, scoring API
  gateway/       LLM gateway: PII masking, budgets, retries, cache, recording, replay
  agent/         LangGraph investigation agent, read-only tools, policy RAG, prompts
  report/        cited STR drafting and the deterministic validator
  cases/         case store (memory/Postgres), state machine, decisions, hash-chained audit
  workflow/      Temporal CaseWorkflow, activities, workers
  evals/         golden set, adversarial cases, metrics, gates.yaml, LLM-judge calibration
  api/           investigator API (FastAPI, roles)
  ui/            Streamlit investigator UI
  ops/           tracing, PSI drift, shadow/canary, case replay
infra/postgres/migrations/  numbered schema migrations (never edited once applied)
docs/            design, ADRs, model risk, model cards, runbook, interview stories
deploy/k8s/      reference Kubernetes manifests
Dockerfile       one image: api | worker | scorer | ui
tests/           unit + integration tests
```

## Data notice

All data is synthetic: IBM's AML transaction simulation plus generated customer profiles.
Results demonstrate the method, not real-world performance.
