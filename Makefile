.DEFAULT_GOAL := help
SIZE ?= small
COMPOSE := docker compose

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install Python deps (all extras) and git hooks
	uv sync --all-extras
	uv run pre-commit install

up: ## Start the full stack (Postgres, Redpanda, Neo4j, Temporal, MLflow) and migrate
	$(COMPOSE) --profile full up -d --wait
	$(MAKE) db-migrate

up-lite: ## Start core only (Postgres, Redpanda) and migrate; for machines under 16 GB RAM
	$(COMPOSE) up -d --wait
	$(MAKE) db-migrate

db-migrate: ## Apply pending schema migrations (safe to re-run)
	uv run aml-db migrate

down: ## Stop all services (keeps data)
	$(COMPOSE) --profile full down

reset: ## Stop everything and DELETE all data volumes
	$(COMPOSE) --profile full down -v

ps: ## Show service status
	$(COMPOSE) --profile full ps

download: ## Download IBM AML dataset from Kaggle (needs ~/.kaggle/kaggle.json)
	./scripts/download_ibm.sh

data-synthetic: ## Prepare synthetic data: make data-synthetic SIZE=small|medium|full
	uv run aml-data prepare --source synthetic --size $(SIZE)

data: ## Prepare IBM dataset: make data SIZE=small|medium|full
	uv run aml-data prepare --source ibm --size $(SIZE)

load: ## Load prepared data into Postgres (safe to re-run)
	uv run aml-data load

quality: ## Data quality report on the database (exit code 1 on failure)
	uv run aml-data quality --from db

# ---------------- Step 2: streaming, features, graph ----------------
RATE ?= 500
CORRUPT ?= 50

topics: ## Create Kafka topics (raw + dead-letter) if missing
	uv run aml-stream topics

produce: ## Replay synthetic transactions into Kafka: make produce RATE=500 CORRUPT=50
	uv run aml-stream produce --source synthetic --rate $(RATE) --corrupt-every $(CORRUPT)

consume: ## Validate the stream into Postgres; bad messages to the DLQ (stops when idle)
	uv run aml-stream consume

stream-demo: topics produce consume quality ## Topics -> produce -> consume -> quality report

features: ## Build point-in-time features from the database into data/features/
	uv run aml-features build --from db

feast: ## Register Feast views and materialize the online store
	uv run aml-features feast-apply

graph: ## Load accounts and transactions into Neo4j (needs make up)
	uv run aml-graph load --from db

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Mypy strict
	uv run mypy

test: ## Unit tests (no services needed; temporal tests start a local dev server)
	uv run pytest -m "not integration and not kafka and not neo4j"

test-int: ## Postgres integration tests (needs make up-lite); uses a throwaway database
	uv run pytest -m integration

test-services: ## Kafka and Neo4j tests (needs make up)
	uv run pytest -m "kafka or neo4j"

check: lint typecheck test ## Everything CI runs

# ---------------- Steps 3-9: detection, agent, workflow, evals, API, ops ----------------
CAPACITY ?= 100
N ?= 20

train: ## Train, calibrate and register the risk scorer: make train CAPACITY=100
	uv run aml-detect train --capacity $(CAPACITY)

score: ## Score every transaction into alerts (data/features/alerts.parquet)
	uv run aml-detect score

investigate: ## Investigate one alert offline: make investigate TXN=<txn_id>
	uv run aml-agent investigate --txn $(TXN)

worker: ## Temporal workers (case queue + rate-limited agent queue)
	uv run aml-workflow worker

workflow: ## Start one case workflow per top-N alert: make workflow N=20
	uv run aml-workflow start --limit $(N)

eval: ## Golden-set evaluation with release gates (exit 1 on any failed gate)
	uv run aml-eval run

api: ## Investigator API backed by Postgres + Temporal
	uv run aml-api serve --temporal

demo: ## API with an in-memory store seeded from the top N alerts (no services needed)
	uv run aml-api serve --demo $(N)

ui: ## Streamlit investigator UI (talks to the API at AML_API_URL)
	uv run --extra ui streamlit run src/aml/ui/app.py

drift: ## PSI drift report: training period vs the latest week
	uv run aml-ops drift

audit-verify: ## Verify the audit log hash chain
	uv run aml-ops audit-verify

test-temporal: ## Workflow tests on a local Temporal dev server
	uv run pytest -m temporal

image: ## Build the application image
	docker build -t aml-copilot:local .

pipeline: data-synthetic features-local train score eval ## Offline end to end: data -> features -> model -> alerts -> evals

features-local: ## Features from the prepared Parquet files (no database)
	uv run aml-features build --from parquet

# ---------------- Cloud (free tier) ----------------
TF := $(shell command -v terraform || command -v tofu)
TF_DIR := infra/terraform/oci
CLOUD := AML_ENV_FILE=.env.cloud

cloud-init: ## Terraform init for the Oracle VM
	$(TF) -chdir=$(TF_DIR) init

cloud-plan: ## Show what Terraform will create
	$(TF) -chdir=$(TF_DIR) plan

cloud-apply: ## Create VM, network, bucket, budget alert
	$(TF) -chdir=$(TF_DIR) apply

cloud-destroy: ## Delete all Oracle resources created by Terraform
	$(TF) -chdir=$(TF_DIR) destroy

migrate: ## Apply schema migrations to Neon (needs AML_POSTGRES_MIGRATE_URL in .env.cloud)
	set -a && . ./.env.cloud && set +a && ./scripts/migrate.sh

cloud-deploy: ## Ship and start the VM stack over Tailscale, then smoke test
	./scripts/deploy_vm.sh

cloud-ps: ## Status of services on the VM
	ssh ubuntu@aml-vm 'cd /opt/aml && docker compose -f docker-compose.cloud.yml ps'

cloud-load: ## Load prepared data into Neon
	$(CLOUD) uv run aml-data load

cloud-test-int: ## Integration tests against Neon (creates and drops a temporary database)
	$(CLOUD) uv run pytest -m integration

.PHONY: help install up up-lite db-migrate down reset ps download data-synthetic data load quality \
	topics produce consume stream-demo features feast graph lint typecheck test test-int test-services check \
	cloud-init cloud-plan cloud-apply cloud-destroy migrate cloud-deploy cloud-ps cloud-load cloud-test-int \
	train score investigate worker workflow eval api demo ui drift audit-verify test-temporal image \
	pipeline features-local
