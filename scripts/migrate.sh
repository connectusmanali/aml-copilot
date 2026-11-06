#!/usr/bin/env bash
# Apply pending schema migrations to Neon (or any Postgres). Safe on every deploy: applied
# migrations are skipped, edited ones are refused (see src/aml/db/migrate.py).
# Uses the DIRECT (non-pooled) URL: DDL and advisory locks should not go through a pooler.
set -euo pipefail
: "${AML_POSTGRES_MIGRATE_URL:?export AML_POSTGRES_MIGRATE_URL (direct Neon connection string)}"
uv run aml-db migrate --dsn "$AML_POSTGRES_MIGRATE_URL"
