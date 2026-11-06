#!/usr/bin/env bash
# Ship the VM stack to aml-vm over Tailscale and (re)start it. Used by CI and by `make cloud-deploy`.
# Requires: Tailscale connected, .env.cloud present locally.
set -euo pipefail
HOST="${AML_VM_SSH:-ubuntu@aml-vm}"
DEST=/opt/aml

[[ -f .env.cloud ]] || { echo ".env.cloud missing"; exit 1; }

rsync -az --delete --relative \
  docker-compose.cloud.yml infra/postgres/ops-init.sql infra/mlflow/ \
  Dockerfile .dockerignore pyproject.toml uv.lock README.md src/ \
  "$HOST:$DEST/"
rsync -az .env.cloud "$HOST:$DEST/.env.cloud"
ssh "$HOST" "chmod 600 $DEST/.env.cloud && cd $DEST && \
  docker compose -f docker-compose.cloud.yml --env-file .env.cloud up -d --build --remove-orphans && \
  docker compose -f docker-compose.cloud.yml ps"

# Smoke tests: fail the deploy if a service did not come up.
for i in $(seq 1 30); do
  if curl -fsS "http://aml-vm:5000/health" >/dev/null && curl -fsS "http://aml-vm:8080" >/dev/null \
     && curl -fsS "http://aml-vm:8000/health" >/dev/null; then
    echo "smoke tests passed"; exit 0
  fi
  sleep 10
done
echo "smoke tests failed"; exit 1
