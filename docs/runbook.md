# Runbook

## Daily checks
| Check | Command | Healthy |
|---|---|---|
| Workers polling | Temporal UI → Workers on `aml-cases`, `aml-agent` | both queues have pollers |
| Queue health | `GET /cases?status=needs_human,escalated` | small and not growing |
| Drift | `make drift` | overall `stable` or `watch` |
| Audit chain | `make audit-verify` | `"ok": true` |
| Eval gates | CI `evals` job on main | green |

## Incidents

### LLM provider down or slow
Symptom: `investigate` activities retrying; cases piling up in `investigating`.
1. Gateway retries with backoff; after 4 attempts Temporal retries the activity.
2. If it persists, set `AML_LLM_PROVIDER=offline` and restart workers: the rule analyzer and
   template drafter take over (degraded but safe; drafts are valid by construction).
3. Workflows resume where they stopped; nothing is lost.

### Many drafts failing validation
Symptom: rising `needs_human` with "draft invalid".
1. Check `cases.reports.validation` issues for the common problem.
2. Compare prompt/model versions in `cases.case_files`; roll back the prompt or model.
3. Run `make eval` with the new version before re-enabling.

### Drift report says `act`
1. Read which features moved (top of the report). Upstream data change? Check `raw.ingest_runs`
   and `raw.rejected_records` for the same period.
2. If the business changed (new product, season), retrain → shadow → canary → promote.
3. If alerts/day is far from capacity, recalibrate the threshold (`make train CAPACITY=...`).

### Audit chain verification fails
Treat as a security incident. The report gives the first bad entry id. Freeze access, export
the log, compare with backups, involve security and compliance.

### SLA breaches
Escalated cases need a senior. Check staffing vs alerts/day; consider raising the threshold
temporarily (fewer alerts) with the model owner's approval, recorded in the audit log.

## Routine operations
- **Retrain:** `make features train` → read `evaluation.md` → `make eval` → shadow
  (`aml-ops shadow --challenger data/models/<version>`) → canary → update `CURRENT`.
- **Rollback model:** write the previous version into `data/models/CURRENT`; restart scorer.
- **Replay a case for audit:** `aml-ops replay --case case-<txn_id>`.
- **Rotate API keys:** update `AML_API_KEYS` in the secret store; restart the API.
