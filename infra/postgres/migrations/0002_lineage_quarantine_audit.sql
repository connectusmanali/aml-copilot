-- 0002 (Step 2): lineage, quarantine table, audit-log hardening, time zones.

-- 1. Audit log: also block TRUNCATE. Row-level triggers never fire on TRUNCATE, so the
--    0001 trigger alone let `TRUNCATE cases.audit_log` wipe it (tested).
--    Remaining gap: the table owner can drop triggers. Step 5 runs the app as a role that only
--    has INSERT/SELECT; Step 9 adds a hash chain so tampering is detectable.
DROP TRIGGER IF EXISTS audit_log_no_truncate ON cases.audit_log;
CREATE TRIGGER audit_log_no_truncate
    BEFORE TRUNCATE ON cases.audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION cases.forbid_audit_change();

-- 2. Moments the system records are TIMESTAMPTZ (stored as UTC, shown in any zone).
--    raw.transactions.ts stays TIMESTAMP: the source data has no time zone.
--    Existing values came from now() written into a TIMESTAMP column, i.e. in the server's
--    TimeZone setting, so they are interpreted in that same zone (UTC on Docker and Neon).
ALTER TABLE cases.audit_log  ALTER COLUMN at        TYPE TIMESTAMPTZ
    USING at AT TIME ZONE current_setting('TimeZone');
ALTER TABLE raw.transactions ALTER COLUMN loaded_at TYPE TIMESTAMPTZ
    USING loaded_at AT TIME ZONE current_setting('TimeZone');

-- 3. Lineage: every batch load and every stream session is a run.
CREATE TABLE IF NOT EXISTS raw.ingest_runs (
    run_id         UUID PRIMARY KEY,
    source         TEXT NOT NULL,          -- 'batch:synthetic', 'batch:ibm', 'stream:transactions.raw'
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    rows_valid     BIGINT NOT NULL DEFAULT 0,
    rows_rejected  BIGINT NOT NULL DEFAULT 0,
    rows_inserted  BIGINT NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running', 'succeeded', 'failed'))
);

-- The run that FIRST inserted each transaction (reloads never overwrite it).
-- Nullable: rows loaded before this migration have no run.
ALTER TABLE raw.transactions
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES raw.ingest_runs (run_id);

-- 4. Quarantine: rejected records are kept with their reason, never dropped.
--    record_hash = sha256(source | payload): re-loading the same bad row is idempotent.
CREATE TABLE IF NOT EXISTS raw.rejected_records (
    record_hash  TEXT PRIMARY KEY,
    run_id       UUID REFERENCES raw.ingest_runs (run_id),
    source       TEXT NOT NULL,
    payload      JSONB NOT NULL,
    error        TEXT NOT NULL,
    rejected_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_rejected_run ON raw.rejected_records (run_id);
