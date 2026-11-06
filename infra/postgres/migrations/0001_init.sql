-- 0001: initial schema (Step 1). Applied by `aml-db migrate`; never edit after it has run.

CREATE EXTENSION IF NOT EXISTS vector;

-- raw   = data as it landed, never edited
-- ref   = reference/master data (customers, accounts)
-- cases = case management + audit (used from Step 5)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS cases;

CREATE TABLE IF NOT EXISTS ref.customers (
    customer_id              TEXT PRIMARY KEY,
    segment                  TEXT NOT NULL CHECK (segment IN ('individual', 'business')),
    occupation               TEXT NOT NULL,
    country                  CHAR(2) NOT NULL,
    risk_rating              TEXT NOT NULL CHECK (risk_rating IN ('low', 'medium', 'high')),
    expected_monthly_volume  NUMERIC(18, 2) NOT NULL,
    onboarded_at             DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS ref.accounts (
    account_id   TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES ref.customers (customer_id),
    bank_id      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    txn_id              TEXT PRIMARY KEY,           -- deterministic hash => idempotent loads
    ts                  TIMESTAMP NOT NULL,
    from_account        TEXT NOT NULL,
    to_account          TEXT NOT NULL,
    amount_paid         NUMERIC(18, 2) NOT NULL CHECK (amount_paid > 0),
    payment_currency    TEXT NOT NULL,
    amount_received     NUMERIC(18, 2) NOT NULL CHECK (amount_received > 0),
    receiving_currency  TEXT NOT NULL,
    payment_format      TEXT NOT NULL,
    is_laundering       BOOLEAN NOT NULL,
    loaded_at           TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_txn_from_ts ON raw.transactions (from_account, ts);
CREATE INDEX IF NOT EXISTS ix_txn_to_ts   ON raw.transactions (to_account, ts);
CREATE INDEX IF NOT EXISTS ix_txn_ts      ON raw.transactions (ts);

-- Append-only audit log. Hash-chained in Step 9; the table exists from day one.
CREATE TABLE IF NOT EXISTS cases.audit_log (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMP NOT NULL DEFAULT now(),
    actor       TEXT NOT NULL,       -- 'system:scorer', 'agent:v1', 'user:jdoe'
    entity      TEXT NOT NULL,       -- 'case:123'
    action      TEXT NOT NULL,
    details     JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Enforce append-only at the database level: no UPDATE or DELETE on the audit log.
CREATE OR REPLACE FUNCTION cases.forbid_audit_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_no_update ON cases.audit_log;
CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON cases.audit_log
    FOR EACH ROW EXECUTE FUNCTION cases.forbid_audit_change();
