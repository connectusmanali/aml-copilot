-- 0003 (Steps 4-9): alerts, cases, case files, reports, decisions, feedback, LLM call log,
-- watchlist, policy store (pgvector), and the audit log hash chain.

-- ---------- reference data for the agent's tools ----------
CREATE TABLE IF NOT EXISTS ref.watchlist (
    entry_id    TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL,
    list        TEXT NOT NULL CHECK (list IN ('sanctions', 'pep', 'adverse_media')),
    reason      TEXT NOT NULL,
    listed_at   DATE NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_watchlist_account ON ref.watchlist (account_id);

-- Policy and typology guidance, chunked and embedded for retrieval (RAG).
CREATE TABLE IF NOT EXISTS ref.policy_chunks (
    chunk_id     TEXT PRIMARY KEY,           -- 'structuring#2'
    doc          TEXT NOT NULL,
    heading      TEXT NOT NULL,
    body         TEXT NOT NULL,
    embedding    vector(256) NOT NULL,
    embed_model  TEXT NOT NULL
);

-- ---------- detection output ----------
CREATE TABLE IF NOT EXISTS cases.alerts (
    txn_id         TEXT PRIMARY KEY,
    ts             TIMESTAMP NOT NULL,
    from_account   TEXT NOT NULL,
    to_account     TEXT NOT NULL,
    amount_usd     DOUBLE PRECISION NOT NULL,
    score          DOUBLE PRECISION NOT NULL,
    priority       TEXT NOT NULL CHECK (priority IN ('high', 'low')),
    rules          TEXT NOT NULL DEFAULT '',
    typologies     TEXT NOT NULL DEFAULT '',
    reasons        JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_version  TEXT NOT NULL,
    shadow_score   DOUBLE PRECISION,          -- Step 9: score from a model in shadow mode
    shadow_version TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- case management ----------
CREATE TABLE IF NOT EXISTS cases.cases (
    case_id      TEXT PRIMARY KEY,           -- 'case-<txn_id>': one case per alert, idempotent
    txn_id       TEXT NOT NULL UNIQUE REFERENCES cases.alerts (txn_id),
    account_id   TEXT NOT NULL,
    priority     TEXT NOT NULL,
    score        DOUBLE PRECISION NOT NULL,
    status       TEXT NOT NULL CHECK (status IN (
                     'new', 'scored', 'investigating', 'drafted', 'in_review',
                     'needs_human', 'escalated', 'filed', 'closed')),
    sla_due_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_cases_queue ON cases.cases (status, score DESC);

CREATE TABLE IF NOT EXISTS cases.case_files (
    case_id        TEXT NOT NULL REFERENCES cases.cases (case_id),
    version        INT NOT NULL,
    payload        JSONB NOT NULL,
    agent_version  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, version)
);

CREATE TABLE IF NOT EXISTS cases.reports (
    case_id     TEXT NOT NULL REFERENCES cases.cases (case_id),
    version     INT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('draft', 'edit', 'final')),
    body        TEXT NOT NULL,
    validation  JSONB NOT NULL DEFAULT '{}'::jsonb,
    author      TEXT NOT NULL,             -- 'agent:report-v1' or 'user:<id>'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, version, kind)
);

CREATE TABLE IF NOT EXISTS cases.decisions (
    case_id        TEXT PRIMARY KEY REFERENCES cases.cases (case_id),
    decision       TEXT NOT NULL CHECK (decision IN ('file_str', 'close')),
    reason         TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    decided_by     TEXT NOT NULL,
    role           TEXT NOT NULL,
    edit_distance  DOUBLE PRECISION,       -- how much the investigator changed the draft (0-1)
    decided_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Investigator feedback on the score: future training labels.
CREATE TABLE IF NOT EXISTS cases.feedback (
    case_id            TEXT NOT NULL REFERENCES cases.cases (case_id),
    user_id            TEXT NOT NULL,
    agrees_with_score  BOOLEAN NOT NULL,
    comment            TEXT NOT NULL DEFAULT '',
    at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, user_id)
);

-- Every LLM call (masked text only), so a case can be replayed exactly and costs audited.
CREATE TABLE IF NOT EXISTS cases.llm_calls (
    call_id         TEXT PRIMARY KEY,       -- hash of (task, model, prompt version, masked prompt)
    case_id         TEXT,
    task            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    response        TEXT NOT NULL,
    input_tokens    INT NOT NULL,
    output_tokens   INT NOT NULL,
    latency_ms      INT NOT NULL,
    cost_usd        DOUBLE PRECISION NOT NULL,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_case ON cases.llm_calls (case_id);

-- ---------- audit hash chain (Step 9) ----------
-- Each entry stores the previous entry's hash and its own: editing or deleting any row breaks
-- every hash after it, so tampering is detectable even by someone with full access.
ALTER TABLE cases.audit_log ADD COLUMN IF NOT EXISTS prev_hash TEXT;
ALTER TABLE cases.audit_log ADD COLUMN IF NOT EXISTS row_hash TEXT;
