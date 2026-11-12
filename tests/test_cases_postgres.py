"""Step 5/9 on real Postgres: case store, decision rules, and the hash-chained audit log."""

import psycopg
import pytest

from aml.cases.audit import verify
from aml.cases.service import CaseService
from aml.cases.store import DecisionRefused, PostgresStore
from tests.test_agent import world  # noqa: F401
from tests.test_workflow import alert_row

pytestmark = pytest.mark.integration


def test_case_lifecycle_in_postgres(pg_dsn: str, world: dict) -> None:  # noqa: F811
    store = PostgresStore(pg_dsn)
    svc = CaseService(store, world["data"])
    cid = svc.open(alert_row(world, "Cash"))
    assert svc.open(alert_row(world, "Cash")) == cid
    svc.investigate(cid)
    assert svc.draft(cid) == "in_review"
    edit = svc.edit_report(cid, store.reports(cid)[0]["body"] + "\nExtra note.\n", "alice")
    assert not edit["validation"]["ok"]  # the added sentence has no citation
    with pytest.raises(DecisionRefused):
        svc.approve_report(cid, "alice", "investigator")  # latest edit is invalid
    svc.edit_report(cid, store.reports(cid)[0]["body"], "alice")
    svc.approve_report(cid, "alice", "investigator")
    assert svc.decide(cid, "file_str", "structuring confirmed", "alice", "investigator") == "filed"
    assert store.decision(cid)["decision"] == "file_str"  # type: ignore[index]
    assert verify(store.audit()).ok


def test_audit_log_is_append_only_and_tampering_is_detected(pg_dsn: str) -> None:
    store = PostgresStore(pg_dsn)
    store.append_audit("user:test", "case:x", "note", {"n": 1})
    with psycopg.connect(pg_dsn) as conn, pytest.raises(psycopg.errors.RaiseException):
        conn.execute("UPDATE cases.audit_log SET details = '{}'")
    # Someone with owner rights disables the trigger and rewrites history ...
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("ALTER TABLE cases.audit_log DISABLE TRIGGER audit_log_no_update")
        conn.execute("UPDATE cases.audit_log SET details = '{\"n\": 2}' WHERE action = 'note'")
        conn.execute("ALTER TABLE cases.audit_log ENABLE TRIGGER audit_log_no_update")
    # ... and the hash chain shows exactly where.
    report = verify(store.audit())
    assert not report.ok and report.problem == "entry content does not match its hash"
