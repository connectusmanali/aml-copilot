"""Case management storage: alerts, cases, case files, reports, decisions, feedback, audit.

Two implementations of one interface:
    MemoryStore     in-process, for tests, evals and the offline demo
    PostgresStore   the real system of record (`cases.*` tables)

Every state change goes through `transition`, which enforces the case state machine and writes
a hash-chained audit entry in the SAME transaction (Postgres) so the audit log can never miss a
change that happened.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from aml.cases.audit import GENESIS, AuditEntry, entry_hash

Status = Literal[
    "new",
    "scored",
    "investigating",
    "drafted",
    "in_review",
    "needs_human",
    "escalated",
    "filed",
    "closed",
]
Decision = Literal["file_str", "close"]
Role = Literal["investigator", "senior", "auditor"]

# The case state machine. Anything not listed is refused.
TRANSITIONS: dict[str, set[str]] = {
    "new": {"scored"},
    "scored": {"investigating", "in_review"},
    "investigating": {"drafted", "needs_human"},
    "drafted": {"in_review", "needs_human"},
    "needs_human": {"in_review", "escalated", "filed", "closed"},
    "in_review": {"escalated", "filed", "closed"},
    "escalated": {"filed", "closed"},
    "filed": set(),
    "closed": set(),
}
TERMINAL = {"filed", "closed"}


class IllegalTransition(ValueError):
    pass


class DecisionRefused(ValueError):
    pass


def now() -> datetime:
    return datetime.now(UTC)


def case_id_for(txn_id: str) -> str:
    return f"case-{txn_id}"


class CaseStore(Protocol):
    def open_case(self, alert: dict[str, Any], sla_hours: float) -> str: ...
    def get_case(self, case_id: str) -> dict[str, Any] | None: ...
    def queue(self, statuses: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]: ...
    def transition(
        self, case_id: str, to: str, actor: str, details: dict[str, Any] | None = None
    ) -> bool: ...
    def save_case_file(
        self, case_id: str, payload: dict[str, Any], versions: dict[str, str]
    ) -> int: ...
    def latest_case_file(self, case_id: str) -> dict[str, Any] | None: ...
    def save_report(
        self, case_id: str, kind: str, body: str, validation: dict[str, Any], author: str
    ) -> int: ...
    def reports(self, case_id: str) -> list[dict[str, Any]]: ...
    def record_decision(
        self,
        case_id: str,
        decision: str,
        reason: str,
        user: str,
        role: str,
        edit_distance: float | None,
    ) -> None: ...
    def decision(self, case_id: str) -> dict[str, Any] | None: ...
    def add_feedback(self, case_id: str, user: str, agrees: bool, comment: str) -> None: ...
    def audit(self, entity: str | None = None) -> list[AuditEntry]: ...
    def append_audit(
        self, actor: str, entity: str, action: str, details: dict[str, Any]
    ) -> AuditEntry: ...


def check_transition(current: str, to: str) -> None:
    if to not in TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"case cannot move from {current!r} to {to!r}")


def check_decision(
    case: dict[str, Any], decision: str, reason: str, role: str, has_final: bool
) -> None:
    """Who may decide what. Enforced in code, below the API, so no client can skip it."""
    if role not in {"investigator", "senior"}:
        raise DecisionRefused(f"role {role!r} may not decide cases")
    if case["status"] in TERMINAL:
        raise DecisionRefused(f"case is already {case['status']}")
    if case["status"] not in {"in_review", "needs_human", "escalated"}:
        raise DecisionRefused(f"case is {case['status']}; it is not ready for a decision")
    if not reason.strip():
        raise DecisionRefused("a decision needs a written reason")
    if case["status"] == "escalated" and role != "senior":
        raise DecisionRefused("an escalated (SLA-breached) case needs a senior investigator")
    if decision == "file_str" and not has_final:
        raise DecisionRefused("filing needs a final report approved by the investigator")


# ------------------------------------------------------------------------------------------
@dataclass
class MemoryStore:
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)
    alerts: dict[str, dict[str, Any]] = field(default_factory=dict)
    case_files: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    report_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    feedback: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    log: list[AuditEntry] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def append_audit(
        self, actor: str, entity: str, action: str, details: dict[str, Any]
    ) -> AuditEntry:
        with self._lock:
            prev = self.log[-1].row_hash if self.log else GENESIS
            at = now().isoformat()
            e = AuditEntry(
                len(self.log) + 1,
                at,
                actor,
                entity,
                action,
                details,
                prev,
                entry_hash(prev, at, actor, entity, action, details),
            )
            self.log.append(e)
            return e

    def open_case(self, alert: dict[str, Any], sla_hours: float) -> str:
        with self._lock:
            cid = case_id_for(str(alert["txn_id"]))
            if cid in self.cases:
                return cid  # idempotent: a retried activity does not open a second case
            self.alerts[str(alert["txn_id"])] = dict(alert)
            self.cases[cid] = {
                "case_id": cid,
                "txn_id": str(alert["txn_id"]),
                "account_id": str(alert["from_account"]),
                "priority": alert["priority"],
                "score": float(alert["score"]),
                "status": "new",
                "sla_due_at": now() + timedelta(hours=sla_hours),
                "created_at": now(),
                "updated_at": now(),
            }
            self.append_audit(
                "system:workflow",
                f"case:{cid}",
                "opened",
                {"txn_id": alert["txn_id"], "score": float(alert["score"])},
            )
            return cid

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        c = self.cases.get(case_id)
        return dict(c) if c else None

    def queue(self, statuses: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        rows = [c for c in self.cases.values() if not statuses or c["status"] in statuses]
        return [dict(c) for c in sorted(rows, key=lambda c: -c["score"])[:limit]]

    def transition(
        self, case_id: str, to: str, actor: str, details: dict[str, Any] | None = None
    ) -> bool:
        with self._lock:
            c = self.cases[case_id]
            if c["status"] == to:
                return False  # idempotent replay of the same step
            check_transition(c["status"], to)
            self.append_audit(
                actor, f"case:{case_id}", f"status:{c['status']}->{to}", details or {}
            )
            c["status"], c["updated_at"] = to, now()
            return True

    def save_case_file(
        self, case_id: str, payload: dict[str, Any], versions: dict[str, str]
    ) -> int:
        with self._lock:
            rows = self.case_files.setdefault(case_id, [])
            rows.append({"version": len(rows) + 1, "payload": payload, **versions})
            self.append_audit(
                f"agent:{versions.get('agent', '?')}",
                f"case:{case_id}",
                "case_file_saved",
                {"version": len(rows)},
            )
            return len(rows)

    def latest_case_file(self, case_id: str) -> dict[str, Any] | None:
        rows = self.case_files.get(case_id)
        return rows[-1] if rows else None

    def save_report(
        self, case_id: str, kind: str, body: str, validation: dict[str, Any], author: str
    ) -> int:
        with self._lock:
            rows = self.report_rows.setdefault(case_id, [])
            version = 1 + sum(1 for r in rows if r["kind"] == kind)
            rows.append(
                {
                    "case_id": case_id,
                    "version": version,
                    "kind": kind,
                    "body": body,
                    "validation": validation,
                    "author": author,
                    "created_at": now(),
                }
            )
            self.append_audit(
                author,
                f"case:{case_id}",
                f"report_{kind}",
                {"version": version, "ok": validation.get("ok")},
            )
            return version

    def reports(self, case_id: str) -> list[dict[str, Any]]:
        return list(self.report_rows.get(case_id, []))

    def record_decision(
        self,
        case_id: str,
        decision: str,
        reason: str,
        user: str,
        role: str,
        edit_distance: float | None,
    ) -> None:
        with self._lock:
            if case_id in self.decisions:
                return  # idempotent
            case = self.cases[case_id]
            has_final = any(r["kind"] == "final" for r in self.report_rows.get(case_id, []))
            check_decision(case, decision, reason, role, has_final)
            self.decisions[case_id] = {
                "case_id": case_id,
                "decision": decision,
                "reason": reason,
                "decided_by": user,
                "role": role,
                "edit_distance": edit_distance,
                "decided_at": now(),
            }
            self.transition(
                case_id,
                "filed" if decision == "file_str" else "closed",
                f"user:{user}",
                {"decision": decision, "reason": reason, "role": role},
            )

    def decision(self, case_id: str) -> dict[str, Any] | None:
        return self.decisions.get(case_id)

    def add_feedback(self, case_id: str, user: str, agrees: bool, comment: str) -> None:
        self.feedback[(case_id, user)] = {"agrees_with_score": agrees, "comment": comment}
        self.append_audit(
            f"user:{user}", f"case:{case_id}", "feedback", {"agrees_with_score": agrees}
        )

    def audit(self, entity: str | None = None) -> list[AuditEntry]:
        return [e for e in self.log if entity is None or e.entity == entity]


# ------------------------------------------------------------------------------------------
AUDIT_LOCK = 72_001  # advisory lock id that serializes the audit hash chain


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _conn(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    @staticmethod
    def _audit(
        conn: psycopg.Connection[dict[str, Any]],
        actor: str,
        entity: str,
        action: str,
        details: dict[str, Any],
    ) -> AuditEntry:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (AUDIT_LOCK,))
        last = conn.execute(
            "SELECT row_hash FROM cases.audit_log WHERE row_hash IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev = last["row_hash"] if last else GENESIS
        at = now().isoformat()
        h = entry_hash(prev, at, actor, entity, action, details)
        row = conn.execute(
            "INSERT INTO cases.audit_log (at, actor, entity, action, details, prev_hash, row_hash)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (at, actor, entity, action, Jsonb(details), prev, h),
        ).fetchone()
        assert row is not None
        return AuditEntry(row["id"], at, actor, entity, action, details, prev, h)

    def append_audit(
        self, actor: str, entity: str, action: str, details: dict[str, Any]
    ) -> AuditEntry:
        with self._conn() as conn:
            return self._audit(conn, actor, entity, action, details)

    def upsert_alert(self, conn: psycopg.Connection[dict[str, Any]], a: dict[str, Any]) -> None:
        reasons = a.get("reasons") or "[]"
        conn.execute(
            "INSERT INTO cases.alerts (txn_id, ts, from_account, to_account, amount_usd, score, "
            "priority, rules, typologies, reasons, model_version, shadow_score, shadow_version) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (txn_id) DO NOTHING",
            (
                a["txn_id"],
                a["ts"],
                a["from_account"],
                a["to_account"],
                float(a["amount_usd"]),
                float(a["score"]),
                a["priority"],
                a.get("rules") or "",
                a.get("typologies") or "",
                Jsonb(json.loads(reasons) if isinstance(reasons, str) else reasons),
                a["model_version"],
                a.get("shadow_score"),
                a.get("shadow_version"),
            ),
        )

    def open_case(self, alert: dict[str, Any], sla_hours: float) -> str:
        cid = case_id_for(str(alert["txn_id"]))
        with self._conn() as conn:
            self.upsert_alert(conn, alert)
            cur = conn.execute(
                "INSERT INTO cases.cases (case_id, txn_id, account_id, priority, score, status, "
                "sla_due_at) VALUES (%s,%s,%s,%s,%s,'new', now() + make_interval(secs => %s)) "
                "ON CONFLICT (case_id) DO NOTHING",
                (
                    cid,
                    alert["txn_id"],
                    alert["from_account"],
                    alert["priority"],
                    float(alert["score"]),
                    sla_hours * 3600,
                ),
            )
            if cur.rowcount:
                self._audit(
                    conn,
                    "system:workflow",
                    f"case:{cid}",
                    "opened",
                    {"txn_id": alert["txn_id"], "score": float(alert["score"])},
                )
        return cid

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cases.cases WHERE case_id = %s", (case_id,)
            ).fetchone()

    def queue(self, statuses: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if statuses:
                return conn.execute(
                    "SELECT * FROM cases.cases WHERE status = ANY(%s) ORDER BY score DESC LIMIT %s",
                    (statuses, limit),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM cases.cases ORDER BY score DESC LIMIT %s", (limit,)
            ).fetchall()

    def transition(
        self, case_id: str, to: str, actor: str, details: dict[str, Any] | None = None
    ) -> bool:
        with self._conn() as conn:
            return self._transition(conn, case_id, to, actor, details or {})

    def _transition(
        self,
        conn: psycopg.Connection[dict[str, Any]],
        case_id: str,
        to: str,
        actor: str,
        details: dict[str, Any],
    ) -> bool:
        row = conn.execute(
            "SELECT status FROM cases.cases WHERE case_id = %s FOR UPDATE", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        if row["status"] == to:
            return False
        check_transition(row["status"], to)
        conn.execute(
            "UPDATE cases.cases SET status = %s, updated_at = now() WHERE case_id = %s",
            (to, case_id),
        )
        self._audit(conn, actor, f"case:{case_id}", f"status:{row['status']}->{to}", details)
        return True

    def save_case_file(
        self, case_id: str, payload: dict[str, Any], versions: dict[str, str]
    ) -> int:
        with self._conn() as conn:
            conn.execute("SELECT 1 FROM cases.cases WHERE case_id = %s FOR UPDATE", (case_id,))
            row = conn.execute(
                "SELECT coalesce(max(version), 0) + 1 AS v FROM cases.case_files "
                "WHERE case_id = %s",
                (case_id,),
            ).fetchone()
            assert row is not None
            v = int(row["v"])
            conn.execute(
                "INSERT INTO cases.case_files (case_id, version, payload, agent_version, "
                "prompt_version, model_version) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    case_id,
                    v,
                    Jsonb(payload),
                    versions.get("agent", "?"),
                    versions.get("prompt", "?"),
                    versions.get("llm", "?"),
                ),
            )
            self._audit(
                conn,
                f"agent:{versions.get('agent', '?')}",
                f"case:{case_id}",
                "case_file_saved",
                {"version": v},
            )
            return v

    def latest_case_file(self, case_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cases.case_files WHERE case_id = %s ORDER BY version DESC LIMIT 1",
                (case_id,),
            ).fetchone()

    def save_report(
        self, case_id: str, kind: str, body: str, validation: dict[str, Any], author: str
    ) -> int:
        with self._conn() as conn:
            conn.execute("SELECT 1 FROM cases.cases WHERE case_id = %s FOR UPDATE", (case_id,))
            row = conn.execute(
                "SELECT coalesce(max(version), 0) + 1 AS v FROM cases.reports "
                "WHERE case_id = %s AND kind = %s",
                (case_id, kind),
            ).fetchone()
            assert row is not None
            v = int(row["v"])
            conn.execute(
                "INSERT INTO cases.reports (case_id, version, kind, body, validation, "
                "author) VALUES (%s,%s,%s,%s,%s,%s)",
                (case_id, v, kind, body, Jsonb(validation), author),
            )
            self._audit(
                conn,
                author,
                f"case:{case_id}",
                f"report_{kind}",
                {"version": v, "ok": validation.get("ok")},
            )
            return v

    def reports(self, case_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cases.reports WHERE case_id = %s ORDER BY created_at, kind, version",
                (case_id,),
            ).fetchall()

    def record_decision(
        self,
        case_id: str,
        decision: str,
        reason: str,
        user: str,
        role: str,
        edit_distance: float | None,
    ) -> None:
        with self._conn() as conn:
            case = conn.execute(
                "SELECT * FROM cases.cases WHERE case_id = %s FOR UPDATE", (case_id,)
            ).fetchone()
            if case is None:
                raise KeyError(case_id)
            if conn.execute(
                "SELECT 1 FROM cases.decisions WHERE case_id = %s", (case_id,)
            ).fetchone():
                return
            has_final = (
                conn.execute(
                    "SELECT 1 FROM cases.reports WHERE case_id = %s AND kind = 'final'", (case_id,)
                ).fetchone()
                is not None
            )
            check_decision(case, decision, reason, role, has_final)
            conn.execute(
                "INSERT INTO cases.decisions (case_id, decision, reason, decided_by, "
                "role, edit_distance) VALUES (%s,%s,%s,%s,%s,%s)",
                (case_id, decision, reason, user, role, edit_distance),
            )
            self._transition(
                conn,
                case_id,
                "filed" if decision == "file_str" else "closed",
                f"user:{user}",
                {"decision": decision, "reason": reason, "role": role},
            )

    def decision(self, case_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cases.decisions WHERE case_id = %s", (case_id,)
            ).fetchone()

    def add_feedback(self, case_id: str, user: str, agrees: bool, comment: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cases.feedback (case_id, user_id, agrees_with_score, "
                "comment) VALUES (%s,%s,%s,%s) ON CONFLICT (case_id, user_id) DO UPDATE "
                "SET agrees_with_score = EXCLUDED.agrees_with_score, "
                "comment = EXCLUDED.comment, at = now()",
                (case_id, user, agrees, comment),
            )
            self._audit(
                conn, f"user:{user}", f"case:{case_id}", "feedback", {"agrees_with_score": agrees}
            )

    def audit(self, entity: str | None = None) -> list[AuditEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, at, actor, entity, action, details, prev_hash, row_hash "
                "FROM cases.audit_log WHERE row_hash IS NOT NULL "
                + ("AND entity = %s " if entity else "")
                + "ORDER BY id",
                (entity,) if entity else (),
            ).fetchall()
        return [
            AuditEntry(
                r["id"],
                r["at"].astimezone(UTC).isoformat(),
                r["actor"],
                r["entity"],
                r["action"],
                r["details"],
                r["prev_hash"],
                r["row_hash"],
            )
            for r in rows
        ]
