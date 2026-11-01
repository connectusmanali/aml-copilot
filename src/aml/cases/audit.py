"""Hash-chained audit log.

Each entry stores the hash of the entry before it and a hash of its own content plus that
previous hash. Changing, deleting or re-ordering any entry breaks every hash after it, so
`verify` finds tampering even by someone with full database access (the triggers in 0001/0002
stop accidental UPDATE/DELETE/TRUNCATE; the chain makes deliberate tampering detectable).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    id: int
    at: str
    actor: str
    entity: str
    action: str
    details: dict[str, Any]
    prev_hash: str
    row_hash: str


def entry_hash(
    prev: str, at: str, actor: str, entity: str, action: str, details: dict[str, Any]
) -> str:
    body = json.dumps(
        [prev, at, actor, entity, action, details],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class ChainReport:
    ok: bool
    entries: int
    first_bad_id: int | None = None
    problem: str = ""


def verify(entries: list[AuditEntry]) -> ChainReport:
    """Verify a full chain (all entries, in id order)."""
    prev = GENESIS
    for e in entries:
        if e.prev_hash != prev:
            return ChainReport(False, len(entries), e.id, "link to previous entry is broken")
        if entry_hash(e.prev_hash, e.at, e.actor, e.entity, e.action, e.details) != e.row_hash:
            return ChainReport(False, len(entries), e.id, "entry content does not match its hash")
        prev = e.row_hash
    return ChainReport(True, len(entries))
