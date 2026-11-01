"""Schema migrations: numbered SQL files applied once each, in order, and recorded.

    infra/postgres/migrations/0001_init.sql
    infra/postgres/migrations/0002_lineage_quarantine_audit.sql

Rules (the same ones Flyway and Alembic enforce):
  - A migration runs exactly once per database; public.schema_migrations records it.
  - Each migration runs in its own transaction: it applies completely or not at all.
  - A migration is never edited after it has run. Its checksum is stored, and a changed file
    stops the runner. To change the schema, add a new file.
  - An advisory lock stops two deploys migrating the same database at the same moment.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "infra" / "postgres" / "migrations"
NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
LOCK_ID = 7_220_001  # arbitrary, constant: identifies "aml migrations" for pg_advisory_lock

CREATE_TABLE = b"""
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    version     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str
    checksum: str


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    found: dict[str, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        m = NAME.match(path.name)
        if not m:
            raise MigrationError(f"{path.name}: expected NNNN_lowercase_name.sql")
        version = m.group(1)
        if version in found:
            raise MigrationError(f"two migrations share version {version}")
        text = path.read_text()
        found[version] = Migration(
            version, path.name, text, hashlib.sha256(text.encode()).hexdigest()
        )
    return [found[v] for v in sorted(found)]


def apply(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations. Returns the names applied (empty when up to date)."""
    migrations = discover(directory)
    done: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
        try:
            conn.execute(CREATE_TABLE)
            applied: dict[str, str] = dict(
                conn.execute("SELECT version, checksum FROM public.schema_migrations").fetchall()
            )
            for mig in migrations:
                if mig.version in applied:
                    if applied[mig.version] != mig.checksum:
                        raise MigrationError(
                            f"{mig.name} was edited after it was applied. "
                            "Never edit an applied migration: add a new one instead."
                        )
                    continue
                with conn.transaction():
                    # bytes = several statements in one call (simple query protocol)
                    conn.execute(mig.sql.encode())
                    conn.execute(
                        "INSERT INTO public.schema_migrations (version, name, checksum) "
                        "VALUES (%s, %s, %s)",
                        (mig.version, mig.name, mig.checksum),
                    )
                done.append(mig.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
    return done


def main(argv: list[str] | None = None) -> None:
    from aml.config import get_settings

    p = argparse.ArgumentParser(prog="aml-db")
    sub = p.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("migrate", help="apply pending schema migrations")
    mp.add_argument("--dsn", default=None, help="defaults to the configured database")
    args = p.parse_args(argv)

    dsn = args.dsn or get_settings().postgres_dsn
    names = apply(dsn)
    print("\n".join(f"applied {n}" for n in names) if names else "schema up to date")


if __name__ == "__main__":
    main()
