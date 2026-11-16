import uuid
from pathlib import Path

import psycopg
import pytest

from aml.config import get_settings
from aml.db.migrate import MIGRATIONS_DIR, MigrationError, apply, discover


def test_repo_migrations_are_well_formed() -> None:
    migrations = discover(MIGRATIONS_DIR)
    assert [m.version for m in migrations] == sorted(m.version for m in migrations)
    assert migrations[0].name == "0001_init.sql"


@pytest.mark.parametrize("name", ["1_init.sql", "0001-init.sql", "0001_Init.sql"])
def test_bad_names_are_refused(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("SELECT 1;")
    with pytest.raises(MigrationError):
        discover(tmp_path)


def test_duplicate_versions_are_refused(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("SELECT 1;")
    (tmp_path / "0001_b.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError, match="share version"):
        discover(tmp_path)


@pytest.mark.integration
def test_apply_once_in_order_and_refuse_edits(tmp_path: Path) -> None:
    s = get_settings()
    name = f"aml_mig_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(s.postgres_dsn, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}".encode())
    dsn = s.model_copy(update={"postgres_db": name}).postgres_dsn
    try:
        (tmp_path / "0001_t.sql").write_text("CREATE TABLE t (x int); INSERT INTO t VALUES (1);")
        (tmp_path / "0002_t.sql").write_text("ALTER TABLE t ADD COLUMN y int;")
        assert apply(dsn, tmp_path) == ["0001_t.sql", "0002_t.sql"]
        assert apply(dsn, tmp_path) == []  # already applied: nothing runs twice

        (tmp_path / "0003_bad.sql").write_text("ALTER TABLE t ADD COLUMN z int; SELECT 1/0;")
        with pytest.raises(psycopg.errors.DivisionByZero):
            apply(dsn, tmp_path)
        with psycopg.connect(dsn) as conn:  # the failed migration left no trace
            cols = [
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='t'"
                )
            ]
            assert "z" not in cols
        (tmp_path / "0003_bad.sql").unlink()

        (tmp_path / "0002_t.sql").write_text("ALTER TABLE t ADD COLUMN y bigint;")  # edited
        with pytest.raises(MigrationError, match="edited after it was applied"):
            apply(dsn, tmp_path)
    finally:
        with psycopg.connect(s.postgres_dsn, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)".encode())
