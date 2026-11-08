"""Shared fixtures.

Integration tests never touch your working database: `pg_dsn` creates a throwaway database,
migrates it, and drops it when the test session ends.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from aml.config import get_settings
from aml.data import ibm, synthetic
from aml.data.cli import prepare
from aml.data.validate import validate_transactions
from aml.db.migrate import apply


@pytest.fixture(scope="session")
def txns() -> pd.DataFrame:
    """Small validated synthetic dataset: ~2k transactions, 200 accounts, 3 typologies."""
    valid, _ = validate_transactions(
        ibm.normalize(synthetic.generate(n_accounts=200, n_normal=2_000, n_patterns=9, seed=7))
    )
    return valid


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    prepare("synthetic", "small", None, out_dir=out)
    return out


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    settings = get_settings()
    name = f"aml_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(settings.postgres_dsn, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}".encode())
    dsn = settings.model_copy(update={"postgres_db": name}).postgres_dsn
    try:
        apply(dsn)
        yield dsn
    finally:
        with psycopg.connect(settings.postgres_dsn, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)".encode())
