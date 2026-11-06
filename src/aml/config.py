"""Typed configuration, loaded from environment variables (prefix AML_) or a .env file.

Every module reads settings from here instead of calling os.environ directly, so
configuration is validated once at startup and is easy to override in tests.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AML_ENV_FILE lets you switch between local (.env) and cloud (.env.cloud) settings.
    model_config = SettingsConfigDict(
        env_prefix="AML_", env_file=os.environ.get("AML_ENV_FILE", ".env"), extra="ignore"
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "aml"
    postgres_user: str = "aml"
    postgres_password: SecretStr = SecretStr("change-me-locally")
    postgres_sslmode: str = "prefer"  # "require" for Neon

    kafka_bootstrap: str = "localhost:19092"
    kafka_topic_raw: str = "transactions.raw"
    kafka_topic_dlq: str = "transactions.dlq"
    kafka_group: str = "aml-ingest"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-locally")
    neo4j_database: str = "neo4j"
    # Neo4j AuraDB Free limits; the loader refuses to exceed them.
    neo4j_max_nodes: int = 200_000
    neo4j_max_relationships: int = 400_000

    temporal_host: str = "localhost:7233"
    mlflow_uri: str = "http://localhost:5000"

    data_dir: Path = Path("./data")
    random_seed: int = 42

    # offline (default: deterministic, no network) | gemini | anthropic
    llm_provider: str = "offline"
    llm_api_key: SecretStr | None = None
    llm_model: str = ""  # empty = provider default (gemini-flash-latest / claude-haiku-4-5)
    llm_embed_model: str = "gemini-embedding-001"
    case_token_budget: int = 20_000
    agent_max_tool_calls: int = 12
    agent_max_seconds: float = 60.0
    # Investigator API keys -> "user:role" (stand-in for SSO). Format: key1=alice:investigator,...
    api_keys: str = (
        "dev-investigator=alice:investigator,dev-senior=bob:senior,dev-auditor=carol:auditor"
    )
    case_sla_hours: float = 48.0

    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def feast_dir(self) -> Path:
        return self.data_dir / "feast"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
