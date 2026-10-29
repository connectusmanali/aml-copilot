"""Policy retrieval (RAG): find the guidance passages relevant to a case.

Two interchangeable indexes:
    TfidfPolicyIndex    in memory, no services (default, tests)
    PgVectorPolicyIndex embeddings in Postgres (pgvector), cosine search in SQL

Retrieved passages become `pol:` evidence. They support the agent's reasoning ("the guidance
says payroll is a common false positive") but are never cited as facts about the customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import psycopg
from sklearn.feature_extraction.text import TfidfVectorizer

from aml.gateway.providers import Provider

POLICY_DIR = Path(__file__).with_name("policies")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # 'structuring#2'
    doc: str
    heading: str
    body: str


def load_chunks(directory: Path = POLICY_DIR) -> list[Chunk]:
    """One chunk per '## heading' section; the doc title is kept as context."""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        doc = path.stem
        sections = re.split(r"^## ", path.read_text(), flags=re.M)
        for i, sec in enumerate(sections[1:], start=1):
            heading, _, body = sec.partition("\n")
            chunks.append(Chunk(f"{doc}#{i}", doc, heading.strip(), " ".join(body.split())))
    return chunks


class PolicyIndex(Protocol):
    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]: ...


class TfidfPolicyIndex:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or load_chunks()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([f"{c.doc} {c.heading} {c.body}" for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vec.transform([query])
        scores = np.asarray((self.matrix @ q.T).todense()).ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class PgVectorPolicyIndex:
    def __init__(self, dsn: str, provider: Provider) -> None:
        self.dsn, self.provider = dsn, provider

    def build(self, chunks: list[Chunk] | None = None) -> int:
        chunks = chunks or load_chunks()
        vectors = self.provider.embed([f"{c.heading}. {c.body}" for c in chunks])
        with psycopg.connect(self.dsn) as conn:
            for c, v in zip(chunks, vectors, strict=True):
                conn.execute(
                    "INSERT INTO ref.policy_chunks (chunk_id, doc, heading, body, embedding, "
                    "embed_model) VALUES (%s, %s, %s, %s, %s::vector, %s) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET body = EXCLUDED.body, "
                    "embedding = EXCLUDED.embedding, embed_model = EXCLUDED.embed_model",
                    (c.chunk_id, c.doc, c.heading, c.body, _vec(v), self.provider.model),
                )
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.provider.embed([query])[0]
        with psycopg.connect(self.dsn) as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc, heading, body, 1 - (embedding <=> %s::vector) AS sim "
                "FROM ref.policy_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vec(q), _vec(q), k),
            ).fetchall()
        return [(Chunk(r[0], r[1], r[2], r[3]), float(r[4])) for r in rows]


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
