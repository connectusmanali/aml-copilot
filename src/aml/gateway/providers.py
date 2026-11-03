"""Model providers behind one small interface. The gateway never knows which vendor it talks to.

    GeminiProvider     Google AI Studio REST API (free tier: synthetic data only)
    AnthropicProvider  Anthropic Messages API
    OfflineProvider    deterministic, no network: tests, CI, and a safe default

Only HTTP is used (httpx), no vendor SDKs: three providers do not justify LiteLLM's dependency
tree, and swapping a vendor is a 40-line class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

EMBED_DIM = 256  # matches ref.policy_chunks.embedding vector(256)


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> Completion: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        retryable = resp.status_code == 429 or resp.status_code >= 500
        raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:300]}", retryable)


def hashing_embed(texts: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic bag-of-words embedding (feature hashing, L2-normalised). No network."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            word = "".join(ch for ch in word if ch.isalnum())
            if len(word) < 3:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = np.linalg.norm(out[i])
        if norm:
            out[i] /= norm
    return out


class OfflineProvider:
    """Deterministic stand-in. complete() echoes a JSON summary of the request; agents use their
    rule-based analyzer/drafter in offline mode rather than pretending this is a model."""

    name = "offline"

    def __init__(self, model: str = "offline-v1") -> None:
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int, json_mode: bool) -> Completion:
        digest = hashlib.sha256((system + user).encode()).hexdigest()[:12]
        text = json.dumps({"offline": True, "digest": digest}) if json_mode else f"offline:{digest}"
        return Completion(text, len(user.split()), len(text.split()), self.model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return hashing_embed(texts)


class GeminiProvider:
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, embed_model: str, timeout: float = 60) -> None:
        self.model, self.embed_model, self._key = model, embed_model, api_key
        self._http = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, *, max_tokens: int, json_mode: bool) -> Completion:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                **({"responseMimeType": "application/json"} if json_mode else {}),
            },
        }
        resp = self._http.post(
            f"{self.URL}/{self.model}:generateContent", params={"key": self._key}, json=body
        )
        _raise_for(resp)
        data = resp.json()
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise ProviderError(f"unexpected response: {str(data)[:300]}", False) from e
        usage = data.get("usageMetadata", {})
        return Completion(
            text,
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
            self.model,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            resp = self._http.post(
                f"{self.URL}/{self.embed_model}:embedContent",
                params={"key": self._key},
                json={"content": {"parts": [{"text": t}]}, "outputDimensionality": EMBED_DIM},
            )
            _raise_for(resp)
            vectors.append(resp.json()["embedding"]["values"])
        arr = np.asarray(vectors, dtype=np.float32)
        out: np.ndarray = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        return out


class AnthropicProvider:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, timeout: float = 60) -> None:
        self.model, self._key = model, api_key
        self._http = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, *, max_tokens: int, json_mode: bool) -> Completion:
        if json_mode:
            user += "\n\nRespond with a single JSON object and nothing else."
        resp = self._http.post(
            self.URL,
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        _raise_for(resp)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage", {})
        return Completion(
            text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), self.model
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return hashing_embed(texts)  # no embeddings endpoint: local hashing embedding
